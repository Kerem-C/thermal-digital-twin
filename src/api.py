import torch
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from train_pinn import ThermalPINN

app = FastAPI(title="Thermal Digital Twin Inference API", version="1.0")

model = ThermalPINN()
try:
    model.load_state_dict(torch.load("data/thermal_pinn.pth"))
    model.eval()
except FileNotFoundError:
    print("Warning: thermal_pinn.pth not found. Ensure weights are in /data.")

X_MEAN = np.array([650.0, 0.005]) 
X_STD = np.array([200.0, 0.001])
Y_MEAN = np.array([0.05, 45.0])
Y_STD = np.array([0.02, 10.0])

class TelemetryPayload(BaseModel):
    server_id: str
    p_gpu_avg_watts: float
    r_tim_degC_W: float

class FlowPrediction(BaseModel):
    server_id: str
    required_m_dot_kg_s: float
    predicted_return_temp_C: float

@app.post("/predict_flow", response_model=FlowPrediction)
async def predict_coolant_flow(payload: TelemetryPayload):
    try:
        x_raw = np.array([payload.p_gpu_avg_watts, payload.r_tim_degC_W])
        x_norm = (x_raw - X_MEAN) / X_STD
        
        x_tensor = torch.tensor(x_norm, dtype=torch.float32).unsqueeze(0)
       
        with torch.no_grad():
            y_pred_norm = model(x_tensor).numpy()[0]
            
        y_pred = (y_pred_norm * Y_STD) + Y_MEAN

        return FlowPrediction(
            server_id=payload.server_id,
            required_m_dot_kg_s=max(0.0, float(y_pred[0])), 
            predicted_return_temp_C=float(y_pred[1])
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



