import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Tuple

class ThermalDataset(Dataset):
    def __init__(self, csv_file: str) -> None:
        df = pd.read_csv(csv_file)

        self.X = df[['p_gpu_1', 'p_gpu_2', 'p_gpu_3', 'p_gpu_4']].values
        self.Y = df[['m_dot_steady_kg_s', 't_return_manifold']].values

        self.X_mean, self.X_std = self.X.mean(axis=0), self.X.std(axis=0)
        self.Y_mean, self.Y_std = self.Y.mean(axis=0), self.Y.std(axis=0)

        self.X_norm = (self.X - self.X_mean) / self.X_std
        self.Y_norm = (self.Y - self.Y_mean) / self.Y_std

        self.X_tensor = torch.tensor(self.X_norm, dtype=torch.float32)
        self.Y_tensor = torch.tensor(self.Y_norm, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.X_tensor)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X_tensor[idx], self.Y_tensor[idx]
    
class ThermalPINN(nn.Module):
    def __init__(self) -> None:
        super(ThermalPINN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(4, 128),
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Linear(128, 2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
    
def pinn_loss(predictions_norm: torch.Tensor, targets_norm: torch.Tensor, inputs_norm: torch.Tensor, dataset: ThermalDataset, lambda_phys: float=0.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mse_loss = nn.MSELoss()(predictions_norm, targets_norm)

    device = predictions_norm.device
    predictions = predictions_norm * torch.tensor(dataset.Y_std, dtype=torch.float32, device=device) + torch.tensor(dataset.Y_mean, dtype=torch.float32, device=device)
    inputs = inputs_norm * torch.tensor(dataset.X_std, dtype=torch.float32, device=device) + torch.tensor(dataset.X_mean, dtype=torch.float32, device=device)

    p_in_total = torch.sum(inputs, dim=1)
    m_dot = predictions[:, 0]
    t_return = predictions[:, 1]

    q_out_total = m_dot * 4184.0 * (t_return - 20.0)

    physics_loss = torch.mean(((p_in_total - q_out_total) / p_in_total) ** 2)
    scaled_physics_loss = physics_loss * lambda_phys

    total_loss = mse_loss + scaled_physics_loss
    return total_loss, mse_loss, scaled_physics_loss

def train_model() -> None:
    print("Loading Monte Carlo Dataset and initiating Tensor mappings...")
    dataset = ThermalDataset("data/thermal_training_data_v4.csv")
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

    model = ThermalPINN()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 4000

    print("Initiating Physics-Informed Training Loop...")
    for epoch in range(epochs):
        total_loss, total_mse, total_phys = 0.0, 0.0, 0.0
        
        if epoch < 500:
            current_lambda = 0.0
        else:
            current_lambda = min(0.1, (epoch - 500) * 0.0001)

        for X_batch, Y_batch in dataloader:
            optimizer.zero_grad()
            predictions = model(X_batch)

            loss, mse, phys = pinn_loss(predictions, Y_batch, X_batch, dataset, lambda_phys=current_lambda)
            
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_mse += mse.item()
            total_phys += phys.item()

        if (epoch + 1) % 200 == 0:
            avg_loss = total_loss / len(dataloader)
            avg_mse = total_mse / len(dataloader)
            avg_phys = total_phys / len(dataloader)
            print(f"Epoch {epoch+1:04d} | Lambda: {current_lambda:.4f} | Total Loss: {avg_loss:.5f} | Data MSE: {avg_mse:.5f} | Physics: {avg_phys:.5f}")
    
    print("Training Complete. Thermodynamics successfully mapped to tensor weights.")
    torch.save(model.state_dict(), "data/thermal_pinn.pth")

if __name__ == "__main__":
    train_model()