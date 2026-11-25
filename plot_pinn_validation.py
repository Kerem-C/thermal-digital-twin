import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), "scripts"))
from train_pinn import ThermalDataset, ThermalPINN

def main() -> None:

    dataset = ThermalDataset("data/thermal_training_data_v4.csv")

    model = ThermalPINN()
    model.load_state_dict(torch.load("data/thermal_pinn.pth", map_location="cpu"))
    model.eval()

    with torch.no_grad():
        Y_pred_norm = model(dataset.X_tensor).numpy()

    Y_true = dataset.Y
    Y_pred = (Y_pred_norm * dataset.Y_std) + dataset.Y_mean

    def metrics(y_t: np.ndarray, y_p: np.ndarray):
        ss_res = np.sum((y_t - y_p) ** 2)
        ss_tot = np.sum((y_t - np.mean(y_t)) ** 2)
        r2 = 1.0 - (ss_res / ss_tot)
        mae = np.mean(np.abs(y_t - y_p))
        return r2, mae

    r2_flow, mae_flow = metrics(Y_true[:, 0], Y_pred[:, 0])
    r2_temp, mae_temp = metrics(Y_true[:, 1], Y_pred[:, 1])

    # Check thermodynamic conservation: P_in vs Q_out = m_dot * c_p * (T_return - 20)
    p_in_total = np.sum(dataset.X, axis=1)
    q_out_pred = Y_pred[:, 0] * 4184.0 * (Y_pred[:, 1] - 20.0)
    mean_energy_err = np.mean(np.abs(p_in_total - q_out_pred) / p_in_total) * 100.0
    print(f"Mean Thermodynamic Energy Balance Error: {mean_energy_err:.2f}%")

    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    # Subplot 1: Steady-State Coolant Flow Rate
    ax1.scatter(Y_true[:, 0], Y_pred[:, 0], alpha=0.55, s=18, color="#1f77b4", edgecolors="none", label="PINN Predictions")
    lims_flow = [min(Y_true[:, 0].min(), Y_pred[:, 0].min()), max(Y_true[:, 0].max(), Y_pred[:, 0].max())]
    ax1.plot(lims_flow, lims_flow, "k--", linewidth=1.2, label="Ideal Fit (y = x)")
    ax1.set_title("Coolant Flow Rate Prediction", fontweight="bold", pad=8)
    ax1.set_xlabel("RK4 Ground Truth (kg/s)", fontweight="bold")
    ax1.set_ylabel("PINN Predicted (kg/s)", fontweight="bold")
    ax1.text(0.05, 0.84, f"$R^2$ = {r2_flow:.4f}\nMAE = {mae_flow:.4f} kg/s",
             transform=ax1.transAxes, bbox=dict(facecolor="white", alpha=0.9, edgecolor="#cccccc"))
    ax1.legend(loc="lower right", frameon=True, framealpha=0.95)
    ax1.grid(True, linestyle="--", alpha=0.45)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Subplot 2: Return Manifold Temperature
    ax2.scatter(Y_true[:, 1], Y_pred[:, 1], alpha=0.55, s=18, color="#d62728", edgecolors="none", label="PINN Predictions")
    lims_temp = [min(Y_true[:, 1].min(), Y_pred[:, 1].min()), max(Y_true[:, 1].max(), Y_pred[:, 1].max())]
    ax2.plot(lims_temp, lims_temp, "k--", linewidth=1.2, label="Ideal Fit (y = x)")
    ax2.set_title("Return Manifold Temp Prediction", fontweight="bold", pad=8)
    ax2.set_xlabel("RK4 Ground Truth (°C)", fontweight="bold")
    ax2.set_ylabel("PINN Predicted (°C)", fontweight="bold")
    ax2.text(0.05, 0.84, f"$R^2$ = {r2_temp:.4f}\nMAE = {mae_temp:.2f} °C",
             transform=ax2.transAxes, bbox=dict(facecolor="white", alpha=0.9, edgecolor="#cccccc"))
    ax2.legend(loc="lower right", frameon=True, framealpha=0.95)
    ax2.grid(True, linestyle="--", alpha=0.45)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    fig.suptitle("PINN Surrogate Validation (500 Monte Carlo Scenarios)", fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig("portfolio_pinn_validation.png", dpi=300, bbox_inches="tight")
    plt.show()

if __name__ == "__main__":
    main()