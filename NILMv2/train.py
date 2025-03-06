import os
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np

from data_preprocessing import NILMDataset, device_thresholds
from models import AdvancedSeq2PointCNN

def teca(actual, prediction):
    mask = ~np.isnan(actual)
    if np.sum(mask) > 0:
        actual = actual[mask]
        prediction = prediction[mask]
    else:
        return 0.0
    nomin = np.abs(actual - prediction).sum()
    print('nomin', nomin)
    dnomin = 2 * np.sum(actual)
    print('dnomin', dnomin)
    if dnomin == 0:
        return 0.0
    value = nomin / dnomin
    teca_score = 1 - value
    return float(teca_score) * 100

def validate_model(model,
                   loader,
                   criterion,
                   device_list,
                   device_thresholds,
                   input_scaler,
                   output_scaler,
                   device):
    model.eval()
    total_loss = 0.0

    all_true_scaled = []
    all_pred_scaled = []

    with torch.no_grad():
        for batch_x, batch_y, _ in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            total_loss += loss.item()

            all_true_scaled.append(batch_y.cpu().numpy())
            all_pred_scaled.append(outputs.cpu().numpy())

    avg_loss_scaled = total_loss / len(loader)

    true_scaled = np.concatenate(all_true_scaled, axis=0)
    pred_scaled = np.concatenate(all_pred_scaled, axis=0)

    true_unscaled = output_scaler.inverse_transform(true_scaled)
    pred_unscaled = output_scaler.inverse_transform(pred_scaled)

    metrics = {}
    for i, dev in enumerate(device_list):
        t = true_unscaled[:, i]
        p = pred_unscaled[:, i]

        mask = (t != 0)
        if mask.sum() == 0:
            metrics[dev] = {'teca': None, 'r2': None, 'mae': None, 'mse': None}
        else:
            t_masked = t[mask]
            p_masked = p[mask]
            metrics[dev] = {
                'teca': teca(t_masked, p_masked),
                'r2': r2_score(t_masked, p_masked),
                'mae': mean_absolute_error(t_masked, p_masked),
                'mse': mean_squared_error(t_masked, p_masked)
            }

    return avg_loss_scaled, metrics

def train_model(model,
                train_loader,
                val_loader,
                optimizer,
                scheduler,
                criterion,
                device_list,
                device_thresholds,
                input_scaler,
                output_scaler,
                device,
                epochs=20,
                patience=5):

    best_val_loss = float('inf')
    epochs_no_improve = 0

    train_losses = []
    val_losses = []

    model.to(device)

    for epoch in range(1, epochs+1):
        model.train()
        running_loss = 0.0

        for batch_x, batch_y, _ in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}"):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)

        val_loss, val_metrics = validate_model(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device_list=device_list,
            device_thresholds=device_thresholds,
            input_scaler=input_scaler,
            output_scaler=output_scaler,
            device=device
        )
        val_losses.append(val_loss)

        print(f"\nEpoch {epoch}/{epochs}")
        print(f"  Training Loss (scaled)  = {train_loss:.6f}")
        print(f"  Validation Loss (scaled)= {val_loss:.6f}")
        for dev, m in val_metrics.items():
            print(f"    {dev}: TECA={m['teca']}, R²={m['r2']}, MAE={m['mae']}, MSE={m['mse']}")

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), "rNILMv2/esults/models/best_model.pth")
            joblib.dump(train_loader.dataset.input_scaler, "NILMv2/results/models/input_scaler.save")
            joblib.dump(train_loader.dataset.output_scaler, "NILMv2/results/models/output_scaler.save")
            print("  [*] Best model + scalers saved.")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopping triggered.")
                break

    plt.figure(figsize=(10,5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('NILMv2/results/plots/training_validation_loss.png')
    plt.close()

def main(config):
    import torch
    from torch.utils.data import DataLoader
    from sklearn.preprocessing import StandardScaler
    from data_preprocessing import NILMDataset, device_thresholds
    from models import AdvancedSeq2PointCNN

    paths = config["paths"]
    train_params = config["train"]

    train_data_path = paths["train_data"]
    val_data_path = paths["val_data"]

    window_size = train_params["window_size"]
    batch_size = train_params["batch_size"]
    epochs = train_params["epochs"]
    learning_rate = train_params["learning_rate"]
    patience = train_params["patience"]
    device_list = train_params["device_list"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if os.path.exists(paths["model_save"]) and os.path.exists(paths["input_scaler"]) and os.path.exists(paths["output_scaler"]):
        print("Model/scalers exist. Skipping training.")
        return

    input_scaler = StandardScaler()
    output_scaler = StandardScaler()

    train_dataset = NILMDataset(
        data_path=train_data_path,
        window_size=window_size,
        device_list=device_list,
        device_thresholds=device_thresholds,
        input_scaler=input_scaler,
        output_scaler=output_scaler,
        stride=1,
        is_training=True
    )
    val_dataset = NILMDataset(
        data_path=val_data_path,
        window_size=window_size,
        device_list=device_list,
        device_thresholds=device_thresholds,
        input_scaler=input_scaler,
        output_scaler=output_scaler,
        stride=1,
        is_training=False
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = AdvancedSeq2PointCNN(input_dim=1, output_dim=len(device_list), window_size=window_size).to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        device_list=device_list,
        device_thresholds=device_thresholds,
        input_scaler=input_scaler,
        output_scaler=output_scaler,
        device=device,
        epochs=epochs,
        patience=patience
    )

if __name__ == "__main__":
    print("Please run the program using run.py with a config file.")
