import os
import joblib
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import plotly.express as px
from plotly.offline import plot
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from data_preprocessing import NILMDataset, device_thresholds
from models import AdvancedSeq2PointCNN
from postprocessing import advanced_postprocess_predictions

def teca(actual, prediction):
    mask = ~np.isnan(actual)
    if np.sum(mask) > 0:
        actual = actual[mask]
        prediction = prediction[mask]
    else:
        return 0.0
    nomin = np.abs(actual - prediction).sum()
    dnomin = 2 * np.sum(actual)
    if dnomin == 0:
        return 0.0
    value = nomin / dnomin
    teca_score = 1 - value
    return float(teca_score) * 100

def line_plot(df, x_col, y_cols, title="", filename='line_plot.html'):
    if df[x_col].dtype != 'datetime64[ns]':
        df[x_col] = pd.to_datetime(df[x_col], errors='coerce')
    fig = px.line(df, x=x_col, y=y_cols, title=title)
    fig.update_xaxes(
        rangeslider_visible=True,
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=1, label="YTD", step="year", stepmode="todate"),
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(step="all")
            ])
        )
    )
    plot(fig, show_link=True, filename=filename)

def evaluate_model(model,
                   test_data_path,
                   device_list,
                   device_postprocessing_params,
                   input_scaler,
                   output_scaler,
                   window_size,
                   batch_size,
                   device,
                   stride=1,
                   save_results=True,
                   postprocess_real_data=True):

    test_dataset = NILMDataset(
        data_path=test_data_path,
        window_size=window_size,
        device_list=device_list,
        device_thresholds=device_thresholds,
        input_scaler=input_scaler,
        output_scaler=output_scaler,
        stride=stride,
        is_training=False
    )

    from torch.utils.data import DataLoader
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    criterion = nn.MSELoss()
    model.eval()

    total_loss = 0
    all_true_scaled = []
    all_pred_scaled = []
    all_mains_unscaled = []
    all_datetimes = []

    with torch.no_grad():
        for batch_x, batch_y, batch_dt in test_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            pred_scaled = model(batch_x)
            loss = criterion(pred_scaled, batch_y)
            total_loss += loss.item()
            all_true_scaled.append(batch_y.cpu().numpy())
            all_pred_scaled.append(pred_scaled.cpu().numpy())

            mid_idx = window_size // 2
            mids = batch_x[:, mid_idx, :].cpu().numpy()
            mains_un = input_scaler.inverse_transform(mids)
            all_mains_unscaled.append(mains_un)
            all_datetimes.extend(batch_dt)

    scaled_loss = total_loss / len(test_loader)
    print(f"\n[TEST] MSE (scaled): {scaled_loss:.6f}")

    all_true_scaled = np.concatenate(all_true_scaled, axis=0)
    all_pred_scaled = np.concatenate(all_pred_scaled, axis=0)
    all_mains_unscaled = np.concatenate(all_mains_unscaled, axis=0).flatten()
    true_unscaled = output_scaler.inverse_transform(all_true_scaled)
    pred_unscaled = output_scaler.inverse_transform(all_pred_scaled)

    for i, dev in enumerate(device_list):
        dev_params = device_postprocessing_params[dev]
        pred_unscaled[:, i] = advanced_postprocess_predictions(
            predictions=pred_unscaled[:, i],
            min_duration=dev_params['min_duration'],
            min_energy_value=dev_params['min_energy_value']
        )
        if postprocess_real_data:
            true_unscaled[:, i] = advanced_postprocess_predictions(
                predictions=true_unscaled[:, i],
                min_duration=dev_params['min_duration'],
                min_energy_value=dev_params['min_energy_value']
            )

    for i, dev in enumerate(device_list):
        t = true_unscaled[:, i]
        p = pred_unscaled[:, i]
        mask = (t != 0)
        if mask.sum() == 0:
            print(f"{dev} has no nonzero ground truth.")
            continue
        t_masked = t[mask]
        p_masked = p[mask]
        m_teca = teca(t_masked, p_masked)
        m_r2   = r2_score(t_masked, p_masked)
        m_mae  = mean_absolute_error(t_masked, p_masked)
        m_mse  = mean_squared_error(t_masked, p_masked)
        print(f"Device={dev} TECA={m_teca:.3f}, R²={m_r2:.3f}, MAE={m_mae:.3f}, MSE={m_mse:.3f}")

    if save_results:
        df_save = pd.DataFrame()
        df_save['datetime'] = all_datetimes
        df_save['mains'] = all_mains_unscaled
        for i, dev in enumerate(device_list):
            df_save[f'{dev}_true'] = true_unscaled[:, i]
            df_save[f'{dev}_pred'] = pred_unscaled[:, i]
        df_save.to_csv('NILMv2/results/csv/test_predictions.csv', index=False)
        print("Predictions saved to 'test_predictions.csv'")

    for i, dev in enumerate(device_list):
        t = true_unscaled[:, i]
        p = pred_unscaled[:, i]
        df_plot = pd.DataFrame({
            "datetime": all_datetimes,
            "Mains": all_mains_unscaled,
            f"{dev}_true": t,
            f"{dev}_pred": p
        })
        df_plot = df_plot.head(9000)
        line_plot(
            df_plot,
            x_col="datetime",
            y_cols=["Mains", f"{dev}_true", f"{dev}_pred"],
            title=f"Disaggregation for {dev}",
            filename=f"NILMv2/results/plots/{dev}_test_plot.html"
        )

def main(config):
    import torch
    import os
    import joblib
    from models import AdvancedSeq2PointCNN

    paths = config["paths"]
    eval_params = config["evaluate"]

    test_data_path = paths["test_data"]
    window_size = eval_params["window_size"]
    batch_size = eval_params["batch_size"]
    stride = eval_params.get("stride", 1)
    device_list = eval_params["device_list"]
    device_postprocessing_params = eval_params["device_postprocessing_params"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(paths["model_save"]):
        print("No saved model found. Please train first.")
        return

    input_scaler = joblib.load(paths["input_scaler"])
    output_scaler = joblib.load(paths["output_scaler"])
    model = AdvancedSeq2PointCNN(input_dim=1, output_dim=len(device_list), window_size=window_size)
    model.load_state_dict(torch.load(paths["model_save"], map_location=device))
    model.to(device)

    evaluate_model(
        model=model,
        test_data_path=test_data_path,
        device_list=device_list,
        device_postprocessing_params=device_postprocessing_params,
        input_scaler=input_scaler,
        output_scaler=output_scaler,
        window_size=window_size,
        batch_size=batch_size,
        device=device,
        stride=stride,
        save_results=True,
        postprocess_real_data=True
    )

if __name__ == "__main__":
    print("Please run the program using run.py with a config file.")
