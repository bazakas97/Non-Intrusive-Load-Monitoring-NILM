import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from scipy.ndimage import gaussian_filter1d

##############################################################################
# Dictionary of device thresholds
##############################################################################
device_thresholds = {
        'energy_dish_washer': {
            'min_duration': 3,
            'min_energy_value': 100,
            
        },
        'energy_oven': {
            'min_duration': 3,
            'min_energy_value': 100,
            
        },
        'energy_washing_machine': {
            'min_duration': 3,
            'min_energy_value': 100,
            
        },
        'energy_fridge_freezer': {
            'min_duration': 0,
            'min_energy_value': 0,
            
        }
    }

##############################################################################
# Helper Functions
##############################################################################

def remove_negative_values(df, col_list):
    """
    Set negative values to zero for specified columns.
    """
    for col in col_list:
        df.loc[df[col] < 0, col] = 0
    return df


def correct_mains_vs_devices(df, device_list, mains_col='energy_mains'):
    """
    For each row, if sum of devices > energy_mains, 
    scale device consumption down proportionally.
    NOTE: This is step #1 in your preprocessing report.
    """
    # Sum of devices
    sum_dev = df[device_list].sum(axis=1)
    # Identify rows where device sum exceeds mains
    mask = (sum_dev > df[mains_col]) & (df[mains_col] > 0)

    # Scale factor
    scale_factor = df.loc[mask, mains_col] / sum_dev.loc[mask]

    for dev in device_list:
        df.loc[mask, dev] = df.loc[mask, dev] * scale_factor

    return df


def moving_average_smoothing(series, window=5):
    """
    Apply a simple moving average smoothing with the specified window size.
    We'll fill edges by forward/backward fill to avoid NaNs.
    """
    ma = series.rolling(window=window, center=True).mean()
    # Fill edges
    ma = ma.fillna(method='bfill').fillna(method='ffill')
    return ma


def apply_moving_average_smoothing(df, col_list, window=5):
    """
    Apply moving average smoothing to specified columns.
    NOTE: This is step #2 in your preprocessing report.
    """
    for col in col_list:
        df[col] = moving_average_smoothing(df[col], window=window)
    return df


def apply_gaussian_smoothing(df, col_list, sigma=2, exclude=None):
    """
    Apply gaussian smoothing (1D) to specified columns,
    optionally excluding some columns.
    """
    if exclude is not None:
        col_list = [col for col in col_list if col not in exclude]
    
    for col in col_list:
        df[col] = gaussian_filter1d(df[col].values, sigma=sigma)
    return df



def final_align_mains_with_devices(df, device_list, mains_col='energy_mains'):
    """
    At the end, align energy_mains with the sum of the devices.
    If you want them EXACTLY equal, we can set:
       df[mains_col] = df[device_list].sum(axis=1)
    """
    df[mains_col] = df[device_list].sum(axis=1)
    return df


##############################################################################
# Main NILMDataset Class
##############################################################################

class NILMDataset(Dataset):
    def __init__(
        self,
        data_path,
        window_size,
        device_list,
        device_thresholds,
        input_scaler=None,
        output_scaler=None,
        stride=1,
        is_training=True
    ):
        # Load data
        self.data = pd.read_csv(data_path)
        # If you have a datetime column (uncomment if needed):
        # self.data['datetime'] = pd.to_datetime(self.data['datetime'])

        self.window_size = window_size
        self.device_list = device_list
        self.device_thresholds = device_thresholds
        self.stride = stride
        self.is_training = is_training

        if 'energy_mains' not in self.data.columns:
            raise ValueError("Dataset must contain 'energy_mains' column.")

        ######################################################################
        # 0) Basic NA fill , make sure device columns exist
        ######################################################################
        self.data[self.device_list] = self.data[self.device_list].fillna(0)
        self.data['energy_mains'] = self.data['energy_mains'].fillna(0)

        ######################################################################
        # 1) Remove negative values (mains + devices)
        ######################################################################
        all_cols = ['energy_mains'] + self.device_list
        self.data = remove_negative_values(self.data, col_list=all_cols)

        ######################################################################
        # 1.5) Correction of Energy Mains vs Device Consumption
        ######################################################################
        self.data = correct_mains_vs_devices(self.data, device_list=self.device_list)

        ######################################################################


        ######################################################################
        # 2) Gaussian Filtering (sigma=2) for devices
        ######################################################################
        self.data = apply_gaussian_smoothing(self.data, col_list=self.device_list, sigma=2, exclude=[''])



        # ######################################################################
        # Final alignment: set energy_mains = sum(devices) (row by row)
        ######################################################################
        self.data = final_align_mains_with_devices(self.data, self.device_list)

        ######################################################################
        # SCALING (StandardScaler)
        ######################################################################
        self.input_scaler = input_scaler if input_scaler is not None else StandardScaler()
        self.output_scaler = output_scaler if output_scaler is not None else StandardScaler()

        # Fit or transform
        if self.is_training:
            self.data['energy_mains'] = self.input_scaler.fit_transform(
                self.data['energy_mains'].values.reshape(-1, 1)
            ).flatten()
            self.data[self.device_list] = self.output_scaler.fit_transform(
                self.data[self.device_list]
            )
        else:
            self.data['energy_mains'] = self.input_scaler.transform(
                self.data['energy_mains'].values.reshape(-1, 1)
            ).flatten()
            self.data[self.device_list] = self.output_scaler.transform(
                self.data[self.device_list]
            )

        ######################################################################
        # Precompute valid centers for windows
        ######################################################################
        half_w = self.window_size // 2
        self.valid_centers = list(range(half_w, len(self.data) - half_w, self.stride))

    def __len__(self):
        return len(self.valid_centers)

    def __getitem__(self, idx):
        center_idx = self.valid_centers[idx]
        half_w = self.window_size // 2

        start_idx = center_idx - half_w
        end_idx   = center_idx + half_w

        X = self.data['energy_mains'].iloc[start_idx:end_idx].values.astype(np.float32)
        X = X.reshape(-1, 1)  # (window_size, 1)

        y = self.data.loc[center_idx, self.device_list].values.astype(np.float32)

        # If your CSV has 'datetime' column:
        dt = None
        if 'datetime' in self.data.columns:
            dt = self.data['datetime'].iloc[center_idx]

        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32), dt
