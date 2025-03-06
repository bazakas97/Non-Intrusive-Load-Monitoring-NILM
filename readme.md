## NILM
This codebase provides a practical example of the approach that is used for residential NILM (Non-Intrusive Load Monitoring) by ITI team. This README file aims to document the main elements and functionalities of the pipeline used.

## General approach

### Data structure
The approach utilizes appliance-level (labeled) data for training. The pipeline operates on data with a 1-minute sampling frequency. The approach has been tested on a dataset from SEL, containing appliance-level and aggregate energy consumption data for six residences over approximately three months. The data is in CSV format and includes the following appliances: refrigerator, oven, dishwasher, and washing machine. Each CSV file contains `timestamp`, `voltage`, `current`, and `energy`.

### Data Splitting Strategy
To effectively train and evaluate the NILM model, the dataset is split into distinct training, validation, and test sets. Two approaches have been implemented for data splitting:

#### Per-House Split (60/20/20 Approach):
In this approach, each house’s data is individually partitioned into 60% for training, 20% for validation, and 20% for testing. This method ensures that every residence contributes data across all stages of the pipeline, which can help in assessing model performance uniformly across different houses.
`NILMv2\DATA\RealData\allhouses\data`

#### House-Based Split (Training vs. Testing Approach):
Alternatively, the dataset can be split based on house identifiers. For example, data from houses 1 through 4 is used exclusively for training, while data from houses 5 and 6 is reserved for testing. This strategy is particularly useful for evaluating the model’s generalization to completely unseen residences.
`NILMv2\DATA\RealData\splitted\data`

Choose the splitting method that best suits your experimental setup and evaluation requirements.

### Preprocess
The pre-process steps that are followed to improve the prediction performance are the following:

#### Resampling
The data is resampled to a consistent 1-minute frequency to ensure uniformity.
#### Handling missing values
Missing values in the dataset are handled by filling them with zeros.

#### Filtering Device Cycles

The `filter_device_cycles` function removes short activations and noise from appliance data. It identifies periods where appliance power consumption exceeds `min_energy_value` for at least `min_duration`. Activations not meeting these criteria are considered noise and set to zero.

### Gaussian smoothing

Apply gaussian smoothing (1D) to smooth spikes

#### Data Scaling

Input (mains energy) and output (appliance energy) data are scaled using `StandardScaler` from scikit-learn to ensure zero mean and unit variance, improving model training.
#### Sequence processing with sliding windows
The `NILMDataset` class creates sliding windows of `window_size` minutes from the mains energy data, pairing them with the corresponding appliance energy consumption at the window's midpoint. This transforms the time series data into a supervised learning problem. The `stride` parameter controls the overlap between consecutive windows.

### Model description
This codebase uses the `AdvancedSeq2PointCNN` model, a convolutional neural network designed for NILM.  It takes a sequence of mains power readings as input and predicts the power consumption of individual appliances at a single point in time.

### Post-process
Post-process is essential to ensure that model predictions match real-world constraints and minimize prediction noise. It consists of different thresholding filters described below:
- Predictions close to zero (typically around 10% of peak values) are set to zero, reducing noise that might appear as minor false positives.


## Setup and replication

### Configurable parameters
Key parameters are defined in the `config.yaml`:

`action`: chose the desired option (train, evaluate, extractsynthetic),

`train_data`: choose between synthetic , Per-House Split  or House-Based  Split   Dataset `NILMv2\DATA`

`test_data`: choose between synthetic , Per-House Split  or House-Based  Split   Dataset `NILMv2\DATA`

`val_data`: choose between synthetic , Per-House Split  or House-Based  Split   Dataset `NILMv2\DATA`

`model_save`,`input_scaler`,`output_scaler`: train the model so you can save those files in `NILMv2\results\models\` or move the files in the  same directory  from the folder you choose to use the Dataset (example for Per-House Split Approach use the files from:`NILMv2\DATA\RealData\allhouses\allhmodel`)

`window_size`: size of the sliding window,

`stride`: stride of the sliding window,

`batch_size`: Number of samples per iteration,

`learning_rate`:learning_rate,

`epochs`: Total training iterations count,

`patience`: Allowed epochs without improvement,

`device_list`: List of target devices,


`min_energy_value`: minimum energy value for device cycle filtering,

`min_duration`: minimum duration for device cycle filtering,

`days`: how many days of synthetic data you want to generate




### Initiate virtual environment
To run the pipeline and replicate the results first create a virtual environment (e.g. for linux) with `python3 -m venv venv`, activate it (`source venv/bin/activate`) and install the require packages with `pip install -r requirements.txt`.
### Initialize dataset
If there is no available dataset then choose  the `extractsynthetic` as`action` in `config.yaml` file and run the following :
```bash
python  NILMv2/run.py --config NILMv2/config.yaml
```
located in the `NILMv2` folder. The data is saved in the `DATA/SyntheticData/data` directory.

### Training
To train the prediction model  choose the `train` as `action` in `config.yaml` file and choose the desired dataset you want to train from `DATA` throught RealData or SyntheticData, then run the following:
```bash
python  NILMv2/run.py --config NILMv2/config.yaml
```
### Cross-validation and results
Finally, to evaluate the performance of the trained model choose the `train` as `action` in `config.yaml` file and run the following:
```bash
python  NILMv2/run.py --config NILMv2/config.yaml
```
The results of the cross-validation are stored in the `results/csv/test_predictions.csv` / `results/plots/energy_[device]_test_plot.html` file.

### Results on synthetic dataset
The performance using the synthetic dataset can be described by the following table / figures.

#### Table for synthetic Metrics
![Metrics for synthetic dataset](NILMv2\DATA\SyntheticData\synthPlots\device_metrics.png)

#### Figures
![Plot for dishwasher](NILMv2\DATA\SyntheticData\synthPlots\dish_synth.png)

![Plot for fridge](NILMv2\DATA\SyntheticData\synthPlots\fridge_synth.png)

![Plot for triangular](NILMv2\DATA\SyntheticData\synthPlots\trianglular_synth.png)

![Plot for washingmachine](NILMv2\DATA\SyntheticData\synthPlots\washingmachine_synth.png)

### Results on SEL's dataset
The performance using the data provided by SEL can be described by the following table / figures.
##### The SEL dataset can be split in two ways. In the first approach, the houses are partitioned into training, validation, and testing sets using a 60/20/20 ratio. In the alternative approach, houses 1 through 4 are used for training, while houses 5 and 6 serve as the test set.

#### 1st Approach
#### Table for  Metrics
![Metrics for 1st Approach dataset](NILMv2\DATA\RealData\allhouses\allplots\device_metrics.png)

#### Figures
![Plot for dishwasher](NILMv2\DATA\RealData\allhouses\allplots\dishwaher_all.png)

![Plot for fridge](NILMv2\DATA\RealData\allhouses\allplots\fridge.png)

![Plot for oven](NILMv2\DATA\RealData\allhouses\allplots\oven.png)

![Plot for washingmachine](NILMv2\DATA\RealData\allhouses\allplots\/washing_machine.png)

#### 2nd Approach
#### Table for  Metrics
![Metrics for 2nd Approach dataset](NILMv2\DATA\RealData\splitted\plots\device_metrics.png)

#### Figures
![Plot for dishwasher](NILMv2\DATA\RealData\splitted\plots\dishwasher.png)

![Plot for fridge](NILMv2\DATA\RealData\splitted\plots\fridge.png)

![Plot for oven](NILMv2\DATA\RealData\splitted\plots\oven.png)

![Plot for washingmachine](NILMv2\DATA\RealData\splitted\plots\washing_machine.png)

###### Disclaimer: In the second approach, please note that two lines may appear on the energy main. This occurs because a second house is operating concurrently during the same hours, which can lead to some confusion when interpreting the results.




