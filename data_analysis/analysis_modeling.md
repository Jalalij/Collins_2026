# Available Dataset, Data Loading/Sync, and LEO Temporal Dynamics Modeling

## Available Datasets
This dataset characterizes the temporal dynamics of OneWeb LEO satellite links. It captures physical-layer metrics and end-to-end transport-layer performance.

The dataset includes 3 subdirectories
```pgsql
data/
├── iperf
│   └── timestamp_DL_targetBW_iperf3.csv /* throughput,jitter, packet_loss */
│   └── timestamp_DL_targetBW_iperf3.json /* metadata included */
└── ping
│   └── timestamp_ping.csv /* rtt_ms */
│   └── timestamp_ping.json /* metadata included */
├── sattrace
│   └── AIM_timestamp.csv /* average_SINR,satellite_ID */
├── experiment_metadata_timestamp.json /* experiment setup metadata */
```

## Data Loading and Time Sync
Check the [data_sync_analysis.py](data_sync_analysis.py) for data loading and time sync.
- Data loading: The csv files in *ping/*, *iperf/*, and *sattrace/* directories will be loaded
- Time sync: It aligns each `iperf` and `sattrace` record with the first available `ping` measurement occurring at or after that timestamp within the specified tolerance window.
- Output: 
    1. All processed files are stored in `./processed_data` (changeable)

    2. Csv files named in `aligned_experiment_timestamp.csv` and separate uplink (UL) and downlink (DL) csv files with `_UL` and `_DL` appended to the filename

    3. `all_processed_experiments_combined.csv` and `all_processed_experiments_metadata.csv` for all aligned experiments in a single file, one for data and the other for metadata


## Temporal Dynamics Modeling

### 1. File Structure

Modules under `src` include:

- models: Neural network and statistical model implementations
- evaluation: Model evaluation and comparison utilities
- data: Data processing and preparation tools
- analysis: Main analysis orchestration
- visualization: Plotting and visualization utilities
- utils: Helper functions and utilities

### 2. Model Training

Files within `scripts` are for model training:

- analysis: the main script for cross-layer model training 
- config: config file for model training args
- full_analysis: another main training script focusing on VAR, LSTM, and transformers
- run_tests: bash script for main training commands

Input and output for model training:

- Input: csv files ('aligned_experiment_timestamp.csv) in the `./processed_data` folder
- Output: Prediction results are stored at `results.csv`

### 3. Model Use

- Model: load the model with trained parameters into the multipath scheduler.
    * Option 1:
        ```bash
        # Use LSTM as an example
        import torch

        # Save the entire model
        torch.save(model, 'full_lstm_model.pt')

        # Load the model
        model = torch.load('full_lstm.pt')
        model.eval() # Set to inference mode
        ```
    * Option 2:
        ```bash
        # Use LSTM as an example
        import torch

        # 1. Save the trained model weights after training
        torch.save(model.state_dict(), 'lstm_weights.pth')

        # 2. Load the model with trained weights for prediction
        model = LSTMModel(*args, **kwargs) # LSTMModel is the defined LSTM model class
        model.load_state_dict(torch.load('lstm_weights.pth'))
        model.eval() # Set to inference mode
        ```

- Input: cross-layer new data including SINR, handover, throughput, packet loss, and RTT. Details can be found at the function 'load_and_split_experiments' within 'analysis.py'

- Output: feed the input into the trained model and we can get throughput, packet loss, and RTT predictions stored in variable 'y_pred'.
