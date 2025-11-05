# src/train.py

import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd

# Step 1: Import all our custom modules and configurations
from config import MODEL_CONFIG, TRAINING_CONFIG, FEATURE_COLS, TARGET_COL
from data_preparation import load_data
from preprocessing import process_splits_to_kbins, build_model_arrays
from dataset import KBinsDataset
from models import *
from utils import DataManager, ExperimentTracker

def main():
    # Step 2: Initialize managers
    data_manager = DataManager()
    tracker = ExperimentTracker(experiment_name="LSTM_Baseline_Run")
    tracker.save_config({'model': MODEL_CONFIG, 'training': TRAINING_CONFIG})

    # Step 3: Load and Preprocess Data (or load from cache)
    # This part could be run once and then loaded via DataManager
    # For simplicity, we show the full flow here.
    try:
        X, Y, labels = data_manager.load_arrays(filename_prefix="processed_data")
        print("Loaded pre-processed arrays from cache.")
    except FileNotFoundError:
        print("Pre-processed data not found. Running preprocessing...")
        df = load_data('../data/raw_sensor_data.csv')
        # ... (Code to get splits and process them into norm_df) ...
        # norm_df = process_splits_to_kbins(...)
        X, Y, labels_list = build_model_arrays(norm_df, FEATURE_COLS, TARGET_COL)
        labels = pd.DataFrame(index=pd.to_datetime(labels_list))
        data_manager.save_arrays(X, Y, labels, filename_prefix="processed_data")

    # Step 4: Load cross-validation folds
    folds_data = data_manager.load_rolling_splits()

    # Step 5: Loop through each fold for training and evaluation
    for i, fold in enumerate(folds_data['folds']):
        print(f"\n===== FOLD {i+1}/{len(folds_data['folds'])} =====")
        
        # Get train/validation indices for this fold
        train_idx, val_idx = data_manager.get_fold_indices(X, labels, fold)

        # Create Datasets and DataLoaders
        train_dataset = KBinsDataset(X[train_idx], Y[train_idx], fit_scalers=True)
        feature_scaler, target_scaler = train_dataset.get_scalers() # Get fitted scalers
        
        val_dataset = KBinsDataset(X[val_idx], Y[val_idx], 
                                   feature_scaler=feature_scaler, 
                                   target_scaler=target_scaler)

        train_loader = DataLoader(train_dataset, batch_size=TRAINING_CONFIG['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=TRAINING_CONFIG['batch_size'])

        # Step 6: Initialize model, optimizer, etc. for this fold
        model = GasTurbineLSTM(input_features=len(FEATURE_COLS), **MODEL_CONFIG)
        optimizer = torch.optim.Adam(model.parameters(), lr=TRAINING_CONFIG['learning_rate'])
        criterion = torch.nn.MSELoss()

        # Step 7: Run the training loop (a separate function is best)
        # history = train_model(model, train_loader, val_loader, optimizer, criterion, ...)
        # tracker.save_model(model, fold=i+1, is_best=True)
        # tracker.save_training_history(history, fold=i+1)
        # tracker.plot_training_history(history, fold=i+1)

        # Step 8: Evaluate the best model
        # metrics = evaluate_model(model, val_loader, target_scaler, ...)
        # tracker.save_metrics(metrics, fold=i+1, split='validation')

    # Step 9: Summarize results from all folds
    tracker.summarize_results()
    print("✅ Pipeline finished successfully.")


if __name__ == "__main__":
    main()