import pandas as pd
from darts import TimeSeries
from darts.models import TFTModel
from darts.dataprocessing.transformers import Scaler
import joblib
import warnings
warnings.filterwarnings("ignore")

def train_forecaster():
    print("Loading data for DLR Forecaster...")
    df = pd.read_csv('historical_grid_data_v2.csv')
    
    # Train on one highly volatile corridor for the PoC (Line 4)
    df_line = df[df['Line_ID'] == 'Line_4'].copy()
    df_line['Timestamp'] = pd.to_datetime(df_line['Timestamp'])
    df_line.sort_values('Timestamp', inplace=True)
    
    series = TimeSeries.from_dataframe(df_line, time_col='Timestamp', value_cols=['DLR_Ampacity_Limit', 'Load_Amps'])
    
    scaler = Scaler()
    series_scaled = scaler.fit_transform(series)
    train, val = series_scaled.split_before(0.9)
    
    print("Training Temporal Fusion Transformer (Predicting 6 hours ahead)...")
    
    # --- THIS IS THE FIX ---
    # We added 'add_relative_index' and 'add_encoders' to give the AI time-awareness
    model = TFTModel(
        input_chunk_length=48, 
        output_chunk_length=24, 
        hidden_size=16, 
        lstm_layers=1, 
        num_attention_heads=4, 
        batch_size=32, 
        n_epochs=3, 
        random_state=42,
        add_relative_index=True,  # Fixes the ValueError
        add_encoders={            # Gives the model a calendar
            'cyclic': {'future': ['hour', 'dayofweek']}
        }
    )
    
    model.fit(train, val_series=val, verbose=True)
    
    model.save("dlr_forecaster.pt")
    joblib.dump(scaler, "forecaster_scaler.pkl")
    print("Success! Forecaster saved.")

if __name__ == "__main__":
    train_forecaster()