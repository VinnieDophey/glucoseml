# This is the one with API
# Load model
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
import io
from fastapi import FastAPI, File, UploadFile
import os

# Api
app = FastAPI()

# Load model
model_path = os.path.join(
    os.path.dirname(__file__),
    "saved_models",
    "2301TrainedModel.keras"
)

model = None

def load_model_once():
    global model
    if model is None:
        model = tf.keras.models.load_model(model_path)
    return model    

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    
    contents = await file.read()
    newdf = pd.read_csv(io.BytesIO(contents))
    
    # Load data
    newdf['Timestamp'] = pd.to_datetime(newdf['Timestamp'], format='%d/%m/%Y %H:%M')
    newdf = newdf.dropna(subset=['Timestamp'])
    newdf = newdf.sort_values(by=['Timestamp'], axis=0, ascending=True)
    newdf = newdf.reset_index(drop=True)

    # Window
    
    
    lookback = 24 # = to uni_past_history value
    if len(newdf) < lookback:
        return {'error': 'Not enough data to predict'}
    current_window = newdf["Glucose"].values[-lookback:].reshape(-1, 1)

    last_timestamp = newdf["Timestamp"].iloc[-1]
    
    # Prediction
    predictedList = []

    for step in range(1, 13): #1hr
        pred = model.predict(current_window.reshape(1, *current_window.shape), verbose=0)
        pred_value = pred[0][0]
        future_time = last_timestamp + pd.Timedelta(minutes=5 * step)
        current_window = np.concatenate([current_window[1:], [[pred_value]]], axis=0)
        
        predictedList.append((future_time, pred_value))
    
    return {
        "predictions": [
        {
            "timestamp": timestamp.isoformat() + "Z",
            "glucose": float(prediction)/18
        }
        for timestamp, prediction in predictedList
    ]
        
    }

#note to self - rem fastapi turns dicts --> json so convert everything to dicts



# Save
# predicted_df = pd.DataFrame(predictedList, columns=["Timestamp", "Predicted_Glucose"])

# output_path = "/Users/vincentpham/Documents/glucoseML/predicted/glucosePred2302.csv"
# predicted_df.to_csv(output_path, index=False)

# print(f"\nPredictions saved to {output_path}")