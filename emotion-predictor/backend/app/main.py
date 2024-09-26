from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
from sklearn.cluster import KMeans
from tensorflow.keras.models import load_model
import pickle
import io
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define model directory
model_dir = 'app/model'

# Load your trained model
try:
    model = load_model(f'{model_dir}/emotion_model.h5')
    logger.info("Model loaded successfully")
except FileNotFoundError:
    logger.error("emotion_model.h5 file not found")
    model = None

# Load your scaler
try:
    with open(f'{model_dir}/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    logger.info("Scaler loaded successfully")
except FileNotFoundError:
    logger.error("scaler.pkl file not found")
    scaler = None

# Load your label encoder
try:
    with open(f'{model_dir}/label_encoder.pkl', 'rb') as f:
        le = pickle.load(f)
    logger.info("Label encoder loaded successfully")
except FileNotFoundError:
    logger.error("label_encoder.pkl file not found")
    le = None

def extract_colors(image, n_colors=3):
    """Extract dominant colors from an image using KMeans clustering."""
    pixels = image.reshape(-1, 3)
    kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
    kmeans.fit(pixels)
    colors = kmeans.cluster_centers_
    return colors.astype(int)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Predict the emotion from the uploaded image."""
    if model is None or scaler is None or le is None:
        raise HTTPException(status_code=500, detail="Model, scaler, or label encoder not loaded")

    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Invalid image file")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        raise HTTPException(status_code=400, detail="Invalid image file")

    try:
        # Extract colors using k-means
        colors = extract_colors(img)
        
        # Prepare input for the model
        model_input = colors.flatten() / 255.0  # Normalize to [0, 1]
        model_input = scaler.transform(model_input.reshape(1, -1))
        
        # Make prediction
        prediction = model.predict(model_input)
        predicted_class = np.argmax(prediction)
        
        # Convert numerical prediction to emotion label
        predicted_emotion = le.inverse_transform([predicted_class])[0]
    except Exception as e:
        logger.error(f"Error making prediction: {e}")
        raise HTTPException(status_code=500, detail="Error making prediction")

    return {
        "predicted_emotion": predicted_emotion,
        "colors": colors.tolist()
    }

@app.get("/")
async def root():
    return {"message": "Hello World"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)