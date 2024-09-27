import tensorflow as tf
import numpy as np
import cv2
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sklearn.cluster import KMeans
from tensorflow.keras.models import load_model
import pickle
import logging
import os

# Print versions
import sklearn
print(f"TensorFlow version: {tf.__version__}")
print(f"scikit-learn version: {sklearn.__version__}")

# Set up logging
logging.basicConfig(level=logging.DEBUG)
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

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(current_dir, 'model')

logger.debug(f"Current working directory: {os.getcwd()}")
logger.debug(f"Model directory: {model_dir}")
logger.debug(f"Contents of model directory: {os.listdir(model_dir)}")

# Load model
try:
    model = load_model(os.path.join(model_dir, 'emotion_model.h5'))
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    model = None

# Load scaler
try:
    with open(os.path.join(model_dir, 'scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)
    logger.info("Scaler loaded successfully")
except Exception as e:
    logger.error(f"Error loading scaler: {e}")
    scaler = None

# Load label encoder
try:
    with open(os.path.join(model_dir, 'label_encoder.pkl'), 'rb') as f:
        le = pickle.load(f)
    logger.info("Label encoder loaded successfully")
except Exception as e:
    logger.error(f"Error loading label encoder: {e}")
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
        raise HTTPException(status_code=500, detail="Model components not loaded")

    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Invalid image file")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")

    try:
        colors = extract_colors(img)
        model_input = colors.flatten() / 255.0
        model_input = scaler.transform(model_input.reshape(1, -1))
        
        prediction = model.predict(model_input)
        predicted_class = np.argmax(prediction)
        predicted_emotion = le.inverse_transform([predicted_class])[0]
    except Exception as e:
        logger.error(f"Error making prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Error making prediction: {str(e)}")

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