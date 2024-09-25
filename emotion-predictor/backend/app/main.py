from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
from sklearn.cluster import KMeans
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import pickle
import io
import logging
import json
import webcolors
import math

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the color names JSON file
try:
    with open('app/data/colors_name.json') as json_file:
        color_names = json.load(json_file)
except FileNotFoundError:
    logger.error("colors_name.json file not found")
    color_names = {}

# Load your trained model
try:
    model = load_model('app/model/emotion_model.h5')
except FileNotFoundError:
    logger.error("emotion_model.h5 file not found")
    model = None

# Load your scaler
try:
    with open('app/model/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
except FileNotFoundError:
    logger.error("scaler.pkl file not found")
    scaler = None

# Define the emotion labels
emotion_labels = [
    "Pure_Simplicity", "Untamed", "Young", "Fresh", "Ethnic", 
    "Tranquil", "Elegant", "Alluring", "Youthful", "Quiet", 
    "Intellectual", "Masculine", "Elaborate", "Casual", "Interesting", 
    "Mellow", "Subtle", "Natural", "Deep", "Mysterious", 
    "Calm", "Conservative", "Decorative", "Pastoral", "Sleek", 
    "Graceful", "Serious", "Soft", "Robust", "Solemn", 
    "Dignified", "Peaceful", "Modern", "Sophisticated", "Brilliant", 
    "Wild", "Composed", "Majestic", "Agile", "Enjoyable", 
    "Open", "Noble", "Progressive", "Dreamy", "Heavy", 
    "Provincial", "Classic", "Traditional", "Crystalline", "Sturdy", 
    "Bright", "Chic", "Rich", "Extravagant", "Feminine", 
    "Pretty", "Tender", "Pleasant", "Dynamic", "Emotional", 
    "Mature", "Glossy", "Sedate", "Luxurious", "Sweet", 
    "Diligent", "Lively", "Complex", "Refined", "Tasteful", 
    "Provocative", "Neat", "Active", "Showy", "Refreshing", 
    "Merry", "Supple", "Sharp", "Distinguished", "Romantic", 
    "Sublime", "Gentle", "Cute", "Domestic", "Free", 
    "Intimate", "Healthy", "Placid", "Amusing", "Flamboyant", 
    "Lighthearted", "Abundant", "Friendly", "Nostalgic", "Delicate", 
    "Delicious", "Aromatic", "Stylish", "Tropical", "Vigorous", 
    "Polished", "Precious", "Cheerful", "Childlike", "Metallic", 
    "Charming", "Colourful", "Restful", "Earnest", "Rational", 
    "Fashionable", "Grand", "Plain", "Mild", "Festive", 
    "Innocent", "Amiable", "Rustic", "Striking", "Generous", 
    "Formal", "Sunny", "Eminent", "Wholesome", "Cultured", 
    "Authoritative", "Vivid", "Smooth", "Bold", "Intrepid", 
    "Dazzling", "Cultivated", "Dry", "Gorgeous", "Citrus", 
    "Substantial", "Modest", "Clear", "Bitter", "Aristocratic", 
    "Fascinating", "Dapper", "Fiery", "Intense", "Hot", 
    "Adult", "Speedy", "Forceful", "Magnificent", "Light", 
    "Fruitful", "Nimble", "Fleet", "Joyful", "Relaxed", 
    "Splendid", "Lightl", "Artistic", "Bitterl", "Genteed", 
    "Happy", "None" 
]

def closest_colour(requested_colour):
    min_colors = {}
    for key, name in color_names.items():
        r_c, g_c, b_c = webcolors.hex_to_rgb("#"+key)
        rd = (r_c - requested_colour[0]) ** 2
        gd = (g_c - requested_colour[1]) ** 2
        bd = (b_c - requested_colour[2]) ** 2
        min_colors[math.sqrt(rd + gd + bd)] = name
    return min_colors[min(min_colors.keys())]

def get_colour_name(requested_hex):
    try:
        closest_name = actual_name = color_names[requested_hex.lstrip('#')]
    except KeyError:
        closest_name = closest_colour(webcolors.hex_to_rgb(requested_hex))
        actual_name = None
    return actual_name, closest_name

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
    if model is None or scaler is None:
        raise HTTPException(status_code=500, detail="Model or scaler not loaded")

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
        predicted_emotion = emotion_labels[predicted_class]

        # Get color names
        color_names = [get_colour_name(webcolors.rgb_to_hex(color))[1] for color in colors]
    except Exception as e:
        logger.error(f"Error making prediction: {e}")
        raise HTTPException(status_code=500, detail="Error making prediction")

    return {
        "predicted_emotion": predicted_emotion,
        "colors": colors.tolist(),
        "color_names": color_names
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)