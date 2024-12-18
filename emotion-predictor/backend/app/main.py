import tensorflow as tf
import numpy as np
import cv2
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
import pickle
import logging
import os
import webcolors
import traceback
from typing import Dict, List, Optional
import sys

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('app.log')
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Emotion Predictor API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Constants
SUPPORTED_MODELS = {"mlp", "svm", "random_forest"}
MODEL_FILES = {
    "mlp": "emotion_model.h5",
    "svm": "svm_model.pkl",
    "random_forest": "random_forest_model.pkl"
}

current_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(current_dir, 'model')

class ModelLoader:
    @staticmethod
    def load_pickle_model(file_path: str, model_type: str) -> Optional[object]:
        """Load a model from pickle file with error handling"""
        try:
            if not os.path.exists(file_path):
                logger.error(f"{model_type} file not found at {file_path}")
                return None
                
            with open(file_path, 'rb') as f:
                model = pickle.load(f)
                
            if isinstance(model, LabelEncoder):
                if not hasattr(model, 'classes_'):
                    logger.error(f"Loaded LabelEncoder has no classes_ attribute")
                    return None
                logger.info(f"Loaded LabelEncoder with classes: {model.classes_}")
            
            logger.info(f"{model_type} loaded successfully from {file_path}")
            return model
            
        except Exception as e:
            logger.error(f"Error loading {model_type}: {str(e)}\n{traceback.format_exc()}")
            return None

    @staticmethod
    def load_tf_model(file_path: str) -> Optional[tf.keras.Model]:
        """Load a TensorFlow model with error handling"""
        try:
            if not os.path.exists(file_path):
                logger.error(f"TF model file not found at {file_path}")
                return None
                
            model = tf.keras.models.load_model(file_path)
            logger.info("TensorFlow model loaded successfully")
            return model
            
        except Exception as e:
            logger.error(f"Error loading TF model: {str(e)}\n{traceback.format_exc()}")
            return None

class ImageProcessor:
    @staticmethod
    def extract_colors(image: np.ndarray, n_colors: int = 3) -> np.ndarray:
        """Extract dominant colors from image"""
        try:
            pixels = image.reshape(-1, 3)
            kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
            kmeans.fit(pixels)
            return kmeans.cluster_centers_.astype(int)
        except Exception as e:
            logger.error(f"Error extracting colors: {str(e)}")
            raise

    @staticmethod
    def get_color_name(rgb_color: np.ndarray) -> str:
        """Get the name of a color from RGB values"""
        try:
            return webcolors.rgb_to_name(tuple(rgb_color))
        except ValueError:
            min_colors = {}
            for hex_value, name in webcolors.CSS3_HEX_TO_NAMES.items():
                r_c, g_c, b_c = webcolors.hex_to_rgb(hex_value)
                rd, gd, bd = (r_c - rgb_color[0])**2, (g_c - rgb_color[1])**2, (b_c - rgb_color[2])**2
                min_colors[(rd + gd + bd)] = name
            return min_colors[min(min_colors.keys())]

class PredictionService:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.label_encoder = None
        self.load_all_models()

    def load_all_models(self):
        """Load all required models and components"""
        try:
            # Load TF model and its scaler
            self.models["mlp"] = ModelLoader.load_tf_model(
                os.path.join(model_dir, MODEL_FILES["mlp"])
            )
            self.scalers["mlp"] = ModelLoader.load_pickle_model(
                os.path.join(model_dir, 'mlp_scaler.pkl'),
                "MLP Scaler"
            )

            # Load sklearn models (SVM already includes scaler in pipeline)
            self.models["svm"] = ModelLoader.load_pickle_model(
                os.path.join(model_dir, MODEL_FILES["svm"]),
                "SVM"
            )

            # Load Random Forest (doesn't need scaling)
            self.models["random_forest"] = ModelLoader.load_pickle_model(
                os.path.join(model_dir, MODEL_FILES["random_forest"]),
                "Random Forest"
            )

            # Load label encoder (common for all models)
            self.label_encoder = ModelLoader.load_pickle_model(
                os.path.join(model_dir, 'label_encoder.pkl'),
                "Label Encoder"
            )

            # Validate components
            if not self.label_encoder:
                raise ValueError("Failed to load label encoder")

            if not self.models["mlp"] or not self.scalers["mlp"]:
                logger.error("Failed to load MLP model or its scaler")

            logger.info("All models and components loaded successfully")
            
        except Exception as e:
            logger.error(f"Error initializing models: {str(e)}")
            raise

    def prepare_input(self, input_data: np.ndarray, model_type: str) -> np.ndarray:
        """Prepare input data based on model type"""
        try:
            if model_type == "mlp":
                if self.scalers["mlp"] is None:
                    raise ValueError("MLP scaler not available")
                return self.scalers["mlp"].transform(input_data)
            elif model_type == "svm":
                # SVM model has scaler in pipeline, return as is
                return input_data
            elif model_type == "random_forest":
                # Random Forest doesn't need scaling
                return input_data
            else:
                raise ValueError(f"Unknown model type: {model_type}")
        except Exception as e:
            logger.error(f"Error preparing input: {str(e)}")
            raise

    def get_predictions(self, model_input: np.ndarray, model_type: str) -> np.ndarray:
        """Get predictions based on model type"""
        try:
            if model_type not in self.models or self.models[model_type] is None:
                raise ValueError(f"Model {model_type} is not available")

            model = self.models[model_type]
            
            # Prepare input according to model type
            prepared_input = self.prepare_input(model_input, model_type)
            
            if model_type == "mlp":
                prediction = model.predict(prepared_input)
            else:
                prediction = model.predict_proba(prepared_input)
            
            logger.debug(f"Raw prediction shape: {prediction.shape}")
            logger.debug(f"Raw prediction values: {prediction}")
            
            return prediction
            
        except Exception as e:
            logger.error(f"Error in prediction: {str(e)}\n{traceback.format_exc()}")
            raise

# Initialize prediction service
prediction_service = PredictionService()

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    model_type: str = Query("mlp", description="Model type to use for prediction")
) -> Dict:
    """
    Make emotion predictions from an image using the specified model.
    """
    logger.info(f"Received prediction request for model type: {model_type}")
    
    if model_type not in SUPPORTED_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model type. Choose from {', '.join(SUPPORTED_MODELS)}"
        )

    try:
        # Read and process image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Failed to decode image")
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Extract colors and prepare input
        colors = ImageProcessor.extract_colors(img)
        model_input = colors.flatten() / 255.0
        
        # Get predictions
        prediction = prediction_service.get_predictions(model_input.reshape(1, -1), model_type)
        
        # Process prediction results
        if len(prediction.shape) == 2:
            top_indices = prediction.argsort()[0][::-1][:2]
            top_emotions = prediction_service.label_encoder.inverse_transform(top_indices)
            top_probabilities = prediction[0][top_indices]
        else:
            raise ValueError(f"Unexpected prediction shape: {prediction.shape}")
        
        # Get color names
        color_names = [ImageProcessor.get_color_name(color) for color in colors]
        
        # Prepare response
        response = {
            "predicted_emotions": [
                {"emotion": str(top_emotions[0]), "probability": float(top_probabilities[0])},
                {"emotion": str(top_emotions[1]), "probability": float(top_probabilities[1])},
            ],
            "colors": [{"rgb": color.tolist(), "name": name} for color, name in zip(colors, color_names)],
        }
        
        logger.info("Prediction successful")
        return response

    except Exception as e:
        error_msg = str(e)
        logger.error(f"Error in prediction endpoint: {error_msg}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=error_msg)

@app.get("/health")
async def health_check() -> Dict:
    """Check API health and model availability"""
    return {
        "status": "healthy",
        "models_available": {
            model_type: prediction_service.models.get(model_type) is not None
            for model_type in SUPPORTED_MODELS
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)