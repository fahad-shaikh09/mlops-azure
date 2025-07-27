import json
import joblib
import numpy as np
import pandas as pd
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def init():
    """Initialize the model for scoring"""
    global model, scaler, label_encoder
    
    try:
        # Get model path from environment variable (set by Azure ML v2)
        model_dir = os.environ.get('AZUREML_MODEL_DIR', '.')
        
        # Load the main model
        model_path = os.path.join(model_dir, 'trained-model.pkl')
        if not os.path.exists(model_path):
            # Fallback to different naming patterns
            possible_paths = [
                os.path.join(model_dir, 'model.pkl'),
                os.path.join(model_dir, 'trained_model.pkl'),
                # Try in subdirectories
                os.path.join(model_dir, 'trained-model', 'trained-model.pkl'),
                os.path.join(model_dir, 'trained-model', 'model.pkl')
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    model_path = path
                    break
            else:
                # List all files in model directory for debugging
                files = []
                for root, dirs, filenames in os.walk(model_dir):
                    for filename in filenames:
                        files.append(os.path.join(root, filename))
                logger.error(f"Model file not found. Available files: {files}")
                raise FileNotFoundError(f"Model file not found in {model_dir}")
        
        logger.info(f"Loading model from: {model_path}")
        model = joblib.load(model_path)
        
        # Initialize optional preprocessors to None
        scaler = None
        label_encoder = None
        
        # Try to load preprocessors if they exist
        try:
            scaler_path = os.path.join(model_dir, 'scaler.pkl')
            if os.path.exists(scaler_path):
                scaler = joblib.load(scaler_path)
                logger.info("Scaler loaded successfully")
            else:
                logger.info("No scaler found, skipping scaling step")
        except Exception as e:
            logger.warning(f"Could not load scaler: {e}")
        
        try:
            label_encoder_path = os.path.join(model_dir, 'label_encoder.pkl')
            if os.path.exists(label_encoder_path):
                label_encoder = joblib.load(label_encoder_path)
                logger.info("Label encoder loaded successfully")
            else:
                logger.info("No label encoder found, using raw predictions")
        except Exception as e:
            logger.warning(f"Could not load label encoder: {e}")
        
        logger.info("Model initialization completed successfully")
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

def run(raw_data):
    """Score the model with input data"""
    try:
        # Parse input data
        data = json.loads(raw_data)
        
        # Convert to DataFrame
        if isinstance(data, dict):
            if 'data' in data:
                df = pd.DataFrame(data['data'])
            else:
                df = pd.DataFrame([data])
        else:
            df = pd.DataFrame(data)
        
        logger.info(f"Received data with shape: {df.shape}")
        
        # Preprocess the data (apply same transformations as training)
        df_processed = preprocess_data(df)
        
        # Scale features if scaler is available
        if scaler is not None:
            X_scaled = scaler.transform(df_processed)
        else:
            X_scaled = df_processed.values
        
        # Make predictions
        predictions = model.predict(X_scaled)
        prediction_probs = model.predict_proba(X_scaled) if hasattr(model, 'predict_proba') else None
        
        # Decode predictions if label encoder was used
        if label_encoder is not None:
            try:
                predictions_decoded = label_encoder.inverse_transform(predictions)
            except Exception as e:
                logger.warning(f"Could not decode predictions: {e}")
                predictions_decoded = predictions
        else:
            predictions_decoded = predictions
        
        # Prepare response
        response = {
            'predictions': predictions_decoded.tolist(),
            'probabilities': prediction_probs.tolist() if prediction_probs is not None else None
        }
        
        logger.info(f"Predictions generated for {len(predictions)} samples")
        return json.dumps(response)
        
    except Exception as e:
        error_msg = f"Error during prediction: {str(e)}"
        logger.error(error_msg)
        return json.dumps({'error': error_msg})

def preprocess_data(df):
    """Apply preprocessing steps to input data"""
    # Handle missing values (same strategy as training)
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        df[col] = df[col].fillna(df[col].median())
    
    categorical_columns = df.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        mode_val = df[col].mode()
        fill_val = mode_val[0] if not mode_val.empty else 'Unknown'
        df[col] = df[col].fillna(fill_val)
    
    # Feature engineering (same as training)
    for col in numeric_columns:
        df[f'{col}_squared'] = df[col] ** 2
    
    # One-hot encode categorical variables
    df_encoded = pd.get_dummies(df, drop_first=True)
    
    return df_encoded