import numpy as np
import pandas as pd
import logging
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
def generate_synthetic_data(n_samples=1000):
    np.random.seed(42)
    n_samples = 1000
    age = np.random.randint(30, 81, size=n_samples)
    sex = np.random.binomial(1, 0.55, size=n_samples)
    systolic_bp = np.random.normal(120, 20, size=n_samples).astype(int)
    diastolic_bp = np.random.normal(80, 10, size=n_samples).astype(int)
    cholesterol = np.random.randint(150, 301, size=n_samples)
    glucose = np.random.randint(70, 201, size=n_samples)
    bmi = np.random.normal(25, 5, size=n_samples).clip(18, 45)
    smoking = np.random.binomial(1, 0.3, size=n_samples)
    physical_activity = np.random.normal(3, 2, size=n_samples).clip(0, 10)
    sleep_time = np.random.normal(7, 1.5, size=n_samples).clip(4, 10)
    risk_score = (
        0.05 * (age - 50) + 
        0.5 * sex + 
        0.02 * (systolic_bp - 120) + 
        0.01 * (cholesterol - 200) +
        0.01 * (glucose - 100) +
        0.5 * smoking
    )
    probability = 1 / (1 + np.exp(-risk_score))
    target = np.random.binomial(1, probability)
    data = {
        'age': age,
        'sex': sex,
        'systolic_bp': systolic_bp,
        'diastolic_bp': diastolic_bp,
        'cholesterol': cholesterol,
        'glucose': glucose,
        'bmi': bmi,
        'smoking': smoking,
        'physical_activity': physical_activity,
        'sleep_time': sleep_time,
        'target': target
    }
    df = pd.DataFrame(data)
    return df
def train_model():
    logger.info("Generating synthetic data...")
    df = generate_synthetic_data()
    X = df.drop('target', axis=1)
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    logger.info("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    logger.info("Training Random Forest model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    train_accuracy = model.score(X_train_scaled, y_train)
    test_accuracy = model.score(X_test_scaled, y_test)
    logger.info(f"Training accuracy: {train_accuracy:.2f}")
    logger.info(f"Test accuracy: {test_accuracy:.2f}")
    return model, scaler
def predict_risk(patient_data: dict) -> tuple:
    try:
        logger.info("Starting prediction for patient data")
        logger.debug(f"Raw patient data: {patient_data}")
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(script_dir, 'heart_disease_model.pkl')
        scaler_path = os.path.join(script_dir, 'scaler.pkl')
        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            logger.info("Model or scaler not found. Training new model...")
            model, scaler = train_model()
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            joblib.dump(model, model_path)
            joblib.dump(scaler, scaler_path)
            logger.info(f"Model and scaler saved to {script_dir}")
        else:
            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
            logger.info("Loaded existing model and scaler")
        default_values = {
            'age': 50,
            'sex': 1,  
            'systolic_bp': 120,
            'diastolic_bp': 80,
            'cholesterol': 200,
            'glucose': 100,
            'bmi': 25,
            'smoking': 0,  
            'physical_activity': 3,  
            'sleep_time': 7  
        }
        features = [patient_data.get(key, default) for key, default in default_values.items()]
        features_array = np.array(features).reshape(1, -1)
        features_scaled = scaler.transform(features_array)
        prediction = int(model.predict(features_scaled)[0])
        probability = float(model.predict_proba(features_scaled)[0][1])
        logger.info(f"Prediction complete. Risk: {'High' if prediction == 1 else 'Low'}, Probability: {probability:.2f}")
        return prediction, probability
    except Exception as e:
        logger.error(f"Error in predict_risk: {str(e)}", exc_info=True)
        return 0, 0.5
if __name__ == "__main__":
    try:
        logger.info("Starting model training...")
        model, scaler = train_model()
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(script_dir, 'heart_disease_model.pkl')
        scaler_path = os.path.join(script_dir, 'scaler.pkl')
        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)
        logger.info(f"Model and scaler saved to {script_dir}")
        print("Model trained and saved successfully!")
    except Exception as e:
        logger.error(f"Error in training: {str(e)}", exc_info=True)
        raise