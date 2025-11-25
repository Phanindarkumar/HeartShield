import os
import json
import uuid
from fastapi import FastAPI, Request, Form, File, UploadFile, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict
from models.train_model import predict_risk
from utils.ocr_processor import process_medical_document
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
app = FastAPI(title="HeartShield: Heart Disease Prediction System")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")
class PatientData(BaseModel):
    age: int
    sex: int
    bmi: float
    systolic_bp: int
    diastolic_bp: int
    glucose: float
    smoking: int
    alcohol: int
    physical_activity: float
    sleep_time: float
class PredictionResponse(BaseModel):
    status: str
    risk_level: str
    probability: float
    recommendations: List[str]
os.makedirs("uploads", exist_ok=True)
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
@app.post("/predict", response_model=PredictionResponse)
async def predict(patient_data: PatientData):
    try:
        data_dict = patient_data.dict()
        model_input = {
            'age': data_dict['age'],
            'sex': data_dict['sex'],
            'cp': 0,  
            'trestbps': data_dict['systolic_bp'],  
            'chol': 200,  
            'fbs': 1 if data_dict['glucose'] > 140 else 0,  
            'restecg': 0,  
            'thalach': 150,  
            'exang': 1 if data_dict['physical_activity'] < 2 else 0,  
            'oldpeak': 0.0,  
            'slope': 1,  
            'ca': 0,  
            'thal': 2  
        }
        prediction, probability = predict_risk(model_input)
        if probability >= 0.7:
            risk_level = "High"
            recommendations = [
                "Consult a cardiologist immediately",
                "Consider lifestyle changes: exercise, healthy diet",
                "Monitor your blood pressure regularly"
            ]
        elif probability >= 0.4:
            risk_level = "Medium"
            recommendations = [
                "Schedule a check-up with your doctor",
                "Consider lifestyle improvements",
                "Monitor your heart health indicators"
            ]
        else:
            risk_level = "Low"
            recommendations = [
                "Maintain your healthy lifestyle",
                "Regular exercise is recommended",
                "Annual check-ups are advised"
            ]
        return {
            "status": "success",
            "risk_level": risk_level,
            "probability": float(probability),
            "recommendations": recommendations
        }
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))
@app.post("/upload")
async def upload_file(
    files: List[UploadFile] = File(...),
    manualData: Optional[str] = Form(None)
):
    logger.info("=== File Upload Endpoint Hit ===")
    logger.info(f"Received {len(files)} files")
    try:
        extracted_data = {}
        for file in files:
            logger.info(f"\nProcessing file: {file.filename}")
            file_extension = os.path.splitext(file.filename)[1].lower()
            logger.info(f"File extension: {file_extension}")
            if file_extension not in ['.pdf', '.jpg', '.jpeg', '.png']:
                logger.warning(f"Skipping unsupported file type: {file_extension}")
                continue
            temp_file_path = f"uploads/{str(uuid.uuid4())}{file_extension}"
            try:
                with open(temp_file_path, "wb") as buffer:
                    buffer.write(await file.read())
                logger.info(f"File saved to {temp_file_path}")
                logger.info("Processing with OCR...")
                result = process_medical_document(temp_file_path)
                logger.info(f"OCR Result: {json.dumps(result, indent=2)}")
                if result.get("status") == "success" and result.get("data"):
                    extracted_data.update(result["data"])
                    logger.info("Successfully extracted data from document")
            except Exception as e:
                logger.error(f"Error processing file {file.filename}: {str(e)}", exc_info=True)
                continue
            finally:
                try:
                    if os.path.exists(temp_file_path):
                        os.remove(temp_file_path)
                        logger.info(f"Temporary file {temp_file_path} removed")
                except Exception as e:
                    logger.error(f"Error removing temporary file {temp_file_path}: {str(e)}")
        manual_data = json.loads(manualData) if manualData else {}
        logger.info(f"Manual data: {json.dumps(manual_data, indent=2)}")
        merged_data = {**extracted_data, **manual_data}
        logger.info(f"Merged data: {json.dumps(merged_data, indent=2)}")
        return {
            "status": "success",
            "data": merged_data,
            "message": "Files processed successfully"
        }
    except Exception as e:
        logger.error(f"Error in upload endpoint: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=400,
            detail=f"Failed to process upload: {str(e)}"
        )
@app.post("/predict", response_model=PredictionResponse)
async def predict(patient_data: PatientData):
    try:
        data_dict = patient_data.dict()
        prediction, probability = predict_risk(data_dict)
        if probability >= 0.7:
            risk_level = "High"
            recommendations = [
                "Consult a cardiologist immediately",
                "Consider lifestyle changes: exercise, healthy diet",
                "Monitor your blood pressure and glucose levels regularly"
            ]
        elif probability >= 0.4:
            risk_level = "Medium"
            recommendations = [
                "Schedule a check-up with your doctor",
                "Consider lifestyle improvements",
                "Monitor your heart health indicators"
            ]
        else:
            risk_level = "Low"
            recommendations = [
                "Maintain your healthy lifestyle",
                "Regular exercise is recommended",
                "Annual check-ups are advised"
            ]
        return {
            "status": "success",
            "risk_level": risk_level,
            "probability": float(probability),
            "probability_percentage": f"{probability*100:.1f}%",
            "recommendations": recommendations
        }
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)