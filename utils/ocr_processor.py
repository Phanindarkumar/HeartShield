import os
import re
import logging
import pytesseract
from typing import Dict, Optional, Union, List, Any
from PIL import Image
import cv2
import numpy as np
import fitz  
from pathlib import Path
import json
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
def ensure_upload_dir() -> str:
    upload_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'uploads')
    os.makedirs(upload_dir, exist_ok=True)
    return upload_dir
def preprocess_image(image: np.ndarray) -> np.ndarray:
    try:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        denoised = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)
        thresh = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
        kernel = np.ones((1, 1), np.uint8)
        dilated = cv2.dilate(opening, kernel, iterations=1)
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(dilated, -1, kernel)
        return sharpened
    except Exception as e:
        logger.error(f"Image preprocessing failed: {str(e)}", exc_info=True)
        return image
def extract_text_from_pdf(pdf_path: str) -> str:
    try:
        text = ""
        images = []
        with fitz.open(pdf_path) as doc:
            for page in doc:
                page_text = page.get_text()
                if page_text.strip():
                    text += page_text + "\n"
                else:
                    pix = page.get_pixmap()
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    images.append(img)
        if images or not text.strip():
            logger.info("No text found in PDF, attempting OCR...")
            ocr_text = ""
            for img in images:
                ocr_text += pytesseract.image_to_string(img) + "\n"
            text = ocr_text if not text.strip() else text + "\n" + ocr_text
        return text.strip()
    except Exception as e:
        logger.error(f"Error processing PDF {pdf_path}: {str(e)}", exc_info=True)
        raise Exception(f"Failed to process PDF: {str(e)}")
def extract_text_from_image(image_path: str) -> str:
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")
        processed_img = preprocess_image(img)
        custom_config = r'--oem 3 --psm 6'
        text = pytesseract.image_to_string(processed_img, config=custom_config)
        if not text.strip():
            text = pytesseract.image_to_string(Image.open(image_path), config=custom_config)
        return text.strip()
    except Exception as e:
        logger.error(f"Error extracting text from image: {str(e)}", exc_info=True)
        raise Exception(f"Failed to process image: {str(e)}")
def extract_text(file_path: str) -> str:
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        logger.info(f"Extracting text from: {file_path}")
        if file_path.lower().endswith('.pdf'):
            return extract_text_from_pdf(file_path)
        else:
            return extract_text_from_image(file_path)
    except Exception as e:
        logger.error(f"Error in extract_text: {str(e)}", exc_info=True)
        raise
def extract_measurement(text: str, patterns: List[str], unit_map: Dict[str, float], default_unit: str) -> Optional[float]:
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                value = float(match.group(1).replace(',', '.'))
                unit = match.group(2).lower() if match.lastindex > 1 else default_unit
                conversion = unit_map.get(unit, 1.0)
                return value * conversion
            except (ValueError, AttributeError, IndexError):
                continue
    return None
def extract_medical_info(text: str) -> Dict[str, Union[float, int, str]]:
    try:
        logger.info("Extracting medical information with enhanced patterns")
        text_lower = text.lower()
        result = {}
        def find_best_match(patterns, group=1, default=None, type_cast=int):
            for pattern in patterns:
                match = re.search(pattern, text_lower, re.IGNORECASE)
                if match:
                    try:
                        return type_cast(match.group(group).replace(',', '.')) if match.lastindex >= group else default
                    except (ValueError, IndexError, AttributeError) as e:
                        logger.debug(f"Error in find_best_match: {e}")
                        continue
            return default
        age = find_best_match([
            r'(?:age|ağe|yaş|yas|y\.a|y\/a|y\/o|y\/o:|y\/o\s*:)[\s:]*(\d{1,3})',
            r'\b(?:age|ağe|yaş|yas)\s*[:=]?\s*(\d{1,3})',
            r'\b(\d{1,3})\s*(?:years?|yrs?|yaş|yas|year|yr|y)\b',
            r'd\.o\.?b[:\s]*(?:\d{1,2}[\/\-\.]\d{1,2}[\/\-\.](\d{4})|\d{4})',
            r'(?:date\s*of\s*birth|doğum\s*tarihi)[:\s]*(?:\d{1,2}[\/\-\.]\d{1,2}[\/\-\.](\d{4})|\d{4})'
        ])
        if age and 0 < age <= 120:
            result['age'] = age
        if any(re.search(p, text_lower) for p in [r'\b(male|m\b|man|erkek|eril|bay|m\/f\s*m)', r'sex[:\s]*(m|male)']):
            result['sex'] = 1  
        elif any(re.search(p, text_lower) for p in [r'\b(female|f\b|woman|kadın|kız|kadin|kiz|bayan|f\/m\s*f)', r'sex[:\s]*(f|female)']):
            result['sex'] = 0  
        height = extract_measurement(
         text_lower,
         patterns=[
         r'(?:height|boy|boyu|length|hgt|h\.t)[\s:]*(\d+\.?\d*)\s*(cm|m|ft|in|"|\'\s*\d*")?',
         r'(\d+\.?\d*)\s*(?:cm|m|ft|in|")\s*(?:height|tall|boy)'
         ],
          unit_map={
         'm': 100, 'meter': 100, 'metre': 100,
         'ft': 30.48, 'feet': 30.48, "'": 30.48,
         'in': 2.54, 'inch': 2.54, '"': 2.54
         },
         default_unit='cm'  
         )
        weight = extract_measurement(
          text_lower,
          patterns=[
          r'(?:weight|ağırlık|kilo|kiloğram|kg|wgt|w\.t)[\s:]*(\d+\.?\d*)\s*(kg|g|lb|lbs|kgr)?',
          r'(\d+\.?\d*)\s*(?:kg|g|lb|lbs)\s*(?:weight|ağırlık|kilo)'
          ],
          unit_map={
           'g': 0.001,
           'lb': 0.453592, 'lbs': 0.453592, 'pound': 0.453592
            },
        default_unit='kg'  
           )
        if height and weight and height > 0:
            height_m = height / 100  
            result['bmi'] = round(weight / (height_m ** 2), 1)
        else:
            bmi = find_best_match([
                r'(?:bmi|vücut\s*kitle\s*indeksi|vki|body\s*mass\s*index)[\s:]*(\d+\.?\d*)',
                r'\bbmi\s*[=:]\s*(\d+\.?\d*)'
            ], type_cast=float)
            if bmi and 10 < bmi < 70:
                result['bmi'] = bmi
        bp_patterns = [
            r'(?:blood\s*pressure|tansiyon|kan\s*basıncı|kan\s*basinci|bp)[\s:]*(\d+)\s*[/-]\s*(\d+)',
            r'(\d+)\s*/\s*(\d+)\s*(?:mmhg|mm\s*hg|mm)',
            r'(?:systolic|sistolik|büyük\s*tansiyon|buyuk\s*tansiyon)[\s:]*(\d+).*?(?:diastolic|diyastolik|küçük\s*tansiyon|kucuk\s*tansiyon)[\s:]*(\d+)'
        ]
        bp_found = False
        for pattern in bp_patterns:
            match = re.search(pattern, text_lower, re.IGNORECASE)
            if match:
                result['systolic_bp'] = int(match.group(1))
                result['diastolic_bp'] = int(match.group(2))
                bp_found = True
                break
        if not bp_found:
            sys_bp = find_best_match([
                r'(?:systolic|sistolik|büyük\s*tansiyon|buyuk\s*tansiyon|sys|syst)[\s:]*(\d+)',
                r'bp\s*:\s*(\d+)\s*/\s*\d+',
                r'(\d+)\s*/\s*\d+\s*(?:mmhg|mm\s*hg)'
            ])
            dia_bp = find_best_match([
                r'(?:diastolic|diyastolik|küçük\s*tansiyon|kucuk\s*tansiyon|dia|diast)[\s:]*(\d+)',
                r'bp\s*:\s*\d+\s*/\s*(\d+)',
                r'\d+\s*/\s*(\d+)\s*(?:mmhg|mm\s*hg)'
            ])
            if sys_bp:
                result['systolic_bp'] = sys_bp
            if dia_bp:
                result['diastolic_bp'] = dia_bp
        glucose = find_best_match([
            r'(?:glucose|glukoz|kan\s*şekeri|kan\s*sekeri|açlık\s*kan\s*şekeri|aclik\s*kan\s*sekeri|fbs|rbs)[\s:]*(\d+\.?\d*)',
            r'(\d+)\s*(?:mg/dl|mg\s*dl|mg\s*\/\s*dl|mg)',
            r'blood\s*sugar[:\s]*(\d+)'
        ], type_cast=float)
        if glucose:
            result['glucose'] = glucose
        if re.search(r'\b(smok(?:ing|er)|sigara|tütün|tutun|cigarette|tobacco)\b.*\b(yes|current|evet|içiyorum|içiyor|kullanıyorum|kullaniyorum|present|positive)\b', text_lower):
            result['smoking'] = 1  
        elif re.search(r'\b(former|ex-?smoker|bıraktım|biraktim|bırakmış|birakmis|bıraktı|birakti|eski|quit|stopped)\b', text_lower):
            result['smoking'] = 2  
        else:
            result['smoking'] = 0  
        if re.search(r'\b(alcohol|alkol|içki|alkol\s*kullanımı)\b.*\b(never|none|hayır|yok|içmiyorum|kullanmıyorum|kullanmiyorum|no|negative)\b', text_lower):
            result['alcohol'] = 0  
        elif re.search(r'\b(alcohol|alkol|içki)\b.*\b(regular|often|düzenli|sık\s*sık|her\s*gün|haftada\s*birkaç|haftada\s*birkac|frequently|daily)\b', text_lower):
            result['alcohol'] = 2  
        else:
            result['alcohol'] = 1  
        activity = find_best_match([
            r'(?:physical\s*activity|fiziksel\s*aktivite|egzersiz|spor|hareket|exercise)[\s:]*(\d+\.?\d*)\s*(?:hours?|saat|hrs|h)',
            r'(\d+\.?\d*)\s*(?:hours?|saat|hrs|h)\s*(?:per\s*week|haftada|haftalık|haftalik)',
            r'(\d+)\s*(?:hours?|saat|hrs|h)\s*exercise'
        ], type_cast=float)
        if activity is not None:
            result['physical_activity'] = activity
        else:
            result['physical_activity'] = 0  
        sleep = find_best_match([
            r'(?:sleep|uyku|uyku\s*süresi|gece\s*uykusu|uyku\s*sürem)[\s:]*(\d+\.?\d*)\s*(?:hours?|saat|hrs|h)',
            r'(\d+\.?\d*)\s*(?:hours?|saat|hrs|h)\s*(?:of\s*sleep|uyku)',
            r'sleep\s*:\s*(\d+\.?\d*)\s*(?:hours?|saat|hrs|h)'
        ], type_cast=float)
        if sleep is not None:
            result['sleep_time'] = sleep
        else:
            result['sleep_time'] = 7  
        logger.info(f"Extracted medical data: {result}")
        return result
    except Exception as e:
        logger.error(f"Error in extract_medical_info: {str(e)}", exc_info=True)
        return {}
def process_medical_document(file_path: str) -> Dict[str, Any]:
    try:
        logger.info(f"Processing medical document: {file_path}")
        if not os.path.exists(file_path):
            error_msg = f"File not found: {file_path}"
            logger.error(error_msg)
            return {
                "status": "error",
                "message": error_msg,
                "data": {}
            }
        try:
            text = extract_text(file_path)
            if not text.strip():
                logger.warning("Extracted text is empty")
                return {
                    "status": "success",
                    "data": {},
                    "message": "Document is empty or could not be read"
                }
        except Exception as e:
            error_msg = f"Failed to extract text: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return {
                "status": "error",
                "message": error_msg,
                "data": {}
            }
        extracted_data = extract_medical_info(text)
        if not extracted_data:
            logger.warning("No medical data could be extracted from the document")
            return {
                "status": "success",
                "data": {},
                "message": "No medical data could be extracted from the document"
            }
        logger.info(f"Successfully extracted {len(extracted_data)} fields from document")
        return {
            "status": "success",
            "data": extracted_data,
            "message": "Document processed successfully"
        }
    except Exception as e:
        error_msg = f"Failed to process document: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return {
            "status": "error",
            "message": error_msg,
            "data": {}
        }