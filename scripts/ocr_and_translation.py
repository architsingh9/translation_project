import pytesseract
from PIL import Image
import logging
from google.api_core.exceptions import GoogleAPIError
import mlflow
from scripts.image_enhancement import enhance_image
import os

logger = logging.getLogger(__name__)

def ocr_image(image_path):
    """
    Performs OCR on the image and extracts Latin text.
    """
    try:
        # Enhance the image
        enhanced_image_path = enhance_image(image_path)

        # Open the enhanced image
        image = Image.open(enhanced_image_path)
    except Exception as e:
        logger.error(f"Unable to process image: {e}")
        raise

    try:
        # Perform OCR using Tesseract
        extracted_text = pytesseract.image_to_string(image, lang='lat')
        if not extracted_text.strip():
            logger.warning("No text found in the image.")
            raise ValueError("No text extracted from image.")
        # Log the length of the extracted text
        mlflow.log_metric("extracted_text_length", len(extracted_text))
        return extracted_text
    except pytesseract.TesseractError as e:
        logger.error(f"OCR failed: {e}")
        raise
    finally:
        # Clean up temporary files
        if os.path.exists(enhanced_image_path):
            os.remove(enhanced_image_path)

def correct_text(ocr_text, correction_model, correction_tokenizer):
    """
    Corrects OCR errors using the OCR correction model.
    """
    try:
        inputs = correction_tokenizer(ocr_text, return_tensors="pt", padding=True, truncation=True, max_length=128)
        outputs = correction_model.generate(**inputs)
        corrected_text = correction_tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Log the length of the corrected text
        mlflow.log_metric("corrected_text_length", len(corrected_text))
        return corrected_text
    except Exception as e:
        logger.error(f"OCR correction failed: {e}")
        raise

def translate_text_custom_model(text, model, tokenizer):
    """
    Translates the given Latin text to English using the fine-tuned model.
    """
    try:
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
        translated = model.generate(**inputs)
        english_text = tokenizer.decode(translated[0], skip_special_tokens=True)
        # Log the length of the translated text
        mlflow.log_metric("translated_text_length", len(english_text))
        return english_text
    except Exception as e:
        logger.error(f"Translation failed: {e}")
        raise

def store_translation(db, document_id, english_text):
    """
    Stores the translation in Google Cloud Firestore.
    """
    try:
        doc_ref = db.collection('translations').document(document_id)
        doc_ref.set({
            'english_text': english_text
        })
        logger.info(f"Translation stored in database with document ID: {document_id}")
        # Log a success metric
        mlflow.log_metric("translation_stored", 1)
    except GoogleAPIError as e:
        logger.error(f"Failed to store data in Firestore: {e}")
        # Log a failure metric
        mlflow.log_metric("translation_stored", 0)
        raise

def process_image(image_path, correction_model, correction_tokenizer, translation_model, translation_tokenizer, page_number):
    """
    Processes the image: enhancement, OCR, correction, translation.
    """
    # Step 1: OCR to extract Latin text
    ocr_text = ocr_image(image_path)
    mlflow.log_text(ocr_text, f"ocr_text_page_{page_number}.txt")

    # Step 2: Correct OCR errors
    corrected_text = correct_text(ocr_text, correction_model, correction_tokenizer)
    mlflow.log_text(corrected_text, f"corrected_text_page_{page_number}.txt")

    # Step 3: Translate corrected text to English
    english_text = translate_text_custom_model(corrected_text, translation_model, translation_tokenizer)
    mlflow.log_text(english_text, f"translated_text_page_{page_number}.txt")

    return english_text
