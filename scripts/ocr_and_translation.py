import pytesseract
from PIL import Image
import logging
from scripts.image_enhancement import enhance_image
import os
import json

logger = logging.getLogger(__name__)

def ocr_image(image_path):
    """
    Performs OCR on the image and extracts Latin text.
    """
    try:
        enhanced_image_path = enhance_image(image_path)
        image = Image.open(enhanced_image_path)
    except Exception as e:
        logger.error(f"Unable to process image: {e}")
        raise

    try:
        extracted_text = pytesseract.image_to_string(image, lang='lat')
        if not extracted_text.strip():
            logger.warning("No text found in the image.")
            raise ValueError("No text extracted from image.")
        logger.info(f"Extracted text length: {len(extracted_text)}")
        return extracted_text
    except pytesseract.TesseractError as e:
        logger.error(f"OCR failed: {e}")
        raise
    finally:
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
        logger.info(f"Corrected text length: {len(corrected_text)}")
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
        logger.info(f"Translated text length: {len(english_text)}")
        return english_text
    except Exception as e:
        logger.error(f"Translation failed: {e}")
        raise

def store_translation_local(file_path, key, value):
    try:
        if os.path.exists(file_path):
            with open(file_path, 'r') as file:
                data = json.load(file)
        else:
            data = {}
        data[key] = value
        with open(file_path, 'w') as file:
            json.dump(data, file, indent=4)
        logger.info(f"Stored translation in '{file_path}' with key '{key}'.")
    except Exception as e:
        logger.error(f"Failed to store translation locally: {e}")

def store_translation(db, document_id, english_text):
    """
    Stores the translation locally using the provided database interface.
    """
    db.set_document(document_id, {'english_text': english_text})
    logger.info(f"Translation stored locally with document ID: {document_id}")

def process_image(image_path, correction_model, correction_tokenizer, translation_model, translation_tokenizer, page_number):
    """
    Processes the image: enhancement, OCR, correction, and translation.
    """
    os.makedirs("logs", exist_ok=True)
    ocr_text = ocr_image(image_path)
    with open(f"logs/ocr_text_page_{page_number}.txt", "w") as f:
        f.write(ocr_text)
    corrected_text = correct_text(ocr_text, correction_model, correction_tokenizer)
    with open(f"logs/corrected_text_page_{page_number}.txt", "w") as f:
        f.write(corrected_text)
    english_text = translate_text_custom_model(corrected_text, translation_model, translation_tokenizer)
    with open(f"logs/translated_text_page_{page_number}.txt", "w") as f:
        f.write(english_text)
    return english_text
