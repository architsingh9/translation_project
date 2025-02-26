import pytesseract
from PIL import Image
import logging
import os
from scripts.image_enhancement import enhance_image
from transformers import T5ForConditionalGeneration, T5Tokenizer, MarianMTModel, MarianTokenizer

logger = logging.getLogger(__name__)

def ocr_image(image_path):
    """
    Enhances the image and performs OCR to extract raw Latin text.
    """
    enhanced_image_path = enhance_image(image_path)
    try:
        image = Image.open(enhanced_image_path)
    except Exception as e:
        logger.error(f"Error opening enhanced image: {e}")
        raise
    try:
        raw_text = pytesseract.image_to_string(image, lang='lat')
        if not raw_text.strip():
            logger.warning("No text found in image after OCR.")
            raise ValueError("No text extracted.")
        logger.info(f"Raw OCR text length: {len(raw_text)}")
    except Exception as e:
        logger.error(f"OCR extraction failed: {e}")
        raise
    finally:
        if os.path.exists(enhanced_image_path):
            os.remove(enhanced_image_path)
    return raw_text

def correct_text(ocr_text, correction_model, correction_tokenizer):
    """
    Corrects OCR text using the OCR correction model.
    """
    inputs = correction_tokenizer(ocr_text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    outputs = correction_model.generate(**inputs)
    corrected_text = correction_tokenizer.decode(outputs[0], skip_special_tokens=True)
    logger.info(f"Corrected OCR text length: {len(corrected_text)}")
    return corrected_text

def translate_text(latin_text, translation_model, translation_tokenizer):
    """
    Translates Latin text to English using the translation model.
    """
    inputs = translation_tokenizer(latin_text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    outputs = translation_model.generate(**inputs)
    english_text = translation_tokenizer.decode(outputs[0], skip_special_tokens=True)
    logger.info(f"Translated text length: {len(english_text)}")
    return english_text

def process_image(image_path, correction_model_directory, translation_model_directory, page_number):
    """
    Processes the image:
    1. Enhances the image.
    2. Extracts raw OCR text.
    3. Corrects the OCR text.
    4. Translates the corrected text into English.
    Returns a dictionary with keys: "raw_ocr", "corrected_ocr", "translated_text".
    """
    os.makedirs("logs", exist_ok=True)
    
    # Step 1: Extract raw OCR text
    raw_ocr = ocr_image(image_path)
    with open(f"logs/ocr_text_p{page_number:04d}.txt", "w") as f:
        f.write(raw_ocr)
    
    # Step 2: Load OCR correction model and correct text
    correction_tokenizer = T5Tokenizer.from_pretrained(correction_model_directory)
    correction_model = T5ForConditionalGeneration.from_pretrained(correction_model_directory)
    corrected_ocr = correct_text(raw_ocr, correction_model, correction_tokenizer)
    with open(f"logs/corrected_text_p{page_number:04d}.txt", "w") as f:
        f.write(corrected_ocr)
    
    # Step 3: Load translation model and translate corrected text
    translation_tokenizer = MarianTokenizer.from_pretrained(translation_model_directory)
    translation_model = MarianMTModel.from_pretrained(translation_model_directory)
    translated_text = translate_text(corrected_ocr, translation_model, translation_tokenizer)
    with open(f"logs/translated_text_p{page_number:04d}.txt", "w") as f:
        f.write(translated_text)
    
    return {
        "raw_ocr": raw_ocr,
        "corrected_ocr": corrected_ocr,
        "translated_text": translated_text
    }
