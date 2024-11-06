import logging
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch

logger = logging.getLogger(__name__)

def correct_ocr_errors(text, model_directory):
    """
    Uses the trained OCR correction model to correct errors in the OCR-extracted text.
    """
    try:
        # Load the model and tokenizer
        tokenizer = T5Tokenizer.from_pretrained(model_directory)
        model = T5ForConditionalGeneration.from_pretrained(model_directory)

        # Prepare the input
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)

        # Generate the corrected text
        outputs = model.generate(**inputs)
        corrected_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        logger.info("OCR correction completed.")
        return corrected_text

    except Exception as e:
        logger.error(f"OCR correction failed: {e}")
        raise
