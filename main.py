import logging
import os
import json
from scripts.data_preparation import fetch_page_data, create_translation_dataset, create_correction_dataset
from scripts.model_training import fine_tune_translation_model, train_ocr_correction_model
from scripts.ocr_and_translation import process_image
from rapidfuzz import fuzz

# Configure logging with timestamp
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class LocalDB:
    """A simple JSON-based local storage for results."""
    def __init__(self, data_file):
        self.data_file = data_file
        if os.path.exists(self.data_file):
            with open(self.data_file, 'r') as f:
                self.data = json.load(f)
        else:
            self.data = {}

    def get_document(self, doc_id):
        return self.data.get(doc_id)

    def set_document(self, doc_id, value):
        self.data[doc_id] = value
        with open(self.data_file, 'w') as f:
            json.dump(self.data, f, indent=4)

def main():
    # Define paths
    images_directory = 'data/Book_pages/'
    xml_directory = 'data/xml/'
    translation_model_directory = 'models/fine_tuned_model/'
    correction_model_directory = 'models/ocr_correction_model/'
    local_db_file = 'local_database.json'

    # Ensure required directories exist
    os.makedirs(images_directory, exist_ok=True)
    os.makedirs(xml_directory, exist_ok=True)
    os.makedirs(translation_model_directory, exist_ok=True)
    os.makedirs(correction_model_directory, exist_ok=True)

    # Initialize local JSON database
    db = LocalDB(local_db_file)

    # Train models if they are not already present
    if (not os.listdir(translation_model_directory) or not os.listdir(correction_model_directory)):
        logger.info("Models not found. Starting training...")
        try:
            latin_texts_xml, english_texts_xml = fetch_page_data(xml_directory)
            translation_dataset = create_translation_dataset(latin_texts_xml, english_texts_xml)
            correction_dataset = create_correction_dataset(latin_texts_xml)
            fine_tune_translation_model(translation_dataset, translation_model_directory)
            train_ocr_correction_model(correction_dataset, correction_model_directory)
            logger.info("Model training completed.")
        except Exception as e:
            logger.error(f"Model training failed: {e}")
            return
    else:
        logger.info("Models found. Skipping training.")

    # Process each image file in the images directory
    image_files = [f for f in os.listdir(images_directory) if os.path.isfile(os.path.join(images_directory, f))]
    if not image_files:
        logger.error("No image files found in the images directory.")
        return

    for image_file in image_files:
        image_path = os.path.join(images_directory, image_file)
        image_name = os.path.basename(image_file)  # e.g., "prayerbook_0015.jpg"
        filename_without_ext = os.path.splitext(image_name)[0]  # "prayerbook_0015"
        parts = filename_without_ext.split('_')
        if len(parts) < 2:
            logger.error(f"Filename format incorrect: {image_name}. Expected format: 'prayerbook_XXXX.jpg'")
            continue
        page_number_str = parts[-1]
        try:
            page_number = int(page_number_str)
        except Exception as e:
            logger.error(f"Invalid page number in filename {image_name}: {e}")
            continue

        doc_id = f"p{page_number:04d}"  # e.g., "p0015"
        if db.get_document(doc_id):
            logger.info(f"Results for page {doc_id} already exist. Skipping.")
            continue

        # Fetch corresponding XML data for the page
        try:
            xml_latin_list, xml_english_list = fetch_page_data(xml_directory, page_number)
            if not xml_latin_list or not xml_english_list:
                logger.error(f"Missing XML data for page {doc_id}.")
                continue
            xml_latin_text = xml_latin_list[0]
            xml_english_text = xml_english_list[0]
        except Exception as e:
            logger.error(f"Error fetching XML for page {doc_id}: {e}")
            continue

        logger.info(f"Processing page {doc_id}...")
        try:
            # Process image: enhancement, OCR extraction, correction, and translation
            results = process_image(
                image_path,
                correction_model_directory,
                translation_model_directory,
                page_number
            )
            raw_ocr = results.get("raw_ocr")
            corrected_ocr = results.get("corrected_ocr")
            translated_text = results.get("translated_text")
        except Exception as e:
            logger.error(f"Processing failed for page {doc_id}: {e}")
            continue

        # Compute similarity metrics
        raw_similarity = fuzz.ratio(xml_latin_text, raw_ocr)
        corrected_similarity = fuzz.ratio(xml_latin_text, corrected_ocr)
        translation_similarity = fuzz.ratio(xml_english_text, translated_text)
        logger.info(f"Page {doc_id} - Raw OCR similarity: {raw_similarity}")
        logger.info(f"Page {doc_id} - Corrected OCR similarity: {corrected_similarity}")
        logger.info(f"Page {doc_id} - Translation similarity: {translation_similarity}")

        # Store all outputs and metrics locally
        db.set_document(doc_id, {
            "raw_ocr": raw_ocr,
            "corrected_ocr": corrected_ocr,
            "translated_text": translated_text,
            "xml_latin_text": xml_latin_text,
            "xml_english_text": xml_english_text,
            "raw_similarity": raw_similarity,
            "corrected_similarity": corrected_similarity,
            "translation_similarity": translation_similarity
        })
        logger.info(f"Processing completed for page {doc_id}.")

if __name__ == "__main__":
    logger.info("Starting program...")
    main()
