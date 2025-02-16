import logging
import os
import json
from scripts.data_preparation import fetch_page_data, create_translation_dataset, create_correction_dataset
from scripts.model_training import fine_tune_translation_model, train_ocr_correction_model
from scripts.ocr_and_translation import process_image

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Define fixed paths for images, XML data, models, and logs
    images_directory = 'data/images/'
    xml_directory = 'data/xml/'
    translation_model_directory = 'models/fine_tuned_model/'
    correction_model_directory = 'models/ocr_correction_model/'
    local_db_file = 'local_database.json'

    # Ensure required directories exist
    os.makedirs(images_directory, exist_ok=True)
    os.makedirs(xml_directory, exist_ok=True)
    os.makedirs(translation_model_directory, exist_ok=True)
    os.makedirs(correction_model_directory, exist_ok=True)

    # Initialize local "database" using JSON storage
    db = MockFirestoreClient(local_db_file)

    # Step 0: Fine-tune the models if they are not already trained
    if not os.path.exists(translation_model_directory) or not os.path.exists(correction_model_directory) or not os.listdir(translation_model_directory) or not os.listdir(correction_model_directory):
        logger.info("Models not found. Starting training...")
        try:
            # Fetch all Latin and English texts from XML files
            latin_texts_xml, english_texts_xml = fetch_page_data(xml_directory)
            # Prepare datasets for translation and OCR correction
            translation_dataset = create_translation_dataset(latin_texts_xml, english_texts_xml)
            correction_dataset = create_correction_dataset(latin_texts_xml)
            # Train models
            fine_tune_translation_model(translation_dataset, translation_model_directory)
            train_ocr_correction_model(correction_dataset, correction_model_directory)
            logger.info("Model training completed.")
        except Exception as e:
            logger.error(f"Model training failed: {e}")
            return
    else:
        logger.info("Models found. Skipping training.")

    # Load the models
    try:
        from transformers import MarianMTModel, MarianTokenizer
        translation_tokenizer = MarianTokenizer.from_pretrained(translation_model_directory)
        translation_model = MarianMTModel.from_pretrained(translation_model_directory)
        from transformers import T5ForConditionalGeneration, T5Tokenizer
        correction_tokenizer = T5Tokenizer.from_pretrained(correction_model_directory)
        correction_model = T5ForConditionalGeneration.from_pretrained(correction_model_directory)
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        return

    # Process each image in the images directory
    image_files = [f for f in os.listdir(images_directory) if os.path.isfile(os.path.join(images_directory, f))]
    if not image_files:
        logger.error("No image files found in the images directory.")
        return

    for image_file in image_files:
        image_path = os.path.join(images_directory, image_file)
        image_name = os.path.basename(image_path)

        # Extract page number from the image filename (expects format: page_<number>.png)
        try:
            page_number = int(os.path.splitext(image_name)[0].split('_')[1])
        except (IndexError, ValueError):
            logger.error(f"Invalid image filename format: {image_name}. Expected format: 'page_<number>.png'")
            continue

        # Check if translation already exists in the local database
        try:
            doc = db.get_document(f"page_{page_number}")
            if doc is not None:
                logger.info(f"Translation for page {page_number} found. Skipping processing.")
                continue
        except Exception as e:
            logger.error(f"Failed to retrieve data from local storage: {e}")
            continue

        # Fetch the corresponding Latin and English texts from XML (for logging or comparison purposes)
        try:
            latin_text_page, english_text_page = fetch_page_data(xml_directory, page_number)
        except Exception as e:
            logger.error(f"Failed to fetch XML data for page {page_number}: {e}")
            continue

        # Process the image: enhancement, OCR, correction, and translation
        logger.info(f"Processing page {page_number}...")
        try:
            english_text = process_image(
                image_path,
                correction_model,
                correction_tokenizer,
                translation_model,
                translation_tokenizer,
                page_number
            )
            # Store the translation locally
            db.set_document(f"page_{page_number}", {'english_text': english_text})
        except Exception as e:
            logger.error(f"Processing failed for page {page_number}: {e}")
            continue

        logger.info(f"Processing completed for page {page_number}.")

def run():
    logger.info("Starting program...")
    main()

class MockFirestoreClient:
    """
    A simple JSON-based storage to mimic Firestore functionality locally.
    """
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

if __name__ == "__main__":
    run()
