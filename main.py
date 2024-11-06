import logging
import os
from scripts.data_preparation import fetch_page_data, create_translation_dataset, create_correction_dataset
from scripts.model_training import fine_tune_translation_model, train_ocr_correction_model
from scripts.ocr_and_translation import process_image, store_translation
from google.cloud import firestore
from google.auth.exceptions import DefaultCredentialsError
import mlflow

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Paths
    images_directory = 'data/images/'
    xml_directory = 'data/xml/'
    translation_model_directory = 'models/fine_tuned_model/'
    correction_model_directory = 'models/ocr_correction_model/'

    # Initialize Firestore
    try:
        db = firestore.Client()
    except DefaultCredentialsError as e:
        logger.error(f"Google Cloud credentials not found: {e}")
        return
    except Exception as e:
        logger.error(f"Failed to initialize Firestore client: {e}")
        return

    # Start MLflow experiment
    mlflow.set_experiment("Latin to English Translation with OCR Correction")
    with mlflow.start_run(run_name="Main Script Execution"):
        # Step 0: Fine-tune the models (if not already done)
        if not os.path.exists(translation_model_directory) or not os.path.exists(correction_model_directory):
            os.makedirs(translation_model_directory, exist_ok=True)
            os.makedirs(correction_model_directory, exist_ok=True)
            logger.info("Models not found. Starting training...")
            try:
                # Fetch all Latin and English texts from XML files
                latin_texts_xml, english_texts_xml = fetch_page_data(xml_directory)
                # Prepare datasets
                translation_dataset = create_translation_dataset(latin_texts_xml, english_texts_xml)
                correction_dataset = create_correction_dataset(latin_texts_xml)
                # Train models
                fine_tune_translation_model(translation_dataset, translation_model_directory)
                train_ocr_correction_model(correction_dataset, correction_model_directory)
                # Log models
                mlflow.log_artifacts(translation_model_directory, artifact_path="fine_tuned_model")
                mlflow.log_artifacts(correction_model_directory, artifact_path="ocr_correction_model")
            except Exception as e:
                logger.error(f"Model training failed: {e}")
                mlflow.end_run(status="FAILED")
                return
        else:
            logger.info("Models found. Skipping training.")
            mlflow.log_artifact(translation_model_directory, artifact_path="fine_tuned_model")
            mlflow.log_artifact(correction_model_directory, artifact_path="ocr_correction_model")

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
            mlflow.end_run(status="FAILED")
            return

        # Process each image in the images directory
        image_files = [f for f in os.listdir(images_directory) if os.path.isfile(os.path.join(images_directory, f))]
        if not image_files:
            logger.error("No image files found in the images directory.")
            mlflow.end_run(status="FAILED")
            return

        for image_file in image_files:
            with mlflow.start_run(run_name=f"Processing {image_file}", nested=True):
                image_path = os.path.join(images_directory, image_file)
                image_name = os.path.basename(image_path)

                # Extract page number from the image filename
                try:
                    page_number = int(os.path.splitext(image_name)[0].split('_')[1])
                except (IndexError, ValueError) as e:
                    logger.error(f"Invalid image filename format: {image_name}. Expected format: 'page_<number>.png'")
                    mlflow.end_run(status="FAILED")
                    continue

                # Check if translation already exists in the database
                try:
                    doc_ref = db.collection('translations').document(f"page_{page_number}")
                    doc = doc_ref.get()
                    if doc.exists:
                        data = doc.to_dict()
                        logger.info(f"Translation for page {page_number} found in database. Skipping processing.")
                        continue
                except Exception as e:
                    logger.error(f"Failed to retrieve data from Firestore: {e}")
                    mlflow.end_run(status="FAILED")
                    continue

                # Fetch the corresponding Latin text and translation from XML files
                try:
                    latin_text_page, english_text_page = fetch_page_data(xml_directory, page_number)
                except Exception as e:
                    logger.error(f"Failed to fetch data for page {page_number} from XML files: {e}")
                    mlflow.end_run(status="FAILED")
                    continue

                # Process the image (enhancement, OCR, correction, translation)
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
                    # Store translation in the database
                    store_translation(db, f"page_{page_number}", english_text)
                except Exception as e:
                    logger.error(f"Processing failed for page {page_number}: {e}")
                    mlflow.end_run(status="FAILED")
                    continue

                logger.info(f"Processing completed for page {page_number}.")
                mlflow.end_run(status="FINISHED")

    mlflow.end_run(status="FINISHED")

if __name__ == "__main__":
    main()
