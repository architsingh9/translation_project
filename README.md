# translation_project

# Latin to English Translation Project with Oxygen XML Editor Integration

This project translates Latin text from images into English. It includes OCR processing with image enhancement, OCR correction, custom model training, integration with Oxygen XML Editor for XML data, Google Cloud Firestore for data storage, and MLflow for experiment tracking and model management.

## **Project Structure**

- `main.py`: The main script orchestrating the entire process.
- `data/`
  - `images/`: Directory containing images with Latin text to be translated.
- `models/`
  - `fine_tuned_model/`: Directory where the fine-tuned translation model is saved.
  - `ocr_correction_model/`: Directory where the OCR correction model is saved.
- `scripts/`
  - `data_preparation.py`: Script for fetching XML data from Oxygen XML Editor and preparing the dataset.
  - `model_training.py`: Scripts for fine-tuning translation and OCR correction models.
  - `ocr_and_translation.py`: Script for OCR processing and translation.
  - `image_enhancement.py`: Script for enhancing images before OCR.
- `requirements.txt`: List of required Python packages.
- `.gitignore`: Specifies files and directories to be ignored by Git.

- Would be trying to directly fetch the translated XML files from the Oxygen XML editor

## **Prerequisites**

- Python 3.7 or higher.
- Tesseract OCR installed with the Latin language pack.
- Google Cloud account with Firestore enabled.
- Service account JSON key file for Google Cloud authentication.
- MLflow installed and configured.
- OpenCV installed for image processing.
- Oxygen XML Editor installed and accessible.
- Java installed (required for Oxygen XML Editor integration).

## **Steps to run:**

1. First run the requirement.txt

pip install -r requirements.txt

2. Install Teserract

brew install tesseract

3. Set up ML Flow

Start MLflow Tracking Server:

mlflow ui

Access the MLflow UI at http://localhost:5000

## **Naming Convention Using**

1. Images: Place the images you want to translate in data/images/
Naming Convention: Name the images as page_<number>.png (e.g., page_15.png)

2. XML Files: Place the XML files from the GitHub repository into data/xml/
Naming Convention: XML files are named like p0015.xml1

