# translation_project

## Latin to English Translation Project with Oxygen XML Editor Integration

This project translates Latin text from images into English. It includes local OCR processing with image enhancement, OCR correction, custom model training, and integration with Oxygen XML Editor for XML data extraction. All data storage, logging, and model management are handled locally—no external cloud services are required.

## Project Structure

- **`main.py`**: The main script orchestrating the entire process.
- **`data/`**
  - **`images/`**: Directory containing images with Latin text to be translated.
  - **`xml/`**: Directory containing XML files exported from Oxygen XML Editor.
- **`models/`**
  - **`fine_tuned_model/`**: Directory where the fine-tuned translation model is saved.
  - **`ocr_correction_model/`**: Directory where the OCR correction model is saved.
- **`scripts/`**
  - **`data_preparation.py`**: Script for fetching XML data from Oxygen XML Editor and preparing the dataset.
  - **`model_training.py`**: Scripts for fine-tuning translation and OCR correction models.
  - **`ocr_and_translation.py`**: Script for OCR processing and translation.
  - **`image_enhancement.py`**: Script for enhancing images before OCR.
- **`requirements.txt`**: List of required Python packages.
- **`.gitignore`**: Specifies files and directories to be ignored by Git.
- **Local JSON Database**: A JSON file (e.g., `local_database.json`) is used to store translations.

*Note: The project will fetch translated XML files directly from Oxygen XML Editor.*

## Prerequisites

- Python 3.7 or higher.
- Tesseract OCR installed with the Latin language pack.
  - **macOS**: `brew install tesseract`
  - **Other OS**: Please refer to [Tesseract OCR documentation](https://github.com/tesseract-ocr/tesseract).
- OpenCV installed for image processing.
- Oxygen XML Editor installed and accessible.
- Java installed (required for Oxygen XML Editor integration).

## Steps to Run

1. **Install Python Dependencies**  
   Run the following command to install all required packages:
   ```bash
   pip install -r requirements.txt
