import logging
from datasets import Dataset, DatasetDict
import mlflow
from lxml import etree
import jpype
import jpype.imports
from jpype.types import *
import os

logger = logging.getLogger(__name__)

def start_jvm():
    """
    Starts the Java Virtual Machine for interacting with Oxygen XML Editor.
    """
    if not jpype.isJVMStarted():
        # Adjust the classpath to your Oxygen installation
        oxygen_lib_path = '/path/to/oxygen/lib/*'  # Replace with the actual path
        jpype.startJVM(classpath=[oxygen_lib_path])
        logger.info("JVM started.")

def fetch_page_data(page_number=None):
    """
    Fetches Latin and English texts from Oxygen XML Editor.
    If page_number is specified, fetches data for that page only.
    """
    try:
        start_jvm()

        # Import Java classes
        from ro.sync.exml.workspace.api.standalone import StandalonePluginWorkspace

        # Get the current plugin workspace
        workspace = StandalonePluginWorkspace.getInstance()
        if workspace is None:
            logger.error("Oxygen XML Editor is not running or accessible.")
            raise Exception("Oxygen XML Editor is not running.")

        # Get all open XML documents
        editorAccesses = workspace.getAllEditorAccesses(StandalonePluginWorkspace.MAIN_EDITING_AREA)
        latin_texts = []
        english_texts = []

        for editorAccess in editorAccesses:
            if editorAccess.getCurrentPageID() == "Author":
                # Get the page number from the document name or metadata
                doc_url = editorAccess.getEditorLocation().toString()
                # Assuming the document name contains the page number, e.g., page_10.xml
                try:
                    doc_name = os.path.basename(doc_url)
                    doc_page_number = int(os.path.splitext(doc_name)[0].split('_')[1])
                except (IndexError, ValueError):
                    logger.warning(f"Invalid document name format: {doc_name}. Expected format: 'page_<number>.xml'")
                    continue

                if page_number is not None and doc_page_number != page_number:
                    continue  # Skip if not the target page

                # Get the XML content
                xml_content = editorAccess.createContentReader().read()
                # Parse the XML content
                root = etree.fromstring(xml_content)
                # Modify XPath expressions as needed
                latin_elements = root.xpath('//LatinTextElement')
                english_elements = root.xpath('//EnglishTextElement')

                if len(latin_elements) != len(english_elements):
                    logger.warning(f"Mismatch in Latin and English elements in {doc_name}.")
                    continue  # Skip files with mismatched entries

                for lat_elem, eng_elem in zip(latin_elements, english_elements):
                    latin_text = lat_elem.text.strip()
                    english_text = eng_elem.text.strip()

                    if latin_text and english_text:
                        latin_texts.append(latin_text)
                        english_texts.append(english_text)

                if page_number is not None:
                    # Return the texts for the specific page
                    return latin_texts, english_texts

        if not latin_texts:
            logger.error("No Latin-English pairs extracted.")
            raise ValueError("Dataset is empty.")

        logger.info(f"Extracted {len(latin_texts)} sentence pairs from Oxygen XML Editor.")

        # Log metrics to MLflow
        mlflow.log_metric("extracted_sentence_pairs", len(latin_texts))

        return latin_texts, english_texts

    except Exception as e:
        logger.error(f"Failed to fetch XML data from Oxygen XML Editor: {e}")
        raise

def create_translation_dataset(latin_texts, english_texts):
    """
    Creates a dataset for translation model training and splits it into training, validation, and test sets.
    """
    dataset_dict = {'translation': [{'la': la, 'en': en} for la, en in zip(latin_texts, english_texts)]}
    full_dataset = Dataset.from_list(dataset_dict['translation'])

    # Split the dataset
    split_dataset = full_dataset.train_test_split(test_size=0.2, seed=42)
    test_dataset = split_dataset['test']
    temp_dataset = split_dataset['train'].train_test_split(test_size=0.25, seed=42)
    train_dataset = temp_dataset['train']
    validation_dataset = temp_dataset['test']

    # Combine into a DatasetDict
    dataset = DatasetDict({
        'train': train_dataset,
        'validation': validation_dataset,
        'test': test_dataset
    })

    logger.info(f"Dataset split into {len(train_dataset)} training, {len(validation_dataset)} validation, and {len(test_dataset)} test examples.")

    return dataset

def create_correction_dataset(latin_texts):
    """
    Creates a dataset for OCR correction model training and splits it into training, validation, and test sets.
    """
    # Simulate OCR errors by introducing noise
    import random

    def introduce_noise(text):
        noise_level = 0.1  # Adjust noise level as needed
        noisy_text = ''
        for char in text:
            if random.random() < noise_level:
                noisy_text += random.choice('abcdefghijklmnopqrstuvwxyz ')
            else:
                noisy_text += char
        return noisy_text

    ocr_texts = [introduce_noise(text) for text in latin_texts]
    dataset_dict = {'input_text': ocr_texts, 'target_text': latin_texts}
    full_dataset = Dataset.from_dict(dataset_dict)

    # Split the dataset
    split_dataset = full_dataset.train_test_split(test_size=0.2, seed=42)
    test_dataset = split_dataset['test']
    temp_dataset = split_dataset['train'].train_test_split(test_size=0.25, seed=42)
    train_dataset = temp_dataset['train']
    validation_dataset = temp_dataset['test']

    # Combine into a DatasetDict
    dataset = DatasetDict({
        'train': train_dataset,
        'validation': validation_dataset,
        'test': test_dataset
    })

    logger.info(f"Correction dataset split into {len(train_dataset)} training, {len(validation_dataset)} validation, and {len(test_dataset)} test examples.")

    return dataset
