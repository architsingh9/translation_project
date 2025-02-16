import logging
from datasets import Dataset, DatasetDict
from lxml import etree
import os

logger = logging.getLogger(__name__)

def fetch_page_data(xml_directory, page_number=None):
    """
    Fetches Latin and English texts from XML files in the given directory.
    If page_number is specified, returns data for that page only.
    """
    try:
        latin_texts = []
        english_texts = []

        xml_files = [f for f in os.listdir(xml_directory) if f.endswith('.xml')]
        if not xml_files:
            logger.error("No XML files found in the XML directory.")
            raise FileNotFoundError("No XML files found.")

        for xml_file in xml_files:
            file_path = os.path.join(xml_directory, xml_file)
            try:
                # Extract page number from filename (e.g., p0015.xml -> 15)
                doc_name = os.path.basename(file_path)
                doc_page_number = int(doc_name[1:5])
            except (IndexError, ValueError):
                logger.warning(f"Invalid XML filename format: {doc_name}. Expected format: 'p<page_number>.xml'")
                continue

            if page_number is not None and doc_page_number != page_number:
                continue  # Process only the target page if specified

            # Parse the XML file
            tree = etree.parse(file_path)
            root = tree.getroot()
            ns = {'tei': 'http://www.tei-c.org/ns/1.0'}

            # Locate transcription and translation elements
            transcription = root.xpath('.//tei:ab[@type="transcription"]', namespaces=ns)
            translation = root.xpath('.//tei:ab[@type="translation"]', namespaces=ns)

            if not transcription or not translation:
                logger.warning(f"Transcription or translation not found in {doc_name}.")
                continue

            # Extract Latin text from transcription element
            latin_lines = []
            for choice in transcription[0].xpath('.//tei:choice', namespaces=ns):
                reg = choice.xpath('.//tei:reg', namespaces=ns)
                if reg and reg[0].text:
                    latin_lines.append(reg[0].text.strip())
                else:
                    orig = choice.xpath('.//tei:orig', namespaces=ns)
                    if orig and orig[0].text:
                        latin_lines.append(orig[0].text.strip())
            # Also capture any direct text within the transcription element
            direct_texts = transcription[0].xpath('./text()', namespaces=ns)
            for text in direct_texts:
                if text.strip():
                    latin_lines.append(text.strip())

            latin_text = ' '.join(latin_lines)
            english_text = ' '.join(translation[0].itertext()).strip()

            if latin_text and english_text:
                latin_texts.append(latin_text)
                english_texts.append(english_text)

            if page_number is not None:
                return [latin_text], [english_text]

        if not latin_texts:
            logger.error("No Latin-English pairs extracted.")
            raise ValueError("Dataset is empty.")

        logger.info(f"Extracted {len(latin_texts)} sentence pairs from XML files.")
        return latin_texts, english_texts

    except Exception as e:
        logger.error(f"Failed to fetch XML data: {e}")
        raise

def create_translation_dataset(latin_texts, english_texts):
    """
    Creates and splits a dataset for translation model training.
    """
    dataset_dict = {'translation': [{'la': la, 'en': en} for la, en in zip(latin_texts, english_texts)]}
    full_dataset = Dataset.from_dict(dataset_dict)
    split_dataset = full_dataset.train_test_split(test_size=0.2, seed=42)
    test_dataset = split_dataset['test']
    temp_dataset = split_dataset['train'].train_test_split(test_size=0.25, seed=42)
    train_dataset = temp_dataset['train']
    validation_dataset = temp_dataset['test']

    dataset = DatasetDict({
        'train': train_dataset,
        'validation': validation_dataset,
        'test': test_dataset
    })

    logger.info(f"Dataset split into {len(train_dataset)} training, {len(validation_dataset)} validation, and {len(test_dataset)} test examples.")
    return dataset

def create_correction_dataset(latin_texts):
    """
    Creates and splits a dataset for OCR correction model training by introducing simulated noise.
    """
    import random

    def introduce_noise(text):
        noise_level = 0.1  # Adjust as needed
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
    split_dataset = full_dataset.train_test_split(test_size=0.2, seed=42)
    test_dataset = split_dataset['test']
    temp_dataset = split_dataset['train'].train_test_split(test_size=0.25, seed=42)
    train_dataset = temp_dataset['train']
    validation_dataset = temp_dataset['test']

    dataset = DatasetDict({
        'train': train_dataset,
        'validation': validation_dataset,
        'test': test_dataset
    })

    logger.info(f"Correction dataset split into {len(train_dataset)} training, {len(validation_dataset)} validation, and {len(test_dataset)} test examples.")
    return dataset
