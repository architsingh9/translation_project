import logging
from transformers import MarianMTModel, MarianTokenizer, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
from transformers import T5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments
import evaluate

logger = logging.getLogger(__name__)

def fine_tune_translation_model(dataset, model_directory):
    """
    Fine-tunes a pre-trained translation model on the custom dataset.
    """
    model_name = 'Helsinki-NLP/opus-mt-zh-en'
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)

    def preprocess_function(examples):
        inputs = [ex['la'] for ex in examples['translation']]
        targets = [ex['en'] for ex in examples['translation']]
        model_inputs = tokenizer(inputs, text_target=targets, max_length=128, truncation=True)
        return model_inputs

    tokenized_datasets = dataset.map(preprocess_function, batched=True)
    training_args = Seq2SeqTrainingArguments(
        output_dir=model_directory,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=3,
        predict_with_generate=True,
        logging_dir='./logs',
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model='eval_bleu',
        greater_is_better=True,
    )
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    metric = evaluate.load("sacrebleu")

    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [[label.strip()] for label in labels]
        return preds, labels

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
        result = metric.compute(predictions=decoded_preds, references=decoded_labels)
        return {"bleu": result["score"]}

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['validation'],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    logger.info(f"Translation model: {model_name}")
    logger.info(f"Training epochs: {training_args.num_train_epochs}")
    logger.info(f"Learning rate: {training_args.learning_rate}")
    logger.info(f"Train batch size: {training_args.per_device_train_batch_size}")
    logger.info(f"Eval batch size: {training_args.per_device_eval_batch_size}")

    trainer.train()
    test_metrics = trainer.evaluate(tokenized_datasets['test'])
    logger.info(f"Test metrics: {test_metrics}")

    trainer.save_model(model_directory)
    tokenizer.save_pretrained(model_directory)
    logger.info(f"Translation model fine-tuning complete and saved to '{model_directory}'")

def train_ocr_correction_model(dataset, model_directory):
    """
    Trains an OCR correction model using T5.
    """
    model_name = 't5-small'
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)

    def preprocess_function(examples):
        inputs = [ex for ex in examples['input_text']]
        targets = [ex for ex in examples['target_text']]
        model_inputs = tokenizer(inputs, padding=True, truncation=True, max_length=128)
        labels = tokenizer(targets, padding=True, truncation=True, max_length=128).input_ids
        model_inputs['labels'] = labels
        return model_inputs

    tokenized_datasets = dataset.map(preprocess_function, batched=True)
    training_args = TrainingArguments(
        output_dir=model_directory,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        learning_rate=2e-4,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=5,
        logging_dir='./logs',
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model='eval_loss',
        greater_is_better=False,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['validation'],
        tokenizer=tokenizer
    )

    logger.info(f"OCR Correction model: {model_name}")
    logger.info(f"Training epochs: {training_args.num_train_epochs}")
    logger.info(f"Learning rate: {training_args.learning_rate}")
    logger.info(f"Train batch size: {training_args.per_device_train_batch_size}")
    logger.info(f"Eval batch size: {training_args.per_device_eval_batch_size}")

    trainer.train()
    test_metrics = trainer.evaluate(tokenized_datasets['test'])
    logger.info(f"Test metrics: {test_metrics}")

    trainer.save_model(model_directory)
    tokenizer.save_pretrained(model_directory)
    logger.info(f"OCR correction model training complete and saved to '{model_directory}'")
