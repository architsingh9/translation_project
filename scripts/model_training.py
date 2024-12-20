import logging
from transformers import MarianMTModel, MarianTokenizer, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
from transformers import T5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments
import mlflow
from datasets import load_metric

logger = logging.getLogger(__name__)

def fine_tune_translation_model(dataset, model_directory):
    """
    Fine-tunes a pre-trained translation model on the custom dataset.
    """
    model_name = 'Helsinki-NLP/opus-mt-zh-en'
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)

    # Tokenization function
    def preprocess_function(examples):
        inputs = [ex['la'] for ex in examples['translation']]
        targets = [ex['en'] for ex in examples['translation']]
        model_inputs = tokenizer(inputs, text_target=targets, max_length=128, truncation=True)
        return model_inputs

    # Preprocess the datasets
    tokenized_datasets = dataset.map(preprocess_function, batched=True)

    # Training arguments
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
        save_steps=500,
        eval_steps=500,
        report_to=["mlflow"],  # Report to MLflow
        load_best_model_at_end=True,
        metric_for_best_model='eval_bleu',
        greater_is_better=True,
    )

    # Data collator
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    # Define compute metrics function
    metric = load_metric("sacrebleu")

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
        # Post-processing
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
        result = metric.compute(predictions=decoded_preds, references=decoded_labels)
        result = {"bleu": result["score"]}
        return result

    # Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['validation'],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    # Log parameters to MLflow
    mlflow.log_param("translation_model_name", model_name)
    mlflow.log_param("translation_num_train_epochs", training_args.num_train_epochs)
    mlflow.log_param("translation_learning_rate", training_args.learning_rate)
    mlflow.log_param("translation_train_batch_size", training_args.per_device_train_batch_size)
    mlflow.log_param("translation_eval_batch_size", training_args.per_device_eval_batch_size)

    # Training
    trainer.train()

    # Evaluate the model on the test set
    test_metrics = trainer.evaluate(tokenized_datasets['test'])
    mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})

    # Save the model
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

    # Preprocess function
    def preprocess_function(examples):
        inputs = [ex for ex in examples['input_text']]
        targets = [ex for ex in examples['target_text']]
        model_inputs = tokenizer(inputs, padding=True, truncation=True, max_length=128)
        labels = tokenizer(targets, padding=True, truncation=True, max_length=128).input_ids
        model_inputs['labels'] = labels
        return model_inputs

    # Preprocess the datasets
    tokenized_datasets = dataset.map(preprocess_function, batched=True)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=model_directory,
        evaluation_strategy='epoch',
        learning_rate=2e-4,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=5,
        logging_dir='./logs',
        logging_steps=10,
        save_steps=500,
        eval_steps=500,
        report_to=["mlflow"],  # Report to MLflow
        load_best_model_at_end=True,
        metric_for_best_model='eval_loss',
        greater_is_better=False,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['validation'],
        tokenizer=tokenizer
    )

    # Log parameters to MLflow
    mlflow.log_param("correction_model_name", model_name)
    mlflow.log_param("correction_num_train_epochs", training_args.num_train_epochs)
    mlflow.log_param("correction_learning_rate", training_args.learning_rate)
    mlflow.log_param("correction_train_batch_size", training_args.per_device_train_batch_size)
    mlflow.log_param("correction_eval_batch_size", training_args.per_device_eval_batch_size)

    # Training
    trainer.train()

    # Evaluate the model on the test set
    test_metrics = trainer.evaluate(tokenized_datasets['test'])
    mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})

    # Save the model
    trainer.save_model(model_directory)
    tokenizer.save_pretrained(model_directory)

    logger.info(f"OCR correction model training complete and saved to '{model_directory}'")
