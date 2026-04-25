import time
import torch
import pandas as pd
import sentencepiece
from datetime import datetime, timedelta
from preprocessing import TextPreprocessor
from datasets import load_dataset, Dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification, TrainingArguments, Trainer


class ModelTrainer:
    def __init__(self, preprocessor):
        self.preprocessor = preprocessor
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def prepare_dataframe(self):
        df = self.preprocessor.run()
        df = df[["Text", "labels"]]
        return df

    def prepare_dataset(self, df):
        dataset = Dataset.from_pandas(df)
        return dataset.train_test_split(test_size=0.2, seed=42)

    def tokenize(self, dataset):
        self.tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")

        def tokenize_function(example):
            return self.tokenizer(
                example["Text"],
                padding="max_length",
                truncation=True,
                max_length=128
            )

        return dataset.map(tokenize_function, batched=True)

    def build_model(self):
        self.model = XLMRobertaForSequenceClassification.from_pretrained(
            "xlm-roberta-base",
            num_labels=2
        )
        self.model.to(self.device)

    def compute_metrics(self, pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        return {
            "accuracy": accuracy_score(labels, preds),
            "f1": f1_score(labels, preds),
            "precision": precision_score(labels, preds),
            "recall": recall_score(labels, preds),
        }

    def train(self, tokenized_dataset):
        training_args = TrainingArguments(
            output_dir="./xlmr_disinfo_model",
            eval_strategy="epoch",
            save_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=3,
            weight_decay=0.01,
            logging_dir="./logs",
            logging_steps=10,
        )

        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["test"],
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics,
        )

        start_time = time.time()
        self.trainer.train()
        elapsed = time.time() - start_time

        self.training_duration = str(timedelta(seconds=int(elapsed)))

    def save(self):
        self.trainer.save_model("xlm_trained_model")
        self.tokenizer.save_pretrained("xlm_trained_model")

        duration = getattr(self, "training_duration", "невідомо")
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with open("xlm_trained_model/training_info.txt", "w", encoding="utf-8") as f:
            f.write(f"Дата навчання: {timestamp}\n")
            f.write(f"Час навчання: {duration}\n")

    def run(self):
        df = self.prepare_dataframe()
        dataset = self.prepare_dataset(df)
        tokenized_dataset = self.tokenize(dataset)
        self.build_model()
        self.train(tokenized_dataset)
        self.save()

if __name__ == "__main__":
    preprocessor = TextPreprocessor("model_training/raw_news_train/raw_news.csv")
    trainer = ModelTrainer(preprocessor)
    trainer.run()