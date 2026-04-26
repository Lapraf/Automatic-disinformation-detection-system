import re
import torch
import unicodedata
import pandas as pd
from preprocessing import TextPreprocessor
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification


class NewsPredictor:
    def __init__(self, input_path: str):
        self.model_path = "xlm_trained_model"
        self.preprocessor = TextPreprocessor(input_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None

    def load_model(self):
        self.tokenizer = XLMRobertaTokenizer.from_pretrained(self.model_path)
        self.model = XLMRobertaForSequenceClassification.from_pretrained(
            self.model_path, num_labels=2
        )
        self.model.to(self.device)
        self.model.eval()

    def predict_one(self, text: str) -> dict:
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

        predicted_class = torch.argmax(probs, dim=-1).item()
        chance_fake = round(probs[0][0].item() * 100, 2)

        return {
            "Text": text,
            "Class": predicted_class,
            "Fake Chance": chance_fake
        }

    def run(self) -> pd.DataFrame:
        df = self.preprocessor.run()
        self.load_model()

        predictions = df["Text"].apply(self.predict_one)
        pred_df = pd.DataFrame(predictions.tolist())

        result_df = pd.concat([df, pred_df], axis=1)

        return result_df