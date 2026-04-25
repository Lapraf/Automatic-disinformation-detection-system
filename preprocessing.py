import re
import unicodedata
import pandas as pd

class TextPreprocessor:
    def __init__(self, input_path: str):
        self.input_path = input_path
        self.df = None

    def load_data(self):
        self.df = pd.read_csv(self.input_path, encoding="utf-8-sig", header=0, delimiter = "|")

    def clean_text(self, text):
        text = str(text)
        text = text.replace('""', '"')
        text = re.sub(r"<.*?>", "", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def normalize_punctuation(self, text):
        text = text.replace("…", "")
        text = text.replace("...", "")
        text = text.replace("«", "")
        text = text.replace("»", "")
        return text.strip()

    def remove_quotes(self, text):
        return "".join(
            ch for ch in text
            if unicodedata.category(ch) not in ["Pi", "Pf"]
        )

    def process(self):
        self.df["Text"] = self.df["Text"].apply(self.clean_text)
        self.df["Text"] = self.df["Text"].apply(self.normalize_punctuation)
        self.df["Text"] = self.df["Text"].apply(self.remove_quotes)

        self.df = self.df.drop_duplicates(subset="Text")
        self.df = self.df.dropna(subset=["Text"])
        self.df = self.df[self.df["Text"].str.len() > 10]

    def run(self):
        self.load_data()
        self.process()
        return self.df