import json
import re
from string import punctuation
from transformers import pipeline
import pandas as pd
import torch
from transformers import BertConfig, BertModel, BertTokenizer


def lower(text):
    return text.lower()


def remove_punctuation(text):
    return text.translate(str.maketrans("", "", punctuation))


def remove_digits(text):
    return re.sub(r"\d+", "", text)


def remove_emoji(text):
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE,
    )
    return emoji_pattern.sub(r"", text)


def remove_non_printable(text):
    text = text.encode("ascii", "ignore")
    return text.decode()


def clean_text(text):
    text = lower(text)
    text = remove_punctuation(text)
    text = remove_digits(text)
    text = remove_emoji(text)
    text = remove_non_printable(text)
    return text


pipe = pipeline("text-classification", model="LiYuan/amazon-review-sentiment-analysis")
sentiment_model = pipeline(
    "text-classification", model="distilbert-base-uncased-finetuned-sst-2-english"
)
model_name = "bert-base-uncased"
config = BertConfig.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name, config=config)
tokenizer = BertTokenizer.from_pretrained(model_name)


def analyze_sentiment(text):
    sentiment = pipe(text)[0]
    emotion = sentiment["label"]
    score = sentiment["score"]
    return emotion, score


def analyze_sentiment_distilbert(text):
    sentiment = sentiment_model(text)[0]
    emotion = sentiment["label"]
    score = sentiment["score"]
    return emotion, score


def analyze_sentiment_bert(text):
    encoding = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
    input_ids = encoding["input_ids"]
    attention_mask = encoding["attention_mask"]
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    last_hidden_state = outputs.last_hidden_state
    pooled_output = torch.mean(last_hidden_state, dim=1)
    logits = torch.nn.Linear(pooled_output.shape[-1], config.num_labels)(pooled_output)
    probabilities = torch.softmax(logits, dim=1).tolist()[0]
    sentiment = "Positive" if probabilities[1] > probabilities[0] else "Negative"
    return sentiment
