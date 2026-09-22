# Sentiment Analysis using Hugging Face Transformers

A simple sentiment analysis application built using **Hugging Face Transformers, PyTorch, and Gradio**.

This project demonstrates two different ways of performing sentiment analysis using the same pretrained DistilBERT model:

1. Using the Hugging Face `pipeline()` API.
2. Loading the tokenizer and model directly using `AutoTokenizer` and `AutoModelForSequenceClassification`.

The outputs from both approaches are displayed side-by-side in a Gradio web interface.

## Project Overview

The application accepts a sentence from the user and predicts whether its sentiment is:

* **POSITIVE**
* **NEGATIVE**

The same input sentence is passed through both implementations, allowing the results of the two approaches to be compared.

### Model Used

The project uses:

`distilbert/distilbert-base-uncased-finetuned-sst-2-english`

This is a DistilBERT model fine-tuned on the **SST-2 (Stanford Sentiment Treebank)** dataset for binary sentiment classification.

## Features

* Sentiment analysis using Hugging Face Transformers.
* Demonstrates the high-level `pipeline()` approach.
* Demonstrates direct model and tokenizer usage.
* Uses PyTorch for softmax function.
* Interactive Gradio user interface.
* Displays the predictions from both approaches simultaneously.

## Project Structure

```text
.
├── app.py
├── requirements.txt
└── README.md
```


## How It Works

### 1. Pipeline Method

The `pipeline()` function provides a convenient high-level API for sentiment analysis.

```python
pipeline_demo = pipeline("sentiment-analysis")
return pipeline_demo(sentence1)[0]['label']
```

The pipeline automatically handles the tokenizer, model loading, input processing, and prediction.

### 2. Direct Model Method

The second approach loads the tokenizer and model explicitly:

```python
tokenizer = AutoTokenizer.from_pretrained(
    "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
)

model_gen = AutoModelForSequenceClassification.from_pretrained(
    "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
)
```

The input sentence is then tokenized and passed to the model.

The model produces logits, which are converted into probabilities using PyTorch:

```python
probabilities = torch.softmax(outputs.logits, dim=-1)
```

The class with the highest probability is selected as the prediction.
Based on that value a sentiment is produced.

## Application Flow

```text
User enters a sentence
        │
        ▼
   Gradio Interface
        │
        ▼
    master_fn()
       /     \
      /       \
     ▼         ▼
Pipeline     Direct Model
Method        Method
     │         │
     ▼         ▼
Sentiment    Sentiment
Prediction   Prediction
     \         /
      \       /
       ▼     ▼
   Display both
    predictions
```

## Example

Input:

```text
I really enjoyed this workshop.
```

Possible output:

```text
Pipeline: POSITIVE
Model: POSITIVE
```

Another example:

```text
The experience was disappointing.
```

Possible output:

```text
Pipeline: NEGATIVE
Model: NEGATIVE
```

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/parthu17/huggingFaceWorkshopAssignment.git
cd Project-Files
```

### 2. Install the required libraries

```bash
pip install -r requirements.txt
```

### 3. Run the application

```bash
python app.py
```

The Gradio application will provide a local web interface that can be opened in a browser.

## Requirements

The project uses the libraries specified in `requirements.txt`, including:

* **Gradio** – for the interactive web interface.
* **PyTorch** – for model inference and probability calculation.
* **Transformers and its sublibraries** – for loading the pretrained tokenizer, model, and sentiment-analysis pipeline.

The pretrained model is downloaded automatically from Hugging Face when it is used for the first time.

## Technologies Used

| Technology                | Purpose                                     |
| ------------------------- | ------------------------------------------- |
| Python                    | Programming language                        |
| Hugging Face Transformers | NLP model and tokenizer                     |
| DistilBERT                | Sentiment classification model              |
| PyTorch                   | Model inference and probability calculation |
| Gradio                    | Interactive web interface                   |

## Learning Objectives

This project demonstrates:

* Using pretrained NLP models from Hugging Face.
* Using the Transformers `pipeline()` API.
* Loading a Transformer model manually.
* Tokenizing text using `AutoTokenizer`.
* Performing inference using `AutoModelForSequenceClassification`.
* Converting model logits into probabilities with PyTorch.
* Building a simple machine-learning interface using Gradio.
* Comparing a high-level API with the underlying model implementation.

## Notes

The application uses a binary sentiment classification model trained for English text. Therefore, predictions are intended for English sentences and may not be reliable for other languages or unusual inputs.

The first execution may take longer because the tokenizer and pretrained model need to be downloaded and cached locally.

## Workshop

This project was completed as part of the **Getting Started with Hugging Face: Models, Datasets and AI Applications Workshop** which was one of the **IITM BS Degree End Term Workshops**. The workshop was on getting started with Hugging Face,usage and introduction of pretrained Transformer models for NLP,basics of Hugging Face Datasets and launching basic web interfaces using Gradio.
