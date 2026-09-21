import gradio as gr
import torch
from transformers import pipeline
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
def pipelinefn(sentence1):
  pipeline_demo= pipeline("sentiment-analysis")
  return pipeline_demo(sentence1)[0]['label']
def modelfn(sentence2):
  tokenizer=AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased-finetuned-sst-2-english")
  model_gen=AutoModelForSequenceClassification.from_pretrained("distilbert/distilbert-base-uncased-finetuned-sst-2-english")
  inputs=tokenizer(sentence2,return_tensors="pt")
  outputs=model_gen(**inputs)
  probabilities=torch.softmax(outputs.logits,dim=-1)
  predicted_outcome=torch.argmax(probabilities,dim=-1).tolist()[0]
  if predicted_outcome == 1:
    return "POSITIVE"
  else :
    return "NEGATIVE"
def master_fn(sentence3):
  pipeline_outcome=pipelinefn(sentence3)
  model_outcome=modelfn(sentence3)
  return pipeline_outcome,model_outcome
application=gr.Interface(fn=master_fn,inputs=gr.Textbox(label="Enter sentence for sentiment analysis"),outputs=[gr.Textbox(label="Pipeline"),gr.Textbox(label="Model")],title="Sentiment Analysis using same model using Pipeline and Model methods")
application.launch()