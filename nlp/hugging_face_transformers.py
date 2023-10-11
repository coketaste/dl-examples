# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Hugging Face Transformers
# MAGIC
# MAGIC This notebook will give an introduction to the Hugging Face Transformers Python library and some common patterns that you can use to take advantage of it. It is most useful for using or fine-tuning pretrained transformer models for your projects.
# MAGIC
# MAGIC Hugging Face provides access to models (both the code that implements them and their pre-trained weights), model-specific tokenizers, as well as pipelines for common NLP tasks, and datasets and metrics in a separate datasets package. It has implementations in PyTorch, Tensorflow, and Flax (though we'll be using the PyTorch versions here!)
# MAGIC
# MAGIC We're going to go through a few use cases:
# MAGIC
# MAGIC * Overview of Tokenizers and Models
# MAGIC * Finetuning - for your own task. We'll use a sentiment-classification example.

# COMMAND ----------

!pip install transformers
!pip install datasets
!pip install torch

# COMMAND ----------

from collections import defaultdict, Counter
import json

from matplotlib import pyplot as plt
import numpy as np
import torch

def print_encoding(model_inputs, indent=4):
    indent_str = " " * indent
    print("{")
    for k, v in model_inputs.items():
        print(indent_str + k + ":")
        print(indent_str + indent_str + str(v))
    print("}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 0: Common Pattern for using Hugging Face Transformers
# MAGIC
# MAGIC We're going to start off with a common usage pattern for Hugging Face Transformers, using the example of Sentiment Analysis.
# MAGIC
# MAGIC First, find a model on the hub. Anyone can upload their model for other people to use. (I'm using a sentiment analysis model from this paper).
# MAGIC
# MAGIC Then, there are two objects that need to be initialized - a tokenizer, and a model
# MAGIC
# MAGIC * Tokenizer converts strings to lists of vocabulary ids that the model requires
# MAGIC * Model takes the vocabulary ids and produces a prediction

# COMMAND ----------

from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained("siebert/sentiment-roberta-large-english")
# Initialize the model
model = AutoModelForSequenceClassification.from_pretrained("siebert/sentiment-roberta-large-english")

# COMMAND ----------

inputs = "I'm excited to learn about Hugging Face Transformers!"
tokenized_inputs = tokenizer(inputs, return_tensors="pt")
outputs = model(**tokenized_inputs)

labels = ['NEGATIVE', 'POSITIVE']
prediction = torch.argmax(outputs.logits)


print("Input:")
print(inputs)
print()
print("Tokenized Inputs:")
print_encoding(tokenized_inputs)
print()
print("Model Outputs:")
print(outputs)
print()
print(f"The prediction is {labels[prediction]}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 0.1 Tokenizers
# MAGIC
# MAGIC Pretrained models are implemented along with tokenizers that are used to preprocess their inputs. The tokenizers take raw strings or list of strings and output what are effectively dictionaries that contain the the model inputs.
# MAGIC
# MAGIC You can access tokenizers either with the Tokenizer class specific to the model you want to use (here DistilBERT), or with the AutoTokenizer class. Fast Tokenizers are written in Rust, while their slow versions are written in Python.
