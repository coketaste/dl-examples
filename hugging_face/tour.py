# Databricks notebook source
# MAGIC %md
# MAGIC # Hugging Face Transformers
# MAGIC
# MAGIC Transformers provides APIs and tools to easily download and train state-of-the-art pretrained models. Using pretrained models can reduce your compute costs, carbon footprint, and save you the time and resources required to train a model from scratch.
# MAGIC
# MAGIC ## Pipelines for inference
# MAGIC
# MAGIC The pipeline() makes it simple to use any model from the Hub for inference on any language, computer vision, speech, and multimodal tasks. 

# COMMAND ----------

!pip install --upgrade pip
!pip install 'transformers[torch,sentencepiece,vision,optuna,sklearn]'
!pip install datasets[audio]

# COMMAND ----------

text = """Dear Amazon, last week I ordered an Optimus Prime action figure \
from your online store in Germany. Unfortunately, when I opened the package, \
I discovered to my horror that I had been sent an action figure of Megatron \
instead! As a lifelong enemy of the Decepticons, I hope you can understand my \
dilemma. To resolve the issue, I demand an exchange of Megatron for the \
Optimus Prime figure I ordered. Enclosed are copies of my records concerning \
this purchase. I expect to hear from you soon. Sincerely, Bumblebee."""

# COMMAND ----------

# MAGIC %md
# MAGIC ### Text Classification

# COMMAND ----------

from transformers import pipeline

classifier = pipeline("text-classification")

# COMMAND ----------

import pandas as pd

outputs = classifier(text)
pd.DataFrame(outputs)    

# COMMAND ----------

# MAGIC %md
# MAGIC ### Named Entity Recognition

# COMMAND ----------

ner_tagger = pipeline("ner", aggregation_strategy="simple")
outputs = ner_tagger(text)
pd.DataFrame(outputs)   

# COMMAND ----------

# MAGIC %md
# MAGIC ### Question Answering 

# COMMAND ----------

reader = pipeline("question-answering")
question = "What does the customer want?"
outputs = reader(question=question, context=text)
pd.DataFrame([outputs])

# COMMAND ----------

# MAGIC %md
# MAGIC ### Summarization

# COMMAND ----------

summarizer = pipeline("summarization")
outputs = summarizer(text, max_length=45, clean_up_tokenization_spaces=True)
print(outputs[0]['summary_text'])

# COMMAND ----------

# MAGIC %md
# MAGIC ### Translation

# COMMAND ----------

translator = pipeline("translation_en_to_de", model="Helsinki-NLP/opus-mt-en-de")
outputs = translator(text, clean_up_tokenization_spaces=True, min_length=100)
print(outputs[0]['translation_text'])

# COMMAND ----------

# MAGIC %md
# MAGIC ### Speech Recognition
# MAGIC
# MAGIC To run the speech recognition, we need to install ffmpeg

# COMMAND ----------

# Wav2Vec2 model
# transcriber = pipeline(task="automatic-speech-recognition")
# transcriber("https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac")

# COMMAND ----------

# Whisper large-v2 model from OpenAI
# transcriber = pipeline(model="openai/whisper-large-v2")
# transcriber("https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac")

# COMMAND ----------

# transcriber(
#     [
#         "https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac",
#         "https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/1.flac",
#     ]
# )
