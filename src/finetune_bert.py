"""
Title: Text Extraction with BERT
Author: [Apoorv Nandan](https://twitter.com/NandanApoorv)
Date created: 2020/05/23
Last modified: 2020/05/23
Description: Fine tune pretrained BERT from HuggingFace Transformers on SQuAD.
"""
"""
## Introduction

This demonstration uses SQuAD (Stanford Question-Answering Dataset).
In SQuAD, an input consists of a question, and a paragraph for context.
The goal is to find the span of text in the paragraph that answers the question.
We evaluate our performance on this data with the "Exact Match" metric,
which measures the percentage of predictions that exactly match any one of the
ground-truth answers.

We fine-tune a BERT model to perform this task as follows:

1. Feed the context and the question as inputs to BERT.
2. Take two vectors S and T with dimensions equal to that of
   hidden states in BERT.
3. Compute the probability of each token being the start and end of
   the answer span. The probability of a token being the start of
   the answer is given by a dot product between S and the representation
   of the token in the last layer of BERT, followed by a softmax over all tokens.
   The probability of a token being the end of the answer is computed
   similarly with the vector T.
4. Fine-tune BERT and learn S and T along the way.

**References:**

- [BERT](https://arxiv.org/pdf/1810.04805.pdf)
- [SQuAD](https://arxiv.org/abs/1606.05250)
"""
"""
## Setup
"""
import os
import re
import json
import pickle
import string
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tokenizers import BertWordPieceTokenizer
from transformers import BertTokenizer, TFBertModel, BertConfig

max_len = 384
configuration = BertConfig()  # default parameters and configuration for BERT

"""
## Set-up BERT tokenizer
"""
# Save the slow pretrained tokenizer
save_path = "bert/bert_base_uncased/"


with open("bert_data_extra_small.pkl", "rb") as f:
    x_train, y_train, x_eval, y_eval = pickle.load(f)

"""
Create the Question-Answering Model using BERT and Functional API
"""


def create_model():
    ## BERT encoder
    encoder = TFBertModel.from_pretrained("bert-base-uncased")

    ## QA Model
    input_ids = layers.Input(shape=(max_len,), dtype=tf.int32)
    token_type_ids = layers.Input(shape=(max_len,), dtype=tf.int32)
    attention_mask = layers.Input(shape=(max_len,), dtype=tf.int32)
    embedding = encoder(
        input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask
    )[0]

    start_logits = layers.Dense(1, name="start_logit", use_bias=False)(embedding)
    start_logits = layers.Flatten()(start_logits)

    end_logits = layers.Dense(1, name="end_logit", use_bias=False)(embedding)
    end_logits = layers.Flatten()(end_logits)

    start_probs = layers.Activation(keras.activations.softmax)(start_logits)
    end_probs = layers.Activation(keras.activations.softmax)(end_logits)

    model = keras.Model(
        inputs=[input_ids, token_type_ids, attention_mask],
        outputs=[start_probs, end_probs],
    )
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    optimizer = keras.optimizers.Adam(lr=5e-5)
    model.compile(optimizer=optimizer, loss=[loss, loss])
    return model


"""
This code should preferably be run on Google Colab TPU runtime.
With Colab TPUs, each epoch will take 5-6 minutes.
"""
use_tpu = False
if use_tpu:
    # Create distribution strategy
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)

    # Create model
    with strategy.scope():
        model = create_model()
else:
    model = create_model()

# model.summary()

"""
## Create evaluation Callback

This callback will compute the exact match score using the validation data
after every epoch.
"""


def normalize_text(text):
    text = text.lower()

    # Remove punctuations
    exclude = set(string.punctuation)
    text = "".join(ch for ch in text if ch not in exclude)

    # Remove articles
    regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
    text = re.sub(regex, " ", text)

    # Remove extra white space
    text = " ".join(text.split())
    return text


class ExactMatch(keras.callbacks.Callback):
    """
    Each `SquadExample` object contains the character level offsets for each token
    in its input paragraph. We use them to get back the span of text corresponding
    to the tokens between our predicted start and end tokens.
    All the ground-truth answers are also present in each `SquadExample` object.
    We calculate the percentage of data points where the span of text obtained
    from model predictions matches one of the ground-truth answers.
    """

    def __init__(self, x_eval, y_eval):
        self.x_eval = x_eval
        self.y_eval = y_eval

    def on_epoch_end(self, epoch, logs=None):
        pred_start, pred_end = self.model.predict(self.x_eval)
        count = 0
        eval_examples_no_skip = [_ for _ in eval_squad_examples if _.skip == False]
        for idx, (start, end) in enumerate(zip(pred_start, pred_end)):
            squad_eg = eval_examples_no_skip[idx]
            offsets = squad_eg.context_token_to_char
            start = np.argmax(start)
            end = np.argmax(end)
            if start >= len(offsets):
                continue
            pred_char_start = offsets[start][0]
            if end < len(offsets):
                pred_char_end = offsets[end][1]
                pred_ans = squad_eg.context[pred_char_start:pred_char_end]
            else:
                pred_ans = squad_eg.context[pred_char_start:]

            normalized_pred_ans = normalize_text(pred_ans)
            normalized_true_ans = [normalize_text(_) for _ in squad_eg.all_answers]
            if normalized_pred_ans in normalized_true_ans:
                count += 1
        acc = count / len(self.y_eval[0])
        print(f"\nepoch={epoch+1}, exact match score={acc:.2f}")


"""
## Train and Evaluate
"""
exact_match_callback = ExactMatch(x_eval, y_eval)
model.fit(
    x_train,
    y_train,
    epochs=1,  # For demonstration, 3 epochs are recommended
    verbose=1,
    batch_size=32,
   # callbacks=[exact_match_callback],
)
