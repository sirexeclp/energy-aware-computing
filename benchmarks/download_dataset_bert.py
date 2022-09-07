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
slow_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
save_path = "bert/bert_base_uncased/"
if not os.path.exists(save_path):
    os.makedirs(save_path)
slow_tokenizer.save_pretrained(save_path)

# Load the fast tokenizer from saved file
tokenizer = BertWordPieceTokenizer(save_path + "vocab.txt", lowercase=True)

"""
## Load the data
"""
train_data_url = "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json"
train_path = keras.utils.get_file("train.json", train_data_url)
eval_data_url = "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json"
eval_path = keras.utils.get_file("eval.json", eval_data_url)

"""
## Preprocess the data

1. Go through the JSON file and store every record as a `SquadExample` object.
2. Go through each `SquadExample` and create `x_train, y_train, x_eval, y_eval`.
"""


class SquadExample:
    def __init__(self, question, context, start_char_idx, answer_text, all_answers):
        self.question = question
        self.context = context
        self.start_char_idx = start_char_idx
        self.answer_text = answer_text
        self.all_answers = all_answers
        self.skip = False

    def preprocess(self):
        context = self.context
        question = self.question
        answer_text = self.answer_text
        start_char_idx = self.start_char_idx

        # Clean context, answer and question
        context = " ".join(str(context).split())
        question = " ".join(str(question).split())
        answer = " ".join(str(answer_text).split())

        # Find end character index of answer in context
        end_char_idx = start_char_idx + len(answer)
        if end_char_idx >= len(context):
            self.skip = True
            return

        # Mark the character indexes in context that are in answer
        is_char_in_ans = [0] * len(context)
        for idx in range(start_char_idx, end_char_idx):
            is_char_in_ans[idx] = 1

        # Tokenize context
        tokenized_context = tokenizer.encode(context)

        # Find tokens that were created from answer characters
        ans_token_idx = []
        for idx, (start, end) in enumerate(tokenized_context.offsets):
            if sum(is_char_in_ans[start:end]) > 0:
                ans_token_idx.append(idx)

        if len(ans_token_idx) == 0:
            self.skip = True
            return

        # Find start and end token index for tokens from answer
        start_token_idx = ans_token_idx[0]
        end_token_idx = ans_token_idx[-1]

        # Tokenize question
        tokenized_question = tokenizer.encode(question)

        # Create inputs
        input_ids = tokenized_context.ids + tokenized_question.ids[1:]
        token_type_ids = [0] * len(tokenized_context.ids) + [1] * len(
            tokenized_question.ids[1:]
        )
        attention_mask = [1] * len(input_ids)

        # Pad and create attention masks.
        # Skip if truncation is needed
        padding_length = max_len - len(input_ids)
        if padding_length > 0:  # pad
            input_ids = input_ids + ([0] * padding_length)
            attention_mask = attention_mask + ([0] * padding_length)
            token_type_ids = token_type_ids + ([0] * padding_length)
        elif padding_length < 0:  # skip
            self.skip = True
            return

        self.input_ids = input_ids
        self.token_type_ids = token_type_ids
        self.attention_mask = attention_mask
        self.start_token_idx = start_token_idx
        self.end_token_idx = end_token_idx
        self.context_token_to_char = tokenized_context.offsets


with open(train_path) as f:
    raw_train_data = json.load(f)

with open(eval_path) as f:
    raw_eval_data = json.load(f)


def create_squad_examples(raw_data):
    squad_examples = []
    for item in raw_data["data"]:
        for para in item["paragraphs"]:
            context = para["context"]
            for qa in para["qas"]:
                question = qa["question"]
                answer_text = qa["answers"][0]["text"]
                all_answers = [_["text"] for _ in qa["answers"]]
                start_char_idx = qa["answers"][0]["answer_start"]
                squad_eg = SquadExample(
                    question, context, start_char_idx, answer_text, all_answers
                )
                squad_eg.preprocess()
                squad_examples.append(squad_eg)
    return squad_examples


def create_inputs_targets(squad_examples):
    dataset_dict = {
        "input_ids": [],
        "token_type_ids": [],
        "attention_mask": [],
        "start_token_idx": [],
        "end_token_idx": [],
    }
    for item in squad_examples:
        if item.skip == False:
            for key in dataset_dict:
                dataset_dict[key].append(getattr(item, key))
    for key in dataset_dict:
        dataset_dict[key] = np.array(dataset_dict[key])

    x = [
        dataset_dict["input_ids"],
        dataset_dict["token_type_ids"],
        dataset_dict["attention_mask"],
    ]
    y = [dataset_dict["start_token_idx"], dataset_dict["end_token_idx"]]
    return x, y


train_squad_examples = create_squad_examples(raw_train_data)
x_train, y_train = create_inputs_targets(train_squad_examples)
print(f"{len(train_squad_examples)} training points created.")

eval_squad_examples = create_squad_examples(raw_eval_data)
x_eval, y_eval = create_inputs_targets(eval_squad_examples)
print(f"{len(eval_squad_examples)} evaluation points created.")

with open("benchmarks/bert_data.pkl", "wb") as f:
    pickle.dump([x_train, y_train, x_eval, y_eval], f)


def subset(data, fraction=0.001):
    return [x[: int(len(x) * fraction)] for x in data]


x_train_small = subset(x_train)
y_train_small = subset(y_train)

x_eval_small = subset(x_eval)
y_eval_small = subset(y_eval)

# with open("benchmarks/bert_data_extra_small.pkl", "wb") as f:
#    pickle.dump([x_train_small, y_train_small, x_eval_small, y_eval_small], f)

np.save("benchmarks/x_train_small.npy", x_train_small)
np.save("benchmarks/y_train_small.npy", y_train_small)

np.save("benchmarks/x_eval_small.npy", x_eval_small)
np.save("benchmarks/y_eval_small.npy", y_eval_small)
