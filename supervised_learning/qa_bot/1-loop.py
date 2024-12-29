#!/usr/bin/env python3

"""
question answer task 0
"""

import tensorflow_hub as hub
import tensorflow as tf
from transformers import BertTokenizer


def question_answer(question, reference):
    """
    question answer task
    """
    tokenizer = BertTokenizer.from_pretrained(
        'bert-large-uncased-whole-word-masking-finetuned-squad')
    model = hub.load("https://tfhub.dev/see--/bert-uncased-tf2-qa/1")

    """Tokenize the input question and reference text"""
    reference = tokenizer.tokenize(reference)
    reference = reference + ['[SEP]']
    reference_ids = tokenizer.convert_tokens_to_ids(reference)
    question = tokenizer.tokenize(question)
    question = ['[CLS]'] + question + ['[SEP]']
    question_tokens_ids = tokenizer.convert_tokens_to_ids(question)
    input_ids = question_tokens_ids + reference_ids
    input_mask = [1] * len(input_ids)
    input_types = [0] * len(question) + [1] * len(reference)

    """convert list to tensors and expand dimensions"""
    input_ids = tf.convert_to_tensor(input_ids, dtype=tf.int32)
    input_mask = tf.convert_to_tensor(input_mask, dtype=tf.int32)
    input_types = tf.convert_to_tensor(input_types, dtype=tf.int32)
    input_ids = tf.expand_dims(input_ids, axis=0)
    input_mask = tf.expand_dims(input_mask, axis=0)
    input_types = tf.expand_dims(input_types, axis=0)

    """Get the model's predictions"""
    outputs = model([input_ids, input_mask, input_types])
    start_logits = tf.argmax(outputs[0][0][1:-1]) + 1
    end_logits = tf.argmax(outputs[1][0][1:-1]) + 1

    """extract answer tokens and convert them to a string"""
    tokens = question + reference
    answer_tokens = tokens[start_logits: end_logits + 1]
    if len(answer_tokens) == 0:
        return None
    else:
        answer = tokenizer.convert_tokens_to_string(answer_tokens)
    return answer

def main():
    reference = input("Please enter the reference text: ")
    while True:
        question = input("Q: ")
        if question.lower() in ['exit', 'quit', 'goodbye', 'bye']:
            print("A: Goodbye")
            break
        answer = question_answer(question, reference)
        if answer:
            print(f"Q: {question}")
            print(f"A: {answer}")
        else:
            print(f"Q: {question}")
            print("A: I don't know the answer to that question.")

if __name__ == "__main__":
    main()