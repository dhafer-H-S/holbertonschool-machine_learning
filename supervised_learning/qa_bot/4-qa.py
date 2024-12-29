#!/usr/bin/env python3

import tensorflow as tf
semantic_search = __import__('3-semantic_search').semantic_search
single_question_answer = __import__('0-qa').answer_question


""" task 4 multi reference question answer """
def question_answer(corpus_path):
    """answers questions from multiple reference texts:"""
    while True:
        question = input("Q: ").lower()

        if question in ["exit", "quit", "goodbye", "bye"]:
            print("A: Goodbye")
            exit()
        else:
            text = semantic_search(corpus_path, question)
            answer = single_question_answer(question, text)
            if answer is None:
                print("A: Sorry, I do not understand your question.")
            else:
                print("A:", answer)
