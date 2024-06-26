import string
from collections import defaultdict
import random

words = "C:\\Users\\LENOVO\PycharmProjects\Mosaic24\Mosaic-24\Mosaic PS2\\training.txt"

with open(words, 'r') as file:
    corpus = file.readlines()

word_list = [word.strip() for word in corpus]

processed_words = []
for word in word_list:
    if word.isalpha():
        processed_words.append(word.lower())

unique_words = list(set(processed_words))

import numpy as np

unique_words = np.array(unique_words)
np.random.shuffle(unique_words)
corpus = unique_words.tolist()

## UNIGRAM LANGUAGE MODEL ##

from collections import Counter


def unigram(corpus):
    unigram_counts = Counter()

    for word in corpus:
        for char in word:
            unigram_counts[char] += 1

    return unigram_counts


unigram_counts = unigram(corpus)


def convert_word(word):
    return "$" + word


def bigram(corpus):
    bigram_counts = defaultdict(Counter)

    for word in corpus:
        word = convert_word(word)

        bigram_list = zip(word[:-1], word[1:])

        for bigram in bigram_list:
            first, second = bigram
            bigram_counts[first][second] += 1

    return bigram_counts


bigram_counts = bigram(corpus)


def bigram_prob(key, char, bigram_counts):
    prev_word_counts = bigram_counts[key]
    total_counts = float(sum(prev_word_counts.values()))

    return prev_word_counts[char] / float(sum(prev_word_counts.values()))


def bigram_guesser(mask, guessed, bigram_counts=bigram_counts, unigram_counts=unigram_counts):
    available = list(set(string.ascii_lowercase) - guessed)

    bigram_probs = []

    for char in available:
        char_prob = 0
        for index in range(len(mask)):

            if index == 0 and mask[index] == '_':
                char_prob += bigram_prob('$', char, bigram_counts)

            elif mask[index] == '_':

                if not mask[index - 1] == '_':
                    char_prob += bigram_prob(mask[index - 1], char, bigram_counts)

                else:
                    char_prob += unigram_counts[char] / float(sum(unigram_counts.values()))

            else:
                continue

        bigram_probs.append(char_prob)

    return available[bigram_probs.index(max(bigram_probs))]


def unigram_guesser(mask, guessed, unigram_counts=unigram_counts):
    copy_dict = unigram_counts.copy()

    for char in guessed:
        del copy_dict[char]

    return max(copy_dict, key=copy_dict.get)


def suggest_next_letter_sol(displayed_word, guessed_letters):
    guessed =set(guessed_letters)
    pred_letter = bigram_guesser(displayed_word, guessed,bigram_counts=bigram_counts, unigram_counts=unigram_counts)

    return pred_letter
