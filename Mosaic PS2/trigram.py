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


## NOW LET'S MAKE SOME COOL SHIT TRIGRAM MODEL WITH SOME SMOOTHING ##

def trigram_convert_word(word):
    return "$$" + word


def trigram(corpus):
    trigram_counts = Counter()
    bigram_counts = defaultdict()

    for word in corpus:
        word = trigram_convert_word(word)

        trigram_list = zip(word[:-2], word[1:-1], word[2:])

        bigram_list = zip(word[:-1], word[1:])

        for trigram in trigram_list:
            first, second, third = trigram
            element = first + second + third
            trigram_counts[element] += 1

        for bigrams in bigram_list:
            first, second = bigram
            bigram_counts[first][second] += 1

    return trigram_counts, bigram_counts


trigram_counts, bigram_counts_for_trigram = trigram(corpus)

def trigram_prob(wi_2,wi_1, char, trigram_counts, bigram_counts):
    return trigram_counts[wi_2 + wi_1 + char] / float(bigram_counts[wi_2][wi_1])

def final_guesser(mask, guessed, bigram_counts = bigram_counts_for_trigram, trigram_counts = trigram_counts, unigram_counts=unigram_counts):

    available = list(set(string.ascii_lowercase)-guessed)

    trigram_probs = []

    mask = ['$','$'] + mask

    trigram_lambda = 0.6
    bigram_lambda = 0.3
    unigram_lambda = 0.1

    for char in available:
        char_prob = 0
        for index in range(len(mask)):

            if index == 0 and mask[index] == '_':
                if index == 0 and mask[index] == '_':
                    char_prob += trigram_lambda* trigram_prob('$','$',char, trigram_counts, bigram_counts)

            if index == 1 and mask[index] == '_':
                if not mask[index-1]=='_':
                    char_prob += trigram_lambda*trigram_prob('$', mask[index-1], char, trigram_counts, bigram_counts)

                elif mask[index -2] == '_' and not mask[index - 1] =='_':
                    char_prob += bigram_lambda*bigram_prob(mask[index-1],char,bigram_counts)

                else:
                    char_prob += unigram_lambda*unigram_counts[char]/float(sum(unigram_counts.values()))

            else:
                continue

        trigram_probs.append(char_prob)

    return available[trigram_probs.index()]

def suggest_next_letter_sol(displayed_word, guessed_letters):
    guessed = set(guessed_letters)
    pred_letter = final_guesser(displayed_word, guessed, bigram_counts = bigram_counts_for_trigram, trigram_counts = trigram_counts, unigram_counts=unigram_counts)

    return pred_letter
