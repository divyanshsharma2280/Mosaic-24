from collections import defaultdict
import random

words = "C:\\Users\\LENOVO\PycharmProjects\Mosaic24\Mosaic-24\Mosaic PS2\\training.txt"

with open(words, 'r') as file:
    corpus = file.readlines()


def generate_ngrams(text, n):
    ngrams = []
    for i in range(len(text) - n + 1):
        ngrams.append(text[i:i + n])
    return ngrams


def build_n_gram_model(n, corpus):
    ngram_model = defaultdict(lambda: defaultdict(int))
    for sentence in corpus:
        ngrams = generate_ngrams(sentence, n)
        for ngram in ngrams:
            if n == 1:
                context = ''  # No context for unigram
            else:
                context = ngram[:-1]  # Use all but the last character as context
            target = ngram[-1]  # Predict the last character given the context
            ngram_model[context][target] += 1

    return ngram_model


def build_ngram_model_reverse(n, corpus):
    ngram_model_reverse = defaultdict(lambda: defaultdict(int))
    for sentence in corpus:
        sentence = "^" + sentence[::-1]  # Reverse the sentence and add start token
        ngrams = generate_ngrams(sentence, n)
        for ngram in ngrams:
            if n == 1:
                context = ''  # No context for unigram
            else:
                context = ngram[1:]  # Use the context excluding the first character
            target = ngram[0]  # Predict the first character given the context
            ngram_model_reverse[context][target] += 1
    return ngram_model_reverse


bigram_model = build_n_gram_model(2, corpus)
trigram_model = build_n_gram_model(3, corpus)
fourgram_model = build_n_gram_model(4, corpus)
fivegram_model = build_n_gram_model(5, corpus)

bigram_model_reverse = build_ngram_model_reverse(2, corpus)
trigram_model_reverse = build_ngram_model_reverse(3, corpus)
fourgram_model_reverse = build_ngram_model_reverse(4, corpus)
fivegram_model_reverse = build_ngram_model_reverse(5, corpus)


def suggest_next_character_ngram(displayed_word, guessed_letters, n, ngram_model):
    context = displayed_word[-(n - 1):] if len(
        displayed_word) >= n - 1 else ''  # Use the last n-1 characters of displayed word as context
    possible_next_chars = ngram_model[context]
    total_occurrences = sum(possible_next_chars.values())
    probabilities = {char: count / total_occurrences for char, count in possible_next_chars.items() if
                     char not in guessed_letters}
    if not probabilities:
        return None  # No available characters to suggest
    next_char = random.choices(list(probabilities.keys()), weights=list(probabilities.values()))[0]
    return next_char


def suggest_previous_character_ngram(displayed_word, n, ngram_model_reverse):
    context = displayed_word[-n:] if len(
        displayed_word) >= n else ''  # Use the last n characters of displayed word as context
    possible_prev_chars = ngram_model_reverse[context]
    total_occurrences = sum(possible_prev_chars.values())
    prev_probabilities = {char: count / total_occurrences for char, count in possible_prev_chars.items()}
    prev_char = random.choices(list(prev_probabilities.keys()), weights=list(prev_probabilities.values()))[0]
    return prev_char


def build_unigram_model(corpus):
    # Initialize an empty dictionary to store character counts
    char_counts = defaultdict(int)

    # Iterate through each word in the corpus
    for word in corpus:
        # Iterate through each character in the word
        for char in word:
            # Increment the count for the character
            char_counts[char] += 1

    # Calculate total number of characters in the corpus
    total_chars = sum(char_counts.values())

    # Calculate probabilities for each character
    unigram_model = {char: count / total_chars for char, count in char_counts.items()}

    return unigram_model


unigram_model = build_unigram_model(corpus)


def suggest_next_letter_sol(displayed_word, guessed_letters):
    """_summary_

    This function takes in the current state of the game and returns the next letter to be guessed.
    displayed_word: str: The word being guessed, with underscores for unguessed letters.
    guessed_letters: list: A list of the letters that have been guessed so far.
    Use python hangman.py to check your implementation.
    """
    ################################################
    ################################################
    ################################################
    # Filter out characters already guessed
    available_chars = [char for char in unigram_model if char not in guessed_letters]

    print("Available chars:", available_chars)

    # Calculate total probability for normalization
    total_probability = sum(unigram_model[char] for char in available_chars)

    print("Total probability:", total_probability)

    # Normalize probabilities
    probabilities = {char: unigram_model[char] / total_probability for char in available_chars}

    print("Probabilities:", probabilities)

    # Predict character with highest probability
    pred_letter = max(probabilities, key=probabilities.get)

    return pred_letter
    ################################################
    ################################################
    ################################################
    raise NotImplementedError
