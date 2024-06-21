import heapq
import itertools
import os
import random
import pickle

# gensim
from gensim.corpora import Dictionary
import gensim.downloader as api

# nltk
from nltk.stem import PorterStemmer

# Embeddings
from embedding.fasttext import FastText

# stopwords
from utils.stopwords import stopwords

# heuristic score
from utils.utils import get_dict2vec_score

import time

idf_lower_bound = 0.0006

fasttext = FastText()


def load_document_frequencies():
    """
    Sets up a dictionary from words to their document frequency
    """
    if os.path.exists("./models/word_to_df.pkl"):
        with open("./models/word_to_df.pkl", "rb") as f:
            word_to_df = pickle.load(f)

    else:
        dataset = api.load("text8")
        dct = Dictionary(dataset)
        id_to_doc_freqs = dct.dfs
        word_to_df = {dct[id]: id_to_doc_freqs[id] for id in id_to_doc_freqs}

    return word_to_df


def generate_board_words(first="red"):
    idx_to_word = dict()

    with open("./utils/codewords.txt") as file:
        for i, line in enumerate(file):
            word = line.strip().lower()
            idx_to_word[i] = word

    rand_idxs = random.sample(range(0, len(idx_to_word.keys())), 25)

    if first == "red":
        num_red_words = 9
    else:
        num_red_words = 8

    red_words = set([idx_to_word[idx] for idx in rand_idxs[:num_red_words]])
    blue_words = set([idx_to_word[idx] for idx in rand_idxs[num_red_words:17]])
    bystanders = set([idx_to_word[idx] for idx in rand_idxs[17:24]])
    assassin = idx_to_word[rand_idxs[24]]

    word_dic = []

    index = 0

    for word in red_words:
        word_dic.append(
            {"index": index, "type": "red", "word": word, "selected": False}
        )
        index += 1

    for word in blue_words:
        word_dic.append(
            {"index": index, "type": "blue", "word": word, "selected": False}
        )
        index += 1

    for word in bystanders:
        word_dic.append(
            {"index": index, "type": "bystander", "word": word, "selected": False}
        )
        index += 1

    word_dic.append(
        {"index": index, "type": "assassin", "word": assassin, "selected": False}
    )

    return word_dic


def get_clue(player_words, other_words, n):
    word_to_df = load_document_frequencies()
    weighted_nn = dict()
    words = player_words + other_words

    for word in words:
        weighted_nn[word] = fasttext.get_weighted_nn(word)

    pq = []
    for word_set in itertools.combinations(player_words, n):
        highest_clues, score = get_highest_clue(
            word_set, other_words, word_to_df, weighted_nn
        )
        # min heap, so push negative score
        heapq.heappush(pq, (-1 * score, highest_clues, word_set))

    # sliced_labels = self.get_cached_labels_from_synset(clue)
    # main_sense, _senses = self.get_cached_labels_from_synset_v5(clue)

    best_clues = []
    best_board_words_for_clue = []
    best_scores = []
    count = 0

    while pq and count <= 5:
        score, clues, word_set = heapq.heappop(pq)

        best_clues.append(clues)
        best_scores.append(score)
        best_board_words_for_clue.append(word_set)

        count += 1

    return best_scores, best_clues, best_board_words_for_clue


def is_valid_clue(words, clue):
    stemmer = PorterStemmer()
    for board_word in words:
        # Check if clue or board_word are substring of each other, or if they share the same word stem
        if (
            clue in board_word
            or board_word in clue
            or stemmer.stem(clue) == stemmer.stem(board_word)
            or not clue.isalpha()
        ):
            return False
    return True


def get_highest_clue(chosen_words, other_words, word_to_df, weighted_nn):

    words = list(chosen_words) + list(other_words)

    potential_clues = set()
    for word in chosen_words:
        nns = weighted_nn[word]
        potential_clues.update(nns)

    highest_scoring_clues = []
    highest_score = float("-inf")

    for clue in potential_clues:
        # don't consider clues which are a substring of any board words
        if not is_valid_clue(words, clue):
            continue
        player_word_counts = []
        for chosen_word in chosen_words:
            if clue in weighted_nn[chosen_word]:
                player_word_counts.append(weighted_nn[chosen_word][clue])
            else:
                player_word_counts.append(
                    fasttext.get_word_similarity(chosen_word, clue)
                )

        heuristic_score = 0

        # the larger the idf is, the more uncommon the word
        idf = (1.0 / word_to_df[clue]) if clue in word_to_df else 1.0

        # prune out super common words (e.g. "get", "go")
        if clue in stopwords or idf < idf_lower_bound:
            idf = 1.0
        dict2vec_weight = fasttext.dict2vec_embedding_weight()
        dict2vec_score = dict2vec_weight * get_dict2vec_score(
            chosen_words, clue, other_words
        )

        heuristic_score = dict2vec_score + (-2 * idf)

        # Give embedding methods the opportunity to rescale the score using their own heuristics
        embedding_score = fasttext.rescale_score(clue, other_words)

        score = sum(player_word_counts) + embedding_score + heuristic_score

        if score > highest_score:
            highest_scoring_clues = [clue]
            highest_score = score
        elif score == highest_score:
            highest_scoring_clues.append(clue)

    return highest_scoring_clues, highest_score


def generate_game_clues(player_words, other_words):

    if len(player_words) < 3:
        num_intended_words = len(player_words)
    else:
        num_intended_words = random.choice([1, 2])

    best_scores, best_clues, best_board_words_for_clue = get_clue(
        player_words, other_words, num_intended_words
    )

    return {
        "clue": best_clues[0],
        "intended_words": best_board_words_for_clue[0],
        "num_intended_words": num_intended_words,
    }
