from gensim.models import KeyedVectors


class FastText(object):

    def __init__(self, configuration=None):

        # Initialize variables
        self.configuration = configuration

        self.fasttext_model = KeyedVectors.load_word2vec_format(
            "./models/fasttext-wiki-news-300d-1M-subword.vec.gz"
        )

    """
	Required codenames methods
	"""

    def get_weighted_nn(self, word, n=500):
        nn_w_similarities = dict()

        if word not in self.fasttext_model.key_to_index:
            return nn_w_similarities
        neighbors_and_similarities = self.fasttext_model.most_similar(word, topn=n)
        for neighbor, similarity in neighbors_and_similarities:
            if len(neighbor.split("_")) > 1 or len(neighbor.split("-")) > 1:
                continue
            neighbor = neighbor.lower()
            if neighbor not in nn_w_similarities:
                nn_w_similarities[neighbor] = similarity
            nn_w_similarities[neighbor] = max(similarity, nn_w_similarities[neighbor])

        return {k: v for k, v in nn_w_similarities.items() if k != word}

    def rescale_score(self, potential_clue, other_words):
        """
        :param chosen_words: potential board words we could apply this clue to
        :param clue: potential clue
        :param red_words: opponent's words
        returns: penalizes a potential_clue for being have high fasttext similarity with opponent's words
        """
        if potential_clue not in self.fasttext_model:
            return 0.0

        max_other_similarity = float("-inf")
        for other_word in other_words:
            if other_word in self.fasttext_model:
                similarity = self.fasttext_model.similarity(other_word, potential_clue)
                if similarity > max_other_similarity:
                    max_other_similarity = similarity

        return -0.5 * max_other_similarity

    def dict2vec_embedding_weight(self):
        return 2.0

    def get_word_similarity(self, word1, word2):
        try:
            return self.fasttext_model.similarity(word1, word2)
        except KeyError:
            return -1.0
