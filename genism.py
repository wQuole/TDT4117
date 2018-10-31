import random;

random.seed(123)
import codecs
import string
import re
from nltk.stem.porter import PorterStemmer
import gensim


def filter_func(paragraphs, filterWord):
    """
    Filters out paragraphs which does not contain the filterWord
    :param paragraphs: List[String]
    :param filterWord: String
    :return: List[String]
    """
    result = []
    for paragraph in paragraphs:
        if len(paragraph) > 2:
            if filterWord not in paragraph.lower():
                result.append(paragraph)
    return result


def text_to_paragraph(filePath):
    """
    Returns a list containing the paragraphs
    :param filePath: URL_PATH
    :return: List[String]
    """
    f = codecs.open(filePath, "r", "utf-8")
    paragraphs = f.read()
    f.close()
    return paragraphs.split("\n\n")


def tokenize(paragraphs):
    """
    a list of the words in the paragraphs with punctuations and whitespace removed
    :param paragraphs: list[String], list containing paragraphs
    :return: List[List[String]]
    """
    tokens = []
    for paragraph in paragraphs:
        token = re.sub('[' + string.punctuation + ']', '', paragraph)
        token = re.sub('[\n\t\r]', '', token)
        token = token.split(" ")
        tokens.append(token)
    return tokens


def stem(tokens):
    """
    :param paragraphs: list[String], list containing paragraphs
    :return: a list of word where each paragraph is a list of processed lower-case words
    """
    stemmer = PorterStemmer()
    stemmedTokens = []
    for token in tokens:
        stemmedToken = []
        for word in token:
            stemmedToken.append(stemmer.stem(word.lower()))
        stemmedTokens.append(stemmedToken)
    return stemmedTokens


def get_stopwords():
    """
    :return: stopwords
    """
    f = open("stopwords.txt", "r")
    words = f.read().split(",")
    f.close()
    return words


def remove_stopwords(dictionary):
    """
    :param dictionary: gensim.corpa Dictionary
    :return: NULL
    """
    # Fetch stopwords from file
    stop_words = get_stopwords()

    # Filter out stopwords that are not in the dictionary's keys
    stop_words = list(filter(lambda w: w not in dictionary.token2id.keys(), stop_words))

    # Map the remaining stopwords to the ID's
    stop_ids = list(map(lambda w: dictionary.token2id[w], stop_words))

    # Remove the stopwords from the dictionary
    dictionary.filter_tokens(stop_ids)


def map_paragraph_to_bow(tokens, dictionary):
    """
    Maps paragraphs into Bags-of-Words. Each paragraph is now a list of tuples (word-index, word-count)
    :param paragraphs: List[List[String]]
    :param dictionary: gensim.corpa Dictionary
    :return: List[List[(int,int)]]
    """
    return list(map(lambda p: dictionary.doc2bow(p), tokens))


"""
M A I N 
"""


def main():
    # Data loading and preprocessing
    original_paragraphs = text_to_paragraph("gutenberg.txt")
    filtered = filter_func(original_paragraphs, "gutenberg")
    tokenized = tokenize(filtered)
    print(tokenized[:2])
    stemmed = stem(tokenized)
    print()
    print(stemmed[:2])

    # Dictionary building
    dictionary = gensim.corpora.Dictionary(stemmed)
    remove_stopwords(dictionary)


if __name__ == '__main__':
    main()
