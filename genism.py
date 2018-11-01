import random
#random.seed(123)
import codecs
import string
import re
from nltk.stem.porter import PorterStemmer
import gensim
import logging

# Logging events
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


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
        token = re.sub('[' + string.punctuation + ']', ' ', paragraph)
        token = re.sub('[\n\t\r]', ' ', token)
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


def remove_stopwords(dictionary) -> object:
    """
    Removes stopwords from dictionary
    :param dictionary: gensim.corpa Dictionary
    """
    # Fetch stopwords from file
    stop_words = get_stopwords()

    # Filter out stopwords that are not in the dictionary's keys
    stop_words = list(filter(lambda w: w in dictionary.token2id.keys(), stop_words))

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


def preprocessing(query, dictionary):
    query[0] = query[0].lower()
    tokenized = tokenize(query)
    stemmed = stem(tokenized)
    #print("\nS T E M M E D:\n",stemmed,"\n")
    bags_of_words = map_paragraph_to_bow(stemmed, dictionary)
    return bags_of_words


def main():
    """
    M A I N
    """

    """
    Data loading and preprocessing
    """
    original_paragraphs = text_to_paragraph("gutenberg.txt")
    filtered = filter_func(original_paragraphs, "gutenberg")
    tokenized = tokenize(filtered)
    stemmed = stem(tokenized)


    """
    Dictionary building
    """
    dictionary = gensim.corpora.Dictionary(stemmed)
    remove_stopwords(dictionary)
    bags_of_words = map_paragraph_to_bow(stemmed, dictionary)


    """
    Retrieval models
    """
    # Building TF-IDF model
    tfidf_model = gensim.models.TfidfModel(bags_of_words)
    # Map BOW into TF-IDF weights
    tfidf_corpus = tfidf_model[bags_of_words]
    # Construct MatrixSimilarity object to calculate similarities between paragraph and queries
    tfidf_index = gensim.similarities.MatrixSimilarity(tfidf_corpus, num_features=len(dictionary))

    # Initialize an LSI transformation and create a double wrapper over the original corpus:
    # bow->tfidf->fold-in-lsi
    lsi_model = gensim.models.LsiModel(tfidf_corpus, id2word=dictionary, num_topics=100)
    lsi_corpus = lsi_model[tfidf_corpus]
    lsi_index = gensim.similarities.MatrixSimilarity(lsi_corpus, num_features=len(dictionary))

    lsi_model.show_topics(3)

    """
    Querying
    """
    # Prep query - T F I D F
    #query = ["What is the function of money?"]
    query = ["How taxes influences Economics?"]
    bow_query = preprocessing(query, dictionary)

    # Converting BOW to TF-IDF representation
    tfidf_query = tfidf_model[bow_query][0]

    # Report top 3 most relevant paragraphs for query
    doc2similarity = enumerate(tfidf_index[tfidf_query])
    rel_tfidf = sorted(doc2similarity, key=lambda kv: -kv[1])[:3]
    for rel in rel_tfidf:
        i, score = rel
        paragraph = original_paragraphs[i]
        print("[paragraph {}]".format(i))
        print("\n", paragraph.split("\n")[:6], "\n")

    # Prep query - L S I
    # Convert query tf_idf representation into LSI-topics representation (weights)
    lsi_query = lsi_model[tfidf_query]
    sorted_lsi_query = sorted(lsi_query, key=lambda kv: -abs(kv[1]))[:3]

    # Report the top 3 (most significant weight)
    topic_ids = list(map(lambda p: p[0], sorted_lsi_query))
    for id in topic_ids:
        print("[Topic {}]".format(id))
        print(lsi_model.print_topic(id),"\n")


    # Report the top 3 (most relevant paragraphs according to LSI model)
    doc2similarity = enumerate(lsi_index[lsi_query])
    rel_lsi = sorted(doc2similarity, key=lambda kv: -kv[1])[:3]
    # Print
    for rel in rel_lsi:
        i, score = rel
        paragraph = original_paragraphs[i]
        print("[paragraph {}]".format(i))
        print("\n",paragraph.split("\n")[:6],"\n") #can use split("\n")[:6] on paragraph to only show first 5 lines.




if __name__ == '__main__':
    main()
