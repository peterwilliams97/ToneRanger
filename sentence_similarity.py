# coding: utf-8

import codecs
import glob
import logging
import multiprocessing
import os
import re
import nltk
import gensim.models.word2vec as w2v
import sklearn.manifold
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


do_train = False
do_plots = False

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# **Download NLTK tokenizer models (only the first time)**
# nltk.download("punkt")
# nltk.download("stopwords")

if do_train:
    book_filenames = sorted(glob.glob("data/*.txt"))
    print("Found books:", book_filenames)

    # **Combine the books into one string**
    corpus_raw = ""
    for book_filename in book_filenames:
        print("Reading '{0}'...".format(book_filename))
        with codecs.open(book_filename, "r", "utf-8") as book_file:
            corpus_raw += book_file.read()
        print("Corpus is now {0} characters long".format(len(corpus_raw)))
        print()

    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    raw_sentences = tokenizer.tokenize(corpus_raw)

    # convert into a list of words
    # remove unnnecessary,, split into words, no hyphens
    # list of words
    def sentence_to_wordlist(raw):
        clean = re.sub("[^a-zA-Z]", " ", raw)
        words = clean.split()
        return words

    # sentence where each word is tokenized
    sentences = []
    for raw_sentence in raw_sentences:
        if len(raw_sentence) > 0:
            sentences.append(sentence_to_wordlist(raw_sentence))

    print(raw_sentences[5])
    print(sentence_to_wordlist(raw_sentences[5]))

    token_count = sum([len(sentence) for sentence in sentences])
    print("The book corpus contains {0:,} tokens".format(token_count))


    # ## Train Word2Vec

    #ONCE we have vectors
    #step 3 - build model
    #3 main tasks that vectors help with
    #DISTANCE, SIMILARITY, RANKING

    # Dimensionality of the resulting word vectors.
    # More dimensions, more computationally expensive to train but also more accurate
    # more dimensions = more generalized
    num_features = 300
    # Minimum word count threshold.
    min_word_count = 3

    # Number of threads to run in parallel.
    num_workers = multiprocessing.cpu_count()

    # Context window length.
    context_size = 7

    # Downsample setting for frequent words.
    # 0 - 1e-5 is good for this
    downsampling = 1e-3

    # Seed for the RNG, to make the results reproducible.
    seed = 1

    thrones2vec = w2v.Word2Vec(
        sg=1,
        seed=seed,
        workers=num_workers,
        size=num_features,
        min_count=min_word_count,
        window=context_size,
        sample=downsampling
        )

    thrones2vec.build_vocab(sentences)
    print("Word2Vec vocabulary length:", len(thrones2vec.wv.vocab))

    thrones2vec.train(sentences,
                      total_examples=thrones2vec.corpus_count,
                      epochs=thrones2vec.iter)

    # **Save to file, can be useful later**
    os.makedirs("w2v.trained", exist_ok=True)
    thrones2vec.save(os.path.join("w2v.trained", "thrones2vec.w2v"))

# ## Explore the trained model.
thrones2vec = w2v.Word2Vec.load(os.path.join("w2v.trained", "thrones2vec.w2v"))


if do_plots:
    # ### Compress the word vectors into 2D space and plot them
    tsne = sklearn.manifold.TSNE(n_components=2, random_state=0)

    all_word_vectors_matrix = thrones2vec.wv.syn0

    print('t-sne')
    all_word_vectors_matrix_2d = tsne.fit_transform(all_word_vectors_matrix)


    # **Plot the big picture**
    points = pd.DataFrame(
        [
            (word, coords[0], coords[1])
            for word, coords in [
                (word, all_word_vectors_matrix_2d[thrones2vec.wv.vocab[word].index])
                for word in thrones2vec.wv.vocab
            ]
        ],
        columns=["word", "x", "y"]
    )

    print(points.head(10))
    print(points.describe())

    sns.set_context("poster")
    print('plot')
    points.plot.scatter("x", "y", s=10, figsize=(20, 12))
    plt.show()


    def plot_region(x_bounds, y_bounds):
        """ Zoom in to some interesting places """
        slice = points[
            (x_bounds[0] <= points.x) & (points.x <= x_bounds[1]) &
            (y_bounds[0] <= points.y) & (points.y <= y_bounds[1])
        ]
        print(type(slice))
        print(slice.describe())
        ax = slice.plot.scatter("x", "y", s=35, figsize=(10, 8))
        for i, point in slice.iterrows():
            ax.text(point.x + 0.005, point.y + 0.005, point.word, fontsize=11)


    # **People related to Kingsguard ended up together**
    plot_region(x_bounds=(0, 0.8), y_bounds=(0, 2))
    # plot_region(x_bounds=(4.0, 4.2), y_bounds=(-0.5, -0.1))

    # **Food products are grouped nicely as well. Aerys (The Mad King) being close to "roasted" also looks sadly correct**
    plot_region(x_bounds=(0, 1), y_bounds=(3, 4.5))

# ### Explore semantic similarities between book characters
# **Words closest to the given word**
thrones2vec.most_similar("Stark")
thrones2vec.most_similar("Aerys")
thrones2vec.most_similar("direwolf")


# **Linear relationships between word pairs**
def nearest_similarity_cosmul(start1, end1, end2):
    similarities = thrones2vec.most_similar_cosmul(
        positive=[end2, start1],
        negative=[end1]
    )
    start2 = similarities[0][0]
    print("{start1} is related to {end1}, as {start2} is related to {end2}".format(**locals()))
    return start2


nearest_similarity_cosmul("Stark", "Winterfell", "Riverrun")
nearest_similarity_cosmul("Jaime", "sword", "wine")
nearest_similarity_cosmul("Arya", "Nymeria", "dragons")
