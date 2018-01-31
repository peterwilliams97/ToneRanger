# coding: utf-8

import codecs
import glob
import logging
import time
import multiprocessing
import os
import re
import nltk
import gensim.models.word2vec as w2v
import sklearn.manifold
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import spacy
from spacy.symbols import SPACE, PUNCT, NUM, SYM
import utils


do_got = False
do_lowercase = False
do_spacy = True

do_tokenize = False
do_train = False
do_plots = False

# Dimensionality of the resulting word vectors.
# More dimensions, more computationally expensive to train but also more accurate
# more dimensions = more generalized
num_features = 600
# Minimum word count threshold.
min_word_count = 1
# Number of threads to run in parallel.
num_workers = multiprocessing.cpu_count()
# Context window length.
context_size = 7
# Downsample setting for frequent words.
# 0 - 1e-5 is good for this
downsampling = 1e-3

# Seed for the RNG, to make the results reproducible.
seed = 1

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


# **Download NLTK tokenizer models (only the first time)**
# nltk.download("punkt")
# nltk.download("stopwords")
if do_got:
    train_dir = "w2v.trained"
    sentences_path = os.path.join(train_dir, 'sentences.json')
    model_name = "thrones2vec"

    book_filenames = sorted(glob.glob("data/*.txt"))
    my_print("Found books:", book_filenames)

    # **Combine the books into one string**
    corpus_raw = ""
    for book_filename in book_filenames:
        my_print("Reading '{0}'... ".format(book_filename), end='')
        with codecs.open(book_filename, "r", "utf-8") as book_file:
            corpus_raw += book_file.read()
        my_print("Corpus is now {0} characters long".format(len(corpus_raw)))

    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    raw_sentences = tokenizer.tokenize(corpus_raw)

    utils.save_json(sentences_path, raw_sentences)

else:
    train_dir = "tone.ranger.trained"
    sentences_path = 'all.sentences.json'
    model_name = "papercut"

model_name = '%s.w%d.f%d.w2v' % (model_name, min_word_count, num_features)
log_name = '%s.w%d.f%d.log' % (model_name, min_word_count, num_features)

if do_lowercase:
    train_dir = '%s.lower' % train_dir
if do_spacy:
    train_dir = '%s.spacy' % train_dir
    nlp = spacy.load('en')
words_path = os.path.join(train_dir, 'words.json')
log_path = os.path.join(train_dir, log_name)

log_f = open(log_path, 'wt')


def my_print(*args):
    print(args)
    print(args, file=log_f)


if do_tokenize:
    raw_sentences = utils.load_json(sentences_path)
    if do_lowercase:
        raw_sentences = [raw.lower() for raw in raw_sentences]

    # convert into a list of words
    # remove unnnecessary,, split into words, no hyphens
    # list of words
    if do_spacy:
        def sentence_to_wordlist(raw):
            sent = nlp(raw)
            words = []
            for tok in sent:
                word = tok.text
                if tok.pos in {SPACE, NUM, PUNCT, SYM}:
                    continue
                words.append(tok.text)
            return words
    else:
        def sentence_to_wordlist(raw):
            clean = re.sub("[^a-zA-Z]", " ", raw)
            words = clean.split()
            return words

    # sentence where each word is tokenized
    sentences = []
    my_print('%7d sentences' % (len(raw_sentences)))
    interval = max(len(raw_sentences) // 50, 100)

    t0 = time.clock()
    for i, raw in enumerate(raw_sentences):
        if len(raw) > 0:
            sentences.append(sentence_to_wordlist(raw))
        if i % interval == 0 and interval:
            my_print('%7d (%5.1f%%) %6.1f sec' % (i, 100.0 * i / len(raw_sentences), time.clock() - t0))

    my_print('raw sentence 5:', raw_sentences[5])
    my_print('word list 5:', sentence_to_wordlist(raw_sentences[5]))

    token_count = sum([len(sentence) for sentence in sentences])
    my_print("The book corpus contains {0:,} tokens".format(token_count))

    utils.save_json(words_path, sentences)
    # assert False


if do_train:

    sentences = utils.load_json(words_path)

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
    my_print("Word2Vec vocabulary length:", len(thrones2vec.wv.vocab))
    my_print(type(thrones2vec.wv.vocab))
    assert 'Kelby' in thrones2vec.wv.vocab
    for sent in sentences:
        for word in sent:
            assert word in thrones2vec.wv.vocab, word
    # assert False

    thrones2vec.train(sentences,
                      total_examples=thrones2vec.corpus_count,
                      epochs=thrones2vec.iter)

    # **Save to file, can be useful later**
    os.makedirs(train_dir, exist_ok=True)
    thrones2vec.save(os.path.join(train_dir, model_name))

# ## Explore the trained model.
thrones2vec = w2v.Word2Vec.load(os.path.join(train_dir, model_name))


if do_plots:
    # ### Compress the word vectors into 2D space and plot them
    tsne = sklearn.manifold.TSNE(n_components=2, random_state=0)

    all_word_vectors_matrix = thrones2vec.wv.syn0

    my_print('t-sne')
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

    my_print(points.head(10))
    my_print(points.describe())

    sns.set_context("poster")
    my_print('plot')
    points.plot.scatter("x", "y", s=10, figsize=(20, 12))
    plt.show()


    def plot_region(x_bounds, y_bounds):
        """ Zoom in to some interesting places """
        slice = points[
            (x_bounds[0] <= points.x) & (points.x <= x_bounds[1]) &
            (y_bounds[0] <= points.y) & (points.y <= y_bounds[1])
        ]
        my_print(type(slice))
        my_print(slice.describe())
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
def show_similer(term):
    if do_lowercase:
        term = term.lower()
    try:
        similar = thrones2vec.most_similar(term)
    except KeyError as e:
        try:
            similar = thrones2vec.most_similar(term.lower())
        except KeyError as e:
            my_print(e)
            return

    my_print('Most similar to %r' % term)
    for i, (t, v) in enumerate(similar):
        my_print('%4d: %15s %.3f' % (i, t, v))


if do_got:
    show_similer("Stark")
    show_similer("Aerys")
    show_similer("direwolf")
else:
    for term in ["PaperCut", "copier", "Dance", "print", "printer", "printing", "user", "server",
                 'support', 'NG', 'MF', 'Mobility', 'software', 'company', 'version',
                 'Windows', 'DNS', 'network', 'PostScript', 'Xerox', 'Konica', 'Ricoh',
                 'archiving', 'watermark', 'cloud', 'LDAP',
                 'Peter', 'Kelby', 'Damien', 'Jason', 'Thom', 'Alec', 'Travis', 'Geoff', 'Chris', 'Matt',
                 'Julie', 'Dean', 'Maria', 'Hendrik', 'Tom', 'Tim',
                 'Troubleshoot', 'License', 'email', 'Administrator', 'password', 'account',
                 'Williams', 'White', 'Clark', 'Beresford', 'Doran', 'Clews', 'Smith', 'Dance',
                 'PaperCutty', 'PaperCutter', 'good', 'bad', 'clever', 'Agile', 'buzzword',
                 'scrum', 'kanban'
                 ]:
        show_similer(term)

# **Linear relationships between word pairs**
def nearest_similarity_cosmul(start1, end1, end2):
    if do_lowercase:
        start1, end1, end2 = (w.lower() for w in (start1, end1, end2))
    similarities = thrones2vec.most_similar_cosmul(
        positive=[end2, start1],
        negative=[end1]
    )
    start2 = similarities[0][0]
    my_print("{start1} is related to {end1}, as {start2} is related to {end2}".format(**locals()))
    return start2


nearest_similarity_cosmul("Geoff", "Chris", "paper")
nearest_similarity_cosmul("Chris", "Matt", "Peter")
nearest_similarity_cosmul("copier", "MFP", "save")
nearest_similarity_cosmul("print", "scan", "Peter")
nearest_similarity_cosmul("printer", "copier", "print")
nearest_similarity_cosmul("printer", "copier", "watermark")
