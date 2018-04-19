# coding: utf-8
"""
    Use word2vec to explore words that appear in similar contexts
"""
import logging
import time
import multiprocessing
import os
import gensim.models.word2vec as w2v
import sklearn.manifold
from sklearn.externals import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import spacy
from spacy.symbols import SPACE, PUNCT, NUM, SYM
import utils


do_lowercase = False

force_tokenize = False
force_train = False
force_tnse = False
do_plots = True

# Dimensions of word vectors.
num_features = 200
# Minimum number of times word must occur in sentences to be used in model.
min_word_count = 3
# Number of threads to run in parallel.
num_workers = multiprocessing.cpu_count()
# Context window length.
context_size = 10
# Downsample setting for frequent words. 0 - 1e-5 is good for this
downsampling = 1e-3
# Seed for the RNG
seed = 114

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

train_dir = "tone.ranger.trained"
sentences_path = 'all.sentences.json'
model_name = "papercut"

base_name = '%s.w%d.f%d.c%d' % (model_name, min_word_count, num_features, context_size)
model_name = '%s.w2v' % base_name
log_name = '%s.log' % base_name
tsne_name = '%s.tsne.pkl' % base_name

if do_lowercase:
    train_dir = '%s.lower' % train_dir
train_dir = '%s.spacy' % train_dir
words_path = os.path.join(train_dir, 'words.json')
log_path = os.path.join(train_dir, log_name)
model_path = os.path.join(train_dir, model_name)
tsne_path = os.path.join(train_dir, tsne_name)

log_f = open(log_path, 'wt')


def my_print(*args):
    print(args)
    print(args, file=log_f)


if force_tokenize or not os.path.exists(words_path):
    nlp = spacy.load('en')

    raw_sentences = utils.load_json(sentences_path)
    if do_lowercase:
        raw_sentences = [raw.lower() for raw in raw_sentences]

    # convert into a list of words
    def sentence_to_wordlist(raw):
        sent = nlp(raw)
        words = []
        for tok in sent:
            if tok.pos in {SPACE, NUM, PUNCT, SYM}:
                continue
            words.append(tok.text)
        return words

    # sentence where each word is tokenized
    sentences = []
    my_print('%7d sentences' % (len(raw_sentences)))
    interval = max(len(raw_sentences) // 50, 100)

    t0 = time.perf_counter()
    for i, raw in enumerate(raw_sentences):
        if len(raw) > 0:
            sentences.append(sentence_to_wordlist(raw))
        if i % interval == 0 and interval:
            my_print('%7d (%5.1f%%) %6.1f sec' % (i, 100.0 * i / len(raw_sentences),
                time.perf_counter() - t0))

    my_print('raw sentence 5:', raw_sentences[5])
    my_print('word list 5:', sentence_to_wordlist(raw_sentences[5]))

    token_count = sum([len(sentence) for sentence in sentences])
    my_print("The book corpus contains {0:,} tokens".format(token_count))

    utils.save_json(words_path, sentences)

if force_train or not os.path.exists(model_path):

    sentences = utils.load_json(words_path)

    tranger2vec = w2v.Word2Vec(
        sg=1,
        seed=seed,
        workers=num_workers,
        size=num_features,
        min_count=min_word_count,
        window=context_size,
        sample=downsampling
        )

    tranger2vec.build_vocab(sentences)
    my_print("Word2Vec vocabulary length:", len(tranger2vec.wv.vocab))
    my_print(type(tranger2vec.wv.vocab))

    tranger2vec.train(sentences,
                      total_examples=tranger2vec.corpus_count,
                      epochs=tranger2vec.iter)

    os.makedirs(train_dir, exist_ok=True)
    tranger2vec.save(model_path)

#
# Show word similarities
#
tranger2vec = w2v.Word2Vec.load(model_path)

if force_tnse or not os.path.exists(tsne_path):

    # ### Compress the word vectors into 2D space and plot them
    tsne = sklearn.manifold.TSNE(n_components=2, random_state=0)

    all_word_vectors_matrix = tranger2vec.wv.syn0

    print('t-sne')
    t0 = time.perf_counter()
    all_word_vectors_matrix_2d = tsne.fit_transform(all_word_vectors_matrix)
    my_print('t-sne took %1.f sec' % (time.perf_counter() - t0))

    joblib.dump(all_word_vectors_matrix_2d, tsne_path)

if do_plots:
    all_word_vectors_matrix_2d = joblib.load(tsne_path)
    # **Plot the big picture**
    points = pd.DataFrame(
        [
            (word, coords[0], coords[1])
            for word, coords in [
                (word, all_word_vectors_matrix_2d[tranger2vec.wv.vocab[word].index])
                for word in tranger2vec.wv.vocab
            ]
        ],
        columns=["word", "x", "y"]
    )

    my_print(points.head(10))
    my_print(points.describe())

    sns.set_context("poster")
    my_print('plot')
    points.plot.scatter("x", "y", s=10, figsize=(10, 6))
    plt.show()

    def plot_region(x_bounds, y_bounds):
        """ Zoom in to some interesting places """
        slice = points[
            (x_bounds[0] <= points.x) & (points.x <= x_bounds[1]) &
            (y_bounds[0] <= points.y) & (points.y <= y_bounds[1])
        ]
        my_print(type(slice))
        my_print(slice.describe())
        if slice.empty:
            print('nothing to plot')
            return
        ax = slice.plot.scatter("x", "y", s=35, figsize=(10, 8))
        for i, point in slice.iterrows():
            ax.text(point.x + 0.005, point.y + 0.005, point.word, fontsize=11)

    plot_region(x_bounds=(0, 0.8), y_bounds=(0, 2))
    # plot_region(x_bounds=(4.0, 4.2), y_bounds=(-0.5, -0.1))
    plot_region(x_bounds=(0, 1), y_bounds=(3, 4.5))


def show_closest(term):
    """Show words that are closest to `term`"""
    if do_lowercase:
        term = term.lower()
    try:
        similar = tranger2vec.most_similar(term)
    except KeyError as e:
        try:
            similar = tranger2vec.most_similar(term.lower())
        except KeyError as e:
            my_print(e)
            return

    my_print('Most similar to %r' % term)
    for i, (t, v) in enumerate(similar):
        my_print('%4d: %15s %.3f' % (i, t, v))


def show_related_(start1, end1, end2):
    """Show word that is to `start1` as `end1` is to `end2 """
    if do_lowercase:
        start1, end1, end2 = (w.lower() for w in (start1, end1, end2))

    try:
        similarities = tranger2vec.most_similar_cosmul(
            positive=[end2, start1],
            negative=[end1]
        )
    except KeyError as e:
        my_print(e)
        return None

    start2 = similarities[0][0]
    my_print("{start1} is related to {end1}, as {start2} is related to {end2}".format(**locals()))
    return start2


def show_related(start1, end1, end2):
    show_related_(start1, end1, end2)
    show_related_(end1, start1, end2)


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
    show_closest(term)

show_related("Geoff", "Chris", "paper")
show_related("Geoff", "Chris", "boy")
show_related("Geoff", "Chris", "man")
show_related("Geoff", "Chris", "dog")
show_related("Chris", "Matt", "Peter")
show_related("copier", "MFP", "save")
show_related("copier", "printer", "stoner")
show_related("copier", "MFP", "save")
show_related("print", "scan", "Peter")
show_related("copier", "printer", "print")
show_related("copier", "printer", "watermark")
