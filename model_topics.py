"""
    Topic modelling

    Follow https://github.com/RaRe-Technologies/gensim/blob/develop/docs/notebooks/topic_methods.ipynb

    Better: https://github.com/RaRe-Technologies/gensim/blob/develop/docs/notebooks/gensim_news_classification.ipynb
"""
from glob import glob
import os
from os.path import join
import spacy
from spacy.symbols import SPACE, PUNCT, NUM, SYM
from utils import summaries_dir, load_json, save_json
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import logging
from pprint import pprint
from gensim import corpora
from gensim.models import Word2Vec, KeyedVectors
from gensim.models.word2vec import Text8Corpus
from gensim.corpora import Dictionary
from gensim.models import ldamodel


np.set_printoptions(linewidth=150)

TEMP_FOLDER = 'temp'
os.makedirs(TEMP_FOLDER, exist_ok=True)

logging.basicConfig(format='%(asctime)s %(filename)s:%(lineno)s %(levelname)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)

logger = logging.getLogger()
logger.info('Hello')
log = logger.info
log('Folder "{}" will be used to save temporary dictionary and corpus.'.format(TEMP_FOLDER))


texts = [['bank', 'river', 'shore', 'water'],
        ['river', 'water', 'flow', 'fast', 'tree'],
        ['bank', 'water', 'fall', 'flow'],
        ['bank', 'bank', 'water', 'rain', 'river'],
        ['river', 'water', 'mud', 'tree'],
        ['money', 'transaction', 'bank', 'finance'],
        ['bank', 'borrow', 'money'],
        ['bank', 'finance'],
        ['finance', 'money', 'sell', 'bank'],
        ['borrow', 'sell'],
        ['bank', 'loan', 'sell']]

dictionary = Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]

pprint(texts)
pprint(dictionary)
# corpus1 = [[x for x, y in c] for c in corpus]
# corpus2 = [[y for x, y in c] for c in corpus]
# pprint(corpus1)
# pprint(corpus2)
# print(type(corpus))
# print(type(corpus[0]))
# print(type(corpus[0][0]))


np.random.seed(1)  # setting random seed to get the same results each time.
models = []
for n in range(2, 3):
    print('-' * 100)
    model = ldamodel.LdaModel(corpus, id2word=dictionary, num_topics=n)
    models.append((n, model))

for n, model in models:
    print('-' * 100)
    print(n)
    model = ldamodel.LdaModel(corpus, id2word=dictionary, num_topics=n)
    model.show_topics()

    for term in ('water', 'money', 'bank', 'transaction', 'mud'):
        print('^' * 100)
        print(term)
        topics = model.get_term_topics(term)
        print(topics)


bow_water = ['bank', 'water', 'bank']
bow_finance = ['bank', 'finance', 'bank']


for terms in (bow_water, bow_finance):
    print('~' * 100)
    print(terms)
    bow = model.id2word.doc2bow(terms)  # convert to bag of words format first
    doc_topics, word_topics, phi_values = model.get_document_topics(bow, per_word_topics=True)
    pprint(doc_topics)
    pprint(word_topics)
    pprint(phi_values)


all_topics = model.get_document_topics(corpus, per_word_topics=True)

for text, (doc_topics, word_topics, phi_values) in zip(texts, all_topics):
    print('@' * 100)
    print(len(text), text)
    pprint(doc_topics)
    pprint(word_topics)
    pprint(phi_values)


def color_words(model, doc):
    import matplotlib.pyplot as plt
    # import matplotlib.patches as patches

    # make into bag of words
    doc = model.id2word.doc2bow(doc)
    # get word_topics
    doc_topics, word_topics, phi_values = model.get_document_topics(doc, per_word_topics=True)

    # color-topic matching
    topic_colors = {1: 'red', 0: 'blue'}

    # set up fig to plot
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])

    # a sort of hack to make sure the words are well spaced out.
    word_pos = 1 / len(doc)

    # use matplotlib to plot words
    for word, topics in word_topics:
        ax.text(word_pos, 0.8, model.id2word[word],
                horizontalalignment='center',
                verticalalignment='center',
                fontsize=20, color=topic_colors[topics[0]],  # choose just the most likely topic
                transform=ax.transAxes)
        word_pos += 0.2  # to move the word for the next iter

    ax.set_axis_off()
    plt.show()

# bow_water = ['bank', 'water', 'bank']
# color_words(model, bow_water)

# bow_finance = ['bank', 'finance', 'bank']
# color_words(model, bow_finance)

doc = ['bank', 'water', 'bank', 'finance', 'money','sell','river','fast','tree']
color_words(model, doc)

def color_words_dict(model, dictionary):
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    word_topics = []
    for word_id in dictionary:
        word = str(dictionary[word_id])
        # get_term_topics returns static topics, as mentioned before
        probs = model.get_term_topics(word)
        # we are creating word_topics which is similar to the one created by get_document_topics
        try:
            if probs[0][1] >= probs[1][1]:
                word_topics.append((word_id, [0, 1]))
            else:
                word_topics.append((word_id, [1, 0]))
        # this in the case only one topic is returned
        except IndexError:
            word_topics.append((word_id, [probs[0][0]]))

    # color-topic matching
    topic_colors = { 1:'red', 0:'blue'}

    # set up fig to plot
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])

    # a sort of hack to make sure the words are well spaced out.
    word_pos = 1/len(doc)

    # use matplotlib to plot words
    for word, topics in word_topics:
        ax.text(word_pos, 0.8, model.id2word[word],
                horizontalalignment='center',
                verticalalignment='center',
                fontsize=20, color=topic_colors[topics[0]],  # choose just the most likely topic
                transform=ax.transAxes)
        word_pos += 0.2 # to move the word for the next iter

    ax.set_axis_off()
    plt.show()

color_words_dict(model, dictionary)


assert False
################################################################################
documents = ["Human machine interface for lab abc computer applications",
             "A survey of user opinion of computer system response time",
             "The EPS user interface management system",
             "System and human system engineering testing of EPS",
             "Relation of user perceived response time to error measurement",
             "The generation of random binary unordered trees",
             "The intersection graph of paths in trees",
             "Graph minors IV Widths of trees and well quasi ordering",
             "Graph minors A survey"]

pprint(documents)


# remove common words and tokenize
stoplist = set('for a of the and to in'.split())
texts = [[word for word in document.lower().split() if word not in stoplist]
         for document in documents]
pprint(texts)

# remove words that appear only once

frequency = defaultdict(int)
for text in texts:
    for token in text:
        frequency[token] += 1

texts = [[token for token in text if frequency[token] > 1] for text in texts]

pprint(texts)


dictionary = corpora.Dictionary(texts)
dictionary.save(join(TEMP_FOLDER, 'deerwester.dict'))  # store the dictionary, for future reference
print(dictionary)
print(dictionary.token2id)

corpus = [dictionary.doc2bow(text) for text in texts]
corpora.MmCorpus.serialize(os.path.join(TEMP_FOLDER, 'deerwester.mm'), corpus)  # store to disk, for later use
for c in corpus:
    print(c)

print('=' * 100)

from gensim.models import Word2Vec, KeyedVectors
from gensim.models.word2vec import Text8Corpus
# Using params from Word2Vec_FastText_Comparison

params = {
    'alpha': 0.05,
    'size': 100,
    'window': 5,
    'iter': 5,
    'min_count': 5,
    'sample': 1e-4,
    'sg': 1,
    'hs': 0,
    'negative': 5
}

text8 = os.path.expanduser('~/data/text8/text8')
model = Word2Vec(Text8Corpus(text8), **params)
print(model)

assert False


threshold_words = 10


def describe(x):
    return '\n'.join([
        'num: %d' % len(x),
        'min-max: %d - %d' % (x.min(), x.max()),
        'mean: %.3f' % x.mean(),
        'median: %.3f' % np.median(x),
        '25_50_75: %s' % ['%.1f' % a for a in [np.percentile(x, q) for q in (25, 50, 75, 90)]],
    ])


def plot(arr, title):
    x = np.array(arr)
    x_min = x.min()
    x_max = np.percentile(x, 90)
    bins = np.arange(x_min, x_max)
    plt.xlim((x_min, x_max))
    plt.hist(x, bins=bins)
    plt.title(title)
    plt.figtext(0.4, 0.6, describe(x))
    print('-' * 80)
    print(title)
    print(describe(x))
    plt.show()


def summary_key(o):
    return -o['count:page_paras'], -o['count:page_chars'], o['path']


def compute_metrics(max_processed=-1):
    nlp = spacy.load('en')

    files = glob(join(summaries_dir, '*.json'))
    print('%4d files' % len(files))
    summaries = [load_json(path) for path in files]
    summaries.sort(key=summary_key)

    para_dist = defaultdict(int)
    sent_dist = defaultdict(int)
    word_dist = defaultdict(int)
    para_docs = defaultdict(set)
    sent_docs = defaultdict(set)
    word_docs = defaultdict(set)
    cnt_page_para = []
    cnt_para_sent = []
    cnt_para_word = []
    cnt_para_char = []
    cnt_sent_word = []
    cnt_sent_char = []
    cnt_word_char = []
    n_processed = 0
    for summary in summaries:
        page_path = summary['path']
        n_page_words = sum(sum(v) for v in summary['count:sent_chars'])
        if n_page_words < threshold_words:
            continue
        n_processed += 1
        if max_processed > 0:
            if n_processed > max_processed:
               break

        print('%3d: %s' % (n_processed, summary['path']))
        cnt_page_para.append(len(summary['text:paras']))

        for para in summary['text:paras']:
            para_dist[para] += 1
            para_docs[para].add(page_path)
            doc = nlp(para)

            sp = 0
            wp = 0
            cp = 0
            for sent in doc.sents:
                sent_dist[sent.text] += 1
                sent_docs[sent.text].add(page_path)
                sp += 1
                ws = 0
                cs = 0
                for tok in sent:
                    word = tok.text
                    if tok.pos in {SPACE, NUM, PUNCT, SYM}:
                        continue
                    if tok.is_punct or tok.is_space or tok.is_stop:
                        # assert False, (tok, tok.is_punct, tok.is_space, tok.is_stop)
                        continue
                    word_dist[word] += 1
                    word_docs[word].add(page_path)
                    wp += 1
                    ws += 1
                    cp += len(word)
                    cs += len(word)
                    cnt_word_char.append(len(word))
                cnt_sent_word.append(ws)
                cnt_sent_char.append(cs)
            cnt_para_sent.append(sp)
            cnt_para_word.append(wp)
            cnt_para_char.append(cp)

    cnt_metrics = {
        'cnt_page_para': cnt_page_para,
        'cnt_para_sent': cnt_para_sent,
        'cnt_para_word': cnt_para_word,
        'cnt_para_char': cnt_para_char,
        'cnt_sent_word': cnt_sent_word,
        'cnt_sent_char': cnt_sent_char,
        'cnt_word_char': cnt_word_char,
    }

    dist_metrics = {
        'para_dist': para_dist,
        'sent_dist': sent_dist,
        'word_dist': word_dist,
    }

    n_dup_thresh = max(2, n_processed // 50)
    doc_metrics = {
        'para_docs': {k: len(v) for k, v in para_docs.items()},
        'sent_docs': {k: len(v) for k, v in sent_docs.items()},
        'word_docs': {k: len(v) for k, v in word_docs.items()},
        'para_docs_vals': {k: sorted(v)[:20] for k, v in para_docs.items() if len(v) >= n_dup_thresh},
        'sent_docs_vals': {k: sorted(v)[:20] for k, v in sent_docs.items() if len(v) >= n_dup_thresh},
    }

    all_metrics = {
        'cnt_metrics': cnt_metrics,
        'dist_metrics': dist_metrics,
        'doc_metrics': doc_metrics,
        'n_files': len(files),
        'n_processed': n_processed,
    }

    return all_metrics


def show_dist(term_dist, title, max_dist, n_pages=-1):
    """
        Show cumulative if not a docs count (n_pages == -11)
    """
    total = sum(term_dist.values())
    print('-' * 80)
    print('%s most frequent. %d counts %d total' % (title, len(term_dist), total))
    t = 0
    for i, term in enumerate(sorted(term_dist, key=lambda w: (-term_dist[w], -len(w), w))[:max_dist]):
        n = term_dist[term]
        t += n
        if n_pages <= 0:
            print('%5d: %7d %4.1f%% (%4.1f%%) %r' % (i, n, n / total * 100.0, t / total * 100.0, term))
        else:
            print('%5d: %7d (%4.1f%% docs) %r' % (i, n, n / n_pages * 100.0, term))


def show_metrics(all_metrics, max_dist=50, do_plot=False):

    cnt_metrics = all_metrics['cnt_metrics']
    dist_metrics = all_metrics['dist_metrics']
    doc_metrics = all_metrics['doc_metrics']

    cnt_page_para = cnt_metrics['cnt_page_para']
    cnt_para_sent = cnt_metrics['cnt_para_sent']
    cnt_para_word = cnt_metrics['cnt_para_word']
    cnt_para_char = cnt_metrics['cnt_para_char']
    cnt_sent_word = cnt_metrics['cnt_sent_word']
    cnt_sent_char = cnt_metrics['cnt_sent_char']
    cnt_word_char = cnt_metrics['cnt_word_char']
    para_dist = dist_metrics['para_dist']
    sent_dist = dist_metrics['sent_dist']
    word_dist = dist_metrics['word_dist']
    para_docs = doc_metrics['para_docs']
    sent_docs = doc_metrics['sent_docs']
    word_docs = doc_metrics['word_docs']

    n_chars = sum(cnt_sent_char)
    n_words = sum(cnt_para_word)
    n_words2 = sum(cnt_sent_word)
    n_words3 = sum(word_dist.values())
    n_sents = sum(cnt_para_sent)
    n_paras = sum(cnt_page_para)
    n_pages = len(cnt_page_para)

    print('=' * 80)
    print('pages: %8d' % n_pages)
    print('paras: %8d' % n_paras)
    print('sents: %8d' % n_sents)
    print('words: %8d=%d=%d' % (n_words, n_words2, n_words3))
    print('chars: %8d' % n_chars)
    print('word_dist: %8d' % len(word_dist))
    print('cnt_page_para: %8d' % len(cnt_page_para))
    print('cnt_para_sent: %8d' % len(cnt_para_sent))
    print('cnt_para_word: %8d' % len(cnt_para_word))
    print('cnt_para_char: %8d' % len(cnt_para_char))
    print('cnt_sent_word: %8d' % len(cnt_sent_word))
    print('cnt_word_char: %8d' % len(cnt_word_char))

    if max_dist > 0:
        show_dist(para_dist, 'para_dist', max_dist)
        show_dist(sent_dist, 'sent_dist', max_dist)
        show_dist(word_dist, 'word_dist', max_dist)
        show_dist(para_docs, 'para_docs', max_dist, n_pages=n_pages)
        show_dist(sent_docs, 'sent_docs', max_dist, n_pages=n_pages)
        show_dist(word_docs, 'word_docs', max_dist, n_pages=n_pages)

    if do_plot:
        # plt.xscale('log')
        plot(cnt_page_para, 'Paragraphs per Page')
        plot(cnt_para_sent, 'Sentences per Paragraph')
        plot(cnt_sent_word, 'Words per Sentence')
        plot(cnt_word_char, 'Characters per Word')

        if False:
            plt.xscale('log')
            plt.yscale('log')
            plt.plot(sorted(word_dist.values())[::-1])
            plt.title('Word Frequencies (log : log)')
            plt.show()


if False:
    all_metrics = compute_metrics(max_processed=-1)
    save_json('all_metrics.json', all_metrics)
if False:
    all_metrics = load_json('all_metrics.json')
    show_metrics(all_metrics, max_dist=250)
