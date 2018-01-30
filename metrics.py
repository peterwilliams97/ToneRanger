"""
    Compute some metrics

    Sentences / paragraph
    Words / paragrah
    Characters / paragraph
    Words / sentence
    Characters / sentence
    Characters / word
    Word counts
"""
from glob import glob
from os.path import join
import spacy
from spacy.symbols import SPACE, PUNCT, NUM, SYM
from utils import summaries_dir, load_json, save_json
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

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


if True:
    all_metrics = compute_metrics(max_processed=-1)
    save_json('all_metrics.json', all_metrics)
if True:
    all_metrics = load_json('all_metrics.json')
    show_metrics(all_metrics, max_dist=250)
