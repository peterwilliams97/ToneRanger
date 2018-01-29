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
    uniques = {hash(o['text:page']): o for o in summaries}
    print('%4d uniques' % len(uniques))
    summaries = sorted(uniques.values(), key=summary_key)

    para_count = defaultdict(int)
    sent_count = defaultdict(int)
    word_count = defaultdict(int)
    cnt_page_para = []
    cnt_para_sent = []
    cnt_para_word = []
    cnt_para_char = []
    cnt_sent_word = []
    cnt_sent_char = []
    cnt_word_char = []
    n_processed = 0
    for summary in summaries:
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
            para_count[para] += 1
            doc = nlp(para)

            sp = 0
            wp = 0
            cp = 0
            for sent in doc.sents:
                sent_count[sent.text] += 1
                sp += 1
                ws = 0
                cs = 0
                for tok in sent:
                    word = tok.text
                    if tok.pos in {SPACE, NUM, PUNCT, SYM}:
                        continue
                    word_count[word] += 1
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

    all_metrics = {
        'cnt_page_para': cnt_page_para,
        'cnt_para_sent': cnt_para_sent,
        'cnt_para_word': cnt_para_word,
        'cnt_para_char': cnt_para_char,
        'cnt_sent_word': cnt_sent_word,
        'cnt_sent_char': cnt_sent_char,
        'cnt_word_char': cnt_word_char,
        'para_count': para_count,
        'sent_count': sent_count,
        'word_count': word_count,
        'n_files': len(files),
        'n_uniques': len(uniques),
    }
    return all_metrics


def show_count(term_count, title, max_count):
    total = sum(term_count.values())
    print('-' * 80)
    print('%s most frequent. %d counts %d total' % (title, len(term_count), total))
    t = 0
    for i, term in enumerate(sorted(term_count, key=lambda w: (-term_count[w], -len(w), w))[:max_count]):
        n = term_count[term]
        t += n
        print('%5d: %7d %4.1f%% (%4.1f%%) %r' % (i, n, n / total * 100.0, t / total * 100.0, term))


def show_metrics(all_metrics, max_count=50, do_plot=False):

    cnt_page_para = all_metrics['cnt_page_para']
    cnt_para_sent = all_metrics['cnt_para_sent']
    cnt_para_word = all_metrics['cnt_para_word']
    cnt_para_char = all_metrics['cnt_para_char']
    cnt_sent_word = all_metrics['cnt_sent_word']
    cnt_sent_char = all_metrics['cnt_sent_char']
    cnt_word_char = all_metrics['cnt_word_char']
    para_count = all_metrics['para_count']
    sent_count = all_metrics['sent_count']
    word_count = all_metrics['word_count']

    n_chars = sum(cnt_sent_char)
    n_words = sum(cnt_para_word)
    n_words2 = sum(cnt_sent_word)
    n_words3 = sum(word_count.values())
    n_sents = sum(cnt_para_sent)
    n_paras = sum(cnt_page_para)
    n_pages = len(cnt_page_para)
    print('=' * 80)
    print('pages: %8d' % n_pages)
    print('paras: %8d' % n_paras)
    print('sents: %8d' % n_sents)
    print('words: %8d=%d=%d' % (n_words, n_words2, n_words3))
    print('chars: %8d' % n_chars)
    print('word_count: %8d' % len(word_count))
    print('cnt_page_para: %8d' % len(cnt_page_para))
    print('cnt_para_sent: %8d' % len(cnt_para_sent))
    print('cnt_para_word: %8d' % len(cnt_para_word))
    print('cnt_para_char: %8d' % len(cnt_para_char))
    print('cnt_sent_word: %8d' % len(cnt_sent_word))
    print('cnt_word_char: %8d' % len(cnt_word_char))

    if max_count > 0:
        show_count(para_count, 'para_count', max_count)
        show_count(sent_count, 'send_count', max_count)
        show_count(word_count, 'word_count', max_count)

    if False:
        print('-' * 80)
        for i, word in enumerate(sorted(word_count, key=lambda w: (-word_count[w], w))[:max_count]):
            n = word_count[word]
            print('%5d: %7d %4.1f%% %r' % (i, n, n / n_words * 100.0, word))
        print('-' * 80)
        for i, sent in enumerate(sorted(sent_count, key=lambda w: (-sent_count[w], w))[:max_count]):
            n = sent_count[sent]
            print('%5d: %7d %4.1f%% %r' % (i, n, n / n_sents * 100.0, sent))
        print('-' * 80)
        for i, para in enumerate(sorted(para_count, key=lambda w: (-para_count[w], w))[:max_count]):
            n = par_count[sent]
            print('%5d: %7d %4.1f%% %r' % (i, n, n / n_sents * 100.0, sent))

    if do_plot:
        # plt.xscale('log')
        plot(cnt_page_para, 'Paragraphs per Page')
        plot(cnt_para_sent, 'Sentences per Paragraph')
        plot(cnt_sent_word, 'Words per Sentence')
        plot(cnt_word_char, 'Characters per Word')

        if False:
            plt.xscale('log')
            plt.yscale('log')
            plt.plot(sorted(word_count.values())[::-1])
            plt.title('Word Frequencies (log : log)')
            plt.show()


if True:
    all_metrics = compute_metrics(max_processed=-1)
    save_json('all_metrics.json', all_metrics)
if True:
    all_metrics = load_json('all_metrics.json')
    show_metrics(all_metrics, max_count=250)
