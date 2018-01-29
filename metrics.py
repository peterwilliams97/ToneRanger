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
from utils import summaries_dir, load_json, save_json
from collections import defaultdict

threshold_words = 10


def summary_key(o):
    return o['count:page_paras'], o['count:page_chars'], o['path']


def compute_metrics():
    nlp = spacy.load('en')

    files = glob(join(summaries_dir, '*.json'))
    print('%4d files' % len(files))
    summaries = [load_json(path) for path in files]
    uniques = {hash(o['text:page']): o for o in summaries}
    print('%4d uniques' % len(uniques))
    summaries = sorted(uniques.values(), key=summary_key)

    vocab = defaultdict(int)
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
        # if n_processed > 5:
        #     break

        print('%3d: %s' % (n_processed, summary['path']))
        cnt_page_para.append(len(summary['text:paras']))

        for para in summary['text:paras']:
            doc = nlp(para)
            sp = 0
            wp = 0
            cp = 0
            for sent in doc.sents:
                sp += 1
                ws = 0
                cs = 0
                for tok in sent:
                    word = tok.text
                    vocab[word] += 1
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
        'vocab': vocab
    }
    return all_metrics


def show_metrics(all_metrics, max_vocab=50):

    cnt_page_para = all_metrics['cnt_page_para']
    cnt_para_sent = all_metrics['cnt_para_sent']
    cnt_para_word = all_metrics['cnt_para_word']
    cnt_para_char = all_metrics['cnt_para_char']
    cnt_sent_word = all_metrics['cnt_sent_word']
    cnt_sent_char = all_metrics['cnt_sent_char']
    cnt_word_char = all_metrics['cnt_word_char']
    vocab = all_metrics['vocab']

    n_chars = sum(cnt_sent_char)
    n_words = sum(cnt_para_word)
    n_words2 = sum(cnt_sent_word)
    n_words3 = sum(vocab.values())
    n_sents = sum(cnt_para_sent)
    n_paras = sum(cnt_page_para)
    n_pages = len(cnt_page_para)
    print('=' * 80)
    print('pages: %8d' % n_pages)
    print('paras: %8d' % n_paras)
    print('sents: %8d' % n_sents)
    print('words: %8d=%d=%d' % (n_words, n_words2, n_words3))
    print('chars: %8d' % n_chars)
    print('vocab: %8d' % len(vocab))
    print('cnt_page_para: %8d' % len(cnt_page_para))
    print('cnt_para_sent: %8d' % len(cnt_para_sent))
    print('cnt_para_word: %8d' % len(cnt_para_word))
    print('cnt_para_char: %8d' % len(cnt_para_char))
    print('cnt_sent_word: %8d' % len(cnt_sent_word))
    print('cnt_word_char: %8d' % len(cnt_word_char))
    print('-' * 80)

    for i, word in enumerate(sorted(vocab, key=lambda w: (-vocab[w], w))[:max_vocab]):
        n = vocab[word]
        print('%5d: %7d %4.1f%% %r' % (i, n, n / n_words * 100.0, word))


if False:
    all_metrics = compute_metrics()
    save_json('all_metrics.json', all_metrics)
if True:
    all_metrics = load_json('all_metrics.json')
    show_metrics(all_metrics)

