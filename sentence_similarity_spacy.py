# coding: utf-8
"""
    Use word2vec to explore words that appear in similar contexts
"""
import logging
import time
import queue
import os
import pickle
import numpy as np
import spacy
from spacy.symbols import SPACE, PUNCT, NUM, SYM
import utils

np.set_printoptions(edgeitems=10, precision=4)
np.core.arrayprint._line_width = 180


do_vector = True
do_lowercase = False
force_parse = True
max_sentences = 200

# Seed for the RNG
seed = 114

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


if do_vector:
    train_dir = "tone.spacy.sentences.vector"
else:
    train_dir = "tone.spacy.sentences"
sentence_text_path = 'all.sentences.json'
sentence_path = os.path.join(train_dir, 'sentences.pkl')
log_path = os.path.join(train_dir, 'xx.log')

os.makedirs(train_dir, exist_ok=True)
log_f = open(log_path, 'wt')


def my_print(*args):
    print(args)
    print(args, file=log_f)


class FiniteQueue:
    """ Priority queue that stores the `max_length` items with highest scores
    """

    def __init__(self, max_length):
        self.q = queue.PriorityQueue(max_length=0)
        self.max_length = max_length
        self.n = 0

    def put(self, score, item):
        self.q.put((score, item))
        self.n += 1
        while self.n > self.max_length:
            self.q.get()
            self.n -= 1

    def get(self):
        score_items = []
        while not self.q.empty():
            score, item = self.q.get()
            score_items.append((score, item))
            self.n -= 1
        score_items.reverse()
        return score_items


if False:
    q = FiniteQueue(3)
    for n in range(20):
        q.put(n, n)
    print(q.get())


if do_vector:
    nlp = spacy.load('en_core_web_lg')
else:
    nlp = spacy.load('en')

if False:
    sent1 = nlp('This is some text')
    sent2 = nlp('To be, or not to be')
    for token in (sent1, sent2):
        print('%30s -- %5s %.3f' % (token.text, token.has_vector, token.vector_norm))
        print(len(token._vector), token._vector[:5], token._vector_norm)
    doc = token
    for token in doc:
         print('%30s -- %5s %.3f %s' % (token.text, token.has_vector, token.vector_norm, token.is_oov))

    sim = sent1.similarity(sent2)
    print('sim=%s' % sim)
    assert False


if force_parse or not os.path.exists(sentence_path):
    sentence_texts = utils.load_json(sentence_text_path)
    sentence_texts = [text for text in sentence_texts if len(text) >= 20]
    sentence_texts.reverse()
    if max_sentences > 0:
        sentence_texts = sentence_texts[:max_sentences]
    my_print('%8d sentences' % len(sentence_texts))

    interval = max(len(sentence_texts) // 50, 20)
    t0 = time.clock()
    sentences = {}
    for i, sent in enumerate(sentence_texts):
        sentences[sent] = nlp(sent)
        if i % interval == 0:
            print('%8d %4.1f%% %5.1f sec' % (i,
                i / len(sentence_texts) * 100.0, time.clock() - t0))
    print('sentence parsing: %.1f sec' % (time.clock() - t0))

    with open(sentence_path, 'wb') as f:
        pickle.dump(sentences, f)

# with open(sentence_path, 'rb') as f:
#     sentences = pickle.load(f)

import inspect


def find_closest(text0, max_length):
    q = FiniteQueue(max_length)
    sent0 = sentences[text0]
    print(type(sent0))
    # members = inspect.getmembers(sent0)
    # for i, m in enumerate(members):
    #     print('%3d: %s' % (i, m))
    # print('text0=%s' % text0[:50])
    # print('vector0=%s' % sent0.vector[:3])
    assert sent0, text0
    for text, sent in sentences.items():
        assert sent, text
        # print('vector=%s' % sent.vector[:3])
        try:
            score = sent0.similarity(sent)
        except AttributeError:
            print('text=%s' % text)
            raise

        q.put(score, text)
    return q.get()


print('Find closest')
# sent0 = "Traveling to Fort Lauderdale in Florida took a long time, about 8 hours total travel, with 2 planes and a taxi ride (although I'm sure the Australian contingent of Chris and Damien won't feel too sorry for me, 26 hours from Melbourne!)."
sent0 = sorted(sentences, key=lambda x: (len(x), x))[0]
sent1 = sorted(sentences, key=lambda x: (len(x), x))[-1]

closest0 = find_closest(sent0, max_length=10)
print('sent0= %d "%s"' % (len(sent0), sent0))
for i, (score, sent) in enumerate(closest0):
    my_print('%3d: %.3f %3d "%s"' % (i, score, len(sent), sent[:50]))

closest1 = find_closest(sent1, max_length=20)
print('sent1= %d "%s"' % (len(sent1), sent1))
for i, (score, sent) in enumerate(closest1):
    my_print('%3d: %.3f %3d "%s"' % (i, score, len(sent), sent[:50]))


closest1 = [text for text in closest1 if text not in closest0]
closest = closest0[:5] + closest1[:5]
print('-' * 80)
for i, (_, text) in enumerate(closest):
    print('%3d: "%s" %s' % (i, text, sentences[text].vector[:3]))

similarity = np.zeros((len(closest), len(closest)))
for i1, (_, text1) in enumerate(closest):
    sent1 = sentences[text1]
    for i2, (_, text2) in enumerate(closest[:i1 + 1]):
        sent2 = sentences[text2]
        score = sent1.similarity(sent2)
        similarity[i1, i2] = score
my_print(similarity)
