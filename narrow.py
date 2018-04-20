import spacy
from utils import load_jsonl
import numpy as np
import annoy
# import faiss
from annoy import AnnoyIndex
import random

if False:
    f = 7
    t = AnnoyIndex(f)  # Length of item vector that will be indexed
    for i in range(1000):
        v = [random.gauss(0, 1) for z in range(f)]
        t.add_item(i, v)

    t.build(10)  # 10 trees
    t.save('test.ann')

    # ...

    u = AnnoyIndex(f)
    u.load('test.ann')  # super fast, will just mmap the file
    print(u.get_nns_by_item(0, 1000))  # will find the 1000 nearest neighbors

    assert False


nlp = spacy.load('predict/my_model')

texts = [
'Today is sunny',
'I hate bunnies',
'''Chris Goult recently joined us as our shiny new channel marketing manager.  Chris comes to us from Konica Minolta Business Solutions Australia, with loads of knowledge and enthusiasm for all things marketing and channel related. ''',
'''You don't want your precious funds ending up as wasted paper in recycle bins.''',
'''At the core of PaperCut MF is the ability to interface directly with MFD hardware to track off-the-glass functions such as copy, scan, fax and secure print release. PaperCut has worked directly with leading MFD manufacturers to bring our software directly to the MFD at a firmware level. To complete the solution offering across all devices PaperCut MF also supports hardware copier terminals from multiple vendors. ''',
]
for text in texts:
    print('-' * 80)
    print(text)
    doc = nlp(text)
    print(doc.cats)

print('=' * 80)
paras = load_jsonl('pc.paragraphs.jsonl')
print(len(paras))
print(paras[-1])
results = []
for i, p in enumerate(paras):
    text = p['text']
    url = p['meta']['url']
    doc = nlp(text)
    score = doc.cats
    # print(sorted(score))
    # assert False

    results.append((text, url, score))
    if i % 1000 == 50:
        # text = text.decode('utf-8', errors='ignore')
        text = text.encode('cp850', 'replace').decode('cp850')
        print('%4d: %-80s %s' % (i, text[:80], score))

f = 5
ASPECTS = ['BRIGHT', 'CARING', 'CONVERSATIONAL', 'EMOTIVE', 'PROFESSIONAL']
N = len(results)

for aspect in ASPECTS:
    print('~' * 80)
    print(aspect)
    results.sort(key=lambda x: x[2][aspect])
    for i, (text, url, score) in enumerate(results[:3]):
        print('%d: %.4f %s' % (i, score[aspect], url))
    print('...')
    for n in N // 4, N // 2, 3 * N // 4:
        for i, (text, url, score) in enumerate(results[n - 1: n + 2]):
            print('%d: %.4f %s' % (i, score[aspect], url))
        print('...')
    for i, (text, url, score) in enumerate(results[-3:]):
        print('%d: %.4f %s' % (i, score[aspect], url))

t = AnnoyIndex(f)  # Length of item vector that will be indexed
for i, (text, url, score) in enumerate(results):
    v = [score[k] for k in ASPECTS]
    t.add_item(i, v)

t.build(10)  # 10 trees
t.save('test.ann')

# ...

u = AnnoyIndex(f)
u.load('test.ann')  # super fast, will just mmap the file
for i in range(10):
    text, url, score = results[i]
    print('^' * 80)
    print('Docs near %d: %s' % (i, url))
    nn = u.get_nns_by_item(i, 4)
    for j, idx in enumerate(nn[1:]):
        text2, url2, score2 = results[idx]
        print('%2d: %4d: %s' % (j, idx, url2))

# print(u.get_nns_by_item(0, 10))  # will find the 100 nearest neighbors

