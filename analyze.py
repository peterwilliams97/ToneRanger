"""
    Analyze a web page
"""
from glob import glob
from os.path import join, abspath
import spacy
from utils import summaries_dir, load_json, save_json


def summary_key(a):
    path, o = a
    return -o['page_chars'], -o['page_paras'], path


nlp = spacy.load('en')

files = glob(join(summaries_dir, '*.json'))
print('%d files' % len(files))
# path_summary = {path: i for i, path in enumerate(files)}
# summary_path = {i: path for path, i in path_summary}
path_summaries = [(path, load_json(path)) for path in files]
path_summaries.sort(key=summary_key)
n = 0
for i, (path, o) in enumerate(path_summaries[:25]):
    print('%3d: %6d %4d %s' % (i, o['page_chars'], o['page_paras'], o['path']))
    o['sents'] = []
    o['para_sents'] = []
    o['sent_words'] = []
    o['sent_chars'] = []
    for para in o['paras']:
        # print('-' * 80)
        doc = nlp(para)
        sents = [str(s) for s in doc.sents]
        sent_chars = [len(s) for s in doc.sents]
        o['sents'].append(sents)
        o['sent_words'].append([len(s) for s in sents])
        o['para_sents'].append(len(sents))
        o['sent_chars'].append(sent_chars)
    print(sorted(o))
    for k in sorted(o):
        v = o[k]
        t = type(v)
        t0 = type(v[0]) if isinstance(v, list) else None
        print('%10s: %s %s' % (k, t, t0))
    save_json(path, o)
    assert False, abspath(path)
        # para2 = ' '.join(sents)
        # print(para)
        # print(para2)
        # assert para == para2
        # for j, sent in enumerate(doc.sents):
        #     print
        #     print('%3d: %s' % (j, sent))
        # n += 1
        # assert n < 20
    break

