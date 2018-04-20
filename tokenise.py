"""
    Analyze a web page
"""
from glob import glob
from os.path import join, basename
import spacy
from utils import summaries_dir, load_json, save_json
import time


def summary_key(a):
    path, o = a
    return -o['count:page_paras'], -o['count:page_chars'], path


nlp = spacy.load('en')

files = glob(join(summaries_dir, '*.json'))
print('%d files' % len(files))

path_summaries = [(path, load_json(path)) for path in files]
path_summaries.sort(key=summary_key)

n = 0
for i, (path, o) in enumerate(path_summaries):
    print('%3d: %4d %6d %s' % (i,
         o['count:page_paras'], o['count:page_chars'], basename(path)), end=' ')
    t0 = time.clock()
    o['text:sents'] = []
    o['count:para_sents'] = []
    o['count:sent_words'] = []
    o['count:sent_chars'] = []
    if 'text:paras' not in o:
        print('!!! bad file')
        continue
    for para in o['text:paras']:
        doc = nlp(para)
        sents = [str(s) for s in doc.sents]
        sent_chars = [len(s) for s in doc.sents]
        o['text:sents'].append(sents)
        o['count:sent_words'].append([len(s) for s in sents])
        o['count:para_sents'].append(len(sents))
        o['count:sent_chars'].append(sent_chars)

    # del o['text:page']  # !@# Don't add in spider
    # del o['text:paras']

    save_json(path, o)
    dt = time.clock() - t0
    print('%4.1f sec' % dt)
