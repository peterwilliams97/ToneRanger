"""
    Extract all sentences from all paragraphs from all pages
"""
from glob import glob
from os.path import join
from collections import defaultdict
from utils import summaries_dir, load_json, save_json, load_jsonl, save_jsonl


threshold_words = 10


def extract_sentences(max_processed=-1):

    files = glob(join(summaries_dir, '*.json'))
    print('%4d files' % len(files))

    para_count = defaultdict(int)
    sent_count = defaultdict(int)
    for path in files:
        summary = load_json(path)
        for para in summary['text:paras']:
            para_count[para] += 1
        for para2 in summary['text:sents']:
            for sent in para2:
                sent_count[sent] += 1

    print('%d paragraphs %d unique' % (sum(para_count.values()), len(para_count)))
    print('%d sentences %d unique' % (sum(sent_count.values()), len(sent_count)))

    def sent_key(sent):
        return -len(sent), sent_count[sent], sent

    paras = sorted(para_count, key=sent_key)
    sents = sorted(sent_count, key=sent_key)

    paras = [{'text': text} for text in paras]
    sents = [{'text': text} for text in sents]

    # {"text":"Uber\u2019s Lesson: Silicon Valley\u2019s Start-Up Machine Needs Fixing","meta":{"source":"The New York Times"}}

    save_jsonl('pc.paragraphs.jsonl', paras)
    save_jsonl('pc.sentences.jsonl', sents)


extract_sentences()
