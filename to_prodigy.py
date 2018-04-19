"""
    Extract all sentences from all paragraphs from all pages
"""
from glob import glob
from os.path import join, abspath, exists
from collections import defaultdict
from utils import summaries_dir, load_json, save_json, load_jsonl, save_jsonl


threshold_words = 10
root = 'c:/code/ToneRanger/paper_spider'


def fix(s):
    return s.replace('/', '\\').lower()


def extract_sentences(max_processed=-1):

    path_raw = load_json('summary_raw.json')
    path_raw = {fix(k): fix(v) for k, v in path_raw.items()}
    file_url_path = join(root, 'file_url.json')
    raw_url = load_json(file_url_path)
    raw_url = {fix(k): v for k, v in raw_url.items()}

    print('+' * 80)
    print('path_raw', len(path_raw))
    for i, k in enumerate(sorted(path_raw)[:5]):
        print('%d: %s %s %s %s' % (i, k, exists(k), path_raw[k], exists(path_raw[k])))
    print('#' * 80)
    print('raw_url', len(raw_url))
    for i, k in enumerate(sorted(raw_url)[:5]):
        print('%d: %s %s' % (i, k, exists(k)))

    path_url = {path: raw_url[raw] for path, raw in path_raw.items() if raw in raw_url}

    files = glob(join(summaries_dir, '*.json'))
    print('%4d files' % len(files))
    print('%4d path_url' % len(path_url))

    para_count = defaultdict(int)
    sent_count = defaultdict(int)
    para_url = {}
    sent_url = {}
    for path in files:
        path = fix(abspath(path))
        summary = load_json(path)

        for para in summary['text:paras']:
            para_count[para] += 1
            if len(para) < 30:
                continue
            if para not in para_url:
                assert path in path_url, path
                para_url[para] = path_url.get(path, "UNKNOWN")
        # for para2 in summary['text:sents']:
        #     for sent in para2:
        #         sent_count[sent] += 1
        #         # if sent not in sent_url:
        #         #     sent_url[sent] = path_url[name]

    print('%d paragraphs %d unique' % (sum(para_count.values()), len(para_count)))
    # print('%d sentences %d unique' % (sum(sent_count.values()), len(sent_count)))

    def sent_key(sent):
        return -len(sent), sent_count[sent], sent

    paras = sorted(para_count, key=sent_key)
    # sents = sorted(sent_count, key=sent_key)

    paras = [{'text': text, 'meta': {'url': para_url[text]}} for text in paras if text in para_url]
    # sents = [{'text': text, 'meta': {'url': sent_url[text]}} for text in sents]

    # paras = [{'text': text} for text in paras]
    # sents = [{'text': text} for text in sents]

    # "meta":{"source":"GitHub","url":"https://github.com/rdbc-io/rdbc/issues/86"}}
    # {"text":"Uber\u2019s Lesson: Silicon Valley\u2019s Start-Up Machine Needs Fixing","meta":{"source":"The New York Times"}}

    save_jsonl('pc.paragraphs.jsonl', paras)
    # save_jsonl('pc.sentences.jsonl', sents)


extract_sentences()
