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
from os.path import join, basename
from os import makedirs, rename
from collections import defaultdict
import spacy
from utils import summaries_dir, load_json
from collections import defaultdict
import shutil


def summary_key(o):
    return -o['count:page_paras'], -o['count:page_chars'], o['path']


def deduplicate(verbose=False):
    nlp = spacy.load('en')

    files = glob(join(summaries_dir, '*.json'))
    print('%4d files' % len(files))
    unique_lists = defaultdict(list)
    for path in files:
        o = load_json(path)
        unique_lists[hash(o['text:page'])].append(path)
    print('%4d uniques' % len(unique_lists))

    def uniques_key(k):
        v = unique_lists[k]
        return -len(v), v[0]

    if verbose:
        for i, k in enumerate(sorted(unique_lists, key=uniques_key)):
            v = unique_lists[k]
            print('%4d: %3d' % (i, len(v)))
            for j, path in enumerate(sorted(v, key=lambda p: (len(p), p))):
                print('%10d: %s' % (j, path))

    # unique_lists = {k: v for k, v in unique_lists.items() if len(v) > 1}  # !@#$
    if len(unique_lists) < len(files):
        uniques_dir = '%s.uniques' % summaries_dir
        backup_dir = '%s.backup' % summaries_dir
        makedirs(uniques_dir)
        for v in unique_lists.values():
            path = max(v, key=lambda p: (len(p), p))
            name = basename(path)
            unique_path = join(uniques_dir, name)
            # print('%s -> %s' % (path, unique_path))
            shutil.copy(path, unique_path)
        rename(summaries_dir, backup_dir)
        rename(uniques_dir, summaries_dir)

deduplicate()
