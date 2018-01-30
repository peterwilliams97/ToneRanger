import json
from os.path import relpath
from os import renames
import re

summaries_dir = 'page.summaries.scrapy'


def read_file(path):
    try:
        with open(path, 'r') as f:
            return f.read()
    except UnicodeDecodeError:
        print('read_file failed: path=%s' % path)
        return ''


def load_json(path):
    try:
        with open(path, 'r') as f:
            obj = json.load(f)
    except json.decoder.JSONDecodeError:
        print('load_json failed: path=%r' % path)
        raise
    return obj


temp_name = 'temp.json'


def save_json(path, obj):
    with open(temp_name, 'w') as f:
        json.dump(obj, f, indent=4, sort_keys=True)
    # remove(path)
    renames(temp_name, path)


RE_SEP = re.compile(r'[\\/\s\.:]+')


def path_to_name(root, path):
    rel = relpath(path, root)
    return RE_SEP.sub('#', rel)


RE_SPACE = re.compile(r'[ \t]+')
RE_NL = re.compile('\n+')
REPLACEMENTS = [
    ('\u00a0', ' '),  # npbs --> ' '
    ('\u2009', ' '),  #      --> ' '
    ('\u2018', "'"),  #    ‘ --> '
    ('\u2019', "'"),  #    ’ --> '
    ('\u201c', '"'),  #    “ --> "
    ('\u201d', '"'),  #    ” --> "
    ('\u2026', '...'),  #  … --> ...
    ('\uff01', '! '),  #  ！ --> !
    ('\uff08', ' ('),  #  （ --> (!)
    ('\uff09', ') '),  #  ） --> )
    ('\uff0c', ', '),  #  ， --> ,
    ('\uff1a', ': '),  #  ： --> :
]


def clean_text(text):
    text = text.strip()
    for old, new in REPLACEMENTS:
        text = text.replace(old, new)
    text = RE_SPACE.sub(' ', text)
    text = RE_NL.sub('\n', text)
    return text


if False:
    from pprint import pprint
    pprint(REPLACEMENTS)
    text = '''

             x


      y
     the \u2018Configure PaperCut\u2018 button
    '''
    # text2 = clean_text(text)
    # print('>%s<-' % text)
    # print('>%s<+' % text2)
    json_kv = '''{"k": "\u2018 Older posts"}'''
    kv = json.loads(json_kv)
    v = kv['k']
    v2 = clean_text(v)
    print('>%s<1' % v)
    print('>%s<2' % v2)
    assert False
