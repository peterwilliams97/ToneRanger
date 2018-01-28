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
