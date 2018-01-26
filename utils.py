import json
from os.path import relpath
import re

summaries_dir = 'data'


def load_json(path):
    with open(path, 'r') as f:
        obj = json.load(f)
    return obj


def read_file(path):
    try:
        with open(path, 'r') as f:
            return f.read()
    except UnicodeDecodeError:
        print('read_file failed: path=%s' % path)
        return ''


def save_json(path, obj):
    with open(path, 'w') as f:
        json.dump(obj, f, indent=4, sort_keys=True)


RE_SEP = re.compile(r'[\\/s\.]+')


def path_to_name(root, path):
    rel = relpath(path, root)
    return RE_SEP.sub('#', rel)
