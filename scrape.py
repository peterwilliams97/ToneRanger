from bs4 import BeautifulSoup
from glob import glob
# from fnmatch import fnmatch
from os.path import expanduser, join, exists, splitext, abspath
from os import makedirs
from collections import defaultdict
# from pprint import pprint
import re
from utils import summaries_dir, read_file, save_json, path_to_name


# <?php /* End PaperCut Lifestyle */ ?>
RE_PHP = re.compile(r'<\?php.+?\?>', re.DOTALL | re.MULTILINE)


def html_to_text(path):
    page0 = read_file(path)
    page = RE_PHP.sub('<!-- ***PHP*** -->', page0)
    if page == page0:
        print('bad: %s -- %d' % (path, len(page0)))
    soup = BeautifulSoup(page, 'html5lib')  # 'html.parser')
    for script in soup(["script", "style"]):
        script.extract()    # rip it out
    return soup.get_text()


def save_summary(in_root, summaries_dir, php_path):
    """Extract text from `php_path`, break it into pages and write the summary to 'summary_path
    """
    summary_name = path_to_name(in_root, php_path)
    summary_path = abspath(join(summaries_dir, '%s.json' % summary_name))
    # print('save_summary: %s->%s' % (php_path, summary_path))
    text = html_to_text(php_path)

    lines = lines = (line.strip() for line in text.splitlines())
    lines = [ln for ln in lines if ln]
    text = '\n'.join(lines)

    if not text:
        return

    summary = {
        'path': php_path,
        'n_chars': len(text),
        'text': text,
        'n_paras': len(lines),
        'para_lens': [len(ln) for ln in lines],
        'paras': lines,
    }

    save_json(summary_path, summary)


def br():
    print('-' * 60)



def describe(path):
    text = html_to_text(path)
    # break into lines and remove leading and trailing space on each
    lines = (line.strip() for line in text.splitlines())
    # break multi-headlines into a line eac
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    # drop blank lines
    text = '\n'.join(chunk for chunk in chunks if chunk)

    br()
    print(text)
    if False:
        br()
        print(soup.p.contents)
        br()
        print(soup.prettify()[0:300])
        br()
        print(soup.div.contents)
        br()
        divs = soup.find_all('div')
        print(divs)
        br()
        print(sorted(divs[0].__dict__))
        for i, d in enumerate(divs):
            print('%3d: %s' % (i, d.contents))
    print('=' * 60)


root = expanduser('~/code/website')
assert exists(root), root
makedirs(summaries_dir, exist_ok=True)

if False:
    # describe('x.php')
    save_summary(root, summaries_dir, 'x.php')
    assert False


files = glob(join(root, '**'), recursive=True)
types = defaultdict(list)
for path in files:
    types[splitext(path)[1]].append(path)
# files = [path for path in files if fnmatch(path, '*.html')]
print('%d %s' % (len(files), files[:5]))
# pprint(types)
types = {k: sorted(v) for k, v in types.items()}
if True:
    for i, t in enumerate(sorted(types, key=lambda x: (-len(types[x]), x))):
        lst = types[t]
        print('%3d: %-6s %4d %s' % (i, t, len(lst), lst[0]))
        if len(lst) < 10:
            break

php_files = types['.php']

for php_path in php_files:
    # print('-' * 80)
    # print(php_path)
    save_summary(root, summaries_dir, php_path)
    # describe(path)
    # assert False
