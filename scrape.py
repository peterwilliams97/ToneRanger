from bs4 import BeautifulSoup
from glob import glob
# from fnmatch import fnmatch
from os.path import expanduser, join, exists, splitext, abspath, isfile
from os import makedirs
from collections import defaultdict
# from pprint import pprint
import re
from utils import summaries_dir, read_file, save_json, path_to_name


# <?php /* End PaperCut Lifestyle */ ?>
RE_PHP = re.compile(r'<\?php.+?\s+\?>', re.DOTALL | re.MULTILINE)
# <iframe src="https://www.googletagmanager.com/ns.html?id=GTM-WP9VKT" height="0" width="0" style="display:none;visibility:hidden"></iframe>
RE_IFRAME = re.compile(r'<iframe.*?>*.?</iframe>', re.DOTALL | re.MULTILINE)


def html_to_paras(path):
    page = read_file(path)
    # page = RE_IFRAME.sub('<!-- ***IFRAME*** -->', page)
    page = RE_IFRAME.sub('', page)
    # # if '?>' not in page0:
    # #     print('bad*: %s -- %d' % (path, len(page0)))
    # #     return ''
    # page = RE_PHP.sub('<!-- ***PHP*** -->', page0)
    # if page == page0:
    #     print('bad: %s -- %d' % (path, len(page0)))
    soup = BeautifulSoup(page, 'html5lib')  # 'html.parser')
    for script in soup(["script", "style"]):
        script.extract()    # rip it out
    body = soup.find('body')
    if not body:
        return ''
    paras = [p.get_text().strip().replace('\u00a0', ' ') for p in body.find_all('p', recursive=True)]
    paras = [p for p in paras if p]
    return paras
    # print('###', len(paras), type(paras[0]))
    # print(paras[0])
    # assert False
    # return body.findChildren()
    # # return soup.get_text()


def save_summary(in_root, summaries_dir, php_path, visited):
    """Extract text from `php_path`, break it into pages and write the summary to 'summary_path
    """
    summary_name = path_to_name(in_root, php_path)
    summary_path = abspath(join(summaries_dir, '%s.json' % summary_name))
    # print('save_summary: %s->%s' % (php_path, summary_path))
    # text = html_to_text(php_path)

    # if '@var string' in text or '$_POST[' in text:
    #     print('bad php extraction: %s %s' % (php_path, text[:60]))
    #     return

    # # assert '>' not in text, php_path

    # paras = (line.strip() for line in text.splitlines())
    # paras = [p for p in paras if p]

    paras = html_to_paras(php_path)
    text = '\n'.join(paras)

    if not text:
        return

    hsh = hash(text)
    if hsh in visited:
        return
    visited.add(hsh)

    summary = {
        'path': php_path,
        'count:page_chars': len(text),
        'count:page_paras': len(paras),
        'count:para_chars': [len(p) for p in paras],
        'text:paras': paras,
        'text:page': text,
    }

    save_json(summary_path, summary)


def br():
    print('-' * 60)


def describe(path):
    text = html_to_text(path)
    # break into paras and remove leading and trailing space on each
    paras = (line.strip() for line in text.splitlines())
    # break multi-headlines into a line eac
    chunks = (phrase.strip() for line in paras for phrase in line.split("  "))
    # drop blank paras
    text = '\n'.join(chunk for chunk in chunks if chunk)

    br()
    print(text)
    print('=' * 60)


# root = expanduser('~/code/website')
root = '/Users/pcadmin/code/ToneRanger/paper_spider/pc_data'
assert exists(root), root
makedirs(summaries_dir, exist_ok=True)

if False:
    # describe('x.php')
    save_summary(root, summaries_dir, 'x.php')
    assert False


files = glob(join(root, '**'), recursive=True)
files = [path for path in files if isfile(path)]

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

# php_files = types['.php']
# php_files = ['/Users/pcadmin/code/ToneRanger/paper_spider/data/@_']

print('%d files' % len(files))
visited = set()
for i, path in enumerate(files):
    # if not isfile(path):
    #     continue
    # print('-' * 80)
    print('%4d: %s' % (i, path))
    save_summary(root, summaries_dir, path, visited)
    # # describe(path)
    False
print('%d files' % len(files))
print('%d unique' % len(visited))

