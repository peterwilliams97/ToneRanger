from bs4 import BeautifulSoup
from glob import glob
from os.path import join, exists, splitext, abspath, isfile, expanduser
from os import makedirs
from collections import defaultdict
import re
from utils import summaries_dir, read_file, save_json, path_to_name, clean_text


# <?php /* End PaperCut Lifestyle */ ?>
RE_PHP = re.compile(r'<\?php.+?\s+\?>', re.DOTALL | re.MULTILINE)
# <iframe src="https://www.googletagmanager.com/ns.html?id=GTM-WP9VKT" height="0" width="0" style="display:none;visibility:hidden"></iframe>
RE_IFRAME = re.compile(r'<iframe.*?>*.?</iframe>', re.DOTALL | re.MULTILINE)


def is_text_popup(tag):
    if not tag:
        return False
    if not tag.name:
        return False
    if tag.name not in {'div', 'span', 'a'}:
        return False
    cls = tag.get('class')
    if not cls:
        return False
    # print('@@@', tag)
    # print('!!!', tag.get('class'))
    return any('MCText' in t for t in tag.get('class'))
    # return tag and tag.name in {'div', 'span'} and 'MCText' in tag.get('class')


def html_to_paras(path):
    page0 = read_file(path)
    if not page0:
        return []
    page = RE_IFRAME.sub('', page0)

    soup = BeautifulSoup(page, 'html5lib')  # 'html.parser')
    for script in soup(["script", "style"]):
        script.extract()    # rip it out
    body = soup.find('body')
    if not body:
        return ''
    # if (div["class"]=="stylelistrow"):

    filtered = False
    for p in body.find_all('p', recursive=True):
        for div in p.find_all(is_text_popup):
            # "span", {'class': 'MCTextPopup'}):
            # print('$$$', div)
            div.decompose()
            filtered = True
    paras = [p.get_text() for p in body.find_all('p', recursive=True)]
    # for p in body.find_all('p', recursive=True):

    paras = [clean_text(p) for p in paras]
    paras = [p for p in paras if p]

    # if filtered:
    #     print('$' * 80)
    #     print(path)
    #     print(len(page0))
    #     print(page0[:200])
    #     print(p[:200] for p in paras[:2])
    #     assert False

    return paras


def save_summary(php_path, summary_path, visited):
    """Extract text from `php_path`, break it into pages and write the summary to 'summary_path
    """
    # summary_name = path_to_name(in_root, php_path)
    # summary_path = abspath(join(summaries_dir, '%s.json' % summary_name))

    paras = html_to_paras(php_path)
    text = '\n'.join(paras)
    if not text:
        return False

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
    }

    save_json(summary_path, summary)
    return True


root = 'c:/code/ToneRanger/paper_spider/pc_data'

root = 'c:/code/ToneRanger/paper_spider'
root = '~/code/ToneRanger/paper_spider'
root = expanduser(root)
root = join(root, 'pc_data')
print('root=%s' % root)
assert exists(root), root
makedirs(summaries_dir, exist_ok=True)

files = glob(join(root, '**'), recursive=True)
files = [path for path in files if isfile(path)]
print('%d files' % len(files))

visited = set()
summary_raw = {}
for i, path in enumerate(files):
    path = abspath(path)
    print('%4d: %s' % (i, path))
    summary_name = path_to_name(summaries_dir, path)
    summary_path = abspath(join(summaries_dir, '%s.json' % summary_name))
    if not save_summary(path, summary_path, visited):
        continue
    summary_raw[summary_path] = path
    if i % 100 == 99:
        save_json('summary_raw.json', summary_raw)
    assert exists(path), path
    assert exists(summary_path), summary_path

save_json('summary_raw.json', summary_raw)
print('%d files' % len(files))
print('%d unique' % len(visited))
