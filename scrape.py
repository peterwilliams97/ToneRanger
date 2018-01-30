from bs4 import BeautifulSoup
from glob import glob
from os.path import join, exists, splitext, abspath, isfile
from os import makedirs
from collections import defaultdict
import re
from utils import summaries_dir, read_file, save_json, path_to_name, clean_text


# <?php /* End PaperCut Lifestyle */ ?>
RE_PHP = re.compile(r'<\?php.+?\s+\?>', re.DOTALL | re.MULTILINE)
# <iframe src="https://www.googletagmanager.com/ns.html?id=GTM-WP9VKT" height="0" width="0" style="display:none;visibility:hidden"></iframe>
RE_IFRAME = re.compile(r'<iframe.*?>*.?</iframe>', re.DOTALL | re.MULTILINE)


def html_to_paras(path):
    page = read_file(path)
    page = RE_IFRAME.sub('', page)

    soup = BeautifulSoup(page, 'html5lib')  # 'html.parser')
    for script in soup(["script", "style"]):
        script.extract()    # rip it out
    body = soup.find('body')
    if not body:
        return ''
    paras = [p.get_text() for p in body.find_all('p', recursive=True)]
    paras = [clean_text(p) for p in paras]
    paras = [p for p in paras if p]

    return paras


def save_summary(in_root, summaries_dir, php_path, visited):
    """Extract text from `php_path`, break it into pages and write the summary to 'summary_path
    """
    summary_name = path_to_name(in_root, php_path)
    summary_path = abspath(join(summaries_dir, '%s.json' % summary_name))

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
    }

    save_json(summary_path, summary)


root = '/Users/pcadmin/code/ToneRanger/paper_spider/pc_data'
assert exists(root), root
makedirs(summaries_dir, exist_ok=True)

files = glob(join(root, '**'), recursive=True)
files = [path for path in files if isfile(path)]
print('%d files' % len(files))

visited = set()
for i, path in enumerate(files):
    print('%4d: %s' % (i, path))
    save_summary(root, summaries_dir, path, visited)

print('%d files' % len(files))
print('%d unique' % len(visited))
