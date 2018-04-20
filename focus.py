from utils import load_jsonl, save_jsonl
from urllib.parse import urlparse

# paras = load_jsonl('pc.paragraphs.jsonl')
paras = load_jsonl('blog_kb.paragraphs.jsonl')
print(len(paras))
print(paras[-1])
results = []
out = []
for i, p in enumerate(paras):
    text = p['text'].lower()
    url = p['meta']['url']
    if not any(k in url.lower() for k in ('kb', 'tour', 'faq', 'blog')):
        continue
    out.append(p)
    results.append(url)

print('%d total' % len(results))
results = sorted(set(results))
print('%d unique' % len(results))

for i, url in enumerate(results[:20]):
    o = urlparse(url)
    print('%4d: %s' % (i, o))

# save_jsonl('kb.paragraphs.jsonl', out)
save_jsonl('final.paragraphs.jsonl', out)
