from utils import load_jsonl, save_jsonl

paras1 = load_jsonl('kb.paragraphs.jsonl')
paras2 = load_jsonl('blog.paragraphs.jsonl')

print('kb')
print(len(paras1))
print(paras1[-1])
print('blog')
print(len(paras2))
print(paras2[-1])

out = []
out_text = set()

for paras in (paras1, paras2):
    for i, p in enumerate(paras):
        text = p['text']
        if text in out_text:
            continue
        out_text.add(text)
        out.append(p)

print('%d total' % len(out))
print('%d unique' % len(out_text))

save_jsonl('blog_kb.paragraphs.jsonl', out)
