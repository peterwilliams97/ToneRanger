import spacy
from utils import load_jsonl
nlp = spacy.load('predict/my_model')

texts = [
'Today is sunny',
'I hate bunnies',
'''Chris Goult recently joined us as our shiny new channel marketing manager.  Chris comes to us from Konica Minolta Business Solutions Australia, with loads of knowledge and enthusiasm for all things marketing and channel related. ''',
'''You don't want your precious funds ending up as wasted paper in recycle bins.''',
'''At the core of PaperCut MF is the ability to interface directly with MFD hardware to track off-the-glass functions such as copy, scan, fax and secure print release. PaperCut has worked directly with leading MFD manufacturers to bring our software directly to the MFD at a firmware level. To complete the solution offering across all devices PaperCut MF also supports hardware copier terminals from multiple vendors. ''',
]
for text in texts:
    print('-' * 80)
    print(text)
    doc = nlp(text)
    print(doc.cats)

print('=' * 80)
paras = load_jsonl('pc.paragraphs.jsonl')
print(len(paras))
print(paras[-1])
results = []
for i, p in enumerate(paras):
    text = p['text']
    url = p['meta']['url']
    doc = nlp(text)
    score = doc.cats
    results.append((text, url, score))
    if i % 100 == 50:
        # text = text.decode('utf-8', errors='ignore')
        text = text.encode('cp850','replace').decode('cp850')
        print('%4d: %-80s %s' % (i, text[:80], score))
