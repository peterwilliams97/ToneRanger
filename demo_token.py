"""
    Show SpaCy tokenization
"""
import spacy
from collections import defaultdict


nlp = spacy.load('en')

para = '''
"The final release for the year is here and it\u2019s a cracker!",
We\u2019ve included loads of enhancements across a wide range of devices, including Integrated Scanning on Ricoh SOP SmartSDK devices.
Read more...
'''
para = para.strip(' \t\n')
doc = nlp(para)

vocab = defaultdict(int)
for token in doc:
    vocab[token.text] += 1

for i, token in enumerate(doc[:5]):
    print('%3d: %r %s %s' % (i, token.text,
        [token.lemma_, token.pos_, token.tag_, token.dep_, token.shape_],
        [token.is_alpha, token.is_stop]))

doc_char = len(para)
doc_word = len(doc)
doc_sent = len(list(doc.sents))
sent_word = [len(s) for s in doc.sents]
word_char = [[len(w) for w in s] for s in doc.sents]

for i, sent in enumerate(doc.sents):
    print('%2d: %3d %s' % (i, len(sent), ('-' * 80)))
    for j, token in enumerate(sent[:3]):
        print('%3d: %-10r %-30s %s' % (j, token.text,
            [token.lemma_, token.pos_, token.tag_, token.dep_, token.shape_],
            [token.is_alpha, token.is_stop]))
    print(sent)
print('-' * 80)
print(para)
print('=' * 80)

print('%d sentences : %d words : %d chars : %d vocab' % (doc_sent, doc_word, doc_char, len(vocab)))
print('sentence lengths: %d sentences : %d words : %s' % (doc_sent, sum(sent_word), sent_word))
print('word lengths: %d words : %d chars' % (doc_word, sum([sum(s) for s in word_char])))
print('  words / sentence %s' % [sum(s) for s in word_char])
print('  chars / word %s' % word_char)


# sents = [str(s) for s in doc.sents]
# sent_words = [len(s) for s in doc.sents]
# o['sents'].append(sents)
# o['sent_lens'].append(len(s) for s in sents)
# o['para_sents'].append(len(sents))
# o['sent_words'].append(sent_words)

