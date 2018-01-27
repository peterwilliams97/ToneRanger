# ToneRanger
A program to detect the tone of documents.

# Document structure
* `page`: Web page
* `para`: Paragraph
* `sent`: Sentence
* `word`: Word
* `char`: Character

`page`es contain `para`s contain `sent`s contain `word`s contain `char`s.

# Keys
* `page_paras`: Number of paragraphs per page.
* `page_chars`: Number of characters per page.

 'para_chars', 'para_sents', 'paras', 'path', 'sent_lens', 'sent_words', 'sents', 'text']
['n_chars', 'n_paras', 'para_lens', 'para_sents', 'paras', 'path', 'sent_lens', 'sent_words', 'sents', 'text']

doc_char = len(para)
doc_word = len(doc)
doc_sent = len(list(doc.sents))
sent_word = [len(s) for s in doc.sents]
word_char = [[len(w) for w in s] for s in doc.sents]


