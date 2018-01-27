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
* `path`: Path of web page.
* `text`: The page text. String.
* `paras`: The paragraphs on the page. List of strings.
* `sents`: The sentences on the page. List of list of strings.
* `page_paras`: Number of paragraphs per page. Integer.
* `para_sents`: Number of sentences per page. List of integers.
* `page_chars`: Number of characters per page. List of integers.
* `sent_chars`: Number of chars per word per sentence. List of list of integers.

