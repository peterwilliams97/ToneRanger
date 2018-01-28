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

# Installation and Running
    git clone https://github.com/peterwilliams97/ToneRanger
    pushd ToneRanger/paper_spider
    scrapy crawl --logfile=mylog.log --loglevel=INFO pcbot
    cd ..
    python scrape.py
    python tokenise.py
    python metrics.py

* `scrapy crawl pcbot` crawls (www.papercut.com)[https://www.papercut.com/] and dumps all
html pages to the `ToneRanger/paper_spider/pc_data/` directory
* `python scrape.py` extracts the html pages `<p>` tags and writes the results in json format
    in `ToneRanger/page.summaries/`
* `python tokenise.py` updates the json summaries in `ToneRanger/page.summaries/` by breaking paragraphs into sentences
* `python metrics.py` computes metrics on the summaries in `ToneRanger/page.summaries/`

# Spider
(web-scraping-in-python-using-scrapy)[https://www.analyticsvidhya.com/blog/2017/07/web-scraping-in-python-using-scrapy/]
