# -*- coding: utf-8 -*-
import scrapy
from scrapy.http import TextResponse, HtmlResponse
from os import getcwd, makedirs
from os.path import abspath, join
from time import clock


web_root = 'https://www.papercut.com/'
data_root = 'pc_data'


def relative(url):
    if url.startswith(web_root):
        return 'pc_' + url[len(web_root):]
    return url


def local(url):
    return '@_%s' % relative(url).replace('/', '#')


def url_key(url):
    digits = [c for c in url if c.isdigit()]
    return url.count('&'), max(len(digits) - 5, 0), url.count('?'), len(url), len(digits), url


if False:
    print(local(web_root))
    assert False


class PcbotSpider(scrapy.Spider):
    name = 'pcbot'
    allowed_domains = [
        'www.papercut.com',
        'blog.papercut.com',
        'portal.papercut.com',
        # 'papercut.com',
    ]
    start_urls = [web_root]

    def __init__(self):
        super().__init__()
        self.data_dir = abspath(data_root)
        print('!!! %s' % getcwd())
        print('!!! %s' % self.data_dir)
        makedirs(self.data_dir, exist_ok=True)
        self.visited_url = set()
        self.visited_body = set()
        self.n_saved = 0
        self.t0 = clock()

    def _save_response(self, response):
        filename = join(self.data_dir, local(response.url))
        with open(filename, 'wb') as f:
            f.write(response.body)
        self.n_saved += 1
        # scrapy.log('Saved file: %4d: %s' % (self.n_saved, filename))

    def parse(self, response):
        # print('*** %s %r %4d' % (response.url, local(response.url), len(self.visited_url)))
        if response.url in self.visited_url:
            # print('####1')
            return
        self.visited_url.add(response.url)

        hsh = hash(response.body)
        if hsh in self.visited_body:
            # print('####2')
            return
        self.visited_body.add(hsh)

        if not isinstance(response, HtmlResponse):
            return

        self._save_response(response)

        print('visited=%d %d saved=%d %s' % (len(self.visited_url), len(self.visited_body),
            self.n_saved, response.url), flush=True)

        assert not response.url.endswith('.jpg')
        assert not response.url.endswith('.png')

        linked_pages = [link.extract() for link in response.css('a::attr(href)')]
        linked_pages = [page for page in linked_pages if page]
        linked_pages = [response.urljoin(page) for page in linked_pages]
        linked_pages = [page for page in linked_pages if page not in self.visited_url]
        linked_pages = [page for page in linked_pages if not
                        ('?action=edit' in page or
                         '?action=diff' in page or
                         '?action=print' in page)]
        linked_pages.sort(key=url_key)
        # print(linked_pages[:3], len(linked_pages))
        # print('%6d linked pages' % len(linked_pages))

        for i, next_page in enumerate(linked_pages):

            # print('- %5d: %s %s %s' % (i, next_page, next_page is not None, next_page not in self.visited_url))
            # if next_page is None:
            #     continue
            # next_page = response.urljoin(next_page)
            # if next_page in self.visited_url:
            #     continue

            # if '?action=edit' in next_page or '?action=diff' or '?action=print' in next_page:
            #     continue
            # print('%3d: %s %d' % (i, next_page, len(self.visited_url)))
            # yield response.follow()
            yield scrapy.Request(next_page, callback=self.parse)
            # print('  @ %3d: %s %d' % (i, next_page, len(self.visited_url)))
