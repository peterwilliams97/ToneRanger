# -*- coding: utf-8 -*-
import scrapy
from scrapy.http import TextResponse, HtmlResponse
from os import getcwd, makedirs
from os.path import abspath, join
from time import clock


web_root = 'https://www.papercut.com/'
data_root = 'pc_data'


def relative(url):
    assert url.startswith(web_root), (web_root, url)
    return url[len(web_root):]


def local(url):
    return '@_%s' % relative(url).replace('/', '#')


if False:
    print(local(web_root))
    assert False


class PcbotSpider(scrapy.Spider):
    name = 'pcbot'
    allowed_domains = ['www.papercut.com']
    start_urls = [web_root]

    def __init__(self):
        super().__init__()
        self.data_dir = abspath(data_root)
        print('!!! %s' % getcwd())
        print('!!! %s' % self.data_dir)
        makedirs(self.data_dir, exist_ok=True)
        self.visited = set()
        self.n_saved = 0
        self.t0 = clock()

    def _save_response(self, response):
        filename = join(self.data_dir, local(response.url))
        with open(filename, 'wb') as f:
            f.write(response.body)
        self.n_saved += 1
        # scrapy.log('Saved file: %4d: %s' % (self.n_saved, filename))

    def parse(self, response):
        # print('*** %s %r %4d' % (response.url, local(response.url), len(self.visited)))
        self.visited.add(response.url)

        if not isinstance(response, HtmlResponse):
            return

        self._save_response(response)

        print('visited=%d saved=%d t=%.1f %s' % (len(self.visited), self.n_saved,
            clock() - self.t0, response.url), flush=True)

        assert not response.url.endswith('.jpg')
        assert not response.url.endswith('.png')

        for i, link in enumerate(response.css('a::attr(href)')):
            next_page = link.extract()
            # print('- %5d: %s %s %s' % (i, next_page, next_page is not None, next_page not in self.visited))
            if (next_page is not None) and (next_page not in self.visited):
                next_page = response.urljoin(next_page)
                # print('%3d: %s %d' % (i, next_page, len(self.visited)))
                # yield response.follow()
                yield scrapy.Request(next_page, callback=self.parse)
                # print('  @ %3d: %s %d' % (i, next_page, len(self.visited)))
