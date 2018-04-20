import scrapy


class PaperCutSpider(scrapy.Spider):
    name = "papercut_spider"
    start_urls = ['https://www.papercut.com']

    def parse(self, response):
        SET_SELECTOR = '.set'
        for selection in response.css(SET_SELECTOR):
            pass
