import scrapy

class AllSidesArticle(scrapy.Item):
    search_term = scrapy.Field()
    page_num = scrapy.Field()
    listing_title = scrapy.Field()
    detail_title = scrapy.Field()
    bias = scrapy.Field()
    tags = scrapy.Field()
    publish_date = scrapy.Field()
    source_url = scrapy.Field()
    article_text = scrapy.Field()
    allsides_text = scrapy.Field()
    allsides_url = scrapy.Field()

class AllSidesArticleURL(scrapy.Item):
    search_term = scrapy.Field()
    page_num = scrapy.Field()
    allsides_url = scrapy.Field()
    datetime = scrapy.Field()
    secondary_pull = scrapy.Field()
