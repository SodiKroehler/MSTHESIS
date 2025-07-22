from flask import request
import scrapy
from datetime import datetime
from allsides_scraper.items import AllSidesArticle, AllSidesArticleURL
from scrapy_playwright.page import PageMethod
from pathlib import Path
import time
import os

CUTOFF_DATE = datetime(2025, 1, 1)

class AllSidesSpider(scrapy.Spider):
    name = "allsides"
    allowed_domains = ["allsides.com"]
    search_terms = ["ai", "immigration"]


    # async def block_ads(self, route, request):
    #     url = request.url
    #     # Always allow internal URLs
    #     if url.startswith("data:") or url.startswith("about:") or url.startswith("chrome-extension:"):
    #         await route.continue_()
    #     elif "allsides.com" in url:
    #         await route.continue_()
    #     else:
    #         await route.abort()


    def start_requests(self):


        screenshot_dir = Path(__file__).parent.parent.parent / "screenshots"
        screenshot_dir.mkdir(parents=True, exist_ok=True)

        start_page_num = 1  # Start from page 5 as per the original code
        for term in self.search_terms:
            for page_num in range(start_page_num, 2):
                yield scrapy.Request(
                    url=f"https://www.allsides.com/search?search={term}&page={page_num}",
                    callback=self.parse_search_results,
                    meta={"search_term": term, 
                        "page_num": page_num, 
                        "playwright_include_page": True,
                        "playwright": True, 
                        "playwright_page_methods": [
                                # PageMethod("route", "**/*", self.block_ads),
                                # PageMethod("wait_for_timeout", 5000),
                                PageMethod("wait_for_selector", "div.view-content", timeout=10000),
                                PageMethod("screenshot", 
                                        path=str(screenshot_dir / f"screenshot-{term}-pg{start_page_num}.png"),
                                        full_page=True)                            
                            ]
                        }

                )
                page_num += 1



    async def parse_search_results(self, response):
        search_term = response.meta["search_term"]
        page_num = response.meta["page_num"]
        try:
            articles = response.css("div.view-content > div.views-row")

            too_old = False
            for result in articles:
                item = AllSidesArticleURL()
                url = result.css("h3.search-result-title a::attr(href)").get()
                date_str = result.css("p.search-result-publish-date span::text").get(default="").strip()
                item["search_term"] = search_term
                item["page_num"] = page_num
                item["allsides_url"] = url
                item['datetime']   = date_str

                # Try several possible date formats
                for fmt in ("%B %d, %Y", "%b %d, %Y", "%b %d %Y", "%B %d %Y"):
                    try:
                        pub_date = datetime.strptime(date_str, fmt)
                        break
                    except ValueError:
                        pub_date = None

                # if pub_date and pub_date < CUTOFF_DATE:
                #     too_old = True
                #     break

                if url:
                    yield dict(item)
                    # yield response.follow(
                    #     url,
                    #     callback=self.parse_article,
                    #     meta={
                    #         "playwright": True,
                    #         "playwright_include_page": True,
                    #         "playwright_page_methods": [
                    #             # PageMethod("route", "**/*", self.block_ads),
                    #             # PageMethod("wait_for_timeout", 5000),  # wait 5s
                    #             PageMethod("wait_for_selector", "div.article-name", timeout=10000),
                    #             PageMethod("screenshot", 
                    #                     path=str(f"./screenshot-article-{search_term}-pg{page_num}.png"),
                    #                     full_page=True)                        
                    #         ],
                    #         "search_term": search_term,
                    #         "page_num": page_num
                    #     }
                    # )

            # if not too_old and page_num < 10:
            #     # print("Continuing, since last date {} was not less than cutoff date {}".format(pub_date, CUTOFF_DATE))
            #     next_page = page_num + 1
            #     # print(f"\033[93mNext page: {next_page}\033[0m")
            #     yield scrapy.Request(
            #         url=f"https://www.allsides.com/search?search={search_term}&page={next_page}",
            #         callback=self.parse_search_results,
            #         meta={"search_term": search_term, "page_num": next_page, "playwright": True}
            #     )
            # too_old = True #only run once
        finally:
            page = response.meta.get("playwright_page")
            if page:
                await page.close()

    def parse_article(self, response):
        item = AllSidesArticle()

        # #DEBUG
        # print(f"\033[93m{response.text[:500]}\033[0m")
        # print(f"\033[93mTitle: {response.css('div.article-name h1 span::text').get(default='').strip()}\033[0m")
        # print(f"\033[93mBias: {response.css('div.article-media-bias- > span > span > a::text').get()}\033[0m")
        # print(f"\033[93mTags: {response.css('div.article-page-detail div.page-tags a::text').getall()}\033[0m")
        print(f"\033[93mPublish Date: {response.css('div.article-page-detail div.article-posted-date::text').get()}\033[0m")
        print(f"\033[93mSource URL: {response.css('div.read-more-story a::attr(href)').get()}\033[0m")
        pull_time = time.time()
        print(f"\033[93mPull Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(pull_time))}\033[0m")
        # #DEBUG

        item["search_term"] = response.meta["search_term"]
        item["page_num"] = response.meta["page_num"]
        item["allsides_url"] = response.url

        item["detail_title"] = response.css("div.article-name h1 span::text").get(default="").strip()
        item["bias"] = response.css("div.article-media-bias- > span > span > a::text").get()
        tags = response.css("div.article-page-detail div.page-tags a::text").getall()
        combined_tags = " ".join(tag.strip() for tag in tags if tag.strip())
        item["tags"] = combined_tags

        pub_date_str = response.css("div.article-page-detail div.article-posted-date::text").get()
        if "Posted on AllSides" in pub_date_str:
            pub_date_str = pub_date_str.replace("Posted on AllSides ", "").strip()
        for fmt in ("%B %d, %Y", "%b %d, %Y", "%b %d %Y", "%B %d %Y"):
            try:
                pub_date = datetime.strptime(pub_date_str, fmt)
                break
            except ValueError:
                pub_date = None
        item["publish_date"] = pub_date

        source_url = response.css("div.read-more-story a::attr(href)").get()
        item["source_url"] = source_url
        

        # if source_url:
        #     yield scrapy.Request(
        #         source_url,
        #         callback=self.parse_source,
        #         meta={"item": item},
        #         dont_filter=True
        #     )
        # else:
        #     yield item
        yield dict(item)


    def parse_source(self, response):
        item = response.meta["item"]
        item["article_text"] = " ".join(response.xpath('//body//*[not(self::script or self::style)]/text()').getall()).strip()
        yield item
