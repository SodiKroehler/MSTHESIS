from flask import request
import scrapy
from datetime import datetime
from allsides_scraper.items import AllSidesArticle, AllSidesArticleURL
from scrapy_playwright.page import PageMethod
from pathlib import Path
import time
import os
import json

CUTOFF_DATE = datetime(2025, 1, 1)

class AllSidesSpider(scrapy.Spider):
    name = "allsides"
    allowed_domains = ["allsides.com"]
    search_terms = ["ai", "immigration"]

    def start_requests(self):


        # Read URLs from a JL file
        urls = []
        jl_path = Path("allsides_articles_toplevel.jl")
        if jl_path.exists():
            with jl_path.open("r") as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        pdate = datetime.strptime(data.get("datetime", ""), "%b %d %Y")
                        if pdate < CUTOFF_DATE:
                            continue
                        itemDict = AllSidesArticleURL()
                        itemDict["search_term"] = data.get("search_term")
                        itemDict["page_num"] = data.get("page_num")
                        itemDict["allsides_url"] = data.get("allsides_url")
                        itemDict["datetime"] = pdate
                        urls.append(itemDict)
                    except Exception:
                        continue

            for urlitem in urls:
                yield scrapy.Request(
                    url=f"https://www.allsides.com{urlitem['allsides_url']}",
                    callback=self.parse_article,
                    meta={"search_term": urlitem["search_term"], 
                        "page_num": urlitem["page_num"], 
                        "playwright_include_page": True,
                        "playwright": True, 
                        "playwright_page_methods": [
                                # PageMethod("route", "**/*", self.block_ads),
                                # PageMethod("wait_for_timeout", 5000),
                                PageMethod("wait_for_selector", "div.view-content", timeout=10000),
                                PageMethod("screenshot", 
                                        path=str(f"../screenshot-{urlitem['search_term']}-pg{urlitem['page_num']}.png"),
                                        full_page=True)                            
                                ]
                        }

                )
                page_num += 1

    def parse_article(self, response):
        item = AllSidesArticle()
        item["search_term"] = response.meta["search_term"]
        item["page_num"] = response.meta["page_num"]
        item["allsides_url"] = response.url
        try: 
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
            

            if source_url:
                yield scrapy.Request(
                    source_url,
                    callback=self.parse_source,
                    meta={"item": item},
                    dont_filter=True
                )
            else:
                yield dict(item)
        except Exception as e:
            item["article_text"] = f"Error parsing article {response.url}: {e}"
            yield dict(item)


    def parse_source(self, response):
        item = response.meta["item"]
        item["article_text"] = " ".join(response.xpath('//body//*[not(self::script or self::style)]/text()').getall()).strip()
        yield dict(item)
