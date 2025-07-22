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

    def start_requests(self):


        screenshot_dir = Path(__file__).parent.parent.parent / "screenshots"
        screenshot_dir.mkdir(parents=True, exist_ok=True)

        start_page_num = 1  # Start from page 5 as per the original code
        for term in self.search_terms:
            for page_num in range(start_page_num, 900):
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
                yield dict(item)
                    
        finally:
            page = response.meta.get("playwright_page")
            if page:
                await page.close()
