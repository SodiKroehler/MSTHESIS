from flask import request
import scrapy
from datetime import datetime
from allsides_scraper.items import AllSidesArticle, AllSidesArticleURL
from scrapy_playwright.page import PageMethod
from pathlib import Path
import time
import os
import json
import pandas as pd
import sys
from scrapy.spidermiddlewares.httperror import HttpError
import copy
import random

CUTOFF_DATE = datetime(2025, 1, 1)
CUSTOM_ERROR_LOC = "./scrapy_errors.log"

class AllSidesSpider(scrapy.Spider):
    name = "allsides"
    allowed_domains = ["allsides.com"]
    # search_terms = ["immigration"]
    search_terms = ["ai", "immigration"]
    handle_httpstatus_all = True

    error_dir = Path(__file__).parent.parent.parent / "errors"
    error_dir.mkdir(parents=True, exist_ok=True)
    ssdir = Path(__file__).parent.parent.parent / "screenshots"
    ssdir.mkdir(parents=True, exist_ok=True)
    article_dir = Path(__file__).parent.parent.parent / "articles"
    article_dir.mkdir(parents=True, exist_ok=True)
    c1df = pd.read_csv("ALLSIDES/jul23_complete_top_level_pulls.csv")
    c2df = pd.read_csv("ALLSIDES/jul23_incomplete_secondary_pulls.csv")

    completed = pd.read_json('allsides_articles_stage2.jl', lines=True)
    completed['urlie'] = completed['allsides_url'].str.replace('https://www.allsides.com/', "")
    completed = completed['urlie'].tolist()

    def get_object(self, url):

        theRing = None
        stageComplete = 0
        if url in self.c2df["allsides_url"].values:
            theRing = AllSidesArticle()
            row = self.c2df[self.c2df["allsides_url"] == url].iloc[0]
            for key in row.index:
                theRing[key] = row[key]
                stageComplete = 2
        elif url in self.c1df["allsides_url"].values:
            theRing = AllSidesArticleURL()
            row = self.c1df[self.c1df["allsides_url"] == url].iloc[0]
            for key in row.index:
                theRing[key] = row[key]
                stageComplete = 1
        return theRing, stageComplete

    def start_requests(self):
        start_num = 1000
        for page in range(start_num, len(self.c1df) + 1):
        # for page in range(start_num, start_num + 1):
            theRing, stageComplete = self.get_object(self.c1df.iloc[page - 1]["allsides_url"])
            # time.sleep(5) skipping since one at a time rn
            if theRing['allsides_url'] in self.completed:
                print(f"skipping {page} as its already done")
                continue

            print(f"Processing page {page} of {len(self.c1df)}: {theRing['allsides_url']}")
            yield scrapy.Request(
                url=f"https://www.allsides.com/{theRing['allsides_url']}",
                callback=self.parse_article,
                errback= self.errback,
                meta={"search_term": theRing["search_term"], 
                    "page_num": theRing["page_num"], 
                    "source_url": theRing["source_url"] if stageComplete == 2 else None,
                    "allsides_url": f"https://www.allsides.com/{theRing['allsides_url']}",
                    "playwright_include_page": True,
                    "playwright": True, 
                    "playwright_page_methods": [
                            PageMethod("wait_for_timeout", random.uniform(6, 20) * 10000),
                            PageMethod("wait_for_selector", "#block-views-article-page-redesign-block-1 > div > div", timeout=10000) #block-views-article-page-redesign-block-1 > div > div
                            # PageMethod("screenshot", 
                            #             path=str(self.ssdir / f"screenshot-{theRing['allsides_url']}.png"),
                            #             full_page=True)                            
                        ]
                },
            )

    async def errback(self, failure):
        # fail = await failure
        page = failure.request.meta.get("playwright_page")
        if isinstance(failure.value, HttpError):
            real_resp = failure.value.response
            if real_resp.status == 404:
                print(f"\033[91m[404] Page not found: {real_resp.url}\033[0m", file=sys.stderr)
                return
        allsides_url = failure.request.meta.get("allsides_url", "unknown")
        if allsides_url == "unknown":
            item = failure.request.meta.get("item", {})
            if item:
                allsides_url = item.get("allsides_url", "unknown")
        print(f"got to errback for {allsides_url}")
        # breakpoint()
        if page:
            try:
                timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
                screenshot_path = self.error_dir / f"screenshot_error_{timestamp}.png"
                await page.screenshot(path=screenshot_path)
                

    

                html_path = self.error_dir / f"failure_{timestamp}.html"
                with open(html_path, 'w') as f:
                    html_content = await page.content()
                    f.write(html_content)

                logMsg = ""
                error_message = str(failure.value)
                request_url = str(failure.request.url)
                logMsg += "___________________________________________\n\n"
                logMsg += f"Page failed to load: {request_url}\n"
                logMsg += f"Error message: {error_message}\n"
                logMsg += f"Page HTML saved to: {html_path}\n"
                log_path = self.error_dir / f"log_error_{timestamp}.errlog"
                with open(log_path, 'w') as log_file:
                    log_file.write(logMsg)
                logMsg += f"Error log saved to: {log_path}\n"

            except Exception as e:
                print(f"\033[91mError in error handling: {str(e)}\033[0m", file=sys.stderr)

            finally:
                # Ensure the page is properly closed after handling the error
                await page.close()
        else:
            print(f"\033[91mError in error handling: No page object found\033[0m", file=sys.stderr)
            if page:
                print(f"\033[93mClosing page for {allsides_url}\033[0m")
                await page.close()


    async def parse_search_results(self, response):
        # repo = await response
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
                print(f"\033[93mClosing page for {response.url}\033[0m")
                await page.close()



    async def parse_article(self, response):
        item = AllSidesArticle()
        # repo = await response
        item["search_term"] = response.meta["search_term"]
        item["page_num"] = response.meta["page_num"]
        item["allsides_url"] = response.url
        source_url = response.meta.get("source_url", None)
        #there's not a huge reason to call this out ahead, we already had to do the request.
        # if source_url:
        #     item["source_url"] = source_url
        #     yield scrapy.Request(
        #         source_url,
        #         callback=self.parse_source,
        #         errback=self.errback,
        #         meta={"item": item},
        #         dont_filter=True
        #     )
        # else:
        try: 
            item["detail_title"] = response.css("div.article-name h1 span::text").get(default="").strip()
            item["bias"] = response.css("div.article-media-bias- > span > span > a::text").get()
            tags = response.css("div.article-page-detail div.page-tags a::text").getall()
            combined_tags = " ".join(tag.strip() for tag in tags if tag.strip())
            item["tags"] = combined_tags

            pub_date_str = response.css("div.article-page-detail div.article-posted-date::text").get()
            if "Posted on AllSides" in pub_date_str:
                pub_date_str = pub_date_str.replace("Posted on AllSides ", "").strip()
            item["publish_date"] = pub_date_str
            # item["allsides_text"] = response.css('div.article-description::text').getall()
            item["allsides_text"] = response.css('div.article-description p::text').get()

            source_url = response.css("div.read-more-story a::attr(href)").get()
            item["source_url"] = source_url
            if source_url:
                yield scrapy.Request(
                    source_url,
                    callback=self.parse_source,
                    errback=self.errback, #worrried thsi will break and we can always do another pass
                    # handle_httpstatus_all=True, # got an unexpected keyword argument 'handle_httpstatus_all'
                    dont_filter=True,
                    meta={
                        "item": copy.deepcopy(item),
                        "playwright": True,
                        "playwright_include_page": True,
                    }

                )
            else:
                yield dict(item)
        except Exception as e:
            # item["article_text"] = f"Error parsing article {response.url}: {e}"
            print(f"\033[91mError parsing article {response.url}: {e}\033[0m" , file=sys.stderr)

        page = response.meta.get("playwright_page")
        if page:
            print(f"\033[93mClosing page for {response.url}\033[0m")
            await page.close()
        yield dict(item)


    async def parse_source(self, response):
        # repo = await response
        item = response.meta["item"]
        try: 
            full_text = response.xpath('//body//*[not(self::script or self::style)]/text()').getall()
            custom_id = int(time.time() * 1000)
            file_path = self.article_dir / f"article_{custom_id}.txt"
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(" ".join(full_text).strip())
            item["article_text"] = str(custom_id)
            # item["article_text"] = " ".join(response.xpath('//body//*[not(self::script or self::style)]/text()').getall()).strip()
        except Exception as e:
            item["article_text"] = f"Error parsing source {response.url}: {e}"
            print(f"\033[91mError parsing source {response.url}: {e}\033[0m", file=sys.stderr)
        finally:
            # Ensure the page is properly closed after parsing
            page = response.meta.get("playwright_page")
            if page:
                print(f"\033[93mClosing page for {response.url}\033[0m")
                await page.close()
            # Yield the item
            yield dict(item)