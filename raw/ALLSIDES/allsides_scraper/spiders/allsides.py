from flask import request
import scrapy
from datetime import datetime
from allsides_scraper.items import AllSidesArticle
from scrapy_playwright.page import PageMethod
from pathlib import Path
import time
import os
import json
import pandas as pd
import sys
from scrapy.spidermiddlewares.httperror import HttpError
import copy
from urllib.parse import urlparse
import re
import random

CUTOFF_DATE = datetime(2025, 1, 1)
CUSTOM_ERROR_LOC = "./scrapy_errors.log"

class AllSidesSpider(scrapy.Spider):
    name = "allsides"
    allowed_domains = ["allsides.com"]
    search_terms = ["ai", "immigration"]
    handle_httpstatus_all = True

    error_dir = Path(__file__).parent.parent.parent / "errors"
    error_dir.mkdir(parents=True, exist_ok=True)
    ssdir = Path(__file__).parent.parent.parent / "screenshots"
    ssdir.mkdir(parents=True, exist_ok=True)
    article_dir = Path(__file__).parent.parent.parent / "articles"
    article_dir.mkdir(parents=True, exist_ok=True)
    c1df = pd.read_csv("ALLSIDES/jul23_complete_top_level_pulls.csv")

    completed = pd.read_json('allsides_articles_stage2.jl', lines=True)
    completed['urlie'] = completed['allsides_url'].str.replace('https://www.allsides.com/', "")
    completed = completed['urlie'].tolist()

    approved = {
        "nytimes.com": "lean_left",
        "msnbc.com": "left",
        "theguardian.com": "left",
        "reuters.com": "center",
        "apnews.com": "left",
        "npr.org": "lean_left",
        "foxnews.com": "right",
        "dailycaller.com": "right",
        "washingtonexaminer.com": "lean_right",
        "bbc.com": "center",
        "bbc.co.uk": "center",
        "aljazeera.com": "lean_left",
        "english.news.cn": "left",
    }


    def get_object(self, url):

        theRing = None
        stageComplete = 0
        # if url in self.c2df["allsides_url"].values:
        #     theRing = AllSidesArticle()
        #     row = self.c2df[self.c2df["allsides_url"] == url].iloc[0]
        #     for key in row.index:
        #         theRing[key] = row[key]
        #         stageComplete = 2
        if url in self.c1df["allsides_url"].values:
            theRing = AllSidesArticle()
            row = self.c1df[self.c1df["allsides_url"] == url].iloc[0]
            for key in row.index:
                if key in theRing.fields:
                    theRing[key] = row[key]
            stageComplete = 1
        return theRing, stageComplete

    def start_requests(self):
        start_page_num = 25
        end_page_num = 30
        for term in self.search_terms:
            for page in range(start_page_num, end_page_num + 1):

                yield scrapy.Request(
                    url=f"https://www.allsides.com/search?search={term}&page={page}",
                    callback=self.parse_search_results,
                    errback= self.errback,
                    meta={"search_term": term, 
                        "page_num": page,
                        "playwright_include_page": True,
                        "playwright": True, 
                        "playwright_page_methods": [
                                PageMethod("wait_for_timeout", random.uniform(1, 10) * 1000),
                                PageMethod("wait_for_selector", "div.view-content", timeout=10000),
                                # PageMethod("screenshot", 
                                #             path=str(self.ssdir / f"screenshot-{theRing['allsides_url']}.png"),
                                #             full_page=True)                            
                            ]
                    },
                )
                print("completed page")

    async def errback(self, failure):
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
        search_term = response.meta["search_term"]
        page_num = response.meta["page_num"]

        try:
            articles = response.css("div.view-content > div.views-row")
            for result in articles:
                item = AllSidesArticle()
                url = result.css("h3.search-result-title a::attr(href)").get()
                date_str = result.css("p.search-result-publish-date span::text").get(default="").strip()
                theRing, stageComplete = self.get_object(url)

                uuid = hash(url)

                if theRing:
                    item.update({k: v for k, v in theRing.items() if k in item.fields})
                    yield dict(item)
                else:
                    if not url.startswith("https://www.allsides.com"):
                        url = "https://www.allsides.com" + url
                    #parse away
                    yield response.follow(
                        url,
                        callback=self.parse_article,
                        meta={
                            "playwright": True,
                            "playwright_include_page": True,
                            "playwright_page_methods": [
                                #simulates full page load (e.g cloudflare bugger)
                                PageMethod(
                                        "goto",
                                        url,
                                        wait_until="networkidle",
                                        timeout=30000  # optional: increase if the site is slow
                                    ),
            
                                PageMethod("wait_for_selector", "#block-views-article-page-redesign-block-1 > div > div", timeout=10000), #block-views-article-page-redesign-block-1 > div > div

                                PageMethod("screenshot", 
                                        path=str(f"./screenshot-article-{search_term}-pg{page_num}.png"),
                                        full_page=True)                        
                            ],
                            "search_term": search_term,
                            "page_num": page_num,
                            "date_str": date_str, 
                            "uuid" : uuid
                        }
                    )
                    
        finally:
            page = response.meta.get("playwright_page")
            if page:
                print(f"\033[93mClosing page for {response.url}\033[0m")
                await page.close()



    async def parse_article(self, response):
        item = AllSidesArticle()

        item["search_term"] = response.meta["search_term"]
        item["page_num"] = response.meta["page_num"]
        date_str = response.meta["date_str"]
        item["uuid"] = response.meta["uuid"]
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
            item["publish_date"] = pub_date_str if pub_date_str else date_str
            item["allsides_text"] = response.css('div.article-description p::text').get()

            source_url = response.css("div.read-more-story a::attr(href)").get()
            item["source_url"] = source_url
            if source_url:
                    source = urlparse(source_url).netloc
                    if source not in self.approved:
                        item['source_url'] = source_url
                        item['tags'] = "unapproved"
                        yield dict(item) #will be empty
                    else:
                        yield scrapy.Request(
                            source_url,
                            callback=self.parse_source,
                            errback=self.errback, 
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
            print(f"\033[91mError parsing article {response.url}: {e}\033[0m" , file=sys.stderr)
        
        finally: 

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
            full_text = re.sub(r'\r\n?|\n', '<<LINEBREAK>>', full_text)
            full_text = re.sub(r'\t', '<<TAB>>', full_text)
            full_text = re.sub(r'["\'“”‘’`´]', '<<QUOTE>>', full_text)
            full_text = json.dumps({"full_source_text": full_text}) #to escape stuff
            item["article_text"] = full_text
        except Exception as e:
            item["article_text"] = f"Error parsing source {response.url}: {e}"
            print(f"\033[91mError parsing source {response.url}: {e}\033[0m", file=sys.stderr)
        finally:
            page = response.meta.get("playwright_page")
            if page:
                print(f"\033[93mClosing page for {response.url}\033[0m")
                await page.close()
            yield dict(item)