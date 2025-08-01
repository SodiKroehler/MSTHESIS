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
import re

CUTOFF_DATE = datetime(2025, 1, 1)
CUSTOM_ERROR_LOC = "./scrapy_errors.log"

class AllSidesSpider(scrapy.Spider):
    name = "allsides"
    # allowed_domains = ["allsides.com"]
    # search_terms = ["immigration"]
    search_terms = ["ai", "immigration"]
    handle_httpstatus_all = True

    error_dir = Path(__file__).parent.parent.parent / "errors"
    error_dir.mkdir(parents=True, exist_ok=True)
    ssdir = Path(__file__).parent.parent.parent / "screenshots"
    ssdir.mkdir(parents=True, exist_ok=True)
    article_dir = Path(__file__).parent.parent.parent / "articles"
    article_dir.mkdir(parents=True, exist_ok=True)
    c1df = pd.read_csv("manual_pull_jul29/firstpass_manual_scraper_allsides.csv")
    # c1df = c1df['source_url'].tolist()
    c1df = c1df.head(2)

    def start_requests(self):
        start_num = 0
        for idx, row in self.c1df.iterrows():
            item = AllSidesArticle()
            item['source_url'] = row['source_url']
            item['detail_title'] = row['title']
            item['search_term'] = "immigration" if row['pull'] == "jul25_allsides_imm" else "ai"
            item['page_num'] = None
            item['bias'] = row['allsides_leaning']
            item['publish_date'] = row['date']
            item['allsides_text'] = row['text'] if pd.notna(row['text']) else None

            yield scrapy.Request(
                url=row['source_url'],
                callback=self.parse_source,
                errback= self.errback,
                meta={"item": copy.deepcopy(item),
                    "playwright_include_page": True,
                    "playwright": True, 
                    "playwright_page_methods": [
                            PageMethod("wait_for_timeout", random.uniform(6, 20) * 10000),                          
                        ]
                },
            )

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
                page = response.meta.get("playwright_page")
                if page:
                    print(f"\033[93mClosing page for {response.url}\033[0m")
                    await page.close()
                yield dict(item)
            except Exception as e:
                item["article_text"] = f"Error parsing source {response.url}: {e}"
                print(f"\033[91mError parsing source {response.url}: {e}\033[0m", file=sys.stderr)
                page = response.meta.get("playwright_page")
                if page:
                    print(f"\033[93mClosing page for {response.url}\033[0m")
                    await page.close()
                yield dict(item)
            finally:
                page = response.meta.get("playwright_page")
                if page:
                    print(f"\033[93mClosing page for {response.url}\033[0m")
                    await page.close()
                yield dict(item)
