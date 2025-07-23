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
CUSTOM_ERROR_LOC = "./scrapy_errors.log"

class AllSidesSpider(scrapy.Spider):
    name = "allsides"
    allowed_domains = ["allsides.com"]
    # search_terms = ["immigration"]
    search_terms = ["ai", "immigration"]

    already_parsed_ai = []
    already_parsed_imm = []
    with open('/home/tia/Documents/WORKBENCH/MSTHESIS/MSTHESIS/raw/ALLSIDES/allsides_articles_toplevel.jl', 'r') as file:
        for line in file:
            data = json.loads(line.strip())
            if data.get("search_term") == "ai":
                already_parsed_ai.append(data.get("page_num"))
            else:
                already_parsed_imm.append(data.get("page_num"))

    def start_requests(self):


        screenshot_dir = Path(__file__).parent.parent.parent / "screenshots"
        screenshot_dir.mkdir(parents=True, exist_ok=True)

        start_page_num = 20

        for term in self.search_terms:
            for page_num in range(start_page_num, 250):
                if str(page_num) in self.already_parsed_ai or len(self.already_parsed_ai) > 100:
                    print(f"skipping page {page_num} for {term}")
                    yield None
                elif str(page_num) in self.already_parsed_imm or len(self.already_parsed_imm) > 100:
                    print(f"skipping page {page_num} for {term}")
                    yield None
                else:
                    time.sleep(5)  # Non-blocking sleep
                    yield scrapy.Request(
                        url=f"https://www.allsides.com/search?search={term}&page={page_num}",
                        callback=self.parse_search_results,
                        errback= self.errback,
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
                            
                            },
                        

                    )
                page_num += 1

    # async def errback(self, failure): 
    #     page = failure.request.meta("playwright-page")
    #     print("page failed")
    #     await page.close()

    async def errback(self, failure):
        page = failure.request.meta.get("playwright-page")
        if page:
            try:
                timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
                screenshot_path = f'screenshots/failure_{timestamp}.png'
                await page.screenshot(path=screenshot_path)
                logMsg = ""

                error_message = str(failure.value)
                request_url = failure.request.url
                logMsg += "___________________________________________\n\n"
                logMsg += f"Page failed to load: {request_url}\n"
                logMsg += f"Error message: {error_message}\n"

                html_path = f'errors/html/failure_{timestamp}.html'
                with open(html_path, 'wb') as f:
                    html_content = await page.content()
                    f.write(html_content)
                logMsg += f"Page HTML saved to: {html_path}\n"

            except Exception as e:
                self.logger.error(f"Error in error handling: {str(e)}")

            finally:
                # Ensure the page is properly closed after handling the error
                await page.close()
        else:
            self.logger.error("No page object found in error callback")

        # You can also log the failure to a custom log file
        self.logger.error(f"Request failed: {failure.request.url} | Error: {str(failure.value)}")

    async def parse_search_results(self, response):

        #trying manual sleep
        await asyncio.sleep(2)



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
