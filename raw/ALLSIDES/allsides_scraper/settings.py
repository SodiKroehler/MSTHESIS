import random 
BOT_NAME = "allsides_scraper"

SPIDER_MODULES = ["allsides_scraper.spiders"]
NEWSPIDER_MODULE = "allsides_scraper.spiders"

ROBOTSTXT_OBEY = False
DOWNLOAD_DELAY = random.uniform(1, 5)

FEEDS = {
    'manual_allsides_articles_stage2.jl': {
        'format': 'jsonlines',
        'encoding': 'utf8', 
    }
}

# Playwright
DOWNLOAD_HANDLERS = {
    "http": "scrapy_playwright.handler.ScrapyPlaywrightDownloadHandler",
    "https": "scrapy_playwright.handler.ScrapyPlaywrightDownloadHandler",
}
TWISTED_REACTOR = "twisted.internet.asyncioreactor.AsyncioSelectorReactor"
# PLAYWRIGHT_BROWSER_TYPE = "chromium"
PLAYWRIGHT_BROWSER_TYPE = "firefox"

USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
# FEED_EXPORT_BATCH_ITEM_COUNT = 10
PLAYWRIGHT_DEFAULT_NAVIGATION_TIMEOUT = 15000  # 15 seconds
# DOWNLOAD_TIMEOUT = 20  # apply at Scrapy layer too
# Lower concurrency to force tabs to close quicker
CONCURRENT_REQUESTS = 4
PLAYWRIGHT_MAX_CONTEXTS = 2
PLAYWRIGHT_ABORT_REQUESTS_ON_DISCONNECT = True
PLAYWRIGHT_LAUNCH_OPTIONS = {
    "headless": True,
    "args": ["--disable-dev-shm-usage", "--no-sandbox"],
}
RETRY_ENABLED = True
RETRY_TIMES = 5
LOG_LEVEL = 'INFO'

# DOWNLOADER_MIDDLEWARES = {
#     'scrapy.downloadermiddlewares.httpproxy.HttpProxyMiddleware': 110,
# }

# HTTP_PROXY = 'socks5://127.0.0.1:9050'