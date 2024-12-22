#! ScraperConfig
SC__REQUESTS_PER_SECOND = 10
SC__MAX_RETIES = 3
SC__HOTEL_REVIEWS_PAGE = "https://www.booking.com/reviewlist.en-gb.html"
SC__OUTPUT_DIR = "output"

# Common
PROCESS_POOL_SIZE = 5
SAFARI_UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15"
BASE_HEADERS = {"User-Agent": SAFARI_UA}
