import csv
import logging
import multiprocessing as mp
import os
import re
from concurrent.futures import ThreadPoolExecutor, wait
from datetime import datetime
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs, urlparse

import numpy as np
import requests
from bs4 import BeautifulSoup, Tag
from dateutil import parser

from scraper.booking._constants import (
    BASE_HEADERS,
    PROCESS_POOL_SIZE,
    SC__HOTEL_REVIEWS_PAGE,
    SC__MAX_RETIES,
    SC__OUTPUT_DIR,
    SC__REQUESTS_PER_SECOND,
)
from scraper.booking.models.scraper import ScraperConfig, ScraperInput


def _setup_logger() -> logging.Logger:
    os.makedirs("logs", exist_ok=True)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "\033[1;32m%(asctime)s\033[0m - \033[1;34m%(name)s\033[0m - \033[1;31m%(levelname)s\033[0m - %(message)s"
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


class ReviewScraper:
    def __init__(
        self,
        input_params: Dict[str, Any],
        save_to_disk: bool = True,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.BASE_HEADERS = BASE_HEADERS
        self.PROCESS_POOL_SIZE = PROCESS_POOL_SIZE

        self.job_id = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        os.environ["job_id"] = self.job_id

        self.logger = logger or self._setup_logger()
        self.config = self._load_config()
        self.input_params = ScraperInput(**input_params)

        self._parsed_pages = mp.Manager().list()
        self._execution_finished = mp.Event()
        self._save_to_disk = save_to_disk

        self._log_input_params()

    def _setup_logger(self) -> logging.Logger:
        return _setup_logger()

    def _load_config(self) -> ScraperConfig:
        return ScraperConfig(
            REQUESTS_PER_SECOND=SC__REQUESTS_PER_SECOND,
            MAX_RETIES=SC__MAX_RETIES,
            HOTEL_REVIEWS_PAGE=SC__HOTEL_REVIEWS_PAGE,
            OUTPUT_DIR=SC__OUTPUT_DIR,
        )

    def _log_input_params(self) -> None:
        params = "\n".join(f"+ {k}: {v}" for k, v in self.input_params.__dict__.items())
        self.logger.info(
            f"Scraping reviews of {self.input_params.hotel_name} with the following params:\n{params}"
        )

    def _get_max_offset(self) -> int:
        response = requests.get(
            self.config.HOTEL_REVIEWS_PAGE,
            params={
                "cc1": self.input_params.country,
                "pagename": self.input_params.hotel_name,
                "rows": 10,
            },
            headers=self.BASE_HEADERS,
        )

        soup = BeautifulSoup(response.content.decode(), "html.parser")
        page_links = soup.select(
            "div.bui-pagination__pages > div.bui-pagination__list > div.bui-pagination__item > a"
        )

        last_page = [a for a in page_links if a.find("span") and "Page " in a.text]
        if not last_page:
            return 0

        parsed_url = urlparse(last_page[-1].get("href", ""))
        offset = parse_qs(parsed_url.query).get("offset", ["0"])[0].split(";")[0]
        return int(offset) if offset.isdigit() else 0

    def _create_urls(self) -> List[Dict[str, Any]]:
        max_offset = self._get_max_offset()
        sort_by_mapping = {
            "most_relevant": "",
            "newest_first": "f_recent_desc",
            "oldest_first": "f_recent_asc",
            "highest_scores": "f_score_desc",
            "lowest_scores": "f_score_asc",
        }

        urls = []
        for offset in range(0, max_offset + 1, 10):
            params = {
                "cc1": self.input_params.country,
                "pagename": self.input_params.hotel_name,
                "rows": 10,
                "sort": sort_by_mapping[self.input_params.sort_by],
            }
            if offset:
                params["offset"] = offset

            url = (
                requests.Request("GET", self.config.HOTEL_REVIEWS_PAGE, params=params)
                .prepare()
                .url
            )
            urls.append({"idx": offset, "url": url})

        self.logger.info(f"Created URLs: {len(urls)}")
        return urls

    def _extract_text(self, element: Optional[Tag]) -> Optional[str]:
        if not element:
            return None
        text = re.sub(
            r"\s+", " ", element.text if isinstance(element, Tag) else element
        ).strip()
        return text if text else None

    def _parse_review(self, review: Tag) -> Dict[str, Any]:
        def get_review_text() -> tuple:
            texts = review.select("div.c-review span.c-review__body")
            if not texts:
                return None, None, None, None

            liked = self._extract_text(texts[0])
            if liked and "no comments available" in liked.lower():
                liked = None

            disliked = self._extract_text(texts[1] if len(texts) > 1 else None)
            if not disliked and len(texts) > 2:
                disliked = self._extract_text(texts[2])

            original_lang = texts[0].get("lang") if texts else None

            parts = []
            if liked:
                parts.append(f"liked: {liked}")
            if disliked:
                parts.append(f"disliked: {disliked}")

            full_text = " ".join(parts)
            return liked, disliked, original_lang, full_text

        username = self._extract_text(
            review.select_one("div.c-review-block__guest span.bui-avatar-block__title")
        )
        date_elem = review.find(
            lambda tag: tag.name == "span" and "Reviewed:" in tag.get_text()
        )
        review_date = (
            parser.parse(date_elem.text.split(":")[-1].strip()).strftime(
                "%m-%d-%Y %H:%M:%S"
            )
            if date_elem
            else None
        )

        liked, disliked, lang, full_review = get_review_text()

        return {
            "username": username,
            "user_country": self._extract_text(
                review.select_one("span.bui-avatar-block__subtitle")
            ),
            "room_view": self._extract_text(
                review.select_one(
                    "div.c-review-block__room-info-row div.bui-list__body"
                )
            ),
            "stay_duration": (
                self._extract_text(
                    review.select_one("ul.c-review-block__stay-date div.bui-list__body")
                )
                or ""
            ).split(" ·")[0],
            "stay_type": self._extract_text(
                review.select_one(
                    "ul.review-panel-wide__traveller_type div.bui-list__body"
                )
            ),
            "review_post_date": review_date,
            "review_title": self._extract_text(
                review.select_one("h3.c-review-block__title")
            ),
            "rating": float(
                self._extract_text(review.select_one("div.bui-review-score__badge"))
                or 0
            ),
            "original_lang": lang,
            "review_text_liked": liked,
            "review_text_disliked": disliked,
            "full_review": full_review,
            "en_full_review": full_review if lang == "en" else None,
            "found_helpful": self._extract_text(
                review.select_one("p.review-helpful__vote-others-helpful")
            ),
            "found_unhelpful": self._extract_text(review.select_one("p.--unhelpful")),
            "owner_resp_text": self._extract_text(
                review.select(
                    "div.c-review-block__response span.c-review-block__response__body"
                )[-1]
                if review.select("div.c-review-block__response")
                else None
            ),
        }

    def _scrape_page(self, url_dict: Dict[str, str]) -> Dict[str, Any]:
        for _ in range(self.config.MAX_RETIES):
            response = requests.get(url_dict["url"], headers=self.BASE_HEADERS)
            if response.status_code == 200:
                return {"idx": url_dict["idx"], "response": response}
            self.logger.warning(f"Retrying... {url_dict['url']}")
        return {"idx": url_dict["idx"], "response": None}

    def _parse_response(self, response_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
        if not response_dict["response"]:
            return []

        soup = BeautifulSoup(response_dict["response"].content.decode(), "html.parser")
        reviews = [
            self._parse_review(review) for review in soup.select("ul.review_list > li")
        ]

        self._parsed_pages.append({"idx": response_dict["idx"], "reviews": reviews})
        return reviews

    def _save_reviews(self, reviews: List[Dict[str, Any]]) -> None:
        if not reviews or not self._save_to_disk:
            return

        output_path = (
            f"{self.config.OUTPUT_DIR}/{self.input_params.hotel_name}_{self.job_id}"
        )
        os.makedirs(output_path, exist_ok=True)

        filename = f"{output_path}/reviews_{self.input_params.sort_by}.csv"
        write_header = not os.path.exists(filename)

        with open(filename, "a", newline="") as file:
            writer = csv.writer(file)
            if write_header:
                writer.writerow(reviews[0].keys())
            for review in reviews:
                writer.writerow(review.values())

    def _scrape_all(self, urls: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        with ThreadPoolExecutor(
            max_workers=self.config.REQUESTS_PER_SECOND
        ) as executor:
            futures = []
            for i, url in enumerate(urls, 1):
                futures.append(executor.submit(self._scrape_page, url))
                if i % self.config.REQUESTS_PER_SECOND == 0:
                    import time

                    time.sleep(1)

            wait(futures)
            responses = list(executor.map(lambda f: f.result(), futures))

        processes = []
        for split in np.array_split(responses, self.PROCESS_POOL_SIZE):
            if len(split):
                p = mp.Process(
                    target=lambda x: [self._parse_response(r) for r in x], args=(split,)
                )
                p.start()
                processes.append(p)

        for p in processes:
            p.join()

        sorted_reviews = sorted(self._parsed_pages, key=lambda x: x["idx"])
        return [review for page in sorted_reviews for review in page["reviews"]]

    def _scrape_conditional(self, urls: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        reviews = []
        for url in urls:
            page_reviews = self._parse_response(self._scrape_page(url))
            reviews.extend(page_reviews)

            if 0 < self.input_params.n_rows <= len(reviews):
                return reviews[: self.input_params.n_rows]

        return reviews

    def run(self) -> List[Dict[str, Any]]:
        start_time = datetime.now()
        urls = self._create_urls()

        reviews = (
            self._scrape_all(urls)
            if self.input_params.n_rows == -1
            else self._scrape_conditional(urls)
        )

        self.logger.info(
            f"Scraped {len(reviews)} reviews of {self.input_params.hotel_name} in {(datetime.now() - start_time).total_seconds():.1f} seconds"
        )
        self._execution_finished.set()

        if self._save_to_disk:
            self._save_reviews(reviews)

        return reviews
