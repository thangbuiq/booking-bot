from booking.utils import ReviewScraper

from logging import Logger
from typing import List

import pandas as pd
import os


def scrape_reviews_of_hotel(
    hotel_name: str,
    country: str,
    sort_by: str = "most_relevant",
    n_reviews: int = 100,
    logger: Logger | None = None,
) -> List[dict]:
    """To run the scrapper as module of a single hotel on Booking.com

    Args:
        hotel_name: Hotel name from booking.com url
        country: "vn" in this case
        sort_by: Sort the reviews by  ['most_relevant', 'newest_first', 'oldest_first', 'highest_scores' or 'lowest_scores']
        n_reviews: -1 means scrape all.
    """

    input_params = {
        "hotel_name": hotel_name,
        "country": country,
        "sort_by": sort_by,
        "n_rows": n_reviews,
    }

    scraper = ReviewScraper(input_params, save_to_disk=False, logger=logger)
    ls_reviews = scraper.run()

    return ls_reviews


def scrape_reviews_multiple_hotels(
    hotels: List[str],
    country: str,
    sort_by: str = "most_relevant",
    n_reviews: int = 100,
    logger: Logger | None = None,
) -> List[dict]:
    """To run the scrapper as module of a list of hotels on Booking.com

    Args:
        hotels: List of hotel names from booking.com url
        country: "vn" in this case
        sort_by: Sort the reviews by  ['most_relevant', 'newest_first', 'oldest_first', 'highest_scores' or 'lowest_scores']
        n_reviews: -1 means scrape all.
    """

    ls_reviews = []

    try:
        for hotel_name in hotels:
            reviews = scrape_reviews_of_hotel(
                hotel_name=hotel_name,
                country=country,
                sort_by=sort_by,
                n_reviews=n_reviews,
                logger=logger,
            )
            ls_reviews.append({"hotel_name": hotel_name, "reviews": reviews})
    except Exception as e:
        logger.error(f"Error: {e}")
    finally:
        return ls_reviews


def get_hotel_list(csv_file: str) -> List[str]:
    """Get a list of hotel names from a csv file"""

    return pd.read_csv(csv_file)["hotel_slug"].tolist()


if __name__ == "__main__":

    dir_path = os.path.dirname(__file__)
    input_path = os.path.join(dir_path, "input", "vn_hotels.csv")
    output_path = os.path.join(dir_path, "output", "vn_hotels_reviews.parquet")

    hotels = get_hotel_list(csv_file=input_path)
    ls_reviews = scrape_reviews_multiple_hotels(hotels, country="vn", n_reviews=500)

    reviews_df = pd.DataFrame(ls_reviews)
    reviews_df.to_parquet(path=output_path, index=False)
