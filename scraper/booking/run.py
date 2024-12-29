import argparse
import os
from logging import Logger
from typing import List

import pandas as pd

from scraper.booking.utils import ReviewScraper


def scrape_reviews_of_hotel(
    hotel_name: str,
    country: str,
    sort_by: str = "most_relevant",
    n_reviews: int = 100,
    logger: Logger | None = None,
) -> List[dict]:
    """To run the scrapper as module of a single hotel on Booking.com

    Args:
        hotel_name: Hotel name from scraper.booking.com url
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
        hotels: List of hotel names from scraper.booking.com url
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


def get_hotel_list(csv_file: str, filter_location: str = "Đà Lạt") -> List[str]:
    """Get a list of hotel names from a csv file"""

    hotel_dataframes = pd.read_csv(csv_file)
    hotel_dataframes = hotel_dataframes[hotel_dataframes["location"] == filter_location]
    return hotel_dataframes["hotel_slug"].tolist()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape reviews from Booking.com")
    parser.add_argument(
        "--input_csv",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "input", "vn_hotels.csv"),
        help="Path to the input CSV file containing hotel information",
    )
    parser.add_argument(
        "--output_parquet",
        type=str,
        default=os.path.join(
            os.path.dirname(__file__), "output", "vn_hotels_reviews.parquet"
        ),
        help="Path to the output Parquet file to save the reviews",
    )
    parser.add_argument(
        "--filter_location",
        type=str,
        default="Đà Lạt",
        help="Location to filter hotels by",
    )
    parser.add_argument(
        "--country",
        type=str,
        default="vn",
        help="Country code for the hotels",
    )
    parser.add_argument(
        "--n_reviews",
        type=int,
        default=20,
        help="Number of reviews to scrape per hotel",
    )

    args = parser.parse_args()

    hotels = get_hotel_list(
        csv_file=args.input_csv, filter_location=args.filter_location
    )
    ls_reviews = scrape_reviews_multiple_hotels(
        hotels, country=args.country, n_reviews=args.n_reviews
    )

    reviews_df = pd.DataFrame(ls_reviews)
    reviews_df.to_parquet(path=args.output_parquet, index=False)
