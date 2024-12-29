import os
from datetime import datetime
from typing import Optional

import pandas as pd

from scraper.booking.models.hotels import Hotels
from scraper.booking.models.reviews import Reviews
from scraper.booking.models.users import Users
from scraper.booking.utils import _setup_logger

logger = _setup_logger()


class BaseWarehouse:
    def __init__(self, df: Optional[pd.DataFrame] = None):
        self.data = df
        self.output_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "output/warehouse"
        )

    def _ensure_output_dir(self):
        os.makedirs(self.output_dir, exist_ok=True)

    def _write_output(self, filename: str) -> None:
        try:
            self._ensure_output_dir()
            path = f"{self.output_dir}/{filename}.parquet"
            self.data.to_parquet(path, index=False)
            logger.info(f"Successfully wrote {filename} to {path}")
        except Exception as e:
            logger.error(f"Error writing {filename}: {str(e)}")
            raise


class FactReview(BaseWarehouse):
    def sync(self, reviews_df: pd.DataFrame) -> None:
        try:
            self.data = reviews_df[
                [
                    "review_id",
                    "hotel_id",
                    "user_id",
                    "review_post_date",
                    "review_rating",
                    "review_title",
                    "review_text_full",
                    "review_text_full_annot",
                    "review_text_disliked",
                    "review_text_liked",
                    "stay_duration",
                    "stay_type",
                    "user_country",
                    "room_view",
                ]
            ]
            logger.info("Fact reviews synced successfully")
            self._write_output("fact_review")
        except Exception as e:
            logger.error(f"Error syncing fact reviews: {str(e)}")
            raise


class DimUser(BaseWarehouse):
    def sync(self, users_df: pd.DataFrame) -> None:
        try:
            self.data = users_df.copy()
            logger.info("Dim users synced successfully")
            self._write_output("dim_user")
        except Exception as e:
            logger.error(f"Error syncing dim users: {str(e)}")
            raise


class DimHotel(BaseWarehouse):
    def sync(self, hotels_df: pd.DataFrame) -> None:
        try:
            self.data = hotels_df.copy()
            logger.info("Dim hotels synced successfully")
            self._write_output("dim_hotel")
        except Exception as e:
            logger.error(f"Error syncing dim hotels: {str(e)}")
            raise


if __name__ == "__main__":
    try:
        reviews_model = Reviews()
        hotels_model = Hotels()

        reviews_model.load_data("scraper/booking/output/vn_hotels_reviews.parquet")
        reviews_model.process_data()

        users_model = Users()
        users_model.load_data(reviews_model.data)
        users_model.process_data()

        hotels_model.load_data("scraper/booking/input/vn_hotels.csv")
        hotels_model.process_data()

        fact_reviews = FactReview()
        fact_reviews.sync(reviews_model.data)

        dim_user = DimUser()
        dim_user.sync(users_model.data)

        dim_hotel = DimHotel()
        dim_hotel.sync(hotels_model.data)

        logger.info("Data processing completed successfully")

    except Exception as e:
        logger.error(f"Error in main process: {str(e)}")
        raise
