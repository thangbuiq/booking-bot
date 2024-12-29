import logging

import pandas as pd

from scraper.booking.models.base import BaseModel

logger = logging.getLogger(__name__)


class Hotels(BaseModel):
    def load_data(self, path: str, location: str = "Đà Lạt") -> None:
        try:
            self.data = pd.read_csv(path)
            self.data = self.data[self.data["location"] == location]
            logger.info("Hotel data loaded successfully")
        except Exception as e:
            logger.error(f"Error loading hotel data: {str(e)}")
            raise

    def process_data(self) -> None:
        try:
            self.data["hotel_id"] = self.data.apply(
                lambda x: self.hash_md5(x["hotel_slug"]), axis=1
            )
            self.data = self.data[
                [
                    "hotel_id",
                    "hotel_slug",
                    "name_hotel",
                    "descriptions",
                    "address",
                    "location",
                    "country",
                    "url_hotel",
                ]
            ]
            self.data.reset_index(drop=True, inplace=True)
            logger.info("Hotel data processed successfully")
        except Exception as e:
            logger.error(f"Error processing hotel data: {str(e)}")
            raise
