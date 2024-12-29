import logging

import pandas as pd

from scraper.booking.models.base import BaseModel

logger = logging.getLogger(__name__)


class Users(BaseModel):
    def load_data(self, reviews_df: pd.DataFrame) -> None:
        try:
            self.data = reviews_df[["username", "user_country"]].drop_duplicates()
            logger.info("User data loaded successfully")
        except Exception as e:
            logger.error(f"Error loading user data: {str(e)}")
            raise

    def process_data(self) -> None:
        try:
            self.data["user_id"] = self.data.apply(
                lambda x: self.hash_md5(str(x["username"])), axis=1
            )
            self.data = self.data[["user_id", "username", "user_country"]]
            self.data.reset_index(drop=True, inplace=True)
            logger.info("User data processed successfully")
        except Exception as e:
            logger.error(f"Error processing user data: {str(e)}")
            raise
