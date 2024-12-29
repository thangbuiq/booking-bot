import logging

import pandas as pd

from scraper.booking.models.base import BaseModel

logger = logging.getLogger(__name__)


class Reviews(BaseModel):
    def load_data(self, path: str) -> None:
        try:
            self.data = pd.read_parquet(path)
            self.data = self.data.explode("reviews")
            self.data = pd.concat(
                [
                    self.data.drop(["reviews"], axis=1),
                    self.data["reviews"].apply(pd.Series),
                ],
                axis=1,
            )
            logger.info("Review data loaded successfully")
        except Exception as e:
            logger.error(f"Error loading review data: {str(e)}")
            raise

    def process_data(self) -> None:
        try:
            self._rename_columns()
            self._generate_ids()
            self._process_review_text()
            self._clean_data_types()
            self._reorder_columns()
            self.data.reset_index(drop=True, inplace=True)
            logger.info("Review data processed successfully")
        except Exception as e:
            logger.error(f"Error processing review data: {str(e)}")
            raise

    def _rename_columns(self) -> None:
        self.data = self.data.rename(
            columns={
                "hotel_name": "hotel_slug",
                "full_review": "review_text_full_annot",
                "rating": "review_rating",
            }
        )

    def _generate_ids(self) -> None:
        self.data["review_id"] = self.data.apply(
            lambda x: self.hash_md5(str(x["hotel_slug"]) + str(x["username"])), axis=1
        )
        self.data["hotel_id"] = self.data.apply(
            lambda x: self.hash_md5(x["hotel_slug"]), axis=1
        )
        self.data["user_id"] = self.data.apply(
            lambda x: self.hash_md5(str(x["username"])), axis=1
        )

    def _process_review_text(self) -> None:
        self.data["review_text_full"] = (
            self.data["review_title"].fillna("")
            + ". "
            + self.data["review_text_liked"].fillna("")
            + ". "
            + self.data["review_text_disliked"].fillna("")
        )

    def _clean_data_types(self) -> None:
        self.data["stay_duration"] = self.data["stay_duration"].str.extract(r"(\d+)")
        self.data["stay_duration"] = (
            pd.to_numeric(self.data["stay_duration"], errors="coerce")
            .fillna(0)
            .astype(int)
        )
        self.data["review_post_date"] = pd.to_datetime(
            self.data["review_post_date"], format="%m-%d-%Y %H:%M:%S"
        )
        self.data["review_rating"] = pd.to_numeric(
            self.data["review_rating"], errors="coerce"
        ).astype(float)

    def _reorder_columns(self) -> None:
        self.data = self.data[
            [
                "review_id",
                "hotel_id",
                "user_id",
                "username",
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
