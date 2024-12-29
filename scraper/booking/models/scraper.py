from typing import Literal
from typing import Optional

from pydantic import BaseModel
from pydantic import Field
from pydantic import PositiveInt


class ScraperInput(BaseModel):
    country: Literal["vn"]
    hotel_name: str = Field(..., min_length=2)
    sort_by: Optional[
        Literal[
            "most_relevant",
            "newest_first",
            "oldest_first",
            "highest_scores",
            "lowest_scores",
        ]
    ] = "most_relevant"

    n_rows: Optional[int] = -1


class ScraperConfig(BaseModel):
    REQUESTS_PER_SECOND: Optional[PositiveInt] = 10
    HOTEL_REVIEWS_PAGE: str
    MAX_RETIES: Optional[int] = 3
    OUTPUT_DIR: str
