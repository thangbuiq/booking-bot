import hashlib
import logging
from abc import ABC
from abc import abstractmethod
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


class BaseModel(ABC):
    def __init__(self):
        self.data: Optional[pd.DataFrame] = None

    @staticmethod
    def hash_md5(text: str) -> str:
        return str(hashlib.md5(str(text).encode()).hexdigest())

    @abstractmethod
    def load_data(self, path: str) -> None:
        pass

    @abstractmethod
    def process_data(self) -> None:
        pass

    def save_data(self, path: str) -> None:
        try:
            self.data.to_parquet(path)
            logger.info(f"Data saved successfully to {path}")
        except Exception as e:
            logger.error(f"Error saving data: {str(e)}")
            raise
