import re
from typing import Any
from typing import List
from typing import Tuple

from recommendation.constants import ENTITIES_RESPONSE_PATTERN
from recommendation.constants import RELATIONSHIPS_RESPONSE_PATTERN


def parse_fn(response_str: str) -> Tuple[List[Any], List[Any]]:
    """
    Parse the response from the language model.

    Args:
        response_str (str): The response from the language model.

    Returns:
        Tuple[List[Any], List[Any]]: A tuple of entities and relationships.
    """
    entities = re.findall(ENTITIES_RESPONSE_PATTERN, response_str)
    relationships = re.findall(RELATIONSHIPS_RESPONSE_PATTERN, response_str)

    return entities, relationships
