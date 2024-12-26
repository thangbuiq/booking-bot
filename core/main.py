from llama_index.llms.openai import OpenAI

from core.recommendation.constants import DATA_FILE_PATH
from core.recommendation.helpers import create_recommendation_system
from core.recommendation.helpers import read_data

llm = OpenAI(model="gpt-4o-mini")

data = read_data(file_path=DATA_FILE_PATH)

recommendation_query_engine = create_recommendation_system(nodes=data, llm=llm)

response = recommendation_query_engine.query(
    "What are the best places to visit in Paris?"
)
