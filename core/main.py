from llama_index.llms.groq import Groq
from recommendation.constants import DATA_FILE_PATH
from recommendation.helpers import read_data

# from recommendation.helpers import create_recommendation_system

llm = Groq(model="llama3-8b-8192")

nodes = read_data(file_path=DATA_FILE_PATH)

print(nodes)

# recommendation_query_engine = create_recommendation_system(nodes=data, llm=llm)

# response = recommendation_query_engine.query(
#     "What are the best places to visit in Paris?"
# )
