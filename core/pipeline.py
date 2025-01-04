import argparse
import json
import logging
from typing import Any, Dict, List

import openai
from llama_index.llms.openai import OpenAI

from core.recommendations.cypher_graph import CypherGraphHotelRecommender
from core.recommendations.graph_rag import GraphRAGHotelRecommender

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RecommendationPipeline")


class RecommendationPipeline:
    def __init__(
        self,
        openai_model: str = "gpt-4o-mini",
        neo4j_uri: str = None,
        neo4j_username: str = None,
        neo4j_password: str = None,
    ):
        """
        Initialize GraphRAG with LLM-driven functionality.

        Args:
            openai_model (str): OpenAI model name. Defaults to "gpt-4o-mini".
            neo4j_uri (str): Neo4j URI.
            neo4j_username (str): Neo4j username.
            neo4j_password (str): Neo4j password.
        """
        self.openai_model = openai_model

        # Initialize the recommender
        self.graph_rag_hotel_recommender = GraphRAGHotelRecommender(
            neo4j_uri, neo4j_username, neo4j_password, openai_model=openai_model
        )

        self.cypher_graph_hotel_recommender = CypherGraphHotelRecommender(
            neo4j_uri,
            neo4j_username,
            neo4j_password,
        )

    def run(self, query: str) -> Dict[str, Any]:
        """
        Run the recommendation pipeline.

        Args:
            query (str): User query.

        Returns:
            Dict[str, Any]: Recommended hotels.
        """
        logger.info("Starting recommendation pipeline.")
        try:
            # First get recommendations from both methods
            cypher_recommendations = self.process_cypher_recommendations(query=query)
            graph_rag_recommendations = self.process_graph_rag_recommendations(
                query=query
            )

            # Then combine using hybrid approach
            final_recommendations = []
            if cypher_recommendations and graph_rag_recommendations:
                final_recommendations = self.process_hybrid_recommendations(
                    cypher_recommendations=cypher_recommendations,
                    graph_rag_recommendations=graph_rag_recommendations,
                    query=query,
                )
            else:
                final_recommendations = (
                    cypher_recommendations or graph_rag_recommendations
                )

            prompt = (
                "Here are the raw recommendations:\n"
                f"Recommendations: {final_recommendations}\n"
                f"User Query: {query}\n\n"
                "Your task is to format the output of the recommendations as per the following template. "
                "Please strictly follow the below format and ensure the content is well-structured, clear, and concise:\n\n"
                "1. **[Hotel Name]**: [Short description of the hotel]. Include key features and amenities such as [specific features mentioned in the query]. Provide a brief evaluation of the hotel and its suitability based on the query.\n"
                "For example, if the user query is 'I am looking for a hotel with air conditioning and TV', the output should be formatted as follows:\n\n"
                "Based on your query for a hotel with air conditioning and TV, I recommend considering the following options:\n\n"
                "1. **Dalat Flower Hotel & Spa**: This luxury hotel features comfortable rooms with air conditioning and TVs, "
                "along with upscale amenities including a spa and fine dining. It's perfect for those looking to indulge while ensuring "
                "all essential amenities are covered.\n"
                "2. **Du Miên Hotel**: Known for its strategic location near attractions, Du Miên Hotel offers essential amenities like "
                "air conditioning and televisions. The comfortable and well-appointed rooms make it a good option for both leisure and "
                "business travelers.\n\n"
                "Make sure to strictly follow the structure and format in your response. Each recommendation should be clearly numbered "
                "with the hotel's name in bold, followed by a concise description of the amenities and features. Ensure proper punctuation "
                "and clarity in the sentences. The response should directly address the user's query while maintaining a professional tone."
                "\nMUST return the total response matches the language of the User Query."
            )

            response = openai.chat.completions.create(
                model=self.openai_model,
                messages=[{"role": "system", "content": prompt}],
            )
            content = response.choices[0].message.content
            return content
        except Exception as e:
            logger.error(f"Error running recommendation pipeline: {e}")
            raise

    def process_cypher_recommendations(self, query: str) -> List[Dict[str, Any]]:
        """
        Process Cypher-based recommendations.

        Args:
            query (str): User query.

        Returns:
            List[Dict[str, Any]]: Recommended hotels.
        """
        logger.info("Processing Cypher recommendations.")

        try:
            # Extract parameters from the query using LLM
            prompt = (
                "You are an AI assistant specializing in hotel recommendations. "
                "Your task is to extract key details from the user query and return them as a structured JSON object. "
                "The details to extract are:\n"
                "- **amenities**: A list of amenities mentioned in the query (e.g., Wifi, Air Conditioning). Ensure each amenity is properly capitalized.\n"
                "- **stay_type**: The type of stay mentioned, if any (e.g., Family, Business, Solo). If not mentioned, set it to None.\n"
                "- **stay_duration**: The duration of the stay, if specified (e.g., Short, Long, Weekend). If not mentioned, set it to None.\n\n"
                f"User Query: {query}\n\n"
                "Provide the output strictly as a JSON object in the following format:\n"
                "{'amenities': [<list of amenities>], 'stay_type': <stay type>, 'stay_duration': <stay duration>}\n\n"
                "Important:\n"
                "1. Ensure the JSON keys are always lowercase (e.g., 'amenities', 'stay_type', 'stay_duration').\n"
                "2. Capitalize the values appropriately (e.g., 'Wifi', 'Air Conditioning').\n"
                "3. If a parameter is missing or not mentioned, set it to None.\n"
                "4. The response must be a valid JSON object without additional text or explanations.\n\n"
                "Example:\n"
                "For the user query: 'I need a hotel with Wifi and Air Conditioning for a short stay', the output should be:\n"
                '{"amenities": ["Wifi", "Air Conditioning"], "stay_type": None, "stay_duration": "Short"}\n\n'
                "I expect the property names enclosed in double quotes."
                "Now, extract the details and provide the output in the specified JSON format."
            )

            response = openai.chat.completions.create(
                model=self.openai_model,
                messages=[{"role": "system", "content": prompt}],
            )
            content = response.choices[0].message.content.strip()

            # Clean the response
            if content.startswith("```") and content.endswith("```"):
                content = content.strip("```").strip("json").strip()
            cleaned_content = content.replace("None", "null")

            # Extract the parameters
            response_json = json.loads(cleaned_content)
            amenities = response_json.get("amenities", [])
            stay_type = response_json.get("stay_type", "")
            stay_duration = response_json.get("stay_duration", "")

            recommended_hotels = self.cypher_graph_hotel_recommender.recommend_hotels(
                amenities=amenities, stay_type=stay_type, stay_duration=stay_duration
            )

            self.cypher_graph_hotel_recommender.close()
            return recommended_hotels
        except Exception as e:
            logger.error(f"Error in Cypher recommendations: {e}")
            return []

    def process_graph_rag_recommendations(self, query: str) -> List[Dict[str, Any]]:
        """
        Process recommendations using the Graph-based RAG method.

        Params:
            - query: Natural language query.

        Returns:
            List[Dict[str, Any]]: Structured element instances.
        """
        logger.info("Processing Graph RAG recommendations.")

        try:
            recommended_hotels = self.graph_rag_hotel_recommender.recommend_hotels(
                query=query
            )
            self.graph_rag_hotel_recommender.close()
            return recommended_hotels
        except Exception as e:
            logger.error(f"Error in Graph RAG recommendations: {e}")
            return []

    def process_hybrid_recommendations(
        self,
        cypher_recommendations: List[Dict[str, Any]],
        graph_rag_recommendations: List[Dict[str, Any]],
        query: str,
    ) -> List[Dict[str, Any]]:
        """
        Combine recommendations using a hybrid approach.

        Args:
            cypher_recommendations (List[Dict[str, Any]]): Cypher-based recommendations.
            graph_rag_recommendations (List[Dict[str, Any]]): Graph RAG-based recommendations.
            query (str): User query.

        Returns:
            List[Dict[str, Any]]: Combined recommendations.
        """
        logger.info("Combining recommendations using hybrid approach.")
        try:
            # Combine the recommendations
            combined = []
            if cypher_recommendations:
                combined.append(cypher_recommendations)

            if graph_rag_recommendations:
                combined.append(graph_rag_recommendations)

            # Prepare data for LLM-based reranking
            prompt = (
                "You are an AI assistant specializing in hotel recommendations. "
                "Given the user query and the list of five hotel recommendations, your task is to rank them "
                "from most to least relevant based on the user's preferences. Please pay close attention to the following steps:\n\n"
                "1. Carefully analyze the user's query to understand the key preferences and requirements. Focus on specific amenities, "
                "location, price range, or any other details mentioned in the query.\n"
                "2. For each hotel recommendation, evaluate how well it aligns with the user's stated preferences. "
                "Consider factors such as the presence of requested amenities (e.g., air conditioning, TV, proximity to landmarks, etc.), "
                "hotel features, and overall suitability based on the user's needs.\n"
                "3. Rank the recommendations from most to least relevant. The most relevant recommendation should be the one that best "
                "meets the user's needs and preferences, while the least relevant recommendation should be the one that least aligns with "
                "those needs.\n\n"
                f"User Query: {query}\n\n"
                "Recommendations:\n"
                "1. **[Hotel Name]**: [Brief description of the hotel and how it fits the user's preferences].\n"
                "2. **[Hotel Name]**: [Brief description of the hotel and how it fits the user's preferences].\n"
                "3. **[Hotel Name]**: [Brief description of the hotel and how it fits the user's preferences].\n"
                "4. **[Hotel Name]**: [Brief description of the hotel and how it fits the user's preferences].\n"
                "5. **[Hotel Name]**: [Brief description of the hotel and how it fits the user's preferences].\n\n"
                "For each recommendation, consider the following when ranking:\n"
                "- **Amenity match**: How well the hotel matches the specific amenities mentioned in the query (e.g., air conditioning, TV, etc.).\n"
                "- **Location relevance**: How relevant the hotel's location is to the user's query, if applicable (e.g., proximity to tourist attractions, business areas, etc.).\n"
                "- **Suitability**: How well the hotel meets the user's overall needs (e.g., luxury, budget, family-friendly, etc.).\n"
                "- **User preferences**: Any other preferences stated in the query (e.g., preferred hotel type, specific features).\n\n"
                "Finally, return the recommendations in the following order, starting with the most relevant and ending with the least relevant."
            )

            for i, rec in enumerate(combined, 1):
                prompt += f"{i}. {rec}\n"

            prompt += "\nReturn the ranked recommendations as a numbered list."

            # Use the LLM to rerank
            llm = OpenAI(self.openai_model)
            response = llm.complete(prompt).text.strip()

            # Parse the response into a ranked list
            ranked_hotels = []
            for line in response.split("\n"):
                line = line.strip()
                if line and line[0].isdigit():
                    try:
                        index = int(line.split(".")[0]) - 1
                        if 0 <= index < len(combined):
                            ranked_hotels.append(combined[index])
                    except (ValueError, IndexError):
                        continue

            return ranked_hotels or combined
        except Exception as e:
            logger.error(f"Error in hybrid recommendations: {e}")
            return []


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get hotel recommendations.")
    parser.add_argument(
        "-m",
        "--message",
        type=str,
        required=False,
        help="Message for hotel recommendations",
        default="I am looking for a hotel with a air conditioning and TV.",
    )

    args = parser.parse_args()

    pipeline = RecommendationPipeline(
        neo4j_uri="bolt://localhost:7687",
        neo4j_username="neo4j",
        neo4j_password="password",
    )
    query = args.message

    recommendations = pipeline.run(query)
    print(recommendations)
