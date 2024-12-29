import os
from typing import Any, Dict, List

import pandas as pd
from dotenv import load_dotenv
from neo4j import GraphDatabase


class HotelGraphDB:
    def __init__(self, uri: str, username: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(username, password))

    def close(self):
        self.driver.close()

    def _create_constraints(self):
        with self.driver.session() as session:
            constraints = [
                "CREATE CONSTRAINT IF NOT EXISTS FOR (h:Hotel) REQUIRE h.hotel_id IS UNIQUE",
                "CREATE CONSTRAINT IF NOT EXISTS FOR (u:User) REQUIRE u.user_id IS UNIQUE",
                "CREATE CONSTRAINT IF NOT EXISTS FOR (r:Review) REQUIRE r.review_id IS UNIQUE",
            ]
            for constraint in constraints:
                session.run(constraint)

    def load_data(
        self, fact_review: pd.DataFrame, dim_hotel: pd.DataFrame, dim_user: pd.DataFrame
    ):
        self._create_constraints()

        with self.driver.session() as session:
            hotels_query = """
            UNWIND $hotels AS hotel
            MERGE (h:Hotel {hotel_id: hotel.hotel_id})
            SET h.name = hotel.name_hotel,
                h.address = hotel.address,
                h.location = hotel.location,
                h.country = hotel.country
            """
            session.run(hotels_query, hotels=dim_hotel.to_dict("records"))

            users_query = """
            UNWIND $users AS user
            MERGE (u:User {user_id: user.user_id})
            SET u.username = user.username,
                u.country = user.user_country
            """
            session.run(users_query, users=dim_user.to_dict("records"))

            reviews_query = """
            UNWIND $reviews AS review
            MATCH (u:User {user_id: review.user_id})
            MATCH (h:Hotel {hotel_id: review.hotel_id})
            MERGE (r:Review {review_id: review.review_id})
            SET r.rating = review.review_rating,
                r.date = datetime(review.review_post_date),
                r.stay_duration = review.stay_duration,
                r.stay_type = review.stay_type
            MERGE (u)-[:WROTE]->(r)
            MERGE (r)-[:ABOUT]->(h)
            """
            session.run(reviews_query, reviews=fact_review.to_dict("records"))

    def query(self, query: str, **kwargs) -> List[Dict[Any, Any]]:
        with self.driver.session() as session:
            results = session.run(query, **kwargs)
            return [dict(record) for record in results]

    def get_hotel_recommendations(
        self, user_id: str, limit: int = 5
    ) -> List[Dict[Any, Any]]:
        with self.driver.session() as session:
            query = """
            MATCH (target_user:User {user_id: $user_id})-[:WROTE]->(r:Review)-[:ABOUT]->(h:Hotel)
            WITH target_user, avg(r.rating) as user_avg_rating
            MATCH (other_user:User)-[:WROTE]->(r1:Review)-[:ABOUT]->(h1:Hotel)
            WHERE other_user <> target_user 
            AND r1.rating >= user_avg_rating
            AND NOT EXISTS((target_user)-[:WROTE]->(:Review)-[:ABOUT]->(h1))
            WITH h1, count(r1) as review_count, avg(r1.rating) as avg_rating
            RETURN h1.hotel_id as hotel_id,
                   h1.name as name,
                   avg_rating,
                   review_count
            ORDER BY avg_rating DESC, review_count DESC
            LIMIT $limit
            """
            results = session.run(query, user_id=user_id, limit=limit)
            return [dict(record) for record in results]

    def get_similar_hotels(self, hotel_id: str, limit: int = 5) -> List[Dict[Any, Any]]:
        with self.driver.session() as session:
            query = """
            MATCH (h:Hotel {hotel_id: $hotel_id})<-[:ABOUT]-(r:Review)<-[:WROTE]-(u:User)
            MATCH (u)-[:WROTE]->(r2:Review)-[:ABOUT]->(other:Hotel)
            WHERE other <> h
            WITH other, count(DISTINCT u) as common_users, 
                 avg(r2.rating) as avg_rating,
                 count(r2) as review_count
            RETURN other.hotel_id as hotel_id,
                   other.name as name,
                   avg_rating,
                   review_count,
                   common_users
            ORDER BY common_users DESC, avg_rating DESC
            LIMIT $limit
            """
            results = session.run(query, hotel_id=hotel_id, limit=limit)
            return [dict(record) for record in results]


if __name__ == "__main__":
    load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))
    graph_db = HotelGraphDB(
        os.getenv("NEO4J_URI"), os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD")
    )

    base_df_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "scraper",
        "booking",
        "output",
        "warehouse",
    )

    fact_review_df = pd.read_parquet(os.path.join(base_df_path, "fact_review.parquet"))
    dim_hotel_df = pd.read_parquet(os.path.join(base_df_path, "dim_hotel.parquet"))
    dim_user_df = pd.read_parquet(os.path.join(base_df_path, "dim_user.parquet"))

    graph_db.load_data(fact_review_df, dim_hotel_df, dim_user_df)

    test_user, test_hotel = (
        "d49aedca2af303f3439c9ddcfaa6c534",
        "d25d0d6a3f0f42b19482c3d6f16b7fbf",
    )
    print("Recommendations for user:", test_user)
    print(graph_db.get_hotel_recommendations(test_user))

    print("\nSimilar hotels to:", test_hotel)
    print(graph_db.get_similar_hotels(test_hotel))

    print("\nSample query:")
    query = """
    MATCH (h:Hotel)
    RETURN h.hotel_id as hotel_id, h.name as name
    LIMIT 5
    """

    print(graph_db.query(query))
