import argparse
import os
from typing import Any, Dict, List

import pandas as pd
from dotenv import load_dotenv
from neo4j import GraphDatabase


class HotelRecommender:
    def __init__(self, uri: str, username: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(username, password))

    def close(self) -> None:
        self.driver.close()

    def _create_constraints(self) -> None:
        with self.driver.session() as session:
            constraints = [
                "CREATE CONSTRAINT IF NOT EXISTS FOR (h:Hotel) REQUIRE h.hotel_id IS UNIQUE",
                "CREATE CONSTRAINT IF NOT EXISTS FOR (u:User) REQUIRE u.user_id IS UNIQUE",
                "CREATE CONSTRAINT IF NOT EXISTS FOR (a:Amenity) REQUIRE a.name IS UNIQUE",
                "CREATE CONSTRAINT IF NOT EXISTS FOR (d:StayDuration) REQUIRE d.duration IS UNIQUE",
                "CREATE CONSTRAINT IF NOT EXISTS FOR (t:StayType) REQUIRE t.type IS UNIQUE",
            ]
            for constraint in constraints:
                session.run(constraint)

    def _create_static_nodes(self, session) -> None:
        amenities = [
            "Air Conditioning",
            "TV",
            "Balcony",
            "Food Service",
            "Parking",
            "Vehicle Hire",
        ]
        query = "UNWIND $amenities AS name MERGE (a:Amenity {name: name})"
        session.run(query, amenities=amenities)

        durations = ["Short", "Medium", "Long"]
        query = (
            "UNWIND $durations AS duration MERGE (d:StayDuration {duration: duration})"
        )
        session.run(query, durations=durations)

        types = ["Couple", "Family", "Group", "Solo traveller"]
        query = "UNWIND $types AS type MERGE (t:StayType {type: type})"
        session.run(query, types=types)

    def load_data(
        self,
        hotels_df: pd.DataFrame,
        reviews_df: pd.DataFrame,
        batch_size: int = 1000,
    ) -> None:
        self._create_constraints()

        reviews_df = reviews_df.dropna(subset=["stay_duration", "stay_type"])
        reviews_df = reviews_df[
            reviews_df["stay_duration"].isin(["Short", "Medium", "Long"])
        ]
        reviews_df = reviews_df[
            reviews_df["stay_type"].isin(
                ["Couple", "Family", "Group", "Solo traveller"]
            )
        ]

        with self.driver.session() as session:
            self._create_static_nodes(session)

            # Create hotels and their amenity relationships
            hotels_query = """
            UNWIND $hotels AS hotel
            MERGE (h:Hotel {hotel_id: hotel.hotel_id})
            SET h.name = hotel.name_hotel,
                h.description = hotel.descriptions,
                h.url = hotel.url_hotel,
                h.address = hotel.address
            WITH h, hotel
            UNWIND [
                {name: 'Air Conditioning', has: hotel.has_air_conditioning},
                {name: 'TV', has: hotel.has_tv},
                {name: 'Balcony', has: hotel.has_balcony},
                {name: 'Food Service', has: hotel.has_food_serving},
                {name: 'Parking', has: hotel.has_parking},
                {name: 'Vehicle Hire', has: hotel.has_hire_vehicle}
            ] AS amenity
            MATCH (a:Amenity {name: amenity.name})
            WHERE amenity.has = true
            MERGE (h)-[:HAS_AMENITY]->(a)
            """

            for i in range(0, len(hotels_df), batch_size):
                batch = hotels_df.iloc[i : i + batch_size]
                session.run(hotels_query, hotels=batch.to_dict("records"))

            # Create reviews with stay duration and type relationships
            reviews_query = """
            UNWIND $reviews AS review
            MATCH (h:Hotel {hotel_id: review.hotel_id})
            MERGE (u:User {user_id: review.user_id})
            SET u.username = review.username
            MERGE (d:StayDuration {duration: review.stay_duration})
            MERGE (t:StayType {type: review.stay_type})
            CREATE (r:Review {
                rating: review.review_rating,
                title: review.review_title,
                text: review.review_text_full,
                room_view: review.room_view
            })
            MERGE (u)-[:REVIEWED]->(r)
            MERGE (r)-[:REVIEWED]->(h)
            MERGE (r)-[:HAS_DURATION]->(d)
            MERGE (r)-[:HAS_TYPE]->(t)
            """

            for i in range(0, len(reviews_df), batch_size):
                batch = reviews_df.iloc[i : i + batch_size]
                session.run(reviews_query, reviews=batch.to_dict("records"))

    def recommend_hotels(
        self,
        amenities: List[str] = None,
        stay_type: str = None,
        stay_duration: str = None,
        min_rating: float = 5.0,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        matches = ["MATCH (h:Hotel)"]

        if amenities:
            for amenity in amenities:
                matches.append(
                    f"MATCH (h)-[:HAS_AMENITY]->(:Amenity {{name: '{amenity}'}})"
                )

        if stay_type:
            matches.append("MATCH (r)-[:HAS_TYPE]->(:StayType {type: $stay_type})")

        if stay_duration:
            matches.append(
                "MATCH (r)-[:HAS_DURATION]->(:StayDuration {duration: $stay_duration})"
            )

        query = f"""
        {' '.join(matches)}
        MATCH (r:Review)-[:REVIEWED]->(h)
        WITH h, avg(r.rating) as avg_rating, count(r) as review_count
        WHERE avg_rating IS NULL OR avg_rating >= $min_rating
        RETURN h.hotel_id as hotel_id,
            h.name as name,
            h.description as description,
            h.address as address,
            coalesce(avg_rating, 0) as avg_rating,
            review_count,
            coalesce(avg_rating * 0.7 + log(review_count + 1) * 0.3, 0) as score
        ORDER BY score DESC
        LIMIT $limit
        """

        with self.driver.session() as session:
            results = session.run(
                query,
                stay_type=stay_type,
                stay_duration=stay_duration,
                min_rating=min_rating,
                limit=limit,
            )
            return [dict(record) for record in results]


def main():
    parser = argparse.ArgumentParser(description="Hotel Recommender CLI")
    parser.add_argument("--load", action="store_true", help="Load data into Neo4j")
    parser.add_argument(
        "--hotels",
        type=str,
        help="Path to hotels parquet file",
        default="data/vn_hotels.parquet",
    )
    parser.add_argument(
        "--reviews",
        type=str,
        help="Path to reviews parquet file",
        default="data/vn_hotels_reviews.parquet",
    )
    args = parser.parse_args()

    load_dotenv()

    recommender = HotelRecommender(
        os.getenv("NEO4J_URI"), os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD")
    )

    if args.load and args.hotels and args.reviews:
        hotels_df = pd.read_parquet(args.hotels)
        reviews_df = pd.read_parquet(args.reviews)
        recommender.load_data(hotels_df, reviews_df)
        print("Data successfully loaded into Neo4j.")

    results = recommender.recommend_hotels(
        amenities=["Parking", "TV"], stay_type="Family", stay_duration="Long"
    )
    for result in results:
        print(result)

    recommender.close()


if __name__ == "__main__":
    main()
