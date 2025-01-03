from typing import Any, Dict, List

import pandas as pd
from neo4j import Session

from core.recommendations.base import BaseHotelRecommender


class CypherGraphHotelRecommender(BaseHotelRecommender):
    def __init__(self, uri: str, username: str, password: str):
        """
        Initialize the CypherGraphHotelRecommender.
        
        Args:
            uri (str): URI for the Neo4j database.
            username (str): Username for the Neo4j database.
            password (str): Password for the Neo4j database.
        """
        super().__init__(uri=uri, username=username, password=password)

    def _create_constraints(self) -> None:
        """
        Create constraints in the Neo4j database.
        """
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

    def _create_static_nodes(self, session: Session) -> None:
        """
        Create static nodes in the Neo4j database.
        
        Args:
            session: Neo4j session object.
        """
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
        """
        Load data into the Neo4j database.
        
        Args:
            hotels_df (pd.DataFrame): DataFrame containing hotel data.
            reviews_df (pd.DataFrame): DataFrame containing review data.
            batch_size (int): Batch size for loading data
        """
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
        """
        Recommend hotels to users.
        
        Args:
            amenities (List[str]): List of amenities to filter by.
            stay_type (str): Type of stay to filter by.
            stay_duration (str): Duration of stay to filter by.
            min_rating (float): Minimum rating for hotels.
            limit (int): Number of hotels to return.
            
        Returns:
            List[Dict[str, Any]]: List of recommended hotels.
        """
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
