import os

# Neo4j
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
NEO4J_URL = os.getenv("NEO4J_URL", "bolt://localhost:7687")

# Groq
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
