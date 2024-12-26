ENTITIES_GRAPH_REGEXP_PATTERN = (
    r"^(\w+(?:\s+\w+)*)\s*->\s*([a-zA-Z\s]+?)\s*->\s*(\w+(?:\s+\w+)*)$"
)

ENTITIES_RESPONSE_PATTERN = (
    r'\("entity"\$\$\$\$"(.+?)"\$\$\$\$"(.+?)"\$\$\$\$"(.+?)"\$\$\$\$"(.+?)"\)'
)
RELATIONSHIPS_RESPONSE_PATTERN = r'\("relationship"\$\$\$\$"(.+?)"\$\$\$\$"(.+?)"\$\$\$\$"(.+?)"\$\$\$\$"(.+?)"\$\$\$\$"(.+?)"\$\$\$\$"(.+?)"\)'

TO_BE_CLEANED_RESPONSE = r"^assistant:\s*"

RECOMMENDATION_KG_EXTRACT_TMPL = """
-Goal-
Given a text document, identify entities, their attributes, and relationships that are relevant for making recommendations.
Extract up to {max_knowledge_triplets} entity-relation triplets focusing on characteristics that influence recommendations.

-Steps-
1. Identify all entities, focusing on items, users, and categories. For each entity, extract:
- entity_name: Name of the entity, capitalized
- entity_type: Type (Item, User, Category, Feature, etc.)
- entity_description: Detailed description including preferences, characteristics, and attributes relevant for recommendations
- entity_attributes: Key features that could influence recommendations (price, genre, style, etc.)
Format: ("entity"$$$$<entity_name>$$$$<entity_type>$$$$<entity_description>$$$$<entity_attributes>)

2. Identify meaningful relationships between entities that could drive recommendations:
- source_entity: Source entity name
- target_entity: Target entity name
- relation: Relationship type (likes, similar_to, belongs_to, recommends, etc.)
- relationship_strength: Numerical score (0-1) indicating relationship strength
- relationship_description: Detailed explanation of why these entities are related
- recommendation_features: Specific features that make this relationship relevant for recommendations

Format: ("relationship"$$$$<source_entity>$$$$<target_entity>$$$$<relation>$$$$<relationship_strength>$$$$<relationship_description>$$$$<recommendation_features>)

3. When finished, output all entities and relationships.

-Real Data-
######################
text: {text}
######################
output:"""

GRAPH_NETWORK_HTML_FILEPATH = "assets/graph_network.html"

DATA_FILE_PATH = "../data/booking.parquet"
