from nano_llm import Plugin
from semantic_map import annotation_manager
from semantic_map import index_manager

semantic_map_root = "/opt/SemanticMap/maps/"

class MapQuery(Plugin):
    """
    Map search tools for the bot.
    """
    def __init__(self, 
                 json_raw_path = f"{semantic_map_root}annotations/instances_default.json",
                 json_processed_path = f"{semantic_map_root}annotations/processed.json",
                 map_file_path = f"{semantic_map_root}images/turtlebot3_world.pgm",
                 coco_schema_path = f"{semantic_map_root}annotations/coco_schema.txt",
                 map_schema_path = f"{semantic_map_root}annotations/semantic_map_schema.txt",
                 index_file_dir = f"{semantic_map_root}annotations",**kwargs):
        
        super().__init__(outputs=None, threaded=False, **kwargs)

        self.json_raw_path = json_raw_path
        self.json_processed_path = json_processed_path
        self.map_file_path = map_file_path
        self.coco_schema_path = coco_schema_path
        self.map_schema_path = map_schema_path
        self.index_file_dir = index_file_dir

        self.ann_mgr = annotation_manager.AnnotationManager(self.json_raw_path, self.map_file_path, self.coco_schema_path)
        self.ann_mgr.write_json(self.json_processed_path)
        del self.ann_mgr
        self.ind_mgr = index_manager.IndexManager(self.json_processed_path, self.map_schema_path, index_file_dir=self.index_file_dir)    
        self.ind_mgr.push_json_blob()

        # Add tools
        self.add_tool(self.get_location_from_query)

    def get_location_from_query(self, query: str) -> dict:
        """
        Look up a location on a map based on the user's query. Use this tool when you are asked to
        tell where something is located, when you are told to go to a location where something is, or when
        you are, in general, asked to reason about a query in order to provide a location.

        Args:
            query: A string representing a user question or command that requires finding a location on the map.
        
        Returns:
            dict: A dictionary containing the name of location on the map and its coordinates.
            Keys:
                - 'name': The name of the location on the map.
                - 'coordinates': The coordinates of the location on the map.
        """
        query_embedding = self.ind_mgr.embedding_model.embed_text(query)
        #root_pol_docs, sub_pol_docs, pol_scores = self.ind_mgr.polygon_doc_index.find_subindex(query_embedding, subindex = 'strings', search_field='embedding', limit=2)
        root_reg_docs, sub_reg_docs, scores = self.ind_mgr.region_doc_index.find_subindex(query_embedding, subindex = 'strings', search_field='embedding', limit=2)
        name = root_reg_docs[0].name
        polygon_id = root_reg_docs[0].polygon_ids[0]
        coords = self.ind_mgr.polygon_docs[polygon_id].polygon_centroid
        coordinates = ', '.join([str(round(coord)) for coord in coords])
        return {'name': name, 'coordinates': coordinates}
        
    @classmethod
    def type_hints(cls):
        return {
            'json_raw_path': {
                'suggestions': [f"{semantic_map_root}annotations/instances_default.json",]
            },
            'json_processed_path': {
                'suggestions': [f"{semantic_map_root}annotations/processed.json"],
            },
            'map_file_path': {
                'suggestions': [f"{semantic_map_root}images/map.pgm"],
            },
            'coco_schema_path': {
                'suggestions': [f"{semantic_map_root}annotations/coco_schema.txt"],
            },
            'map_schema_path': {
                'suggestions': [f"{semantic_map_root}annotations/semantic_map_schema.txt"],
            },
            'index_file_dir': {
                'suggestions': [f"{semantic_map_root}annotations"],
            },
        }