import logging
from nano_llm import Plugin
from semantic_map import annotation_manager
from semantic_map import index_manager
from numpy import ndarray
from typing import Union
from PIL import Image

SEMANTIC_MAP_ROOT = "/opt/SemanticMap/maps/"

class MapQuery(Plugin):
    """
    Map search tools for the bot.
    """
    def __init__(self, 
                 json_raw_path = f"{SEMANTIC_MAP_ROOT}annotations/instances_default.json",
                 json_processed_path = f"{SEMANTIC_MAP_ROOT}annotations/processed.json",
                 map_file_path = f"{SEMANTIC_MAP_ROOT}images/map.png",
                 coco_schema_path = f"{SEMANTIC_MAP_ROOT}annotations/coco_schema.txt",
                 map_schema_path = f"{SEMANTIC_MAP_ROOT}annotations/semantic_map_schema.txt",
                 index_file_dir = f"{SEMANTIC_MAP_ROOT}annotations", **kwargs):

        """
        Load Map Query for reasoning about annotated maps.
        
        Args:
            json_raw_path (str): The path to the raw JSON file containing the map annotations.
            json_processed_path (str): The path to the processed JSON file containing the map annotations.
            map_file_path (str): The path to the map file.
            coco_schema_path (str): The path to the COCO schema file.
            map_schema_path (str): The path to the map schema file.
            index_file_dir (str): The directory where the index files are stored.
        """
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

        self.add_parameter('json_raw_path', default=json_raw_path) 
        self.add_parameter('json_processed_path', default=json_processed_path)
        self.add_parameter('map_file_path', default=map_file_path)
        self.add_parameter('coco_schema_path', default=coco_schema_path)
        self.add_parameter('map_schema_path', default=map_schema_path)
        self.add_parameter('index_file_dir', default=index_file_dir)
        
        self.current_location = self.get_your_current_location()

        logging.info(f"Map Query plugin initialized with {self.json_raw_path}")

        # Add tools
        self.add_tool(self.get_location_of_something)
        self.add_tool(self.get_your_current_location)
        self.add_tool(self.go_to_location_on_map)
        self.add_tool(self.generate_path_plan)
        # self.add_tool(self.move_forward)

    def get_your_current_location(self) -> dict:
        """
        Get your current location on the map. Use this tool whenever the user asks where you are located on the map.
        Only use this tool when you are specifically asked to provide YOUR current location. If the user asks for the
        location of something else or instructs you to go to a location, then use one of the other tools provided.

        Returns:
            dict: A dictionary containing the name of the current location on the map and its coordinates.
            Keys:
                - 'name': The name of the current location on the map.
                - 'coordinates': The coordinates of the current location on the map.
        """
        current_name = "Paradise"
        current_coords = "100 101"
        return {'name': current_name, 'coordinates': current_coords}
    
    def get_location_of_something(self, query: str) -> dict:
        """
        Look up a location on a map based on the user's query. Use this tool when you are asked to
        tell where something is located, or when you are, in general, asked to reason about a user's
        query regarding the map in order to provide a location to the user. Don't use this tool if the
        user asks where YOU are or about YOUR location.

        Args:
            query: A string representing a user question or instruction that requires finding a location on the map.
        
        Returns:
            dict: A dictionary containing the name of location on the map and its coordinates.
            Keys:
                - 'name': The name of the location on the map.
                - 'coordinates': The coordinates of the location on the map.
        """
        results_dict = self.text_map_query(query)
        root_reg_docs = results_dict['region docs']['root docs']
        name = root_reg_docs[0].name
        polygon_id = root_reg_docs[0].polygon_ids[0]
        coords = self.ind_mgr.polygon_docs[polygon_id].polygon_centroid
        coordinates = ', '.join([str(round(coord)) for coord in coords])
        
        return {'name': name, 'coordinates': coordinates}
    
    def go_to_location_on_map(self, query: str) -> dict:
        """
        Navigate to a location on the map based on the user's query. Use this tool when the user gives you some
        information and asks you to go to some location on the map based on that information. 

        Args:
            query: A string representing the user instruction to go to a location on the map.
        
        Returns:
            str: A list of coordinates representing the path plan from the starting location to the ending location.
        """
        loc_dict = self.get_location_of_something(query)
        logging.info(f"Going to {loc_dict['name']} at coordinates {loc_dict['coordinates']} on the map")
        return self.get_path(self.current_location['coordinates'], loc_dict['coordinates'])

    def generate_path_plan(self, start: str, end: str) -> str:
        """
        Generate a path plan to navigate from one location to another on the map. Use this tool when the user asks
        you to generate a path plan to navigate from one location to another on the map or asks you more generally 
        about the best way to get from a given start location to a given end location.

        Args:
            start: A string, inferred from user input, representing the starting location on the map.
            end: A string, inferred from user input, representing the ending location on the map.
        
        Returns:
            str: A list of coordinates representing the path plan from the starting location to the ending location.
        """
        start_dict = self.get_location_of_something(start)
        end_dict = self.get_location_of_something(end)
        path_plan = self.get_path(start_dict['coordinates'], end_dict['coordinates'])
        logging.info(f"Path plan generated from {start_dict['name']} at {start_dict['coordinates']} to {end_dict['name']} and {end_dict['coordinates']}")
        return path_plan
    
    def move_forward(self, distance: int) -> str:
        """
        Move forward by a certain distance on the map. Use this tool when the user asks you to move forward, go straight ahead,
        or continue on by a certain distance.

        Args:
            distance: An integer representing the distance to move forward on the map.
        
        Returns:
            str: A string representing the new coordinates after moving forward by the given distance.
        """
        logging.info(f"Moving forward by {distance} units on the map")

        return "100 101"

    def get_path(self, start: str, end: str) -> str:
        """
        NOT A TOOL
        Get a path plan to navigate from one location to another on the map.
        Args:
            start: A string representing the starting location on the map.
            end: A string representing the ending location on the map.
        
        Returns:
            str: A list of coordinates representing the path plan from the starting location to the ending location.
        """
        path_plan = "100 101 102 103"
        logging.info(f"Getting path from {start} to {end}")
        return path_plan

    def text_map_query(self, query: str, limit: int = 3) -> dict:
        """
        NOT A TOOL
        Query the map index for information based on a user's text query.
        Args:
            query: A string representing the user's query.
        
        Returns:
            dict: A dictionary containing the root docs, subindex docs, and scores returned by the search.
            Keys:
                - 'region docs': The region documents turned up by the search.
                - 'polygon docs': The polygon documents turned up by the search.
        """
        results_dict = {}
        query_embedding = self.ind_mgr.embedding_model.embed_text(query)

        root_region_docs, sub_region_docs, region_scores = self.ind_mgr.region_doc_index.find_subindex(query_embedding, subindex = 'strings', search_field='embedding', limit=limit)
        root_polygon_docs, sub_polygon_docs, polygon_scores = self.ind_mgr.polygon_doc_index.find_subindex(query_embedding, subindex = 'strings', search_field='embedding', limit=limit)

        results_dict['region docs'] = {'root docs': root_region_docs, 'subindex docs': sub_region_docs, 'scores': region_scores}
        results_dict['polygon docs'] = {'root docs': root_polygon_docs, 'subindex docs': sub_polygon_docs, 'scores': polygon_scores}

        return results_dict

    def image_map_query(self, image: Union[str, ndarray], limit: int = 3):

        """
        NOT A TOOL
        Query the map index for information based on an image.
        Args:
            image: A string representing the image URL or a numpy array representing the image.
        
        Returns:
            dict: A dictionary containing the root docs, subindex docs, and scores returned by the search.
            Keys:
                - 'region docs': The region documents turned up by the search.
                - 'polygon docs': The polygon documents turned up by the search.
        """
        results_dict = {}

        if isinstance(image, str):
            image = Image.open(image)
        elif isinstance(image, ndarray):
            image = Image.fromarray(image)

        image_embedding = self.ind_mgr.embedding_model.embed_image(image)

        # Remember only polygons have images associated with them
        root_polygon_docs, sub_polygon_docs, polygon_scores = self.ind_mgr.polygon_doc_index.find_subindex(image_embedding, subindex = 'images', search_field='embedding', limit=limit)
        results_dict['polygon docs'] = {'root docs': root_polygon_docs, 'subindex docs': sub_polygon_docs, 'scores': polygon_scores}

        return results_dict

    @classmethod
    def type_hints(cls):
        return {
            'json_raw_path': {
                'display_name': 'Raw JSON path',
                'suggestions': [f"{SEMANTIC_MAP_ROOT}annotations/instances_default.json",]
            },
            'json_processed_path': {
                'display_name': 'Processed JSON path',
                'suggestions': [f"{SEMANTIC_MAP_ROOT}annotations/processed.json"],
            },
            'map_file_path': {
                'display_name': 'Map image file path',
                'suggestions': [f"{SEMANTIC_MAP_ROOT}images/map.pgm"],
            },
            'coco_schema_path': {
                'display_name': 'COCO JSON schema path',
                'suggestions': [f"{SEMANTIC_MAP_ROOT}annotations/coco_schema.txt"],
            },
            'map_schema_path': {
                'display_name': 'Map JSON schema path',
                'suggestions': [f"{SEMANTIC_MAP_ROOT}annotations/semantic_map_schema.txt"],
            },
            'index_file_dir': {
                'display_name': 'DocArray indexes directory',
                'suggestions': [f"{SEMANTIC_MAP_ROOT}annotations"],
            },
        }