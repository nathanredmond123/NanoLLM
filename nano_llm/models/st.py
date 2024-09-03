#!/usr/bin/env python3
import time
import logging
import os

import torch
import numpy as np

from accelerate import init_empty_weights

from nano_llm import NanoLLM


class STModel(NanoLLM):
    """
    Sentence Transformers model. Model names must be "sentence-transformers/{model}" or "cross-encoder/{model}".
    """
    def __init__(self, model_path, load=True, init_empty_weights=False, **kwargs):
        """
        Load model from path on disk or HuggingFace repo name.
        Model types are bi-encoder (text and multi-modal/clip) and cross-encoder.
        
        Args:
          
          model_path (str): Path to model on disk or HuggingFace repo name.
          load (bool): Load model on initialization.
          init_empty_weights (bool): Initialize model with empty weights.

        **model_kwargs: Additional optional keyword arguments for either SentenceTransformer or CrossEncoder:
        For detailed kwarg descriptions, 
        see SentenceTransformer at`https://www.sbert.net/docs/package_reference/sentence_transformer/SentenceTransformer.html#id1`
        and CrossEncoder at 'https://www.sbert.net/docs/package_reference/cross_encoder/cross_encoder.html#id1'.
        """
        super(STModel, self).__init__(model_path, **kwargs)
    
        # Parse kwargs
        model_kwargs = kwargs.get('model_kwargs', {})
        tokenizer_kwargs = kwargs.get('tokenizer_kwargs', {})
        config_kwargs = kwargs.get('config_kwargs', {})
        if 'name' in kwargs: del kwargs['name']
        if 'api' in kwargs: del kwargs['api']
        if 'model_kwargs' in kwargs: del kwargs['model_kwargs']
        if 'tokenizer_kwargs' in kwargs: del kwargs['tokenizer_kwargs']
        if 'config_kwargs' in kwargs: del kwargs['config_kwargs']

        self.model_path = model_path
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        if not (st_type := kwargs['st_type']):
            raise ValueError("for Sentence Transformers, st_type must be set to either 'bi-encoder (for SentenceTransformer) or 'cross-encoder' (for CrossEncoder).")
        else:
            self.st_type = st_type
            del kwargs['st_type']

        torch_dtype = model_kwargs.get('torch_dtype', torch.float16)

        if not load:
            return
        
        if init_empty_weights:
            with init_empty_weights():
                if self.st_type == 'bi-encoder':
                    from sentence_transformers import SentenceTransformer
                    self.model = SentenceTransformer(model_path, **kwargs, 
                                                     model_kwargs=model_kwargs, 
                                                     tokenizer_kwargs=tokenizer_kwargs, 
                                                     config_kwargs=config_kwargs,
                                                     ).to(torch_dtype) # model_kwargs['torch_dtype'] not being passed to CLIP models.
                    self.has_embed = True
                else:
                    from sentence_transformers import CrossEncoder
                    self.model = CrossEncoder(model_path, **kwargs, 
                                                     model_kwargs=model_kwargs, 
                                                     tokenizer_kwargs=tokenizer_kwargs, 
                                                     config_kwargs=config_kwargs,
                                                     ).to(torch_dtype)
                    self.has_embed = False
        else:
            if self.st_type == 'bi-encoder':
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer(model_path, **kwargs, 
                                                     model_kwargs=model_kwargs, 
                                                     tokenizer_kwargs=tokenizer_kwargs, 
                                                     config_kwargs=config_kwargs,
                                                     ).to(torch_dtype).to(self.device).eval()
                self.has_embed = True
                
            else:
                from sentence_transformers import CrossEncoder
                self.model = CrossEncoder(model_path, **kwargs, 
                                                     model_kwargs=model_kwargs, 
                                                     tokenizer_kwargs=tokenizer_kwargs, 
                                                     config_kwargs=config_kwargs,
                                                     ).to(torch_dtype).to(self.device).eval()
                self.has_embed = False

        self.config.torch_dtype = next(self.model.parameters()).dtype

    def generate(self, inputs, **generate_kwargs):
        """
        Generate embeddings from input text or input images with bi-encoder, or return pairwise similarity scores
        from cross-encoder.
        For detailed kwarg descriptions, see `https://www.sbert.net/docs/package_reference/sentence_transformer/SentenceTransformer.html#id1`
                                         and 'https://www.sbert.net/docs/package_reference/cross_encoder/cross_encoder.html#id1'.
        Args:
          inputs (str|ndarray): Text or image inputs to the model/

        Returns:
          Text embeddings, image embeddings, or similarity scores.
        """
        if self.st_type == 'bi-encoder':
            try:
                if isinstance(inputs, np.ndarray) or os.path.isfile(inputs): # it's an image array or file path to an image
                    return self.embed_image(inputs, **generate_kwargs)
                else:
                    return self.embed_text(inputs, **generate_kwargs)
            except ValueError as e:
                logging.error(f"Error generating embeddings with bi-encoder, make sure input is str or list[str]: {e}")
                return None
        else:
            try:
                return self.model.predict(inputs, **generate_kwargs)
            except ValueError as e:
                logging.error(f"Error generating similarity score with cross-encoder,  make sure input is list of sentence pair tuples: {e}")
                return None
            
    def embed_text(self, text, **generate_kwargs):
        """
        Embed text using the model.
        """
        return self.model.encode(text, **generate_kwargs)
    
    def embed_image(self, image, **generate_kwargs):
        """
        Embed image using the model.
        """
        return self.model.encode(image, **generate_kwargs)
    
    def config_vision(self, **kwargs):
        print('Vision config not implemented for Sentence Transformer CLIP models')
        return
    
    def init_vision(self, **kwargs):
        print('Vision init not implemented for Sentence Transformer CLIP models')
        return