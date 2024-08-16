#!/usr/bin/env python3
import os
import re
import time
import json
import glob
import shutil
import logging

import torch
import numpy as np

from transformers import AutoTokenizer

from .vision import CLIPVisionModel, TIMMVisionModel, MMProjector
from .utils import AttributeDict, convert_tensor, download_model, default_model_api, rename_weights, filter_keys, print_table


class NanoLLM():
    """
    LLM interface that model APIs implement, including:
    
      * :func:`generate` for token generation
      * :func:`tokenize` and :func:`detokenize`
      * :func:`embed_text`, :func:`embed_tokens`, and :func:`embed_image`
      
    The static method :func:`from_pretrained` will load the model using the specified API.
    """
    ModelCache={}
    
    @staticmethod
    def from_pretrained(model, api=None, use_cache=False, **kwargs):
        """
        Load a model from the given path or download it from HuggingFace Hub.
        Various inference and quantization APIs are supported, such as MLC and AWQ.
        If the API isn't explicitly specified, it will be inferred from the type of model.
        
        Base class for local LLM APIs. It defines common Huggingface-like interfaces for
        model loading, text generation, tokenization, embeddings, and streaming.
        It also supports multimodal vision models like Llava and generating image embeddings with CLIP.
    
        Args:
          model (str): either the path to the model, or HuggingFace model repo/name.
          api (str): the model backend API to use:  'auto_gptq', 'awq', 'mlc', 'hf', or 'st'
                       if left as None, it will attempt to be automatically determined.

          quantization (str): for AWQ or MLC, either specify the quantization method,
                              or the path to the quantized model (AWQ and MLC API's only)

          vision_model (str): for VLMs, override the vision embedding model 
                              (typically `openai/clip-vit-large-patch14-336 <https://huggingface.co/openai/clip-vit-large-patch14-336>`_).
                              Otherwise, it will use the CLIP variant from the config.
          
          st_type (str): for Sentence Transformers, the model type: 'bi-encoder' or 'cross-encoder'

          model_kwargs, tokenizer_kwargs, config_kwargs: for Sentence Transformers, additional kwargs dictionares for the model, tokenizer,
                              and config. See `SentenceTransformer <https://www.sbert.net/docs/package_reference/sentence_transformer/SentenceTransformer.html#id1>`_.
                              and `CrossEncoder <https://www.sbert.net/docs/package_reference/cross_encoder/cross_encoder.html#id1>`.
                                
        Returns:
          A loaded `NanoLLM` model instance using the determined API.
        """
        if use_cache:
            model_config = frozenset({'model': model, 'api': api, **kwargs}.items())
            cached_model = NanoLLM.ModelCache.get(model_config)
            if cached_model is not None:
                return cached_model
                
        if os.path.isdir(model) or os.path.isfile(model):
            model_path = model
            model_name = os.path.basename(model_path)
        else:
            model_path = download_model(model, **kwargs)
            model_name = os.path.basename(model)
            
        if not api:
            api = default_model_api(model_path, kwargs.get('quantization'))
        
        api = api.lower()
        
        kwargs['name'] = model_name
        kwargs['api'] = api
        
        logging.info(f"loading {model_path} with {api.upper()}")
        load_begin = time.perf_counter()
        
        # doing this imports here avoid circular import, and makes it so these
        # dependencies are only needed if they are actually used to load a model
        if api == 'auto_gptq':
            from nano_llm.models import AutoGPTQModel
            model = AutoGPTQModel(model_path, **kwargs)
        elif api == 'awq':
            from nano_llm.models import AWQModel
            model = AWQModel(model_path, **kwargs)
        elif api == 'mlc':
            from nano_llm.models import MLCModel
            model = MLCModel(model_path, **kwargs)
        elif api == 'hf':
            from nano_llm.models import HFModel
            model = HFModel(model_path, **kwargs)
        elif api == 'st':
            from nano_llm.models import STModel
            model = STModel(model_path, **kwargs)
        else:
            raise ValueError(f"invalid API: {api}")

        # moved CLIP to after LLM is loaded because of MLC CUDA errors when running in subprocess
        model.init_vision(**kwargs)  
        model.restore_config()
        model.config.load_time = time.perf_counter() - load_begin

        print_table(model.config)
        print('')
        
        if use_cache:
            NanoLLM.ModelCache[model_config] = model
            
        return model
     
    def generate(self, inputs, streaming=True, **kwargs):
        """
        Generate output from input text, tokens, or an embedding.
        For detailed kwarg descriptions, see `transformers.GenerationConfig <https://huggingface.co/docs/transformers/main/en/main_classes/text_generation#transformers.GenerationConfig>`_.
        
        Args:
        
          inputs (str|ndarray): Text or embedding inputs to the model/
          
          streaming (bool): If True, an iterator will be returned that returns text chunks.
                            Otherwise, this function will block and return the generated text.
                              
          functions(list[callable]): Dynamic functions or plugins to run inline with token generation 
                                     for things like function calling, guidance, token healing, ect.
                                     These will be passed the text generated by the LLM so far, and any
                                     additional text that these return will be added to the chat.

          max_new_tokens (int): The number of tokens to output in addition to the prompt (default: 128)
          min_new_tokens (int): Force the model to generate a set number of output tokens (default: -1)
          detokenize (bool): If ``True`` (the default), text will be returned (otherwise ``list[int]`` of token ID's)
          do_sample (bool): If ``True``, temperature/top_p will be used.  Otherwise, greedy search (default: ``False``)
          repetition_penalty: The parameter for repetition penalty. 1.0 means no penalty (default: 1.0)
          temperature (float): Randomness token sampling parameter (default=0.7, only used if ``do_sample=True``)
          top_p (float): If set to float < 1 and ``do_sample=True``, only the smallest set of most probable tokens.
                           with probabilities that add up to top_p or higher are kept for generation (default 0.95)
          stop_tokens (list[int]|list[str]): Stop generation if the bot produces tokens or text from this list (defaults to EOS token ID)
          kv_cache (np.ndarray): Previous kv_cache that the inputs will be appended to.  By default, a blank kv_cache 
                                will be created for each generation (i.e. a new chat).  This generation's kv_cache
                                will be set in the returned :class:`StreamingResponse` iterator after the request is complete.

        Returns:
          An asynchronous :class:`StreamingResponse` iterator (when ``streaming=True``) that outputs one decoded token or string at a time.
          Otherwise, this function blocks and a string (or ``list[int]`` if ``detokenize=False``) containing the full reply is returned after it's been completed.
        """
        raise NotImplementedError("use LLM.from_pretrained() as opposed to instantiating an LLM object directly")

    def tokenize(self, text, add_special_tokens=False, dtype=np.int32, return_tensors='np', **kwargs):
        """
        Tokenize the given string and return the encoded token ID's.
        
        Args:
          text (str): the text to tokenize.
          add_special_tokens (str): if BOS/EOS tokens (like ``<s>`` or ``<|endoftext|>``) should automatically be added (default False)
          dtype (type): the numpy or torch datatype of the tensor to return.
          return_tensors (str): ``'np'`` to return a `np.ndarray` or ``'pt'`` to return a `torch.Tensor`
          kwargs:  additional arguments forwarded to the HuggingFace `transformers.AutoTokenizer <https://huggingface.co/docs/transformers/main/en/model_doc/auto#transformers.AutoTokenizer>`_ encode function.
          
        Returns:
          The token ID's with the tensor type as indicated by `return_tensors` (either `'np'` for `np.ndarray`
          or `'pt'` for `torch.Tensor`) and datatype as indicated by `dtype` (by default ``int32``)
        """
        if return_tensors == 'tvm':
            return_tensors = 'np'
  
        tokens = self.tokenizer(
            text, 
            add_special_tokens=add_special_tokens, 
            return_tensors=return_tensors,
            **kwargs
        ).input_ids

        return convert_tensor(tokens, return_tensors=return_tensors, dtype=dtype)

    def detokenize(self, tokens, skip_special_tokens=False, **kwargs) -> str:
        """
        Detokenize the given token ID's and return the decoded string.
        
        Args:
          tokens (list[int], np.ndarray, torch.Tensor): the array of token ID's
          skip_special_tokens (bool): if special tokens (like BOS/EOS) should be supressed from the output or not (default false)
          kwargs:  additional arguments forwarded to the HuggingFace `transformers.AutoTokenizer <https://huggingface.co/docs/transformers/main/en/model_doc/auto#transformers.AutoTokenizer>`_ decode function.
          
        Returns:
          The string containing the decoded text.
        """
        return self.tokenizer.decode(tokens, skip_special_tokens=skip_special_tokens, **kwargs)
        
    def embed_text(self, text, add_special_tokens=False, use_cache=False, return_tensors='np', return_tokens=False, **kwargs):
        """
        Tokenize the string with :meth:`NanoLLM.tokenize` and return its embedding as computed by :meth:`NanoLLM.embed_tokens`.
        Note that if ``model.has_embed=False``, then None will be returned for the embedding and the tokens should be used instead.
        
        Args:
          text (str): the text to tokenize and embed.
          add_special_tokens (str): if BOS/EOS tokens (like ``<s>``, ``<|endoftext|>``) should automatically be added (default False)
          use_cache (bool): if True, the text embedding will be cached and returned without additional computation if
                            the same string was already embedded previously.  This is useful for things like the system prompt
                            that are relatively static, but probably shouldn't be used for dynamic user inputs that are unlikely
                            to be re-used again (leading to unnecessarily increased memory usage).  The default is false.
          return_tensors (str): ``'np'`` to return a `np.ndarray` or ``'pt'`` to return a `torch.Tensor`
          return_tokens (bool): if True, then the tokens will also be returned in addition to the embedding.
          kwargs:  additional arguments forwarded to :meth:`NanoLLM.tokenize` and the HuggingFace `transformers.AutoTokenizer <https://huggingface.co/docs/transformers/main/en/model_doc/auto#transformers.AutoTokenizer>`_ 
          
        Returns:
          The embedding with the tensor type as indicated by `return_tensors` (either `'np'` for `np.ndarray`
          or `'pt'` for `torch.Tensor`) with ``float32`` data.  If ``return_tokens=True``, then an (embedding, tokens)
          tuple will be returned instead of only the embeddings. If ``model.has_embed=False`, then the embedding will be None.
        """
        result = None

        if use_cache:
            result = self.embed_cache.get(text)
            
        if result is None:
            tokens = self.tokenize(text, add_special_tokens=add_special_tokens, return_tensors=return_tensors, **kwargs)
            embed = self.embed_tokens(tokens, return_tensors=return_tensors) if self.has_embed else None
            result = (embed, tokens)
            
            if use_cache:
                self.embed_cache[text] = result
                
            #print(f'NanoLLM text:   `{text}`'.replace('\n', '\\n'))
            #print(f'NanoLLM tokens: {convert_tensor(tokens, return_tensors=list)}'.replace('\n', '\\n'))
        else:
            logging.debug(f'text embedding cache hit `{text}`'.replace('\n', '\\n'))

        if return_tokens:
            return result
        else:
            return result[0]

    def embed_tokens(self, tokens, return_tensors='np', **kwargs):
        """
        Compute the token embedding and return its tensor.  This will raise an exception if ``model.has_embed=False``.
        
        Args:
          tokens (list[int], np.ndarray, torch.Tensor): the array of token ID's
          return_tensors (str): ``'np'`` to return a `np.ndarray` or ``'pt'`` to return a `torch.Tensor`
          
        Returns:
          The embedding with the tensor type as indicated by `return_tensors` (either `'np'` for `np.ndarray`
          or `'pt'` for `torch.Tensor`) with ``float32`` data.
        """
        raise NotImplementedError("embed_tokens() not implemented for this model")
       
    def embed_image(self, image, return_tensors='pt', **kwargs):
        """
        Compute the embedding of an image (for multimodel models with a vision encoder like CLIP),
        and apply any additional projection layers as specified by the model.
        
        Args:
          image (pil.Image, np.ndarray, torch.Tensor, jetson.utils.cudaImage, __cuda_array_interface__): the image
          return_tensors (str): ``'np'`` to return a `np.ndarray` or ``'pt'`` to return a `torch.Tensor` (on the GPU)
          return_dict (bool): if true, return a dict including the vision encoder's `hidden_state` and `embedding`
          kwargs: additional arguments forwarded to the vision encoder (`nano_llm.vision.CLIPImageEmbedding`)
        
        Returns:
          The embedding with the tensor type as indicated by `return_tensors` (either `'np'` for `np.ndarray`
          or `'pt'` for `torch.Tensor`), or a dict containing the embedding and vision encoder's `hidden_state`
          if ``return_dict=True``.
        """  
        assert(self.has_vision)

        embeddings = []
        
        for i, vision in enumerate(self.vision):
            embedding = vision(
                image, #embeddings[i-1] if i > 0 else image,
                hidden_state = self.config.get('mm_vision_select_layer')
            ).to(dtype=torch.float16)
            
            if 'clip' in vision.config.name.lower(): # if not 'mm_projector_cfg' in self.config
                embedding = embedding[:, 1:]
            
            embeddings.append(embedding)

        embedding = torch.cat(embeddings, dim=2)
        embedding = self.mm_projector(embedding)

        logging.debug(f"image embedding  shape={embedding.shape}  dtype={embedding.dtype}  device={embedding.device}")
        
        if False: #return_dict:
            output.embedding = embedding
            for key in output:
                output[key] = convert_tensor(output[key], return_tensors=return_tensors)
            return output
        else:
            return convert_tensor(embedding, return_tensors=return_tensors)
        
    def __init__(self, model_path, **kwargs):
        #: HuggingFace `transformers.AutoTokenizer <https://huggingface.co/docs/transformers/main/en/model_doc/auto#transformers.AutoTokenizer>`_ instance used for tokenization/detokenization.
        self.tokenizer = None

        #: Dict containing the model configuration (inspect it on the HuggingFace model card)
        self.config = AttributeDict()
        
        #: The local path to the model config file (``config.json``)
        #: Sometimes the config.json is one directory deeper than the model checkpoint, like for
        #: Sentence Transformers CLIP models.
        for root, _, files in os.walk(model_path):
            if 'config.json' in files:
                self.config_path = os.path.join(root, 'config.json')
                self.model_path = root #: The local path to the model checkpoint/weights in HuggingFace format.
                print(f"config_path: {self.config_path}")
                print(f"model_path: {self.model_path}")
                break
        
        # load the config file
        if os.path.isfile(self.config_path):
            with open(self.config_path) as config_file:
                self.config = AttributeDict(json.load(config_file))
        else:
            logging.warning(f"could not find model config file at {self.config_path}")
            self.config = AttributeDict()

        self.config.name = kwargs.get('name')
        self.config.api = kwargs.get('api')
        
        if 'max_position_embeddings' not in self.config and self.config.api != 'st':
            self.config.max_position_embeddings = self.config.get('llm_max_length', 4096)
                    
        #: Dict containing the latest generation performance statistics.
        self.stats = AttributeDict()
        
        #: :class:`VLAModel` for vision-language-action models.
        self.vla = None
        
        #: List of vision encoders for vision/language models.
        self.vision = []  
        
        #: True if this is a multimodal vision/language model.
        self.has_vision = self.config_vision()
        
        #: True if this model has a separate text embedding layer for embed_text()
        self.has_embed = False
        
        # token and embedding caches
        self.embed_cache = {}
        
        # create the tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, use_fast=True, trust_remote_code=True)
        except:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, use_fast=False, trust_remote_code=True)
            
    def is_type(self, architectures):
        # Check the architectures list in the HF config and see if any of these are included.
        # They are like "LlamaForCausalLM" and get used because model_type may have been patched.
        if isinstance(architectures, str):
            architectures = [architectures]
            
        for model_arch in self.config.get('architectures', []):
            model_arch = model_arch.lower()
            for arch in architectures:
                if arch.lower() in model_arch:
                    return arch
        
    def patch_config(self, load=None, save=None, **kwargs):
        # Update the original HF model's config.json with different settings from the provided kwargs.
        # The original will be saved under the same directory to 'config.json.backup'
        backup_path = self.config_path + '.backup'
        
        if not os.path.isfile(backup_path):
            logging.info(f"backing up original model config to {backup_path}")
            shutil.copyfile(self.config_path, backup_path)
            
        logging.info(f"patching model config with {kwargs}")
        
        if save:
            if os.path.isfile(save):
                with open(save) as config_file:
                    patched_config = json.load(config_file)
            else:
                patched_config = {}
        else:        
            patched_config = self.config #.copy()
        
        if load:
            with open(load, 'r') as config_file:
                patched_config.update(json.load(config_file))
                
        patched_config.update(kwargs)

        with open(save if save else self.config_path, 'w') as config_file:
            json.dump(patched_config, config_file, indent=2)
            
        return patched_config
    
    def restore_config(self, **kwargs):
        # restore the config file back to the original so that HF can load it again
        backup_path = self.config_path + '.backup'
        
        if os.path.isfile(backup_path): 
            logging.debug(f"restoring original model config from {backup_path}")
            shutil.copyfile(backup_path, self.config_path)
            
    def config_vision(self, **kwargs):
        # Check the model config for multimodal support (can be in a variety of formats)
        model_type = self.config.model_type.lower()
        has_vision = 'llava' in model_type
        
        # patch the config to change llava to llama so the quant tools handle it
        if has_vision:
            if 'stablelm' in model_type:
                self.patch_config(model_type='stablelm_epoch')
            elif 'phi' in model_type:
                self.patch_config(model_type='phi')
            else:
                self.patch_config(model_type='llama')
        else:
            name_or_path = self.config.get('_name_or_path')
            if name_or_path:
                has_vision = 'llava' in name_or_path.lower()

        # check for VLMs in the architectures list, since model_type may have been patched
        arch = self.is_type(['llava', 'bunny', 'openvla'])
        
        if arch:
            has_vision = True
         
        # OpenVLA needs its LLM layer names renamed   
        if arch == 'openvla':
            llm_path = os.path.join(self.model_path, 'llm')
            llm_config = os.path.join(llm_path, 'config.json')
            
            if not os.path.isdir(llm_path):
                os.makedirs(llm_path)

                self.patch_config(
                    load=download_model(os.path.join(self.config.hf_llm_id, 'config.json')),
                    save=llm_config, **self.config.text_config,
                )
                
                with open(llm_config) as file:
                    self.patch_config(**filter_keys(json.load(file), keep=['max_position_embeddings', 'vocab_size']))
                    
                for tokenizer in glob.glob(os.path.join(self.model_path, 'tokenizer*')):
                    shutil.copy(tokenizer, llm_path)
                    
                rename_weights(self.model_path, llm_path, lambda layer: layer.replace('language_model.', ''))

            dataset_config = os.path.join(self.model_path, 'dataset_statistics.json')
            
            if os.path.isfile(dataset_config):
                with open(dataset_config) as f:
                    self.config.norm_stats.update(json.load(f))
                    
            self.model_path = llm_path
             
        # change model_type back to the base model    
        if self.config.model_type == 'bunny-stablelm':
            self.patch_config(model_type='stablelm_epoch')
        elif self.config.model_type == 'bunny-phi':
            self.patch_config(model_type='phi')
            
        # support VILA checkpoints with LLM and vision encoder under separate subdirectories
        if 'vision_tower_cfg' in self.config:
            vision_path = os.path.join(self.model_path, os.path.basename(self.config['vision_tower_cfg']['_name_or_path']))
            if not os.path.isdir(vision_path):
                raise IOError(f"multimodal config was for separate models, but could not find {vision_path}")
            if 'mm_vision_tower' not in self.config:
                self.config['mm_vision_tower'] = vision_path
        
        if 'mm_projector_cfg' in self.config:
            self.config.mm_projector_path = os.path.join(self.model_path, os.path.basename(self.config['mm_projector_cfg']['_name_or_path']))
            if not os.path.isdir(self.config.mm_projector_path):
                raise IOError(f"multimodal config was for separate models, but could not find {self.config.mm_projector_path}")
            if 'mm_projector_type' not in self.config:
                self.config['mm_projector_type'] = self.config['mm_projector_cfg']['mm_projector_type']

        if 'mm_projector_path' not in self.config:
            self.config.mm_projector_path = self.model_path
                          
        if 'llm_cfg' in self.config:
            llm_path = os.path.join(self.model_path, os.path.basename(self.config['llm_cfg']['_name_or_path']))
            if not os.path.isdir(llm_path):
                raise IOError(f"multimodal config was for separate models, but could not find {llm_path}")
            with open(os.path.join(llm_path, 'config.json')) as config_file:
                self.config.update(json.load(config_file))
            self.model_path = llm_path  # redirect downstream LLM APIs to the LLM model
          
        return has_vision
               
    def init_vision(self, vision_model=None, vision_api='auto', **kwargs):
        # Load the vision encoder (CLIP/SigLIP) and mm_projector for multimodal models
        if not self.has_vision:
            return

        if self.is_type('openvla'):
            from nano_llm.vision.vla import VLAModel
            
            weights_key = ['vision_backbone.featurizer.', 'vision_backbone.fused_featurizer.']
            self.vision = [
                TIMMVisionModel(
                    timm_model_id, 
                    weights=self.model_path,
                    weights_key=lambda layer: layer.replace(weights_key[i], '').replace('scale_factor', 'gamma') if weights_key[i] in layer else None,
                    img_size=self.config.image_sizes[i], 
                    act_layer=self.config.timm_override_act_layers[i],
                    hidden_state=-2,
                    num_classes=0,
                    dtype=torch.float16,
                    use_tensorrt=(vision_api == 'auto' or vision_api == 'trt'), 
                )
                for i, timm_model_id in enumerate(self.config.timm_model_ids)
            ]
                
            llm_hidden_size = self.config.get('hidden_size', 4096)
            vision_hidden_size = 0
            
            for vision in self.vision:
                vision_hidden_size += vision.config.output_shape[-1]
                 
            logging.debug(f"{self.config.name} vision_hidden_size {vision_hidden_size}  llm_hidden_size {llm_hidden_size}")
            
            self.mm_projector = MMProjector.from_pretrained(
                self, dtype=torch.float16, 
                input_dim=vision_hidden_size, 
                output_dim=llm_hidden_size
            )
            
            self.vla = VLAModel(self, action_space=self.config.pop('norm_stats', {}))
        else:
            # load the image embedding model
            self.vision = [
                CLIPVisionModel.from_pretrained(
                    vision_model if vision_model else self.config.mm_vision_tower,
                    crop=(kwargs.get('vision_scaling', 'resize') == 'crop'),
                    use_tensorrt=(vision_api == 'auto' or vision_api == 'trt'), 
                    dtype=torch.float16)
            ]
            
            self.mm_projector = MMProjector.from_pretrained(self, dtype=torch.float16)
            
