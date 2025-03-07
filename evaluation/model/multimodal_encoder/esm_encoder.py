import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class ESMProteinEncoder(nn.Module):
    def __init__(self, protein_encoder, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        
        self.protein_encoder_name = protein_encoder
        self.select_layer = args.mm_protein_select_layer
        self.select_feature = getattr(args, 'mm_protein_select_feature', 'cls')  # Default to 'patch'
        # self.max_length = getattr(args, 'mm_protein_max_length', 512)  # Default to 'patch'
        
        self.mm_hidden_size = None

        if not delay_load:
            self.load_model()
        elif getattr(args, 'unfreeze_mm_protein_encoder', False):
            self.load_model()

    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.protein_encoder_name))
            return

        if self.protein_encoder_name.startswith("facebook/esm2"):
            self.protein_encoder = AutoModel.from_pretrained(self.protein_encoder_name, device_map=device_map)
            self.tokenizer = AutoTokenizer.from_pretrained(self.protein_encoder_name)
            self.protein_encoder.requires_grad_(False)
            self.mm_hidden_size = self.protein_encoder.config.hidden_size
            self.is_loaded = True

        else:
            raise ValueError(f'Unknown protein encoder: {self.protein_encoder_name}')



    def feature_select(self, sequence_features, remove_eos=False):
        
        # Apply feature selection based on self.select_feature
        if self.select_feature == 'patch':  # Default behavior for 'patch'
            sequence_features = [features[1:] for features in sequence_features]  # Skipping the CLS token
        elif self.select_feature == 'cls_patch':
            sequence_features = sequence_features  # Keep the entire sequence including CLS
        elif self.select_feature == 'cls':  # Keep only the CLS token
            sequence_features = [features[:1] for features in sequence_features]
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        
        # Optionally remove the <eos> token if specified
        if remove_eos:
            sequence_features = [
                features[:-1] if '<eos>' in self.tokenizer.decode(features[-1]) else features
                for features in sequence_features
            ]
        
        return sequence_features


    def encode_esm2(self, sequences):
        inputs = self.tokenizer(sequences, return_tensors='pt', padding=True)
        with torch.no_grad():
            sequence_forward_outs = self.protein_encoder(**inputs.to(device=self.device), output_hidden_states=True)

        sequence_forward_outs = sequence_forward_outs.hidden_states[self.select_layer]

        attention_mask = inputs["attention_mask"] 
        non_padded_embeddings = [sequence_forward_outs[i, :mask.sum()] for i, mask in enumerate(attention_mask)]

        sequence_features = self.feature_select(non_padded_embeddings) 

        return sequence_features



    @torch.no_grad()
    def forward(self, sequences):
        if self.protein_encoder_name.startswith("facebook/esm2"):
            return self.encode_esm2(sequences)
        else:
            raise ValueError(f'Unknown protein encoder: {self.protein_encoder_name}')
        


    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.protein_encoder.dtype

    @property
    def device(self):
        return self.protein_encoder.device

    @property
    def config(self):
        return self.protein_encoder.config

    @property
    def hidden_size(self):
        return self.config.hidden_size
