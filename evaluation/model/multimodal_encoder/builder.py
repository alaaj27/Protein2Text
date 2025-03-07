import os
from .esm_encoder import ESMProteinEncoder  # Updated to import the ESMProteinEncoder

def build_protein_encoder(protein_encoder_cfg, **kwargs):
    # Extract the protein encoder name from the configuration
    protein_encoder = getattr(protein_encoder_cfg, 'mm_protein_encoder', getattr(protein_encoder_cfg, 'protein_encoder', None))
    
    # Check if the provided path is an absolute path or a valid model name
    is_absolute_path_exists = os.path.exists(protein_encoder)
    
    if is_absolute_path_exists or protein_encoder.startswith("facebook/esm2") or "esm3" in protein_encoder.lower():
        return ESMProteinEncoder(protein_encoder, args=protein_encoder_cfg, **kwargs)

    raise ValueError(f'Unknown protein encoder: {protein_encoder}')
