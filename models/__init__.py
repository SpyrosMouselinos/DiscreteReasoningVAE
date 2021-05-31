from .base import *
from .cat_vae import *
from .vq_vae import *

GumbelVAE = CategoricalVAE

vae_models = {'VQVAE': VQVAE, 'CategoricalVAE': CategoricalVAE, 'GumbelVAE': GumbelVAE}
