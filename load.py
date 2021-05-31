import argparse

import torch.backends.cudnn as cudnn
import yaml
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TestTubeLogger

from experiment import VAEXperiment
from models import *

parser = argparse.ArgumentParser(description='Generic loader for VAE models')
parser.add_argument('--config', '-c',
                    dest="filename",
                    metavar='FILE',
                    help='path to the config file',
                    default='./configs/cat_vae.yaml')

parser.add_argument('--checkpoint_path', '-cp',
                    dest="filename",
                    metavar='FILE',
                    help='path to the checkpoint file')

parser.add_argument('--cont', '-cont',
                    help='whether to continue training ')

args = parser.parse_args()

with open(args.filename, 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)

tt_logger = TestTubeLogger(
    save_dir=config['logging_params']['save_dir'],
    name=config['logging_params']['name'],
    debug=False,
    create_git_tag=False,
)

# For reproducibility
torch.manual_seed(config['logging_params']['manual_seed'])
np.random.seed(config['logging_params']['manual_seed'])
cudnn.deterministic = True
cudnn.benchmark = False

model = vae_models[config['model_params']['name']](**config['model_params'])
experiment = VAEXperiment.load_from_checkpoint(args.checkpoint_path, model, config['exp_params'])

if args.cont == 1:
    runner = Trainer(default_root_dir=f"{tt_logger.save_dir}",
                     min_epochs=1,
                     logger=tt_logger,
                     log_every_n_steps=50,
                     num_sanity_val_steps=-1,
                     **config['trainer_params'])

    print(f"======= Continuing Training {config['model_params']['name']} =======")
    runner.fit(experiment)
