import pytorch_lightning as pl
import torch
import torchvision.utils as vutils
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import FakeData

from models import BaseVAE
from utils import data_loader, ClevrFolderDataset


class VAEXperiment(pl.LightningModule):

    def __init__(self,
                 vae_model: BaseVAE,
                 params: dict) -> None:
        super(VAEXperiment, self).__init__()
        self.ename = vae_model.ename

        self.model = vae_model
        self.params = params
        self.curr_device = None
        self.hold_graph = False
        try:
            self.hold_graph = self.params['retain_first_backpass']
        except:
            pass
        ### Initialize Validation Dataloader ###
        self.val_dataloader()

    def forward(self, input, **kwargs):
        return self.model(input, **kwargs)

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        real_img, labels = batch
        self.curr_device = real_img.device

        results = self.forward(real_img, labels=labels)
        train_loss = self.model.loss_function(*results,
                                              M_N=self.params['batch_size'] / self.num_train_imgs,
                                              optimizer_idx=optimizer_idx,
                                              batch_idx=batch_idx)

        self.logger.experiment.log({key: val.item() for key, val in train_loss.items()})

        return train_loss

    def validation_step(self, batch, batch_idx, optimizer_idx=0):
        real_img, labels = batch
        self.curr_device = real_img.device

        results = self.forward(real_img, labels=labels)
        val_loss = self.model.loss_function(*results,
                                            M_N=self.params['batch_size'] / self.num_val_imgs,
                                            optimizer_idx=optimizer_idx,
                                            batch_idx=batch_idx)

        return val_loss

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        tensorboard_logs = {'avg_val_loss': avg_loss}
        self.sample_images()
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def sample_images(self):
        test_input, test_label = next(iter(self.sample_dataloader))
        test_input = test_input.to(self.curr_device)
        test_label = test_label.to(self.curr_device)
        recons = self.model.generate(test_input, labels=test_label)
        vutils.save_image(test_input.data,
                          f"{self.logger.save_dir}/{self.ename}/version_{self.logger.version}/"
                          f"org_{self.logger.name}_{self.current_epoch}.png",
                          normalize=True,
                          nrow=3)
        vutils.save_image(recons.data,
                          f"{self.logger.save_dir}/{self.ename}/version_{self.logger.version}/"
                          f"recons_{self.logger.name}_{self.current_epoch}.png",
                          normalize=True,
                          nrow=3)
        return

    def configure_optimizers(self):

        optims = []
        scheds = []

        optimizer = optim.Adam(self.model.parameters(),
                               lr=self.params['LR'],
                               weight_decay=self.params['weight_decay'])
        optims.append(optimizer)
        # Check if more than 1 optimizer is required (Used for adversarial training)
        try:
            if self.params['LR_2'] is not None:
                optimizer2 = optim.Adam(getattr(self.model, self.params['submodel']).parameters(),
                                        lr=self.params['LR_2'])
                optims.append(optimizer2)
        except:
            pass

        try:
            if self.params['scheduler_gamma'] is not None:
                scheduler = optim.lr_scheduler.ExponentialLR(optims[0],
                                                             gamma=self.params['scheduler_gamma'])
                scheds.append(scheduler)

                try:
                    if self.params['scheduler_gamma_2'] is not None:
                        scheduler2 = optim.lr_scheduler.ExponentialLR(optims[1],
                                                                      gamma=self.params['scheduler_gamma_2'])
                        scheds.append(scheduler2)
                except:
                    pass
                return optims, scheds
        except:
            return optims

    @data_loader
    def train_dataloader(self):
        transform = self.data_transforms()

        if self.params['dataset'] == 'fake':
            dataset = FakeData(size=200, num_classes=1, image_size=(3, 224, 224), transform=transform)
        elif self.params['dataset'] == 'clevr':
            dataset = ClevrFolderDataset(folder='./data', split='train', max_images=self.params['max_train_images'],
                                         transform=None)
        else:
            raise ValueError('Undefined dataset type')

        self.num_train_imgs = len(dataset)
        return DataLoader(dataset,
                          batch_size=self.params['batch_size'],
                          shuffle=True,
                          drop_last=True,
                          num_workers=self.params['n_workers'])

    @data_loader
    def val_dataloader(self):
        transform = self.data_transforms()

        if self.params['dataset'] == 'fake':
            sample_dataset = FakeData(size=200, num_classes=1, image_size=(3, 224, 224), transform=transform)
        elif self.params['dataset'] == 'clevr':
            sample_dataset = ClevrFolderDataset(folder='./data', split='val', max_images=self.params['max_val_images'],
                                                transform=None)
        else:
            raise ValueError('Undefined dataset type')

        self.num_val_imgs = len(sample_dataset)
        self.sample_dataloader = DataLoader(sample_dataset,
                                            batch_size=self.params['batch_size'],
                                            shuffle=False,
                                            drop_last=False,
                                            num_workers=self.params['n_workers'])
        return self.sample_dataloader

    def data_transforms(self):
        if self.params['dataset'] == 'fake':
            transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                            transforms.CenterCrop(148),
                                            transforms.Resize(self.params['img_size']),
                                            transforms.ToTensor(),
                                            ])
        elif self.params['dataset'] == 'clevr':
            transform = None
        else:
            raise NotImplementedError(f"Dataset {self.params['dataset']} has not been implemented yet!")
        return transform

