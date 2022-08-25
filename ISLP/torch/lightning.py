import torch.nn as nn
from torch.optim import RMSprop

from torchmetrics import Accuracy

from pytorch_lightning import (LightningModule,
                               LightningDataModule)
from pytorch_lightning.utilities.distributed import rank_zero_only
from pytorch_lightning.utilities.seed import seed_everything
from torch.utils.data import (random_split,
                              DataLoader,
                              Dataset)
from torchvision import transforms
from torch.utils.data import TensorDataset

class SimpleDataModule(LightningDataModule):

    def __init__(self,
                 train_dataset,
                 test_dataset,
                 batch_size=32,
                 num_workers=0,
                 persistent_workers=True,
                 validation=None,
                 seed=0):

        super(SimpleDataModule, self).__init__()

        ntrain = len(train_dataset)
        if type(validation) == float:
            nvalidation = int(validation * len(train_dataset))
        elif type(validation) == int:
            nvalidation = validation
        elif validation is None:
            nvalidation = 0

        if isinstance(validation, Dataset):
            (self.train_dataset,
             self.validation_dataset) = (train_dataset,
                                         validation)
        else:
            (self.train_dataset,
             self.validation_dataset) = random_split(train_dataset,
                                                     [ntrain - nvalidation,
                                                      nvalidation])
                
        self.test_dataset = test_dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers and num_workers > 0
        self.seed = seed
        
    def train_dataloader(self):
        seed_everything(self.seed, workers=True)
        return DataLoader(self.train_dataset,
                          shuffle=True,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          persistent_workers=self.persistent_workers)

    def val_dataloader(self):
        seed_everything(self.seed, workers=True)
        return DataLoader(self.validation_dataset,
                          shuffle=False,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          persistent_workers=self.persistent_workers)

    def test_dataloader(self):
        seed_everything(self.seed, workers=True)
        return DataLoader(self.test_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          persistent_workers=self.persistent_workers)

    def predict_dataloader(self):
        seed_everything(self.seed, workers=True)
        return DataLoader(self.test_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          persistent_workers=self.persistent_workers)

    @staticmethod
    def fromarrays(*arrays,
                   test=0,
                   validation=0,
                   batch_size=32,
                   num_workers=0,
                   persistent_workers=True,
                   seed=0):

        tensor_ds = TensorDataset(*arrays)
        npts = len(tensor_ds)
        if type(test) == float:
            test = int(test*npts)
        if type(validation) == float:
            validation = int(validation*npts)
        train_ds, test_ds, valid_ds = random_split(tensor_ds,
                                                   [npts - test - validation,
                                                    test,
                                                    validation])
        return SimpleDataModule(train_ds,
                                test_ds,
                                validation=valid_ds,
                                batch_size=batch_size,
                                num_workers=num_workers,
                                persistent_workers=persistent_workers,
                                seed=seed)

class SimpleModule(LightningModule):

    """
    A simple `pytorch_lightning` module for regression problems.
    """

    def __init__(self,
                 model,
                 loss,
                 optimizer=None,
                 metrics={},
                 on_epoch=True,
                 pre_process_y_for_metrics=lambda y: y):

        super(SimpleModule, self).__init__()

        self.model = model
        self.loss = loss or nn.MSELoss()
        optimizer = optimizer or RMSprop(model.parameters())
        self._optimizer = optimizer
        self._metrics = metrics
        self.on_epoch = on_epoch
        self.pre_process_y_for_metrics = pre_process_y_for_metrics
        
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self.forward(x)
        loss = self.loss(preds, y)
        self.log("train_loss",
                 loss,
                 on_epoch=self.on_epoch,
                 on_step=False)

        y_ = self.pre_process_y_for_metrics(y)
        for _metric in self._metrics.keys():
            self.log(f"train_{_metric}",
                     self._metrics[_metric](preds, y_),
                     on_epoch=self.on_epoch)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        preds = self.forward(x)
        loss = self.loss(preds, y)

        y_ = self.pre_process_y_for_metrics(y)
        for _metric in self._metrics.keys():
            self.log(f"test_{_metric}",
                     self._metrics[_metric](preds, y_),
                     on_epoch=self.on_epoch)
        self.log("test_loss",
                 loss,
                 on_epoch=self.on_epoch)

    @rank_zero_only
    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self.forward(x)
        loss = self.loss(preds, y)
        # convert before computing metrics
        # needed for BCEWithLogitsLoss -- y is float but
        # must be int for classification metrics

        y_ = self.pre_process_y_for_metrics(y)
        for _metric in self._metrics.keys():
            self.log(f"valid_{_metric}",
                     self._metrics[_metric](preds, y_),
                     on_epoch=self.on_epoch)
        self.log("valid_loss",
                 loss,
                 on_epoch=self.on_epoch)

    def predict_step(self, batch, batch_idx):
        x, y = batch
        return y, self.forward(x)

    def configure_optimizers(self):
        return self._optimizer

    @staticmethod
    def regression(model,
                   **kwargs):
        loss = nn.MSELoss()
        return SimpleModule(model,
                            loss,
                            **kwargs)

    @staticmethod
    def binary_classification(model,
                              metrics={'accuracy':Accuracy()},
                              **kwargs):
        loss = nn.BCEWithLogitsLoss()
        return SimpleModule(model,
                            loss,
                            metrics=metrics,
                            pre_process_y_for_metrics = lambda x: x.int(),
                            **kwargs)

    @staticmethod
    def classification(model,
                       metrics={'accuracy':Accuracy()},
                       **kwargs):
        loss = nn.CrossEntropyLoss()
        return SimpleModule(model,
                            loss,
                            metrics=metrics,
                            **kwargs)
    
