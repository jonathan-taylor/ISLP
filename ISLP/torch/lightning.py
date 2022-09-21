import warnings

import torch.nn as nn
from torch.optim import RMSprop

from torch.utils.data import (random_split,
                              DataLoader,
                              Dataset)
from torch import tensor, Generator, concat
from torchvision import transforms
from torch.utils.data import TensorDataset

from torchmetrics import Accuracy

from pytorch_lightning import (LightningModule,
                               LightningDataModule)
from pytorch_lightning.utilities.distributed import rank_zero_only
from pytorch_lightning.callbacks import Callback

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
                                                      nvalidation],
                                                     generator=Generator().manual_seed(seed))
                
        self.test_dataset = test_dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers and num_workers > 0
        self.seed = seed
        
    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          shuffle=True,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          persistent_workers=self.persistent_workers)

    # for validation, test and predict
    # we load the entire data at once in this
    # simple module. otherwise metrics get averaged
    # over minibatch

    def val_dataloader(self):
        return DataLoader(self.validation_dataset,
                          shuffle=False,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          persistent_workers=self.persistent_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          persistent_workers=self.persistent_workers)

    def predict_dataloader(self):
        return DataLoader(self.test_dataset,
                          batch_size=len(self.test_dataset),
                          num_workers=self.num_workers,
                          persistent_workers=self.persistent_workers)

    @staticmethod
    def fromarrays(*arrays,
                   test=0,
                   validation=0,
                   batch_size=32,
                   num_workers=0,
                   persistent_workers=True,
                   test_as_validation=False,
                   seed=0):

        tensor_ds = TensorDataset(*[tensor(arr) for arr in arrays])
        npts = len(tensor_ds)
        if type(test) == float:
            test = int(test*npts)
        if type(validation) == float:
            validation = int(validation*npts)
        if npts <= test + validation:
            raise ValueError('Total test and validation requested exceeds size of dataset: no data left for training.')
        train_ds, test_ds, valid_ds = random_split(tensor_ds,
                                                   [npts - test - validation,
                                                    test,
                                                    validation],
                                                   generator=Generator().manual_seed(seed))
        if test_as_validation:
            valid_ds = test_ds
            if validation != 0:
                warnings.warn('Validation set not empty but `test_as_validation` is True. If using test set for validation set `validation=0`.')

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
        self.metrics = metrics
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
        for _metric in self.metrics.keys():
            self.log(f"train_{_metric}",
                     self.metrics[_metric](preds, y_),
                     on_epoch=self.on_epoch)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch

    @rank_zero_only
    def validation_step(self, batch, batch_idx):
        x, y = batch

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
                              metrics={},
                              device=None,
                              **kwargs):
        loss = nn.BCEWithLogitsLoss()
        if 'accuracy' not in metrics:
            metrics['accuracy'] = Accuracy()
        if device is not None:
            for key, metric in metrics:
                metrics[key] = metric.to(device)
        return SimpleModule(model,
                            loss,
                            metrics=metrics,
                            pre_process_y_for_metrics = lambda x: x.int(),
                            **kwargs)

    @staticmethod
    def classification(model,
                       metrics={},
                       device=None,
                       **kwargs):
        loss = nn.CrossEntropyLoss()
        if 'accuracy' not in metrics:
            metrics['accuracy'] = Accuracy()
        if device is not None:
            for key, metric in metrics:
                metrics[key] = metric.to(device)
        return SimpleModule(model,
                            loss,
                            metrics=metrics,
                            **kwargs)
    
class ErrorTracker(Callback):

    def on_validation_epoch_start(self,
                                  trainer,
                                  pl_module):
        self.val_preds = []
        self.val_targets = []

    def on_validation_batch_start(self,
                                  trainer,
                                  pl_module,
                                  batch,
                                  batch_idx,
                                  dataloader_idx):
        x, y = batch
        self.val_preds.append(pl_module.forward(x))
        self.val_targets.append(y)

    def on_validation_epoch_end(self,
                                trainer,
                                pl_module):

        preds = concat(self.val_preds)
        targets = concat(self.val_targets)
        targets_ = pl_module.pre_process_y_for_metrics(targets)

        loss = pl_module.loss(preds, targets)
        pl_module.log("valid_loss",
                      loss,
                      on_epoch=pl_module.on_epoch)

        for _metric in pl_module.metrics.keys():
            pl_module.log(f"valid_{_metric}",
                          pl_module.metrics[_metric](preds, targets_),
                          on_epoch=pl_module.on_epoch)

    def on_test_epoch_start(self,
                            trainer,
                            pl_module):
        self.test_preds = []
        self.test_targets = []

    def on_test_batch_start(self,
                            trainer,
                            pl_module,
                            batch,
                            batch_idx,
                            dataloader_idx):
        x, y = batch
        self.test_preds.append(pl_module.forward(x))
        self.test_targets.append(y)

    def on_test_epoch_end(self,
                          trainer,
                          pl_module):

        preds = concat(self.test_preds)
        targets = concat(self.test_targets)
        targets_ = pl_module.pre_process_y_for_metrics(targets)

        loss = pl_module.loss(preds, targets)
        pl_module.log("test_loss",
                      loss,
                      on_epoch=pl_module.on_epoch)

        for _metric in pl_module.metrics.keys():
            pl_module.log(f"test_{_metric}",
                          pl_module.metrics[_metric](preds, targets_),
                          on_epoch=pl_module.on_epoch)

