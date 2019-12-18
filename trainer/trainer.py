import numpy as np
import torch
from torchvision.utils import make_grid
from neural_imputation.base import BaseTrainer
from neural_imputation.utils import inf_loop, MetricTracker
from torch.utils.data import DataLoader



class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, batch_size, config,
                 orig_data, train_data, val_data=None, lr_scheduler=None, len_epoch=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config
        
        # load training data
        self.orig_data= orig_data
        self.train_data= train_data
        self.train_dataloader = DataLoader(self.train_data, batch_size=batch_size, shuffle=False)     
        
        # define validation
        self.val_data = val_data
        if self.val_data is not None:
            self.val_dataloader = DataLoader(self.val_data, batch_size=batch_size, shuffle=False)     
            self.do_validation = True
        else:
            self.do_validation = False
            
        # determine epochs
        self.len_epoch = len(self.data_loader)
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(batch_size))

        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

    def _train_epoch(self, epoch, eval_epoch=False):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :param eval_epoch: Modulo remainder of epoch number in which a metric
                evaluation will occur (this will do the correlation metrics)
        :return: A log that contains average loss and metrics in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        
        for batch_idx, inputs in enumerate(self.train_dataloader):
            if len(inputs)==3:
                data, mask, mask_spiked = inputs
            elif len(inputs) == 2:
                data, mask = inputs
            else:
                raise ValueError

            self.optimizer.zero_grad()
            rate, distortion = self.model([data, mask], training=True)
            loss = self.criterion(rate, distortion)
            loss.backward()
            self.optimizer.step()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item())
            self.train_metrics
            #for met in self.metric_ftns:
                #self.train_metrics.update(met.__name__, met(output, target))

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item()))
                self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

            if batch_idx == self.len_epoch:
                break
        log = self.train_metrics.result()
        if eval_epoch:
            eval_sample = self.model.sample([self.train_data.data, self.train_data.mask])
            eval_mask = self.train_data.mask_true.clone().detach()
            if self.train_data.mask_spiked is not None:
                eval_mask_spiked = self.train_data.mask_spiked.clone().detach()
                eval_mask = [eval_mask, eval_mask_spiked]
            target = self.train_data.data.clone().detach()
            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(eval_sample, target, eval_mask))

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            for batch_idx, inputs in enumerate(self.val_dataloader):
                if len(inputs)==3:
                    data, mask, mask_spiked = inputs
                elif len(inputs) == 2:
                    data, mask = inputs
                else:
                    raise ValueError
                    
                rate, distortion = self.model([data,mask], training=False)
                loss = self.criterion(rate, distortion)

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.item())

        eval_sample = self.model.sample([self.val_data.data, self.val_data.mask])
        eval_mask = self.val_data.mask_true.clone().detach()
        if self.val_data.mask_spiked is not None:
            eval_mask_spiked = self.val_data.mask_spiked.clone().detach()
            eval_mask = [eval_mask, eval_mask_spiked]
        target = self.val_data.data.clone().detach()
        for met in self.metric_ftns:
            self.train_metrics.update(met.__name__, met(eval_sample, target, eval_mask))

    
    def impute(self):
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():   
            orig_data = self.orig_data.clone().detach()
            mask = self.mask.clone().detach()            
            sample = self.model.sample([orig_data, mask])
            
        return sample.detach()


    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
