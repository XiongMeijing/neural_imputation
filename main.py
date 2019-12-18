import argparse
import collections
import torch
import numpy as np
import neural_imputation.data_loader.data_loaders as module_data
import neural_imputation.model.loss as module_loss
import neural_imputation.model.metric as module_metric
import neural_imputation.model.model as module_arch
import neural_imputation.data_loader.preprocessors as module_preprocess
from neural_imputation.data_loader.datasets import MaskedDataset
from neural_imputation.parse_config import ConfigParser
from neural_imputation.trainer import Trainer


# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

def main(config, data, mask):
    logger = config.get_logger('train')
    n = data.shape[0]
    p = data.shape[1]
    batch_size = config['batch_size']

    if hasattr(config,'miss_spike_percent'):
        if 0<config['miss_spike_percent']<1:
            miss_spiked_percent = config['miss_spike_percent']
            mask_spiked = np.random.uniform(0,1,[n,p]) # Add validation spiked missingness
            mask_spiked[mask_spiked<miss_spiked_percent] = 0
            mask_spiked[mask_spiked>0] = 1
            masks = [mask, mask_spiked]
        else:
            masks = mask
    else:
        masks = mask

    preprocessor = config.init_obj("preprocessor", module_preprocess)
    preprocessor.fit(data, masks)
    inputs = preprocessor.transform(data, masks, preimpute_transform=True)
    
    train_dataset = MaskedDataset(inputs['train']['data_preimp'],inputs['train']['masks'])
    if preprocessor.validation:
        val_dataset = MaskedDataset(inputs['validation']['data_preimp'],inputs['validation']['masks'])
    else:
        val_dataset = None
    if preprocessor.test:
        test_dataset = MaskedDataset(inputs['test']['data_preimp'],inputs['test']['masks'])
        #test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


    # build model architecture, then print to console
    model = config.init_obj('arch', module_arch)
    logger.info(model)

    # get function handles of loss and metrics
    criterion = getattr(module_loss, config['loss'])
    # criterion = config.init_obj('loss', module_loss)
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj('optimizer', torch.optim, trainable_params)

    lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)

    trainer = Trainer(model, criterion, metrics, optimizer, batch_size,
                      config=config,
                      train_data=train_dataset,
                      val_data=val_dataset,
                      lr_scheduler=lr_scheduler)

    trainer.train()
    
    data_imp = trainer.impute().item()
    np.savetxt(config.save_dir(),data_imp, delimiter=",")
    
    """
    Test phase is not written, but I sort of don't know what the point of it is?
    """
      


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Neural Imputation')
    args.add_argument("datafile", type=str, help="CSV file of the dataset")
    args.add_argument("maskfile", type=str, help="CSV file of the missing mask")
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='batch_size')
    ]
    config = ConfigParser.from_args(args, options)
    
    data = np.genfromtxt(config['datafile'], delimiter=',').astype(float)
    mask = np.genfromtxt(config['maskfile'], delimiter=',',dtype=np.int32)

    main(config, data, mask)
