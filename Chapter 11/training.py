import sys
import argparse
import datetime
import logging 
import numpy as np
import torch
from LunaModel import LunaModel
import torch.nn as nn
import tqdm
from LunaDataset import LunaDataset
from torch.optim import SGD
from torch.utils.data import DataLoader
from util import enumerateWithEstimate

log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
log.setLevel(logging.INFO)
log.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(levelname)s:%(asctime)s:%(module)s:%(message)s')

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
log.addHandler(stream_handler)

file_handler = logging.FileHandler('training.log')
log.addHandler(file_handler)
file_handler.setFormatter(formatter)


# names array indexes are declared at the module-level
# scope -- aka global variables -- these are indices
METRICS_LABEL_NDX=0
METRICS_PRED_NDX=1
METRICS_LOSS_NDX=2
METRICS_SIZE=3

class LunaTrainingApp:

    # class variables/constants
    # METRICS_LABEL_NDX=0
    # METRICS_PRED_NDX=1
    # METRICS_LOSS_NDX=2
    # METRICS_SIZE=3

    def __init__(self, sys_argv=None):
        if sys_argv is None:
            sys_argv = sys.argv[1:]
        
        #  for some reason the default values do not result in -- typerror
        # for just the positively labeled data -- try num workers of 2
        # for ubuntu, you can try num workers 4 and batch size of 16
        parser = argparse.ArgumentParser()
        parser.add_argument('--num-workers',
                            help='Number of worker processes for background data loading',
                            default=1,
                            type=int)
        parser.add_argument('--epochs',
                            help='Number of epochs to train the model',
                            default=1,
                            type=int)
        parser.add_argument('--batch-size',
                            help='Batches to process/load, if vram is a problem choose low batches',
                            default=32,
                            type=int)

        # this should take in inputs from the command line
        self.cli_args = parser.parse_args(sys_argv)

        self.time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H.%M.%S')

        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device('cuda' if self.use_cuda else 'cpu')

        self.model = self.initModel()
        self.optimizer = self.initOptimizer()

        self.totalTrainingsamples_count = 0
    
    def main(self):
        log.info(f'Starting {type(self).__name__}, {self.cli_args}')

        # our training and validaiton set as torch dataloader objects
        # you should expect to see 2 loading bars, this is from the
        # LunaDataset class
        trainset_dl = self.initTrainDl()
        valset_dl = self.initValDl()

        log.info('In LunaTrainingApp.main method -- Starting Training and Validation...')
        # for epoch_ndx in tqdm.tqdm(range(1, self.cli_args.epochs + 1), total=self.cli_args.epochs, desc='Training and Validating'):
        for epoch_ndx in range(1, self.cli_args.epochs + 1):
            trnMetrics_t = self.doTraining(epoch_ndx, trainset_dl)
            self.logMetrics(epoch_ndx, 'trn', trnMetrics_t)

            valMetrics_t = self.doValidation(epoch_ndx, valset_dl)
            self.logMetrics(epoch_ndx, 'val', valMetrics_t)
            


    # this function, initModel, will initialize our model and return the model
    # model will either be in cuda or cpu depending on self.use_cuda attribute
    # additionally we also take advantage of multi gpu system
    def initModel(self):
        log.info('Constructing our LunaModel...')
        model = LunaModel()
        if self.use_cuda:
            log.info(f'Using CUDA; {torch.cuda.device_count()} devices')
            if torch.cuda.device_count() > 1:
                # here if we have more then one GPU we can use nn.DataParallel
                # to distribute the work between all the GPUs in the system
                model = nn.DataParallel(model)
            model = model.to(device=self.device)
        return model
    
    # function that will initialize and stochastic gradient descent optimizer
    def initOptimizer(self):
        return SGD(self.model.parameters(), lr=0.001, momentum=0.99)

    # this method is dedicated for our training dataset -- it should 
    # return a dataloader for our training data
    def initTrainDl(self):
        # creating our training dataset
        # print('Initializing Training Data...')
        log.info('Initializing Training Data...')
        train_ds = LunaDataset(val_stride=10, isValSet_bool=False)

        batch_size = self.cli_args.batch_size 

        # our train loader
        train_dl = DataLoader(dataset=train_ds, 
                              batch_size=batch_size, 
                              num_workers=self.cli_args.num_workers, 
                              pin_memory=self.use_cuda)
        return train_dl
    
    # this method is deidcated for our validation dataset -- it should
    # return a dataloader for our traininf data
    def initValDl(self):
        # creating our validation dataset
        # print('Initializing Validation Data...')
        log.info('Initializing Validation Data...')
        validation_ds = LunaDataset(val_stride=10, isValSet_bool=True)

        batch_size = self.cli_args.batch_size 

        validation_dl = DataLoader(dataset=validation_ds,
                                   batch_size=batch_size,
                                   num_workers=self.cli_args.num_workers,
                                   pin_memory=self.use_cuda)
        return validation_dl

    
    # this is our training method -- it takes in the training dataloader
    def doTraining(self, epoch_ndx, train_dl):
        self.model.train()
        trnMetrics_g = torch.zeros(
            METRICS_SIZE,
            len(train_dl.dataset),
            device=self.device,
            dtype=torch.float32
        )

        # this was a stylistic choice used by the book...try something else
        # batch_iter = enumerateWithEstimate(
        #     itter=train_dl,
        #     desc_str=f'E{epoch_ndx} Training',
        #     start_ndx=train_dl.num_workers,
        #     itter_len=len(train_dl.dataset)
        # )


        log.info('Performing Training')
        # for batch_ndx, batch_tup in tqdm.tqdm(batch_iter, total=len(train_dl.dataset), desc='Running Training'):
        # for batch_ndx, batch_tup in batch_iter:
        # for (batch_ndx, batch_tup) in train_dl:
        length = int(len(train_dl.dataset) / train_dl.batch_size)
        for (batch_ndx, batch_tup) in tqdm.tqdm(enumerate(train_dl), total=length, desc='Training'):
            # zero the gradient
            self.optimizer.zero_grad()

            loss_var = self.computeBatchLoss(
                batch_ndx,
                batch_tup,
                train_dl.batch_size,
                trnMetrics_g
            )

            # here we are calculating the gradient and stepping
            loss_var.backward()
            self.optimizer.step()
        
        self.totalTrainingsamples_count += len(train_dl.dataset)

        # we are returning metrics data to the cpu
        return trnMetrics_g.to('cpu')
    
    # this is our validaiton method -- it takes in epoch and the validation dataloader
    # very similar to the training method
    def doValidation(self, epoch_ndx, val_dl):
        with torch.no_grad():
            self.model.eval()
            valMetrics_g = torch.zeros(
                METRICS_SIZE,
                len(val_dl.dataset),
                device=self.device,
                dtype=torch.float32 # remove this if it dont work
            )

            # this was a stylistic choice used by the book...try something else
            # batch_iter = enumerateWithEstimate(
            #     val_dl,
            #     f'E{epoch_ndx} Validation',
            #     start_ndx=val_dl.num_workers,
            #     itter_len=len(val_dl.dataset)
            # )
            

            log.info('Performing Validation')
            length = int(len(val_dl.dataset) / val_dl.batch_size)
            # for batch_ndx, batch_tup in tqdm.tqdm(batch_iter, total=len(val_dl.dataset), desc='Running Validation'):
            for (batch_ndx, batch_tup) in tqdm.tqdm(enumerate(val_dl), total=length, desc='Validating'):
                self.computeBatchLoss(batch_ndx, batch_tup, val_dl.batch_size, valMetrics_g)

        return valMetrics_g.to('cpu')

    # a function for computing the loss after a batch has been passed through
    def computeBatchLoss(self, batch_ndx, batch_tup, batch_size, metrics_g):
        input_t, label_t, _series_list, _center_list = batch_tup 

        # the non_blocking parameter -- look up docs
        # here we are moving things to gpu
        input_g = input_t.to(self.device, non_blocking=True)
        label_g = label_t.to(self.device, non_blocking=True)

        # here we are performing a forward pass, and passing in our inputs
        # to our mdoel
        logits_g, probability_g = self.model(input_g)

        # reduction set to none means it will calculate the loss 
        # per sample in the batch -- so wen we pass in our logits and labels
        # this should return a tensor of losses corresponding to each sample
        loss_func = nn.CrossEntropyLoss(reduction='none')
        loss_g = loss_func(logits_g, label_g[:,1])

        start_ndx = batch_ndx * batch_size
        end_ndx = start_ndx + label_t.size(0)

        # use detatch since none of our metrics need to keep track of gradient
        metrics_g[METRICS_LABEL_NDX, start_ndx:end_ndx] = label_g[:,1].detach()
        metrics_g[METRICS_PRED_NDX, start_ndx:end_ndx] = probability_g[:,1].detach()
        metrics_g[METRICS_LOSS_NDX, start_ndx:end_ndx] = loss_g.detach()

        # we are returning the average loss in a batch
        return loss_g.mean()
    
    # array masking is implemented in here -- essentiallly it's kind of like logical
    # indexing that we see in pandas -- where elements that satisfy the given logic 
    # are returned to the user -- this whole method is dedicated to logging
    def logMetrics(self, epoch_ndx, mode_str, metrics_t, classificationThreshold=0.5):
        print(metrics_t)
        # here we are createing our own logic statement -- our masks
        negLabel_mask = metrics_t[METRICS_LABEL_NDX] <= classificationThreshold
        negPred_mask = metrics_t[METRICS_PRED_NDX] <= classificationThreshold

        print('neg masks', negLabel_mask, negPred_mask)

        # the '~' symbol denotes the NOT operator
        posLabel_mask = ~negLabel_mask
        posPred_mask = ~negPred_mask

        print('pos mask', posLabel_mask, posPred_mask)

        neg_count = int(negLabel_mask.sum())
        pos_count = int(posLabel_mask.sum())
        
        print('counts', neg_count, pos_count)

        # the '&' is a the bitwise and operator -- it operates on binary numbers
        # specifically 2 binary numbers, it will align them and return the 
        # evaluated new binary number-- ex. 0b101011 & 0b011111 should 
        # result in 0b001011 as the new bitwise output (negLabel_mask and negPred_mask)
        # are binary values of Tru/False or 0/1 so we can do this....
        neg_correct = int((negLabel_mask & negPred_mask).sum())
        pos_correct = int((posLabel_mask & posPred_mask).sum())

        print('neg and pos correct', neg_correct, pos_correct)

        metrics_dict = {}
        metrics_dict['loss/all'] = metrics_t[METRICS_LOSS_NDX].mean()
        metrics_dict['loss/neg'] = metrics_t[METRICS_LOSS_NDX, negLabel_mask].mean()
        metrics_dict['loss/pos'] = metrics_t[METRICS_LOSS_NDX, posLabel_mask].mean()
        metrics_dict['correct/all'] = (pos_correct + neg_correct) / np.float32(metrics_t.shape[1]) * 100

        # according to the logs this line of code right here is throwing an error! probably division by
        # NaN or zero division as the values of neg_count and neg_correct are supposedly 0
        # is this what maybe causing the TypeError: __init__() missing 1 required positional argument: 'dtype'
        metrics_dict['correct/neg'] = neg_correct / np.float32(neg_count) * 100


        metrics_dict['correct/pos'] = pos_correct / np.float32(pos_count) * 100

        log.info('E{} {:8} {loss/all:.4f},' + '{correct/all:-5.1f}% correct'.format(epoch_ndx, mode_str, **metrics_dict))
        log.info('E{} {:8} {loss/neg:.4f},' + '{correct/neg:-5.1f}% correct ({neg_correct:} of {neg_count:})'.format(epoch_ndx, mode_str + '_neg', neg_correct=neg_correct, neg_count=neg_count, **metrics_dict))
        log.info('E{} {:8} {loss/pos:.4f},' + '{correct/pos:-5.1f}% correct ({pos_correct:} of {pos_count:})'.format(epoch_ndx, mode_str + '_pos', pos_correct=pos_correct, pos_count=pos_count, **metrics_dict))
        log.info(metrics_dict)
        log.info(f'{neg_count}, {pos_count}, {neg_correct}, {pos_correct}')

if __name__ == '__main__':
    LunaTrainingApp().main()