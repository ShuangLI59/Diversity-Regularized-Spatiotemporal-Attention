from __future__ import print_function, absolute_import
import time

import torch
from torch.autograd import Variable

from torch.nn import functional as F
from .evaluation_metrics import accuracy
from .loss import OIMLoss, TripletLoss
from .utils.meters import AverageMeter

class BaseTrainer(object):
    def __init__(self, args, model, modelSatn, modelTatn, criterion):
        super(BaseTrainer, self).__init__()
        self.model = model
        self.modelSatn = modelSatn  #Multiple Spatial Attention
        self.modelTatn = modelTatn  #Multiple Temporal Attention
        self.criterion = criterion

        self.args = args

    def train(self, epoch, data_loader, optimizer, writer, print_freq=1, seqlen=6, spanum=3):

        self.model.train()
        self.modelSatn.train()
        self.modelTatn.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        precisions = AverageMeter()

        end = time.time()
        for i, inputs in enumerate(data_loader):
            data_time.update(time.time() - end)

            inputs, targets = self._parse_data(inputs)

            conv_feat = self.model(*inputs)
            satn_out,reg = self.modelSatn(conv_feat)
            tatn_out = self.modelTatn(satn_out)

            loss, prec1 = self._forward(tatn_out, targets, reg)

            losses.update(loss.data[0], targets.size(0))
            precisions.update(prec1, targets.size(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})\t'
                      'Prec {:.2%} ({:.2%})\t'
                      .format(epoch, i + 1, len(data_loader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg,
                              precisions.val, precisions.avg))

    def _parse_data(self, inputs):
        raise NotImplementedError

    def _forward(self, inputs, targets):
        raise NotImplementedError


class Trainer(BaseTrainer):
    def _parse_data(self, inputs):
        imgs, fnames, pids, _, _ = inputs
        imgs = imgs.view(-1,self.args.sampleSeqLength,3,imgs.size(2), imgs.size(3))
        inputs = [Variable(imgs)]
        targets = Variable(pids.cuda())
        return inputs, targets

    def _forward(self, inputs, targets, reg):

        outputs = inputs
        if isinstance(self.criterion, torch.nn.CrossEntropyLoss):
            loss = self.criterion(outputs, targets)
            prec, = accuracy(outputs.data, targets.data)
            prec = prec[0]
        elif isinstance(self.criterion, OIMLoss):
            loss, outputs = self.criterion(outputs, targets)
            prec, = accuracy(outputs.data, targets.data)
            prec = prec[0]
        elif isinstance(self.criterion, TripletLoss):
            loss, prec = self.criterion(outputs, targets)
        else:
            raise ValueError("Unsupported loss:", self.criterion)

        reg = torch.sqrt(reg)
        reg = torch.bmm(reg, reg.transpose(1,2))
        reg = reg-Variable(torch.eye(reg.size(1)).expand_as(reg).cuda())
        reg = torch.pow(reg, 2)
        reg = torch.sum(reg)+1e-5
        reg = torch.sqrt(reg)
        reg = torch.mean(reg)
        loss = loss+reg

        return loss, prec
