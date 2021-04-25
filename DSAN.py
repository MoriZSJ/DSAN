from __future__ import print_function
import torch
import torch.nn.functional as F
import os
import math
import data_loader
import ResNet as models
from Config import *
import time
from torch.utils.tensorboard import SummaryWriter

cuda = True
# torch.manual_seed(seed)
# if cuda:
#     torch.cuda.manual_seed(seed)

kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

source_loader = data_loader.load_training(root_path, source_name, batch_size, kwargs)
target_train_loader = data_loader.load_training(root_path, target_name, batch_size, kwargs)
target_test_loader = data_loader.load_testing(root_path, target_name, batch_size, kwargs)

len_source_dataset = len(source_loader.dataset)
len_target_dataset = len(target_test_loader.dataset)
len_source_loader = len(source_loader)
len_target_loader = len(target_train_loader)


def train(epoch, model, logger):
    LEARNING_RATE = lr / math.pow((1 + 10 * (epoch - 1) / epochs), 0.75)
    print('learning rate{: .4f}'.format(LEARNING_RATE))
    if bottle_neck:
        optimizer = torch.optim.SGD([
            {'params': model.feature_layers.parameters()},
            {'params': model.bottle.parameters(), 'lr': LEARNING_RATE},
            {'params': model.cls_fc.parameters(), 'lr': LEARNING_RATE},
        ], lr=LEARNING_RATE / 10, momentum=momentum, weight_decay=l2_decay)
    else:
        optimizer = torch.optim.SGD([
            {'params': model.feature_layers.parameters()},
            {'params': model.cls_fc.parameters(), 'lr': LEARNING_RATE},
        ], lr=LEARNING_RATE / 10, momentum=momentum, weight_decay=l2_decay)

    model.train()

    iter_source = iter(source_loader)
    iter_target = iter(target_train_loader)
    num_iter = len_source_loader
    loss_sum, loss_cls_sum, loss_mmd_sum = 0, 0, 0
    for i in range(1, num_iter):
        data_source, label_source = iter_source.next()
        data_target, _ = iter_target.next()
        if i % len_target_loader == 0:
            iter_target = iter(target_train_loader)
        if cuda:
            data_source, label_source = data_source.cuda(), label_source.cuda()
            data_target = data_target.cuda()

        optimizer.zero_grad()
        label_source_pred, loss_mmd = model(data_source, data_target, label_source)
        loss_cls = F.nll_loss(F.log_softmax(label_source_pred, dim=1), label_source)
        lambd = 2 / (1 + math.exp(-10 * (epoch) / epochs)) - 1
        loss = loss_cls + param * lambd * loss_mmd

        loss_sum += loss.item()
        loss_cls_sum += loss_cls.item()
        loss_mmd_sum += loss_mmd.item()

        loss.backward()
        optimizer.step()
        if i % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tsoft_Loss: {:.6f}\tlmmd_Loss: {:.6f}'.format(
                epoch, i * len(data_source), len_source_dataset, 100. * i / len_source_loader, loss.item(), loss_cls.item(), loss_mmd.item()))
    # logging to tensorboard
    logger.add_scalar('train_loss/loss_total', loss_sum/len_source_dataset, epoch)
    logger.add_scalar('train_loss/loss_cls', loss_cls_sum/len_source_dataset, epoch)
    logger.add_scalar('train_loss/loss_lmmd', loss_mmd_sum/len_source_dataset, epoch)


def test(model, logger):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in target_test_loader:
            if cuda:
                data, target = data.cuda(), target.cuda()
            s_output, t_output = model(data, data, target)
            # sum up batch loss
            test_loss += F.nll_loss(F.log_softmax(s_output, dim=1), target).item()
            # get the index of the max log-probability
            pred = s_output.data.max(1)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        test_loss /= len_target_dataset
        print('\n{} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            target_name, test_loss, correct, len_target_dataset, 100. * correct / len_target_dataset))

        logger.add_scalar('test_loss', test_loss, epoch)
        logger.add_scalar('test_acc', correct/len_target_dataset, epoch)

    return correct


if __name__ == '__main__':
    # logger = SummaryWriter(log_dir=tensorboard_path+source_name+'To'+target_name)
    # model = models.DSAN(num_classes=class_num)
    # correct = 0
    # # print(model)
    # if cuda:
    #     model.cuda()
    # time_start = time.time()
    # for epoch in range(1, epochs + 1):
    #     train(epoch, model, logger)
    #     t_correct = test(model, logger)
    #     if t_correct > correct:
    #         correct = t_correct
    #         torch.save(model, 'model.pkl')
    #     end_time = time.time()
    #     print('source: {} to target: {} max correct: {} max accuracy{: .2f}%\n'.format(
    #           source_name, target_name, correct, 100. * correct / len_target_dataset))
    #     print('cost time:', end_time - time_start)
    # # torch.save(model, 'model_last.pkl')
    # logger.flush()
    # logger.close()

    '''for testing model'''
    logger = SummaryWriter(log_dir=tensorboard_path+source_name+'To'+target_name)
    model_path = '/home/mori/Programming/DSAN/save_models/office31/d2w/RN50_acc9862.pkl'
    model = torch.load(model_path).cuda()
    correct = test(model, logger)
    logger.flush()
    logger.close()
