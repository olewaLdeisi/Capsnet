import argparse
import os
import shutil
import logging
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim import lr_scheduler
from torch.autograd import Variable
# import torchnet.meter as meter
import torchnet
import pandas as pd

# from OldCapsNet import CapsNet, ReconstructionNet, CapsNetWithReconstruction, MarginLoss
from CapsNet import CapsNet, ReconstructionNet, CapsNetWithReconstruction, MarginLoss
from focalloss import FocalLoss
from utils import *
from DoubleBranch.DoubleBranch import *
from dataset import Dataset
import visdom

# Training settings
parser = argparse.ArgumentParser(description='CapsNet with Glaucoma')
parser.add_argument('--batch_size', type=int, default=10, metavar='N', help='input batch size for training (default: 32)')
parser.add_argument('--test_batch_size', type=int, default=32, metavar='N', help='input batch size for testing (default: 32)')
parser.add_argument('--epochs', type=int, default=100, metavar='N', help='number of epochs to train (default: 200)')
parser.add_argument('--lr', type=float, default=0.0001, metavar='LR', help='learning rate (default: 0.0001)')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--log_interval', type=int, default=1, metavar='N', help='how many batches to wait before logging training status')
parser.add_argument('--routing_iterations', type=int, default=3)
parser.add_argument('--reconstruction_alpha', type=float, default=0.0005, help='reconstruction alpha')
parser.add_argument('--weight_decay', default=1e-4, type=float, metavar='M', help='weight decay')
parser.add_argument('--input_size', type=int, default=32, help='imgsize')
parser.add_argument('--output_size', type=int, default=28, help='imgsize')
parser.add_argument('--block_size', type=int, default=3, help='block_size')
parser.add_argument('--drop_prob', type=float, default=0.2, help='drop_prob')
parser.add_argument('--with_reconstruction', action='store_true', default=True)
parser.add_argument('--with_5c', action='store_true', default=False, help='use the 5 channels image')
parser.add_argument('--block', action='store_true', default=False)
parser.add_argument('--data_repeat', action='store_true', default=False)
parser.add_argument('--no_cuda', action='store_true', default=False, help='disables CUDA training')
parser.add_argument('--save_path', type=str, default='.', help='save_path')
parser.add_argument('--model',type=str,default='CapsNet',help='model_name')
parser.add_argument('--class_num',type=int,default=2,help='分类数量')
parser.add_argument('--dropchannel', action='store_true', default=False, help='use dropchannel')
parser.add_argument('--focalloss', action='store_true', default=False, help='use focalloss')
parser.add_argument('--focalgamma', type=int, default=5, help='focalgamma')
parser.add_argument('--with_4c', action='store_true', default=False, help='use the 4 channels image')


best_prec = 0


def main():
    global args, best_prec
    logging.basicConfig(filename='loss.log', level=logging.INFO, filemode='a',
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s', datefmt='%m-%d %H:%M', )
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    logging.info(args)
    print('基本配置信息:\n{}'.format(args))

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    # Load data
    # if args.with_5c:
    #     dataroot = './data'
    # else:
    dataroot = './data'
    train_loader, test_loader = load_data(dataroot)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    # create model
    # model = CapsNet(args.routing_iterations, 2,block_size=args.block_size,drop_prob=args.drop_prob,img_size=args.output_size,block=args.block)
    if args.model == 'CapsNet':
        model = CapsNet(args.routing_iterations, 2,block_size=args.block_size,drop_prob=args.drop_prob,img_size=args.output_size,block=args.block)
        if args.with_5c:
            model.conv1 = nn.Conv2d(5, 256, kernel_size=9, stride=1)
        elif args.with_4c:
            model.conv1 = nn.Conv2d(4, 256, kernel_size=9, stride=1)
        if args.with_reconstruction:
            reconstruction_model = ReconstructionNet(16, 2,img_size=args.output_size, dropchannel=args.dropchannel)
            if args.with_5c:
                reconstruction_model.fc1 = nn.Linear(16 * 2, 2048)
                reconstruction_model.fc2 = nn.Linear(2048, 4096)
                # reconstruction_model.fc3 = nn.Linear(4096, 28 * 28 * 5)
                reconstruction_model.fc3 = nn.Linear(4096, args.output_size * args.output_size * 5)
            elif args.with_4c:
                reconstruction_model.fc1 = nn.Linear(16 * 2, 2048)
                reconstruction_model.fc2 = nn.Linear(2048, 4096)
                # reconstruction_model.fc3 = nn.Linear(4096, 28 * 28 * 5)
                reconstruction_model.fc3 = nn.Linear(4096, args.output_size * args.output_size * 4)
            model = CapsNetWithReconstruction(model, reconstruction_model)

    elif args.model == 'DoubleBranch':
        model = DoubleBranch()

    if args.cuda:
        model.cuda()

    # optimizer and loss function
    # optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.5,0.999))
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    # scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True, patience=10, min_lr=1e-6)
    # scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True,
    #                                           threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=1e-08, eps=1e-08)
    if args.focalloss:
        loss_fn = FocalLoss(gamma=args.focalgamma)
    else:
        loss_fn = MarginLoss(0.9, 0.1, 0.5)

    env_name = "Caspnet" + "_imgSize=" + str(args.output_size)
    if args.with_5c:
        env_name += "_with_5c"
    elif args.with_4c:
        env_name += "_with_4c"
    if args.drop_prob > 0:
        env_name += "_drop"
    if args.block:
        env_name += "block"
    if args.data_repeat:
        env_name += "_dataRepeat"
    env_name += "_class_num=" + str(args.class_num)
    if args.dropchannel:
        env_name += '__dropchannel'
    viz = visdom.Visdom(env=env_name)

    best_model_wts = model.state_dict()
    best_auc = 0.0, 0.0
    best_acc = 0.0
    best_epoch = 0

    # loss
    train_epoch_loss_list = []
    val_epoch_loss_list = []
    # auc
    train_epoch_auc_list = []
    val_epoch_auc_list = []
    # B-acc
    train_epoch_acc_list = []
    val_epoch_acc_list = []
    # sensitivity
    train_epoch_sen_list = []
    val_epoch_sen_list = []
    # specificity
    train_epoch_spe_list = []
    val_epoch_spe_list = []

    loss_dict = defaultdict(list)
    accuracy_dict = defaultdict(list)
    for epoch in range(args.epochs):
        exp_lr_scheduler.step()
        train_epoch_loss, train_epoch_acc, train_auc, train_epoch_sen, train_epoch_spe = train(train_loader, model, loss_fn, optimizer, epoch)
        val_epoch_loss, val_epoch_acc, auc_values, val_epoch_sen, val_epoch_spe = test(test_loader, model, loss_fn)
        print("val_epoch_acc, val_epoch_sen, val_epoch_spe: ", val_epoch_acc, val_epoch_sen, val_epoch_spe)
        if best_auc[0] < auc_values[0]:
            best_auc = auc_values
            best_acc = val_epoch_acc
            best_epoch = epoch
            best_model_wts = model.state_dict()
        # scheduler.step(test_loss)

        # if (args.with_5c and (test_accuarcy >= 84 and auc_values[0] >= 0.75) or auc_values[0] >= 0.868) or (test_accuarcy >= 80 and auc_values[0] >= 0.75):
        #     torch.save(model.state_dict(),
        #                args.save_path + '/' + '{:03d}_model_dict_5c{}_acc{:.2f}_auc{:.4f}.pth'.format(epoch, args.with_5c, test_accuarcy, auc_values[0]))
        #     save_auc(auc_values, filename=args.save_path + '/' + '{:03d}_AUC{}.png'.format(epoch, auc_values[0]))

        train_epoch_auc = train_auc[0]
        val_epoch_auc = auc_values[0]
        train_epoch_loss_list.append(train_epoch_loss)
        val_epoch_loss_list.append(val_epoch_loss)
        train_epoch_acc_list.append(train_epoch_acc)
        val_epoch_acc_list.append(val_epoch_acc)
        train_epoch_sen_list.append(train_epoch_sen)
        val_epoch_sen_list.append(val_epoch_sen)
        train_epoch_spe_list.append(train_epoch_spe)
        val_epoch_spe_list.append(val_epoch_spe)
        train_epoch_auc_list.append(train_epoch_auc)
        val_epoch_auc_list.append(val_epoch_auc)
        if epoch == 0:
            win = viz.line(X=torch.Tensor([[epoch] * 10]), Y=torch.Tensor([[train_epoch_loss, val_epoch_loss,
                                                                            train_epoch_acc, val_epoch_acc,
                                                                            train_epoch_sen, val_epoch_sen,
                                                                            train_epoch_spe, val_epoch_spe,
                                                                            train_epoch_auc, val_epoch_auc]]),
                           opts=dict(legend=['train_epoch_loss', 'val_epoch_loss', 'train_epoch_acc', 'val_epoch_acc',
                                             'train_epoch_sen', 'val_epoch_sen', 'train_epoch_spe', 'val_epoch_spe',
                                             'train_epoch_auc', 'val_epoch_auc']))
        else:
            viz.line(X=torch.Tensor([[epoch] * 10]), Y=torch.Tensor([[train_epoch_loss, val_epoch_loss, train_epoch_acc,
                                                                      val_epoch_acc, train_epoch_sen, val_epoch_sen,
                                                                      train_epoch_spe, val_epoch_spe, train_epoch_auc,
                                                                      val_epoch_auc]]), opts=dict(
                legend=['train_epoch_loss', 'val_epoch_loss', 'train_epoch_acc', 'val_epoch_acc', 'train_epoch_sen',
                        'val_epoch_sen', 'train_epoch_spe', 'val_epoch_spe', 'train_epoch_auc', 'val_epoch_auc']),
                     win=win, update='append')

        loss_dict['train loss'].append(train_epoch_loss)
        loss_dict['test loss'].append(val_epoch_loss)
        accuracy_dict['train accuracy'].append(float(train_epoch_acc))
        accuracy_dict['test accuracy'].append(float(val_epoch_acc))

        # is_best = test_accuarcy > best_prec
        # best_prec = max(test_accuarcy, best_prec)
        # save_checkpoint({
            # 'epoch': epoch,
            # 'state_dict': model.state_dict(),
            # 'best_prec': best_prec,
            # 'optimizer': optimizer.state_dict(),
        # }, is_best)

    dataframe = pd.DataFrame({'train_epoch_loss':train_epoch_loss_list,
                              'val_epoch_loss':val_epoch_loss_list,
                              'train_epoch_acc':train_epoch_acc_list,
                              'val_epoch_acc':val_epoch_acc_list,
                              'train_epoch_auc':train_epoch_auc_list,
                              'val_epoch_auc':val_epoch_auc_list,
                              'train_epoch_sen':train_epoch_sen_list,
                              'val_epoch_sen':val_epoch_sen_list,
                              'train_epoch_spe':train_epoch_spe_list,
                              'val_epoch_spe':val_epoch_spe_list
                              },index=[i for i in range(len(train_epoch_loss_list))])
    dataframe.to_csv(args.save_path + '/' + '{}_5c{}_4c{}_log.csv'.format(args.epochs, args.with_5c, args.with_4c))
    # csv_dict = {**loss_dict, **accuracy_dict}
    # csv_df = pd.DataFrame(csv_dict)
    # csv_df.to_csv(args.save_path + '/' + '{}_5c{}_log.csv'.format(args.epochs, args.with_5c))
    save_plot(loss_dict,save_path=args.save_path)
    save_plot(accuracy_dict,save_path=args.save_path)
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(),
               args.save_path + '/' + '{:03d}_model_dict_5c{}_4c{}_acc{:.2f}_auc{:.4f}.pth'.format(best_epoch, args.with_5c, args.with_4c,
                                                                                              best_acc, best_auc[0]))
    save_auc(best_auc, filename=args.save_path + '/' + '{:03d}_AUC{}.png'.format(best_epoch, best_auc[0]))


def load_data(path):
    """Load data from path by ImageFloder

    Args:
        path: path which have train and val folder

    Return:
        train_loader: The DataLoader of train dataset
        test_loader:  The DataLoader of test dataset
    """
    kwargs = {'num_workers': 4, 'pin_memory': True, 'drop_last': True} if args.cuda else {'drop_last': True}

    # normalize = transforms.Normalize((0.957, 0.647, 0.349), (0.080, 0.148, 0.153))
    # normalize = transforms.Normalize((0.640, 0.435, 0.240, 0.440), (0.475, 0.342, 0.214, 0.380))
    train_transform = transforms.Compose([
        transforms.Resize(args.input_size),
        transforms.RandomCrop(args.output_size),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # normalize,
    ])
    center_transform = transforms.Compose([
        transforms.Resize(args.input_size),
        transforms.CenterCrop(args.output_size),
        transforms.ToTensor(),
        # normalize,
    ])
    # train_set = Dataset(class_num=2, data_path=os.path.join(path, 'train.txt'),
    if args.data_repeat:
        train_set = Dataset(class_num=2, data_path=os.path.join(path, 'train_new.txt'),
                file_path=path, grayscale=False, p=0.5,transform=train_transform)
    else:
        train_set = Dataset(class_num=2, data_path=os.path.join(path, 'train.txt'),
                file_path=path, grayscale=False, p=0.5,transform=train_transform)
    train_loader = torch.utils.data.DataLoader(train_set,
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_set = Dataset(class_num=2, data_path=os.path.join(path, 'test.txt'),
                      file_path=path, grayscale=False, transform=center_transform)
    test_loader = torch.utils.data.DataLoader(test_set,
        batch_size=args.test_batch_size, shuffle=False, **kwargs)
    return train_loader, test_loader


def train(train_loader, model, criterion, optimizer, epoch):
    """Train the model

    Args:
        train_loader: The DataLoader of train dataset
        model: The Model to train
        criterion: to compute the margin loss
        optimizer: optimizer for model
        epoch: the current epoch to train

    Return:
        losses.avg: The average loss of train loss
    """
    model.train()
    losses = AverageMeter()
    # total_correct = 0
    train_running_corrects = torch.Tensor([0] * args.class_num)
    train_running_total = torch.Tensor([0] * args.class_num)
    # print("train_running_corrects, train_running_total :", train_running_corrects, train_running_total)
    # mtr = meter.AUCMeter()
    mtr = torchnet.meter.AUCMeter()
    if args.with_5c:
        input_img = torch.Tensor(args.batch_size, 5, args.output_size, args.output_size)
    elif args.with_4c:
        input_img = torch.Tensor(args.batch_size, 4, args.output_size, args.output_size)
    else:
        input_img = torch.Tensor(args.batch_size, 3, args.output_size, args.output_size)
    input_target = torch.LongTensor(args.batch_size)
    if args.cuda:
        input_img, input_target = input_img.cuda(), input_target.cuda()

    for batch_idx, (data,_,target) in enumerate(train_loader):
        if args.with_5c:
            data = input_img.copy_(torch.cat((data, *_), 1))
        elif args.with_4c:
            data = input_img.copy_(torch.cat((data, _[0]), 1))
        else:
            data = input_img.copy_(data)
        target = input_target.copy_(target)
        data_var, target_var = Variable(data), Variable(target)

        if args.with_reconstruction:
            output, probs = model(data_var, target_var)
            bs, c, h, w = data.size()
            reconstruction_loss = F.mse_loss(output, data_var.view(-1, c * h * w))
            margin_loss = criterion(probs, target_var)
            loss = args.reconstruction_alpha * reconstruction_loss + margin_loss
        else:
            output, probs = model(data_var)
            loss = criterion(probs, target_var)

        losses.update(loss.item(), data.size(0))
        smax_probs = F.softmax(probs, dim=1)
        mtr.add(smax_probs.data[:, 1], target.cpu()) # AUC

        pred = probs.data.max(1, keepdim=True)[1]

        # 各类别准确数和总数

        # print(f"Epoch{epoch}, {pred.view_as(target)}, {target}")
        train_running_corrects += torch.Tensor([float(torch.sum((pred.view_as(target) == j) & (target == j))) for j in range(args.class_num)])
        train_running_total += torch.Tensor([float(torch.sum(target == j)) for j in range(args.class_num)])
        # print(f"Epoch{epoch} train_running_corrects, train_running_total :", train_running_corrects, train_running_total)

        # correct = pred.eq(target.view_as(pred)).sum()
        # total_correct += correct

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, (batch_idx + 1) * len(data), len(train_loader.dataset),
                100 * (batch_idx + 1) / len(train_loader), loss.item()))

    TP = train_running_corrects[1]
    TN = train_running_corrects[0]
    FP = train_running_total[0] - train_running_corrects[0]
    FN = train_running_total[1] - train_running_corrects[1]
    # print("TP, TN, FP, FN : ", TP, TN, FP, FN)
    train_epoch_acc = float(Baccuracy(TP, TN, FP, FN))
    train_epoch_sen = float(recall(TP, TN, FP, FN))
    train_epoch_spe = float(specificity(TP, TN, FP, FN))

    # train_epoch_auc = float(train_mtr.value()[0])

    # accuracy = 100. * float(total_correct) / float(len(train_loader.dataset))
    mtr_values = mtr.value()
    # print('\nTrain set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%), AUC: {:.4f}'.format(
    #     losses.avg, TP + TN, len(train_loader.dataset), accuracy, mtr_values[0]))

    return losses.avg, train_epoch_acc, mtr_values, train_epoch_sen, train_epoch_spe


def test(test_loader, model, criterion):
    """Test the model

    Args:
        test_loader: The DataLoader of test dataset
        model: The model to test
        criterion: to compute the margin loss

    Return:
        losses.avg: The average loss of test
        accuracy: The accuracy of test, computed by total_correct / len(test_loader.datasets)
    """
    model.eval()
    losses = AverageMeter()
    # total_correct = 0
    val_running_corrects = torch.Tensor([0] * args.class_num)
    val_running_total = torch.Tensor([0] * args.class_num)
    # mtr = meter.AUCMeter()
    mtr = torchnet.meter.AUCMeter()
    if args.with_5c:
        input_img = torch.Tensor(args.test_batch_size, 5, args.output_size, args.output_size)
    elif args.with_4c:
        input_img = torch.Tensor(args.test_batch_size, 4, args.output_size, args.output_size)
    else:
        input_img = torch.Tensor(args.test_batch_size, 3, args.output_size, args.output_size)
    input_target = torch.LongTensor(args.test_batch_size)
    if args.cuda:
        input_img, input_target = input_img.cuda(), input_target.cuda()

    for data,_, target in test_loader:
        if args.with_5c:
            data = input_img.copy_(torch.cat((data, *_), 1))
        elif args.with_4c:
            data = input_img.copy_(torch.cat((data, _[0]), 1))
        else:
            data = input_img.copy_(data)
        target = input_target.copy_(target)
        with torch.no_grad():
            data_var = Variable(data)
        target_var = Variable(target)

        if args.with_reconstruction:
            output, probs = model(data_var, target_var)
            bs, c, h, w = data.size()
            reconstruction_loss = F.mse_loss(output, data_var.view(-1, c * h * w))
            margin_loss = criterion(probs, target_var)
            loss = args.reconstruction_alpha * reconstruction_loss + margin_loss
        else:
            output, probs = model(data_var)
            loss = criterion(probs, target_var)

        smax_probs = F.softmax(probs, dim=1)
        mtr.add(smax_probs.data[:, 1], target.cpu())

        pred = probs.data.max(1, keepdim=True)[1]

        # 各类别准确数和总数
        val_running_corrects += torch.Tensor([float(torch.sum((pred.view_as(target) == j) & (target == j))) for j in range(args.class_num)])
        val_running_total += torch.Tensor([float(torch.sum(target == j)) for j in range(args.class_num)])

        # correct = pred.eq(target.view_as(pred)).sum()
        # total_correct += correct
        losses.update(loss.item(), data.size(0))

    TP = val_running_corrects[1]
    TN = val_running_corrects[0]
    FP = val_running_total[0] - val_running_corrects[0]
    FN = val_running_total[1] - val_running_corrects[1]
    val_epoch_acc = float(Baccuracy(TP, TN, FP, FN))
    val_epoch_sen = float(recall(TP, TN, FP, FN))
    val_epoch_spe = float(specificity(TP, TN, FP, FN))
    # val_epoch_auc = float(val_mtr.value()[0])

    # accuracy = 100. * float(total_correct )/ float(len(test_loader.dataset))
    mtr_values = mtr.value()
    # print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%), AUC: {:.4f}\n'.format(
    #     losses.avg, TP + TN, len(test_loader.dataset), accuracy, mtr_values[0]))
    if str(val_epoch_sen) == 'nan':
        print("nan:TP, TN, FP, FN: ", TP, TN, FP, FN)
        print("val_running_total, val_running_corrects: ", val_running_total, val_running_corrects)

    return losses.avg, val_epoch_acc, mtr_values, val_epoch_sen, val_epoch_spe


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """保存检查点

        Args:
            state: 要保存的对象
            is_best: 是否在验证集准确率最高
            filename: 保存的文件名
    """
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


if __name__ == '__main__':
    main()