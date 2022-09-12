import os
import torch
import torch.nn as nn
import numpy as np
from model import VisionTransformer
import torchvision.transforms as transforms
from config import get_train_config
from checkpoint import load_checkpoint
from data_loaders import *
from utils import setup_device, accuracy, MetricTracker, TensorboardWriter
from data_utils import DataLoader
from sklearn.metrics import *
import torch.utils.data as data
import glob
import pdb
import time


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"


loss_func_mse = nn.MSELoss(reduction='mean')


# def loss_functions(x, x_hat):
#     weights = 0.1
#     x, x_hat= x.cpu(), x_hat.cpu()

#     recons_loss = torch.mean(loss_func_mse(x, x_hat))

#     # kld_loss = torch.mean(-0.5 * torch.sum(1 + sigma - mu ** 2 - sigma.exp(), dim = 1), dim = 0)
#     # total_loss = recons_loss

#     return recons_loss


def train_epoch(epoch, model, data_loader, num_frames, optimizer, lr_scheduler, metrics, device=torch.device('cpu')):
    metrics.reset()
    average_loss = []
    # training loop
    for batch_idx, (batch_data) in enumerate(data_loader):
        batch_data = batch_data.to(device)
        # batch_data = Variable(batch_data).cuda()
        # batch_target = batch_target.to(device)

        optimizer.zero_grad()
        batch_pred = model(batch_data[:,:num_frames])

        loss = loss_func_mse(batch_data[:,num_frames].float(), batch_pred)
        # loss = torch.mean(batch_pred)
        # neptune.log_metric('radius_train', radius_t.item())
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        
        # batch_target = torch.nn.functional.one_hot(batch_target, num_classes=25)
        # acc1, acc5 = accuracy(batch_pred, batch_target, topk=(1, 5))

        metrics.writer.set_step((epoch - 1) * len(data_loader) + batch_idx)
        metrics.update('loss', loss.item())
        # metrics.update('acc1', acc1.item())
        # metrics.update('acc5', acc5.item())
        # OA1.append(acc1.item())
        # OA2.append(acc5.item())
        average_loss.append(loss.item())

        if batch_idx % 100 == 0:
            print("Train Epoch: {:03d} Batch: {:05d}/{:05d} Reconstruction Loss: {:.4f}"
                    .format(epoch, batch_idx, len(data_loader), np.mean(average_loss)))
    return metrics.result()


def valid_epoch(model, data_loader, valid_label, num_frames, device=torch.device('cpu')):
    # metrics.reset()
    losses = []
    acc1s = []
    acc5s = []
    # validation loop
    new_label_path = valid_label
    new_label = np.load(new_label_path)
    with torch.no_grad():
        for batch_idx, (batch_data) in enumerate(data_loader):
            batch_data = batch_data.to(device)
            # batch_target = batch_target.to(device)
            batch_pred = model(batch_data[:, :num_frames])

            loss = loss_func_mse(batch_data[:, num_frames].float(), batch_pred)
            # loss = torch.mean(batch_pred)
            losses.append(loss.item())

            # acc1, acc5 = accuracy(batch_pred, batch_target, topk=(1, 5))

            # losses.append(loss.item())
            # acc1s.append(acc1.item())
            # acc5s.append(acc5.item())

    loss = np.mean(losses)
    frame_auc = roc_auc_score(y_true=new_label[len(new_label) - len(losses):], y_score=losses)
    
    # np.save('loss_%s.npy' % (time.strftime('%d_%H_%M_%S', time.localtime())), losses)
    # acc1 = np.mean(acc1s)
    # acc5 = np.mean(acc5s)
    # metrics.writer.set_step(epoch, 'valid')
    # metrics.update('loss', loss)
    # metrics.update('acc1', frame_auc)
    # metrics.update('acc5', acc5)
    print("AUC@1: {:.2f}".format(frame_auc))
    # return metrics.result()


def save_model(save_dir, epoch, model, optimizer, lr_scheduler, device_ids, best=False):
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict() if len(device_ids) <= 1 else model.module.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict(),
    }
    filename = str('/home/user/anomaly/final_codes/vision-transformer-pytorch-main/check_points/' + 'current.pth')
    torch.save(state, filename)

    if best:
        filename = str('/home/user/anomaly/final_codes/vision-transformer-pytorch-main/check_points/' + 'best.pth')
        torch.save(state, filename)


def data_provider(path):
    # Loading dataset
    config = get_train_config()
    # train_folder = "/home/user/anomaly/dataset/train/"
    # train_dataset = DataLoader(train_folder, transforms.Compose([
    #     transforms.ToTensor(),
    # ]), resize_height=config.image_size, resize_width=config.image_size)
    #
    # train_size = len(train_dataset)
    # print('train size: %d' % train_size)
    # train_batch = data.DataLoader(train_dataset, batch_size=config.batch_size,
    #                               shuffle=True, num_workers=4, drop_last=True)
    scene_names = glob.glob(os.path.join(path, '*'))
    scene_names.sort()
    for s in range(len(scene_names)):
        scene = scene_names[s]
        if s == 0:
            print('--------%s--------' % scene)
            sequence_names = glob.glob(os.path.join(scene, '*'))
            for sequence in sequence_names:
                print('    --------%s--------' % sequence)
                data_paths = glob.glob(os.path.join(sequence, '*'))
                train_paths = data_paths[0]
                
                test_labels = glob.glob(os.path.join(data_paths[1], '*.npy'))
                test_labels.sort(key=lambda x: int(x[len(data_paths[1])+1:-4]))
                test_paths = []
                for root, dirnames, filenames in os.walk(data_paths[1]):
                    for dirname in dirnames:
                        test_paths.append(os.path.join(root, dirname))
                test_paths.sort(key=lambda x: int(x[-2:]))
                print('        --------training--------')
                
                train_batch = load_dataset(train_paths, config.batch_size, 4, config.image_size, config.num_frames)
                train(train_batch, test_paths, test_labels)
                # for t in range(len(test_paths)):
                #     print('        --------testing_%d--------' % t)
                #     test_batch = load_dataset(test_paths[t], 1, 0, config.image_size, config.num_frames)
                #     test(train_model, test_batch, test_labels[t], config.num_frames, device)


def load_dataset(data_path, batch_size, num_workers, image_size, num_frames):

    # Loading dataset
    dataset = DataLoader(data_path, transforms.Compose([transforms.ToTensor(),]), resize_height=image_size, resize_width=image_size, time_step=num_frames)

    data_size = len(dataset)

    data_batch = data.DataLoader(dataset, batch_size=batch_size,
                                  shuffle=False, num_workers=num_workers, drop_last=False)
    print('dataload!')
    print('number of frames: %d' % data_size)

    return data_batch


def train(train_batch, test_paths, test_labels):
    config = get_train_config()

    # device
    device, device_ids = setup_device(config.n_gpu)

    # tensorboard
    writer = TensorboardWriter(config.summary_dir, config.tensorboard)

    # metric tracker
    metric_names = ['loss', 'acc1', 'acc5']
    train_metrics = MetricTracker(*[metric for metric in metric_names], writer=writer)
    valid_metrics = MetricTracker(*[metric for metric in metric_names], writer=writer)
    # test_folder = "/home/user/anomaly/dataset/test/"
    # test_dataset = DataLoader(test_folder, transforms.Compose([
    #     transforms.ToTensor(),
    # ]), resize_height=config.image_size, resize_width=config.image_size)
    #
    # test_size = len(test_dataset)
    # print('test size: %d' % test_size)
    # test_batch = data.DataLoader(test_dataset, batch_size=1,
    #                              shuffle=False, num_workers=4, drop_last=True)
    # print('dataload!')
    # create model
    print("create model")
    model = VisionTransformer(
        image_size=(config.image_size, config.image_size),
        patch_size=(config.patch_size, config.patch_size),
        emb_dim=config.emb_dim,
        mlp_dim=config.mlp_dim,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        num_classes=config.num_classes,
        attn_dropout_rate=config.attn_dropout_rate,
        dropout_rate=config.dropout_rate,
        num_frames=config.num_frames)

    # load checkpoint
    if config.checkpoint_path:
        state_dict = load_checkpoint(config.checkpoint_path)
        # print(state_dict.keys())
        if config.num_classes != state_dict['classifier.weight'].size(0):
            del state_dict['classifier.weight']
            del state_dict['classifier.bias']
            del state_dict['transformer.pos_embedding.pos_embedding']
            # del state_dict['transformer.pos_embedding.bias']
            print("re-initialize fc layer")
            print("re-initialize pos_embedding layer")
            model.load_state_dict(state_dict, strict=False)
        else:
            model.load_state_dict(state_dict)
        print("Load pretrained weights from {}".format(config.checkpoint_path))

    # send model to device
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    # training criterion
    print("create criterion and optimizer")
    criterion = nn.CrossEntropyLoss()

    # create optimizers and learning rate scheduler
    optimizer = torch.optim.SGD(
        params=model.parameters(),
        lr=config.lr,
        weight_decay=config.wd,
        momentum=0.9)
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer=optimizer,
        max_lr=config.lr,
        pct_start=config.warmup_steps / config.train_steps,
        total_steps=config.train_steps)

    # start training
    print("start training")
    best_acc = 0.0
    log = {}
    log['val_acc1'] = 0
    # epochs = config.train_steps // len(train_dataloader)
    epochs = config.epochs
    # model.train()
    for epoch in range(1, epochs + 1):
        log['epoch'] = epoch

        # train the model
        model.train()
        result = train_epoch(epoch, model, train_batch, config.num_frames, optimizer, lr_scheduler, train_metrics, device)
        log.update(result)
    model.eval()
    # val_label = '/home/user/anomaly/dataset/new_labels.npy'
    # valid_epoch(model, test_batch, val_label, config.num_frames, device)
    for t in range(len(test_paths)):
        print('        --------testing_%d--------' % t)
        model.eval()
        test_batch = load_dataset(test_paths[t], 1, 0, config.image_size, config.num_frames)
        # test(model, test_batch, test_labels[t], config.num_frames, device)
        valid_epoch(model, test_batch, test_labels[t], config.num_frames, device)


def test(model, test_batch, test_label, num_frames, device):
    # validate the model
    model.eval()
    valid_epoch(model, test_batch, test_label, num_frames, device)
    # log.update(**{'val_' + k: v for k, v in result.items()})
    # # best acc
    # best = False
    # if log['val_acc1'] > best_acc:
    #     best_acc = log['val_acc1']
    #     best = True
    #
    # # save model
    # # save_model(config.checkpoint_dir, epoch, model, optimizer, lr_scheduler, device_ids, best)
    #
    # # print logged informations to the screen
    # for key, value in log.items():
    #     print('    {:15s}: {}'.format(str(key), value))


def main():
    path = '/home/user/anomaly/final_data/Drone-Anomaly/'
    data_provider(path)


if __name__ == '__main__':
    main()
