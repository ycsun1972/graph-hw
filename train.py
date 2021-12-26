from ogb_dataloader import PglNodePropPredDataset
import os
import argparse
import time
from functools import partial

import numpy as np
import tqdm
import pgl
import paddle
from pgl.utils.logger import log
from pgl.utils.data import Dataloader
from evaluete import Evaluator

from model import GraphSage
from dataset import ShardedDataset, batch_fn
from ogb_dataloader import PglNodePropPredDataset
import logging

from tb_paddle import SummaryWriter


def add_log_to_file(log_path):
    fh = logging.FileHandler(log_path)
    formatter = logging.Formatter(
        fmt='[%(levelname)s] %(asctime)s [%(filename)12s:%(lineno)5d]:\t%(message)s'
    )
    fh.setFormatter(formatter)
    log.addHandler(fh)

def train(dataloader, model, feature, criterion, optim, log_per_step=100):
    model.train()

    batch = 0
    total_loss = 0.
    total_acc = 0.
    total_sample = 0

    for g, sample_index, index, label in dataloader:

        # import pdb; pdb.set_trace()


        batch += 1
        num_samples = len(index)

        g.tensor()
        sample_index = paddle.to_tensor(sample_index)
        index = paddle.to_tensor(index)
        label = paddle.to_tensor(label)

        feat = paddle.gather(feature, sample_index)

        pred = model(g, feat)
        pred = paddle.gather(pred, index)
        loss = criterion(pred, label)
        loss.backward()
        acc = paddle.metric.accuracy(input=pred, label=label, k=1)
        optim.step()
        optim.clear_grad()

        total_loss += loss.numpy() * num_samples
        total_acc += acc.numpy() * num_samples
        total_sample += num_samples

        if batch % log_per_step == 0:
            log.info("Batch %s %s-Loss %s %s-Acc %s" %
                     (batch, "train", loss.numpy(), "train", acc.numpy()))

    return total_loss / total_sample, total_acc / total_sample


@paddle.no_grad()
def eval(dataloader, model, feature, criterion):
    model.eval()
    loss_all, acc_all = [], []
    for g, sample_index, index, label in dataloader:
        g.tensor()
        sample_index = paddle.to_tensor(sample_index)
        index = paddle.to_tensor(index)
        label = paddle.to_tensor(label)

        feat = paddle.gather(feature, sample_index)
        pred = model(g, feat)
        pred = paddle.gather(pred, index)
        loss = criterion(pred, label)
        acc = paddle.metric.accuracy(input=pred, label=label, k=1)
        loss_all.append(loss.numpy())
        acc_all.append(acc.numpy())

    return np.mean(loss_all), np.mean(acc_all)

@paddle.no_grad()
def eval_ogb(dataloader, model, feature, criterion):
    evaluator = Evaluator('ogbn-products')
    model.eval()
    loss_all, pred_all, label_all = [], [], []
    for g, sample_index, index, label in dataloader:
        g.tensor()
        sample_index = paddle.to_tensor(sample_index)
        index = paddle.to_tensor(index)
        label = paddle.to_tensor(label)

        feat = paddle.gather(feature, sample_index)
        pred = model(g, feat)
        pred = paddle.gather(pred, index)
        loss = criterion(pred, label)
        loss_all.append(loss.numpy())

        _, topk_indices = paddle.topk(pred, k=1)
        pred_all.append(topk_indices.numpy())
        label_all.append(label.numpy())
    # import pdb; pdb.set_trace()

    y_pred = np.vstack(pred_all)
    y_true = np.vstack(label_all)

    input_dict = {'y_true': y_true, 'y_pred': y_pred}
    result = evaluator.eval(input_dict)

    return np.mean(loss_all), result['acc']

    


def main(args):
    if paddle.distributed.get_world_size() > 1:
        paddle.distributed.init_parallel_env()

    dataset = PglNodePropPredDataset(name="ogbn-products")
    split_idx=dataset.get_idx_split() #数据集划分
    graph, label = dataset[0] #graph 是pgl.graph.Graph对象，label为节点的标签


    log.info("Preprocess finish")
    log.info("num of nodes: %s" % str(graph.num_nodes))
    log.info("num of edges: %s" % graph.num_edges)
    log.info("dimension of feature: %s" % str(graph.node_feat["feat"].shape))
    log.info("class of labels: %s" % str(int(max(label))+1))
    log.info("Train Examples: %s" % len(split_idx['train']))
    log.info("Val Examples: %s" % len(split_idx['valid']))
    log.info("Test Examples: %s" % len(split_idx['test']))

    writer = SummaryWriter('./log_tb')
    


    train_index = split_idx['train']
    val_index = split_idx['valid']
    test_index = split_idx['valid']

    train_label = label[train_index]
    val_label = label[val_index]
    test_label = label[test_index]

    model = GraphSage(
        input_size=graph.node_feat["feat"].shape[-1],
        num_class=int(max(label))+1,
        hidden_size=args.hidden_size,
        num_layers=len(args.samples))

    # model = paddle.DataParallel(model)

    criterion = paddle.nn.loss.CrossEntropyLoss()

    optim = paddle.optimizer.Adam(
        learning_rate=args.lr,
        parameters=model.parameters(),
        weight_decay=0.001)

    feature = paddle.to_tensor(graph.node_feat["feat"])

    train_ds = ShardedDataset(train_index, train_label)
    val_ds = ShardedDataset(val_index, val_label)
    test_ds = ShardedDataset(test_index, test_label)

    collate_fn = partial(batch_fn, graph=graph, samples=args.samples)

    train_loader = Dataloader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.sample_workers,
        collate_fn=collate_fn)
    val_loader = Dataloader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.sample_workers,
        collate_fn=collate_fn)
    test_loader = Dataloader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.sample_workers,
        collate_fn=collate_fn)

    cal_val_acc = []
    cal_test_acc = []
    cal_val_loss = []
    for epoch in range(args.epoch):
        
        train_loss, train_acc = train(train_loader, model, feature, criterion,
                                      optim)
        log.info("Runing epoch:%s\t train_loss:%s\t train_acc:%s", epoch,
                 train_loss, train_acc)
        val_loss, val_acc = eval(val_loader, model, feature, criterion)
        cal_val_acc.append(val_acc)
        cal_val_loss.append(val_loss)
        log.info("Runing epoch:%s\t val_loss:%s\t val_acc:%s", epoch, val_loss,
                 val_acc)
        test_loss, test_acc = eval_ogb(test_loader, model, feature, criterion)
        cal_test_acc.append(test_acc)
        log.info("Runing epoch:%s\t val_loss:%s\t val_acc:%s", epoch, test_loss,
                 test_acc)

        writer.add_scalar('train_loss', train_loss, epoch)
        writer.add_scalar('val_loss', val_loss, epoch)
        writer.add_scalar('val_acc', val_acc, epoch)
        writer.add_scalar('test_acc', test_acc, epoch)

        paddle.save(model.state_dict(), 'checkpoint/epoch_{}/net.pdparams'.format(epoch))
        # layer_state_dict = paddle.load("linear_net.pdparams")

    log.info("Runs %s: Model: %s Best Test Accuracy: %f" %
             (0, "graphsage", cal_test_acc[np.argmax(cal_val_acc)]))
    log.info("Runs %s: Model: %s Last Test Accuracy: %f" %
             (0, "graphsage", cal_test_acc[-1]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='graphsage')
    parser.add_argument(
        "--normalize", action='store_true', help="normalize features")
    parser.add_argument(
        "--symmetry", action='store_true', help="undirect graph")
    parser.add_argument("--sample_workers", type=int, default=16)
    parser.add_argument("--epoch", type=int, default=10)
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument('--samples', nargs='+', type=int, default=[50, 20])
    parser.add_argument('--log_path',  type=str, default='log.txt')
    args = parser.parse_args()
    add_log_to_file(args.log_path)
    log.info(args)
    main(args)

# python train.py  --epoch 10  --normalize --symmetry --batch_size 256