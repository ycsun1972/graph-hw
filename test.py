from ogb_dataloader import PglNodePropPredDataset
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
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
import os

from tb_paddle import SummaryWriter

def resume(checkpoint_dir, model, optim):
    log.info('resume from : %s' % checkpoint_dir)
    layer_state_dict = paddle.load(os.path.join(checkpoint_dir, "net.pdparams"))
    opt_state_dict = paddle.load(os.path.join(checkpoint_dir, "optim.pdopt"))

    model.set_state_dict(layer_state_dict)
    optim.set_state_dict(opt_state_dict)


@paddle.no_grad()
def eval_ogb(dataloader, model, feature, criterion):
    evaluator = Evaluator('ogbn-products')
    model.eval()
    loss_all, pred_all, label_all = [], [], []

    batch = 0
    for g, sample_index, index, label in dataloader:
        batch += 1
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

        if batch % 100 == 0:
            log.info("Batch %s" %batch)
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


    train_index = split_idx['train']
    val_index = split_idx['valid']
    test_index = split_idx['test']

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
    pre_step = 0
    if args.checkpoint_dir != '':
        resume(args.checkpoint_dir, model, optim)
        pre_step += int(args.checkpoint_dir.split('_')[-1])

    feature = paddle.to_tensor(graph.node_feat["feat"])

    train_ds = ShardedDataset(train_index, train_label)
    val_ds = ShardedDataset(val_index, val_label)
    test_ds = ShardedDataset(test_index, test_label)

    collate_fn = partial(batch_fn, graph=graph, samples=args.samples)


    test_loader = Dataloader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.sample_workers,
        collate_fn=collate_fn)


    test_loss, test_acc = eval_ogb(test_loader, model, feature, criterion)
 
    log.info("Runs %s: Model: %s Best Test Accuracy: %f" % (0, "graphsage", test_acc))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='graphsage')
    parser.add_argument(
        "--normalize", action='store_true', help="normalize features")
    parser.add_argument(
        "--symmetry", action='store_true', help="undirect graph")
    parser.add_argument("--sample_workers", type=int, default=16)
    parser.add_argument("--epoch", type=int, default=10)
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument('--samples', nargs='+', type=int, default=[25, 10])
    parser.add_argument('--log_path',  type=str, default='log.txt')
    parser.add_argument('--checkpoint_dir',  type=str, default='')
    args = parser.parse_args()
    log.info(args)
    main(args)

# python test.py  --epoch 1000  --normalize --symmetry --log_path log1227.txt --batch_size 2560 --checkpoint_dir checkpoint_nv/epoch_500