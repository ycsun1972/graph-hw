from ogb_dataloader import PglNodePropPredDataset
dataset = PglNodePropPredDataset(name="ogbn-products")
split_idx=dataset.get_idx_split() #数据集划分
graph, label = dataset[0] #graph 是pgl.graph.Graph对象，label为节点的标签

print("num of nodes:",graph.num_nodes)
print("num of edges:",graph.num_edges)
print("dimension of feature:",graph.node_feat["feat"].shape)
print("class of labels:", int(max(label))+1)
print("Train Examples:",len(split_idx['train']))
print("Val Examples:",len(split_idx['valid']))
print("Test Examples:",len(split_idx['test']))

import pdb; pdb.set_trace()