import pandas as pd
import os
import os.path as osp
import numpy as np
from tqdm import tqdm
import pgl

def read_csv_graph_raw(raw_dir, add_inverse_edge = True, additional_node_files = [], additional_edge_files = []):

    print('Loading necessary files...')
    print('This might take a while.')
    try:
        edge = pd.read_csv(osp.join(raw_dir, "edge.csv.gz"), compression="gzip", header = None).values.T.astype(np.int64) # (2, num_edge) numpy array
        num_node_list = pd.read_csv(osp.join(raw_dir, "num-node-list.csv.gz"), compression="gzip", header = None).astype(np.int64)[0].tolist() # (num_graph, ) python list
        num_edge_list = pd.read_csv(osp.join(raw_dir, "num-edge-list.csv.gz"), compression="gzip", header = None).astype(np.int64)[0].tolist() # (num_edge, ) python list

    except FileNotFoundError:
        raise RuntimeError("No necessary file")

    try:
        node_feat = pd.read_csv(osp.join(raw_dir, "node-feat.csv.gz"), compression="gzip", header = None).values
        if 'int' in str(node_feat.dtype):
            node_feat = node_feat.astype(np.int64)
        else:
            # float
            node_feat = node_feat.astype(np.float32)
    except FileNotFoundError:
        node_feat = None

    try:
        edge_feat = pd.read_csv(osp.join(raw_dir, "edge-feat.csv.gz"), compression="gzip", header = None).values
        if 'int' in str(edge_feat.dtype):
            edge_feat = edge_feat.astype(np.int64)
        else:
            #float
            edge_feat = edge_feat.astype(np.float32)

    except FileNotFoundError:
        edge_feat = None
    additional_node_info = {}
    for additional_file in additional_node_files:
        temp = pd.read_csv(osp.join(raw_dir, additional_file + ".csv.gz"), compression="gzip", header = None).values

        if 'node_' not in additional_file:
            feat_name = 'node_' + additional_file
        else:
            feat_name = additional_file

        if 'int' in str(temp.dtype):
            additional_node_info[feat_name] = temp.astype(np.int64)
        else:
            # float
            additional_node_info[feat_name] = temp.astype(np.float32)

    additional_edge_info = {}
    for additional_file in additional_edge_files:
        temp = pd.read_csv(osp.join(raw_dir, additional_file + ".csv.gz"), compression="gzip", header = None).values

        if 'edge_' not in additional_file:
            feat_name = 'edge_' + additional_file
        else:
            feat_name = additional_file

        if 'int' in str(temp.dtype):
            additional_edge_info[feat_name] = temp.astype(np.int64)
        else:
            # float
            additional_edge_info[feat_name] = temp.astype(np.float32)
    graph_list = []
    num_node_accum = 0
    num_edge_accum = 0

    print('Processing graphs...')
    for num_node, num_edge in tqdm(zip(num_node_list, num_edge_list), total=len(num_node_list)):

        graph = dict()

        ### handling edge
        if add_inverse_edge:
            ### duplicate edge
            duplicated_edge = np.repeat(edge[:, num_edge_accum:num_edge_accum+num_edge], 2, axis = 1)
            duplicated_edge[0, 1::2] = duplicated_edge[1,0::2]
            duplicated_edge[1, 1::2] = duplicated_edge[0,0::2]

            graph["edge_index"] = duplicated_edge

            if edge_feat is not None:
                graph["edge_feat"] = np.repeat(edge_feat[num_edge_accum:num_edge_accum+num_edge], 2, axis = 0)
            else:
                graph["edge_feat"] = None

            for key, value in additional_edge_info.items():
                graph[key] = np.repeat(value[num_edge_accum:num_edge_accum+num_edge], 2, axis = 0)
        else:
            graph["edge_index"] = edge[:, num_edge_accum:num_edge_accum+num_edge]

            if edge_feat is not None:
                graph["edge_feat"] = edge_feat[num_edge_accum:num_edge_accum+num_edge]
            else:
                graph["edge_feat"] = None

            for key, value in additional_edge_info.items():
                graph[key] = value[num_edge_accum:num_edge_accum+num_edge]

        num_edge_accum += num_edge

        ### handling node
        if node_feat is not None:
            graph["node_feat"] = node_feat[num_node_accum:num_node_accum+num_node]
        else:
            graph["node_feat"] = None

        for key, value in additional_node_info.items():
            graph[key] = value[num_node_accum:num_node_accum+num_node]

        graph["num_nodes"] = num_node
        num_node_accum += num_node
        graph_list.append(graph)
    return graph_list



def read_csv_graph_pgl(raw_dir, add_inverse_edge=False):
    """Read CSV data and build PGL Graph
    """
    graph_list = read_csv_graph_raw(raw_dir, add_inverse_edge)
    pgl_graph_list = []

    for graph in graph_list:
        edges = list(zip(graph["edge_index"][0], graph["edge_index"][1]))
        g = pgl.graph.Graph(num_nodes=graph["num_nodes"], edges=edges)

        if graph["edge_feat"] is not None:
            g.edge_feat["feat"] = graph["edge_feat"]

        if graph["node_feat"] is not None:
            g.node_feat["feat"] = graph["node_feat"]

        pgl_graph_list.append(g)

    return pgl_graph_list


def to_bool(value):
    """to_bool"""
    return np.array([value], dtype="bool")[0]


class PglNodePropPredDataset(object):
    """PglNodePropPredDataset
    """

    def __init__(self, name, root="dataset"):
        self.name = name  ## original name, e.g., ogbn-proteins
        self.dir_name = "_".join(
            name.split("-")
        ) + "_pgl"  ## replace hyphen with underline, e.g., ogbn_proteins_pgl

        self.original_root = root
        self.root = osp.join(root, self.dir_name)
        
        self.num_tasks = int(1)
        self.task_type = "multiclass classification"
        #self.task_type = self.meta_info[self.name]["task type"]

        super(PglNodePropPredDataset, self).__init__()

        self.pre_process()

    def pre_process(self):
        """pre_process downlaoding data
        """
        processed_dir = osp.join(self.root, 'processed')
        pre_processed_file_path = osp.join(processed_dir, 'pgl_data_processed')

        if osp.exists(pre_processed_file_path):
            # TODO: Reload Preprocess files
            graph_list=[]
            graph_list.append(pgl.graph.Graph.load(pre_processed_file_path))
            self.graph=graph_list
            self.labels=np.load(pre_processed_file_path+"/labels.npy")
            pass
        else:

            raw_dir = osp.join(self.root, "raw")

            ### pre-process and save
            #add_inverse_edge = to_bool(self.meta_info[self.name]["add_inverse_edge"])
            add_inverse_edge = True
            self.graph = read_csv_graph_pgl(
                raw_dir, add_inverse_edge=add_inverse_edge)

            ### adding prediction target
            node_label = pd.read_csv(
                osp.join(raw_dir, 'node-label.csv.gz'),
                compression="gzip",
                header=None).values
            if "classification" in self.task_type:
                node_label = np.array(node_label, dtype=np.int64)
            else:
                node_label = np.array(node_label, dtype=np.float32)

            label_dict = {"labels": node_label}

            # TODO: SAVE preprocess graph
            self.labels = label_dict['labels']
            self.graph[0].dump(pre_processed_file_path)

            np.save(pre_processed_file_path+"/labels.npy",self.labels)
            

    def get_idx_split(self):
        """Train/Validation/Test split
        """
        #split_type = self.meta_info[self.name]["split"]
        split_type = "sales_ranking"
        path = osp.join(self.root, "split", split_type)

        train_idx = pd.read_csv(
            osp.join(path, "train.csv.gz"), compression="gzip",
            header=None).values.T[0]
        valid_idx = pd.read_csv(
            osp.join(path, "valid.csv.gz"), compression="gzip",
            header=None).values.T[0]
        test_idx = pd.read_csv(
            osp.join(path, "test.csv.gz"), compression="gzip",
            header=None).values.T[0]
        return {
            "train": np.array(
                train_idx, dtype="int64"),
            "valid": np.array(
                valid_idx, dtype="int64"),
            "test": np.array(
                test_idx, dtype="int64")
        }

    def __getitem__(self, idx):
        assert idx == 0, "This dataset has only one graph"
        return self.graph[idx], self.labels

    def __len__(self):
        return 1

    def __repr__(self):  # pragma: no cover
        return '{}({})'.format(self.__class__.__name__, len(self))