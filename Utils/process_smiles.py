import numpy as np
from torch_geometric.data import Dataset, DataLoader, Batch, InMemoryDataset
from torch_geometric import data as DATA
import torch
from rdkit import Chem
import networkx as nx


class GraphDataset_v(Dataset):
    def __init__(self, root='.', dataset='davis', transform=None, pre_transform=None, dtype=None,
                 xd=None, xc=None, xm=None, y=None, did=None, cid=None, mid=None):
        super(GraphDataset_v, self).__init__(root, transform, pre_transform)
        self.dataset = dataset
        self.dttype = dtype
        self.process(xc,  cid)
    @property
    def raw_file_names(self):
        pass

    @property
    def processed_file_names(self):
        return [self.dataset + f'_data_{self.dttype}.pt']

    def download(self):
        pass

    def _download(self):
        pass

    def _process(self):
        pass

    def process(self, xc, cid):
        data_list = []
        data_len = len(xc)
        for i in range(data_len):
            c_size, features, edge_index = xc[i]
            GCNData = DATA.Data(x=torch.FloatTensor(features),
                                edge_index=torch.LongTensor(edge_index)
                                )
            GCNData.__setitem__('c_size', torch.LongTensor([c_size]))
            # GCNData.__setitem__('disease', torch.FloatTensor([disease]))
            # GCNData.__setitem__('micro_all', torch.FloatTensor([micro]))
            GCNData.__setitem__('cid', torch.LongTensor([cid[i]]))
            # GCNData.__setitem__('did', torch.LongTensor([did[i]]))
            # GCNData.__setitem__('mid', torch.LongTensor([mid[i]]))
            # append graph, label and target sequence to data list
            data_list.append(GCNData)

        self.data = data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate(data_list):
    batchA = Batch.from_data_list([data for data in data_list])
    return batchA


def atom_features(atom):
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),
                                          ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As',
                                           'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se',
                                           'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr',
                                           'Pt', 'Hg', 'Pb', 'Unknown']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    [atom.GetIsAromatic()])


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)

    c_size = mol.GetNumAtoms()

    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append(feature / sum(feature))

    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    g = nx.Graph(edges).to_directed()
    edge_s = []
    edge_d = []
    for e1, e2 in g.edges:
        edge_s.append(e1)
        edge_d.append(e2)
    edge_index = [edge_s, edge_d]
    return c_size, features, edge_index


def drug2embedding(drug, max_drug_len=228):
    drug_iso2char = {"1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9,
                     "0": 10, "#": 11, "(": 12, ")": 13, "[": 14, "]": 15, "-": 16, "=": 17,
                     "@": 18, "+": 19, "C": 20, "O": 21, "N": 22, "H": 23, "B": 24, "r": 25,
                     "F": 26, "S": 27, "\\": 28, "/": 29, "l": 30, "I": 31, ".": 32, "P": 33,
                     "t": 34, " ": 35}
    embeddings = []
    for d in drug:
        d = d.rsplit()
        embeddings.append(embed(d, max_drug_len, drug_iso2char))

    return embeddings


def embed(data, max_len, charset):
    embedding = np.zeros(max_len)
    for n in range(len(data[0])):
        embedding[n] = charset[data[0][n]]
    return embedding
