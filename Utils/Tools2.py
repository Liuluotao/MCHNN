import argparse
import random
from Utils.process_smiles import *
import csv

def drug_fea_process(smiles_file, drug_num):
    reader = csv.reader(open(smiles_file))
    smile_graph = []
    for item in reader:
        smile = item[1]
        g = smile_to_graph(smile)
        smile_graph.append(g)
    dru_data = GraphDataset_v(xc=smile_graph, cid=[i for i in range(drug_num + 1)])
    dru_data = torch.utils.data.DataLoader(dataset=dru_data, batch_size=drug_num, shuffle=False,
                                           collate_fn=collate)
    for step, batch_drug in enumerate(dru_data):
        drug_data = batch_drug
    return drug_data

def parameters_set():
    parser = argparse.ArgumentParser(description='using GAT process association matrix and GIN&Conv process feature')
    parser.add_argument('--gpu', type=int, default=-1,
                        help='which GPU to use. Set -1 to use CPU.')
    parser.add_argument('--hypergraph_loss_ratio', type=float, default=0.8,
                        help='the weight of hypergraph_loss_ratio')
    parser.add_argument('--epochs', type=int, default=3000,
                        help='number of training epochs')
    parser.add_argument('--top_1', type=int, default=1,
                        help='hit@top,ndcg@top')
    parser.add_argument('--top_3', type=int, default=3,
                        help='hit@top,ndcg@top')
    parser.add_argument('--top_5', type=int, default=5,
                        help='hit@top,ndcg@top')

    parser.add_argument('--train_data', type=str, default='4_type',
                        help='train_data_neg_type: '
                             'ratio_4_type,4_type,del_1th_type,del_2th_type,del_3th_type,del_4th_type,'
                             'only_1th_type,only_2th_type,only_3th_type,only_4th_type')  # 4_type

    parser.add_argument('--bio_out_dim', type=int, default=32,
                        help='bio_out_dim of the dru,mic,dis ')  # 32 warm-start最佳, 8 cold-start最佳
    parser.add_argument('--hgnn_dim_1', type=int, default=512,
                        help='hgnn_dim_1 of feature')  # 512

    parser.add_argument('--cold_class', type=str, default='dis')

    parser.add_argument('--lr', type=float, default=0.005,  # 0.005
                        help='learning rate')
    parser.add_argument('--seed', type=int, default=1,
                        help='the random seeds')
    parser.add_argument('--k_fold', type=int, default=5,
                        help='k_fold')
    parser.add_argument('--BATCH_SIZE', type=int, default=30,
                        help='BATCH_SIZE')

    args = parser.parse_args()
    return args

class Metrics():
    def __init__(self, step, test_data, predict_1, batch_size, top):
        self.pair = []
        self.step = step
        self.test_data = test_data
        self.predict_1 = predict_1
        self.top = top
        self.dcgsum = 0
        self.idcgsum = 0
        self.hit = 0
        self.ndcg = 0
        self.batch_size = batch_size
        self.val_top = []

    def hits_ndcg(self):
        for i in range(self.step * self.batch_size, (self.step + 1) * self.batch_size):
            g = []
            g.extend([self.test_data[i, 3], self.predict_1[i].item()])
            self.pair.append(g)
        np.random.seed(1)
        np.random.shuffle(self.pair)
        pre_val = sorted(self.pair, key=lambda item: item[1], reverse=True)
        self.val_top = pre_val[0: self.top]

        for i in range(len(self.val_top)):
            if self.val_top[i][0] == 1:
                self.hit = self.hit + 1
                self.dcgsum = (2 ** self.val_top[i][0] - 1) / np.log2(i + 2)
                break
        ideal_list = sorted(self.val_top, key=lambda item: item[0], reverse=True)
        for i in range(len(ideal_list)):
            if ideal_list[i][0] == 1:
                self.idcgsum = (2 ** ideal_list[i][0] - 1) / np.log2(i + 2)
                break
        if self.idcgsum == 0:
            self.ndcg = 0
        else:
            self.ndcg = self.dcgsum / self.idcgsum
        return self.hit, self.ndcg

def get_metrics(real_score, predict_score):
    sorted_predict_score = np.array(sorted(list(set(np.array(predict_score).flatten()))))
    sorted_predict_score_num = len(sorted_predict_score)
    thresholds = sorted_predict_score[
        (np.array([sorted_predict_score_num]) * np.arange(1, 1000) / np.array([1000])).astype(int)]
    thresholds = np.mat(thresholds)
    thresholds_num = thresholds.shape[1]

    predict_score_matrix = np.tile(predict_score, (thresholds_num, 1))
    negative_index = np.where(predict_score_matrix < thresholds.T)
    positive_index = np.where(predict_score_matrix >= thresholds.T)
    predict_score_matrix[negative_index] = 0
    predict_score_matrix[positive_index] = 1
    TP = predict_score_matrix * real_score.T

    FP = predict_score_matrix.sum(axis=1) - TP
    FN = real_score.sum() - TP
    TN = len(real_score.T) - TP - FP - FN

    fpr = FP / (FP + TN)
    tpr = TP / (TP + FN)
    ROC_dot_matrix = np.mat(sorted(np.column_stack((fpr, tpr)).tolist())).T
    ROC_dot_matrix.T[0] = [0, 0]
    ROC_dot_matrix = np.c_[ROC_dot_matrix, [1, 1]]
    x_ROC = ROC_dot_matrix[0].T
    y_ROC = ROC_dot_matrix[1].T
    auc = 0.5 * (x_ROC[1:] - x_ROC[:-1]).T * (y_ROC[:-1] + y_ROC[1:])

    recall_list = tpr
    precision_list = TP / (TP + FP)
    PR_dot_matrix = np.mat(sorted(np.column_stack((recall_list, -precision_list)).tolist())).T
    PR_dot_matrix[1, :] = -PR_dot_matrix[1, :]
    PR_dot_matrix.T[0] = [0, 1]
    PR_dot_matrix = np.c_[PR_dot_matrix, [1, 0]]
    x_PR = PR_dot_matrix[0].T
    y_PR = PR_dot_matrix[1].T
    aupr = 0.5 * (x_PR[1:] - x_PR[:-1]).T * (y_PR[:-1] + y_PR[1:])

    f1_score_list = 2 * TP / (len(real_score.T) + TP - TN)
    accuracy_list = (TP + TN) / len(real_score.T)
    specificity_list = TN / (TN + FP)

    max_index = np.argmax(f1_score_list)
    f1_score = f1_score_list[max_index, 0]
    accuracy = accuracy_list[max_index, 0]
    specificity = specificity_list[max_index, 0]
    recall = recall_list[max_index, 0]
    precision = precision_list[max_index, 0]
    return [aupr[0, 0], auc[0, 0], f1_score, accuracy, recall, specificity, precision]

def hit_ndcg_ls(top, fold_th, pred_val, val_data, val,
                hit_ls_1, hit_ls_2, hit_ls_3, hit_ls_4, ndcg_ls_1, ndcg_ls_2, ndcg_ls_3, ndcg_ls_4, epoch=False,
                loss_test=False):
    loader_val = torch.utils.data.DataLoader(dataset=pred_val, batch_size=100, shuffle=False)
    hits = 0
    ndcg_val = 0
    for step, batch_val in enumerate(loader_val):
        metrix = Metrics(step, val_data, pred_val, batch_size=100, top=top)
        hit, ndcg = metrix.hits_ndcg()
        hits = hits + hit
        ndcg_val = ndcg_val + ndcg
    hits = hits / int((len(val_data)) / 100)
    ndcg = ndcg_val / int((len(val_data)) / 100)
    if val == 1:
        hit_ls_1.append(hits)
        ndcg_ls_1.append(ndcg)
    if val == 2:
        hit_ls_2.append(hits)
        ndcg_ls_2.append(ndcg)
    if val == 3:
        hit_ls_3.append(hits)
        ndcg_ls_3.append(ndcg)
    if val == 4:
        hit_ls_4.append(hits)
        ndcg_ls_4.append(ndcg)
    if loss_test and epoch:
        print('fold_num:', fold_th, 'epoch:{:02d},'.format(epoch),
              'val_type:{:02d},'.format(val),
              'loss_test:{:.6f},'.format(loss_test), 'hit: {:.6f},'.format(hits),
              'ndcg:{:.6f},'.format(ndcg))
    else:
        print('fold_num:', fold_th,
              'val_type:{:02d},'.format(val),
              'hit: {:.6f},'.format(hits), 'ndcg:{:.6f},'.format(ndcg))
    return hit_ls_1, hit_ls_2, hit_ls_3, hit_ls_4, ndcg_ls_1, ndcg_ls_2, ndcg_ls_3, ndcg_ls_4

def hit_ndcg_value(pred_val, val_data, top):
    loader_val = torch.utils.data.DataLoader(dataset=pred_val, batch_size=30, shuffle=False)
    hits = 0
    ndcg_val = 0
    for step, batch_val in enumerate(loader_val):
        metrix = Metrics(step, val_data, pred_val, batch_size=30, top=top)
        hit, ndcg = metrix.hits_ndcg()
        hits = hits + hit
        ndcg_val = ndcg_val + ndcg
    hits = hits / int((len(val_data)) / 30)
    ndcg = ndcg_val / int((len(val_data)) / 30)
    return hits, ndcg

def write_type_1234(file_name, fold_th, hits_1_max, ndcg_1_max, hits_2_max, ndcg_2_max, hits_3_max, ndcg_3_max,
                    hits_4_max, ndcg_4_max, epoch_max_1=None, epoch_max_2=None, epoch_max_3=None,
                    epoch_max_4=None, top='top1'):
    with open(file_name, 'a') as f:
        f.write(str(fold_th) + '\t' + str(top) + '\t' + str(hits_1_max) + '\t' + str(ndcg_1_max) + '\t' + str(
            epoch_max_1) + '\t' + str(hits_2_max) + '\t' + str(ndcg_2_max) + '\t' + str(
            epoch_max_2) + '\t' + str(hits_3_max) + '\t' + str(ndcg_3_max) + '\t' + str(
            epoch_max_3) + '\t' + str(hits_4_max) + '\t' + str(ndcg_4_max) + '\t' + str(
            epoch_max_4) + '\n')

def build_hypergraph(data):
    pairs = np.array(data[:, 0:3]).reshape(1, -1)
    pairs_num = np.expand_dims(np.arange(len(data)), axis=-1)
    hyper_edge_num = np.concatenate((pairs_num, pairs_num, pairs_num), axis=1)
    hyper_edge_num = np.array(hyper_edge_num).reshape(1, -1)
    hyper_graph = np.concatenate((pairs, hyper_edge_num), axis=0)
    hyper_graph = torch.from_numpy(hyper_graph).type(torch.LongTensor)
    return hyper_graph