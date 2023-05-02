import torch.backends.cudnn
from model_hyper_graph import *
from Utils.Tools2 import *
from Utils.right_negative_sample_generate import *
import os

def series_num(data):
    for line in data:
        line[1] = line[1] + 270
        line[2] = line[2] + 270 + 58
    return data

def random_seed(sd):
    random.seed(sd)
    os.environ['PYTHONHASHSEED'] = str(sd)
    np.random.seed(sd)
    torch.manual_seed(sd)
    torch.cuda.manual_seed(sd)
    torch.cuda.manual_seed_all(sd)

def train(drug_fea, mic_fea, dis_fea, hg_pos, hg_neg_ls, train_data):
    model.train()
    print('--- Start training ---')
    optimizer.zero_grad()
    _, pred, train_embed_pos, graph_embed_neg_ls, train_summary = model(drug_fea, mic_fea, dis_fea, hg_pos,
                                                                     train_data[:, 0], train_data[:, 1],
                                                                     train_data[:, 2], hg_neg_ls)
    loss_1 = my_loss(pred.view(-1, 1), torch.from_numpy(train_data[:, 3]).view(-1, 1).float())
    loss_2 = model.DGI_loss(train_embed_pos, train_summary, graph_embed_neg_ls)
    loss = args.hypergraph_loss_ratio * loss_1 + (1 - args.hypergraph_loss_ratio) * loss_2
    loss.backward()
    optimizer.step()
    print('epoch:{:02d},'.format(e + 1), 'loss_train:{:.6f},'.format(loss.item()))

def test(drug_fea, mic_fea, dis_fea, hg_pos, hg_neg_ls, val_data_1, val_data_2, val_data_3, val_data_4):
    with torch.no_grad():
        model.eval()
        print('--- Start valuating ---')
        val = 0
        for val_data in [val_data_1, val_data_2, val_data_3, val_data_4]:
            val += 1
            _, pred_val, te_embed_pos, graph_embed_neg_ls, te_summary = model(drug_fea, mic_fea,
                dis_fea, hg_pos,val_data[:, 0], val_data[:, 1], val_data[:, 2], hg_neg_ls)
            hits_1, ndcg_1 = hit_ndcg_value(pred_val, val_data, args.top_1)
            hits_3, ndcg_3 = hit_ndcg_value(pred_val, val_data, args.top_3)
            hits_5, ndcg_5 = hit_ndcg_value(pred_val, val_data, args.top_5)
            print('val:{:1d}'.format(val), 'hits_3:{:.6f},'.format(hits_3), 'ndcg_3:{:.6f},'.format(ndcg_3))
            ealy_stop(val, e, hits_1, ndcg_1, hits_3, ndcg_3, hits_5, ndcg_5)

def ealy_stop(val, e, hits_1, ndcg_1, hits_3, ndcg_3, hits_5, ndcg_5):
    if hits_1 >= hits_max_matrix[val-1][0]:
        hits_max_matrix[val-1][0] = hits_1
        ndcg_max_matrix[val-1][0] = ndcg_1
        hits_max_matrix[val-1][1] = hits_3
        ndcg_max_matrix[val-1][1] = ndcg_3
        hits_max_matrix[val-1][2] = hits_5
        ndcg_max_matrix[val-1][2] = ndcg_5
        epoch_max_matrix[0][val-1] = e + 1
        patience_num_matrix[0][val-1] = 0
    else:
        patience_num_matrix[0][val-1] += 1

if __name__ == '__main__':
    args = parameters_set()
    random.seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    patience_num_matrix = np.zeros((1, 4))
    epoch_max_matrix = np.zeros((1, 4))
    hits_max_matrix = np.zeros((4, 3))
    ndcg_max_matrix = np.zeros((4, 3))

    patience = 300
    smiles_file = './Data/drug_smiles_270.csv'
    batch_drug = drug_fea_process(smiles_file, drug_num=270)
    dis_sim = np.loadtxt('./Data/dis_sim.txt', delimiter='\t')
    mic_sim = np.loadtxt('./Data/mic_sim_NinimHMDA.txt', delimiter='\t')
    dis_input = torch.from_numpy(dis_sim).type(torch.FloatTensor)
    mic_input = torch.from_numpy(mic_sim).type(torch.FloatTensor)

    adj_data = np.loadtxt('./Data/adj_del_4mic_myid.txt')
    np.random.shuffle(adj_data)
    test_data = adj_data[:int(0.1 * len(adj_data)), :]
    train_cv_data = adj_data[int(0.1 * len(adj_data)):, :]

    resultFileName = './indep_results.txt'
    train_data_all, tn_1, tn_2, tn_3, tn_4, te_1, te_2, te_3, te_4 = \
        neg_data_generate(adj_data, train_cv_data, test_data, args.seed)

    train_data_pos = series_num(train_cv_data.copy().astype(int))
    train_data_neg_1 = series_num(np.array(tn_1).copy().astype(int))
    train_data_neg_2 = series_num(np.array(tn_2).copy().astype(int))
    train_data_neg_3 = series_num(np.array(tn_3).copy().astype(int))
    train_data_neg_4 = series_num(np.array(tn_4).copy().astype(int))
    train_data_all = series_num(np.array(train_data_all).copy().astype(int))

    hypergraph_pos = build_hypergraph(train_data_pos)
    hypergraph_neg_1 = build_hypergraph(train_data_neg_1)
    hypergraph_neg_2 = build_hypergraph(train_data_neg_2)
    hypergraph_neg_3 = build_hypergraph(train_data_neg_3)
    hypergraph_neg_4 = build_hypergraph(train_data_neg_4)
    hypergraph_neg_ls = [hypergraph_neg_1, hypergraph_neg_2, hypergraph_neg_3, hypergraph_neg_4]

    val_data_1 = series_num(np.array(te_1).copy().astype(int))
    val_data_2 = series_num(np.array(te_2).copy().astype(int))
    val_data_3 = series_num(np.array(te_3).copy().astype(int))
    val_data_4 = series_num(np.array(te_4).copy().astype(int))

    model = HGNN(BioEncoder(mic_sim.shape[0], dis_sim.shape[0], args.bio_out_dim),
                 HgnnEncoder(args.bio_out_dim, args.hgnn_dim_1),
                 Decoder(((args.hgnn_dim_1) // 4) * 3), (args.hgnn_dim_1 // 4))
    my_loss = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), args.lr)

    for e in range(args.epochs):
        train(batch_drug, mic_input, dis_input, hypergraph_pos, hypergraph_neg_ls, train_data_all)
        test(batch_drug, mic_input, dis_input, hypergraph_pos, hypergraph_neg_ls, val_data_1, val_data_2, val_data_3, val_data_4)
        if patience_num_matrix[0][0] >= patience and patience_num_matrix[0][1] >= patience and \
                patience_num_matrix[0][2] >= patience and patience_num_matrix[0][3] >= patience:
            break
    print('success')
    write_type_1234(resultFileName, 'indep', hits_max_matrix[0][0], ndcg_max_matrix[0][0], hits_max_matrix[1][0],
                    ndcg_max_matrix[1][0], hits_max_matrix[2][0], ndcg_max_matrix[2][0], hits_max_matrix[3][0],
                    ndcg_max_matrix[3][0], epoch_max_matrix[0][0], epoch_max_matrix[0][1],
                    epoch_max_matrix[0][2], epoch_max_matrix[0][3], top='top1')
    write_type_1234(resultFileName, 'indep', hits_max_matrix[0][1], ndcg_max_matrix[0][1], hits_max_matrix[1][1],
                    ndcg_max_matrix[1][1], hits_max_matrix[2][1], ndcg_max_matrix[2][1], hits_max_matrix[3][1],
                    ndcg_max_matrix[3][1], epoch_max_matrix[0][0], epoch_max_matrix[0][1],
                    epoch_max_matrix[0][2], epoch_max_matrix[0][3], top='top3')
    write_type_1234(resultFileName, 'indep', hits_max_matrix[0][2], ndcg_max_matrix[0][2], hits_max_matrix[1][2],
                    ndcg_max_matrix[1][2], hits_max_matrix[2][2], ndcg_max_matrix[2][2], hits_max_matrix[3][2],
                    ndcg_max_matrix[3][2], epoch_max_matrix[0][0], epoch_max_matrix[0][1],
                    epoch_max_matrix[0][2], epoch_max_matrix[0][3], top='top5')
