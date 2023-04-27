import random
import numpy as np

def neg_data_generate(adj_data_all,train_data_fix,val_data_fix,seed):
    neg_num_train = 2
    neg_num_test = 29
    random.seed(seed)
    train_neg_1_ls = []; train_neg_2_ls = []; train_neg_3_ls = []; train_neg_4_ls = []
    val_neg_1_ls = []; val_neg_2_ls = []; val_neg_3_ls = []; val_neg_4_ls = []
    arr_true = np.zeros((270,58,167))
    for line in adj_data_all:
        arr_true[int(line[0]), int(line[1]), int(line[2])] = 1
    arr_false_train = np.zeros((len(set(adj_data_all[:,0])), len(set(adj_data_all[:,1])),
                                len(set(adj_data_all[:,2]))))
    for i in train_data_fix:
        ctn_1 = 0
        ctn_2 = 0
        ctn_3 = 0
        ctn_4 = 0
        # Only the negative samples corresponding to the drug and disease will be insufficient,
        # the microbe lists are not used
        tr_dru_ls = [j for j in range(0, arr_true.shape[0])]
        tr_mic_ls = [j for j in range(0, arr_true.shape[1])]
        tr_dis_ls = [j for j in range(0, arr_true.shape[2])]
        while ctn_1 < neg_num_train:
            a = random.randint(0, arr_true.shape[0] - 1)
            b = random.randint(0, arr_true.shape[1] - 1)
            c = random.randint(0, arr_true.shape[2] - 1)
            if arr_true[a, b, c] != 1 and arr_false_train[a, b, c] != 1:
                arr_false_train[a, b, c] = 1
                ctn_1 += 1
                train_neg_1_ls.append((a, b, c, 0))
        while ctn_2 < neg_num_train:
            b = int(i[1])
            c = int(i[2])
            a = random.choice(tr_dru_ls)
            if arr_true[a, b, c] != 1 and arr_false_train[a, b, c] != 1:
                arr_false_train[a, b, c] = 1
                ctn_2 += 1
                train_neg_2_ls.append((a, b, c, 0))
        while ctn_3 < neg_num_train:
            a = int(i[0])
            b = random.randint(0, arr_true.shape[1] - 1)
            c = int(i[2])
            if arr_true[a, b, c] != 1 and arr_false_train[a, b, c] != 1:
                arr_false_train[a, b, c] = 1
                ctn_3 += 1
                train_neg_3_ls.append((a, b, c, 0))
        while ctn_4 < neg_num_train:
            a = int(i[0])
            b = int(i[1])
            if tr_dis_ls != []:
                c = random.choice(tr_dis_ls)
                tr_dis_ls.remove(c)
                if arr_true[a, b, c] != 1 and arr_false_train[a, b, c] != 1:
                    arr_false_train[a, b, c] = 1
                    ctn_4 += 1
                    train_neg_4_ls.append((a, b, c, 0))
            else:
                distance_t4 = neg_num_train - ctn_4
                # print('triplet:', i, 'tr_4:', distance_t4)
                last_ind = len(train_neg_4_ls) - 1
                for k in range(distance_t4):
                    train_neg_4_ls.append(train_neg_4_ls[last_ind])
                break

    train_neg_1_arr = np.array(train_neg_1_ls)
    train_neg_2_arr = np.array(train_neg_2_ls)
    train_neg_3_arr = np.array(train_neg_3_ls)
    train_neg_4_arr = np.array(train_neg_4_ls)
    train_neg_all = np.vstack((np.vstack((np.vstack((np.vstack((train_neg_1_arr, train_neg_2_arr)), train_neg_3_arr)), train_neg_4_arr)),
                               train_data_fix))
    np.random.shuffle(train_neg_all)
    for i in val_data_fix:
        neg_1_i = []; neg_2_i = []; neg_3_i = []; neg_4_i = []
        # Because it is too easy to repeat, it is only guaranteed that multiple negative samples generated for a
        # certain positive sample are not repeated, and negative samples of different positive samples may be repeated.
        arr_false_val_1 = np.zeros((len(set(adj_data_all[:,0])), len(set(adj_data_all[:,1])), len(set(adj_data_all[:,2]))))
        arr_false_val_2 = np.zeros((len(set(adj_data_all[:,0])), len(set(adj_data_all[:,1])), len(set(adj_data_all[:,2]))))
        arr_false_val_3 = np.zeros((len(set(adj_data_all[:,0])), len(set(adj_data_all[:,1])), len(set(adj_data_all[:,2]))))
        arr_false_val_4 = np.zeros((len(set(adj_data_all[:,0])), len(set(adj_data_all[:,1])), len(set(adj_data_all[:,2]))))
        neg_1_i.append(i); neg_2_i.append(i); neg_3_i.append(i); neg_4_i.append(i)
        cva_1 = 0
        cva_2 = 0
        cva_3 = 0
        cva_4 = 0
        dru_ls = [j for j in range(0, arr_true.shape[0])]
        mic_ls = [j for j in range(0, arr_true.shape[1])]
        dis_ls = [j for j in range(0, arr_true.shape[2])]
        while cva_1 < neg_num_test:
            a_1 = random.randint(0, arr_true.shape[0] - 1)
            b_1 = random.randint(0, arr_true.shape[1] - 1)
            c_1 = random.randint(0, arr_true.shape[2] - 1)
            if arr_true[a_1, b_1, c_1] != 1 and arr_false_train[a_1, b_1, c_1] != 1 and arr_false_val_1[
                a_1, b_1, c_1] != 1:
                arr_false_val_1[a_1, b_1, c_1] = 1
                cva_1 += 1
                neg_1_i.append((a_1, b_1, c_1, 0))
        np.random.shuffle(neg_1_i)
        val_neg_1_ls.extend(neg_1_i)
        while cva_2 < neg_num_test:
            b_2 = int(i[1])
            c_2 = int(i[2])
            if dru_ls != []:
                a_2 = random.choice(dru_ls)
                dru_ls.remove(a_2)
                if arr_true[a_2, b_2, c_2] != 1 and arr_false_train[a_2, b_2, c_2] != 1 and arr_false_val_2[
                    a_2, b_2, c_2] != 1:
                    arr_false_val_2[a_2, b_2, c_2] = 1
                    cva_2 += 1
                    neg_2_i.append((a_2, b_2, c_2, 0))
            else:
                distance_2 = neg_num_test - cva_2
                # print('triplet:', i, 'val_2:', distance_2)
                last_ind = len(neg_2_i) - 1
                for k in range(distance_2):
                    neg_2_i.append(neg_2_i[last_ind])
                break
        np.random.shuffle(neg_2_i)
        val_neg_2_ls.extend(neg_2_i)

        while cva_3 < neg_num_test:
            a_3 = int(i[0])
            c_3 = int(i[2])
            if mic_ls != []:
                b_3 = random.choice(mic_ls)
                mic_ls.remove(b_3)
                if arr_true[a_3, b_3, c_3] != 1 and arr_false_train[a_3, b_3, c_3] != 1 and arr_false_val_3[
                    a_3, b_3, c_3] != 1:
                    arr_false_val_3[a_3, b_3, c_3] = 1
                    cva_3 += 1
                    neg_3_i.append((a_3, b_3, c_3, 0))
            else:
                distance_3 = neg_num_test - cva_3
                # print('triplet:', i, 'val_3:', distance_3)
                last_ind = len(neg_3_i) - 1
                for k in range(distance_3):
                    neg_3_i.append(neg_3_i[last_ind])
                break
        np.random.shuffle(neg_3_i)
        val_neg_3_ls.extend(neg_3_i)

        while cva_4 < neg_num_test:
            a_4 = int(i[0])
            b_4 = int(i[1])
            if dis_ls != []:
                c_4 = random.choice(dis_ls)
                dis_ls.remove(c_4)
                if arr_true[a_4, b_4, c_4] != 1 and arr_false_train[a_4, b_4, c_4] != 1 and arr_false_val_4[
                    a_4, b_4, c_4] != 1:
                    arr_false_val_4[a_4, b_4, c_4] = 1
                    cva_4 += 1
                    neg_4_i.append((a_4, b_4, c_4, 0))
            else:
                distance_4 = neg_num_test - cva_4
                # print('triplet:', i, 'val_4:', distance_4)
                last_ind = len(neg_4_i) - 1
                for k in range(distance_4):
                    neg_4_i.append(neg_4_i[last_ind])
                break
        np.random.shuffle(neg_4_i)
        val_neg_4_ls.extend(neg_4_i)
    # print('fold_num:', fold_num, 'neg_2:', neg_2, 'neg_3:', neg_3, 'neg_4:', neg_4)
    return train_neg_all, train_neg_1_ls, train_neg_2_ls, train_neg_3_ls,train_neg_4_ls, val_neg_1_ls, val_neg_2_ls, val_neg_3_ls, val_neg_4_ls
