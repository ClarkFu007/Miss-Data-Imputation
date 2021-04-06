import copy
import csv
import random
import torch
import time
import numpy as np
from math import sqrt
from numpy import fabs, sum, square
from sklearn import preprocessing


def get_dataset(datafile_B1, datafile_FA, datafile_FB, datafile_FC):
    """
       Input the complete dataset (sub-function).
    :param datafile_B1: link of the data of Bus_1
    :param datafile_FA: link of the data of Feeder A
    :param datafile_FB: link of the data of Feeder B
    :param datafile_FC: link of the data of Feeder C

    :return: the complete dataset
    """
    Bus_1_X, Bus_1_y = input_data(datafile_B1, 3, 8760)
        # Data matrix Bus_1_X: (2, 8760)
        # Label matrix Bus_1_y: (1, 8760)
    Feeder_A_X, Feeder_A_y = input_data(datafile_FA, 51, 8760)
        # Data matrix Feeder_A_X: (34, 8760)
        # Label matrix Feeder_A_y: (17, 8760)
    Feeder_B_X, Feeder_B_y = input_data(datafile_FB, 180, 8760)
        # Data matrix Feeder_B_X: (120, 8760)
        # Label matrix Feeder_B_y: (60, 8760)
    Feeder_C_X, Feeder_C_y = input_data(datafile_FC, 486, 8760)
        # Data matrix Feeder_C_X (324, 8760)
        # Label matrix Feeder_C_X (162, 8760)

    # Get the dataset of shape (240, 3, 8760)
    data_set = np.zeros((240, 8760, 2), dtype='float')
    # data_set[0, 0:3, 0:8760] = np.vstack((Bus_1_X, Bus_1_y))
    data_set[0, 0:8760, 0:2] = np.transpose(Bus_1_X)
    data_set[1:18, 0:8760, 0] = Feeder_A_X[0:17, 0:8760]
    data_set[1:18, 0:8760, 1] = Feeder_A_X[17:34, 0:8760]
    # data_set[1:18, 2, 0:8760] = Feeder_A_y
    data_set[18:78, 0:8760, 0] = Feeder_B_X[0:60, 0:8760]
    data_set[18:78, 0:8760, 1] = Feeder_B_X[60:120, 0:8760]
    # data_set[18:78, 2, 0:8760] = Feeder_B_y
    data_set[78:240, 0:8760, 0] = Feeder_C_X[0:162, 0:8760]
    data_set[78:240, 0:8760, 1] = Feeder_C_X[162:324, 0:8760]
    # data_set[78:240, 2, 0:8760] = Feeder_C_y

    return data_set


def input_data(datafile_w, data_m, data_n):
    """
       Input the complete dataset (main function).
    :param datafile_w: link of the data of Bus_1
    :param data_m: link of the data of Feeder A
    :param data_n: link of the data of Feeder B

    :return: data matrix X, label matrix y
    """
    InputData = np.zeros((data_m, data_n), dtype='float')
    with open(datafile_w, 'r') as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            data = [float(datum) for datum in row]
            InputData[i] = data
    bound_num = data_m * 2 // 3
    Data_X = InputData[0:bound_num, 0:data_n]  # Data matrix X
    Data_y = InputData[bound_num:data_m, 0:data_n]  # Label matrix y

    return Data_X, Data_y


def group_sample(data_set, time_size):
    """
       Group dataset in terms of the time size.
    :param data_set: link of the data of Bus_1
    :param time_size: link of the data of Feeder A

    :return: the output window data with different sizes of time
    """
    node_num, total_num, fea_num = data_set.shape[:]

    sample_num = total_num // time_size
    output_data = np.zeros((sample_num, node_num, time_size, fea_num), dtype='float')
    for sample_i in range(0, sample_num):
        sample_l = sample_i * time_size
        sample_u = sample_l + time_size
        output_data[sample_i, 0:node_num, 0:time_size, 0:fea_num] = \
            data_set[0:node_num, sample_l:sample_u, 0:fea_num]

    return output_data


def split_sample(data_set, total_num, tr_num, vali_num):
    """
       Get the training, validation, and test set (sub-function).
    :param data_set: total dataset
    :param total_num: total number of samples
    :param tr_num: number of training set
    :param vali_num: number of validation set

    :return: training set, validation set, test set
    """
    index_list = list(range(total_num))
    train_index = index_list[0: tr_num]
    validation_index = index_list[tr_num: tr_num + vali_num]
    test_index = index_list[tr_num + vali_num: total_num]

    node_num, time_size, fea_num = data_set.shape[1:]

    tr_set = np.zeros((tr_num, node_num, time_size, fea_num), dtype='float')
    for sample_i in range(tr_num):
        tr_set[sample_i] = data_set[train_index[sample_i]]

    vali_set = np.zeros((vali_num, node_num, time_size, fea_num), dtype='float')
    for sample_i in range(vali_num):
        vali_set[sample_i] = data_set[validation_index[sample_i]]

    te_num = total_num - tr_num - vali_num
    te_set = np.zeros((te_num, node_num, time_size, fea_num), dtype='float')
    for sample_i in range(te_num):
        te_set[sample_i] = data_set[test_index[sample_i]]

    return tr_set, vali_set, te_set


def split_tr_val_te(dataset, time_step, train_num, validation_num):
    """
       Get the training, validation, and test set (main function).
    :param dataset: total dataset
    :param time_step: total number of samples
    :param train_num: number of training set
    :param validation_num: number of validation set

    :return: training set, validation set, test set
    """
    dataset = group_sample(dataset, time_step)
    print("The experiment dataset", dataset.shape)
    sample_num = dataset.shape[0]

    train_set, validation_set, test_set = split_sample(data_set=dataset,
                                                       total_num=sample_num,
                                                       tr_num=train_num,
                                                       vali_num=validation_num)

    return train_set, validation_set, test_set


def get_normalized_adj(A):
    """
       Get the degree normalized adjacency matrix.
    :param A: adjacency matrix

    :return: degree normalized adjacency matrix
    """
    A = A + np.diag(np.ones(A.shape[0], dtype=np.float32))
    D = np.array(np.sum(A, axis=1)).reshape((-1,))
    D[D <= 10e-5] = 10e-5    # Prevent infs
    diag = np.reciprocal(np.sqrt(D))
    A_wave = np.multiply(np.multiply(diag.reshape((-1, 1)), A),
                         diag.reshape((1, -1)))
    return A_wave


def degrade_dataset(X, missingness, v, method, size, seed_value=590):
    """
       Degrade the dataset (sub-function).
    :param X: dataset to degrade.
    :param missingness: percentage of data to eliminate[0,1]
    :param v: replace with = 'zero' or 'np.nan'
    :param method: 'block' or 'nonblock'
    :param size: the size of the block
    :param seed_value: the seed value to control corruptation

    :return: corrupted set, binary mask
    """
    if method == 'block':
        X_1d = X.flatten('F')  # According to the columns
        n = len(X_1d)
        mask_1d = np.ones(n)

        random.seed(seed_value)
        corrupt_ids = random.sample(range(n), int(missingness * n))
        for i in corrupt_ids:
            X_1d[i] = v
            mask_1d[i] = 0
            for j in range(size):
                if i - j >= 0:
                    X_1d[i - j] = v
                    mask_1d[i - j] = 0
                else:
                    X_1d[i + j] = v
                    mask_1d[i + j] = 0

        cX = X_1d.reshape(X.shape, order='F')
        mask = mask_1d.reshape(X.shape, order='F')
        return cX, mask

    elif method == 'nonblock':
        X_1d = X.flatten()
        n = len(X_1d)
        mask_1d = np.ones(n)

        random.seed(seed_value)
        corrupt_ids = random.sample(range(n), int(missingness * n))
        for i in corrupt_ids:
            X_1d[i] = v
            mask_1d[i] = 0

        cX = X_1d.reshape(X.shape)
        mask = mask_1d.reshape(X.shape)
        return cX, mask

    else:
        print("Pleast input the correct degrading method!")
        return False


def corrupt_dataset(x_set, missingness, v_value, corrupt_method, block_size):
    """
       Corrupt the dataset (main funcrion).
    :param x_set: dataset to corrupt
    :param missingness: percentage of data to eliminate[0,1]
    :param v_value: replace with = 'zero' or 'np.nan'
    :param corrupt_method: 'block' or 'nonblock'
    :param block_size: the size of the block

    :return: corrupted set, binary mask
    """
    cx_set = np.zeros((x_set.shape[0], x_set.shape[1],
                        x_set.shape[2], x_set.shape[3]), dtype='float')
    cx_set_mask = np.zeros((x_set.shape[0], x_set.shape[1],
                            x_set.shape[2], x_set.shape[3]), dtype='int')
    seed_value = 0
    for sample_i in range(x_set.shape[0]):
        for node_i in range(x_set.shape[1]):
            cx_set[sample_i, node_i], cx_set_mask[sample_i, node_i] = \
                degrade_dataset(X=x_set[sample_i, node_i], missingness=missingness,
                                v=v_value, method=corrupt_method, size=block_size,
                                seed_value=seed_value)
            seed_value += 1

    return cx_set, cx_set_mask


def preprocess_data(x_set, cx_set):  # Pass by reference!
    """
       Preprocess the complete and corrupted data.
    :param x_set: complete dataset
    :param cx_set: corrupted dataset

    :return: corrupted set, binary mask
    """
    x_set0 = copy.deepcopy(x_set)
    cx_set0 = copy.deepcopy(cx_set)
    for sample_i in range(cx_set.shape[0]):
        for node_i in range(cx_set.shape[1]):
            scaler_x = preprocessing.MinMaxScaler()
            x_set0[sample_i, node_i] = \
                scaler_x.fit_transform(x_set0[sample_i, node_i])

            scaler_cx = preprocessing.MinMaxScaler()
            cx_set0[sample_i, node_i] = \
                scaler_cx.fit_transform(cx_set0[sample_i, node_i])

    return x_set0, cx_set0


if __name__ == "__main__":
    """
      The main function.
    """
    import torch.nn as nn
    m = nn.Dropout(p=0.5, inplace=True)
    input0 = torch.ones(5, 3)
    print(input0)
    output0 = m(input0)
    print(output0)

