import csv
import copy
import random
import torch
import numpy as np
from STGCN_utils import corrupt_dataset
from STGCN_utils import preprocess_data
from STGCN_core import STGCN


def do_STGCN(train_set, validation_set, test_set, missingness, verbose, corrupt_method,
             block_size, theta, epoch_num, time_step, size_step, auto_lr, fine_tune=False):
    """
       The core function to do STGCN imputation.
    :param train_set: training set
    :param validation_set: validation set
    :param test_set: test set
    :param missingness: value of missingness
    :param verbose: whther to print the training process
    :param corrupt_method: "block" or "nonblock"
    :param block_size: size of block
    :param theta: adjust the loss function
    :param epoch_num: number of epochs
    :param time_step: value of time step
    :param size_step: adjust the output dimension
    :param auto_lr: learning rate
    :param fine_tune: whether fine tune or not

    :output: the imputation performance
    """
    # Degrade three sets
    x_train = copy.deepcopy(train_set)  # The feature matrix of train
    x_validation = copy.deepcopy(validation_set)  # The feature matrix of train
    x_test = copy.deepcopy(test_set)  # The feature matrix of test

    v_value = 0  # You can replace the missing value as 0 or np.nan
    cx_train, cx_train_mask = corrupt_dataset(x_set=x_train,
                                              missingness=missingness,
                                              v_value=v_value,
                                              corrupt_method=corrupt_method,
                                              block_size=block_size)
    cx_validation, cx_validation_mask = corrupt_dataset(x_set=x_validation,
                                                        missingness=missingness,
                                                        v_value=v_value,
                                                        corrupt_method=corrupt_method,
                                                        block_size=block_size)
    cx_test, cx_test_mask = corrupt_dataset(x_set=x_test,
                                            missingness=missingness,
                                            v_value=v_value,
                                            corrupt_method=corrupt_method,
                                            block_size=block_size)
    # After preprocessing
    x_train0, cx_train0 = preprocess_data(x_set=x_train, cx_set=cx_train)
    x_validation0, cx_validation0 = preprocess_data(x_set=x_validation, cx_set=cx_validation)  # Pass by reference!
    x_test0, cx_test0 = preprocess_data(x_set=x_test, cx_set=cx_test)

    stgcn_model = STGCN(x_train0=x_train0, cx_train0=cx_train0, cx_train_mask=cx_train_mask,
                        cx_validation0=cx_validation0, cx_validation_mask=cx_validation_mask,
                        cx_validation=cx_validation, x_validation=x_validation, theta=theta,
                        time_step=time_step, size_step=size_step, auto_lr=auto_lr)

    stgcn_model.fit(epochs=epoch_num, missingness=missingness, block_size=block_size, verbose=verbose, fine_tune=fine_tune)
    stgcn_model.transform(cx_test0=cx_test0, cx_test_mask=cx_test_mask, cx_test=cx_test,
                          x_test=x_test, missingness=missingness, block_size=block_size)


if __name__ == "__main__":
    """
      The main function.
    """
    a = [1, 2, 3]
    b = a
    c = a[:]
    a[0] = 2
    print(b)
    print(c)