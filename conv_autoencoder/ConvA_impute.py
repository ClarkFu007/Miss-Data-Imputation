import copy
import random
import torch
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from ConvA_utils import corrupt_dataset, merge_feature
from ConvA_utils import preprocess_data


def do_ConvA(train_set, validation_set, test_set, missingness,
             verbose, corrupt_method, block_size, theta, epoch_num):
    """
       The core function to do MIDA imputation.
    :param train_set: training set
    :param validation_set: validation set
    :param test_set: test set
    :param missingness: value of missingness
    :param verbose: whther to print the training process
    :param corrupt_method: "block" or "nonblock"
    :param block_size: size of block
    :param theta: adjust the loss function
    :param epoch_num: number of epochs

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
    # After merging features
    num_cols = [0, 1]
    # For training data
    x_train = merge_feature(x_train, num_cols)
    cx_train = merge_feature(cx_train, num_cols)
    cx_train_mask = merge_feature(cx_train_mask, num_cols)
    # print("After merging features, x_train", x_train.shape)
    # print("After merging features, cx_train", cx_train.shape)
    # print("After merging features, cx_train_mask", cx_train_mask.shape)
    # For validation data
    x_validation = merge_feature(x_validation, num_cols)
    cx_validation = merge_feature(cx_validation, num_cols)
    cx_validation_mask = merge_feature(cx_validation_mask, num_cols)
    # print("After merging features, x_validation", x_validation.shape)
    # print("After merging features, cx_validation", cx_validation.shape)
    # print("After merging features, cx_validation_mask", cx_validation_mask.shape)
    # For test data
    x_test = merge_feature(x_test, num_cols)
    cx_test = merge_feature(cx_test, num_cols)
    cx_test_mask = merge_feature(cx_test_mask, num_cols)
    # print("After merging features, x_test", x_test.shape)
    # print("After merging features, cx_test", cx_test.shape)
    # print("After merging features, cx_test_mask", cx_test_mask.shape)

    x_train0, cx_train0 = preprocess_data(x_train, cx_train)
    x_validation0, cx_validation0 = preprocess_data(x_validation, cx_validation)  # Pass by reference!
    x_test0, cx_test0 = preprocess_data(x_test, cx_test)

    # print("After preprocessing, x_train0, cx_train0", x_train0.shape, cx_train0.shape)
    # print("After preprocessing, x_validation0, cx_validation0", x_validation0.shape, cx_validation0.shape)
    # print("After preprocessing, x_test0, cx_test0", x_test0.shape, cx_test0.shape)

    from ConvA_core import ConvAutoencoder
    my_model = ConvAutoencoder(x_train0=x_train0, cx_train0=cx_train0, cx_train_mask=cx_train_mask,
                               cx_validation0=cx_validation0, cx_validation_mask=cx_validation_mask,
                               cx_validation=cx_validation,
                               x_validation=x_validation, theta=theta)

    my_model.fit(epochs=epoch_num, missingness=missingness, block_size=block_size, verbose=verbose)
    my_model.transform(cx_test0=cx_test0, cx_test_mask=cx_test_mask, cx_test=cx_test, x_test=x_test,
                       missingness=missingness, block_size=block_size)


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