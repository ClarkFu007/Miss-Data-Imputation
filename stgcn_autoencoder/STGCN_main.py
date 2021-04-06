import time
import numpy as np
from STGCN_utils import get_dataset
from STGCN_utils import split_tr_val_te
from STGCN_impute import do_STGCN


def main():
    # Input the original dataset.
    datafile_B1 = "Bus_1.csv"
    datafile_FA = "Feeder A.csv"
    datafile_FB = "Feeder B.csv"
    datafile_FC = "Feeder C.csv"
    dataSet = get_dataset(datafile_B1, datafile_FA, datafile_FB, datafile_FC)
    print("The original dataSet", dataSet.shape)

    # Get the training set, validation set, and test set
    time_step = 24
    train_num, validation_num = 275, 30
    trainSet, validationSet, testSet = \
        split_tr_val_te(dataset=dataSet, time_step=time_step,
                        train_num=train_num,
                        validation_num=validation_num)
    print("Training set", trainSet.shape)
    print("Validation set", validationSet.shape)
    print("Test set", testSet.shape)
    print(" ")

    epoch_num = 150
    verbose = False
    auto_lr, size_step = 0.01, 30

    # The number of nodes in each layer may be important! Smaller learning rate might be okay.0.01, size_step30
    do_STGCN(train_set=trainSet, validation_set=validationSet, test_set=testSet,
             missingness=0.2, verbose=verbose, corrupt_method="nonblock", block_size=1,
             theta=1, epoch_num=epoch_num, time_step=time_step, size_step=size_step, auto_lr=auto_lr, fine_tune=False)
    do_STGCN(train_set=trainSet, validation_set=validationSet, test_set=testSet,
             missingness=0.4, verbose=verbose, corrupt_method="nonblock", block_size=1,
             theta=1, epoch_num=epoch_num, time_step=time_step, size_step=size_step, auto_lr=auto_lr, fine_tune=False)
    do_STGCN(train_set=trainSet, validation_set=validationSet, test_set=testSet,
             missingness=0.5, verbose=verbose, corrupt_method="nonblock", block_size=1,
             theta=1, epoch_num=epoch_num, time_step=time_step, size_step=size_step, auto_lr=auto_lr, fine_tune=False)

    do_STGCN(train_set=trainSet, validation_set=validationSet, test_set=testSet,
             missingness=0.15, verbose=verbose, corrupt_method="block", block_size=3,
             theta=1, epoch_num=epoch_num, time_step=time_step, size_step=size_step, auto_lr=auto_lr, fine_tune=False)
    do_STGCN(train_set=trainSet, validation_set=validationSet, test_set=testSet,
             missingness=0.2, verbose=verbose, corrupt_method="block", block_size=3,
             theta=1, epoch_num=epoch_num, time_step=time_step, size_step=size_step, auto_lr=auto_lr, fine_tune=False)
    do_STGCN(train_set=trainSet, validation_set=validationSet, test_set=testSet,
             missingness=0.1, verbose=verbose, corrupt_method="block", block_size=4,
             theta=1, epoch_num=epoch_num, time_step=time_step, size_step=size_step, auto_lr=auto_lr, fine_tune=False)
    do_STGCN(train_set=trainSet, validation_set=validationSet, test_set=testSet,
             missingness=0.15, verbose=verbose, corrupt_method="block", block_size=4,
             theta=1, epoch_num=epoch_num, time_step=time_step, size_step=size_step, auto_lr=auto_lr, fine_tune=False)

    do_STGCN(train_set=trainSet, validation_set=validationSet, test_set=testSet,
             missingness=0.05, verbose=verbose, corrupt_method="block", block_size=8,
             theta=1, epoch_num=epoch_num, time_step=time_step, size_step=size_step, auto_lr=auto_lr, fine_tune=False)
    do_STGCN(train_set=trainSet, validation_set=validationSet, test_set=testSet,
             missingness=0.08, verbose=verbose, corrupt_method="block", block_size=8,
             theta=1, epoch_num=epoch_num, time_step=time_step, size_step=size_step, auto_lr=auto_lr, fine_tune=False)


if __name__ == '__main__':
    main()





