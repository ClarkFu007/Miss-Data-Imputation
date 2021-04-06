import time
import numpy as np
from ConvA_utils import get_dataset
from ConvA_utils import split_tr_val_te
from ConvA_impute import do_ConvA


def main():
    # Input the original dataset.
    """
    datafile_B1 = r"C:/My Files/Academic Files/Graduate Study/" \
                  "University of Southern California/Academic Material/" \
                  "Research/Codes/ginn/Smart Meter Data/Dataset/Baseline/Bus_1.csv"
    datafile_FA = r"C:/My Files/Academic Files/Graduate Study/" \
                  "University of Southern California/Academic Material/" \
                  "Research/Codes/ginn/Smart Meter Data/Dataset/Baseline/Feeder A.csv"
    datafile_FB = r"C:/My Files/Academic Files/Graduate Study/" \
                  "University of Southern California/Academic Material/" \
                  "Research/Codes/ginn/Smart Meter Data/Dataset/Baseline/Feeder B.csv"
    datafile_FC = r"C:/My Files/Academic Files/Graduate Study/" \
                  "University of Southern California/Academic Material/" \
                  "Research/Codes/ginn/Smart Meter Data/Dataset/Baseline/Feeder C.csv"
    """
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

    do_ConvA(train_set=trainSet, validation_set=validationSet,
             test_set=testSet, missingness=0.2, verbose=False,
             corrupt_method="nonblock", block_size=1, theta=1, epoch_num=100)

    do_ConvA(train_set=trainSet, validation_set=validationSet,
             test_set=testSet, missingness=0.4, verbose=False,
             corrupt_method="nonblock", block_size=1, theta=1, epoch_num=100)

    do_ConvA(train_set=trainSet, validation_set=validationSet,
             test_set=testSet, missingness=0.5, verbose=False,
             corrupt_method="nonblock", block_size=1, theta=1, epoch_num=100)

    do_ConvA(train_set=trainSet, validation_set=validationSet,
             test_set=testSet, missingness=0.15, verbose=False,
             corrupt_method="block", block_size=3, theta=1, epoch_num=100)

    do_ConvA(train_set=trainSet, validation_set=validationSet,
             test_set=testSet, missingness=0.2, verbose=False,
             corrupt_method="block", block_size=3, theta=1, epoch_num=100)

    do_ConvA(train_set=trainSet, validation_set=validationSet,
             test_set=testSet, missingness=0.1, verbose=False,
             corrupt_method="block", block_size=4, theta=1, epoch_num=100)

    do_ConvA(train_set=trainSet, validation_set=validationSet,
             test_set=testSet, missingness=0.15, verbose=False,
             corrupt_method="block", block_size=4, theta=1, epoch_num=100)
             
    do_ConvA(train_set=trainSet, validation_set=validationSet,
             test_set=testSet, missingness=0.05, verbose=False,
             corrupt_method="block", block_size=8, theta=1, epoch_num=100)

    do_ConvA(train_set=trainSet, validation_set=validationSet,
             test_set=testSet, missingness=0.08, verbose=False,
             corrupt_method="block", block_size=8, theta=1, epoch_num=100)


if __name__ == '__main__':
    main()







