import math
import time
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

from numpy import fabs, sum, square, sqrt
# from ConvA_utils import *
from ConvA_models import ConvAutoencoderModel

torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class ConvAutoencoder(object):
    """
    Impute with GINN!
    Functions:
        __init__
        fit()
        transform()
    """

    def __init__(self, x_train0, cx_train0, cx_train_mask, cx_validation0,
                 cx_validation_mask, cx_validation, x_validation, theta,
                 auto_lr=0.01, weight_decay=0):
        """
            Build the graph-structure of the dataset based on the file
            Instantiate the network based on the graph using the dgl library
        """
        self.sample_num = x_train0.shape[0]  # The number of samples
        self.num_nodes = x_train0.shape[1]   # The number of nodes of each graph
        self.feat_num = x_train0.shape[2]    # The number of features

        self.x_train0 = x_train0             # The real training set after preprocessing
        self.cx_train0 = cx_train0           # The corrupted training set after preprocessing
        self.cx_train_mask = cx_train_mask   # The mask of the corrupted training set

        self.cx_validation0 = cx_validation0  # The corrupted validation set after preprocessing
        self.cx_validation_mask = cx_validation_mask  # The mask of the corrupted validation set
        self.cx_validation = cx_validation  # The corrupted validation set before preprocessing
        self.x_validation = x_validation      # The real validation set before preprocessing

        self.theta = theta                  # the theta value for the cost function
        self.auto_lr = auto_lr              # learning rate
        self.weight_decay = weight_decay    # weight decay

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("ConvA is running on", self.device)
        print("The version of PyTorch is", torch.__version__)

        from ConvA_models import ConvAutoencoderModel
        """
        Autoencoder with spatio-temporal graph convolutional layers.
        """
        # construct an autoencoder
        self.conv_autoencoder = ConvAutoencoderModel(num_nodes=self.num_nodes,
                                                     feat_num=self.feat_num).to(self.device)
        # construct an optimizer
        self.optim_auto = torch.optim.Adam(self.conv_autoencoder.parameters(),
                                           lr=self.auto_lr,
                                           betas=(0.9, 0.999),
                                           weight_decay=self.weight_decay)
        # and a learning rate scheduler
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optim_auto,
                                                            step_size=5,
                                                            gamma=0.9)
        return

    def fit(self, epochs, missingness, block_size, fine_tune=False, verbose=False):
        """
            Trains the network, if fine_tune=True uses the previous state
            of the optimizer instantiated before.
        """
        if fine_tune:
            checkpoint = torch.load("best_model" + str(missingness) + str(block_size) + ".pth")
            self.conv_autoencoder.load_state_dict(checkpoint["auto_state_dict"])
            self.optim_auto.load_state_dict(checkpoint["optim_auto_state_dict"])

        """
           Create criterions with respect to categorical columns and
        numerical columns.
        """
        mse_criterion = nn.MSELoss().to(self.device)

        start0 = time.time()
        batch_size = 25
        # early_stop = False
        minimum_MA_error = 10  # Make it big enough
        self.conv_autoencoder.train()
        for epoch in range(epochs):
            """
            if early_stop:
                break
            """
            t0, total_loss = time.time(), 0
            permutation = torch.randperm(self.sample_num)
            for batch_i in range(0, self.sample_num, batch_size):
                indices = permutation[batch_i:batch_i + batch_size]
                featT = self.cx_train0
                featT_r = self.x_train0
                featT = torch.FloatTensor(featT)
                featT_r = torch.FloatTensor(featT_r)

                X_batch, y_batch = featT[indices], featT_r[indices]
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                iX = self.conv_autoencoder(X_batch)  # Reconstruction
                num_loss = mse_criterion(iX, y_batch)
                           # global_criterion(torch.mean(y_batch, dim=0), torch.mean(iX, dim=0))
                self.optim_auto.zero_grad()  # Zero the gradients
                num_loss.backward()  # Calculate the gradients
                self.optim_auto.step()  # Update the weights

                total_loss += num_loss.item()

            self.lr_scheduler.step()  # update the learning rate
            mean_loss = total_loss

            # For validation
            vali_real_mean, vali_MA_error, vali_RME_error, \
            vali_nAE, vali_nAAE = get_results(cx_set0=self.cx_validation0, cx_mask=self.cx_validation_mask,
                                              device=self.device, imputation_model=self.conv_autoencoder,
                                              cx_set=self.cx_validation, x_set=self.x_validation)

            if verbose:
                print("Epoch is %04d/%04d, mean loss is: %f," % (epoch + 1, epochs, mean_loss))
                print('For valadation data, '
                      'real mean value: %.6f, '
                      'mean absolute error: %.6f, '
                      'root mean squared error: %.6f, '
                      'normalized absolute error: %.6f, '
                      'normalized accumulate absolute error: %.6f'
                      % (vali_real_mean, vali_MA_error, vali_RME_error, vali_nAE, vali_nAAE))
                print(" ")

            if minimum_MA_error > vali_MA_error:
                minimum_MA_error = vali_MA_error
                torch.save({"auto_state_dict": self.conv_autoencoder.state_dict(),
                            "optim_auto_state_dict": self.optim_auto.state_dict()},
                           "best_model" + str(missingness) + str(block_size) + ".pth")
            """
            if mean_loss < 1e-05:
                early_stop = True
            """

        end0 = time.time()
        total_time = end0 - start0
        print("The total running time is %f seconds" % (total_time))
        if epochs != 0:
            mean_time = total_time / epochs
            print("The mean training time is %f seconds" % (mean_time))
        # print("The mean training time is %f seconds" % (dur_train.mean()))

        return

    def transform(self, cx_test0, cx_test_mask, cx_test, x_test,
                  missingness, block_size):
        """
           Test the trained model.
        """
        dur_impu = np.zeros(x_test.shape[0], dtype='float')
        # model_path = "best_model" + str(missingness) + str(block_size) + ".pth"
        model_path = "best_model9.pth"
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        self.conv_autoencoder.load_state_dict(checkpoint["auto_state_dict"])
        self.optim_auto.load_state_dict(checkpoint["optim_auto_state_dict"])
        test_real_mean, test_MA_error, test_RME_error, \
        test_nAE, test_nAAE = get_results(cx_set0=cx_test0, cx_mask=cx_test_mask,
                                          device=self.device, imputation_model=self.conv_autoencoder,
                                          cx_set=cx_test, x_set=x_test)

        print('    For test data when missingness is %.2f, block size is %d' % (missingness, block_size))
        print('real mean value: %.6f' % (test_real_mean))
        print('mean absolute error: %.6f' % (test_MA_error))
        print('root mean squared error: %.6f' % (test_RME_error))
        print('normalized absolute error: %.6f' % (test_nAE))
        print('normalized accumulate absolute error: %.6f' % (test_nAAE))
        print(' ')


def do_imputation(sample_i, cx_set0, cx_mask, device, imputation_model, cx_set, x_set):
    """
       Do the imputation with the trained model (sub-function to get final results).
    :param sample_i: the ith sample
    :param cx_set0: corrupted data after preprocessing
    :param cx_mask: mask of corrupted data
    :param device: the running device
    :param imputation_model: the trained model
    :param cx_set: corrupted data before preprocessing
    :param x_set: real data before preprocessing

    :return: miss_num, x_real, error
    """
    missed_data = cx_set0[sample_i]
    mask = cx_mask[sample_i]
    missed_dataT = np.multiply(missed_data, mask)
    missed_dataT = np.expand_dims(missed_dataT, axis=0)
    missed_dataT = torch.FloatTensor(missed_dataT).to(device)

    # Reconstruction
    with torch.no_grad():
        imputation_model.eval()
        filled_data = imputation_model(missed_dataT)
    filled_data = filled_data.cpu().detach().numpy()
    filled_data = np.squeeze(filled_data, axis=0)
    max_vector = np.expand_dims(cx_set[sample_i].max(axis=0), axis=0)
    filled_data = np.multiply(filled_data, max_vector)
    """
    scaler = preprocessing.MinMaxScaler()
    scaler.fit(cx_set[sample_i])
    filled_data = scaler.inverse_transform(filled_data)
    """

    total_num = cx_set0.shape[1] * cx_set0.shape[2]
    miss_num = total_num - np.count_nonzero(cx_mask[sample_i])
    new_mask = np.ones((cx_mask.shape[1], cx_mask.shape[2]))
    new_mask = new_mask - cx_mask[sample_i]

    imputed_data = np.multiply(filled_data, new_mask)
    x_real = np.multiply(x_set[sample_i], new_mask)
    error = imputed_data - x_real

    return miss_num, x_real, error


def get_results(cx_set0, cx_mask, device,
                imputation_model, cx_set, x_set):
    """
       Get the performance of imputation (main function to get final results).
    :param cx_set0: corrupted data after preprocessing
    :param cx_mask: mask of corrupted data
    :param device: the running device
    :param imputation_model: the trained model
    :param cx_set: corrupted data before preprocessing
    :param x_set: real data before preprocessing

    :return: real_mean, MA_error, RME_error, nAE, nAAE
    """
    MA_error, RME_error = 0, 0
    nAE, nAAE = 0, 0
    real_mean = 0
    total_miss = 0
    for sample_i in range(cx_set0.shape[0]):
        miss_num, x_real, error = do_imputation(sample_i=sample_i, cx_set0=cx_set0,
                                                cx_mask=cx_mask, device=device,
                                                imputation_model=imputation_model,
                                                cx_set=cx_set, x_set=x_set)
        mean_sample_i = np.mean(x_set[sample_i])
        total_miss += miss_num
        real_mean += sum(fabs(x_real))
        MA_error += sum(fabs(error))
        RME_error += sum(square(error))
        nAE += sum(fabs(error)) / mean_sample_i
        nAAE += fabs(sum(error)) / mean_sample_i

    return real_mean / total_miss, MA_error / total_miss, \
           sqrt(RME_error / total_miss), nAE / total_miss, nAAE / total_miss