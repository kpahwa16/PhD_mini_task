from CF_GAN import train_cfgan
import argparse
import utils
from sklearn.metrics import mean_squared_error
import numpy as np
from data_loader import Law
import matplotlib.pyplot as plt
import pickle
from scipy.stats import gaussian_kde
import seaborn as sns
from plotutils import *

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=20, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.05, help="learning rate")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=5, help="dimensionality of the latent space")
parser.add_argument("--lam", type=float, default=0.01, help="cf regularisation")
parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")
parser.add_argument("--clip_value", type=float, default=0.01, help="lower and upper clip value for disc. weights")
parser.add_argument("--sensitive_dim", type=int, default=8, help="dimension of sensitive attributes")
opt = parser.parse_args()
print(opt)

# Creating training and test datasets
X_train, y_train, X_test, y_test, df_train, A, norm_fac = Law(norm=True)

X_train_unaware = X_train[:, 8:]  # Removing race columns
# X_train_unaware = X_train_unaware[:, :-2]  # removing gender

X_test_unaware = X_test[:, 8:]  # removing gender
# X_test_unaware = X_test_unaware[:, :-2]   # removing gender

X_in = np.concatenate((X_train, y_train.reshape(-1, 1)), axis=1)
# p_a = np.sum(X_train[:, 0:8], axis=0) / 17432
X_fair = train_cfgan(X_in, opt)
X_fair = X_fair.numpy()
y_fair = X_fair[:, -1]
X_fair = X_fair[:, 0:12]

model_type = 'randomforestReg'
model_full = utils.supervised_model_training(X_train, y_train, model_type)
model_unaware = utils.supervised_model_training(X_train_unaware, y_train, model_type)
model_fair = utils.supervised_model_training(X_fair, y_fair, model_type)


predictions_full = model_full.predict(X_test)
RMSE_full = np.sqrt(mean_squared_error(y_test, predictions_full))

predictions_unaware = model_unaware.predict(X_test_unaware)
RMSE_unaware = np.sqrt(mean_squared_error(y_test, predictions_unaware))

predictions_fair = model_fair.predict(X_test)
RMSE_fair = np.sqrt(mean_squared_error(y_test, predictions_fair))

print("RMSE of full model is", RMSE_full)
print("RMSE of unaware model is", RMSE_unaware)
print("RMSE of fair model is", RMSE_fair)

