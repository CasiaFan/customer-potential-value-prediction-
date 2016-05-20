# This is a script for calculating a customer's potential value in sale for a shop.
# The algorithm is that:
# 1. we assume the total number of transaction of a customer in such kind of shops follows poisson distribution.
# The parameter to influence the distribution is lambda and we speculate it is related to total money have spent
# and the time period.So it could help us to infer the total size of wallet of the customer.
# 2. we assume the number of transaction of a customer in specific shop follows negative binomial distribution whose
# possibility is when choosing this shop is customer-specific constant Pai(i). We speculate that the Pai(i) is related
# to the average time interval and recency to the same shop. This would help us to determine the share of wallet.
# 3. We define a term - potential of wallet which equals N_total - N_cur to describe the consumption potential.
# reference: predicting customer wallet without survey data

import numpy as np
import scipy as sp
import scipy.optimize as op
import scipy.misc
import scipy.stats as ss
import pandas as pd
import datetime as dt
import sklearn.preprocessing as skp
def get_time_interval(pre_date, late_date):
    # calculate period between 2 transaction
    pre = pre_date.split('/')
    late = late_date.split('/')
    # calculate the time interval
    delta = dt.date(int(late[2]), int(late[0]), int(late[1])) - dt.date(int(pre[2]), int(pre[0]), int(pre[1]))
    cyc = str(delta.days)
    return cyc

# GBM_LL: log-likelihood of the Generalized Binomial Model (GBM)
# Inputs: all date should be numpy array
# params: contains alpha (first dim_V) and beta (later dim_W)
# - alpha for V regressor to optimize lambda parameter in poisson distribution (2 x 1)
# - beta for W regressor to optimize Pi parameter in binomial distribution (2 x 1)
# data: contains transaction data and V and W parameters
# - V is the total mount of money and total spent time period of customer i together with intercept column(n x 3)
# - W is the average time interval and recency of customer i together with intercept (n x 3)
# - x is the transaction count in the shop (n x 1)
# Return function f and g: f is log likelihood function during maximum likelihood estimation (equation 18 in reference)
# g is gradient function during maximum likelihood estimation (equation 19, 20)
# the cost objective function
def GBM_LL(params, data):
    # get dimensions of V
    n_V  = int(data[0, 0])
    # split data into x, V, W
    x = np.matrix(data[:, 1]).T
    V = data[:, 2:2+n_V]
    W = data[:, 2+n_V:]
    # get dimension of W
    n_W = int(W.shape[1])
    # number of customer
    n = x.size
    # get alpha and beta
    beta = np.matrix(params[:n_V]).T
    alpha = np.matrix(params[n_V:]).T
    # get lambda i for poisson distribution and Pi i for binomial distribution
    lambdai = np.exp(V.dot(beta))
    pi = 1/(1 + np.exp(-W.dot(alpha)))
    # log-likelihood function
    lp = np.multiply(pi, lambdai)
    f = (-np.multiply(x, np.log(lp)) + lp + sp.misc.factorial(x)).sum()
    return f

# the gradient function of cost function
def GBM_LL_gradient(params, data):
    # get dimensions of V
    n_V  = int(data[0, 0])
    # split data into x, V, W
    x = np.matrix(data[:, 1]).T
    V = data[:, 2:2+n_V]
    W = data[:, 2+n_V:]
    # get dimension of W
    n_W = int(W.shape[1])
    # number of customer
    n = x.size
    # get alpha and beta
    beta = np.matrix(params[:n_V]).T
    alpha = np.matrix(params[n_V:]).T
    # get lambda i for poisson distribution and Pi i for binomial distribution
    lambdai = np.exp(V.dot(beta))
    pi = 1./(1. + np.exp(-W.dot(alpha)))
    # log-likelihood function
    lp = np.multiply(pi, lambdai)
    # compute the gradient
    # for beta gradient
    gbeta = (np.multiply((lp - x) * np.ones((1, n_V)), V)).sum(axis=0)
    # for alpha gradient
    galpha = (np.multiply(np.multiply(lp - x, 1 - pi) * np.ones((1, n_W)), W)).sum(axis=0)
    return [gbeta, galpha]

# This programs estimates the parameters of the model using the number of transactions a customer does at the focal company x_i,
# the explanatory variables accounting for the Size-of-Wallet, V, and for the Share-of-Wallet, W.
# Inputs:
# - xi is the number of transactions done by a customer
# - V is the matrix of explanatory variables accounting for the size of wallet.
# Here for it has relation with money amount and total time period and add a column of 1 for an intercept,
# the dimensions of V should be n x 3
# - W is the matrix of explanatory variables accounting for the share of wallet.
# Here for it has relation with average time interval and recency and add a column of 1 for an intercept,
# the dimensions of W should be n x 3
# Output:
# results_beta: the estimates of the parameters related to the size of wallet
# results_alpha: the estimates of the parameters related to the share of wallet
# results_ni: predicted size of wallet
# results_lambdai: predicted size of wallet parameter
# results_pi: predicted share of wallet parameter
def predict_wallet(xi, V, W):
    # get dimension of V and W
    n_V = V.shape[1]
    n_W = W.shape[1]
    # get number of customers
    n = xi.size
    # set initial parameters
    params_init = np.zeros((n_W+n_V, 1))
    # merge dimension of V, x, V, W to a data.frame for input of GBM_LL
    data = np.c_[n_V * np.ones((n, 1)), xi, V, W]
    # estimate and get beta and alpha parameters that minimize the log-likelihood function
    pre_params = op.minimize(GBM_LL, x0=params_init, args=data, method='TNC', jac=GBM_LL_gradient, options={'maxiter': 400}, tol=1e-6)
    # pre_params = op.minimize(GBM_LL, x0=params_init, args=data, method='Nelder-Mead', options={'maxiter': 400}, tol=1e-6)
    pre_params_ab = pre_params.x
    # get predicted beta and alpha
    pre_beta = pre_params_ab[:n_V].T
    pre_alpha = pre_params_ab[n_V:].T
    # get predicted lambda
    pre_lambda = np.exp(V.dot(pre_beta))
    # for lambda is the estimator of poisson distribution, so it shouldn't be negative.
    pre_lambda = np.where(pre_lambda > 0, pre_lambda, 0)
    # calculate predicted total transaction number of customer i, ni
    ni = [0 for m in range(0, n)]
    for i in range(n):
        # ni should never be less than xi (local shop)
        if pre_lambda[i] < xi[i]:
            ni[i] = xi[i]
        else:
            ni[i] = pre_lambda[i] * (1 - ss.poisson.cdf(xi[i]-2, pre_lambda[i]) / (1 - ss.poisson.cdf(xi[i]-1, pre_lambda[i])))

    # get predicted pi parameter
    pre_pi = 1/(1 + np.exp(-W.dot(pre_alpha)))
    return pre_beta, pre_alpha, ni, pre_lambda, pre_pi

# import real data to test
fileIn = 'C:\Users\Fan\Desktop\T40504.freq.update.signma.csv'
# output file contains alpha and beta parameter
file_ab = 'C:\Users\Fan\Desktop\T40504.pow.ab.csv'
# output file contains predicted ni and xi of local shop
file_pow = 'C:\Users\Fan\Desktop\T40504.pow.csv'
# column need for further analysis
items = ['customer_id', 'total_freq', 'transaction_date', 'total_points', 'mean_freq', 'days_to_last_transaction']
# number of runs and different initial values of the estimation process
iter = 20
df = pd.read_csv(fileIn)
# get mean_init_interval for those with few transaction times (n <= 3)
mean_init_interval = df['mean_initial_freq'].iloc[0]
# retrieve columns for further analysis
df = df[items]
################################################################################################
# data cleaning: remove irregular values
# remove customers those with only one transaction / only analysis customers with more than 4 transactions
df = df[df['total_freq'] > 4]
# remove customers whose points is negative
df = df[df['total_points'] > 1]
# remove irregular high points
df = df[df['total_points'] < 1000]
################################################################################################
# get total transaction time period
n_cust = df.shape[0]
total_period = []
for k in range(0, n_cust):
    # get first date
    first_date = df['transaction_date'].iloc[k].split(';')[0]
    # get last date
    last_date = df['transaction_date'].iloc[k].split(';')[-1]
    total_period.append(get_time_interval(first_date, last_date))

# replace transaction date column with time period and rename the column
# NOTE: the index is important for alignment of the value, so reset the index without gap
df.index = pd.Series(range(n_cust))
df['transaction_date'] = pd.Series(total_period)
# replace NA in mean interval with mean_init_interval
df['mean_freq'].fillna(mean_init_interval, inplace=True)
df.columns = ['customer_id', 'freq', 'time_period', 'points', 'mean_interval', 'recency']
# numeric data with float type and log conversion (except freq)
df[['freq', 'time_period', 'points', 'mean_interval', 'recency']] = df[['freq', 'time_period', 'points', 'mean_interval', 'recency']].astype(float)
# restore initial local transaction count
xi_init = df['freq']
time_period = df['time_period']
recency = df['recency']
points = df['points']
mean_interval = df['mean_interval']
# min and max date set of freq for later restore
min_freq = df['freq'].min()
max_freq = df['freq'].max()
# For we think time_period is negatively related to total money amount and recency & mean_interval are negatively related to share of wallet,
# we will revert their value by reciprocal conversion (1/n)
df[['time_period', 'mean_interval', 'recency']] = 1/df[['time_period', 'mean_interval', 'recency']]
# Note: scale data from 0 to 1
df[['freq', 'time_period', 'points', 'mean_interval', 'recency']] = skp.minmax_scale(df[['freq', 'time_period', 'points', 'mean_interval', 'recency']], feature_range=(0, 1))

# make sure there are no nan in df
try:
    assert df.isnull().any().sum() == 0
except AssertionError:
    print "df has nan element"
    print df.isnull.any()
# get V and W explanatory variables
V = pd.DataFrame(df[['time_period', 'points']])
V['intercept'] = pd.Series([1 for j in range(0, n_cust)])
W = pd.DataFrame(df[['mean_interval', 'recency']])
W['intercept'] = pd.Series([1 for l in range(0, n_cust)])
# get transaction counts in local shop
xi = df['freq']

# repeat iter times, not to use continuous '=' for they will point to one df which means one change all change
est_beta = pd.DataFrame()
est_alpha = pd.DataFrame()
est_ni = pd.DataFrame()
est_lambda = pd.DataFrame()
est_pi = pd.DataFrame()
for count in range(iter):
    # function estimating the parameters for prediction
    pre_beta, pre_alpha, ni, pre_lambda, pre_pi = predict_wallet(xi, V, W)
    est_beta[count] = pd.Series(pre_beta)
    est_alpha[count] = pd.Series(pre_alpha)
    est_ni[count] = pd.Series(ni)
    est_lambda[count] = pd.Series(pre_lambda)
    est_pi[count] = pd.Series(pre_pi)

# calculate the mean estimate parameters
mean_est_beta = est_beta.mean(axis=1)
mean_est_alpha = est_alpha.mean(axis=1)
mean_est_ni = est_ni.mean(axis=1)
# restore scaled predicted transaction number to normal
mean_est_ni = mean_est_ni * (max_freq - min_freq) + min_freq
mean_est_lambda = est_lambda.mean(axis=1)
mean_est_pi = est_pi.mean(axis=1)
# output to csv
para_ab = pd.DataFrame({'beta': mean_est_beta, 'alpha': mean_est_alpha})
pow = pd.DataFrame({'customer_id': df['customer_id'], 'time_period': time_period, 'points': points, 'recency': recency, 'mean_interval': mean_interval, 'local_freq': xi_init, 'total_predicted_freq': mean_est_ni, 'lambda': mean_est_lambda, 'pi': mean_est_pi})
para_ab.to_csv(file_ab)
pow.to_csv(file_pow)
