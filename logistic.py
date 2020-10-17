import numpy as np
import pandas as pd
from sklearn.datasets import load_svmlight_file
from scipy.stats import logistic
import analysis_utils
import time
import random
import json
import string
import math
import pathlib
sigm = logistic.cdf

class LogiBinaryLibSVM():
    def __init__(self, dataset_name, lambd):
        """
        Initialize the dataset with l2-regularization strength lambd. This completely specifies the objectives.

        Arguments:
            dataset_name: name of LibSVM dataset - downloaded from https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/
            lambd: l2-regularization strength
        """
        self.X_train, self.y_train = load_svmlight_file(
            "./libsvm_datasets/"+dataset_name)
        # dimension: (n_samples, n_features)
        self.X_train = self.X_train.toarray()
        # self.y_train = np.expand_dims(self.y_train, axis=1)
 
        self.n_samples, self.n_features = self.X_train.shape
        self.weight_shape = (self.n_features,)
        self.lambd = lambd
        self.dataset = dataset_name
        pass

    def sample_grad(self, weight, batch_size):
        """
        Sample minibatch gradients with size batch_size (For MB-SGD and MB-AC-SGD)

        Arguments:
            weight: w
            batch_size: number of samples queried to compute the sample gradient.

        Return:
            Averaged gradients
        """
        samples_idx = np.random.choice(self.n_samples, batch_size, replace=True)
        X_sampled = self.X_train[samples_idx, :]
        y_sampled = self.y_train[samples_idx]

        grad = -((y_sampled * (1 - sigm((X_sampled @ weight) * y_sampled))).T @ X_sampled) / \
            len(y_sampled) + self.lambd * weight

        return grad

    def sample_grad_pool(self, weight_pool, M, local_batch=1):
        """
        Efficiently sample gradients for M workers for FedAc and FedAvg.      
        
        Arguments: 
            weight_pool: weight held by M workers, dimension (M, n_features)
            M:   number of workers
            local_batch: batch size for each local worker (default to be 1)

        Return: 
            M gradient vectors of dimension (M, n_features)
        """
        # sampling
        samples_idx = np.random.choice(
            self.n_samples, local_batch * M, replace=True)

        X_bmp = self.X_train[samples_idx, :].reshape(
            local_batch, M, self.n_features)
        # (local_batch, M, n_features)
        Y_bm = self.y_train[samples_idx].reshape(local_batch, M)
        # (local_batch, M)

        tmp1 = np.sum(X_bmp * weight_pool, axis=-1)
        tmp2 = (-Y_bm*(1-(sigm(Y_bm * tmp1))))
        grad_likelihood = np.mean(tmp2[:, :, np.newaxis]*X_bmp, axis=0)
        return grad_likelihood + self.lambd * weight_pool

    def loss(self, X, Y, weight):
        """
        Compute the loss.
        """
        return -np.mean(np.log(sigm((X @ weight) * Y))) + 0.5 * self.lambd * np.linalg.norm(weight) ** 2

    def population_loss(self, weight):
        """
        Compute the population loss for all training samples.

        Argument:
            weight: w
        """        
        return self.loss(self.X_train, self.y_train, weight)

    def broadcast_avg(self, pool):
        """
        Helper functions for FedAc and FedAvg, average and broadcast the weights.
        """
        avg = pool.mean(axis=0)
        pool = np.repeat(avg[np.newaxis, :], pool.shape[0], axis=0)
        return pool

    def mbsgd(self, eta, M, K, T, local_batch, record_intvl=512, print_intvl=8192, SEED=0):
        """
        Simulate distributed Minibatch-SGD (MB-SGD)

        Arguments:
            eta: learning rate
            M:   number of workers
            K:   synchronization interval
            T:   total parallel runtime
            record_intvl:   compute the population loss every record_intvl steps.

        Return:
            A pandas.Series object of population loss evaluated.
        """
        np.random.seed(SEED)
        assert(T % K == 0)
        w = np.random.randn(*self.weight_shape)
        seq = pd.Series(name='loss')
        for iter_cnt in range(0, T+1, K):
            if iter_cnt % record_intvl == 0:
                seq.at[iter_cnt] = self.population_loss(w)
            w -= eta * self.sample_grad(w, M*K*local_batch)
        return seq

    def mbasgd(self, eta, gamma, alpha, beta, M, K, T, local_batch, record_intvl=512, print_intvl=8192, SEED=0):
        """
        Simulate distributed Minibatch-Accelerated-SGD (MB-AC-SGD)

        Arguments:
            eta:    learning rate
            gamma, alpha, beta:  hyperparameters
            M:      number of workers
            K:      synchronization interval
            T:      total parallel runtime
            record_intvl:   compute the population loss every record_intvl steps.

        Return:
            A pandas.Series object of population loss evaluated.
        """
        np.random.seed(SEED)
        assert(T % K == 0)
        w = np.random.randn(*self.weight_shape)
        w_ag = np.copy(w)
        seq = pd.Series(name='loss')
        for iter_cnt in range(0, T+1, K):
            if iter_cnt % record_intvl == 0:
                seq.at[iter_cnt] = self.population_loss(w_ag)
            w_md = (1/beta) * w + (1-(1/beta))*w_ag
            grad_md = self.sample_grad(w_md, M*K*local_batch)
            w_ag = w_md - eta * grad_md
            w = (1 - (1/alpha)) * w + (1/alpha) * \
                w_md - gamma * grad_md
        return seq

    def fedavg(self, eta, M, K, T, local_batch, record_intvl=512, print_intvl=8192, SEED=0):
        """
        Simulate Federated Averaging (FedAvg, a.k.a. Local-SGD, or Parallel SGD, etc.)

        Arguments:
            eta:    learning rate
            M:      number of workers
            K:      synchronization interval, (i.e., local steps)
            T:      total parallel runtime
            record_intvl:   compute the population loss every record_intvl steps.

        Return:
            A pandas.Series object of population loss evaluated.
        """
        np.random.seed(SEED)
        common_init_w = np.random.randn(*self.weight_shape)
        w_pool = np.repeat(common_init_w[np.newaxis, :], M, axis=0)

        seq = pd.Series(name='loss')
        for iter_cnt in range(T+1):
            if iter_cnt % K == 0:
                w_pool = self.broadcast_avg(w_pool)

                if iter_cnt % record_intvl == 0:
                    seq.at[iter_cnt] = self.population_loss(w_pool[0, :])

            w_pool -= eta * self.sample_grad_pool(w_pool, M, local_batch)
        return seq

    def fedac(self, eta, gamma, alpha, beta, M, K, T, local_batch, record_intvl=512, print_intvl=8192, SEED=0):
        """
        Simulate FedAc

        Arguments:
            eta:    learning rate
            gamma, alpha, beta:  hyperparameters
            M:      number of workers
            K:      synchronization interval
            T:      total parallel runtime
            record_intvl:   compute the population loss every record_intvl steps.

        Return:
            A pandas.Series object of population loss evaluated.
        """
        np.random.seed(SEED)
        common_init_w = np.random.randn(*self.weight_shape)
        w_pool = np.repeat(common_init_w[np.newaxis, :], M, axis=0)
        w_ag_pool = np.copy(w_pool)

        seq = pd.Series(name='loss')

        for iter_cnt in range(T+1):
            if iter_cnt % K == 0:
                w_pool = self.broadcast_avg(w_pool)
                w_ag_pool = self.broadcast_avg(w_ag_pool)

                if iter_cnt % record_intvl == 0:
                    seq.at[iter_cnt] = self.population_loss(w_ag_pool[0, :])

            w_md_pool = (1/beta) * w_pool + (1-(1/beta))*w_ag_pool
            grad_md_pool = self.sample_grad_pool(w_md_pool, M, local_batch)
            w_ag_pool = w_md_pool - eta * grad_md_pool
            w_pool = (1 - (1/alpha)) * w_pool + (1/alpha) * \
                w_md_pool - gamma * grad_md_pool

        return seq

    def train(self, alg, eta, M, K, T, local_batch, seed, record_intvl):
        """
        Training

        Arguments:
            alg:    algorithms. 
                    -1: Naive FedAc
                    0:  FedAvg
                    1:  FedAc-I
                    2:  FedAc-II
                    3:  MB-SGD
                    4:  MB-AC-SGD
            eta:    learning rate
            M:      number of workers
            K:      synchronization interval
            T:      total parallel runtime
            record_intvl:   compute the population loss every record_intvl steps.

        Return:
            save a config file in .json and a sequence of loss evaluated in .csv.
        """


        args = {'dataset': self.dataset,
                'lambd': self.lambd,
                'alg': alg,
                'eta': eta,
                'M': M,
                'K': K,
                'T': T,
                'local_batch': local_batch,
                'seed': seed,
                'record_intvl': record_intvl}
        expr_mgr = analysis_utils.ExprMgr(args['dataset'], args['lambd'])
        out_dir = expr_mgr.out_dir

        runid = time.strftime(
            "%m%dT%H%M%SZ_") + ''.join(random.choices(string.ascii_letters + string.digits, k=6))

        with open(out_dir / f'{runid}.json', 'w') as f:
            json.dump(args, f, indent=2)

        if args['alg'] == 0:      # FedAvg
            seq = self.fedavg(eta=args['eta'], 
                                 M=args['M'],
                                 K=args['K'],
                                 T=args['T'],
                                 local_batch=args['local_batch'],
                                 record_intvl=args['record_intvl'],
                                 SEED=args['seed'])
        elif args['alg'] == 3:
            seq = self.mbsgd(eta=args['eta'], 
                                 M=args['M'],
                                 K=args['K'],
                                 T=args['T'],
                                 local_batch=args['local_batch'],
                                 record_intvl=args['record_intvl'],
                                 SEED=args['seed'])
        elif args['alg'] == 4:
            gamma = math.sqrt(args['eta']/(args['lambd']))
            alpha = 1/(gamma * args['lambd'])
            beta = alpha + 1
            seq = self.mbasgd(eta=args['eta'], 
                                gamma = gamma,
                                alpha = alpha,
                                beta = beta,
                                 M=args['M'],
                                 K=args['K'],
                                 T=args['T'],
                                 local_batch=args['local_batch'],
                                 record_intvl=args['record_intvl'],
                                 SEED=args['seed'])
        else:
            if args['alg'] == -1: # Naive FedAc
                gamma = math.sqrt(args['eta']/(args['lambd']))
                alpha = 1/(gamma * args['lambd'])
                beta = alpha + 1
            elif args['alg'] == 1: # FedAc-I
                gamma = max(args['eta'], math.sqrt(
                    args['eta']/(args['lambd']*args['K'])))
                alpha = 1/(gamma * args['lambd'])
                beta = alpha + 1
            elif args['alg'] == 2: # FedAc-II
                gamma = max(args['eta'], math.sqrt(
                    args['eta']/(args['lambd']*args['K'])))
                alpha = 3/(2 * gamma * args['lambd']) - (1/2)
                beta = (2*alpha*alpha)/(alpha-1)

            seq = self.fedac(eta=args['eta'], gamma=gamma,
                                  alpha=alpha, beta=beta,
                                  M=args['M'],
                                  K=args['K'],
                                  T=args['T'],
                                  local_batch=args['local_batch'],
                                  record_intvl=args['record_intvl'],
                                  SEED=args['seed'])

        seq.to_csv(out_dir / f'{runid}.csv')
