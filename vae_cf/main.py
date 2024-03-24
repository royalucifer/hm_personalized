import numpy as np
import scipy as sp
import pandas as pd
import tensorflow as tf
import tensorflow_ranking as tfr

from vae_cf.train import train_epoch
from vae_cf.model import VAE
from vae_cf.data import load_data
from vae_cf.dataset import to_tf_train_dataset, to_tf_eval_dataset

transactions_dir = "/Users/roysung/py_projects/hm_personal/dataset/transactions_train.csv"
tensorboard_dir = "/Users/roysung/py_projects/hm_personal/model_output/basic"


def generate_params(n_items):
    model_params = {
        'p_dims': [200, 600, n_items],
        'q_dims': [n_items, 600, 200],
        'drop_prob': 0.5,
        'l2_reg': 0.0,
        'seed': 98765}
    train_params = {
        'metrics': {
            'neg_elbo': tf.keras.metrics.Mean(name='train_neg_elbo'),
            'kl': tf.keras.metrics.Mean(name='train_kl'),
            'neg_ll': tf.keras.metrics.Mean(name='train_neg_ll'),
            'ndcg': tfr.keras.metrics.NDCGMetric(name='test_ndcg', topn=20)},
        'num_epochs': 3,
        'anneal_cap': 0.2,
        'total_anneal_steps': 200000,
        'summary_writer': tf.summary.create_file_writer(tensorboard_dir)}
    return model_params, train_params


if __name__ == "__main__":
    # Data
    tr_data, vd_data, map_dict = load_data(transactions_dir, n_heldout_user=5000)
    n_users, n_items = tr_data.shape
    vd_features, vd_labels = vd_data

    tr_datasets = to_tf_train_dataset(tr_data,  2500, 1)
    vd_datasets = to_tf_eval_dataset(vd_features, vd_labels, 1000)

    print(f'n_users_tr: {n_users} / n_items: {n_items} / batch: {n_users / 2500}')
    print(f'n_users_vd: {vd_features.shape[0]} / batch: {vd_features.shape[0] / 1000}')

    # Model
    model_params, train_params = generate_params(n_items)
    model = VAE(**model_params)
    optimizer = tf.keras.optimizers.Adam(0.001, epsilon=1e-08)
    train_epoch(tr_datasets, vd_datasets, model, optimizer, **train_params)
