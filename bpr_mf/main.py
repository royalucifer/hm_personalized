import tensorflow as tf
import tensorflow_ranking as tfr

from data import load_entity_map, load_interactions
from dataset import to_triplet_train_dataset_random, to_triplet_test_dataset
from model import BPR
from train import train_epoch

TRANSACTIONS_DIR = "/Users/roysung/py_projects/hm_personal/dataset/transactions_train.csv"
CUSTOMERS_DIR = "/Users/roysung/py_projects/hm_personal/dataset/customers.csv"
ARTICLES_DIR = "/Users/roysung/py_projects/hm_personal/dataset/articles.csv"
TENSORBOARD_DIR = "/Users/roysung/py_projects/hm_personal/model_output/test"
TRAIN_BATCH_SIZE = 8192
TEST_BATCH_SIZE = 2000


def generate_params(n_users, n_items):
    model_params = {
        'user_num': n_users,
        'item_num': n_items,
        'embed_dim': 100,
        'l2_reg': 0.0,
        'seed': 98765}
    train_params = {
        'metrics': {
            'loss': tf.keras.metrics.Mean(name='loss'),
            # 'ndcg': tfr.keras.metrics.NDCGMetric(name='test_ndcg', topn=20)
            'ndcg': tf.keras.metrics.Mean(name='ndcg'),
            'precision': tf.keras.metrics.Mean(name='precision'),
            'recall': tf.keras.metrics.Mean(name='recall')
        },
        'num_epochs': 5,
        'summary_writer': tf.summary.create_file_writer(TENSORBOARD_DIR)}
    return model_params, train_params


if __name__ == '__main__':
    user_map = load_entity_map(CUSTOMERS_DIR, 'customer_id')
    item_map = load_entity_map(ARTICLES_DIR, 'article_id')
    n_users, n_items = len(user_map), len(item_map)
    interactions, test_mask = load_interactions(
        TRANSACTIONS_DIR, 'customer_id', user_map, 'article_id', item_map)

    # Dataset
    train_dataset = to_triplet_train_dataset_random(interactions[~test_mask], n_items, 3, TRAIN_BATCH_SIZE)
    test_dataset = to_triplet_test_dataset(interactions, test_mask, TEST_BATCH_SIZE)
    del interactions, test_mask

    # Model
    model_params, train_params = generate_params(n_users, n_items)
    model = BPR(**model_params)
    optimizer = tf.keras.optimizers.Adam(0.001, epsilon=1e-08)
    train_epoch(train_dataset, test_dataset, model, optimizer, **train_params)
