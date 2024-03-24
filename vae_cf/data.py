import scipy as sp
import numpy as np
import pandas as pd
import scipy.sparse as sp


def _load_raw_data(directory):
    data = pd.read_csv(directory)
    id_map = dict()
    for i in ['customer_id', 'article_id']:
        cat = data[i].astype('category').cat
        data[i] = cat.codes
        id_map[i] = cat.categories
    return data, id_map


def _to_sparse_matrix(data, n_items):
    n_users = data['customer_id'].max() + 1
    score = np.ones(data.shape[0])
    interactions = sp.csr_matrix(
        (score, (data['customer_id'], data['article_id'])),
        dtype=np.double,
        shape=(n_users, n_items))
    return interactions


def _split_train_valid(n_user, n_heldout_user=10000):
    np.random.seed(98765)
    random_user_index = np.random.permutation(n_user)

    tr_users = random_user_index[:(n_user - n_heldout_user)]
    vd_users = random_user_index[(n_user - n_heldout_user):]
    return (tr_users, vd_users), random_user_index


def _split_feature_label(data, test_prop=0.2):
    data_grouped_by_user = data.groupby('customer_id')
    feature_list, label_list = list(), list()

    np.random.seed(98765)
    for _, group in data_grouped_by_user:
        n_clicked_items = len(group)
        n_test_items = int(test_prop * n_clicked_items)

        if n_clicked_items >= 5:
            idx = np.zeros(n_clicked_items, dtype='bool')
            idx[np.random.choice(n_clicked_items, size=n_test_items, replace=False)] = True

            feature_list.append(group[np.logical_not(idx)])
            label_list.append(group[idx])
        else:
            feature_list.append(group)

    data_feature = pd.concat(feature_list)
    data_label = pd.concat(label_list)
    return data_feature, data_label


class SparseMatrixEncoder:
    def __init__(self, n_items, n_users, user_index):
        self.n_items = n_items
        self.n_users = n_users
        self.user_id_map = pd.DataFrame(
            {'customer_id': user_index, 'new_customer_id': range(n_users)})

    def _sample_from_uid(self, data, user_index):
        df = data.loc[data['customer_id'].isin(user_index)]
        # convert uid
        df = df.merge(self.user_id_map, on='customer_id')
        df.drop(columns=['customer_id'], inplace=True)

        # new index
        df['new_customer_id'] = df['new_customer_id'] - df['new_customer_id'].min()
        df.rename(columns = {'new_customer_id': 'customer_id'}, inplace=True)
        return df

    def transform(self, data, user_index, method='train'):
        samples = self._sample_from_uid(data, user_index)
        if method == 'train':
            return _to_sparse_matrix(samples, self.n_items)
        else:
            tr_samples, te_samples = _split_feature_label(samples)
            tr_interactions = _to_sparse_matrix(tr_samples, self.n_items)
            te_interactions = _to_sparse_matrix(te_samples, self.n_items)
            return tr_interactions, te_interactions


def load_data(directory, n_heldout_user=1000):
    data, id_map = _load_raw_data(directory)
    n_users = len(id_map['customer_id'])
    n_items = len(id_map['article_id'])

    split_user_index, user_index = _split_train_valid(n_users, n_heldout_user=n_heldout_user)
    train_user_index, valid_user_index = split_user_index

    sparse_encoder = SparseMatrixEncoder(n_items, n_users, user_index)
    tr_data = sparse_encoder.transform(data, train_user_index, 'train')
    vd_data_feature, vd_data_labels = sparse_encoder.transform(data, valid_user_index, 'valid')

    user_dict = dict(zip(id_map['customer_id'][user_index], range(n_users)))
    item_dict = dict(zip(range(n_items), id_map['article_id']))
    return tr_data, (vd_data_feature, vd_data_labels), (user_dict, item_dict)