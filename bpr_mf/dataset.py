import itertools

import numpy as np
import tensorflow as tf


# 19 min
def to_triplet_train_dataset_user(data, n_items, n_neg_item, batch_size):
    neg_item = []
    for _, item_group in data.groupby('user_id', observed=True):
        pos_item = item_group['item_id'].nunique().tolist()
        n_pos_item = len(pos_item)

        prob = np.ones(n_items)
        prob[pos_item] = 0
        prob /= prob.sum()

        np.random.seed(123)
        neg_item.append(np.random.choice(n_items, (n_pos_item, n_neg_item), p=prob))

    dataset = tf.data.Dataset.from_tensor_slices({
        'user': data['user_id'].to_numpy(),
        'pos_item': data['item_id'].to_numpy(),
        'neg_item': np.concatenate(neg_item) })
    return dataset.batch(batch_size)


def to_triplet_train_dataset_random(data, n_items, n_neg_item, batch_size=1024):
    class NegativeItemSample:
        def __init__(self, n_items, n_neg_item):
            self.n_items = n_items - 1
            self.n_neg_item = n_neg_item
            tf.random.set_seed(123)

        def __call__(self, user, pos_id):
            neg_id = tf.random.uniform(
                shape=[self.n_neg_item],
                maxval=self.n_items,
                dtype=tf.int32)
            return {'user': user, 'pos_item': pos_id, 'neg_item': neg_id}

    dataset = tf.data.Dataset.from_tensor_slices((
        data['user_id'].to_numpy(),
        data['item_id'].to_numpy() )). \
        map(
            NegativeItemSample(n_items, n_neg_item),
            tf.data.experimental.AUTOTUNE). \
        batch(batch_size). \
        prefetch(tf.data.AUTOTUNE)
    return dataset


# def to_triplet_test_dataset(data, n_items, test_mask, batch_size=1024):
#     def _assign_arr_val(row, shape):
#         arr = np.zeros(shape, dtype=bool)
#         arr[row] = 1.0
#         return arr

#     train_data = data[~test_mask]. \
#         groupby('user_id', observed=True). \
#         agg(known=('item_id', list))

#     test_data = data[test_mask]. \
#         groupby('user_id', observed=True). \
#         agg(label=('item_id', list)). \
#         join(train_data, how='left'). \
#         dropna(). \
#         applymap(lambda x: _assign_arr_val(x, n_items)). \
#         reset_index()

#     dataset = tf.data.Dataset.from_tensor_slices({
#         'user': test_data['user_id'].to_numpy(),
#         'known': np.stack(test_data['known'].values),
#         'label': np.stack(test_data['label'].values) })
#     return dataset.batch(batch_size)


def to_triplet_test_dataset(data, test_mask, batch_size=1024):
    def _generator():
        n_users = test_data.shape[0]
        for start in range(0, n_users, batch_size):
            end = min(start + batch_size, n_users)

            user, known, label, label_num = [], [], [], []
            for i, row in enumerate(test_data[start:end].itertuples()):
                user.append(getattr(row, 'user_id'))
                known.extend([(i, val) for val in getattr(row, 'known')])
                label.extend([(i, val) for val in getattr(row, 'label')])
                label_num.append(len(getattr(row, 'label')))
            yield (user, known, label, label_num)

    train_data = data[~test_mask]. \
        groupby('user_id', observed=True). \
        agg(known=('item_id', list))

    test_data = data[test_mask]. \
        groupby('user_id', observed=True). \
        agg(label=('item_id', list)). \
        join(train_data, how='left'). \
        dropna(). \
        reset_index()

    dataset = tf.data.Dataset.from_generator(
        _generator,
        output_types=(tf.int32, tf.int32, tf.int32, tf.float32)). \
        prefetch(tf.data.AUTOTUNE)
    return dataset
