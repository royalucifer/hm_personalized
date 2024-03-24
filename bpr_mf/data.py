import pandas as pd


def load_entity_map(path, index_col):
    data = pd.read_csv(
        path,
        index_col=index_col,
        dtype={index_col: 'category'})
    return {index: i for i, index in enumerate(data.index.unique())}


def load_interactions(path, user_index, user_map, item_index, item_map):
    data = pd.read_csv(
        path,
        dtype={user_index: 'category', item_index: 'category'},
        parse_dates=['t_dat']). \
        assign(week=lambda x: (x['t_dat'].max() - x['t_dat']).dt.days // 7)

    # filter
    low_freq_user = data. \
        groupby([user_index, 't_dat'], observed=True). \
        agg(item_num=(item_index, 'size')). \
        reset_index(). \
        groupby(user_index, observed=True). \
        agg(
            buy_num=('t_dat', 'count'),
            avg_item_num=('item_num', 'mean')). \
        query('buy_num == 1 and avg_item_num == 1').index

    data = data[~data[user_index].isin(low_freq_user)]. \
        sort_values([user_index, item_index])
    data['user_id'] = data[user_index].map(user_map)
    data['item_id'] = data[item_index].map(item_map)

    # create the mask for test data
    test_mask = (data['week'] == 0).to_numpy()
    return data[['user_id', 'item_id']], test_mask

