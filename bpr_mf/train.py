import tensorflow as tf


@tf.function
def train_step(inputs, model, optimizer, metrics):
    with tf.GradientTape() as tape:
        loss = model.call(inputs, training=True)['loss']
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    metrics['loss'].update_state(loss)


# @tf.function
# def test_step(inputs, model, metrics):
#     zero = tf.constant(False, dtype=tf.bool)
#     neg_val = tf.constant(-1000, dtype=tf.float32)

#     logits = model.call(inputs)['logits']
#     condition = tf.not_equal(inputs['known'], zero)
#     predictions = tf.where(condition, neg_val, logits)
#     metrics['ndcg'].update_state(inputs['label'], predictions)


def train_epoch(
        train_datasets,
        eval_datasets,
        model,
        optimizer,
        metrics,
        num_epochs,
        summary_writer):
    step = 0
    for epoch in range(1, num_epochs + 1):
        print(f'Epoch {epoch}')
        for inputs in train_datasets:
            train_step(inputs, model, optimizer, metrics)
            step += 1

        for test_inputs in eval_datasets:
            test_step(test_inputs, model, metrics)

        with summary_writer.as_default():
            tf.summary.scalar('train_loss', metrics['loss'].result(), step=epoch)
            tf.summary.scalar('ndcg_at_k', metrics['ndcg'].result(), step=epoch)
            tf.summary.scalar('precision_at_k', metrics['precision'].result(), step=epoch)
            tf.summary.scalar('recall_at_k', metrics['recall'].result(), step=epoch)

        metrics['loss'].reset_states()
        metrics['ndcg'].reset_states()
        metrics['precision'].reset_states()
        metrics['recall'].reset_states()


@tf.function
def _dcg(target, top_k=20):
    rank = tf.range(1, top_k + 1, dtype=tf.float32)
    denom = tf.math.log(rank + 1.0) / tf.math.log(2.0)
    return tf.reduce_sum(target / denom, axis=-1)


@tf.function
def _calc_metrics(logits, ground_truth, label_num, top_k=20):
    topk_indices = tf.math.top_k(logits, k=top_k).indices
    sorted_truth = tf.gather(ground_truth, topk_indices, axis=1, batch_dims=1)
    ideal_truth = tf.sort(ground_truth, direction='DESCENDING')[:, :top_k]

    ndcg = _dcg(sorted_truth, top_k) / _dcg(ideal_truth, top_k)
    precision = tf.reduce_sum(sorted_truth, axis=-1) / top_k
    recall = tf.reduce_sum(sorted_truth, axis=-1) / label_num
    return ndcg, precision, recall


def test_step(inputs, model, metrics, top_k=20):
    user, known_indices, label_indices, label_num = inputs
    logits = model.call(user)['logits']

    # Exclude training items:
    logits = tf.tensor_scatter_nd_update(logits, known_indices, tf.fill([known_indices.shape[0]], -100.0))
    ground_truth = tf.scatter_nd(label_indices, tf.ones_like(label_indices[:, 0], dtype=tf.float32), logits.shape)

    # Computing ndcg, precision and recall:
    ndcg, precision, recall = _calc_metrics(logits, ground_truth, label_num, top_k)
    metrics['ndcg'].update_state(ndcg)
    metrics['precision'].update_state(precision)
    metrics['recall'].update_state(recall)

    del logits, ground_truth, ndcg, precision, recall
