import tensorflow as tf
import tensorflow.keras as tfk


class BPR(tf.keras.Model):
    def __init__(self, user_num, item_num, embed_dim, use_l2norm=False, l2_reg=0., seed=None):
        super(BPR, self).__init__()
        # user embedding
        self.user_embedding = tfk.layers.Embedding(
            input_dim=user_num,
            input_length=1,
            output_dim=embed_dim,
            embeddings_initializer=tfk.initializers.RandomNormal(seed=seed),
            embeddings_regularizer=tfk.regularizers.l2(l2_reg))
        # item embedding
        self.item_embedding = tfk.layers.Embedding(
            input_dim=item_num,
            input_length=1,
            output_dim=embed_dim,
            embeddings_initializer=tfk.initializers.RandomNormal(seed=seed),
            embeddings_regularizer=tfk.regularizers.l2(l2_reg))

        self.item_num = item_num
        self.use_l2norm = use_l2norm
        tf.random.set_seed(seed)

    def bpr_loss(self, pos_scores, neg_scores):
        loss = tf.reduce_mean(-tf.math.log_sigmoid(pos_scores - neg_scores))
        return loss

    @tf.function
    def call(self, inputs, training=False):
        if training:
            # user info
            user_embed = self.user_embedding(tf.reshape(inputs['user'], [-1, ]))  # (None, embed_dim)

            # item info
            pos_info = self.item_embedding(tf.reshape(inputs['pos_item'], [-1, ]))  # (None, embed_dim)
            neg_info = self.item_embedding(inputs['neg_item'])  # (None, neg_num, embed_dim)

            # norm
            if self.use_l2norm:
                pos_info = tf.math.l2_normalize(pos_info, axis=-1)
                neg_info = tf.math.l2_normalize(neg_info, axis=-1)
                user_embed = tf.math.l2_normalize(user_embed, axis=-1)

            # calculate positive item scores and negative item scores
            pos_scores = tf.reduce_sum(tf.multiply(user_embed, pos_info), axis=-1, keepdims=True)  # (None, 1)
            neg_scores = tf.reduce_sum(tf.multiply(tf.expand_dims(user_embed, axis=1), neg_info), axis=-1)  # (None, neg_num)

            # add loss
            logits = pos_scores
            loss = self.bpr_loss(tf.expand_dims(pos_scores, axis=1), neg_scores)
            # self.add_loss(loss)
        else:
            user_embed = self.user_embedding(tf.reshape(inputs, [-1, ]))  # (None, embed_dim)
            item_embed = self.item_embedding(tf.range(self.item_num))
            logits = tf.matmul(user_embed, item_embed, transpose_b=True)
            loss = None
        return {'logits': logits, 'loss': loss}

    def get_user_vector(self, inputs):
        if len(inputs) < 2 and inputs.get('user') is not None:
            return self.user_embedding(inputs['user'])

    def summary(self):
        inputs = {
            'user': tfk.layers.Input(shape=(), dtype=tf.int32),
            'pos_item': tfk.layyers.Input(shape=(), dtype=tf.int32),
            'neg_item': tfk.layyers.Input(shape=(1,), dtype=tf.int32) }
        tf.keras.Model(inputs=inputs, outputs=self.call(inputs)).summary()
