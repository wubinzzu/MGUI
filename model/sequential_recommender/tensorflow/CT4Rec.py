# @Author ：SuXun
# @Time   ：2023/11/20 19:08
# @File     ：CT4Rec.py
from __future__ import print_function
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
import time
import numpy as np
from model.base import AbstractRecommender
from data import TimeOrderPairwiseSampler
from util.tensorflow import get_initializer, get_session
from data.data_iterator import DataIterator
from reckit import pad_sequences
seed = 2023
np.random.seed(seed)

def get_shape_list(tensor):
    """Returns a list of the shape of tensor, preferring static dimensions.

    Args:
        tensor: A tf.Tensor object to find the shape of.

    Returns:
        A list of dimensions of the shape of tensor. All static dimensions will
        be returned as python integers, and dynamic dimensions will be returned
        as tf.Tensor scalars.
    """

    shape = tensor.shape.as_list()
    non_static_indexes = []
    for (index, dim) in enumerate(shape):
        if dim is None:
            non_static_indexes.append(index)

    if not non_static_indexes:
        return shape

    dyn_shape = tf.shape(tensor)
    for index in non_static_indexes:
        shape[index] = dyn_shape[index]
    return shape


def positional_encoding(dim, sentence_length, dtype=tf.float32):
    encoded_vec = np.array([pos / np.power(10000, 2 * i / dim) for pos in range(sentence_length) for i in range(dim)])
    encoded_vec[::2] = np.sin(encoded_vec[::2])
    encoded_vec[1::2] = np.cos(encoded_vec[1::2])

    return tf.convert_to_tensor(encoded_vec.reshape([sentence_length, dim]), dtype=dtype)


def normalize(inputs,
              epsilon=1e-8,
              scope="ln",
              reuse=None):
    '''Applies layer normalization.

    Args:
      inputs: A tensor with 2 or more dimensions, where the first dimension has
        `batch_size`.
      epsilon: A floating number. A very small number for preventing ZeroDivision Error.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns:
      A tensor with the same shape and data dtype as `inputs`.
    '''
    with tf.variable_scope(scope, reuse=reuse):
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]

        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        beta = tf.get_variable('beta', initializer=tf.zeros(params_shape))
        gamma = tf.get_variable('gamma', initializer=tf.ones(params_shape))
        normalized = (inputs - mean) / ((variance + epsilon) ** (.5))
        outputs = gamma * normalized + beta

    return outputs


def layer_norm(input_tensor, name=None):
    """Run layer normalization on the last dimension of the tensor."""
    return tf.contrib.layers.layer_norm(
        inputs=input_tensor, begin_norm_axis=-1, begin_params_axis=-1, scope=name,
        variables_collections=["layer_norm", tf.GraphKeys.GLOBAL_VARIABLES])


def embedding(inputs,
              vocab_size,
              num_units,
              zero_pad=True,
              scale=False,
              l2_reg=0.0,
              scope="embedding",
              with_t=False,
              reuse=None):
    '''Embeds a given tensor.

    Args:
      inputs: A `Tensor` with type `int32` or `int64` containing the ids
         to be looked up in `lookup table`.
      vocab_size: An int. Vocabulary size.
      num_units: An int. Number of embedding hidden units.
      zero_pad: A boolean. If True, all the values of the fist row (id 0)
        should be constant zeros.
      scale: A boolean. If True. the outputs is multiplied by sqrt num_units.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns:
      A `Tensor` with one more rank than inputs's. The last dimensionality
        should be `num_units`.

    For example,

    ```
    import tensorflow as tf

    inputs = tf.to_int32(tf.reshape(tf.range(2*3), (2, 3)))
    outputs = embedding(inputs, 6, 2, zero_pad=True)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print sess.run(outputs)
    >>
    [[[ 0.          0.        ]
      [ 0.09754146  0.67385566]
      [ 0.37864095 -0.35689294]]

     [[-1.01329422 -1.09939694]
      [ 0.7521342   0.38203377]
      [-0.04973143 -0.06210355]]]
    ```

    ```
    import tensorflow as tf

    inputs = tf.to_int32(tf.reshape(tf.range(2*3), (2, 3)))
    outputs = embedding(inputs, 6, 2, zero_pad=False)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print sess.run(outputs)
    >>
    [[[-0.19172323 -0.39159766]
      [-0.43212751 -0.66207761]
      [ 1.03452027 -0.26704335]]

     [[-0.11634696 -0.35983452]
      [ 0.50208133  0.53509563]
      [ 1.22204471 -0.96587461]]]
    ```
    '''
    with tf.variable_scope(scope, reuse=reuse):
        lookup_table = tf.get_variable('lookup_table',
                                       dtype=tf.float32,
                                       shape=[vocab_size, num_units],
                                       initializer=tf.truncated_normal_initializer(stddev=0.02),
                                       regularizer=tf.keras.regularizers.l2(l2_reg))
        if zero_pad:
            lookup_table = tf.concat((tf.zeros(shape=[1, num_units]),
                                      lookup_table[1:, :]), 0)
        outputs = tf.nn.embedding_lookup(lookup_table, inputs)

        if scale:
            outputs = outputs * (num_units ** 0.5)
    if with_t:
        return outputs, lookup_table
    else:
        return outputs


def multihead_attention(queries,
                        keys,
                        num_units=None,
                        num_heads=8,
                        dropout_rate=0,
                        is_training=True,
                        causality=False,
                        scope="multihead_attention",
                        reuse=None,
                        with_qk=False):
    '''Applies multihead attention.

    Args:
      queries: A 3d tensor with shape of [N, T_q, C_q].
      keys: A 3d tensor with shape of [N, T_k, C_k].
      num_units: A scalar. Attention size.
      dropout_rate: A floating point number.
      is_training: Boolean. Controller of mechanism for dropout.
      causality: Boolean. If true, units that reference the future are masked.
      num_heads: An int. Number of heads.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns
      A 3d tensor with shape of (N, T_q, C)
    '''
    with tf.variable_scope(scope, reuse=reuse):
        # Set the fall back option for num_units
        if num_units is None:
            num_units = queries.get_shape().as_list[-1]

        # Linear projections
        # Q = tf.layers.dense(queries, num_units, activation=tf.nn.relu) # (N, T_q, C)
        # K = tf.layers.dense(keys, num_units, activation=tf.nn.relu) # (N, T_k, C)
        # V = tf.layers.dense(keys, num_units, activation=tf.nn.relu) # (N, T_k, C)
        Q = tf.layers.dense(queries, num_units, activation=None)  # (N, T_q, C)
        K = tf.layers.dense(keys, num_units, activation=None)  # (N, T_k, C)
        V = tf.layers.dense(keys, num_units, activation=None)  # (N, T_k, C)

        # Split and concat
        Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)  # (h*N, T_q, C/h)
        K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)
        V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)

        # Multiplication
        outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))  # (h*N, T_q, T_k)

        # Scale
        outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)

        # Key Masking
        key_masks = tf.sign(tf.reduce_sum(tf.abs(keys), axis=-1))  # (N, T_k)
        key_masks = tf.tile(key_masks, [num_heads, 1])  # (h*N, T_k)
        key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1])  # (h*N, T_q, T_k)

        paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
        outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs)  # (h*N, T_q, T_k)

        # Causality = Future blinding
        if causality:
            diag_vals = tf.ones_like(outputs[0, :, :])  # (T_q, T_k)
            tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()  # (T_q, T_k)
            masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(outputs)[0], 1, 1])  # (h*N, T_q, T_k)

            paddings = tf.ones_like(masks) * (-2 ** 32 + 1)
            outputs = tf.where(tf.equal(masks, 0), paddings, outputs)  # (h*N, T_q, T_k)

        # Activation
        outputs = tf.nn.softmax(outputs)  # (h*N, T_q, T_k)

        # Query Masking
        query_masks = tf.sign(tf.reduce_sum(tf.abs(queries), axis=-1))  # (N, T_q)
        query_masks = tf.tile(query_masks, [num_heads, 1])  # (h*N, T_q)
        query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]])  # (h*N, T_q, T_k)
        outputs *= query_masks  # broadcasting. (N, T_q, C)

        # Dropouts
        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))

        # Weighted sum
        outputs = tf.matmul(outputs, V_)  # ( h*N, T_q, C/h)

        # Restore shape
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)  # (N, T_q, C)

        # Residual connection
        outputs += queries

        # Normalize
        # outputs = normalize(outputs) # (N, T_q, C)

    if with_qk:
        return Q, K
    else:
        return outputs


def feedforward(inputs,
                num_units=[2048, 512],
                scope="multihead_attention",
                dropout_rate=0.2,
                is_training=True,
                reuse=None):
    '''Point-wise feed forward net.

    Args:
      inputs: A 3d tensor with shape of [N, T, C].
      num_units: A list of two integers.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns:
      A 3d tensor with the same shape and dtype as inputs
    '''
    with tf.variable_scope(scope, reuse=reuse):
        # Inner layer
        params = {"inputs": inputs, "filters": num_units[0], "kernel_size": 1,
                  "activation": tf.nn.relu, "use_bias": True}
        outputs = tf.layers.conv1d(**params)
        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))
        # Readout layer
        params = {"inputs": outputs, "filters": num_units[1], "kernel_size": 1,
                  "activation": None, "use_bias": True}
        outputs = tf.layers.conv1d(**params)
        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))

        # Residual connection
        outputs += inputs

        # Normalize
        # outputs = normalize(outputs)

    return outputs


def feedforward2(inputs,
                 num_units,
                 scope="feedforward2",
                 dropout_rate=0.2,
                 reuse=None,
                 is_training=True):
    with tf.variable_scope(scope, reuse=reuse):
        intermediate_output = tf.layers.dense(
            inputs,
            num_units,
            activation=tf.nn.relu,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))

        # Down-project back to `hidden_size` then add the residual.
        layer_output = tf.layers.dense(
            intermediate_output,
            num_units,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))
        layer_output = tf.layers.dropout(layer_output, rate=dropout_rate, training=tf.convert_to_tensor(is_training))

    return layer_output


class CT4Rec(AbstractRecommender):
    def __init__(self, config):
        super(CT4Rec, self).__init__(config)
        self.lr = config["lr"]
        self.batch_size = config["batch_size"]
        self.epochs = config["epochs"]
        self.maxlen = config["maxlen"]
        self.neg_sample_n = config["neg_sample_n"]
        self.hidden_units = config["hidden_units"]
        self.l2_emb = config["l2_emb"]
        self.reuse = False
        self.dropout_rate = config["dropout_rate"]
        self.num_blocks = config["num_blocks"]
        self.num_heads = config["num_heads"]
        # self.neg_test = config["neg_test"]
        self.rd_alpha = config["rd_alpha"]
        self.con_alpha = config["con_alpha"]
        self.user_reg_type = config["user_reg_type"]
        self.temperature = config["temperature"]
        self.rd_reduce = config["rd_reduce"]
        self.num_users, self.num_items = self.dataset.num_users, self.dataset.num_items
        self.user_pos_train = self.dataset.train_data.to_user_dict(by_time=True)
        self.sess = get_session(config["gpu_mem"])
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.is_training = tf.placeholder(tf.bool, shape=())
        self.u = tf.placeholder(tf.int32, shape=(None))
        self.input_seq = tf.placeholder(tf.int32, shape=(None, self.maxlen))
        self.pos = tf.placeholder(tf.int32, shape=(None, self.maxlen))
        self.neg = tf.placeholder(tf.int32, shape=(None, self.maxlen, self.neg_sample_n))
        pos = self.pos
        neg = self.neg
        mask_bool = tf.not_equal(self.input_seq, 0)
        mask = tf.expand_dims(tf.to_float(mask_bool), -1)
        batch_size = tf.shape(self.input_seq)[0]

        with tf.variable_scope("SASRec", reuse=tf.AUTO_REUSE):
            # sequence embedding, item embedding table
            seq_input, item_emb_table = embedding(self.input_seq,
                                                 vocab_size=self.num_items + 1,
                                                 num_units=self.hidden_units,
                                                 zero_pad=True,
                                                 scale=False,
                                                 l2_reg=self.l2_emb,
                                                 scope="input_embeddings",
                                                 with_t=True,
                                                 reuse=self.reuse
                                                 )

            # Positional Encoding
            t, pos_emb_table = embedding(
                tf.tile(tf.expand_dims(tf.range(tf.shape(self.input_seq)[1]), 0), [batch_size, 1]),
                vocab_size=self.maxlen,
                num_units=self.hidden_units,
                zero_pad=False,
                scale=False,
                l2_reg=self.l2_emb,
                scope="dec_pos",
                reuse=self.reuse,
                with_t=True
            )
            seq_input  = seq_input + t

            seq_encoder = self.user_encoder(input_seq = seq_input,
                                         dropout_rate = self.dropout_rate,
                                         mask = mask,
                                         num_blocks = self.num_blocks,
                                         hidden_units = self.hidden_units,
                                         num_heads = self.num_heads)

            seq_encoder_second = self.user_encoder(input_seq = seq_input,
                                         dropout_rate = self.dropout_rate,
                                         mask = mask,
                                         num_blocks = self.num_blocks,
                                         hidden_units = self.hidden_units,
                                         num_heads = self.num_heads)

            self.seq = seq_encoder
            self.seq_second = seq_encoder_second
            self.seq_l2 = tf.nn.l2_normalize(seq_encoder, axis=-1)
            self.seq_second_l2 = tf.nn.l2_normalize(seq_encoder_second, axis=-1)

        pos = tf.reshape(pos, [batch_size * self.maxlen])
        neg = tf.reshape(neg, [batch_size * self.maxlen, self.neg_sample_n])
        pos_emb = tf.nn.embedding_lookup(item_emb_table, pos)
        neg_emb = tf.nn.embedding_lookup(item_emb_table, neg)
        seq_emb = tf.reshape(self.seq, [batch_size * self.maxlen, self.hidden_units])
        seq_emb_second = tf.reshape(self.seq_second, [batch_size * self.maxlen, self.hidden_units])
        seq_emb_l2 = tf.reshape(self.seq_l2, [batch_size * self.maxlen, self.hidden_units])
        seq_emb_second_l2 = tf.reshape(self.seq_second_l2, [batch_size * self.maxlen, self.hidden_units])

        # test -----------------------------------------------------------
        # self.test_item = tf.placeholder(tf.int32, shape=(None, self.neg_test + 1))
        # test_item_emb = tf.nn.embedding_lookup(item_emb_table, self.test_item)
        test_seq_emb = tf.expand_dims(self.seq[:, -1, :], 1)
        self.test_logits = tf.matmul(test_seq_emb, item_emb_table, transpose_b=True)
        self.test_logits = tf.reshape(self.test_logits, [batch_size, tf.shape(item_emb_table)[0]])
        # =====================================loss==================================================
        # ignore padding items (0)
        istarget = tf.reshape(tf.to_float(tf.not_equal(pos, 0)), [batch_size * self.maxlen])

        with tf.variable_scope("basic_loss"):
            # prediction layer
            self.pos_logits = tf.reduce_sum(pos_emb * seq_emb, -1, keepdims=True)
            self.neg_logits = tf.reshape(tf.matmul(neg_emb, tf.expand_dims(seq_emb, -1)), [batch_size * self.maxlen, self.neg_sample_n])
            logits_first = tf.concat([self.pos_logits, self.neg_logits], axis=-1)
            logits_first = tf.clip_by_value(logits_first, clip_value_min=-80, clip_value_max=80)
            prob_first = tf.nn.softmax(logits_first)
            self.loss_first = self.get_softmax_loss(prob=prob_first, is_target= istarget)

        with tf.variable_scope("basic_loss2"):
            self.pos_logits_second = tf.reduce_sum(pos_emb * seq_emb_second, -1, keepdims=True)
            self.neg_logits_second = tf.reshape(tf.matmul(neg_emb, tf.expand_dims(seq_emb_second, -1)), [batch_size * self.maxlen, self.neg_sample_n])
            logits_second = tf.concat([self.pos_logits_second, self.neg_logits_second], axis=-1)
            logits_second = tf.clip_by_value(logits_second, clip_value_min=-80, clip_value_max=80)
            prob_second = tf.nn.softmax(logits_second)
            self.loss_second = self.get_softmax_loss(prob=prob_second, is_target=istarget)

        if self.rd_alpha != 0.0:
            self.loss = (self.loss_first + self.loss_second) / 2.0
        else:
            self.loss = self.loss_first

        # seq padding bool
        seq_bool = tf.reshape(mask_bool, [-1])
        seq_emb_l2_not_pad = tf.boolean_mask(seq_emb_l2, seq_bool)
        seq_emb_second_l2_not_pad = tf.boolean_mask(seq_emb_second_l2, seq_bool)
        with tf.variable_scope("ur_loss"):
            if self.con_alpha <= 0:
                self.ur_loss = 0.0
            elif self.user_reg_type == 'cosine':
                cosine_sim = (tf.reduce_sum(seq_emb_l2_not_pad * seq_emb_second_l2_not_pad, axis=-1) + 1) / 2.0
                cosine_sim = tf.clip_by_value(cosine_sim, 0, 2.0 - 1e-10)
                self.ur_loss = - tf.reduce_mean(tf.log(cosine_sim + 1e-10))
            elif self.user_reg_type == 'l2':
                self.ur_loss = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(seq_emb_l2_not_pad - seq_emb_second_l2_not_pad), axis=-1)))
            elif self.user_reg_type == 'kl':
                tmp_size, _ = get_shape_list(seq_emb_l2_not_pad)
                user_inner_match1 = tf.matmul(seq_emb_l2_not_pad, seq_emb_l2_not_pad, transpose_b=True) * 5 - 10000 * tf.eye(tmp_size)
                user_inner_match2 = tf.matmul(seq_emb_second_l2_not_pad, seq_emb_second_l2_not_pad, transpose_b=True) * 5 - 10000 * tf.eye(tmp_size)
                user_inner_match_prob1 = tf.nn.softmax(user_inner_match1)
                user_inner_match_prob2 = tf.nn.softmax(user_inner_match2)
                user_inner_match_prob1 = tf.clip_by_value(user_inner_match_prob1, 1e-10, 1e10)
                user_inner_match_prob2 = tf.clip_by_value(user_inner_match_prob2, 1e-10, 1e10)
                self.ur_loss = self.get_r_dropout_loss(user_inner_match_prob1, user_inner_match_prob2, 'mean')
            elif self.user_reg_type == 'cl':
                seq_emb_union = tf.concat([seq_emb_l2_not_pad, seq_emb_second_l2_not_pad], axis=0)
                con_mask = tf.eye(tf.shape(seq_emb_union)[0])
                con_sim = tf.matmul(seq_emb_union, seq_emb_union, transpose_b=True)
                self.ur_loss, _ = self.weight_info_nce(sim=con_sim, temperature=self.temperature, mask=con_mask)
            else:
                self.ur_loss = 0.0

        self.loss += self.ur_loss * self.con_alpha

        # r-dropout loss----------------------------------------------
        with tf.variable_scope("rd_loss"):
            if self.rd_alpha > 0:
                self.rd_loss = self.get_r_dropout_loss(prob1=prob_first, prob2=prob_second, reduce=self.rd_reduce, w = istarget)
            else:
                self.rd_loss = 0.0
        self.loss += self.rd_loss * self.rd_alpha

        self.update_opt = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
        self.sess.run(tf.global_variables_initializer())

    def user_encoder(self, input_seq, dropout_rate, mask, num_blocks, hidden_units, num_heads, reuse=None):
        with tf.variable_scope("user_encoder", reuse=reuse):
            seq = tf.layers.dropout(input_seq, rate=dropout_rate, training=tf.convert_to_tensor(self.is_training))
            seq *= mask
            seq = normalize(seq, scope='input_ln')
            # Build blocks
            for i in range(num_blocks):
                with tf.variable_scope("num_blocks_%d" % i, reuse=reuse):
                    # Self-attention
                    seq = multihead_attention(queries=seq,
                                                   keys=seq,
                                                   num_units=hidden_units,
                                                   num_heads=num_heads,
                                                   dropout_rate=dropout_rate,
                                                   is_training=self.is_training,
                                                   causality=True,
                                                   scope="self_attention")
                    seq = normalize(seq, scope='attention_out_ln')
                    # Feed forward
                    seq = feedforward(seq, num_units=[hidden_units, hidden_units],
                                           dropout_rate=dropout_rate, is_training = self.is_training)
                    seq *= mask
                    seq = normalize(seq, scope='feedforward_out_ln')
            return seq

    def weight_info_nce(self, sim, temperature=1.0, weight=None, mask=None, name='weight_info_nce'):
        with tf.variable_scope(name_or_scope=name):
            batch_size, col_size = get_shape_list(sim)
            tn = batch_size // 2
            sim_t = sim / temperature
            idx =tf.range(tn)
            idx = tf.reshape(tf.concat([idx + tn, idx], axis=-1), [-1, 1])
            if mask is not None:
                sim_t += -100000 * mask
            prob_sim = tf.nn.softmax(sim_t, axis=-1)
            diag_part_sim = tf.batch_gather(prob_sim, idx)
            if weight is None:
                loss = tf.reduce_mean(-tf.log(diag_part_sim))
            else:
                loss = tf.reduce_sum(-tf.log(diag_part_sim) * weight) / tf.reduce_sum(weight)
            return loss, prob_sim

    def get_softmax_loss(self, prob, is_target):
        prob_t = tf.reshape(prob[:, :1], [-1])
        return tf.reduce_sum( - tf.log(prob_t + 1e-10) * is_target) / (tf.reduce_sum(is_target) + 1e-10)

    def kl_divergence(self, p1, p2):
        return tf.reduce_sum(p1 * (tf.log(p1 + 1e-10) - tf.log(p2 + 1e-10)), axis=-1)

    def get_r_dropout_loss(self, prob1, prob2, reduce='', w=None):
        if w is None:
            w = tf.ones_like(prob1)
        if reduce == 'sum':
            kl_loss = tf.reduce_sum(self.kl_divergence(prob1, prob2) * w)
            kl_loss += tf.reduce_sum(self.kl_divergence(prob2, prob1) * w)
        else:
            kl_loss = tf.reduce_sum(self.kl_divergence(prob1, prob2) * w) / tf.reduce_sum(w)
            kl_loss += tf.reduce_sum(self.kl_divergence(prob2, prob1) * w) / tf.reduce_sum(w)
        return kl_loss / 2.0

    def train_model(self):
        users_list, item_seq_list, item_pos_list, item_neg_list = self.get_traindata()
        self.logger.info(self.evaluator.metrics_info())
        for epoch in range(self.epochs):
            now_training = time.time()
            data_iter = DataIterator(users_list, item_seq_list, item_pos_list, item_neg_list,
                                batch_size=self.batch_size, shuffle=True)
            for bat_users, bat_item_seqs, bat_pos_items, bat_neg_items in data_iter:
                feed = {self.u: bat_users,
                        self.input_seq: bat_item_seqs,
                        self.pos: bat_pos_items,
                        self.neg: bat_neg_items,
                        self.is_training: True}

                self.sess.run(self.update_opt, feed_dict=feed)

            # training_t = time.time() - now_training
            # print("training time：", training_t)
            # now_testing = time.time()
            result = self.evaluate_model()
            # testing_t = time.time() - now_testing
            # print("testing time：", testing_t)
            self.logger.info("epoch %d:\t%s" % (epoch, result))

    def get_traindata(self):
        def random_neq(l, r, s):
            t = np.random.randint(l, r)
            while t in s:
                t = np.random.randint(l, r)
            return t

        userid_set = np.unique(list(self.user_pos_train.keys()))
        users_list, item_seq_list, item_pos_list, item_neg_list = [], [], [], []
        for user_id in userid_set:
            users_list.append(user_id)
            seq = np.zeros([self.maxlen], dtype=np.int32)
            pos = np.zeros([self.maxlen], dtype=np.int32)
            neg = np.zeros([self.maxlen, self.neg_sample_n], dtype=np.int32)
            nxt = self.user_pos_train[user_id][-1]
            idx = self.maxlen - 1
            ts = set(self.user_pos_train[user_id])
            for i in reversed(self.user_pos_train[user_id][:-1]):
                seq[idx] = i
                pos[idx] = nxt
                if nxt != 0:
                    for ni in range(self.neg_sample_n):
                        neg[idx, ni] = random_neq(1, self.num_items + 1, ts)
                nxt = i
                idx -= 1
                if idx == -1: break
            item_seq_list.append(seq)
            item_pos_list.append(pos)
            item_neg_list.append(neg)
        return users_list, item_seq_list, item_pos_list, item_neg_list

    def evaluate_model(self):
        return self.evaluator.evaluate(self)

    def predict(self, users):
        item_seqs = [self.user_pos_train[user][-self.maxlen:] if user in self.user_pos_train else [self.num_items]
                     for user in range(self.num_users)]
        self.test_item_seqs = pad_sequences(item_seqs, value=self.num_items, max_len=self.maxlen,
                                            padding='pre', truncating='pre', dtype=np.int32)
        last_items = [self.test_item_seqs[u] for u in users]
        feed = {self.u: users, self.input_seq: last_items, self.is_training: False}
        bat_ratings = self.sess.run(self.test_logits,feed_dict=feed)

        return bat_ratings