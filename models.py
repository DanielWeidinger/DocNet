from __future__ import absolute_import, division, print_function, unicode_literals

import os
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

import tensorflow as tf
import tensorflow.keras.backend as K


class FFN(tf.keras.layers.Layer):
    def __init__(
            self,
            hidden_size=768,
            dropout=0.1,
            residual=True,
            name='FFN',
            **kwargs):
        """Simple Dense wrapped with various layers
        """

        super(FFN, self).__init__(name=name, **kwargs)
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.residual = residual
        self.ffn_layer = tf.keras.layers.Dense(
            units=hidden_size,
            use_bias=True
        )

    def call(self, inputs):
        ffn_embedding = self.ffn_layer(inputs)
        ffn_embedding = tf.keras.layers.ReLU()(ffn_embedding)
        if self.dropout > 0:
            ffn_embedding = tf.keras.layers.Dropout(
                self.dropout)(ffn_embedding)

        if self.residual:
            ffn_embedding += inputs
        return ffn_embedding


def stack_two_tensor(tensor_a, tensor_b):
    if tensor_a is not None and tensor_b is not None:
        return tf.stack([tensor_a, tensor_b], axis=1)
    elif tensor_a is not None:
        return tf.expand_dims(tensor_a, axis=1)
    elif tensor_b is not None:
        return tf.expand_dims(tensor_b, axis=1)


class MedicalQAModel(tf.keras.Model):
    def __init__(self, name=''):
        super(MedicalQAModel, self).__init__(name=name)
        self.q_ffn = FFN(name='QFFN', input_shape=(768,))
        self.a_ffn = FFN(name='AFFN', input_shape=(768,))

    def call(self, inputs):
        q_bert_embedding, a_bert_embedding = tf.unstack(inputs, axis=1)
        q_embedding, a_embedding = self.q_ffn(
            q_bert_embedding), self.a_ffn(a_bert_embedding)
        return tf.stack([q_embedding, a_embedding], axis=1)


class BioBert(tf.keras.Model):
    def __init__(self, name=''):
        super(BioBert, self).__init__(name=name)

    def call(self, inputs):

        # inputs is dict with input features
        input_ids, input_masks, segment_ids = inputs
        # pass to bert
        # with shape of (batch_size/2*batch_size, max_seq_len, hidden_size)
        # TODO(Alex): Add true bert model
        # Input: input_ids, input_masks, segment_ids all with shape (None, max_seq_len)
        # Output: a tensor with shape (None, max_seq_len, hidden_size)
        fake_bert_output = tf.expand_dims(tf.ones_like(
            input_ids, dtype=tf.float32), axis=-1)*tf.ones([1, 1, 768], dtype=tf.float32)
        max_seq_length = tf.shape(fake_bert_output)[-2]
        hidden_size = tf.shape(fake_bert_output)[-1]

        bert_output = tf.reshape(
            fake_bert_output, (-1, 2, max_seq_length, hidden_size))
        return bert_output


class MedicalQAModelwithBert(tf.keras.Model):
    def __init__(
            self,
            hidden_size=768,
            dropout=0.1,
            residual=True,
            activation=tf.keras.layers.ReLU(),
            name=''):
        super(MedicalQAModelwithBert, self).__init__(name=name)
        self.biobert = BioBert()
        self.qa_ffn_layer = QAFFN(
            hidden_size=hidden_size,
            dropout=dropout,
            residual=residual,
            activation=activation)

    def _avg_across_token(self, tensor):
        if tensor is not None:
            tensor = tf.reduce_mean(tensor, axis=1)
        return tensor

    def call(self, inputs):

        q_bert_embedding, a_bert_embedding = self.biobert(inputs)

        # according to USE, the DAN network average embedding across tokens
        q_bert_embedding = self._avg_across_token(q_bert_embedding)
        a_bert_embedding = self._avg_across_token(a_bert_embedding)

        q_embedding, a_embedding = self.qa_ffn_layer(
            (q_bert_embedding, a_bert_embedding))

        return stack_two_tensor(q_embedding, a_embedding)
