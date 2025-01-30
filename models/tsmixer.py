# coding=utf-8
# Copyright 2024 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Implementation of TSMixer."""

# import tensorflow as tf
# from tensorflow.keras import layers

# def res_block(inputs, norm_type, activation, dropout, ff_dim):
#   """Residual block of TSMixer."""

#   norm = (
#       layers.LayerNormalization
#       if norm_type == 'L'
#       else layers.BatchNormalization
#   )

#   # Temporal Linear
#   x = norm(axis=[-2, -1])(inputs)
#   x = tf.transpose(x, perm=[0, 2, 1])  # [Batch, Channel, Input Length]
#   x = layers.Dense(x.shape[-1], activation=activation)(x)
#   x = tf.transpose(x, perm=[0, 2, 1])  # [Batch, Input Length, Channel]
#   x = layers.Dropout(dropout)(x)
#   res = x + inputs

#   # Feature Linear
#   x = norm(axis=[-2, -1])(res)
#   x = layers.Dense(ff_dim, activation=activation)(
#       x
#   )  # [Batch, Input Length, FF_Dim]
#   x = layers.Dropout(dropout)(x)
#   x = layers.Dense(inputs.shape[-1])(x)  # [Batch, Input Length, Channel]
#   x = layers.Dropout(dropout)(x)
#   return x + res

import tensorflow as tf
import keras.layers as layers
import keras

def res_block(inputs, norm_type, activation, dropout, ff_dim):
    """Residual block of TSMixer with correct normalization strategy.

    Args:
        inputs (tf.Tensor): Input tensor of shape [batch_size, seq_len, channels].
        norm_type (str): 'L' for LayerNorm, 'B' for BatchNorm.
        activation (str): Activation function (e.g., 'relu', 'gelu').
        dropout (float): Dropout rate.
        ff_dim (int): Feature expansion dimension.

    Returns:
        tf.Tensor: Output tensor of shape [batch_size, seq_len, channels].
    """

    # Select normalization type
    norm = layers.LayerNormalization if norm_type == 'L' else layers.BatchNormalization

    # 1️⃣ **Temporal Mixing Block** (Normalize over time)
    if norm_type == 'L':
        x = norm(axis=-1)(inputs)  # LayerNorm across features (channels)
    else:
        x = norm(axis=1)(inputs)  # BatchNorm across time (seq_len)

    # Replaced tf.transpose() with a Keras Lambda layer
    x = layers.Lambda(lambda t: tf.transpose(t, perm=[0, 2, 1]))(x)  # Shape: [Batch, Channels, Seq_Len]
    x = layers.Dense(x.shape[-1], activation=activation)(x)  # Temporal Mixing
    x = layers.Lambda(lambda t: tf.transpose(t, perm=[0, 2, 1]))(x)  # Shape: [Batch, Seq_Len, Channels]
    x = layers.Dropout(dropout)(x)

    # Residual connection for temporal mixing
    res = x + inputs

    # 2️⃣ **Feature Mixing Block** (Normalize over features)
    if norm_type == 'L':
        x = norm(axis=-1)(res)  # LayerNorm across features (channels)
    else:
        x = norm(axis=-1)(res)  # BatchNorm across features (channels)

    x = layers.Dense(ff_dim, activation=activation)(x)  # Expand feature dimension
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(inputs.shape[-1])(x)  # Project back to original feature dim
    x = layers.Dropout(dropout)(x)

    # Residual connection for feature mixing
    return x + res

def build_model(
    input_shape,
    pred_len,
    norm_type,
    activation,
    n_block,
    dropout,
    ff_dim,
    target_slice,
):
  """Build TSMixer model."""

  inputs = tf.keras.Input(shape=input_shape)
  x = inputs  # [Batch, Input Length, Channel]
  for _ in range(n_block):
    x = res_block(x, norm_type, activation, dropout, ff_dim)

  if target_slice:
    x = x[:, :, target_slice]

  x = tf.transpose(x, perm=[0, 2, 1])  # [Batch, Channel, Input Length]
  x = layers.Dense(pred_len)(x)  # [Batch, Channel, Output Length]
  outputs = tf.transpose(x, perm=[0, 2, 1])  # [Batch, Output Length, Channel])

  return tf.keras.Model(inputs, outputs)