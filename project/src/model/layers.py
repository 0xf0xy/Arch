import tensorflow as tf
import keras


@keras.saving.register_keras_serializable()
def scaled_dot_product_attention(query, key, value, mask):
    """
    Compute scaled dot-product attention.

    Args:
        query: Query tensor.
        key: Key tensor.
        value: Value tensor.
        mask: Attention mask.

    Returns:
        Attention output tensor.
    """
    score = tf.matmul(query, key, transpose_b=True)

    depth = tf.cast(tf.shape(key)[-1], tf.float32)
    logits = score / tf.math.sqrt(depth)

    if mask is not None:
        logits += mask * -1e9

    attention_weights = tf.nn.softmax(logits, axis=-1)

    return tf.matmul(attention_weights, value)


@keras.saving.register_keras_serializable()
class MultiheadAttention(keras.layers.Layer):
    """Multi-head attention layer."""

    def __init__(self, d_model: int, n_heads: int, **kwargs):
        """
        Initialize the multi-head attention layer.

        Args:
            d_model (int): Dimensionality of the model.
            n_heads (int): Number of attention heads.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(**kwargs)
        self.n_heads = n_heads
        self.d_model = d_model

        assert d_model % self.n_heads == 0

        self.d_heads = d_model // self.n_heads

        self.query_dense = keras.layers.Dense(d_model, name="query")
        self.key_dense = keras.layers.Dense(d_model, name="key")
        self.value_dense = keras.layers.Dense(d_model, name="value")

        self.dense = keras.layers.Dense(d_model)

    def get_config(self):
        """
        Return config for serialization.

        Returns:
            Config dictionary.
        """
        config = super().get_config()
        config.update(
            {
                "d_model": self.d_model,
                "n_heads": self.n_heads,
            }
        )

        return config

    def split_heads(self, inputs, batch_size):
        """
        Split the last dimension into (n_heads, d_heads).

        Args:
            inputs: Input tensor.
            batch_size: Batch size.

        Returns:
            Tensor with split heads.
        """
        inputs = tf.reshape(inputs, shape=(batch_size, -1, self.n_heads, self.d_heads))

        return tf.transpose(inputs, perm=[0, 2, 1, 3])

    def call(self, query, key, value, mask):
        """
        Forward pass of the multi-head attention layer.

        Args:
            query: Query tensor.
            key: Key tensor.
            value: Value tensor.
            mask: Attention mask.

        Returns:
            Output tensor after attention.
        """
        batch_size = tf.shape(query)[0]

        query = self.query_dense(query)
        key = self.key_dense(key)
        value = self.value_dense(value)

        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)

        scaled_attention = scaled_dot_product_attention(query, key, value, mask)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))

        return self.dense(concat_attention)


@keras.saving.register_keras_serializable()
class TokenandPositionEmbedding(keras.layers.Layer):
    """Token and position embedding layer."""

    def __init__(self, seq_len: int, vocab_size: int, d_model: int, **kwargs):
        """
        Initialize the embedding layer.

        Args:
            seq_len (int): Sequence length.
            vocab_size (int): Vocabulary size.
            d_model (int): Model dimensionality.
            **kwargs: Additional arguments.
        """
        super().__init__(**kwargs)
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.d_model = d_model

        self.token_embedding = keras.layers.Embedding(vocab_size, d_model, name="token")
        self.position_embedding = keras.layers.Embedding(
            seq_len, d_model, name="position"
        )

    def get_config(self):
        """
        Return config for serialization.

        Returns:
            Config dictionary.
        """
        config = super().get_config()
        config.update(
            {
                "seq_len": self.seq_len,
                "vocab_size": self.vocab_size,
                "d_model": self.d_model,
            }
        )

        return config

    def call(self, inputs):
        """
        Compute embeddings.

        Args:
            inputs: Input tokens.

        Returns:
            Embedded tokens with positions.
        """
        length = tf.shape(inputs)[-1]
        positions = tf.range(0, length, 1)
        embedded_tokens = self.token_embedding(inputs)
        embedded_positions = self.position_embedding(positions)
        embedded_tokens *= tf.math.sqrt(
            tf.cast(self.token_embedding.output_dim, tf.float32)
        )

        return embedded_tokens + embedded_positions


@keras.saving.register_keras_serializable()
class FeedForward(keras.layers.Layer):
    """Feed-forward network layer."""

    def __init__(self, d_model: int, units: int, drop_rate: float, **kwargs):
        """
        Initialize the feed-forward layer.

        Args:
            d_model (int): Model dimensionality.
            units (int): Number of units.
            drop_rate (float): Dropout rate.
            **kwargs: Additional arguments.
        """
        super().__init__(**kwargs)
        self.d_model = d_model
        self.units = units
        self.drop_rate = drop_rate

        self.dense_1 = keras.layers.Dense(units, activation="relu", name="ffn_dense_1")
        self.dropout = keras.layers.Dropout(drop_rate, name="ffn_dropout")
        self.dense_2 = keras.layers.Dense(d_model, name="ffn_dense_2")

    def get_config(self):
        """
        Return config for serialization.

        Returns:
            Config dictionary.
        """
        config = super().get_config()
        config.update(
            {
                "d_model": self.d_model,
                "units": self.units,
                "drop_rate": self.drop_rate,
            }
        )

        return config

    def call(self, inputs, training=None):
        """
        Forward pass.

        Args:
            inputs: Input tensor.
            training: Whether in training mode.

        Returns:
            Output tensor.
        """
        x = self.dense_1(inputs)
        x = self.dropout(x, training=training)
        x = self.dense_2(x)

        return x


@keras.saving.register_keras_serializable()
class DecoderLayer(keras.layers.Layer):
    """Decoder layer with self-attention and feed-forward."""

    def __init__(
        self, units: int, d_model: int, n_heads: int, drop_rate: float, **kwargs
    ):
        """Initialize the decoder layer.

        Args:
            units (int): Feed-forward units.
            d_model (int): Model dimensionality.
            n_heads (int): Number of attention heads.
            drop_rate (float): Dropout rate.
            **kwargs: Additional arguments.
        """
        super().__init__(**kwargs)
        self.units = units
        self.d_model = d_model
        self.n_heads = n_heads
        self.drop_rate = drop_rate

        self.mha = MultiheadAttention(d_model, n_heads, name="self_mha")
        self.dropout1 = keras.layers.Dropout(drop_rate)
        self.norm1 = keras.layers.LayerNormalization(epsilon=1e-6)

        self.ffn = FeedForward(d_model, units, drop_rate, name="ffn")

        self.dropout2 = keras.layers.Dropout(drop_rate)
        self.norm2 = keras.layers.LayerNormalization(epsilon=1e-6)

    def get_config(self):
        """
        Return config for serialization.

        Returns:
            Config dictionary.
        """
        config = super().get_config()
        config.update(
            {
                "units": self.units,
                "d_model": self.d_model,
                "n_heads": self.n_heads,
                "drop_rate": self.drop_rate,
            }
        )

        return config

    def call(self, inputs, causal_mask, training=None):
        """
        Forward pass.

        Args:
            inputs: Input tensor.
            causal_mask: Causal mask.
            training: Whether in training mode.

        Returns:
            Output tensor.
        """
        attn = self.mha(inputs, inputs, inputs, causal_mask)
        attn = self.dropout1(attn, training=training)
        out1 = self.norm1(inputs + attn)

        ffn = self.ffn(out1)
        out2 = self.dropout2(ffn, training=training)

        return self.norm2(out1 + out2)


@keras.saving.register_keras_serializable()
class Decoder(keras.layers.Layer):
    """Transformer decoder."""

    def __init__(
        self,
        seq_len: int,
        vocab_size: int,
        n_layers: int,
        units: int,
        d_model: int,
        n_heads: int,
        drop_rate: float,
        **kwargs,
    ):
        """
        Initialize the decoder.

        Args:
            seq_len (int): Sequence length.
            vocab_size (int): Vocabulary size.
            n_layers (int): Number of layers.
            units (int): Feed-forward units.
            d_model (int): Model dimensionality.
            n_heads (int): Number of attention heads.
            drop_rate (float): Dropout rate.
            **kwargs: Additional arguments.
        """
        super().__init__(**kwargs)
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.n_layers = n_layers
        self.units = units
        self.d_model = d_model
        self.n_heads = n_heads
        self.drop_rate = drop_rate

        self.embedding = TokenandPositionEmbedding(seq_len, vocab_size, d_model)
        self.dropout = keras.layers.Dropout(drop_rate)

        self.dec_layers = [
            DecoderLayer(units, d_model, n_heads, drop_rate, name=f"dec_layer_{i}")
            for i in range(n_layers)
        ]

    def get_config(self):
        """
        Return config for serialization.

        Returns:
            Config dictionary.
        """
        config = super().get_config()
        config.update(
            {
                "seq_len": self.seq_len,
                "vocab_size": self.vocab_size,
                "n_layers": self.n_layers,
                "units": self.units,
                "d_model": self.d_model,
                "n_heads": self.n_heads,
                "drop_rate": self.drop_rate,
            }
        )

        return config

    def call(self, inputs, causal_mask, training=None):
        """
        Forward pass.

        Args:
            inputs: Input tensor.
            causal_mask: Causal mask.
            training: Whether in training mode.

        Returns:
            Decoded output.
        """
        x = self.embedding(inputs)
        x = self.dropout(x, training=training)

        for layer in self.dec_layers:
            x = layer(
                x,
                causal_mask,
                training=training,
            )

        return x
