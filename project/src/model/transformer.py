"""
MIT License

Copyright (c) 2026 0xf0xy

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from .masking import CausalMaskLayer
from .layers import Decoder
import keras


@keras.saving.register_keras_serializable()
class Transformer(keras.Model):
    """Transformer model for text generation tasks."""

    def __init__(
        self,
        seq_len: int,
        vocab_size: int,
        n_layers: int,
        d_ffn: int,
        d_model: int,
        n_heads: int,
        drop_rate: float,
        **kwargs
    ):
        """
        Initialize the Transformer model.

        Args:
            seq_len (int): Sequence length.
            vocab_size (int): Vocabulary size.
            n_layers (int): Number of layers.
            d_ffn (int): Feed-forward units.
            d_model (int): Model dimensionality.
            n_heads (int): Number of attention heads.
            drop_rate (float): Dropout rate.
            **kwargs: Additional arguments.
        """
        super().__init__(**kwargs)
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.n_layers = n_layers
        self.d_ffn = d_ffn
        self.d_model = d_model
        self.n_heads = n_heads
        self.drop_rate = drop_rate

        self.mask = CausalMaskLayer(name="attention_mask")

        self.decoder = Decoder(
            seq_len,
            vocab_size,
            n_layers,
            d_ffn,
            d_model,
            n_heads,
            drop_rate,
            name="decoder",
        )

        self.final_dense = keras.layers.Dense(vocab_size, name="output_projection")

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
                "d_ffn": self.d_ffn,
                "d_model": self.d_model,
                "n_heads": self.n_heads,
                "drop_rate": self.drop_rate,
            }
        )

        return config

    def call(self, inputs, training=None):
        """
        Forward pass of the Transformer.

        Args:
            inputs: Tuple of encoder and decoder inputs.
            training: Whether in training mode.

        Returns:
            Model outputs.
        """
        causal_mask = self.mask(inputs)

        dec_outputs = self.decoder(
            inputs,
            causal_mask,
            training=training,
        )

        return self.final_dense(dec_outputs)
