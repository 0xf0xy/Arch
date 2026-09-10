from project.src.model.inference import greedy_decode, top_p_decode
from project.src.model.metrics import masked_accuracy, masked_loss
from project.src.model.transformer import Transformer
from project.src.model.bpe import BPETokenizer
from project.src.model.trainer import Trainer
from project.src.model.noam import Noam

import keras
import tensorflow as tf
import yaml


class Arch:
    """
    Arch: Experimental autoregressive language model.

    Provides a high-level interface for configuring, training,
    loading and running inference with the model.
    """

    def __init__(self):
        self.config = None
        self.model = None
        self.tokenizer = None

    def load_config(self, config_path):
        """Load the model configuration from a YAML file."""
        with open(config_path, "r", encoding="utf-8") as file:
            self.config = yaml.safe_load(file)

    def display_config(self):
        """Display the currently loaded configuration."""
        if self.config is None:
            print("No configuration loaded.")
            return

        print(yaml.dump(self.config, sort_keys=False))

    def train(self):
        """
        Train the model using the loaded configuration.

        Returns:
            The trained model.
        """
        if self.config is None:
            raise RuntimeError("Configuration not loaded.")

        trainer = Trainer(self.config)
        trainer.run()

        self.model = trainer.model
        self.tokenizer = trainer.tokenizer

        return self.model

    def load(self):
        """
        Load a trained model and tokenizer from the configured
        artifact paths.

        Returns:
            The loaded model.
        """
        if self.config is None:
            raise RuntimeError("Configuration not loaded.")

        self.model = keras.models.load_model(self.config["path"]["model"])

        self.tokenizer = BPETokenizer()
        self.tokenizer.load(self.config["path"]["tokenizer"])

        return self.model

    def generate(
        self,
        prompt,
        strategy="greedy",
        max_len=None,
        temperature=None,
        top_p=None,
    ):
        """
        Generate text from a prompt using the selected decoding strategy.

        Args:
            prompt: Input text.
            strategy: Decoding strategy ('greedy' or 'top_p').
            max_len: Maximum generated sequence length.
            temperature: Sampling temperature for top-p decoding.
            top_p: Nucleus sampling probability.

        Returns:
            Generated text.
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model and tokenizer are not loaded.")

        inference_config = self.config.get("inference", {})

        max_len = max_len or inference_config.get("max_len", 128)

        if strategy == "greedy":
            return greedy_decode(
                self.model,
                self.tokenizer,
                prompt,
                max_len,
            )

        if strategy == "top_p":
            temperature = temperature or inference_config.get(
                "temperature",
                1.0,
            )

            top_p = top_p or inference_config.get(
                "top_p",
                0.9,
            )

            return top_p_decode(
                self.model,
                self.tokenizer,
                prompt,
                max_len,
                temperature,
                top_p,
            )

        raise ValueError(f"Unknown decoding strategy: {strategy}")

    @staticmethod
    def gpu_available():
        """Check whether a GPU is available."""
        return bool(tf.config.list_physical_devices("GPU"))


__all__ = [
    "Arch",
    "Transformer",
    "BPETokenizer",
    "Trainer",
    "Noam",
    "greedy_decode",
    "top_p_decode",
    "masked_accuracy",
    "masked_loss",
]
