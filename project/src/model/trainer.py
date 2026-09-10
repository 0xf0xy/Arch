from .metrics import masked_loss, masked_accuracy
from .dataset import DatasetLoader, DataPipeline
from .transformer import Transformer
from .bpe import BPETokenizer
import tensorflow as tf
from .noam import Noam
import numpy as np
import keras


class Trainer:
    """Trainer class for training the Transformer model."""

    def __init__(self, config: dict):
        self.model_config = config["model"]
        self.tokenizer_config = config["tokenizer"]
        self.train_config = config["training"]
        self.path_config = config["path"]
        self.tokenizer = None
        self.model = None

    def load_data(self):
        """Load dataset from file."""
        loader = DatasetLoader(
            self.path_config["dataset"], self.train_config["data_limit"]
        )
        self.inputs = loader.load()

    def split_data(self):
        """Split inputs into train and validation sets."""
        rng = np.random.default_rng(self.train_config["seed"])
        rng.shuffle(self.inputs)

        size = len(self.inputs)
        split = int(size * (1 - self.train_config["val_split"]))

        self.train_in, self.val_in = (self.inputs[:split], self.inputs[split:])

    def build_tokenizer(self):
        """Build tokenizer and train on the training data."""
        self.tokenizer = BPETokenizer(
            max_tokens=self.model_config["vocab_size"],
            output_sequence_length=self.model_config["seq_len"],
            min_frequency=self.tokenizer_config["min_freq"],
            standardize=self.tokenizer_config["standardize"],
            pattern_tokenize=self.tokenizer_config["pattern_tokenize"],
        )

        self.tokenizer.train(self.train_in)

        if self.train_config["save"]:
            self.tokenizer.save(self.path_config["tokenizer"])

    def build_dataset(self):
        """Build TensorFlow datasets for training and validation."""
        pipeline = DataPipeline(
            self.tokenizer,
            batch_size=self.train_config["batch_size"],
            buffer_size=self.train_config["buffer_size"],
        )
        self.train_ds = pipeline.build(self.train_in, training=True)
        self.val_ds = pipeline.build(self.val_in, training=False)

    def build_model(self):
        """Build the Transformer model."""
        self.model = Transformer(
            seq_len=self.model_config["seq_len"],
            vocab_size=self.model_config["vocab_size"],
            n_layers=self.model_config["n_layers"],
            d_ffn=self.model_config["d_ffn"],
            d_model=self.model_config["d_model"],
            n_heads=self.model_config["n_heads"],
            drop_rate=self.model_config["drop_rate"],
            name="transformer",
        )

        dummy = tf.ones((1, self.model_config["seq_len"]))
        _ = self.model(dummy)

    def compile(self):
        """Compile the model with optimizer and loss."""
        lr = Noam(self.model_config["d_model"], self.train_config["warmup_steps"])
        optimizer = keras.optimizers.AdamW(
            lr,
            weight_decay=self.train_config["weight_decay"],
            beta_1=0.9,
            beta_2=0.98,
            epsilon=1e-9,
        )

        self.model.compile(
            optimizer=optimizer, loss=masked_loss, metrics=[masked_accuracy]
        )

    def train(self):
        """Train the model on the training dataset."""
        callbacks = [
            keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)
        ]
        self.model.fit(
            self.train_ds,
            validation_data=self.val_ds,
            epochs=self.train_config["epochs"],
            callbacks=callbacks,
        )

        if self.train_config["save"]:
            self.model.save(self.path_config["model"])

    def run(self):
        """Run the full training pipeline."""
        self.load_data()
        self.split_data()
        self.build_tokenizer()
        self.build_dataset()
        self.build_model()
        self.compile()
        self.train()
