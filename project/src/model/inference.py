import tensorflow as tf


def greedy_decode(model, tokenizer, input_text: str, max_len: int) -> str:
    """
    Greedy Search decoding.

    Args:
        model: The sequence generation model.
        tokenizer: The tokenizer used for encoding/decoding.
        input_text (str): Input prompt.
        max_len (int): Maximum generation length.

    Returns:
        Generated text.
    """
    input_tokens = tokenizer.encode([f"<bos> {input_text} <mos>"])

    prompt_len = input_tokens.shape[1]

    eos_token_id = tokenizer.token_to_id["<eos>"]
    mos_token_id = tokenizer.token_to_id["<mos>"]

    for _ in range(max_len):
        preds = model(input_tokens[:, -model.seq_len :])

        logits = preds[:, -1, :]

        next_token = tf.argmax(logits, axis=-1)

        input_tokens = tf.concat(
            [input_tokens, tf.expand_dims(next_token, axis=1)], axis=1
        )

        if tf.reduce_all(tf.equal(next_token, eos_token_id)) or tf.reduce_all(
            tf.equal(next_token, mos_token_id)
        ):
            break

    generated = input_tokens.numpy()[0]
    generated = generated[prompt_len:]

    response = tokenizer.decode([generated])[0]
    response = response.replace("<eos>", "").replace("<mos>", "").strip()

    return response


def top_p_decode(
    model,
    tokenizer,
    input_text: str,
    max_len: int,
    temperature: float,
    top_p: float,
) -> str:
    """
    Top-P (Nucleus) Sampling with Temperature decoding.

    Args:
        model: The sequence generation model.
        tokenizer: The tokenizer used for encoding/decoding.
        input_text (str): Input prompt.
        max_len (int): Maximum generation length.
        temperature (float): Sampling temperature.
        top_p (float): Cumulative probability threshold.

    Returns:
        Generated text.
    """
    input_tokens = tokenizer.encode([f"<bos> {input_text} <mos>"])

    prompt_len = input_tokens.shape[1]

    eos_token_id = tokenizer.encode(["<eos>"])[0, 0]
    mos_token_id = tokenizer.encode(["<mos>"])[0, 0]

    for _ in range(max_len):
        preds = model(input_tokens[:, -model.seq_len :])

        logits = preds[:, -1, :]
        logits = logits / temperature

        probs = tf.nn.softmax(logits, axis=-1)

        sorted_probs, sorted_indices = tf.math.top_k(probs, k=tf.shape(probs)[-1])

        cumulative_probs = tf.cumsum(sorted_probs, axis=-1)

        mask = cumulative_probs <= top_p
        mask = tf.concat(
            [tf.ones_like(mask[:, :1], dtype=tf.bool), mask[:, 1:]], axis=-1
        )

        filtered_probs = tf.where(mask, sorted_probs, tf.zeros_like(sorted_probs))
        filtered_probs = filtered_probs / tf.reduce_sum(
            filtered_probs, axis=-1, keepdims=True
        )

        sampled_index = tf.random.categorical(
            tf.math.log(filtered_probs), num_samples=1
        )

        next_token = tf.gather(sorted_indices, sampled_index, batch_dims=1)
        next_token = tf.squeeze(next_token, axis=-1)
        next_token = tf.cast(next_token, dtype=tf.int64)

        input_tokens = tf.concat(
            [input_tokens, tf.expand_dims(next_token, axis=1)], axis=1
        )

        if tf.reduce_all(tf.equal(next_token, eos_token_id)) or tf.reduce_all(
            tf.equal(next_token, mos_token_id)
        ):
            break

    generated = input_tokens.numpy()[0]
    generated = generated[prompt_len:]

    response = tokenizer.decode([generated])[0]
    response = response.replace("<eos>", "").replace("<mos>", "").strip()

    return response
