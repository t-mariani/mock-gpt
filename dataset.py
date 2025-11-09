from functools import partial

import torch


def load_text():
    with open("data/shakespear.txt", "r") as f:
        text = f.read()
    return text


def get_alphabet(text):
    chars = sorted(list(set(text)))
    return chars


def wrap_alphabet(coder, chars):
    return partial(coder, chars)


def encode(alphabet, text):
    encoder = {k: i for i, k in enumerate(alphabet)}
    return [encoder[c] for c in text]


def decode(alphabet, list_ind):
    decoder = {i: k for i, k in enumerate(alphabet)}
    return "".join([decoder[e] for e in list_ind])


def load_torch_dataset():
    txt = load_text()
    encoded_txt = encode(get_alphabet(txt), txt)
    return torch.tensor(encoded_txt, dtype=torch.long)


def get_alphabet_size():
    txt = load_text()
    alphabet = get_alphabet(txt)
    return len(alphabet)


def get_batch(cfg, data):
    ix = torch.randint(0, len(data) - cfg.block_size - 1, (cfg.batch_size,))
    x = torch.stack([data[i : i + cfg.block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + cfg.block_size + 1] for i in ix])
    return x, y
