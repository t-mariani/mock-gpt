import numpy as np
import torch
from torch.optim import AdamW
from torchinfo import summary

from utils import load_cfg
from dataset import load_torch_dataset, get_batch, get_alphabet_size
from model import LargeLanguageModel


def run_val_step(model, dataset, n_tests=50):
    model.eval()
    with torch.no_grad():
        test_losses = []
        for _ in range(n_tests):
            xb, yb = get_batch(cfg, dataset)
            _, loss = model(xb, yb)
            test_losses.append(loss.cpu().detach().numpy())
    model.train()
    return test_losses


if __name__ == "__main__":
    cfg = load_cfg("config.yaml")

    if cfg.device == "auto":
        if torch.cuda.is_available():
            torch.set_default_device("cuda")
        elif torch.mps.is_available():
            torch.set_default_device("mps")
    elif cfg.device is not None:
        torch.set_default_device(cfg.device)

    dataset = load_torch_dataset()
    alphabet_size = get_alphabet_size()
    train, valtest = torch.split(dataset, int(cfg.train_split * len(dataset)))
    remaining_ratio = cfg.val_split / (1 - cfg.train_split)
    val, test = torch.split(valtest, int(remaining_ratio * len(valtest)))

    model = LargeLanguageModel(cfg, alphabet_size)
    summary(model)
    optimizer = AdamW(model.parameters(), cfg.lr)

    losses = []
    for i in range(cfg.n_epoch):
        xb, yb = get_batch(cfg, train)
        _, loss = model(xb, yb)
        losses.append(loss.cpu().detach().numpy())
        if i % cfg.log_each_n == 0:
            l = losses[-cfg.log_each_n :]
            print(f"Iter{i} = {sum(l)/len(l)}")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % cfg.eval_interval == 0:
            val_losses = run_val_step(model, val)
            print(f"Average val loss : {np.mean(val_losses)} +/- {np.std(val_losses)}")

    # Test set
    n_tests = 100
    print(f"Running Test ({n_tests} predictions)")
    test_losses = run_val_step(model, test, n_tests=n_tests)
    print(f"Average test loss : {np.mean(test_losses)} +/- {np.std(test_losses)}")
