import numpy as np
import argparse
import torch
from data import get_train_loaders, preprocess
from model import TrafficSignNet
from torch import nn, optim
import torch.nn.functional as F
from tqdm import tqdm


def loss_batch(model, loss_func, x, y, opt=None):
    loss = loss_func(model(x), y)

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item(), len(x)


def valid_batch(model, loss_func, x, y):
    output = model(x)
    loss = loss_func(output, y)
    pred = torch.argmax(output, dim=1)
    correct = pred == y.view(*pred.shape)

    return loss.item(), torch.sum(correct).item(), len(x)


def fit(epochs, model, loss_func, opt, train_dl, valid_dl, patience, checkpoint):
    wait = 0
    valid_loss_min = np.Inf
    for epoch in range(epochs):
        # Train model
        model.train()
        losses, nums = zip(
            *[loss_batch(model, loss_func, x, y, opt) for x, y in tqdm(train_dl, desc=f"[Epoch {epoch+1}/{epochs}]")])
        train_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
        # Validation model
        model.eval()
        with torch.no_grad():
            losses, corrects, nums = zip(
                *[valid_batch(model, loss_func, x, y) for x, y in valid_dl])
            valid_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
            valid_accuracy = np.sum(corrects) / np.sum(nums) * 100
            print(f"- Train loss: {train_loss:.6f}\t"
                  f"Validation loss: {valid_loss:.6f}\t",
                  f"Validation accruacy: {valid_accuracy:.3f}%")
            # save model if validation loss has decreased
            if valid_loss <= valid_loss_min:
                print(
                    f"- Validation loss decreased ({valid_loss_min:.6f} --> {valid_loss:.6f}). Saving model...")
                torch.save(model.state_dict(), checkpoint)
                valid_loss_min = valid_loss
                wait = 0
            # Early stopping
            else:
                wait += 1
                if wait >= patience:
                    print(
                        f"Terminated Training for Early Stopping at Epoch {epoch+1}")
                    return


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(
        description='Traffic sign recognition training script')
    parser.add_argument('--data', type=str, default='data', metavar='D',
                        help="Folder where data is located. train.p and vliad.p need to be found in the folder (default: data)")
    parser.add_argument('--class-count', type=int, default=20000, metavar='C',
                        help='Each class will have this number of samples after extension and balancing (default: 10k)')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='Input batch size for training (default: 64)')
    parser.add_argument('--num-workers', type=int, default=0, metavar='W',
                        help='How many subprocesses to use for data loading (default: 0)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='Number of epochs to train (default: 100)')
    parser.add_argument('--patience', type=int, default=10, metavar='P',
                        help='Number of epochs with no improvement after which training will be stopped (default: 10)')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='Learning rate (default: 0.0001)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='Random seed (default: 1)')
    parser.add_argument('--checkpoint', type=str, default='model.pt', metavar='M',
                        help='checkpoint file name (default: model.pt)')
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    # Data Initialization and Loading
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    preprocess(args.data)
    train_loader, valid_loader = get_train_loaders(
        args.data, device, args.batch_size, args.num_workers, args.class_count)

    # Neural Network and Optimizer
    model = TrafficSignNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Training and Validation
    fit(args.epochs, model, criterion, optimizer,
        train_loader, valid_loader, args.patience, args.checkpoint)
