from distutils.util import strtobool
import numpy as np
import argparse
import torch
from data import get_train_loaders
from model import Net, StnNet
from torch import nn, optim
import torch.nn.functional as F


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


def fit(epochs, model, loss_func, opt, train_dl, valid_dl):
    valid_loss_min = np.Inf
    for epoch in range(epochs):
        # Train model
        model.train()
        losses, nums = zip(
            *[loss_batch(model, loss_func, x, y, opt) for x, y in train_dl])
        train_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
        # Validation model
        model.eval()
        with torch.no_grad():
            losses, corrects, nums = zip(
                *[valid_batch(model, loss_func, x, y) for x, y in valid_dl])
            valid_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
            valid_accuracy = np.sum(corrects) / np.sum(nums) * 100
            print(f"[Epoch {epoch+1}/{epochs}] "
                  f"Train loss: {train_loss:.6f}\t"
                  f"Validation loss: {valid_loss:.6f}\t",
                  f"Validation accruacy: {valid_accuracy:.3f}%")
            # save model if validation loss has decreased
            if valid_loss <= valid_loss_min:
                print(
                    f"Validation loss decreased ({valid_loss_min:.6f} --> {valid_loss:.6f}). Saving model...")
                model_file = 'model_' + str(epoch+1) + '.pt'
                checkpoint = {
                    'state_dict': model.state_dict(),
                    'grayscale': args.grayscale,
                    'stn': args.stn
                }
                torch.save(checkpoint, model_file)
                valid_loss_min = valid_loss
                print(f"You can run `python evaluate.py --model {model_file}`")


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(
        description='Traffic sign recognition training script')
    parser.add_argument('--data', type=str, default='data', metavar='D',
                        help="folder where data is located. train.p and vliad.p need to be found in the folder (default: data)")
    parser.add_argument('--grayscale', type=lambda x: bool(strtobool(x)), default=True, metavar='G',
                        help='convert a RGB image to grayscale (default: true)')
    parser.add_argument('--augmentation', type=lambda x: bool(strtobool(x)), default=True, metavar='A',
                        help='image augmentation (default: true)')
    parser.add_argument('--stn', type=lambda x: bool(strtobool(x)), default=True, metavar='T',
                        help='use spatial transformer networks (default: true)')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--num-workers', type=int, default=0, metavar='W',
                        help='how many subprocesses to use for data loading (default: 0)')
    parser.add_argument('--epochs', type=int, default=30, metavar='N',
                        help='number of epochs to train (default: 30)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    # Data Initialization and Loading
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, valid_loader = get_train_loaders(
        args.data, device, args.batch_size, args.num_workers, args.grayscale, args.augmentation)

    # Neural Network and Optimizer
    if args.stn:
        model = StnNet(gray=args.grayscale).to(device)
    else:
        model = Net(gray=args.grayscale).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=args.momentum)

    # Training and Validation
    fit(args.epochs, model, criterion, optimizer, train_loader, valid_loader)
