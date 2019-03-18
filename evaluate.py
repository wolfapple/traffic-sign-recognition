import matplotlib.pyplot as plt
import numpy as np
import argparse
import torch
from torch import nn
from model import TrafficSignNet
from data import get_test_loader
from torchvision.utils import make_grid
from train import valid_batch
from tqdm import tqdm


def evaluate(model, loss_func, dl):
    model.eval()
    with torch.no_grad():
        losses, corrects, nums = zip(
            *[valid_batch(model, loss_func, x, y) for x, y in tqdm(dl)])
        test_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
        test_accuracy = np.sum(corrects) / np.sum(nums) * 100

        print(f"Test loss: {test_loss:.6f}\t"
              f"Test accruacy: {test_accuracy:.3f}%")


def convert_image_np(img):
    img = img.numpy().transpose((1, 2, 0)).squeeze()
    return img


def visualize_stn(dl, outfile):
    with torch.no_grad():
        data = next(iter(dl))[0]

        input_tensor = data.cpu()
        transformed_tensor = model.stn(data).cpu()

        input_grid = convert_image_np(make_grid(input_tensor))
        transformed_grid = convert_image_np(make_grid(transformed_tensor))

        # Plot the results side-by-side
        fig, ax = plt.subplots(1, 2)
        fig.set_size_inches((16, 16))
        ax[0].imshow(input_grid)
        ax[0].set_title('Dataset Images')
        ax[0].axis('off')

        ax[1].imshow(transformed_grid)
        ax[1].set_title('Transformed Images')
        ax[1].axis('off')

        plt.savefig(outfile)


if __name__ == "__main__":
    # Evaluation settings
    parser = argparse.ArgumentParser(
        description='Traffic sign recognition evaluation script')
    parser.add_argument('--data', type=str, default='data', metavar='D',
                        help="folder where data is located. test.p need to be found in the folder (default: data)")
    parser.add_argument('--model', type=str, default='model.pt', metavar='M',
                        help="the model file to be evaluated. (default: model.pt)")
    parser.add_argument('--outfile', type=str, default='visualize_stn.png', metavar='O',
                        help="visualize the STN transformation on some input batch (default: visualize_stn.png)")

    args = parser.parse_args()

    # Load model checkpoint
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.model, map_location=device)

    # Neural Network and Loss Function
    model = TrafficSignNet().to(device)
    model.load_state_dict(checkpoint)
    model.eval()
    criterion = nn.CrossEntropyLoss()

    # Data Initialization and Loading
    test_loader = get_test_loader(args.data, device)
    evaluate(model, criterion, test_loader)
    visualize_stn(test_loader, args.outfile)
