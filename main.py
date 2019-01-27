import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

# directory for the data used between multiple runs
STATE_DIR = './state'
MNIST_DATA_DIR = os.path.join(STATE_DIR, 'mnist_data/')
MODEL_STATE_PATH = os.path.join(STATE_DIR, 'mnist_cnn.pt')

# directory with the inputs for a singular run
INPUTS_DIR = './data'
X_TRAIN_PATH = os.path.join(INPUTS_DIR, 'X_train.npy')
Y_TRAIN_PATH = os.path.join(INPUTS_DIR, 'y_train.npy')
X_TEST_PATH = os.path.join(INPUTS_DIR, 'X_test.npy')
Y_TEST_PATH = os.path.join(INPUTS_DIR, 'y_test.npy')


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1)
        self.fc1 = nn.Linear(in_features=4*4*64, out_features=10, bias=True)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*64)
        x = self.fc1(x)
        return F.log_softmax(x, dim=1)


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    s_loss, batch_idx = 0., 1
    dataset_size = len(train_loader.dataset)

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        s_loss += loss.item()

        if (batch_idx + 1) % args.log_interval == 0:
            info = f'Epoch: {epoch:2d} [{(batch_idx + 1) * args.batch_size}/{dataset_size} '\
                f'({100. * (batch_idx + 1) / len(train_loader):.0f}%)]\t' \
                f'loss: {s_loss/(batch_idx + 1):.4f}'
            print(info, end='\r')

    info = f'Epoch: {epoch:2d} [{dataset_size}/{dataset_size} ' \
        f'(100%)]\tloss: {s_loss/(batch_idx + 1):.4f}'
    print(info)


def predict(args, model, device, data_loader):
    model.eval()

    s_loss, n_correct = 0., 0
    dataset_size = len(data_loader.dataset)
    y_pred = torch.zeros(dataset_size)

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(data_loader):
            data, target = data.to(device), target.to(device)

            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability

            start = batch_idx * args.test_batch_size
            end = start + args.test_batch_size
            y_pred[start:end] = pred.view(-1)

            if args.mode == 'eval':
                s_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
                n_correct += pred.eq(target.view_as(pred)).sum().item()

            info = f'Inference ' \
                f'[{min((batch_idx + 1) * args.test_batch_size, dataset_size)}/{dataset_size} ' \
                f'({100. * (batch_idx + 1) / len(data_loader):.0f}%)]'
            print(info, end='\r')

        info = f'Inference ' \
            f'[{min((batch_idx + 1) * args.test_batch_size, dataset_size)}/{dataset_size} ' \
            f'(100%)]'
        print(info)

    if args.mode == 'eval':
        s_loss /= dataset_size
        info = f'Evaluation: loss: {s_loss:.4f}, acc.: {n_correct}/{dataset_size} ' \
            f'({100. * n_correct / dataset_size:.2f}%)'
        print(info)

    return y_pred


def build_dataset(X, y=None):
    if len(X.shape) != 4 and not np.array_equal(X.shape[1:], np.array([1, 28, 28])):
        print(f'> Error: Dataset has incorrect shape, found {X.shape}, required (N, 1, 28, 28)')
        return None

    if y is not None:
        y = np.reshape(y, (-1))

        if X.shape[0] != y.shape[0]:
            print(f'> Error: the first dimension of X and y is different')
            return None
    else:
        y = np.zeros(X.shape[0])

    tensor_x = torch.Tensor(X)
    tensor_y = torch.LongTensor(y)

    return torch.utils.data.TensorDataset(tensor_x, tensor_y)


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--mode', type=str, default='fit', metavar='M',
                        help='run mode (`fit`, `predict` or `eval`)')
    parser.add_argument('--model-path', type=str, default=MODEL_STATE_PATH, metavar='P',
                        help='path to save to and load from the model')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for inference (default: 1000)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()

    if args.mode != 'fit' and args.mode != 'predict' and args.mode != 'eval':
        print('Incorrect run mode, use either '
              '\n `--mode fit` to fit a new model'
              '\n `--mode predict` to run an inference'
              '\n `--mode eval` to run model evaluation on the input')
        return

    use_cuda = torch.cuda.is_available()  # use cuda if available
    device = torch.device("cuda" if use_cuda else "cpu")
    torch.manual_seed(args.seed)  # random seed

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    if args.mode == 'fit':
        train_set = None
        if os.path.isdir(STATE_DIR):
            if os.path.exists(X_TRAIN_PATH) and os.path.exists(Y_TRAIN_PATH):
                X_train = np.load(X_TRAIN_PATH)
                y_train = np.load(Y_TRAIN_PATH)

                train_set = build_dataset(X_train, y_train)
            else:
                print('> Training files `X_train.npy` and `y_train.npy` are not found')
        else:
            print('> Training directory is not found')

        if train_set is None:
            print('> Using default MNIST train data')
            train_set = datasets.MNIST(MNIST_DATA_DIR, train=True, download=True,
                                       transform=transforms.Compose([
                                           transforms.ToTensor(),
                                       ]))
        else:
            print('> Using provided files `X_train.npy` and `y_train.npy`')

        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=args.batch_size, shuffle=True, **kwargs
        )

        # Building the model and setting up the optimizer
        model = Net().to(device)
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

        for epoch in range(1, args.epochs + 1):
            train(args, model, device, train_loader, optimizer, epoch)

        print(f'> Saving the model state at {args.model_path}')
        torch.save(model.state_dict(), args.model_path)
    else:
        if os.path.isfile(args.model_path):
            model = Net().to(device)
            model.load_state_dict(torch.load(args.model_path))
            print('> Using saved model state')
        else:
            print('> Model state file is not found, fit a model before the inference')
            print('> Stopping execution')
            return

        test_set = None
        if os.path.isdir(STATE_DIR):
            if os.path.exists(X_TEST_PATH):
                X_test = np.load(X_TEST_PATH)

                if args.mode == 'eval' and os.path.exists(Y_TEST_PATH):
                    y_test = np.load(Y_TEST_PATH)
                else:
                    print('> Targets file `y_test.npy` is not found')
                    y_test = None

                test_set = build_dataset(X_test, y_test)

            else:
                print('> Input file `X_test.npy` is not found')

        else:
            print('> Training data directory is not found')

        if test_set is None:
            print('> Using default MNIST train data')
            test_set = datasets.MNIST(MNIST_DATA_DIR, train=False, download=True,
                                      transform=transforms.Compose([
                                          transforms.ToTensor(),
                                      ]))
        else:
            print('> Using provided files X_test.npy and (y_test.npy)')

        eval_loader = torch.utils.data.DataLoader(
            test_set, batch_size=args.test_batch_size, shuffle=True, **kwargs
        )

        print('> Running inference on input data')
        y_pred = predict(args, model, device, eval_loader)

        print('> Saving predictions at `data/y_pred.npy`')
        if not os.path.exists(INPUTS_DIR):
            os.mkdir(INPUTS_DIR)
        np.save(os.path.join(INPUTS_DIR, 'y_pred.npy'), y_pred.data)


if __name__ == '__main__':
    main()
