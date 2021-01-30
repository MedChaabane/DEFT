from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os

import torch
import torch.utils.data
from opts import opts
from model.model import create_model, load_model, save_model
from model.data_parallel import DataParallel
from logger import Logger
from dataset.dataset_factory import get_dataset
from trainer import Trainer
from dataset.trajectory_dataset import TrajectoryDataset


def get_optimizer(opt, model):
    if opt.optim == "adam":
        optimizer = torch.optim.Adam(model.parameters(), opt.lr)
    elif opt.optim == "sgd":
        print("Using SGD")
        optimizer = torch.optim.SGD(
            model.parameters(), opt.lr, momentum=0.9, weight_decay=0.0001
        )
    else:
        assert 0, opt.optim
    return optimizer


class DecoderRNN(torch.nn.Module):
    def __init__(self, num_hidden, opt):
        super(DecoderRNN, self).__init__()
        self.num_hidden = num_hidden
        if opt.dataset == "nuscenes":
            self.lstm = torch.nn.LSTM(18, self.num_hidden)
            self.out1 = torch.nn.Linear(self.num_hidden, 64)
            self.out2 = torch.nn.Linear(64, 4 * 4)
        else:
            self.lstm = torch.nn.LSTM(11, self.num_hidden)
            self.out1 = torch.nn.Linear(self.num_hidden, 64)
            self.out2 = torch.nn.Linear(64, 4 * 5)

    def forward(self, input_traj):
        # Fully connected
        input_traj = input_traj.permute(1, 0, 2)
        output, (hn, cn) = self.lstm(input_traj)
        x = self.out1(output[-1])
        x = self.out2(x)
        return x


def main(opt):
    torch.manual_seed(opt.seed)
    torch.backends.cudnn.benchmark = not opt.not_cuda_benchmark and not opt.test
    Dataset = get_dataset(opt.dataset, prediction_model=True)
    if not opt.not_set_cuda_env:
        os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus_str
    opt.device = torch.device("cuda" if opt.gpus[0] >= 0 else "cpu")
    device = opt.device
    logger = Logger(opt)

    print("Creating model...")

    model = DecoderRNN(128, opt)
    optimizer = get_optimizer(opt, model)
    start_epoch = 0
    if opt.load_model_traj != "":
        model, optimizer, start_epoch = load_model(
            model, opt.load_model, opt, optimizer
        )
    loss_function = torch.nn.SmoothL1Loss()

    for i, param in enumerate(model.parameters()):
        print("param ", i)
        param.requires_grad = True

    train_loader = torch.utils.data.DataLoader(
        Dataset(opt, "train"),
        batch_size=1,
        shuffle=True,
        num_workers=16,
        pin_memory=True,
        drop_last=True,
    )

    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device=device, non_blocking=True)
    model = model.to(device)
    loss_function = loss_function.to(device)

    print("Starting training...")
    for epoch in range(start_epoch + 1, opt.num_epochs + 1):
        mark = epoch if opt.save_all else "last"
        for iter_id, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(device=device).float()
            targets = targets.to(device=device).view(1, -1).float()
            outputs = model(inputs)
            loss = loss_function(outputs, targets)
            if 100 * loss.item() < 20:
                loss = 100 * loss
            else:
                loss = 10 * loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            del outputs, loss

        save_model(
            os.path.join(opt.save_dir, "model_last.pth"), epoch, model, optimizer
        )
        logger.write("\n")
        save_model(
            os.path.join(opt.save_dir, "model_{}.pth".format(epoch)),
            epoch,
            model,
            optimizer,
        )
        if epoch in opt.lr_step:
            lr = opt.lr * (0.1 ** (opt.lr_step.index(epoch) + 1))
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
    logger.close()


if __name__ == "__main__":
    opt = opts().parse()
    main(opt)
