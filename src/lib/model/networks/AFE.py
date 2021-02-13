# ------------------------------------------------------------------------------
# Large Part of this file is taken from https://github.com/shijieS/SST
# ------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import os
import cv2
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

SELECTOR_INPUT_CHANNEL = [16, 32, 64, 128, 256, 512, 64, 128, 256, 512, 64, 64, 64]


class AFE_module(nn.Module):
    # new: combine two vgg_net
    def __init__(self, opt):
        super(AFE_module, self).__init__()

        if opt.dataset == "nuscenes":
            SELECTOR_OUTPUT_CHANNEL = [
                48,
                48,
                64,
                64,
                64,
                64,
                64,
                64,
                64,
                64,
                32,
                32,
                32,
            ]

        else:
            SELECTOR_OUTPUT_CHANNEL = [
                32,
                32,
                32,
                32,
                32,
                32,
                32,
                32,
                32,
                32,
                32,
                32,
                32,
            ]

        FINAL_NET = [832, 512, 256, 128, 64, 1]
        FINAL_NET[0] = np.sum(SELECTOR_OUTPUT_CHANNEL) * 2
        self.stacker2_bn = nn.BatchNorm2d(int(FINAL_NET[0] / 2))
        self.final_dp = nn.Dropout(0.25)
        self.final_net = nn.ModuleList(add_final(FINAL_NET))

        self.max_object = opt.max_object
        self.selector_input_channel = SELECTOR_INPUT_CHANNEL

        self.selector_output_channel = SELECTOR_OUTPUT_CHANNEL
        self.selector = nn.ModuleList(
            selector(self.selector_input_channel, self.selector_output_channel)
        )
        self.false_objects_column = None
        self.false_objects_row = None
        self.false_constant = 1.0

    def forward(self, sources_pre, sources_next, l_pre, l_next):

        x_pre = self.forward_selector_stacker1(sources_pre, l_pre, self.selector)
        x_next = self.forward_selector_stacker1(sources_next, l_next, self.selector)
        # [B, N, N, C]
        x = self.forward_stacker2(x_pre, x_next)
        #         x = self.final_dp(x)
        # [B, N, N, 1]
        x = self.forward_final(x, self.final_net)

        # add false unmatched row and column
        x = self.add_unmatched_dim(x)
        return x

    def forward_feature_extracter(self, s, l):

        x = self.forward_selector_stacker1(s, l, self.selector)

        return x

    def get_similarity(self, image1, detection1, image2, detection2):
        feature1 = self.forward_feature_extracter(image1, detection1)
        feature2 = self.forward_feature_extracter(image2, detection2)
        return self.forward_stacker_features(feature1, feature2, False)

    def resize_dim(self, x, added_size, dim=1, constant=0):
        if added_size <= 0:
            return x
        shape = list(x.shape)
        shape[dim] = added_size
        if torch.cuda.is_available():
            new_data = Variable(torch.ones(shape) * constant).cuda()
        else:
            new_data = Variable(torch.ones(shape) * constant)
        return torch.cat([x, new_data], dim=dim)

    def forward_stacker_features(self, xp, xn, fill_up_column=True):
        pre_rest_num = self.max_object - xp.shape[1]
        next_rest_num = self.max_object - xn.shape[1]
        pre_num = xp.shape[1]
        next_num = xn.shape[1]
        x = self.forward_stacker2(
            self.resize_dim(xp, pre_rest_num, dim=1),
            self.resize_dim(xn, next_rest_num, dim=1),
        )

        # [B, N, N, 1]
        x = self.forward_final(x, self.final_net)
        x = x.contiguous()
        # add zero
        if next_num < self.max_object:
            x[0, 0, :, next_num:] = 0
        if pre_num < self.max_object:
            x[0, 0, pre_num:, :] = 0
        x = x[0, 0, :]
        # add false unmatched row and column
        x = self.resize_dim(x, 1, dim=0, constant=self.false_constant)
        x = self.resize_dim(x, 1, dim=1, constant=self.false_constant)

        x_f = F.softmax(x, dim=1)
        x_t = F.softmax(x, dim=0)
        # slice
        last_row, last_col = x_f.shape
        row_slice = list(range(pre_num)) + [last_row - 1]
        col_slice = list(range(next_num)) + [last_col - 1]
        x_f = x_f[row_slice, :]
        x_f = x_f[:, col_slice]
        x_t = x_t[row_slice, :]
        x_t = x_t[:, col_slice]

        x = Variable(torch.zeros(pre_num, next_num + 1))
        x[0:pre_num, 0:next_num] = torch.max(
            x_f[0:pre_num, 0:next_num], x_t[0:pre_num, 0:next_num]
        )
        x[:, next_num : next_num + 1] = x_f[:pre_num, next_num : next_num + 1]
        if fill_up_column and pre_num > 1:
            x = torch.cat(
                [x, x[:, next_num : next_num + 1].repeat(1, pre_num - 1)], dim=1
            )

        if torch.cuda.is_available():
            y = x.data.cpu().numpy()

        else:
            y = x.data.numpy()

        return y

    def forward_selector_stacker1(self, sources, labels, selector):
        """
        :param sources: [B, C, H, W]
        :param labels: [B, N, 1, 1, 2]
        :return: the connected feature
        """
        sources = [F.relu(net(x), inplace=True) for net, x in zip(selector, sources)]

        res = list()

        for label_index in range(labels.size(1)):
            label_res = list()
            for source_index in range(len(sources)):
                # [N, B, C, 1, 1]
                label_res.append(
                    # [B, C, 1, 1]
                    F.grid_sample(
                        sources[source_index],  # [B, C, H, W]
                        labels[:, label_index, :],  # [B, 1, 1, 2
                        padding_mode="border",
                    )
                    .squeeze(2)
                    .squeeze(2)
                )
            res.append(torch.cat(label_res, 1))

        return torch.stack(res, 1)

    def forward_stacker2(self, stacker1_pre_output, stacker1_next_output):
        stacker1_pre_output = (
            stacker1_pre_output.unsqueeze(2)
            .repeat(1, 1, self.max_object, 1)
            .permute(0, 3, 1, 2)
        )
        stacker1_next_output = (
            stacker1_next_output.unsqueeze(1)
            .repeat(1, self.max_object, 1, 1)
            .permute(0, 3, 1, 2)
        )

        stacker1_pre_output = self.stacker2_bn(stacker1_pre_output.contiguous())
        stacker1_next_output = self.stacker2_bn(stacker1_next_output.contiguous())

        output = torch.cat([stacker1_pre_output, stacker1_next_output], 1)

        return output

    def forward_final(self, x, final_net):
        x = x.contiguous()
        for f in final_net:
            x = f(x)
        return x

    def add_unmatched_dim(self, x):
        if self.false_objects_column is None:
            self.false_objects_column = (
                Variable(torch.ones(x.shape[0], x.shape[1], x.shape[2], 1))
                * self.false_constant
            )
            if torch.cuda.is_available():
                self.false_objects_column = self.false_objects_column.cuda()
        x = torch.cat([x, self.false_objects_column], 3)

        if self.false_objects_row is None:
            self.false_objects_row = (
                Variable(torch.ones(x.shape[0], x.shape[1], 1, x.shape[3]))
                * self.false_constant
            )
            if torch.cuda.is_available():
                self.false_objects_row = self.false_objects_row.cuda()
        x = torch.cat([x, self.false_objects_row], 2)
        return x

    def loss(self, input, target, mask0, mask1):
        mask_pre = mask0[:, :, :]
        mask_next = mask1[:, :, :]
        mask0 = mask0.unsqueeze(3).repeat(1, 1, 1, self.max_object + 1)
        mask1 = mask1.unsqueeze(2).repeat(1, 1, self.max_object + 1, 1)
        mask0 = Variable(mask0.data)
        mask1 = Variable(mask1.data)
        target = Variable(target.byte().data)

        if torch.cuda.is_available():
            mask0 = mask0.cuda()
            mask1 = mask1.cuda()

        mask_region = (mask0 * mask1).float()
        mask_region_pre = mask_region.clone()
        mask_region_pre[:, :, self.max_object, :] = 0
        mask_region_next = mask_region.clone()
        mask_region_next[:, :, :, self.max_object] = 0
        mask_region_union = mask_region_pre * mask_region_next

        input_pre = nn.Softmax(dim=3)(mask_region_pre * input)
        input_next = nn.Softmax(dim=2)(mask_region_next * input)
        input_all = input_pre.clone()
        input_all[:, :, : self.max_object, : self.max_object] = (
            (input_pre + input_next) / 2.0
        )[:, :, : self.max_object, : self.max_object]
        target = target.float()
        target_pre = mask_region_pre * target
        target_next = mask_region_next * target
        target_union = mask_region_union * target
        target_num = target.sum()
        target_num_pre = target_pre.sum()
        target_num_next = target_next.sum()
        target_num_union = target_union.sum()
        if int(target_num_pre.data.item()):
            loss_pre = -(target_pre * torch.log(input_pre)).sum() / target_num_pre
        else:
            loss_pre = -(target_pre * torch.log(input_pre)).sum()
        if int(target_num_next.data.data.item()):
            loss_next = -(target_next * torch.log(input_next)).sum() / target_num_next
        else:
            loss_next = -(target_next * torch.log(input_next)).sum()
        if int(target_num_pre.data.item()) and int(target_num_next.data.item()):
            loss = -(target_pre * torch.log(input_all)).sum() / target_num_pre
        else:
            loss = -(target_pre * torch.log(input_all)).sum()

        if int(target_num_union.data.item()):
            loss_similarity = (
                target_union * (torch.abs((1 - input_pre) - (1 - input_next)))
            ).sum() / target_num
        else:
            loss_similarity = (
                target_union * (torch.abs((1 - input_pre) - (1 - input_next)))
            ).sum()

        _, indexes_ = target_pre.max(3)
        indexes_ = indexes_[:, :, :-1]
        _, indexes_pre = input_all.max(3)
        indexes_pre = indexes_pre[:, :, :-1]
        mask_pre_num = mask_pre[:, :, :-1].sum().data.item()
        if mask_pre_num:
            accuracy_pre = (
                indexes_pre[mask_pre[:, :, :-1]] == indexes_[mask_pre[:, :, :-1]]
            ).float().sum() / mask_pre_num
        else:
            accuracy_pre = (
                indexes_pre[mask_pre[:, :, :-1]] == indexes_[mask_pre[:, :, :-1]]
            ).float().sum() + 1

        _, indexes_ = target_next.max(2)
        indexes_ = indexes_[:, :, :-1]
        _, indexes_next = input_next.max(2)
        indexes_next = indexes_next[:, :, :-1]
        mask_next_num = mask_next[:, :, :-1].sum().data.item()
        if mask_next_num:
            accuracy_next = (
                indexes_next[mask_next[:, :, :-1]] == indexes_[mask_next[:, :, :-1]]
            ).float().sum() / mask_next_num
        else:
            accuracy_next = (
                indexes_next[mask_next[:, :, :-1]] == indexes_[mask_next[:, :, :-1]]
            ).float().sum() + 1

        return (
            loss_pre,
            loss_next,
            loss_similarity,
            (loss_pre + loss_next + loss + loss_similarity) / 4.0,
            accuracy_pre,
            accuracy_next,
            (accuracy_pre + accuracy_next) / 2.0,
            indexes_pre,
        )


def add_final(cfg, batch_normal=True):
    layers = []
    in_channels = int(cfg[0])
    layers += []
    for v in cfg[1:-2]:
        conv2d = nn.Conv2d(in_channels, v, kernel_size=1, stride=1)
        if batch_normal:
            layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
        else:
            layers += [conv2d, nn.ReLU(inplace=True)]
        in_channels = v
    for v in cfg[-2:]:
        conv2d = nn.Conv2d(in_channels, v, kernel_size=1, stride=1)
        layers += [conv2d, nn.ReLU(inplace=True)]
        in_channels = v

    return layers


def selector(selector_input_channel, selector_output_channel):

    selector_layers = []

    for k, v in enumerate(selector_input_channel):
        selector_layers += [
            nn.Conv2d(
                selector_input_channel[k],
                selector_output_channel[k],
                kernel_size=3,
                padding=1,
            )
        ]

    return selector_layers


def show_batch_circle_image(
    img_pre, img_next, boxes_pre, boxes_next, valid_pre, valid_next, indexes, opt
):
    batch_size = img_pre.shape[0]
    images = list()
    gap = 20 / 900
    i = 0
    img1 = img_pre[i, :].data
    img1 = img1.cpu().numpy()
    height, width, _ = img1.shape

    valid1 = valid_pre[i, 0, :-1].data.cpu().numpy()
    boxes1 = boxes_pre[i, :, 0, 0, :].data.cpu().numpy()
    boxes1 = boxes1[valid1 == 1]
    img1 = np.clip(img1, 0, 255).astype(np.uint8).copy()

    index = indexes[i, 0, :].data.cpu().numpy()[valid1 == 1]

    img2 = img_next[i, :].data
    img2 = img2.cpu().numpy()
    valid2 = valid_next[i, 0, :-1].data.cpu().numpy()
    boxes2 = boxes_next[i, :, 0, 0, :].data.cpu().numpy()
    boxes2 = boxes2[valid2 == 1]
    img2 = np.clip(img2, 0, 255).astype(np.uint8).copy()

    # draw all circle
    for b in boxes1:
        b[0] = (b[0] + 1) / 2.0 * width
        b[1] = (b[1] + 1) / 2.0 * height
        tup = tuple(b.astype(int))
        img1 = cv2.circle(img1, tup, 20, [0, 0, 255], thickness=3)

    for b in boxes2:
        b[0] = (b[0] + 1) / 2.0 * width
        b[1] = (b[1] + 1) / 2.0 * height
        tup = tuple(b.astype(int))
        img2 = cv2.circle(img2, tup, 20, [0, 0, 255], thickness=3)

    gap_pixel = int(gap * 900)
    H, W, C = img1.shape
    img = np.ones((2 * H + gap_pixel, W, C), dtype=np.uint8) * 255
    img[:H, :W, :] = img1
    img[gap_pixel + H :, :] = img2

    for j, b1 in enumerate(boxes1):
        if index[j] >= opt.max_object:
            continue

        color = tuple((np.random.rand(3) * 255).astype(int).tolist())

        start_pt = tuple(b1.astype(int))
        b2 = boxes_next[i, :, 0, 0, :].data.cpu().numpy()[index[j]]
        b2[0] = (b2[0] + 1) / 2.0 * width
        b2[1] = (b2[1] + 1) / 2.0 * height
        end_pt = tuple(b2.astype(int))

        end_pt = (end_pt[0], end_pt[1] + height + gap_pixel)
        img = cv2.circle(img, start_pt, 20, color, thickness=3)
        img = cv2.circle(img, end_pt, 20, color, thickness=3)
        img = cv2.line(img, start_pt, end_pt, color, thickness=3)

    img = img.astype(np.float)
    return img
