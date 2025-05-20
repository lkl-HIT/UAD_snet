import copy
from typing import List

import numpy as np
import scipy.ndimage as ndimage
import torch
import torch.nn.functional as F


class _BaseMerger:
    def __init__(self):
        """Merges feature embedding by name."""

    def merge(self, features: list):
        features = [self._reduce(feature) for feature in features]
        return np.concatenate(features, axis=1)


class AverageMerger(_BaseMerger):
    @staticmethod
    def _reduce(features):
        # NxCxWxH -> NxC
        return features.reshape([features.shape[0], features.shape[1], -1]).mean(
            axis=-1
        )


class ConcatMerger(_BaseMerger):
    @staticmethod
    def _reduce(features):
        # NxCxWxH -> NxCWH
        return features.reshape(len(features), -1)


class Preprocessing(torch.nn.Module):
    def __init__(self, input_dims, output_dim):
        super(Preprocessing, self).__init__()
        self.input_dims = input_dims
        self.output_dim = output_dim

        self.preprocessing_modules = torch.nn.ModuleList()
        for input_dim in input_dims:
            module = MeanMapper(output_dim)
            self.preprocessing_modules.append(module)

    def forward(self, features):
        _features = []
        for module, feature in zip(self.preprocessing_modules, features):
            _features.append(module(feature))
        return torch.stack(_features, dim=1)


class MeanMapper(torch.nn.Module):
    def __init__(self, preprocessing_dim):
        super(MeanMapper, self).__init__()
        self.preprocessing_dim = preprocessing_dim

    def forward(self, features):
        features = features.reshape(len(features), 1, -1)
        return F.adaptive_avg_pool1d(features, self.preprocessing_dim).squeeze(1)


class Aggregator(torch.nn.Module):
    def __init__(self, target_dim):
        super(Aggregator, self).__init__()
        self.target_dim = target_dim

    def forward(self, features):
        """Returns reshaped and average pooled features."""
        # batchsize x number_of_layers x input_dim -> batchsize x target_dim
        features = features.reshape(len(features), 1, -1)
        features = F.adaptive_avg_pool1d(features, self.target_dim)
        return features.reshape(len(features), -1)


class RescaleSegmentor:
    def __init__(self, device, target_size=224):
        self.device = device
        self.target_size = target_size
        self.smoothing = 4

    # def convert_to_segmentation(self, patch_scores, features):

    #     with torch.no_grad():
    #         if isinstance(patch_scores, np.ndarray):
    #             patch_scores = torch.from_numpy(patch_scores)
    #         _scores = patch_scores.to(self.device)
    #         _scores = _scores.unsqueeze(1)
    #         _scores = F.interpolate(
    #             _scores, size=self.target_size, mode="bilinear", align_corners=False
    #         )
    #         _scores = _scores.squeeze(1)
    #         patch_scores = _scores.cpu().numpy()

    #         if isinstance(features, np.ndarray):
    #             features = torch.from_numpy(features)
    #         features = features.to(self.device).permute(0, 3, 1, 2)
    #         if self.target_size[0] * self.target_size[1] * features.shape[0] * features.shape[1] >= 2**31:
    #             subbatch_size = int((2**31-1) / (self.target_size[0] * self.target_size[1] * features.shape[1]))
    #             interpolated_features = []
    #             for i_subbatch in range(int(features.shape[0] / subbatch_size + 1)):
    #                 subfeatures = features[i_subbatch*subbatch_size:(i_subbatch+1)*subbatch_size]
    #                 subfeatures = subfeatures.unsuqeeze(0) if len(subfeatures.shape) == 3 else subfeatures
    #                 subfeatures = F.interpolate(
    #                     subfeatures, size=self.target_size, mode="bilinear", align_corners=False
    #                 )
    #                 interpolated_features.append(subfeatures)
    #             features = torch.cat(interpolated_features, 0)
    #         else:
    #             features = F.interpolate(
    #                 features, size=self.target_size, mode="bilinear", align_corners=False
    #             )
    #         features = features.cpu().numpy()

    #     return [
    #         ndimage.gaussian_filter(patch_score, sigma=self.smoothing)
    #         for patch_score in patch_scores
    #     ], [ 
    #         feature
    #         for feature in features
    #     ]

#采用pytorch实现高斯模糊而并非numpy方法
    def gaussian_kernel(self, kernel_size, sigma):
        """生成高斯核"""
        x_coord = torch.arange(kernel_size, dtype=torch.float32, device=self.device)
        x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
        y_grid = x_grid.t()
        xy_grid = torch.stack([x_grid, y_grid], dim=-1)

        mean = (kernel_size - 1) / 2.
        variance = sigma ** 2.

        # 计算高斯核
        gaussian_kernel = torch.exp(
            -torch.sum((xy_grid - mean) ** 2., dim=-1) / (2 * variance)
        )

        # 归一化
        gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()

        # 调整形状以用于卷积
        gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
        return gaussian_kernel

    def gaussian_blur(self, x, kernel_size, sigma):
        """应用高斯模糊"""
        kernel = self.gaussian_kernel(kernel_size, sigma)
        padding = kernel_size // 2
        
        return F.conv2d(x, kernel, padding=padding, groups=x.size(1))

    def convert_to_segmentation(self, patch_scores, features = None):
        with torch.no_grad():
            # 转换 patch_scores 为张量
            # if isinstance(patch_scores, np.ndarray):
            #     patch_scores = torch.from_numpy(patch_scores)
            # patch_scores = patch_scores.to(self.device)

            # 对 patch scores 进行插值
            patch_scores = F.interpolate(
                patch_scores.unsqueeze(1), size=self.target_size, mode="bilinear", align_corners=False
            )

            # 应用高斯模糊
            kernel_size = int(2 * (self.smoothing * 3) + 1)  # 3-sigma rule
            patch_scores = self.gaussian_blur(patch_scores, kernel_size, self.smoothing).squeeze(1)

            # # 处理特征
            # # if isinstance(features, np.ndarray):
            # #     features = torch.from_numpy(features)
            # features = features.to(self.device).permute(0, 3, 1, 2)

            # # 处理大型特征张量
            # if self.target_size[0] * self.target_size[1] * features.shape[0] * features.shape[1] >= 2**31:
            #     subbatch_size = int((2**31-1) / (self.target_size[0] * self.target_size[1] * features.shape[1]))
            #     interpolated_features = []
            #     for i_subbatch in range(int(features.shape[0] / subbatch_size + 1)):
            #         subfeatures = features[i_subbatch*subbatch_size:(i_subbatch+1)*subbatch_size]
            #         subfeatures = subfeatures.unsqueeze(0) if len(subfeatures.shape) == 3 else subfeatures
            #         subfeatures = F.interpolate(
            #             subfeatures, size=self.target_size, mode="bilinear", align_corners=False
            #         )
            #         interpolated_features.append(subfeatures)
            #     features = torch.cat(interpolated_features, 0)
            # else:
            #     features = F.interpolate(
            #         features, size=self.target_size, mode="bilinear", align_corners=False
            #     )

        return patch_scores

        # return patch_scores, features


class NetworkFeatureAggregator(torch.nn.Module):
    """Efficient extraction of network features."""

    def __init__(self, backbone, layers_to_extract_from, device, train_backbone=False):
        super(NetworkFeatureAggregator, self).__init__()
        """Extraction of network features.

        Runs a network only to the last layer of the list of layers where
        network features should be extracted from.

        Args:
            backbone: torchvision.model
            layers_to_extract_from: [list of str]
        """
        self.layers_to_extract_from = layers_to_extract_from
        self.backbone = backbone
        self.device = device
        self.train_backbone = train_backbone
        if not hasattr(backbone, "hook_handles"):
            self.backbone.hook_handles = []
        for handle in self.backbone.hook_handles:
            handle.remove()
        self.outputs = {}

        for extract_layer in layers_to_extract_from:
            forward_hook = ForwardHook(
                self.outputs, extract_layer, layers_to_extract_from[-1]
            )
            if "." in extract_layer:
                extract_block, extract_idx = extract_layer.split(".")
                network_layer = backbone.__dict__["_modules"][extract_block]
                if extract_idx.isnumeric():
                    extract_idx = int(extract_idx)
                    network_layer = network_layer[extract_idx]
                else:
                    network_layer = network_layer.__dict__["_modules"][extract_idx]
            else:
                network_layer = backbone.__dict__["_modules"][extract_layer]

            if isinstance(network_layer, torch.nn.Sequential):
                self.backbone.hook_handles.append(
                    network_layer[-1].register_forward_hook(forward_hook)
                )
            else:
                self.backbone.hook_handles.append(
                    network_layer.register_forward_hook(forward_hook)
                )
        self.to(self.device)

    def forward(self, images, eval=True):
        self.outputs.clear()
        if self.train_backbone and not eval:
            self.backbone(images)
        else:
            with torch.no_grad():
                # The backbone will throw an Exception once it reached the last
                # layer to compute features from. Computation will stop there.
                try:
                    _ = self.backbone(images)
                except LastLayerToExtractReachedException:
                    pass
        return self.outputs

    def feature_dimensions(self, input_shape):
        """Computes the feature dimensions for all layers given input_shape."""
        _input = torch.ones([1] + list(input_shape)).to(self.device)
        _output = self(_input)
        return [_output[layer].shape[1] for layer in self.layers_to_extract_from]


class ForwardHook:
    def __init__(self, hook_dict, layer_name: str, last_layer_to_extract: str):
        self.hook_dict = hook_dict
        self.layer_name = layer_name
        self.raise_exception_to_break = copy.deepcopy(
            layer_name == last_layer_to_extract
        )

    def __call__(self, module, input, output):
        self.hook_dict[self.layer_name] = output
        # if self.raise_exception_to_break:
        #     raise LastLayerToExtractReachedException()
        return None


class LastLayerToExtractReachedException(Exception):
    pass


