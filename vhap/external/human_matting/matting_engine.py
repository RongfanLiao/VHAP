import os
import torch
import inspect
import warnings
import torchvision
from .stylematte import StyleMatte

class StyleMatteEngine(torch.nn.Module):
    def __init__(self, device='cpu',human_matting_path='./model_zoo/flame_tracking_models/matting/stylematte_synth.pt'):
        super().__init__()
        self._device = device
        self.normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self._init_models(human_matting_path)

    def _init_models(self,_ckpt_path):
        # load dict
        state_dict = torch.load(_ckpt_path, map_location='cpu')
        # build model
        model = StyleMatte()
        model.load_state_dict(state_dict)
        self.model = model.to(self._device).eval()
    
    @torch.no_grad()
    def forward(self, input_image, return_type='matting', background_rgb=1.0):
        if not hasattr(self, 'model'):
            self._init_models()
        # Accept both (3, H, W) and (N, 3, H, W)
        squeeze_output = False
        if input_image.dim() == 3:
            input_image = input_image.unsqueeze(0)
            squeeze_output = True
        if input_image.max() > 2.0:
            warnings.warn('Image should be normalized to [0, 1].')
        _, _, ori_h, ori_w = input_image.shape
        input_image = input_image.to(self._device).float()
        image = input_image.clone()
        # resize
        if max(ori_h, ori_w) > 1024:
            scale = 1024.0 / max(ori_h, ori_w)
            resized_h, resized_w = int(ori_h * scale), int(ori_w * scale)
            image = torchvision.transforms.functional.resize(image, (resized_h, resized_w), antialias=True)
        else:
            resized_h, resized_w = ori_h, ori_w
        # padding
        if resized_h % 8 != 0 or resized_w % 8 != 0:
            image = torchvision.transforms.functional.pad(image, ((8-resized_w % 8)%8, (8-resized_h % 8)%8, 0, 0, ), padding_mode='reflect')
        # normalize and forwarding
        image = torch.stack([self.normalize(img) for img in image])
        predict = self.model(image)
        # undo padding
        predict = predict[:, :, -resized_h:, -resized_w:]
        # undo resize
        if resized_h != ori_h or resized_w != ori_w:
            predict = torchvision.transforms.functional.resize(predict, (ori_h, ori_w), antialias=True)

        if return_type == 'alpha':
            # predict shape: (N, 1, H, W) -> (N, H, W) or (H, W)
            result = predict[:, 0]
            return result[0] if squeeze_output else result
        elif return_type == 'matting':
            predict = predict.expand(-1, 3, -1, -1)
            matting_image = input_image.clone()
            background = matting_image.new_ones(matting_image.shape) * background_rgb
            matting_image = matting_image * predict + (1-predict) * background
            alpha = predict[:, 0]
            if squeeze_output:
                return matting_image[0], alpha[0]
            return matting_image, alpha
        elif return_type == 'all':
            predict = predict.expand(-1, 3, -1, -1)
            background = input_image.new_ones(input_image.shape) * background_rgb
            foreground_image = input_image * predict + (1-predict) * background
            background_image = input_image * (1-predict) + predict * background
            if squeeze_output:
                return foreground_image[0], background_image[0]
            return foreground_image, background_image
        else:
            raise NotImplementedError
