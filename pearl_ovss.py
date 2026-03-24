import logging
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.structures import PixelData
from mmseg.models.data_preprocessor import SegDataPreProcessor
from mmseg.models.segmentors import BaseSegmentor
from mmseg.registry import MODELS

import clip
from pearl.prop import TLP
from prompts.imagenet_template import openai_imagenet_template

sys.path.append("..")

@MODELS.register_module()
class PEARL(BaseSegmentor):
    def __init__(self, clip_path, name_path, device=torch.device('cuda'),
                 arch='reduced', attn_strategy='pearl',
                 prob_thd=0.0, logit_scale=40, slide_stride=112, slide_crop=224,
                 use_kk=True, alpha_kk=0.08, align_solver='polar', polar_iters=5,
                 use_prop="on", grid=80
                 ):

        data_preprocessor = SegDataPreProcessor(
            mean=[122.771, 116.746, 104.094],
            std=[68.501, 66.632, 70.323],
            rgb_to_bgr=True
        )
        super().__init__(data_preprocessor=data_preprocessor)

        self.net, _ = clip.load(clip_path, device=device, jit=False)

        query_words, self.query_idx = get_cls_idx(name_path)
        self.num_queries = len(query_words)
        self.query_idx = torch.tensor(self.query_idx, dtype=torch.int64, device=device)

        query_features = []
        with torch.no_grad():
            for qw in query_words:
                tok = clip.tokenize([temp(qw) for temp in openai_imagenet_template]).to(device)
                feat = self.net.encode_text(tok)
                feat = feat / feat.norm(dim=-1, keepdim=True)
                feat = feat.mean(dim=0)
                feat = feat / feat.norm()
                query_features.append(feat.unsqueeze(0))
        self.query_features = torch.cat(query_features, dim=0)
        self.dtype = self.query_features.dtype

        self.net.visual.set_params(arch, attn_strategy)

        self.logit_scale = logit_scale
        self.prob_thd = prob_thd
        self.slide_stride = slide_stride
        self.slide_crop = slide_crop
        self.align_corners = False

        # PA
        self.net.visual.set_attn_options(
            use_kk=use_kk, alpha_kk=alpha_kk,
            align_solver=align_solver, polar_iters=polar_iters
        )

        # TLP
        if use_prop == "on":
            self.prop = TLP(grid=grid).to(device)
            self.prop.bind_text(self.query_features)
        else:
            self.prop = None

        logging.info(f'attn_strategy is {attn_strategy}, arch is {arch}.')

    def forward_feature(self, img, logit_size=None):
        if isinstance(img, list):
            img = img[0]
        B, _, H, W = img.shape

        tokens_all = self.net.encode_image(img, return_all=True)   # [B, 1+N, Dv]
        patch_tokens = tokens_all[:, 1:, :]                        # [B, N, Dv]
        patch_tokens = F.normalize(patch_tokens, dim=-1)
        Dv = patch_tokens.size(-1)

        text = F.normalize(self.query_features, dim=-1)            # [C, Dt]
        C  = text.size(0)
        text_b = text.unsqueeze(0).expand(B, -1, -1)               # [B, C, Dt]

        logits_patch = torch.bmm(
            patch_tokens,                      # [B, N, Dv]
            text_b.transpose(1, 2)             # [B, Dt, C]
        )                                      # [B, N, C]

        patch_size = self.net.visual.patch_size
        h, w = H // patch_size, W // patch_size
        logits_map = logits_patch.permute(0, 2, 1).reshape(B, C, h, w)
        logits_map = F.interpolate(
            logits_map, size=(H, W), mode='bilinear', align_corners=self.align_corners
        )                                      # [B, C, H, W]
            
        return logits_map

    def forward_slide(self, img, stride=112, crop_size=224):
        if isinstance(img, list):
            img = img[0].unsqueeze(0)

        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(crop_size, int):
            crop_size = (crop_size, crop_size)

        h_stride, w_stride = stride
        h_crop, w_crop = crop_size
        batch_size, _, h_img, w_img = img.shape
        out_channels = self.num_queries

        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1

        preds = img.new_zeros((batch_size, out_channels, h_img, w_img))
        count_mat = img.new_zeros((batch_size, 1, h_img, w_img))

        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)

                crop_img = img[:, :, y1:y2, x1:x2]
                crop_seg_logit = self.forward_feature(crop_img)

                preds += nn.functional.pad(
                    crop_seg_logit,
                    (int(x1), int(preds.shape[3] - x2),
                     int(y1), int(preds.shape[2] - y2))
                )
                count_mat[:, :, y1:y2, x1:x2] += 1

        assert (count_mat == 0).sum() == 0

        logits = preds / count_mat  # [B,C,H_img,W_img]
        return logits

    def _forward_once(self, inputs):
        if self.slide_crop > 0:
            seg_logits = self.forward_slide(inputs, self.slide_stride, self.slide_crop)
        else:
            seg_logits = self.forward_feature(inputs)
        return seg_logits  # [B,C,H_in,W_in]

    def predict(self, inputs, data_samples):
        if data_samples is not None:
            batch_img_metas = [data_sample.metainfo for data_sample in data_samples]
        else:
            batch_img_metas = [dict(
                ori_shape=inputs.shape[2:],
                img_shape=inputs.shape[2:],
                pad_shape=inputs.shape[2:],
                padding_size=[0, 0, 0, 0])
            ] * inputs.shape[0]

        seg_logits = self._forward_once(inputs)  # [B,C,H,W]

        img_size = batch_img_metas[0]['ori_shape']
        seg_logits = nn.functional.interpolate(
            seg_logits,
            size=img_size,
            mode='bilinear',
            align_corners=self.align_corners
        )

        # TLP
        if self.prop:
            seg_logits = self.prop(
                F.interpolate(inputs, size=img_size, mode='bilinear', align_corners=self.align_corners),
                seg_logits.to(inputs.dtype)).to(self.dtype)

        return self.postprocess_result(seg_logits, data_samples)

    def postprocess_result(self, seg_logits, data_samples):
        batch_size = seg_logits.shape[0]
        for i in range(batch_size):
            seg_probs = torch.softmax(seg_logits[i] * self.logit_scale, dim=0)  # [C,H,W]

            num_cls, num_queries = max(self.query_idx) + 1, len(self.query_idx)
            if num_cls != num_queries:
                seg_probs = seg_probs.unsqueeze(0)  # [1,C,H,W]
                cls_index = nn.functional.one_hot(self.query_idx)  # [num_queries, num_cls]
                cls_index = cls_index.T.view(num_cls, num_queries, 1, 1)
                seg_probs = (seg_probs * cls_index).max(1)[0]  # [num_cls,H,W]

            seg_pred = seg_probs.argmax(0, keepdim=True)
            seg_pred[seg_probs.max(0, keepdim=True)[0] < self.prob_thd] = 0
            seg_probs /= seg_probs.sum(0, keepdim=True)

            data_samples[i].set_data({
                'seg_logits': PixelData(**{'data': seg_probs}),
                'pred_sem_seg': PixelData(**{'data': seg_pred})
            })

        return data_samples

    def _forward(data_samples):
        pass

    def inference(self, img, batch_img_metas):
        pass

    def encode_decode(self, inputs, batch_img_metas):
        pass

    def extract_feat(self, inputs):
        pass

    def loss(self, inputs, data_samples):
        pass

def get_cls_idx(path):
    with open(path, 'r') as f:
        name_sets = f.readlines()
    num_cls = len(name_sets)

    class_names, class_indices = list(), list()
    for idx in range(num_cls):
        names_i = name_sets[idx].split(', ')
        class_names += names_i
        class_indices += [idx for _ in range(len(names_i))]
    class_names = [item.replace('\n', '') for item in class_names]
    return class_names, class_indices
