# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved. 
#   
# Licensed under the Apache License, Version 2.0 (the "License");   
# you may not use this file except in compliance with the License.  
# You may obtain a copy of the License at   
#   
#     http://www.apache.org/licenses/LICENSE-2.0    
#   
# Unless required by applicable law or agreed to in writing, software   
# distributed under the License is distributed on an "AS IS" BASIS, 
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
# See the License for the specific language governing permissions and   
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import json
import numpy as np
import paddle
from ppdet.core.workspace import register, create
from tools.x2coco import get_bbox
from .meta_arch import BaseArch

__all__ = ['SplitRCNN']


@register
class SplitRCNN(BaseArch):
    """
    Cascade R-CNN network, see https://arxiv.org/abs/1712.00726

    Args:
        backbone (object): backbone instance
        rpn_head (object): `RPNHead` instance
        bbox_head (object): `BBoxHead` instance
        bbox_post_process (object): `BBoxPostProcess` instance
        neck (object): 'FPN' instance
        mask_head (object): `MaskHead` instance
        mask_post_process (object): `MaskPostProcess` instance
    """
    __category__ = 'architecture'
    __inject__ = [
        'bbox_post_process',
        'mask_post_process',
    ]

    def __init__(self,
                 backbone,
                 rpn_head,
                 bbox_head,
                 bbox_post_process,
                 neck=None,
                 mask_head=None,
                 mask_post_process=None,
                 multi_dataset=None):
        super(SplitRCNN, self).__init__()
        self.backbone = backbone
        self.rpn_head = rpn_head
        self.bbox_head = bbox_head
        self.bbox_post_process = bbox_post_process
        self.neck = neck
        self.mask_head = mask_head
        self.mask_post_process = mask_post_process
        self.with_mask = mask_head is not None
        
        self.unified_eval = multi_dataset['unified_eval']
        self.datasets = multi_dataset['datasets']
        self.num_datasets = len(self.datasets)
        self.dataset_name_to_id = {k: i for i, k in enumerate(self.datasets)}
        self.cpu_post_process = False # due to memory issue on mask
        self.eval_datatset=-1
        # label_map = json.load(
        #     open(multi_dataset['unified_label_file'], 'r'))['label_map']
        # self.label_map = {
        #     self.datasets.index(d): paddle.to_tensor(x) \
        #     for d, x in label_map.items() if d in self.datasets} 
    @classmethod
    def from_config(cls, cfg, *args, **kwargs):
        backbone = create(cfg['backbone'])
        kwargs = {'input_shape': backbone.out_shape}
        neck = cfg['neck'] and create(cfg['neck'], **kwargs)

        out_shape = neck and neck.out_shape or backbone.out_shape
        kwargs = {'input_shape': out_shape}
        rpn_head = create(cfg['rpn_head'], **kwargs)
        bbox_head = create(cfg['bbox_head'], **kwargs)

        out_shape = neck and out_shape or bbox_head.get_head().out_shape
        kwargs = {'input_shape': out_shape}
        mask_head = cfg['mask_head'] and create(cfg['mask_head'], **kwargs)
        return {
            'backbone': backbone,
            'neck': neck,
            "rpn_head": rpn_head,
            "bbox_head": bbox_head,
            "mask_head": mask_head,
        }

    def _forward(self):
        '''
        self.inputs['gt_class'][0]=paddle.to_tensor([[23],[23]])
        self.inputs['gt_bbox'][0]=paddle.to_tensor([[545.6920, 130.3771, 902.8917, 487.6388],[228.5795, 377.1871, 563.8398, 510.1152]])
        self.inputs['dataset_source'][0]=paddle.to_tensor([1])
        if self.training:
            for i in range(len(self.inputs['gt_class'])):
                dataset_source = self.inputs['dataset_source'][i]
                # self.inputs['gt_class'][i] = \
                #     self.label_map[int(dataset_source)][self.inputs['gt_class'][i]]
        self.inputs['image']=paddle.to_tensor(np.load("image_test.npy"))
        self.inputs['im_shape']=paddle.to_tensor([[736., 1103]])
        '''
        #'''
        if self.training:
            for i in range(len(self.inputs['gt_class'])):
                dataset_source = self.inputs['dataset_source'][i]
        #'''
        body_feats = self.backbone(self.inputs)
        if self.neck is not None:
            body_feats = self.neck(body_feats)
        # print('backbone_time:{}'.format((time.time()-model_before)))

        if self.training:
            rois, rois_num, rpn_loss = self.rpn_head(body_feats, self.inputs)
            # print('rpn_time:{}'.format((time.time()-model_before)))
            bbox_loss, bbox_feat = self.bbox_head(body_feats, rois, rois_num,
                                                  self.inputs)
            # print('head_time:{}'.format((time.time()-model_before)))
            return rpn_loss, bbox_loss, {}
        else:
            rois, rois_num, _ = self.rpn_head(body_feats, self.inputs)
            preds, _ = self.bbox_head(body_feats, rois, rois_num, self.inputs,self.eval_dataset)
            refined_rois = self.bbox_head.get_refined_rois()

            im_shape = self.inputs['im_shape']
            scale_factor = self.inputs['scale_factor']
            bbox, bbox_num, nms_keep_idx = self.bbox_post_process(
                preds, (refined_rois, rois_num), im_shape, scale_factor,self.eval_dataset)
            # rescale the prediction back to origin image
            bbox, bbox_pred, bbox_num = self.bbox_post_process.get_pred(
                bbox, bbox_num, im_shape, scale_factor)
            return bbox_pred, bbox_num, None

    def get_loss(self, ):
        rpn_loss, bbox_loss, mask_loss = self._forward()
        loss = {}
        loss.update(rpn_loss)
        loss.update(bbox_loss)
        if self.with_mask:
            loss.update(mask_loss)
        total_loss = paddle.add_n(list(loss.values()))
        loss.update({'loss': total_loss})
        return loss

    def get_pred(self):
        bbox_pred, bbox_num, mask_pred = self._forward()
        output = {
            'bbox': bbox_pred,
            'bbox_num': bbox_num,
        }
        if self.with_mask:
            output.update({'mask': mask_pred})
        return output
    
    
    def set_eval_dataset(self, dataset_name):
        meta_datase_name = dataset_name[:dataset_name.find('_')]
        if self.unified_eval:
            self.eval_dataset = -1
        else:
            self.eval_dataset = \
                self.dataset_name_to_id[meta_datase_name]