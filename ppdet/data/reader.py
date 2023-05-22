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

import copy
import os
import traceback
import six
import sys
if sys.version_info >= (3, 0):
    pass
else:
    pass
import numpy as np
import paddle
import paddle.nn.functional as F

from copy import deepcopy

from paddle.io import DataLoader,DistributedBatchSampler
from .utils import default_collate_fn
import itertools
import math
from collections import defaultdict
from typing import Optional
from ppdet.core.workspace import register
from . import transform
from .shm_utils import _get_shared_memory_size_in_M
from paddle.io import DataLoader, BatchSampler
from ppdet.utils.logger import setup_logger
paddle.set_printoptions(precision=8)
logger = setup_logger('reader')

MAIN_PID = os.getpid()


class Compose(object):
    def __init__(self, transforms, num_classes=80):
        self.transforms = transforms
        self.transforms_cls = []
        for t in self.transforms:
            for k, v in t.items():
                op_cls = getattr(transform, k)
                f = op_cls(**v)
                if hasattr(f, 'num_classes'):
                    f.num_classes = num_classes

                self.transforms_cls.append(f)

    def __call__(self, data):
        for f in self.transforms_cls:
            try:
                data = f(data)
            except Exception as e:
                stack_info = traceback.format_exc()
                logger.warning("fail to map sample transform [{}] "
                               "with error: {} and stack:\n{}".format(
                                   f, e, str(stack_info)))
                raise e

        return data


class BatchCompose(Compose):
    def __init__(self, transforms, num_classes=80, collate_batch=True):
        super(BatchCompose, self).__init__(transforms, num_classes)
        self.collate_batch = collate_batch

    def __call__(self, data):
        for f in self.transforms_cls:
            try:
                data = f(data)
            except Exception as e:
                stack_info = traceback.format_exc()
                logger.warning("fail to map batch transform [{}] "
                               "with error: {} and stack:\n{}".format(
                                   f, e, str(stack_info)))
                raise e

        # remove keys which is not needed by model
        extra_key = ['h', 'w', 'flipped']
        for k in extra_key:
            for sample in data:
                if k in sample:
                    sample.pop(k)

        # batch data, if user-define batch function needed
        # use user-defined here
        if self.collate_batch:
            batch_data = default_collate_fn(data)
        else:
            batch_data = {}
            for k in data[0].keys():
                tmp_data = []
                for i in range(len(data)):
                    # if k=='im_id' and type(data[0]['im_id'][0])==np.str_:
                    #     tmp_data.append(str(data[i]['im_id'][0]))
                    # else:
                        tmp_data.append(data[i][k])
                # if not 'gt_' in k and not 'is_crowd' in k and not 'difficult' in k and not 'im_id' in k:
                if not 'gt_' in k and not 'is_crowd' in k and not 'difficult' in k:
                    tmp_data = np.stack(tmp_data, axis=0)
                batch_data[k] = tmp_data
        return batch_data


class BaseDataLoader_u(object):
    """
    Base DataLoader implementation for detection models

    Args:
        sample_transforms (list): a list of transforms to perform
                                  on each sample
        batch_transforms (list): a list of transforms to perform
                                 on batch
        batch_size (int): batch size for batch collating, default 1.
        shuffle (bool): whether to shuffle samples
        drop_last (bool): whether to drop the last incomplete,
                          default False
        num_classes (int): class number of dataset, default 80
        collate_batch (bool): whether to collate batch in dataloader.
            If set to True, the samples will collate into batch according
            to the batch size. Otherwise, the ground-truth will not collate,
            which is used when the number of ground-truch is different in 
            samples.
        use_shared_memory (bool): whether to use shared memory to
                accelerate data loading, enable this only if you
                are sure that the shared memory size of your OS
                is larger than memory cost of input datas of model.
                Note that shared memory will be automatically
                disabled if the shared memory of OS is less than
                1G, which is not enough for detection models.
                Default False.
    """

    def __init__(self,
                sample_transforms=[],
                batch_transforms=[],
                batch_size=1,
                shuffle=False,
                drop_last=False,
                num_classes=80,
                collate_batch=True,
                use_shared_memory=False,
                sample_epoch_size=1600,
                dataset_ratio=[1,1,1],
                muti_dataset_use_cas=[False,True,True],
                **kwargs):
        self.sample_epoch_size=sample_epoch_size
        self.dataset_ratio=dataset_ratio
        self.muti_dataset_use_cas=muti_dataset_use_cas
        # sample transform
        self._sample_transforms = Compose(
            sample_transforms, num_classes=num_classes)

        # batch transfrom 
        self._batch_transforms = BatchCompose(batch_transforms, num_classes,
                                              collate_batch)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.use_shared_memory = use_shared_memory
        self.kwargs = kwargs
        
    def __call__(self,
                dataset_coco=None,
                dataset_obj365=None,
                dataset_oid=None,
                dataset_voc=None,
                worker_num=None,
                batch_sampler=None,
                return_list=False,
                 ):
        dataset_names=[]
        self.datasets=[]
  
        if dataset_coco is not None:
            self.dataset_coco = dataset_coco
            self.dataset_coco.check_or_download_dataset()
            self.dataset_coco.parse_dataset()
            self.dataset_coco.set_transform(self._sample_transforms)
            self.dataset_coco.set_kwargs(**self.kwargs)
            dataset_names.append('coco')
            self.datasets.append(self.dataset_coco)
        if dataset_obj365 is not None:
            self.dataset_obj365 = dataset_obj365
            self.dataset_obj365.check_or_download_dataset()
            self.dataset_obj365.parse_dataset()
            self.dataset_obj365.set_transform(self._sample_transforms)
            self.dataset_obj365.set_kwargs(**self.kwargs)
            dataset_names.append('obj365')
            self.datasets.append(self.dataset_obj365)
        if dataset_oid is not None:
            self.dataset_oid = dataset_oid    
            self.dataset_oid.check_or_download_dataset()
            self.dataset_oid.parse_dataset()
            self.dataset_oid.set_transform(self._sample_transforms)
            self.dataset_oid.set_kwargs(**self.kwargs)
            dataset_names.append('oid')
            self.datasets.append(self.dataset_oid)
        if dataset_voc is not None:
            self.dataset_voc=dataset_voc   
            self.dataset_voc.check_or_download_dataset()
            self.dataset_voc.parse_dataset()
            self.dataset_voc.set_transform(self._sample_transforms)
            self.dataset_voc.set_kwargs(**self.kwargs)
            dataset_names.append('voc')
            self.datasets.append(self.dataset_voc)
        # batch sampler
        self.dataset_dicts = [dataset.roidbs for dataset in self.datasets]
        for source_id, (dataset_name, dicts) in \
            enumerate(zip(dataset_names, self.dataset_dicts)):
            assert len(dicts), "Dataset '{}' is empty!".format(dataset_name)
            for d in dicts:
                d['dataset_source'] = source_id
        import itertools
        dataset_dicts = list(itertools.chain.from_iterable(self.dataset_dicts))
        sizes = [0 for _ in range(len(dataset_names))]       
        for d in dataset_dicts:
            sizes[d['dataset_source']] += 1
        import copy
        
        self.dataset=copy.deepcopy(self.dataset_coco)
        self.dataset.roidbs=dataset_dicts
        if batch_sampler is None:
            self._batch_sampler = DistributedBatchSampler_u(
                self.dataset,
                sizes,
                batch_size=self.batch_size,
                shuffle=self.shuffle,
                drop_last=self.drop_last,
                sample_epoch_size=self.sample_epoch_size,
                dataset_ratio=self.dataset_ratio,
                muti_dataset_use_cas=self.muti_dataset_use_cas
                )
        else:
            self._batch_sampler = batch_sampler

        # DataLoader do not start sub-process in Windows and Mac
        # system, do not need to use shared memory
        use_shared_memory = self.use_shared_memory and \
                            sys.platform not in ['win32', 'darwin']
        # check whether shared memory size is bigger than 1G(1024M)
        if use_shared_memory:
            shm_size = _get_shared_memory_size_in_M()
            if shm_size is not None and shm_size < 1024.:
                logger.warning("Shared memory size is less than 1G, "
                               "disable shared_memory in DataLoader")
                use_shared_memory = False

        self.dataloader = DataLoader(
            dataset=self.dataset,
            batch_sampler=self._batch_sampler,
            collate_fn=self._batch_transforms,
            num_workers=worker_num,
            return_list=return_list,
            use_shared_memory=use_shared_memory)
        self.loader = iter(self.dataloader)

        return self

    def __len__(self):
        return len(self._batch_sampler)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return next(self.loader)
        except StopIteration:
            self.loader = iter(self.dataloader)
            six.reraise(*sys.exc_info())

    def next(self):
        # python2 compatibility
        return self.__next__()

class BaseDataLoader(object):
    """
    Base DataLoader implementation for detection models
    Args:
        sample_transforms (list): a list of transforms to perform
                                  on each sample
        batch_transforms (list): a list of transforms to perform
                                 on batch
        batch_size (int): batch size for batch collating, default 1.
        shuffle (bool): whether to shuffle samples
        drop_last (bool): whether to drop the last incomplete,
                          default False
        num_classes (int): class number of dataset, default 80
        collate_batch (bool): whether to collate batch in dataloader.
            If set to True, the samples will collate into batch according
            to the batch size. Otherwise, the ground-truth will not collate,
            which is used when the number of ground-truch is different in 
            samples.
        use_shared_memory (bool): whether to use shared memory to
                accelerate data loading, enable this only if you
                are sure that the shared memory size of your OS
                is larger than memory cost of input datas of model.
                Note that shared memory will be automatically
                disabled if the shared memory of OS is less than
                1G, which is not enough for detection models.
                Default False.
    """

    def __init__(self,
                 sample_transforms=[],
                 batch_transforms=[],
                 batch_size=1,
                 shuffle=False,
                 drop_last=False,
                 num_classes=80,
                 collate_batch=True,
                 use_shared_memory=False,
                 **kwargs):
        # sample transform
        self._sample_transforms = Compose(
            sample_transforms, num_classes=num_classes)

        # batch transfrom 
        self._batch_transforms = BatchCompose(batch_transforms, num_classes,
                                              collate_batch)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.use_shared_memory = use_shared_memory
        self.kwargs = kwargs

    def __call__(self,
                 dataset,
                 worker_num,
                 batch_sampler=None,
                 return_list=False):
        self.dataset = dataset
        self.dataset.check_or_download_dataset()
        self.dataset.parse_dataset()
        # get data
        self.dataset.set_transform(self._sample_transforms)
        # set kwargs
        self.dataset.set_kwargs(**self.kwargs)
        # batch sampler
        if batch_sampler is None:
            self._batch_sampler = DistributedBatchSampler(
                self.dataset,
                batch_size=self.batch_size,
                shuffle=self.shuffle,
                drop_last=self.drop_last)
        else:
            self._batch_sampler = batch_sampler

        # DataLoader do not start sub-process in Windows and Mac
        # system, do not need to use shared memory
        use_shared_memory = self.use_shared_memory and \
                            sys.platform not in ['win32', 'darwin']
        # check whether shared memory size is bigger than 1G(1024M)
        if use_shared_memory:
            shm_size = _get_shared_memory_size_in_M()
            if shm_size is not None and shm_size < 1024.:
                logger.warning("Shared memory size is less than 1G, "
                               "disable shared_memory in DataLoader")
                use_shared_memory = False

        self.dataloader = DataLoader(
            dataset=self.dataset,
            batch_sampler=self._batch_sampler,
            collate_fn=self._batch_transforms,
            num_workers=worker_num,
            return_list=return_list,
            use_shared_memory=use_shared_memory)
        self.loader = iter(self.dataloader)

        return self

    def __len__(self):
        return len(self._batch_sampler)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return next(self.loader)
        except StopIteration:
            self.loader = iter(self.dataloader)
            six.reraise(*sys.exc_info())

    def next(self):
        # python2 compatibility
        return self.__next__()
@register
class TrainReader(BaseDataLoader):
    __shared__ = ['num_classes']

    def __init__(self,
                 sample_transforms=[],
                 batch_transforms=[],
                 batch_size=1,
                 shuffle=True,
                 drop_last=True,
                 num_classes=80,
                 collate_batch=True,
                 **kwargs):
        super(TrainReader, self).__init__(sample_transforms, batch_transforms,
                                          batch_size, shuffle, drop_last,
                                          num_classes, collate_batch, **kwargs)

@register
class TrainReader_u(BaseDataLoader_u):
    __shared__ = ['num_classes']

    def __init__(self,
                 sample_transforms=[],
                 batch_transforms=[],
                 batch_size=1,
                 shuffle=True,
                 drop_last=True,
                 num_classes=80,
                 collate_batch=True,
                 **kwargs):
        super(TrainReader_u, self).__init__(sample_transforms, batch_transforms,
                                          batch_size, shuffle, drop_last,
                                          num_classes, collate_batch, **kwargs)
@register
class EvalReader(BaseDataLoader):
    __shared__ = ['num_classes']

    def __init__(self,
                 sample_transforms=[],
                 batch_transforms=[],
                 batch_size=1,
                 shuffle=False,
                 drop_last=False,
                 num_classes=80,
                 **kwargs):
        super(EvalReader, self).__init__(sample_transforms, batch_transforms,
                                         batch_size, shuffle, drop_last,
                                         num_classes, **kwargs)


@register
class TestReader(BaseDataLoader):
    __shared__ = ['num_classes']

    def __init__(self,
                 sample_transforms=[],
                 batch_transforms=[],
                 batch_size=1,
                 shuffle=False,
                 drop_last=False,
                 num_classes=80,
                 **kwargs):
        super(TestReader, self).__init__(sample_transforms, batch_transforms,
                                         batch_size, shuffle, drop_last,
                                         num_classes, **kwargs)


@register
class EvalMOTReader(BaseDataLoader):
    __shared__ = ['num_classes']

    def __init__(self,
                 sample_transforms=[],
                 batch_transforms=[],
                 batch_size=1,
                 shuffle=False,
                 drop_last=False,
                 num_classes=1,
                 **kwargs):
        super(EvalMOTReader, self).__init__(sample_transforms, batch_transforms,
                                            batch_size, shuffle, drop_last,
                                            num_classes, **kwargs)


@register
class TestMOTReader(BaseDataLoader):
    __shared__ = ['num_classes']

    def __init__(self,
                 sample_transforms=[],
                 batch_transforms=[],
                 batch_size=1,
                 shuffle=False,
                 drop_last=False,
                 num_classes=1,
                 **kwargs):
        super(TestMOTReader, self).__init__(sample_transforms, batch_transforms,
                                            batch_size, shuffle, drop_last,
                                            num_classes, **kwargs)


# For Semi-Supervised Object Detection (SSOD)
class Compose_SSOD(object):
    def __init__(self, base_transforms, weak_aug, strong_aug, num_classes=80):
        self.base_transforms = base_transforms
        self.base_transforms_cls = []
        for t in self.base_transforms:
            for k, v in t.items():
                op_cls = getattr(transform, k)
                f = op_cls(**v)
                if hasattr(f, 'num_classes'):
                    f.num_classes = num_classes
                self.base_transforms_cls.append(f)

        self.weak_augs = weak_aug
        self.weak_augs_cls = []
        for t in self.weak_augs:
            for k, v in t.items():
                op_cls = getattr(transform, k)
                f = op_cls(**v)
                if hasattr(f, 'num_classes'):
                    f.num_classes = num_classes
                self.weak_augs_cls.append(f)

        self.strong_augs = strong_aug
        self.strong_augs_cls = []
        for t in self.strong_augs:
            for k, v in t.items():
                op_cls = getattr(transform, k)
                f = op_cls(**v)
                if hasattr(f, 'num_classes'):
                    f.num_classes = num_classes
                self.strong_augs_cls.append(f)

    def __call__(self, data):
        for f in self.base_transforms_cls:
            try:
                data = f(data)
            except Exception as e:
                stack_info = traceback.format_exc()
                logger.warning("fail to map sample transform [{}] "
                               "with error: {} and stack:\n{}".format(
                                   f, e, str(stack_info)))
                raise e

        weak_data = deepcopy(data)
        strong_data = deepcopy(data)
        for f in self.weak_augs_cls:
            try:
                weak_data = f(weak_data)
            except Exception as e:
                stack_info = traceback.format_exc()
                logger.warning("fail to map weak aug [{}] "
                               "with error: {} and stack:\n{}".format(
                                   f, e, str(stack_info)))
                raise e

        for f in self.strong_augs_cls:
            try:
                strong_data = f(strong_data)
            except Exception as e:
                stack_info = traceback.format_exc()
                logger.warning("fail to map strong aug [{}] "
                               "with error: {} and stack:\n{}".format(
                                   f, e, str(stack_info)))
                raise e

        weak_data['strong_aug'] = strong_data
        return weak_data


class BatchCompose_SSOD(Compose):
    def __init__(self, transforms, num_classes=80, collate_batch=True):
        super(BatchCompose_SSOD, self).__init__(transforms, num_classes)
        self.collate_batch = collate_batch

    def __call__(self, data):
        # split strong_data from data(weak_data)
        strong_data = []
        for sample in data:
            strong_data.append(sample['strong_aug'])
            sample.pop('strong_aug')

        for f in self.transforms_cls:
            try:
                data = f(data)
                strong_data = f(strong_data)
            except Exception as e:
                stack_info = traceback.format_exc()
                logger.warning("fail to map batch transform [{}] "
                               "with error: {} and stack:\n{}".format(
                                   f, e, str(stack_info)))
                raise e

        # remove keys which is not needed by model
        extra_key = ['h', 'w', 'flipped']
        for k in extra_key:
            for sample in data:
                if k in sample:
                    sample.pop(k)
            for sample in strong_data:
                if k in sample:
                    sample.pop(k)

        # batch data, if user-define batch function needed
        # use user-defined here
        if self.collate_batch:
            batch_data = default_collate_fn(data)
            strong_batch_data = default_collate_fn(strong_data)
            return batch_data, strong_batch_data
        else:
            batch_data = {}
            for k in data[0].keys():
                tmp_data = []
                for i in range(len(data)):
                    tmp_data.append(data[i][k])
                if not 'gt_' in k and not 'is_crowd' in k and not 'difficult' in k:
                    tmp_data = np.stack(tmp_data, axis=0)
                batch_data[k] = tmp_data

            strong_batch_data = {}
            for k in strong_data[0].keys():
                tmp_data = []
                for i in range(len(strong_data)):
                    tmp_data.append(strong_data[i][k])
                if not 'gt_' in k and not 'is_crowd' in k and not 'difficult' in k:
                    tmp_data = np.stack(tmp_data, axis=0)
                strong_batch_data[k] = tmp_data

        return batch_data, strong_batch_data


class CombineSSODLoader(object):
    def __init__(self, label_loader, unlabel_loader):
        self.label_loader = label_loader
        self.unlabel_loader = unlabel_loader

    def __iter__(self):
        while True:
            try:
                label_samples = next(self.label_loader_iter)
            except:
                self.label_loader_iter = iter(self.label_loader)
                label_samples = next(self.label_loader_iter)

            try:
                unlabel_samples = next(self.unlabel_loader_iter)
            except:
                self.unlabel_loader_iter = iter(self.unlabel_loader)
                unlabel_samples = next(self.unlabel_loader_iter)

            yield (
                label_samples[0],  # sup weak
                label_samples[1],  # sup strong
                unlabel_samples[0],  # unsup weak
                unlabel_samples[1]  # unsup strong
            )

    def __call__(self):
        return self.__iter__()


class BaseSemiDataLoader(object):
    def __init__(self,
                 sample_transforms=[],
                 weak_aug=[],
                 strong_aug=[],
                 sup_batch_transforms=[],
                 unsup_batch_transforms=[],
                 sup_batch_size=1,
                 unsup_batch_size=1,
                 shuffle=True,
                 drop_last=True,
                 num_classes=80,
                 collate_batch=True,
                 use_shared_memory=False,
                 **kwargs):
        # sup transforms
        self._sample_transforms_label = Compose_SSOD(
            sample_transforms, weak_aug, strong_aug, num_classes=num_classes)
        self._batch_transforms_label = BatchCompose_SSOD(
            sup_batch_transforms, num_classes, collate_batch)
        self.batch_size_label = sup_batch_size

        # unsup transforms
        self._sample_transforms_unlabel = Compose_SSOD(
            sample_transforms, weak_aug, strong_aug, num_classes=num_classes)
        self._batch_transforms_unlabel = BatchCompose_SSOD(
            unsup_batch_transforms, num_classes, collate_batch)
        self.batch_size_unlabel = unsup_batch_size

        # common
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.use_shared_memory = use_shared_memory
        self.kwargs = kwargs

    def __call__(self,
                 dataset_label,
                 dataset_unlabel,
                 worker_num,
                 batch_sampler_label=None,
                 batch_sampler_unlabel=None,
                 return_list=False):
        # sup dataset 
        self.dataset_label = dataset_label
        self.dataset_label.check_or_download_dataset()
        self.dataset_label.parse_dataset()
        self.dataset_label.set_transform(self._sample_transforms_label)
        self.dataset_label.set_kwargs(**self.kwargs)
        if batch_sampler_label is None:
            self._batch_sampler_label = DistributedBatchSampler(
                self.dataset_label,
                batch_size=self.batch_size_label,
                shuffle=self.shuffle,
                drop_last=self.drop_last)
        else:
            self._batch_sampler_label = batch_sampler_label

        # unsup dataset
        self.dataset_unlabel = dataset_unlabel
        self.dataset_unlabel.length = self.dataset_label.__len__()
        self.dataset_unlabel.check_or_download_dataset()
        self.dataset_unlabel.parse_dataset()
        self.dataset_unlabel.set_transform(self._sample_transforms_unlabel)
        self.dataset_unlabel.set_kwargs(**self.kwargs)
        if batch_sampler_unlabel is None:
            self._batch_sampler_unlabel = DistributedBatchSampler(
                self.dataset_unlabel,
                batch_size=self.batch_size_unlabel,
                shuffle=self.shuffle,
                drop_last=self.drop_last)
        else:
            self._batch_sampler_unlabel = batch_sampler_unlabel

        # DataLoader do not start sub-process in Windows and Mac
        # system, do not need to use shared memory
        use_shared_memory = self.use_shared_memory and \
                            sys.platform not in ['win32', 'darwin']
        # check whether shared memory size is bigger than 1G(1024M)
        if use_shared_memory:
            shm_size = _get_shared_memory_size_in_M()
            if shm_size is not None and shm_size < 1024.:
                logger.warning("Shared memory size is less than 1G, "
                               "disable shared_memory in DataLoader")
                use_shared_memory = False

        self.dataloader_label = DataLoader(
            dataset=self.dataset_label,
            batch_sampler=self._batch_sampler_label,
            collate_fn=self._batch_transforms_label,
            num_workers=worker_num,
            return_list=return_list,
            use_shared_memory=use_shared_memory)

        self.dataloader_unlabel = DataLoader(
            dataset=self.dataset_unlabel,
            batch_sampler=self._batch_sampler_unlabel,
            collate_fn=self._batch_transforms_unlabel,
            num_workers=worker_num,
            return_list=return_list,
            use_shared_memory=use_shared_memory)

        self.dataloader = CombineSSODLoader(self.dataloader_label,
                                            self.dataloader_unlabel)
        self.loader = iter(self.dataloader)
        return self

    def __len__(self):
        return len(self._batch_sampler_label)

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.loader)

    def next(self):
        # python2 compatibility
        return self.__next__()


@register
class SemiTrainReader(BaseSemiDataLoader):
    __shared__ = ['num_classes']

    def __init__(self,
                 sample_transforms=[],
                 weak_aug=[],
                 strong_aug=[],
                 sup_batch_transforms=[],
                 unsup_batch_transforms=[],
                 sup_batch_size=1,
                 unsup_batch_size=1,
                 shuffle=True,
                 drop_last=True,
                 num_classes=80,
                 collate_batch=True,
                 **kwargs):
        super(SemiTrainReader, self).__init__(
            sample_transforms, weak_aug, strong_aug, sup_batch_transforms,
            unsup_batch_transforms, sup_batch_size, unsup_batch_size, shuffle,
            drop_last, num_classes, collate_batch, **kwargs)






class DistributedBatchSampler_u(BatchSampler):
    """Sampler that restricts data loading to a subset of the dataset.

    In such case, each process can pass a DistributedBatchSampler instance 
    as a DataLoader sampler, and load a subset of the original dataset that 
    is exclusive to it.

    .. note::
        Dataset is assumed to be of constant size.
        
    Args:
        dataset(paddle.io.Dataset): this could be a `paddle.io.Dataset` implement
                     or other python object which implemented
                     `__len__` for BatchSampler to get sample
                     number of data source.
        batch_size(int): sample indice number in a mini-batch indices.
        num_replicas(int, optional): porcess number in distributed training.
            If :attr:`num_replicas` is None, :attr:`num_replicas` will be
            retrieved from :code:`paddle.distributed.ParallenEnv`.
            Default None.
        rank(int, optional): the rank of the current process among :attr:`num_replicas`
            processes. If :attr:`rank` is None, :attr:`rank` is retrieved from
            :code:`paddle.distributed.ParallenEnv`. Default None.
        shuffle(bool): whther to shuffle indices order before genrating
            batch indices. Default False.
        drop_last(bool): whether drop the last incomplete batch dataset size
            is not divisible by the batch size. Default False

    Examples:
        .. code-block:: python

            import numpy as np

            from paddle.io import Dataset, DistributedBatchSampler

            # init with dataset
            class RandomDataset(Dataset):
                def __init__(self, num_samples):
                    self.num_samples = num_samples
            
                def __getitem__(self, idx):
                    image = np.random.random([784]).astype('float32')
                    label = np.random.randint(0, 9, (1, )).astype('int64')
                    return image, label
                
                def __len__(self):
                    return self.num_samples
  
            dataset = RandomDataset(100)
            sampler = DistributedBatchSampler(dataset, batch_size=64)

            for data in sampler:
                # do something
                break
    """

    def __init__(self,
                 dataset,
                 sizes,
                 batch_size,
                 num_replicas=None,
                 rank=None,
                 shuffle=False,
                 drop_last=False,
                 sample_epoch_size=1600,
                 dataset_ratio=[1,1,1],
                 muti_dataset_use_cas=[False,True,True]):
        self.dataset = dataset

        assert isinstance(batch_size, int) and batch_size > 0, \
                "batch_size should be a positive integer"
        self.batch_size = batch_size
        assert isinstance(shuffle, bool), \
                "shuffle should be a boolean value"
        self.shuffle = shuffle
        assert isinstance(drop_last, bool), \
                "drop_last should be a boolean number"

        from paddle.fluid.dygraph.parallel import ParallelEnv

        if num_replicas is not None:
            assert isinstance(num_replicas, int) and num_replicas > 0, \
                    "num_replicas should be a positive integer"
            self.nranks = num_replicas
        else:
            self.nranks = ParallelEnv().nranks

        if rank is not None:
            assert isinstance(rank, int) and rank >= 0, \
                    "rank should be a non-negative integer"
            self.local_rank = rank
        else:
            self.local_rank = ParallelEnv().local_rank

        self.drop_last = drop_last
        self.epoch = 0
        self.sample_epoch_size=sample_epoch_size
        self.num_samples = self.sample_epoch_size
        self.total_size = self.num_samples
        assert len(dataset_ratio) == len(sizes), \
            'length of dataset ratio {} should be equal to number if dataset {}'.format(
                len(dataset_ratio), len(sizes)
            )
        dataset_weight = [paddle.ones([s]) * max(sizes) / s * r / sum(dataset_ratio) \
            for i, (r, s) in enumerate(zip(dataset_ratio, sizes))]
        st=0
        cas_factors = []
        for i, s in enumerate(sizes):
            
            if muti_dataset_use_cas[i]:
                cas_factor = self._get_class_balance_factor_per_dataset(
                    self.dataset.roidbs[st: st + s],
                    l=1.0)
                cas_factor = cas_factor * (s / cas_factor.sum())
            else:
                cas_factor = paddle.ones([s])
            cas_factors.append(cas_factor)
            st = st + s
        cas_factors = paddle.concat(cas_factors)
        dataset_weight = paddle.concat(dataset_weight)
        self.weights = dataset_weight * cas_factors
    def __iter__(self):
        num_samples = self.sample_epoch_size
        indices = paddle.multinomial(
            self.weights, num_samples,
            replacement=True)
        # indices=paddle.to_tensor(np.load('ids.npy'))
        indices =indices.tolist()
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        def _get_indices_by_batch_size(indices):
            subsampled_indices = []
            last_batch_size = self.total_size % (self.batch_size * self.nranks)
            assert last_batch_size % self.nranks == 0
            last_local_batch_size = last_batch_size // self.nranks

            for i in range(self.local_rank * self.batch_size,
                           len(indices) - last_batch_size,
                           self.batch_size * self.nranks):
                subsampled_indices.extend(indices[i:i + self.batch_size])

            indices = indices[len(indices) - last_batch_size:]
            subsampled_indices.extend(indices[
                self.local_rank * last_local_batch_size:(
                    self.local_rank + 1) * last_local_batch_size])
            return subsampled_indices

        if self.nranks > 1:
            indices = _get_indices_by_batch_size(indices)
        assert len(indices) == self.num_samples//self.nranks
        _sample_iter = iter(indices)
        num_datasets=3
        self._buckets = [[] for _ in range(2 * num_datasets)]
        for idx in _sample_iter:
            d=self.dataset.roidbs[idx]           
            w, h = d["w"], d["h"]
            aspect_ratio_bucket_id = 0 if w > h else 1
            bucket_id = d['dataset_source'] * 2 + aspect_ratio_bucket_id
            bucket = self._buckets[bucket_id]
            bucket.append(idx)
            if len(bucket) == self.batch_size:
                yield bucket[:]
                del bucket[:]

    def __len__(self):
        num_samples = self.num_samples
        num_samples += int(not self.drop_last) * (self.batch_size - 1)
        return num_samples // self.batch_size//self.nranks

    def set_epoch(self, epoch):
        self.epoch = epoch




    def _get_class_balance_factor_per_dataset(self, dataset_dicts, l=1.):
        ret = []
        category_freq = defaultdict(int)
        for dataset_dict in dataset_dicts:  # For each image (without repeats)
            cat_ids = {ann for ann in dataset_dict['gt_class'][0]}
            for cat_id in cat_ids:
                category_freq[cat_id] += 1
        for i, dataset_dict in enumerate(dataset_dicts):
            cat_ids = {ann for ann in dataset_dict['gt_class'][0]}
            ret.append(sum(
                [1. / (category_freq[cat_id] ** l) for cat_id in cat_ids]))
        return paddle.to_tensor(ret,dtype='float32')
    
    
    

