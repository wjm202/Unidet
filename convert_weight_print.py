import paddle
import torch
import pickle
state_dict = torch.load('Unified_learned_OCIM_R50_6x+2x.pth')
paddle_model=paddle.load('model_partion.pdparams')
state_dict_backbone=pickle.load(open(r'R-50.pkl','rb'))
print(state_dict['model'].keys())
with open("paddle_paration.txt", "w+") as f:
    for key, value in paddle_model.items():
            print(key, value.shape)
            f.write('{:<60}\t{:<20}\t\n'.format(str(key), str(value.shape)))
    f.close()
with open("partition_torch.txt", "w+") as f:
    for key, value in state_dict['model'].items():
            print(key, value.shape)
            f.write('{:<60}\t{:<20}\t\n'.format(str(key), str(list(value.cpu().numpy().shape))))
    f.close()
with open("resnet50_torch.txt", "w+") as f:
    for key, value in state_dict_backbone.items():
            print(key, value.shape)
            f.write('{:<60}\t{:<20}\t\n'.format(str(key), str(list(value.shape))))
    f.close()


import os,sys
# 按行合并 两个txt文件
# 打开所需要合并的两个txt文件
file1path = 'paddle_paration.txt'
file2path = 'partion_match.txt'
 
file_1 = open(file1path, 'r', encoding='utf-8')
file_2 = open(file2path, 'r', encoding='utf-8')
 
list1 = []
for line in file_1.readlines():
    ss =line[:60].strip()
    list1.append(ss)
file_1.close()
 
list2 = []
for line in file_2.readlines():
    ss =line[:60].strip()
    list2.append(ss)
file_2.close()
 
# 创建新的txt文件，用来保存，'annotation/result2.txt'为保存路径
file_new = open('torch2paddle_partion.txt', 'w', encoding='utf-8')
for i in range(len(list1)):
    # 将两个txt文件合并到一行 中间用分隔符隔开
    sline = list1[i] + '\t' + list2[i]
    # 写入新的txt文件 换行
    file_new.write('{:<60}\t{:<60}\t\n'.format(list1[i] , list2[i]))

import paddle 
import torch 
import pickle
weight_name_map = {}
with open('torch2paddle_partion.txt') as f:
    for line in f.readlines():
        fields = line.split()
        weight_name_map[fields[0]] = fields[1]
dst = {}
src = torch.load('mode_backbone.pth')['model']
for k, v in weight_name_map.items():
    if 'roi_heads.box_' in v and len(src[v].shape)>1:
        dst[k] = src[v].transpose(0,1).cpu().numpy().astype('float32')
    else:
        dst[k] = src[v].cpu().numpy().astype('float32')
paddle.save(dst, "resnet_torch.pdparams")




import paddle 
import torch 
import pickle
dst = paddle.load('resnet_torch.pdparams')
src = paddle.load('/root/.cache/paddle/weights/ResNet50_cos_pretrained.pdparams')
for k, v in src.items():
    dst[k] = src[k].cpu().numpy().astype('float32')
paddle.save(dst, "resnet_paddle_init_weight.pdparams")