# Unidet

# Simple multi-dataset detection
An object detector trained on multiple large-scale datasets with a unified label space; Winning solution of ECCV 2020 Robust Vision Challenges.

<p align="center"> <img src='docs/unidet_teaser.jpg' align="center" height="170px"> </p>

> [**Simple multi-dataset detection**](http://arxiv.org/abs/2102.13086),            
> Xingyi Zhou, Vladlen Koltun, Philipp Kr&auml;henb&uuml;hl,        
> *CVPR 2022 ([arXiv 2102.13086](http://arxiv.org/abs/2102.13086))*         

<!-- Contact: [zhouxy@cs.utexas.edu](mailto:zhouxy@cs.utexas.edu). Any questions or discussions are welcomed!  -->

## Features at a glance

- We trained a unified object detector on 4 large-scale detection datasets: COCO, Objects365, OpenImages, and Mapillary, with state-of-the-art performance on all of them.

- The model predicts class labels in a **learned** unified label space.

- The model can be directly used to test on novel datasets outside the training datasets.

- In this repo, we also provide state-of-the-art baselines for Objects365 and OpenImages.

## Inference
        {
            "name": "Python:cascade_rcnn_infer",
            "type": "python",
            "request": "launch",
            "program": "tools/infer.py",
            "args": ["-c","configs/cascade_rcnn/cascade_rcnn_r50_fpn_1x_coco.yml","-o","weights=model_torch.pdparams","--infer_img=17790319373_bd19b24cfc_k.jpg"],
            "console": "integratedTerminal",
            "justMyCode": true,
            "env": {"CUDA_VISIBLE_DEVICES":"4",
            },
        }
~~~
python tools/infer.py -c configs/cascade_rcnn/Partitioned_COI_R50_2x.yml "-o","weights=model_torch.pdparams","--infer_img=17790319373_bd19b24cfc_k.jpg"
~~~

~~~
python tools/infer.py -c configs/cascade_rcnn/Unified_learned_OCI_R50_2x.yaml "-o","weights=model_torch.pdparams","--infer_img=17790319373_bd19b24cfc_k.jpg"
~~~

## Training Partitioned Detector
~~~
python tools/train.py -c configs/cascade_rcnn/Partitioned_COI_R50_2x.yml --eval
~~~
## Learning a unified label space

请使用官方提供的 datasets/label_space/learned_mAP.json 开启Unified Detector训练
## Training Unified detector

~~~
python train_net.py --config-file -c configs/Unified_COI_R50_2x.yaml --eval
~~~
官方代码存在bug,reg loss nan 无法复现

## Benchmark evaluation and training

After installation, follow the instructions in [DATASETS.md](DATASETS.md) to setup the (many) datasets.


## Citation

If you find this project useful for your research, please use the following BibTeX entry.

    @inproceedings{zhou2021simple,
      title={Simple multi-dataset detection},
      author={Zhou, Xingyi and Koltun, Vladlen and Kr{\"a}henb{\"u}hl, Philipp},
      booktitle={CVPR},
      year={2022}
    }