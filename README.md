# Revealing the Dark Secrets of Masked Image Modeling (Depth Estimation) [[Paper]](https://arxiv.org/abs/2205.13543)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/revealing-the-dark-secrets-of-masked-image/monocular-depth-estimation-on-nyu-depth-v2)](https://paperswithcode.com/sota/monocular-depth-estimation-on-nyu-depth-v2?p=revealing-the-dark-secrets-of-masked-image)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/revealing-the-dark-secrets-of-masked-image/monocular-depth-estimation-on-kitti-eigen)](https://paperswithcode.com/sota/monocular-depth-estimation-on-kitti-eigen?p=revealing-the-dark-secrets-of-masked-image)

### Main results
#### Results on NYUv2
| Backbone | d1 | d2 | d3 | abs_rel | rmse | rmse_log |
|-------------------|-------|-------|--------|--------|--------|-------|
| **Swin-v2-Base** |  0.935 | 0.991 | 0.998 | 0.044 | 0.304 | 0.109 | 
| **Swin-v2-Large** |  0.949 | 0.994 | 0.999 | 0.036 | 0.287 | 0.102 | 

#### Results on KITTI
| Backbone | d1 | d2 | d3 | abs_rel | rmse | rmse_log |
|-------------------|-------|-------|--------|--------|--------|-------|
| **Swin-v2-Base** |  0.976 | 0.998 | 0.999 | 0.052 | 2.050 | 0.078 |
| **Swin-v2-Large** |  0.977 | 0.998   | 1.000 | 0.050 | 1.966 | 0.075 | 

### Preparation
Please refer to [[GLPDepth]](https://github.com/vinvino02/GLPDepth) for configuring the environment and preparing the NYUV2 and KITTI datasets. 
You can download pretrained models and our well-trained models from zoo([OneDrive](https://mailustceducn-my.sharepoint.com/:f:/g/personal/aa397601_mail_ustc_edu_cn/EkoYQyhiD6hJu9CGYLOwiF8BRqHgk8kX61NUcyfmdOUV7Q?e=h2uctw)).


### Training


- Training with model (NYU Depth V2 Swin-Base)
  
  ```
  $ python3 train.py --dataset nyudepthv2 --data_path ../data/ --max_depth 10.0 --max_depth_eval 10.0  --backbone swin_base_v2 --depths 2 2 18 2 --num_filters 32 32 32 --deconv_kernels 2 2 2 --window_size 30 30 30 15 --pretrain_window_size 12 12 12 6 --use_shift True True False False --flip_test --shift_window_test --shift_size 2 --pretrained weights/swin_v2_base_simmim.pth --save_model --crop_h 480 --crop_w 480 --layer_decay 0.9 --drop_path_rate 0.3 --log_dir logs/ 
  ```

- Training with model (NYU Depth V2 Swin-Large)
  
  ```
  $ python3 train.py --dataset nyudepthv2 --data_path ../data/ --max_depth 10.0 --max_depth_eval 10.0  --backbone swin_large_v2 --depths 2 2 18 2 --num_filters 32 32 32 --deconv_kernels 2 2 2 --window_size 30 30 30 15 --pretrain_window_size 12 12 12 6 --use_shift True True False False --flip_test --shift_window_test --shift_size 2 --pretrained weights/swin_v2_large_simmim.pth --save_model --crop_h 480 --crop_w 480 --layer_decay 0.85 --drop_path_rate 0.5 --log_dir logs/ 
  ```

- Training with model (KITTI Swin-Base)
  
  ```
  $ python3 train.py --dataset kitti --kitti_crop garg_crop --data_path ../data/ --max_depth 80.0 --max_depth_eval 80.0 --backbone swin_base_v2 --depths 2 2 18 2 --num_filters 32 32 32 --deconv_kernels 2 2 2 --window_size 22 22 22 11 --pretrain_window_size 12 12 12 6 --use_shift True True False False --flip_test --shift_window_test --shift_size 16 --pretrained weights/swin_v2_base_simmim.pth --save_model --crop_h 352 --crop_w 352 --layer_decay 0.9 --drop_path_rate 0.3 --log_dir logs/ 
  ```

- Training with model (KITTI Swin-Large)
  
  ```
  $ python3 train.py --dataset kitti --kitti_crop garg_crop --data_path ../data/ --max_depth 80.0 --max_depth_eval 80.0 --backbone swin_base_v2 --depths 2 2 18 2 --num_filters 32 32 32 --deconv_kernels 2 2 2 --window_size 22 22 22 11 --pretrain_window_size 12 12 12 6 --use_shift True True False False --flip_test --shift_window_test --shift_size 16 --pretrained weights/swin_v2_large_simmim.pth --save_model --crop_h 352 --crop_w 352 --layer_decay 0.85 --drop_path_rate 0.5 --log_dir logs/ 
  ```


#### Evaluation


- Evaluate with model (NYU Depth V2 Swin-Base)
  
  ```
  $ python3 test.py --dataset nyudepthv2 --data_path ../data/ --max_depth 10.0 --max_depth_eval 10.0  --backbone swin_base_v2 --depths 2 2 18 2 --num_filters 32 32 32 --deconv_kernels 2 2 2 --window_size 30 30 30 15 --pretrain_window_size 12 12 12 6 --use_shift True True False False --flip_test --shift_window_test --shift_size 2 --do_evaluate --ckpt_dir ckpt/nyudepthv2_swin_base.ckpt
  ```

- Evaluate with model (NYU Depth V2 Swin-Large)
  
  ```
  $ python3 test.py --dataset nyudepthv2 --data_path ../data/ --max_depth 10.0 --max_depth_eval 10.0  --backbone swin_large_v2 --depths 2 2 18 2 --num_filters 32 32 32 --deconv_kernels 2 2 2 --window_size 30 30 30 15 --pretrain_window_size 12 12 12 6 --use_shift True True False False --flip_test --shift_window_test --shift_size 2 --do_evaluate --ckpt_dir ckpt/nyudepthv2_swin_large.ckpt
  ```

- Evaluate with model (KITTI Swin-Base)
  
  ```
  $ python3 test.py --dataset kitti --kitti_crop garg_crop --data_path ../data/ --max_depth 80.0 --max_depth_eval 80.0 --backbone swin_base_v2 --depths 2 2 18 2 --num_filters 32 32 32 --deconv_kernels 2 2 2 --window_size 22 22 22 11 --pretrain_window_size 12 12 12 6 --use_shift True True False False --flip_test --shift_window_test --shift_size 16 --do_evaluate --ckpt_dir ckpt/kitti_swin_base.ckpt
  ```

- Evaluate with model (KITTI Swin-Large)
  
  ```
  $ python3 test.py --dataset kitti --kitti_crop garg_crop --data_path ../data/ --max_depth 80.0 --max_depth_eval 80.0 --backbone swin_base_v2 --depths 2 2 18 2 --num_filters 32 32 32 --deconv_kernels 2 2 2 --window_size 22 22 22 11 --pretrain_window_size 12 12 12 6 --use_shift True True False False --flip_test --shift_window_test --shift_size 16 --do_evaluate --ckpt_dir ckpt/kitti_swin_base.ckpt
  ```

### Citation

```
@article{xie2023darkmim,
  title={Revealing the Dark Secrets of Masked Image Modeling},
  author={Zhenda Xie, Zigang Geng, Jingcheng Hu, Zheng Zhang, Han Hu, Yue Cao},
  journal={arXiv preprint arXiv:2205.13543},
  year={2022}
}
```

### Acknowledge

Our code is mainly based on GLPDepth[1]. The code of the model is from SwinTransformer[2] and Simple Baseline[3].

[1] Global-Local Path Networks for Monocular Depth Estimation with Vertical CutDepth. [[code]](https://github.com/vinvino02/GLPDepth)

[2] Swin Transformer: Hierarchical Vision Transformer using Shifted Windows. [[code]](https://github.com/microsoft/Swin-Transformer)

[3] Simple Baselines for Human Pose Estimation and Tracking. [[code]](https://github.com/microsoft/human-pose-estimation.pytorch)
