# Normalized-disparity-loss

## Introduction

The code is a pytorch implementation of the paper: A Normalized Disparity Loss for Stereo Matching Networks, IEEE Robotics and Automation Letters, 2022. 

## Usage

1. Requirements: PyTorch>=1.0.0, Python3, tensorboardX

2. Train:
    1) Train PSMNet on SceneFlow

   '''
   sh tool/run.sh SF PSMNet
   '''
   
   Adjust the "xx.py" to "train_PSMNet.py" in run.sh and the paths or parameters in ./config/SF/SF_PSMNet.yaml

    2) Train PSMNet with normalized loss on SceneFlow
  
   '''
   sh tool/run.sh SF PSMNet_normloss
   '''
   
   Adjust the "xx.py" to "train_PSMNet_normloss.py" in run.sh and the paths or parameters in ./config/SF/SF_PSMNet_normloss.yaml

    3) For the training on other datasets, the procedure is similar.

3. Test:
   1) Test the trained model:
   
   '''
   sh tool/run.sh SF PSMNet_normloss
   '''
   
   Adjust the "xx.py" in run.sh to "test_PSMNet_normloss.py" and the paths or parameters in ./config/SF/PSMNet_normloss.yaml

4. Visualization:
   tensorboard --logdir=xx --port=6789 
   
## Citation

If you find this code useful in your research, please cite:
```
@article{Chen2022,
  author    = {Shuya Chen and Zhiyu Xiang and Peng Xu and Xijun Zhao},
  title     = {A Normalized Disparity Loss for Stereo Matching Networks},
  journal   = {IEEE Robotics and Automation Letters},
  year      = {2022}
}
```

## Acknowledgements

 The code is partly based on the [PSMNet](https://github.com/JiaRenChang/PSMNet) and [PSPNet](https://github.com/hszhao/semseg). Thanks to these excellent work.
```
@inproceedings{chang2018pyramid,
  title={Pyramid Stereo Matching Network},
  author={Chang, Jia-Ren and Chen, Yong-Sheng},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={5410--5418},
  year={2018}
}

@inproceedings{zhao2017pspnet,
  title={Pyramid Scene Parsing Network},
  author={Zhao, Hengshuang and Shi, Jianping and Qi, Xiaojuan and Wang, Xiaogang and Jia, Jiaya},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year={2017}
}
```
