# Total3D
For the CV course Final project

## environment requirements (Linux recommanded)
+ python 3.7
+ cuda 11.3
+ pytorch 1.12.1
+ torchvision 0.13.1
+ pymesh
+ other missing modules you can use 'pip install xxx' to install

## install pymesh
Follow this [website](https://blog.csdn.net/weixin_46632183/article/details/120553750) to install pymesh.

## dataset
To be filled.





# TODO

- [ ] 接口实现 example 

    ```python
    from argparse import ArgumentParser
    def get_args():
    parser = ArgumentParser()
    
    parser.add_argument(
    "--n_epochs", default=41, type=int, help="the number of epochs to run."
    )
    parser.add_argument(
    "--lr", default = 0.0001, type = float, help="learning rate."
    )
    
    return parser.parse_args()
    
    
    self.__init__(self, opt)
    ```

- [ ] Pipeline

    - [ ] 梯度回传路径？

    - [ ] Loss 构成？

    - [ ] How to Modify the Mesh's Output with the ODN outputs?

    - [ ] 

- [ ] *Layout Estimation Net* 

    - [ ] 图形学相关知识？

    - [ ] output features ?

- [ ] 数据怎么处理！？

    - [x] ODE 需要将场景中的物体划分后，假设有$N$个物品，以$N\times 3 \times 256 \times 256$输入，同时需要得到Geometry features($N \times N \times 64$)

- [x] 数据集处理、接口 dataset.load 适配后续feature extraction

    - [x] 弄明白 data label 比如说 data是image， label是空间坐标
    - [x] Pix 3D 在下载 
    - [x] SUN RGB-D ？ 

    

- [x] Object Detection Network  尽早实现一下

    - [x] Attention Sum 注意一下是怎么实现

- [ ] 提出 IDAES

- [ ] Mesh Generation Net 

    - [x] 除理大小不同的input image
    - [x] 不同size的图片 如何用一个网络做encoding *ReShape!*
    - [x] AtlasNet 边缘检测 、 边缘强化 **实现** （用 某种 net 实现）？ 能否用传统CV实现？
    - [ ] Need to pre-train

- [ ] 2D BOX and Get Geometry Feature

- [ ] 完善一下.gitignore, 数据集目录data、log等不建议上传。

- [ ] 这边建议数据集的路径链接用os.path.join，不然因为/的问题找不到目录很讨厌！

# IDEAS

1.    Style of Square? 与BaseLine不同的“放置”、“形变” 方式？
2.   滤镜
3.   如何充分利用 3D Box ？
4.   图像编辑？ 3D建模层面编辑。
5.   Add another thing into the 3D space! With a box to occupy some space.