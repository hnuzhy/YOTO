# Augmentation of Demonstrations
In this part, we show how to condcut the point cloud-level geometry augmentation of manipulated objects for all valid demonstrations in each task. These pre-processed data will be used for training BiDP models.

## Step 0: Data download and environment configuration

* Download the raw binocular paired left/right images (in *.jpg format) and the corresponding dual robot arm states (in *.json format) for each task. We have uploaded them (`drawer.zip`, `pouring.zip`, `unscrew.zip`, `uncover.zip` and `openbox.zip`) on [huggingface/YOTO](https://huggingface.co/HoyerChou/YOTO/tree/main). We also uploaded the corresponding segmented manipulated objects of all left-view images (`drawer_mask.zip`, `pouring_mask.zip`, `unscrew_mask.zip`, `uncover_mask.zip` and `openbox_mask.zip`). These data will be pre-processed for training.

<table>
  <tr>
    <th> raw images and states </th>
    <th> raw segmented objects </th>
  </tr>
  <tr>
    <td><img src="./materials/raw_images_states.jpg" height="280"></td>
    <td><img src="./materials/raw_segmented_objs.jpg" height="280"></td> 
  </tr>
</table>

* Configure the dependent environment required for binocular stereo matching. We used the [IGEV](https://github.com/gangweiX/IGEV) algorithm and [KingFisher](https://dexforce-3dvision.com/productinfo/1022811.html) binocular camera. Among them, you need to utilize the dependency packages [glia](http://rjyfb:123456@69.235.177.182:10801/simple/glia/) and [glia-trt](http://rjyfb:123456@69.235.177.182:10801/simple/glia-trt/) developed by `DexForce` in file [infer_trt.py](https://github.com/hnuzhy/YOTO/blob/main/AugDemos/infer_trt.py#L12). Despite this, due to hardware differences, readers may need to use commercial binocular cameras (such as RealSense) to collect similar data.


## Step 1: Pre-processing of all valid demonstrations

## Step 2: Geometric transformation in point cloud-level
