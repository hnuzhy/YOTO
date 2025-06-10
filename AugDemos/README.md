# Augmentation of Demonstrations
In this part, we show how to condcut the point cloud-level geometry augmentation of manipulated objects for all valid demonstrations in each task. These pre-processed data will be used for training BiDP models.

## Step 0: Data download and environment configuration

* Download the raw binocular paired left/right images (in *.jpg format) and the corresponding dual robot arm states (in *.json format) for each task. We have uploaded them (`drawer.zip`, `pouring.zip`, `unscrew.zip`, `uncover.zip` and `openbox.zip`) on [huggingface/YOTO](https://huggingface.co/HoyerChou/YOTO/tree/main). We also uploaded the corresponding segmented manipulated objects of all left-view images (`drawer_mask.zip`, `pouring_mask.zip`, `unscrew_mask.zip`, `uncover_mask.zip` and `openbox_mask.zip`). These demonstrations are auto-rollout verified in real-world, and will be pre-processed for training.

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

* Configure the dependent environment required for binocular stereo matching. We used the [IGEV](https://github.com/gangweiX/IGEV) algorithm and [KingFisher](https://dexforce-3dvision.com/productinfo/1022811.html) binocular camera. Among them, you need to utilize the dependency packages [glia](http://rjyfb:123456@69.235.177.182:10801/simple/glia/) and [glia-trt](http://rjyfb:123456@69.235.177.182:10801/simple/glia-trt/) developed by `DexForce` in file [infer_trt.py](https://github.com/hnuzhy/YOTO/blob/main/AugDemos/infer_trt.py#L12). For files [calib_1106.yaml](https://huggingface.co/HoyerChou/YOTO/blob/main/calib_1106.yaml) and [igev.trt](https://huggingface.co/HoyerChou/YOTO/blob/main/igev.trt) which are needed by script [infer_trt.py](https://github.com/hnuzhy/YOTO/blob/main/AugDemos/infer_trt.py#L47), you can download them if you want to reconstruct 3D point clouds of objects. Despite this, due to hardware differences, readers may need to use commercial binocular cameras (such as RealSense) to collect similar data.

I have uploaded these two files related to the used kingfisher camera in huggingface (calib_1106.yaml and igev.trt). Please check it.

## Step 1: Pre-processing of all valid demonstrations
```
python aug_dataset_v1.py
```
Before running, you can adjust many useful parameters in this script. If you cannot reproduce our environment, you can directly download our pre-processed jsons (`drawer_preprocessed.json`, `pouring_preprocessed.json`, `unscrew_preprocessed.json`, `uncover_preprocessed.json` and `openbox_preprocessed.json`) in [huggingface/YOTO](https://huggingface.co/HoyerChou/YOTO/tree/main) for training.
* [`debug_flag`](https://github.com/hnuzhy/YOTO/blob/main/AugDemos/aug_dataset_v1.py#L216): whether using the debug mode to checking pre-processed results, such as matching effect, object masks and segmented object point clouds.
* [`robot2camera_trans`](https://github.com/hnuzhy/YOTO/blob/main/AugDemos/aug_dataset_v1.py#L217): whether tansforming the dual-arm robot states (6-DoF keyposes) from the robot world into the camera world. We have valided that set it as True will not be better. We thus utilize visual observations in the camera view and robot actions in the robot view to train all policy models.
* [`task_names`](https://github.com/hnuzhy/YOTO/blob/main/AugDemos/aug_dataset_v1.py#L219): You can change the task name list for pre-processing. We currently collected a fixed number of demonstrations through *Auto-Rollout Verification in Real-World* for each task (`drawer`, `pouring`, `unscrew`, `uncover`, `openbox`), which are `243/162/54/45/36` respectively.

<table>
  <tr>
    <th> matched binocular images </th>
    <th> objects point clouds </th>
  </tr>
  <tr>
    <td><img src="./materials/paired_raw_images.jpg" height="280"></td>
    <td><img src="./materials/objects_point_clouds.jpg" height="280"></td> 
  </tr>
</table>

## Step 2: Geometric transformation in point cloud-level
```
python aug_dataset_v2.py
```
Before running, you can also adjust many useful parameters in this script. We have also uploaded our pre-processed jsons (`drawer_preprocessed_aug100x_*.json`, `pouring_preprocessed_aug100x_*.json`, `unscrew_preprocessed_aug100x.json`, `uncover_preprocessed_aug100x.json` and `openbox_preprocessed_aug100x.json`) in [huggingface/YOTO](https://huggingface.co/HoyerChou/YOTO/tree/main) for training.
* [`debug_flag`](https://github.com/hnuzhy/YOTO/blob/main/AugDemos/aug_dataset_v2.py#L431): whether using the debug mode to checking pre-processed results, such as matching effect, object masks and segmented object point clouds.
* [`aug_times`](https://github.com/hnuzhy/YOTO/blob/main/AugDemos/aug_dataset_v2.py#L432): set the augmentation times for each single demonstration. We can set it as `1 / 5 / 25 / 100 / 500`. And we have valided that set it as `100` is good enough.
* [`left_pts_num`](https://github.com/hnuzhy/YOTO/blob/main/AugDemos/aug_dataset_v2.py#L433): set the left points number in manipulated object point clouds. We can set it as `32768 / 16384 / 8192 / 4096 / 2048`. In fact, we don’t need to set it too large, because the number of points when sampling objects during training is generally `1024 or 2048`. We set its as `4096` in our 100x augmentation for saving space of final pre-processed results. 
* [`task_names`](https://github.com/hnuzhy/YOTO/blob/main/AugDemos/aug_dataset_v2.py#L434): You can change the task name list for pre-processing. We currently generated 100x demonstrations through *Geometric Transformation of Point Clouds* for each task (`drawer`, `pouring`, `unscrew`, `uncover`, `openbox`), which are `24300/16200/5400/4500/3600` respectively.

<table>
  <tr>
    <th> the matched binocular images </th>
    <th> augmented objects point clouds </th>
  </tr>
  <tr>
    <td><img src="./materials/aug_pcd_drawer_rgbs.jpg" height="280"></td>
    <td><img src="./materials/aug_pcd_drawer_pcds.gif" height="280"></td> 
  </tr>
  <tr>
    <td><img src="./materials/aug_pcd_pouring_rgbs.jpg" height="280"></td>
    <td><img src="./materials/aug_pcd_pouring_pcds.gif" height="280"></td> 
  </tr>
  <tr>
    <td><img src="./materials/aug_pcd_uncover_rgbs.jpg" height="280"></td>
    <td><img src="./materials/aug_pcd_uncover_pcds.gif" height="280"></td> 
  </tr>
</table>

