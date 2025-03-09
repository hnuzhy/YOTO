# YOTO
Code for my paper "*You Only Teach Once: Learn One-Shot Bimanual Robotic Manipulation from Video Demonstrations*" [[arXiv](https://arxiv.org/abs/2501.14208)] / [[Project](https://hnuzhy.github.io/projects/YOTO/)] / [[Dataset](https://huggingface.co/HoyerChou/YOTO)] (coming soon)

[]

<table>
  <tr>
    <th> Task </th>
    <th> BiDP trained without augmentation </th>
    <th> BiDP trained with augmentation </th>
  </tr>
  <tr>
    <th> Drawer </th>
    <td><img src="./BiDP/materials/BiDP_infer_demo1_drawer_noaug.gif" height="240"></td>
    <td><img src="./BiDP/materials/BiDP_infer_demo1_drawer_withaug.gif" height="240"></td> 
  </tr>
  <tr>
    <th> Pouring </th>
    <td><img src="./BiDP/materials/BiDP_infer_demo1_pouring_noaug.gif" height="240"></td>
    <td><img src="./BiDP/materials/BiDP_infer_demo1_pouring_withaug.gif" height="240"></td> 
  </tr>
  <tr>
    <th> Unscrew </th>
    <td><img src="./BiDP/materials/BiDP_infer_demo1_unscrew_noaug.gif" height="240"></td>
    <td><img src="./BiDP/materials/BiDP_infer_demo1_unscrew_withaug.gif" height="240"></td> 
  </tr>
  <tr>
    <th> Uncover </th>
    <td><img src="./BiDP/materials/BiDP_infer_demo1_uncover_noaug.gif" height="240"></td>
    <td><img src="./BiDP/materials/BiDP_infer_demo1_uncover_withaug.gif" height="240"></td> 
  </tr>
  <tr>
    <th> Openbox </th>
    <td><img src="./BiDP/materials/BiDP_infer_demo1_openbox_noaug.gif" height="240"></td>
    <td><img src="./BiDP/materials/BiDP_infer_demo1_openbox_withaug.gif" height="240"></td> 
  </tr>
</table>

## ● Acknowledgement
Our `hand motion extraction and injection` process relies on a variety of vison algorithms, including Hand Detection and 3D Mesh Reconstruction [WiLoR](https://github.com/rolpotamias/WiLoR), Large Vision-Language Model [Florence2](https://huggingface.co/collections/microsoft/florence-6669f44df0d87d9c3bfb76de), Segment Anything Model 2 [SAM2](https://github.com/facebookresearch/segment-anything-2) and Binocular Stereo Matching  [IGEV](https://github.com/gangweiX/IGEV). While, the codebase of our imitation learning algorithm `BiDP` is partly based on [ACT](https://github.com/tonyzhaozh/act), [Diffusion Policy](https://github.com/real-stanford/diffusion_policy), [3D Diffusion Policy](https://github.com/YanjieZe/3D-Diffusion-Policy) and [EquiBot](https://github.com/yjy0625/equibot). We thank them for their open source efforts and contributions.

## ● Citation
If you use our code or models in your research, please cite with:
```bash
@article{zhou2025you,
  title={You Only Teach Once: Learn One-Shot Bimanual Robotic Manipulation from Video Demonstrations},
  author={Zhou, Huayi and Wang, Ruixiang and Tai, Yunxin and Deng, Yueci and Liu, Guiliang and Jia, Kui},
  journal={arXiv preprint arXiv:2501.14208},
  year={2025}
}
```
