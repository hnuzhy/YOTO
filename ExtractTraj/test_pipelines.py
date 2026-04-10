
import os
import pdb
import time
import numpy as np
import torch
import cv2
import json
import pickle
from scipy.spatial.transform import Rotation
from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import WiLorHandPose3dEstimationPipeline

from utils_wilor import Renderer
from utils_self import plot_3axis_Zaxis, batch_rodrigues
from config import video_info_dict


def test_wilor_image_pipeline():

    LIGHT_PURPLE = (0.25098039, 0.274117647, 0.65882353)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    dtype = torch.float16

    pipe = WiLorHandPose3dEstimationPipeline(device=device, dtype=dtype, verbose=False)
    img_path = "./assets/img.png"
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    for _ in range(20):
        t0 = time.time()
        outputs = pipe.predict(image, hand_conf=0.5)
        print("time_cost:", time.time() - t0)
    save_dir = "./results/imgs"
    os.makedirs(save_dir, exist_ok=True)
    renderer = Renderer(pipe.wilor_model.mano.faces)

    render_image = image.copy()
    render_image = render_image.astype(np.float32)[:, :, ::-1] / 255.0
    pred_keypoints_2d_all = []
    pred_keypoints_3d_all = []
    hand_orient_all = []
    for i, out in enumerate(outputs):
        # print(out, out["wilor_preds"]["pred_cam"])
        verts = out["wilor_preds"]['pred_vertices'][0]
        is_right = out['is_right']
        cam_t = out["wilor_preds"]['pred_cam_t_full'][0]
        scaled_focal_length = out["wilor_preds"]['scaled_focal_length']
        pred_keypoints_2d = out["wilor_preds"]["pred_keypoints_2d"]
        pred_keypoints_2d_all.append(pred_keypoints_2d)

        pred_keypoints_3d = out["wilor_preds"]["pred_keypoints_3d"]
        pred_keypoints_3d_all.append(pred_keypoints_3d)
        # https://files.is.tue.mpg.de/black/talks/SMPL-made-simple-FAQs.pdf
        hand_orient = out["wilor_preds"]['global_orient'][0][0]  # Axis-Angle (3,) in radius, SMPL’s default input
        hand_orient_all.append([hand_orient, is_right])
        
        misc_args = dict(
            mesh_base_color=LIGHT_PURPLE,
            scene_bg_color=(1, 1, 1),
            focal_length=scaled_focal_length,
        )
        tmesh = renderer.vertices_to_trimesh(verts, cam_t.copy(), LIGHT_PURPLE, is_right=is_right)
        tmesh.export(os.path.join(save_dir, f'{os.path.basename(img_path)}_hand{i:02d}.obj'))
        cam_view = renderer.render_rgba(verts, cam_t=cam_t, render_res=[image.shape[1], image.shape[0]],
                                        is_right=is_right, **misc_args)

        # Overlay image
        render_image = render_image[:, :, :3] * (1 - cam_view[:, :, 3:]) + cam_view[:, :, :3] * cam_view[:, :, 3:]

    render_image = (255 * render_image).astype(np.uint8)
    for pred_keypoints_2d, pred_keypoints_3d, [hand_orient, is_right] in zip(pred_keypoints_2d_all, pred_keypoints_3d_all, hand_orient_all):
        for j in range(pred_keypoints_2d[0].shape[0]):
            color, radius = (0, 0, 255), 3
            x, y = pred_keypoints_2d[0][j]
            if j in [4, 8, 12, 16, 20]:  # five fingertips
                color = (0, 255, 255) if is_right else (255, 255, 0)
                radius = 6
            cv2.circle(render_image, (int(x), int(y)), radius, color, -1)
            if j in [0, 1, 5, 9, 13, 17]:  # five finger-roots and wrist-root
                 cv2.circle(render_image, (int(x), int(y)), radius, (0, 0, 0), -1)

        # (wrong) method-1: https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.html
        # [pitch, yaw, roll] = Rotation.from_rotvec(hand_orient, degrees=False).as_euler('xyz', degrees=True)
        
        # (right) method-2: https://github.com/hnuzhy/DirectMHP/blob/main/exps/compare_img2pose.py#L91
        # rot_mat = Rotation.from_rotvec(hand_orient, degrees=False).as_matrix()
        # rot_mat_2 = np.transpose(rot_mat)
        # angle = Rotation.from_matrix(rot_mat_2).as_euler('xyz', degrees=True)
        # pitch, yaw, roll = angle[0], -angle[1], -angle[2]
        
        # (right) method-3: https://github.com/hassony2/manopth/blob/master/manopth/rodrigues_layer.py#L43
        # (right) method-3: https://github.com/MandyMo/HAMR/blob/master/src/util.py#L21
        # rot_mat = batch_rodrigues(torch.from_numpy(np.array([hand_orient]))).detach().numpy()
        # rot_mat_2 = np.transpose(rot_mat.reshape(3, 3))
        # angle = Rotation.from_matrix(rot_mat_2).as_euler('xyz', degrees=True)
        # pitch, yaw, roll = angle[0], -angle[1], -angle[2]
        
        # (right) method-4: calculate from root_joint and five fingertips
        root_joint, thumb_fingertip = pred_keypoints_3d[0][0], pred_keypoints_3d[0][4]
        avg_two_kpts = (pred_keypoints_3d[0][16] + pred_keypoints_3d[0][20]) / 2.0
        vec_ori_z = np.cross(pred_keypoints_3d[0][8]-root_joint, avg_two_kpts-root_joint)
        avg_four_fingertips = np.mean(pred_keypoints_3d[0][[8, 12, 16, 20]], axis=0)
        vec_ori_y = avg_four_fingertips - root_joint
        vec_ori_x = np.cross(vec_ori_y, vec_ori_z)
        vec_list = [v / (np.linalg.norm(v) + 1e-16) for v in [vec_ori_x, vec_ori_y, vec_ori_z]]
        angle = Rotation.from_matrix(np.array(vec_list)).as_euler('xyz', degrees=True)
        pitch, yaw, roll = angle[0], -angle[1], -angle[2]
 
        x1, x2 = np.min(pred_keypoints_2d[0][:, 0]), np.max(pred_keypoints_2d[0][:, 0])
        y1, y2 = np.min(pred_keypoints_2d[0][:, 1]), np.max(pred_keypoints_2d[0][:, 1])
        render_image = plot_3axis_Zaxis(render_image, yaw, pitch, roll, 
            tdx=(x1+x2)/2, tdy=(y1+y2)/2, size=max(y2-y1, x2-x1)*0.8, thickness=2)
             
    cv2.imwrite(os.path.join(save_dir, os.path.basename(img_path)), render_image)
    print(os.path.join(save_dir, os.path.basename(img_path)))


def test_wilor_video_pipeline():

    LIGHT_PURPLE = (0.25098039, 0.274117647, 0.65882353)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("Used device:", device)
    dtype = torch.float16

    pipe = WiLorHandPose3dEstimationPipeline(device=device, dtype=dtype)
    
    # video_path = "./assets/video.mp4"
    video_path = "./assets/video_20241202.mp4"
    # video_path = "./assets/cam_left_debug/Video_20241101125735351_drawer(online-video-cutter.com).mp4"
    # video_path = "./assets/cam_left_debug/Video_20241101142445645_teapot(online-video-cutter.com).mp4"
    # video_path = "../assets/cam_left_debug/Video_20241101143058673_uncover(online-video-cutter.com).mp4"
    # video_path = "./assets/cam_left_debug/Video_20241107111136157_drawer(online-video-cutter.com).mp4"
    
    save_dir = "./results"
    os.makedirs(save_dir, exist_ok=True)
    renderer = Renderer(pipe.wilor_model.mano.faces)

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create VideoWriter object
    output_path = os.path.join(save_dir, os.path.basename(video_path))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    vout = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    tracking_joints_poses = []
    using_global_hand_pose = False
    
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        t0 = time.time()
        outputs = pipe.predict(image, hand_conf=0.4)
        print("time_cost:", time.time() - t0)
        render_image = image.copy()
        render_image = render_image.astype(np.float32)[:, :, ::-1] / 255.0

        pred_kpts_2d_all = []
        pred_kpts_3d_all = []
        hand_orient_all = []

        for i, out in enumerate(outputs):
            verts = out["wilor_preds"]['pred_vertices'][0]
            is_right = out['is_right']
            cam_t = out["wilor_preds"]['pred_cam_t_full'][0]
            scaled_focal_length = out["wilor_preds"]['scaled_focal_length']

            pred_kpts_2d = out["wilor_preds"]["pred_keypoints_2d"]
            pred_kpts_2d_all.append(pred_kpts_2d)
            pred_kpts_3d = out["wilor_preds"]["pred_keypoints_3d"]
            pred_kpts_3d_all.append(pred_kpts_3d)
            hand_orient = out["wilor_preds"]['global_orient'][0][0]  # Axis-Angle (3,) in radius, SMPL’s default input
            hand_orient_all.append([hand_orient, is_right])

            misc_args = dict(
                mesh_base_color=LIGHT_PURPLE,
                scene_bg_color=(1, 1, 1),
                focal_length=scaled_focal_length,
            )
            # tmesh = renderer.vertices_to_trimesh(verts, cam_t.copy(), LIGHT_PURPLE, is_right=is_right)
            cam_view = renderer.render_rgba(verts, cam_t=cam_t, render_res=[image.shape[1], image.shape[0]],
                                            is_right=is_right, **misc_args)

            # Overlay image
            render_image = render_image[:, :, :3] * (1 - cam_view[:, :, 3:]) + cam_view[:, :, :3] * cam_view[:, :, 3:]

        render_image = (255 * render_image).astype(np.uint8)
        
        '''
        for pred_kpts_2d, pred_kpts_3d, [hand_orient, is_right] in zip(pred_kpts_2d_all, pred_kpts_3d_all, hand_orient_all):
            tracking_joint, tracking_joint_3d = [0, 0], np.array([0, 0, 0], dtype=np.float64)
            for j in range(pred_kpts_2d[0].shape[0]):
                color = (0, 0, 255)
                x, y = pred_kpts_2d[0][j]
                if j in [4, 8, 12, 16]:  # five fingertips [4, 8, 12, 16, 20]
                    color = (0, 255, 255) if is_right else (255, 255, 0)
                    tracking_joint[0] += x
                    tracking_joint[1] += y
                    tracking_joint_3d += np.array(pred_kpts_3d[0][j])
                cv2.circle(render_image, (int(x), int(y)), 3, color, -1)
            tracking_joint[0] /= 4.0
            tracking_joint[1] /= 4.0
            tracking_joint_3d = [value/4.0 for value in tracking_joint_3d]
            
            if using_global_hand_pose:
                rot_mat = Rotation.from_rotvec(hand_orient, degrees=False).as_matrix()
                rot_mat_2 = np.transpose(rot_mat)
                angle = Rotation.from_matrix(rot_mat_2).as_euler('xyz', degrees=True)
                pitch, yaw, roll = angle[0], -angle[1], -angle[2]
                x1, x2 = np.min(pred_kpts_2d[0][:, 0]), np.max(pred_kpts_2d[0][:, 0])
                y1, y2 = np.min(pred_kpts_2d[0][:, 1]), np.max(pred_kpts_2d[0][:, 1])
                render_image = plot_3axis_Zaxis(render_image, yaw, pitch, roll, 
                    tdx=(x1+x2)/2, tdy=(y1+y2)/2, size=max(y2-y1, x2-x1)*0.6, thickness=2)
            else:
                root_joint, thumb_fingertip = pred_kpts_3d[0][0], pred_kpts_3d[0][4]
                avg_two_kpts = (pred_kpts_3d[0][16] + pred_kpts_3d[0][20]) / 2.0
                vec_ori_z = np.cross(pred_kpts_3d[0][8]-root_joint, avg_two_kpts-root_joint)
                avg_four_fingertips = np.mean(pred_kpts_3d[0][[8, 12, 16, 20]], axis=0)
                vec_ori_y = avg_four_fingertips - root_joint
                vec_ori_x = np.cross(vec_ori_y, vec_ori_z)
                vec_list = [v / (np.linalg.norm(v) + 1e-16) for v in [vec_ori_x, vec_ori_y, vec_ori_z]]
                angle = Rotation.from_matrix(np.array(vec_list)).as_euler('xyz', degrees=True)
                pitch, yaw, roll = angle[0], -angle[1], -angle[2]
                x1, x2 = np.min(pred_kpts_2d[0][:, 0]), np.max(pred_kpts_2d[0][:, 0])
                y1, y2 = np.min(pred_kpts_2d[0][:, 1]), np.max(pred_kpts_2d[0][:, 1])
                render_image = plot_3axis_Zaxis(render_image, yaw, pitch, roll, 
                    tdx=tracking_joint[0], tdy=tracking_joint[1], size=max(y2-y1, x2-x1)*0.6, thickness=3)
            
            tracking_joints_poses.append([frame_count, is_right, [pitch, yaw, roll], tracking_joint, tracking_joint_3d])
                
        for idx, [_, is_right, _, tracking_joint, _] in enumerate(tracking_joints_poses[::-1]):
            if idx == 60:  break
            color = (0, 255, 255) if is_right else (255, 255, 0)
            cv2.circle(render_image, (int(tracking_joint[0]), int(tracking_joint[1])), 6, color, -1)
        '''
            
        # Write the frame to the output video
        vout.write(render_image)

        frame_count += 1
        print(f"Processed frame {frame_count}")

    # Release everything
    cap.release()
    vout.release()
    cv2.destroyAllWindows()
    
    '''
    with open(output_path.replace(".mp4", ".pkl"), "wb") as fp:  # Pickling
        pickle.dump(tracking_joints_poses, fp)
    '''
    
    print(f"Video processing complete. Output saved to {output_path}")


def test_wilor_paired_pipeline(video_id_str):

    LIGHT_PURPLE = (0.25098039, 0.274117647, 0.65882353)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("Used device:", device)
    dtype = torch.float16

    pipe = WiLorHandPose3dEstimationPipeline(device=device, dtype=dtype)

    paired_path_L = os.path.join("./assets/cam_left/", video_info_dict[video_id_str]["cam_l_name"])
    paired_path_R = os.path.join("./assets/cam_right/", video_info_dict[video_id_str]["cam_r_name"])
    keyframe_id_dict = video_info_dict[video_id_str]["keyframe_id"]
    gripper_state_dict = video_info_dict[video_id_str]["gripper_state"]
    showing_ids_list = video_info_dict[video_id_str]["showing_ids"]
    
    save_dir = "./results"
    os.makedirs(save_dir, exist_ok=True)
    task_name = video_id_str.split("_")[0]
    save_dir = "./results/" + task_name
    os.makedirs(save_dir, exist_ok=True)
    
    renderer = Renderer(pipe.wilor_model.mano.faces)

    # Create VideoWriter object
    fps, width, height = 25, 1280, 1024
    output_path = os.path.join(save_dir, os.path.basename(paired_path_L) + "_v2.mp4")
    print("output_path:", output_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    vout = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    tracking_joints_poses = []
    
    frames_L = os.listdir(paired_path_L)
    frames_R = os.listdir(paired_path_R)
    assert len(frames_L) == len(frames_R), "left and right folder with frames of different number!!!"
    frames_L.sort()
    frames_R.sort()
    for frame_count, [frame_l, frame_r] in enumerate(zip(frames_L, frames_R)):
        frame_lp = os.path.join(paired_path_L, frame_l)
        frame_rp = os.path.join(paired_path_R, frame_r)
        real_frame_id = int(frame_l.replace(".jpg", ""))
        
        # Convert frame to RGB
        image = cv2.cvtColor(cv2.imread(frame_lp), cv2.COLOR_BGR2RGB)
        t0 = time.time()
        outputs = pipe.predict(image, hand_conf=0.3)
        print("time_cost:", time.time() - t0)
        render_image = image.copy()
        render_image = render_image.astype(np.float32)[:, :, ::-1] / 255.0

        pred_kpts_2d_all = []
        pred_kpts_3d_all = []
        hand_orient_all = []
        
        if len(outputs) > 2:  # especially for the demonstrarion openbox_01
            print("************ Detecting many wrong hands!!! *************")
            for output in outputs:
                print(output['is_right'], output['hand_bbox'])

        left_right_hand_cnt = [0, 0]
        for i, out in enumerate(outputs):
            verts = out["wilor_preds"]['pred_vertices'][0]
            is_right = out['is_right']
            cam_t = out["wilor_preds"]['pred_cam_t_full'][0]
            scaled_focal_length = out["wilor_preds"]['scaled_focal_length']
            
            if is_right and left_right_hand_cnt[1]==0:
                left_right_hand_cnt[1] += 1
            elif not is_right and left_right_hand_cnt[0]==0:
                left_right_hand_cnt[0] += 1
            else:
                continue  # we do not consider more than one left/right hand
            
            pred_kpts_2d = out["wilor_preds"]["pred_keypoints_2d"]
            pred_kpts_2d_all.append(pred_kpts_2d)
            pred_kpts_3d = out["wilor_preds"]["pred_keypoints_3d"]
            pred_kpts_3d_all.append(pred_kpts_3d)
            hand_orient = out["wilor_preds"]['global_orient'][0][0]  # Axis-Angle (3,) in radius, SMPL’s default input
            hand_orient_all.append([hand_orient, is_right])

            misc_args = dict(
                mesh_base_color=LIGHT_PURPLE,
                scene_bg_color=(1, 1, 1),
                focal_length=scaled_focal_length,
            )
            # tmesh = renderer.vertices_to_trimesh(verts, cam_t.copy(), LIGHT_PURPLE, is_right=is_right)
            cam_view = renderer.render_rgba(verts, cam_t=cam_t, render_res=[image.shape[1], image.shape[0]],
                                            is_right=is_right, **misc_args)

            # Overlay image
            render_image = render_image[:, :, :3] * (1 - cam_view[:, :, 3:]) + cam_view[:, :, :3] * cam_view[:, :, 3:]

        render_image = (255 * render_image).astype(np.uint8)
        
        if real_frame_id in showing_ids_list:
            render_image_canvas = render_image.copy()
        if real_frame_id == 132 and "drawer" in video_id_str:  # for paper writing
            render_image_canvas = render_image.copy()

        for pred_kpts_2d, pred_kpts_3d, [hand_orient, is_right] in zip(pred_kpts_2d_all, pred_kpts_3d_all, hand_orient_all):
            tracking_joint, tracking_joint_3d = [0, 0], np.array([0, 0, 0], dtype=np.float64)
            for j in range(pred_kpts_2d[0].shape[0]):
                color = (0, 0, 255)
                x, y = pred_kpts_2d[0][j]
                if j in [4, 8, 12, 16]:  # five fingertips [4, 8, 12, 16, 20]
                    color = (0, 255, 255) if is_right else (255, 255, 0)
                    tracking_joint[0] += x
                    tracking_joint[1] += y
                    tracking_joint_3d += np.array(pred_kpts_3d[0][j])
                cv2.circle(render_image, (int(x), int(y)), 3, color, -1)
            tracking_joint[0] /= 4.0
            tracking_joint[1] /= 4.0
            tracking_joint_3d = [value/4.0 for value in tracking_joint_3d]
            
            # compute the 3dof hand pose
            # root_joint = np.mean(pred_kpts_3d[0][[0, 1, 5, 9, 13, 17]], axis=0)
            root_joint = pred_kpts_3d[0][0]
            vec_ori_z = np.cross(pred_kpts_3d[0][8]-root_joint, pred_kpts_3d[0][16]-root_joint)
            vec_ori_z = vec_ori_z / (np.linalg.norm(vec_ori_z) + 1e-16)
            avg_two_fingertips = (pred_kpts_3d[0][8] + pred_kpts_3d[0][16]) / 2.0
            vec_ori_y = avg_two_fingertips - root_joint
            vec_ori_y = vec_ori_y / (np.linalg.norm(vec_ori_y) + 1e-16)
            vec_ori_x = np.cross(vec_ori_y, vec_ori_z)
            rot_mat = np.array([vec_ori_x, vec_ori_y, vec_ori_z])  # the shape is (3, 3)
            angle = Rotation.from_matrix(rot_mat).as_euler('xyz', degrees=True)
            pitch, yaw, roll = angle[0], -angle[1], -angle[2]
            x1, x2 = np.min(pred_kpts_2d[0][:, 0]), np.max(pred_kpts_2d[0][:, 0])
            y1, y2 = np.min(pred_kpts_2d[0][:, 1]), np.max(pred_kpts_2d[0][:, 1])
            render_image = plot_3axis_Zaxis(render_image, yaw, pitch, roll, 
                tdx=tracking_joint[0], tdy=tracking_joint[1], size=max(y2-y1, x2-x1)*0.5, thickness=2)
            
            cur_joint_pose = [is_right, [pitch, yaw, roll], rot_mat, tracking_joint, tracking_joint_3d]
            tracking_joints_poses.append(cur_joint_pose)
            if real_frame_id in keyframe_id_dict:  # for finding real 3d points (may taking 0.3 seconds)
                left_right_flag = keyframe_id_dict[real_frame_id]
                # if (is_right and left_right_flag == "R") or (not is_right and left_right_flag == "L"):
                if (is_right and "R" in left_right_flag) or (not is_right and "L" in left_right_flag):
                    tracking_joints_poses[-1].append(1)  # keyframe, we will calculate its real 3d location
                    gripper = gripper_state_dict[real_frame_id]  # string. need to be converted into int
                    tracking_joints_poses[-1].append(gripper)  # gripper state. "1": open, "0": close 
                else:
                    tracking_joints_poses[-1].append(0)  # keyframe but with the another hand
                    tracking_joints_poses[-1].append(-1)  # gripper state. None
            else:
                tracking_joints_poses[-1].append(0)  # not keyframe, do not calculate its real 3d location
                tracking_joints_poses[-1].append(-1)  # gripper state. None
            tracking_joints_poses[-1].append([frame_lp, frame_rp, real_frame_id])  # other_info for running igev in another conda env
            
        for idx, [is_right, _, _, tracking_joint, _, _, _, _] in enumerate(tracking_joints_poses[::-1]):
            if idx == 60:  break
            color = (0, 255, 255) if is_right else (255, 255, 0)
            cv2.circle(render_image, (int(tracking_joint[0]), int(tracking_joint[1])), 5, color, -1)
        for idx, [is_right, hand_pose, rot_mat, point_2d, _, kf_flag, _, _] in enumerate(tracking_joints_poses):
            if kf_flag:  # save these info of keyframe for running in real robot (e.g., Aubo or Rokae)
                [pitch, yaw, roll] = hand_pose
                color = (0, 255, 255) if is_right else (255, 255, 0)
                render_image = plot_3axis_Zaxis(render_image, yaw, pitch, roll, 
                    tdx=point_2d[0], tdy=point_2d[1], size=60, thickness=6)
                cv2.circle(render_image, (int(point_2d[0]), int(point_2d[1])), 9, color, -1)
                cv2.circle(render_image, (int(point_2d[0]), int(point_2d[1])), 6, (0,0,0), -1)
                cv2.circle(render_image, (int(point_2d[0]), int(point_2d[1])), 3, (255,255,255), -1)


        if real_frame_id in showing_ids_list:
            left_hand_pos, right_hand_pos = [], []
            for idx, [is_right, hand_pose, _, point_2d, _, kf_flag, _, _] in enumerate(tracking_joints_poses):
                color_value = 0.55 + 0.39*idx/len(frames_L)  # from 0.55 to 0.99
                covl = int(color_value*255)
                color = (0, covl, covl) if is_right else (covl, covl, 0)
                start_point = (int(point_2d[0]), int(point_2d[1]))
                if is_right:
                    if len(right_hand_pos) != 0:
                        end_point = (right_hand_pos[0], right_hand_pos[1])
                        cv2.line(render_image_canvas, start_point, end_point, color, 12)
                    right_hand_pos = [int(point_2d[0]), int(point_2d[1])]
                else:
                    if len(left_hand_pos) != 0:
                        end_point = (left_hand_pos[0], left_hand_pos[1])
                        cv2.line(render_image_canvas, start_point, end_point, color, 12)
                    left_hand_pos = [int(point_2d[0]), int(point_2d[1])]
                cv2.circle(render_image_canvas, start_point, 12, color, -1)        
                        
            [is_right, [pitch, yaw, roll], _, tracking_joint, _, _, _, _] = tracking_joints_poses[-1]
            render_image_canvas = plot_3axis_Zaxis(render_image_canvas, yaw, pitch, roll, 
                tdx=tracking_joint[0], tdy=tracking_joint[1], size=100, thickness=10)
            cv2.circle(render_image_canvas, (int(tracking_joint[0]), int(tracking_joint[1])), 8, (0,0,0), -1)
            cv2.circle(render_image_canvas, (int(tracking_joint[0]), int(tracking_joint[1])), 4, (255,255,255), -1)
            
            [is_right, [pitch, yaw, roll], _, tracking_joint, _, _, _, _] = tracking_joints_poses[-2]
            render_image_canvas = plot_3axis_Zaxis(render_image_canvas, yaw, pitch, roll, 
                tdx=tracking_joint[0], tdy=tracking_joint[1], size=100, thickness=10)
            cv2.circle(render_image_canvas, (int(tracking_joint[0]), int(tracking_joint[1])), 8, (0,0,0), -1)
            cv2.circle(render_image_canvas, (int(tracking_joint[0]), int(tracking_joint[1])), 4, (255,255,255), -1)

            cv2.imwrite(output_path.replace("_v2.mp4", f"_v2_{frame_l}"), render_image_canvas)
        
        if real_frame_id == 132 and "drawer" in video_id_str:  # for paper writing
            for i, out in enumerate(outputs):
                is_right = out['is_right']
                hand_bbox = out['hand_bbox']
                [x0, y0, x1, y1] = hand_bbox
                cv2.rectangle(render_image_canvas, (int(x0), int(y0)), (int(x1), int(y1)), (255,255,255), 3)
                # text_str = "Right" if is_right else "Left"
                # cv2.putText(render_image_canvas, text_str, (int(x0), int(y0-6)), 
                #     cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)
            cv2.imwrite(output_path.replace("_v2.mp4", f"_v2_pure_{frame_l}"), render_image_canvas)
            
            
        # Write the frame to the output video
        vout.write(render_image)
        print(f"Processed frame {frame_count}")

    # Release everything
    vout.release()
    cv2.destroyAllWindows()
    
    with open(output_path.replace("_v2.mp4", "_v2.pkl"), "wb") as fp:  # Pickling
        pickle.dump(tracking_joints_poses, fp)


    if "drawer" in video_id_str:  # for paper writing
        image_canvas = cv2.imread(os.path.join(paired_path_L, frames_L[0]))
        image_canvas_l = image_canvas.copy()
        image_canvas_r = image_canvas.copy()
        empety_canvas = np.ones((1024, 1280, 3)) * 255
        empety_canvas_l = empety_canvas.copy()
        empety_canvas_r = empety_canvas.copy()
        left_hand_pos, right_hand_pos = [], []
        for idx, [is_right, hand_pose, _, point_2d, _, kf_flag, _, _] in enumerate(tracking_joints_poses):
            color = (0, 255, 255) if is_right else (255, 255, 0)
            start_point = (int(point_2d[0]), int(point_2d[1]))
            if is_right:
                if kf_flag:
                    cv2.circle(image_canvas_r, start_point, 12, color, -1)
                    cv2.circle(image_canvas_r, start_point, 8, (0,0,0), -1)
                    cv2.circle(image_canvas_r, start_point, 4, (255,255,255), -1)
                    cv2.circle(empety_canvas_r, start_point, 12, color, -1)
                    cv2.circle(empety_canvas_r, start_point, 8, (0,0,0), -1)
                    cv2.circle(empety_canvas_r, start_point, 4, (255,255,255), -1)
            else:
                if kf_flag:
                    cv2.circle(image_canvas_l, start_point, 12, color, -1)
                    cv2.circle(image_canvas_l, start_point, 8, (0,0,0), -1)
                    cv2.circle(image_canvas_l, start_point, 4, (255,255,255), -1)
                    cv2.circle(empety_canvas_l, start_point, 12, color, -1)
                    cv2.circle(empety_canvas_l, start_point, 8, (0,0,0), -1)
                    cv2.circle(empety_canvas_l, start_point, 4, (255,255,255), -1)
            if kf_flag:
                cv2.circle(image_canvas, start_point, 12, color, -1)
                cv2.circle(image_canvas, start_point, 8, (0,0,0), -1)
                cv2.circle(image_canvas, start_point, 4, (255,255,255), -1)
                cv2.circle(empety_canvas, start_point, 12, color, -1)
                cv2.circle(empety_canvas, start_point, 8, (0,0,0), -1)
                cv2.circle(empety_canvas, start_point, 4, (255,255,255), -1)
        cv2.imwrite(output_path.replace("_v2.mp4", f"_v4_pure_LR_{frames_L[0]}"), image_canvas)
        cv2.imwrite(output_path.replace("_v2.mp4", f"_v4_pure_L_{frames_L[0]}"), image_canvas_l)
        cv2.imwrite(output_path.replace("_v2.mp4", f"_v4_pure_R_{frames_L[0]}"), image_canvas_r)
        cv2.imwrite(output_path.replace("_v2.mp4", f"_v4_empety_LR_{frames_L[0]}"), empety_canvas)
        cv2.imwrite(output_path.replace("_v2.mp4", f"_v4_empety_L_{frames_L[0]}"), empety_canvas_l)
        cv2.imwrite(output_path.replace("_v2.mp4", f"_v4_empety_R_{frames_L[0]}"), empety_canvas_r)
        for idx, [is_right, hand_pose, _, point_2d, _, kf_flag, _, _] in enumerate(tracking_joints_poses):
            color_value = 0.55 + 0.39*idx/len(frames_L)  # from 0.55 to 0.99
            covl = int(color_value*255)
            color = (0, covl, covl) if is_right else (covl, covl, 0)
            start_point = (int(point_2d[0]), int(point_2d[1]))
            if is_right:
                if len(right_hand_pos) != 0:
                    end_point = (right_hand_pos[0], right_hand_pos[1])
                    cv2.line(image_canvas, start_point, end_point, color, 12)
                    cv2.line(image_canvas_r, start_point, end_point, color, 12)
                    cv2.line(empety_canvas, start_point, end_point, color, 12)
                    cv2.line(empety_canvas_r, start_point, end_point, color, 12)
                right_hand_pos = [int(point_2d[0]), int(point_2d[1])]
                cv2.circle(image_canvas_r, start_point, 12, color, -1)   
                cv2.circle(empety_canvas_r, start_point, 12, color, -1)   
            else:
                if len(left_hand_pos) != 0:
                    end_point = (left_hand_pos[0], left_hand_pos[1])
                    cv2.line(image_canvas, start_point, end_point, color, 12)
                    cv2.line(image_canvas_l, start_point, end_point, color, 12)
                    cv2.line(empety_canvas, start_point, end_point, color, 12)
                    cv2.line(empety_canvas_l, start_point, end_point, color, 12)
                left_hand_pos = [int(point_2d[0]), int(point_2d[1])]
                cv2.circle(image_canvas_l, start_point, 12, color, -1)
                cv2.circle(empety_canvas_l, start_point, 12, color, -1)
            cv2.circle(image_canvas, start_point, 12, color, -1)     
            cv2.circle(empety_canvas, start_point, 12, color, -1)   
        cv2.imwrite(output_path.replace("_v2.mp4", f"_v2_pure_LR_{frames_L[0]}"), image_canvas)
        cv2.imwrite(output_path.replace("_v2.mp4", f"_v2_pure_L_{frames_L[0]}"), image_canvas_l)
        cv2.imwrite(output_path.replace("_v2.mp4", f"_v2_pure_R_{frames_L[0]}"), image_canvas_r)
        cv2.imwrite(output_path.replace("_v2.mp4", f"_v2_empety_LR_{frames_L[0]}"), empety_canvas)
        cv2.imwrite(output_path.replace("_v2.mp4", f"_v2_empety_L_{frames_L[0]}"), empety_canvas_l)
        cv2.imwrite(output_path.replace("_v2.mp4", f"_v2_empety_R_{frames_L[0]}"), empety_canvas_r)
        for idx, [is_right, hand_pose, _, point_2d, _, kf_flag, _, _] in enumerate(tracking_joints_poses):
            color = (0, 255, 255) if is_right else (255, 255, 0)
            start_point = (int(point_2d[0]), int(point_2d[1]))
            if is_right:
                if kf_flag:
                    cv2.circle(image_canvas_r, start_point, 12, color, -1)
                    cv2.circle(image_canvas_r, start_point, 8, (0,0,0), -1)
                    cv2.circle(image_canvas_r, start_point, 4, (255,255,255), -1)
                    cv2.circle(empety_canvas_r, start_point, 12, color, -1)
                    cv2.circle(empety_canvas_r, start_point, 8, (0,0,0), -1)
                    cv2.circle(empety_canvas_r, start_point, 4, (255,255,255), -1)
            else:
                if kf_flag:
                    cv2.circle(image_canvas_l, start_point, 12, color, -1)
                    cv2.circle(image_canvas_l, start_point, 8, (0,0,0), -1)
                    cv2.circle(image_canvas_l, start_point, 4, (255,255,255), -1)
                    cv2.circle(empety_canvas_l, start_point, 12, color, -1)
                    cv2.circle(empety_canvas_l, start_point, 8, (0,0,0), -1)
                    cv2.circle(empety_canvas_l, start_point, 4, (255,255,255), -1)
            if kf_flag:
                cv2.circle(image_canvas, start_point, 12, color, -1)
                cv2.circle(image_canvas, start_point, 8, (0,0,0), -1)
                cv2.circle(image_canvas, start_point, 4, (255,255,255), -1)
                cv2.circle(empety_canvas, start_point, 12, color, -1)
                cv2.circle(empety_canvas, start_point, 8, (0,0,0), -1)
                cv2.circle(empety_canvas, start_point, 4, (255,255,255), -1)
        cv2.imwrite(output_path.replace("_v2.mp4", f"_v3_pure_LR_{frames_L[0]}"), image_canvas)
        cv2.imwrite(output_path.replace("_v2.mp4", f"_v3_pure_L_{frames_L[0]}"), image_canvas_l)
        cv2.imwrite(output_path.replace("_v2.mp4", f"_v3_pure_R_{frames_L[0]}"), image_canvas_r)
        cv2.imwrite(output_path.replace("_v2.mp4", f"_v3_empety_LR_{frames_L[0]}"), empety_canvas)
        cv2.imwrite(output_path.replace("_v2.mp4", f"_v3_empety_L_{frames_L[0]}"), empety_canvas_l)
        cv2.imwrite(output_path.replace("_v2.mp4", f"_v3_empety_R_{frames_L[0]}"), empety_canvas_r)
    
        
    print(f"Video processing complete. Output saved to {output_path}")


if __name__ == '__main__':
    
    # test_wilor_image_pipeline()
    
    
    # test_wilor_video_pipeline()


    # test_wilor_paired_pipeline("drawer_04")
    # test_wilor_paired_pipeline("pouring_05")
    # test_wilor_paired_pipeline("uncover_02")
    # test_wilor_paired_pipeline("unscrew_01")
    # test_wilor_paired_pipeline("openbox_01")
    
    # test_wilor_paired_pipeline("invert_01")
    # test_wilor_paired_pipeline("redirect_02")
    # test_wilor_paired_pipeline("reorient_01")
    # test_wilor_paired_pipeline("dualpap_01")
    # test_wilor_paired_pipeline("insertpen_01")
    # test_wilor_paired_pipeline("stacking_01")
    
    # test_wilor_paired_pipeline("pouring_06")
    test_wilor_paired_pipeline("unscrew_02")
    

