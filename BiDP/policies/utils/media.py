"""
Utilities for loading, saving, and manipulating videos and images.

@yjy0625

"""

import os
import numpy as np
import cv2
import skvideo.io
import imageio


def _make_dir(filename):
    folder = os.path.dirname(filename)
    os.makedirs(folder, exist_ok=True)


def save_image(image, filename):
    _make_dir(filename)
    if np.max(image) < 2:
        image = np.array(image * 255)
    image = image.astype(np.uint8)
    cv2.imwrite(filename, image[..., ::-1])


def save_gif(video_array, file_path, fps=10):
    """
    Save a (T, H, W, 3) numpy array of video into a GIF file.

    Parameters:
        video_array (numpy.ndarray): The video as a 4D numpy array with shape (T, H, W, 3).
        file_path (str): The file path where the GIF will be saved.
        fps (int, optional): Frames per second for the GIF. Default is 10.
    """
    try:
        # Ensure the video array is uint8 (required for GIF)
        video_array = (255 * (1.0 - video_array)).astype("uint8")

        # Save the GIF
        imageio.mimsave(file_path, video_array, duration=len(video_array) / fps, loop=1)
        print(f"Saved GIF to {file_path}")
    except Exception as e:
        print(f"Error saving GIF: {e}")


def save_video(video_frames, filename, fps=10, video_format="mp4"):
    if len(video_frames) == 0:
        return False

    assert fps == int(fps), fps
    _make_dir(filename)

    skvideo.io.vwrite(
        filename,
        video_frames,
        inputdict={
            "-r": str(int(fps)),
        },
        outputdict={"-f": video_format, "-pix_fmt": "yuv420p"},
    )

    return True


def read_video(filename):
    return skvideo.io.vread(filename)


def get_video_framerate(filename):
    videometadata = skvideo.io.ffprobe(filename)
    frame_rate = videometadata["video"]["@avg_frame_rate"]
    return eval(frame_rate)


def add_caption_to_img(img, info, name=None, flip_rgb=False, num_lines=5):
    """Adds caption to an image. info is dict with keys and text/array.
    :arg name: if given this will be printed as heading in the first line
    :arg flip_rgb: set to True for inputs with BGR color channels
    """
    mul = 2.0
    offset = int(12 * mul)

    frame = img * 255.0 if img.max() <= 1.0 else img
    if flip_rgb:
        frame = frame[:, :, ::-1]

    # colors
    blue = (66, 133, 244)
    yellow = (255, 255, 0)
    white = (255, 255, 255)

    # add border to frame if success
    if "is_success" in info.keys() and info["is_success"]:
        border_size = int(10 * mul)
        frame[:border_size, :] = np.array(blue)[None][None]
        frame[-border_size:, :] = np.array(blue)[None][None]
        frame[:, :border_size] = np.array(blue)[None][None]
        frame[:, -border_size:] = np.array(blue)[None][None]

    fheight, fwidth = frame.shape[:2]
    pad_height = int(offset * (num_lines + 2))
    frame = np.concatenate([frame, np.zeros((pad_height, fwidth, 3))], 0)

    font_size = 0.4 * mul
    thickness = int(1 * mul)
    x, y = int(5 * mul), fheight + int(10 * mul)
    if name is not None:
        cv2.putText(
            frame,
            "[{}]".format(name),
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_size,
            yellow,
            thickness,
            cv2.LINE_AA,
        )
    for i, k in enumerate(info.keys()):
        v = info[k]
        if (
            type(v) == np.ndarray
            or type(v) == np.float64
            or type(v) == np.float32
            or type(v) == float
        ):
            if type(v) == np.ndarray:
                v = np.round(v, 3 if np.isscalar(v) or len(v) <= 3 else 2)
            else:
                v = np.round(v, 3)
        key_text = "{}: ".format(k)
        (key_width, _), _ = cv2.getTextSize(
            key_text, cv2.FONT_HERSHEY_SIMPLEX, font_size, thickness
        )

        cv2.putText(
            frame,
            key_text,
            (x, y + offset * (i + 2)),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_size,
            blue,
            thickness,
            cv2.LINE_AA,
        )

        cv2.putText(
            frame,
            str(v),
            (x + key_width, y + offset * (i + 2)),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_size,
            white,
            thickness,
            cv2.LINE_AA,
        )

    if flip_rgb:
        frame = frame[:, :, ::-1]

    return frame


def add_border_to_img(frame, color=(234, 67, 53)):
    border_size = 20
    if np.max(frame) <= 1.0:
        color = tuple([float(x) / 255 for x in color])
    frame[:border_size, :] = np.array(color)[None][None]
    frame[-border_size:, :] = np.array(color)[None][None]
    frame[:, :border_size] = np.array(color)[None][None]
    frame[:, -border_size:] = np.array(color)[None][None]
    return frame


def add_border_to_video(frames, color=(234, 67, 53)):
    border_size = 20
    if np.max(frames) <= 1.0:
        color = tuple([float(x) / 255 for x in color])
    frames[:, :border_size, :] = np.array(color)[None][None][None]
    frames[:, -border_size:, :] = np.array(color)[None][None][None]
    frames[:, :, :border_size] = np.array(color)[None][None][None]
    frames[:, :, -border_size:] = np.array(color)[None][None][None]
    return frames


def combine_videos(images, num_cols=5):
    if len(images) == 1:
        return np.array(images[0])
    max_frames = np.max([len(im) for im in images])
    images = [
        np.concatenate([im[:-1], np.array([im[-1]] * (max_frames - len(im) + 1))])
        for im in images
    ]
    images = np.array(images)
    B = images.shape[0]
    if B % num_cols != 0:
        images = np.concatenate(
            [images, np.zeros((num_cols - (B % num_cols),) + tuple(images.shape[1:]))]
        )
    B, T, H, W, C = images.shape
    images = images.reshape(B // num_cols, num_cols, T, H, W, C).transpose(
        2, 0, 3, 1, 4, 5
    )
    images = images.reshape(T, B // num_cols * H, num_cols * W, C)
    return images
