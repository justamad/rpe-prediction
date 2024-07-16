from typing import Optional, Tuple
from argparse import ArgumentParser
from pyk4a import PyK4APlayback, ImageFormat
from os.path import join
from pathlib import Path
from tqdm import tqdm

import cv2
import numpy as np
import os

parser = ArgumentParser(description="pyk4a player")
parser.add_argument("--seek", type=float, help="Seek file to specified offset in seconds", default=0.0)
parser.add_argument("--path", type=str, help="Seek file to specified offset in seconds", default="G:\\Azure_Kinect_RPE")
parser.add_argument("--dst", type=str, help="Seek file to specified offset in seconds", default="G:\\output")
args = parser.parse_args()


def convert_to_bgra_if_required(color_format: ImageFormat, color_image):
    # examples for all possible pyk4a.ColorFormats
    if color_format == ImageFormat.COLOR_MJPG:
        color_image = cv2.imdecode(color_image, cv2.IMREAD_COLOR)
    elif color_format == ImageFormat.COLOR_NV12:
        color_image = cv2.cvtColor(color_image, cv2.COLOR_YUV2BGRA_NV12)
        # this also works and it explains how the COLOR_NV12 color color_format is stored in memory
        # h, w = color_image.shape[0:2]
        # h = h // 3 * 2
        # luminance = color_image[:h]
        # chroma = color_image[h:, :w//2]
        # color_image = cv2.cvtColorTwoPlane(luminance, chroma, cv2.COLOR_YUV2BGRA_NV12)
    elif color_format == ImageFormat.COLOR_YUY2:
        color_image = cv2.cvtColor(color_image, cv2.COLOR_YUV2BGRA_YUY2)
    return color_image


def colorize(
    image: np.ndarray,
    clipping_range: Tuple[Optional[int], Optional[int]] = (None, None),
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    if clipping_range[0] or clipping_range[1]:
        img = image.clip(clipping_range[0], clipping_range[1])
    else:
        img = image.copy()
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    img = cv2.applyColorMap(img, colormap)
    return img


def info(playback: PyK4APlayback):
    print(f"Record length: {playback.length / 1000000: 0.2f} sec")


def play_mkv_file(playback: PyK4APlayback, file_path: str):
    while True:
        try:
            capture = playback.get_next_capture()
            # if capture.color is not None:
            #     cv2.imshow("Color", convert_to_bgra_if_required(playback.configuration["color_format"], capture.color))
            if capture.depth is not None:
                # cv2.imshow("Depth", colorize(capture.depth, (None, 5000)))
                # cv2.imwrite(join(file_path, f"{capture.color_timestamp_usec}.png"), colorize(capture.depth, (None, 5000)))
                cv2.imwrite(join(file_path, f"{capture.color_timestamp_usec}.png"), capture.depth)
                # cv2.imshow("Depth", capture.depth)
            # key = cv2.waitKey(10)
            # if key != -1:
                # break
        except EOFError:
            break
    # cv2.destroyAllWindows()


def process_video(file_name: str):
    offset: float = args.seek

    # filename = "E:\\2021_Fatigue_Study\\2021-04-15_Arne_FlyWheel_Pilot\\master_0.mkv"
    # filename = "E:\\2021_Fatigue_Study\\2021-04-15_Arne_FlyWheel_Pilot\\sub_0.mkv"
    head, tail = os.path.split(file_name)
    tail = tail.replace(".mkv", "")
    dst_path = join(head, tail)

    if os.path.exists(dst_path):
        print(f"Skipping {file_name}")
        return

    os.makedirs(dst_path, exist_ok=True)

    playback = PyK4APlayback(file_name)
    playback.open()

    # info(playback)
    with open(join(dst_path, "../src/scripts/calibration.json"), "w") as f:
        f.write(playback.calibration_raw)

    if offset != 0.0:
        playback.seek(int(offset * 1000000))
    play_mkv_file(playback, file_path=dst_path)

    playback.close()


if __name__ == "__main__":
 #    all_files = list(Path(args.path).rglob('*.mkv'))
 #    print(all_files)
 #    for file_name in tqdm(all_files):
 #        process_video(str(file_name))
 #
    # process_video("C:\\Users\\Justin\\Desktop\\4-7_kmh_02.mkv")
    process_video("test.mkv")
