import cv2
import numpy as np
import pykinect_azure as pykinect

from os import makedirs
from os.path import join


def render_video(video_file: str, dst_path: str, max_frame: int):
    dst_path = join(dst_path, video_file)
    makedirs(dst_path, exist_ok=True)

    pykinect.initialize_libraries(track_body=True)
    playback = pykinect.start_playback(video_file)
    playback_config = playback.get_record_configuration()
    playback_calibration = playback.get_calibration()

    bodyTracker = pykinect.start_body_tracker(calibration=playback_calibration)
    count = 0
    while True:
        if count >= max_frame:
            break

        ret, capture = playback.update()
        if not ret:
            break

        body_frame = bodyTracker.update(capture=capture)
        ret_color, color_image = capture.get_color_image()
        # ret_depth, depth_color_image = capture.get_colored_depth_image()

        if not ret_color:
            continue

        # Combine both images
        # combined_image = cv2.addWeighted(depth_color_image, 0.6, body_image_color, 0.4, 0)
        # combined_image = cv2.addWeighted(color_image[:, :, :3], 0.7, combined_image, 0.3, 0)

        skeleton_image = np.zeros((color_image.shape[0], color_image.shape[1], 3), dtype=np.uint8)
        skeleton_image = body_frame.draw_bodies(skeleton_image, pykinect.K4A_CALIBRATION_TYPE_COLOR)
        color = (0, 0, 0)
        mask = np.where((skeleton_image == color).all(axis=2), 0, 255)
        skeleton_image = cv2.cvtColor(skeleton_image, cv2.COLOR_BGR2BGRA)
        skeleton_image[:, :, 3] = mask

        cv2.imwrite(join(dst_path, f"{count:03d}_color.png"), color_image)
        cv2.imwrite(join(dst_path, f"{count:03d}_skele.png"), skeleton_image)
        print(f"Saved frame {count}")
        count += 1

        # cv2.imshow('Depth image with skeleton', color_image)
        # Press q key to stop
        # if cv2.waitKey(1) == ord('q'):
            # break


if __name__ == "__main__":
    render_video("../../sub_5.mkv", dst_path="video", max_frame=400)
    render_video("../../master_5.mkv", dst_path="video", max_frame=400)
