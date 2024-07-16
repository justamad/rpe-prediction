import cv2
import numpy as np
import open3d as o3d
import pykinect_azure as pykinect

from os import makedirs
from os.path import join


def normalize_depth_image(depth_image: np.ndarray):
    black_mask = depth_image == 0
    depth_image = ((1 - depth_image / depth_image.max()) * 255).astype(np.uint8)
    depth_image[black_mask] = 0
    return depth_image


def extract_rgb_and_depth(video_file: str, max_frame: int):
    dst_path = join(video_file.replace(".mkv", ""))
    makedirs(dst_path, exist_ok=True)

    pykinect.initialize_libraries(track_body=True)
    playback = pykinect.start_playback(video_file)
    playback_config = playback.get_record_configuration()
    playback_calibration = playback.get_calibration()

    count = 0
    while True:
        if count >= max_frame:
            break

        ret, capture = playback.update()
        if not ret:
            break

        ret_depth, depth_image = capture.get_depth_image()
        depth_image = normalize_depth_image(depth_image)

        ret_pc, point_cloud = capture.get_pointcloud()
        pcd = o3d.geometry.PointCloud()

        distances = np.linalg.norm(point_cloud, axis=1)
        normalized_distances = (distances - np.min(distances)) / (np.max(distances) - np.min(distances))
        # Convert normalized distances to grayscale values
        grayscale_values = (normalized_distances * 255).astype(np.uint8)

        # Create a color array using grayscale values
        colors = np.ones_like(point_cloud, dtype=np.float64) * grayscale_values[:, None]

        pcd.points = o3d.utility.Vector3dVector(point_cloud)
        # pcd.colors = o3d.utility.Vector3dVector(colors)

        o3d.visualization.draw_geometries([pcd],
                                          zoom=0.3412,
                                          front=[0.4257, -0.2125, -0.8795],
                                          lookat=[2.6172, 2.0475, 1.532],
                                          up=[-0.0694, -0.9768, 0.2024])

        # o3d.io.write_point_cloud("./data.ply", pcd)

        # ret_depth, depth_color_image = capture.get_colored_depth_image()

        if not ret_depth or not ret_pc:
            continue

        # Combine both images
        # combined_image = cv2.addWeighted(depth_color_image, 0.6, body_image_color, 0.4, 0)
        # combined_image = cv2.addWeighted(color_image[:, :, :3], 0.7, combined_image, 0.3, 0)

        # skeleton_image = np.zeros((color_image.shape[0], color_image.shape[1], 3), dtype=np.uint8)
        # skeleton_image = body_frame.draw_bodies(skeleton_image, pykinect.K4A_CALIBRATION_TYPE_COLOR)
        # color = (0, 0, 0)
        # mask = np.where((skeleton_image == color).all(axis=2), 0, 255)
        # skeleton_image = cv2.cvtColor(skeleton_image, cv2.COLOR_BGR2BGRA)
        # skeleton_image[:, :, 3] = mask

        cv2.imwrite(join(dst_path, f"{count:03d}_depth.png"), depth_image)
        # cv2.imwrite(join(dst_path, f"{count:03d}_skele.png"), skeleton_image)
        print(f"Saved frame {count}")
        count += 1

        # cv2.imshow('Depth image with skeleton', color_image)
        # Press q key to stop
        # if cv2.waitKey(1) == ord('q'):
            # break


def render_video(video_file: str, max_frame: int):
    dst_path = join(video_file.replace(".mkv", ""))
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
        ret_depth, depth_image = capture.get_depth_image()

        # ret_depth, depth_color_image = capture.get_colored_depth_image()

        if not ret_color or not ret_depth:
            continue

        # Combine both images
        # combined_image = cv2.addWeighted(depth_color_image, 0.6, body_image_color, 0.4, 0)
        # combined_image = cv2.addWeighted(color_image[:, :, :3], 0.7, combined_image, 0.3, 0)
        # depth_skeleton = np.zeros((depth_image.shape[0], depth_image.shape[1]), dtype=np.uint8)
        depth_image = normalize_depth_image(depth_image)
        depth_image = body_frame.draw_bodies(depth_image, pykinect.K4A_CALIBRATION_TYPE_DEPTH, only_segments=True)

        # skeleton_image = np.zeros((color_image.shape[0], color_image.shape[1], 3), dtype=np.uint8)
        color_image = body_frame.draw_bodies(color_image, pykinect.K4A_CALIBRATION_TYPE_COLOR, only_segments=True)
        # color = (0, 0, 0)
        # mask = np.where((skeleton_image == color).all(axis=2), 0, 255)
        # skeleton_image = cv2.cvtColor(skeleton_image, cv2.COLOR_BGR2BGRA)
        # skeleton_image[:, :, 3] = mask

        cv2.imwrite(join(dst_path, f"{count:03d}_color.png"), color_image)
        # cv2.imwrite(join(dst_path, f"{count:03d}_skele.png"), skeleton_image)
        cv2.imwrite(join(dst_path, f"{count:03d}_depth.png"), depth_image)
        print(f"Saved frame {count}")
        count += 1


if __name__ == "__main__":
    render_video("persons.mkv", max_frame=400)
    # extract_rgb_and_depth("test.mkv", max_frame=400)
