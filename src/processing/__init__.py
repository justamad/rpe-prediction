from .segmentation import segment_imu_signal, segment_kinect_signal
from .synchronization import calculate_cross_correlation, calculate_cross_correlation_with_datetime

from .utils import (
    get_joint_names_from_columns_as_list,
    remove_columns_from_dataframe,
    get_hsv_color,
    compute_mean_and_std_of_joint_for_subjects,
    get_all_columns_for_joint,
    align_skeleton_parallel_to_x_axis,
    check_angle_between_x_axis,
)

from .signal_processing import (
    normalize_signal,
    resample_data,
    identify_and_fill_gaps_in_data,
    apply_butterworth_filter,
    find_closest_timestamp,
    butterworth_filter_1d,
    calculate_magnitude,
    calculate_acceleration,
    apply_butterworth_1d_signal,
)

from .geometry import (
    create_rotation_matrix_y_axis,
    calculate_angle_in_radians_between_vectors,
    apply_affine_transformation,
)
