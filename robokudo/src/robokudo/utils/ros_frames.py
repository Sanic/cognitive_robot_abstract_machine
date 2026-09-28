"""
Normalize ROS frame identifiers at camera input boundaries.
"""

from robokudo.exceptions import InvalidCameraObservation


def normalize_ros_frame_id(frame_id: str) -> str:
    """
    Return a frame identifier without ROS's optional leading slash.
    """
    normalized = frame_id.lstrip("/")
    if not normalized:
        raise InvalidCameraObservation(reason="ROS frame identifier is empty")
    return normalized
