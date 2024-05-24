import numpy as np
import rclpy
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from scipy.spatial.transform import Rotation

def create_bbox_marker(bbox, frameID, markerArray, label=None, classNames=None, score=None,
                        withLine=False, colour=[0.58, 0.58, 0.58], a=0.2, lifetime=None, twoDim=False):
    index = len(markerArray.markers)

    if twoDim:
        boxPosX = bbox[0]
        boxPosY = bbox[1]
        boxPosZ = 0
        boxSizeX = bbox[2]
        boxSizeY = bbox[3]
        boxSizeZ = 1
        boxPosRY = bbox[4]
    else:
        boxPosX = bbox[0]
        boxPosY = bbox[1]
        boxPosZ = bbox[2]
        boxSizeX = bbox[3]
        boxSizeY = bbox[4]
        boxSizeZ = bbox[5]
        boxPosRY = bbox[6]

    rot = Rotation.from_euler('xyz', [0, 0, boxPosRY], degrees=False)
    orientationQuat = rot.as_quat()

    markerBox = Marker()
    markerBox.id = index
    markerBox.header.frame_id = frameID
    markerBox.type = Marker.CUBE
    markerBox.action = Marker.ADD
    if lifetime is not None:
        markerBox.lifetime = rclpy.duration.Duration(seconds=lifetime)
    markerBox.pose.position.x = boxPosX
    markerBox.pose.position.y = boxPosY
    markerBox.pose.position.z = boxPosZ
    markerBox.pose.orientation.x = orientationQuat[0]
    markerBox.pose.orientation.y = orientationQuat[1]
    markerBox.pose.orientation.z = orientationQuat[2]
    markerBox.pose.orientation.w = orientationQuat[3]
    markerBox.scale.x = boxSizeX
    markerBox.scale.y = boxSizeY
    markerBox.scale.z = boxSizeZ
    markerBox.color.a = a
    markerBox.color.r = colour[0]
    markerBox.color.g = colour[1]
    markerBox.color.b = colour[2]

    markerArray.markers.append(markerBox)

    if label is not None or score is not None:
        markerLabel = Marker()
        markerLabel.id = index + 1
        markerLabel.header.frame_id = frameID
        markerLabel.type = Marker.TEXT_VIEW_FACING
        markerLabel.action = Marker.ADD
        if lifetime is not None or lifetime is False:
            markerLabel.lifetime = rclpy.duration.Duration(seconds=lifetime)
        markerLabel.pose.position.x = boxPosX
        markerLabel.pose.position.y = boxPosY
        markerLabel.pose.position.z = boxPosZ + 0.5 * abs(boxSizeZ)
        markerLabel.scale.z = 2
        markerLabel.color.a = 1.0
        markerLabel.color.r = 0.58
        markerLabel.color.g = 0.58
        markerLabel.color.b = 0.58

        if classNames is not None:
            if isinstance(label, str):
                label = label
            else:
                label = classNames[label]

        if score is None:
            markerLabel.text = 'ID: {}'.format(label)
        else:
            markerLabel.text = 'Class: {}\n Score: {:.2f}'.format(label, score)
        markerArray.markers.append(markerLabel)

    if withLine:
        markerLine = Marker()
        if label is not None or score is not None:
            markerLine.id = index + 2
        else:
            markerLine.id = index + 1
        markerLine.header.frame_id = frameID
        markerLine.type = Marker.LINE_LIST
        markerLine.action = Marker.ADD
        if lifetime is not None or lifetime is False:
            markerLine.lifetime = rclpy.duration.Duration(seconds=lifetime)
        markerLine.pose.position.x = boxPosX
        markerLine.pose.position.y = boxPosY
        markerLine.pose.position.z = boxPosZ
        markerLine.pose.orientation.x = orientationQuat[0]
        markerLine.pose.orientation.y = orientationQuat[1]
        markerLine.pose.orientation.z = orientationQuat[2]
        markerLine.pose.orientation.w = orientationQuat[3]
        markerLine.scale.x = 0.05
        markerLine.color.a = 1.0
        markerLine.color.r = 1.0
        markerLine.color.g = 1.0
        markerLine.color.b = 1.0

        bLowLeft = Point()
        bLowLeft.x = -0.5 * boxSizeX
        bLowLeft.y = -0.5 * boxSizeY
        bLowLeft.z = -0.5 * boxSizeZ
        # ... (similar assignments for other points)

        markerLine.points.append(bLowLeft)
        # ... (similar append statements for other points)
        markerArray.markers.append(markerLine)


def main(args=None):
    rclpy.init(args=args)
    node = rclpy.create_node('pcdet_node')  # You might want to change the node name
    marker_array = MarkerArray()

    # Example usage
    bbox = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.0]
    frame_id = "base_link"
    create_bbox_marker(bbox, frame_id, marker_array, label="example_label", classNames=["class1", "class2"], score=0.8, withLine=True)

    # Publish the marker array
    marker_publisher = node.create_publisher(MarkerArray, 'visualization_marker_array', 10)
    marker_publisher.publish(marker_array)

    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
