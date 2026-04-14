import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
import numpy as np

class StaticLaneLinesPublisher(Node):
    def __init__(self):
        super().__init__('static_road_line_map_node')
        self.publisher_ = self.create_publisher(MarkerArray, 'road_line_map', 10)

        # Prepare the static lane lines marker array once
        self.marker_array = self.create_lane_lines_marker_array()

        # Publish once on startup, or if needed, set a timer to publish repeatedly
        self.timer = self.create_timer(1.0, self.timer_callback)  # 1 Hz

    def create_lane_lines_marker_array(self):
        marker_array = MarkerArray()
        z = -4.5
        z1 = -4.3
        # Hardcoded static lane lines points
        lane_points = [
            #footh path lines
            [(0.7, 0.0, z1), (0.2, 50.0, z1)], # L1
            [(0.7, 0.0, z1), (1.2, -50.0, z1)], # L2
            [(4.3, 0.0, z1), (3.5, 50.0, z1)], # L3
            [(4.3, 0.0, z1), (5.6, -50.0, z1)], # L4
            [(11.0, 0.0, z1), (11.0, 17.0, z1)], # L5
            [(11.0, 0.0, z1), (12.6, -50.0, z1)], # L6
            [(13.7, 0.0, z1), (13.7, 16.5, z1)],  #L7
            [(13.7, 0.0, z1), (15.5, -50.0, z1)], #L8

            [(7.9, -1.0, z), (10.7, -1.0, z)],  #L9
            [(7.9, -1.0, z), (8.1, -11.4, z)],  #L10
            [(8.1, -14.0, z), (8.15, -17.0, z)], #L11
            [(8.15, -20.0, z), (8.2, -23.0, z)], #L12
            [(8.2, -26.0, z), (8.25, -29.0, z)], #L13

            [(4.6, 5.94, z), (7.62, 5.94, z)], #L14
            [(7.40, 16.19, z), (7.62, 5.94, z)], #L15
            [(7.43, 20.0, z), (7.43, 23.0, z)],  #L16

            [(11.15, -1.0, z1), (12.2, -1.0, z1)], #L17
            [(2.7, 5.94, z1), (4.1, 5.94, z1)], #L18

            [(4.6, 0.0, z), (10.7, 0.0, z)], #L19
            [(4.6, 4.5, z), (10.7, 4.5, z)], #L20

            [(9.7, -11.3, z), (11.1, -11.3, z)], #L21
            [(9.7, -11.3, z), (10.5, -12.8, z)], #L22
            [(10.9, -48.6, z), (12.3, -48.6, z)], #L23
            [(10.9, -48.6, z), (11.7, -47.1, z)], #L24

            [(12.0, 34.0, z1), (12.45, 50.0, z1)], #L25
            [(14.5, 34.0, z1), (15.0, 50.0, z1)], #L26

            [(13.7, 16.5, z1), (18.0, 19.4, z1)], #L27
            [(18.0, 19.4, z1), (25.0, 19.4, z1)], #L28
            [(14.5, 34.0, z1), (17.5, 29.7, z1)], #L29
            [(17.5, 29.7, z1), (25.0, 29.7, z1)], #L30

            [(7.43, 26.0, z), (7.43, 29.0, z)],  #L31
            [(7.43, 32.0, z), (7.53, 35.0, z)],  #L32
            [(7.53, 38.0, z), (7.63, 41.0, z)],  #L33
            [(7.63, 44.0, z), (7.73, 47.0, z)],  #L34

            [(12.0, 34.0, z1), (13.25, 30.6, z1)], #L35
            [(13.25, 30.6, z1), (17.5, 27.5, z1)], #L36
            [(17.5, 27.5, z1), (25.0, 27.5, z1)], #L37

            [(11.0, 17.0, z1), (12.7, 20.0, z1)], #L38
            [(12.7, 20.0, z1), (13.7, 21.0, z1)], #L39
            [(13.7, 21.0, z1), (18.0, 21.5, z1)], #L40
            [(18.0, 21.5, z1), (25.0, 21.5, z1)], #L41

            [(1.8, 50.0, z1), (3.4, -50.0, z1)], #L42
            [(12.4, 0.0, z1), (13.8, -50.0, z1)], #L43
            [(12.4, 0.0, z1), (12.2, 19.25, z1)], #L44

            [(11.0, 17.0, z), (12.0, 34.0, z)], #L45
            [(12.2, 19.25, z), (13.0, 31.0, z)], #L46
            [(13.0, 31.0, z1), (13.75, 50.0, z1)], #L47

            [(13.5, 21.25, z), (14.0, 29.75, z)], #L48

            [(8.3, -32.0, z), (8.4, -35.0, z)], #L49
            [(8.4, -38.0, z), (8.5, -41.0, z)], #L50
            [(8.5, -44.0, z), (8.55, -47.0, z)], #L51
        ]



        for i, points in enumerate(lane_points):
            marker = Marker()
            marker.header.frame_id = 'sys_world'
            marker.ns = 'static_lane_lines'
            marker.id = i
            marker.type = Marker.LINE_STRIP
            marker.action = Marker.ADD
            marker.scale.x = 0.05  # line width
            marker.color.a = 1.0  # fully opaque
            marker.color.r = 1.0
            marker.color.g = 1.0
            marker.color.b = 1.0

#Bus Lane(Lavender)
            if i in [20,21,22,23]:
                marker.color.r = 0.75
                marker.color.g = 0.5
                marker.color.b = 1.0
#Road lines(amber)
            if i in [9,10,11,12,14,15,30,31,32,33,48,49,50]:
                marker.color.r = 1.0
                marker.color.g = 0.8
                marker.color.b = 0.0
#road stop lines(spring green)
            if i in [47,13,18,16,17,19,8]:
                marker.color.r = 0.0
                marker.color.g = 1.0
                marker.color.b = 0.6
#bicycle lane(Crimson Red)
            if i in [41,42,43,46,44,45]:
                marker.color.r = 0.0
                marker.color.g = 0.55
                marker.color.b = 0.55
#Edges(Deep Sky Blue)
            if i in [0,1,7,6,25,28,29,34,35,36,37,38,39,40,26,27, 2,3,4,5,24]: 
                marker.color.r = 0.0
                marker.color.g = 0.6
                marker.color.b = 1.0
                

            for p in points:
                pt = Point()
                pt.x, pt.y, pt.z = p
                marker.points.append(pt)

            marker_array.markers.append(marker)

            # Create label text marker
            text_marker = Marker()
            text_marker.header.frame_id = 'sys_world'
            text_marker.ns = 'static_lane_line_labels'
            text_marker.id = i + 100  # Unique id
            text_marker.type = Marker.TEXT_VIEW_FACING
            text_marker.action = Marker.ADD
            text_marker.scale.z = 0.4  # Text height
            text_marker.color.r = 1.0
            text_marker.color.g = 1.0
            text_marker.color.b = 1.0
            text_marker.color.a = 1.0  # Fully opaque

            #BUS
            if i==6:
                bus_marker = Marker()
                bus_marker.header.frame_id = 'sys_world'
                bus_marker.ns = 'static_lane_line_labels'
                bus_marker.id = i + 1000  # Unique id
                bus_marker.type = Marker.TEXT_VIEW_FACING
                bus_marker.action = Marker.ADD
                bus_marker.scale.z = 2.0  # Text height
                bus_marker.color.r = 0.75
                bus_marker.color.g = 0.5
                bus_marker.color.b = 1.0
                bus_marker.color.a = 1.0  # Fully opaque
                p1 = np.array(first_point)
                p2 = np.array(second_point)
                mid = (p1 + p2) / 2.0

                # Optional offset upward (in road direction)
                offset = np.array([-1.5, 0.0, 0])  # move 1 meter in +Y direction

                bus_marker.pose.position.x = mid[0] + offset[0]
                bus_marker.pose.position.y = mid[1] + offset[1]
                bus_marker.pose.position.z = mid[2]  # lift slightly above ground

                bus_marker.text = "B    U    S"

                marker_array.markers.append(bus_marker)


            #crossing
            if i==19:
                c_marker = Marker()
                c_marker.header.frame_id = 'sys_world'
                c_marker.ns = 'static_lane_line_labels'
                c_marker.id = i + 1000  # Unique id
                c_marker.type = Marker.TEXT_VIEW_FACING
                c_marker.action = Marker.ADD
                c_marker.scale.z = 0.7  # Text height
                c_marker.color.r = 0.0
                c_marker.color.g = 1.0
                c_marker.color.b = 0.6
                c_marker.color.a = 1.0  # Fully opaque
                p1 = np.array(first_point)
                p2 = np.array(second_point)

                c_marker.pose.position.x = 7.75
                c_marker.pose.position.y = 1.9
                c_marker.pose.position.z = 0.5  # lift slightly above ground

                c_marker.text = "C\nR\nO\nS\nS\nI\nN\nG"

                marker_array.markers.append(c_marker)



            # Position text at the first point of the lane line
            first_point = points[0]
            second_point = points[1]
            text_marker.pose.position.x = (first_point[0] + second_point[0])/2
            text_marker.pose.position.y = (first_point[1] + second_point[1]) / 2
            text_marker.pose.position.z = first_point[2] + 0.5

            text_marker.text = f"L{i+1}"

            #marker_array.markers.append(text_marker)

            # Create black points marker for highlighting points
            points_marker = Marker()
            points_marker.header.frame_id = 'sys_world'
            points_marker.ns = 'static_lane_points'
            points_marker.id = i + 200  # Unique id for points
            points_marker.type = Marker.SPHERE_LIST
            points_marker.action = Marker.ADD
            points_marker.scale.x = 0.3  # sphere diameter
            points_marker.scale.y = 0.2
            points_marker.scale.z = 0.2
            points_marker.color.r = 1.0
            points_marker.color.g = 1.0
            points_marker.color.b = 1.0
            points_marker.color.a = 1.0  # fully opaque

            for p in points:
                pt = Point()
                pt.x, pt.y, pt.z = p
                points_marker.points.append(pt)

            #marker_array.markers.append(points_marker)

        cube_marker = Marker()
        cube_marker.header.frame_id = "sys_world"
        cube_marker.ns = "static_cube"
        cube_marker.id = 99990
        cube_marker.type = Marker.CUBE
        cube_marker.action = Marker.ADD
        cube_marker.scale.x = 1.0
        cube_marker.scale.y = 1.0
        cube_marker.scale.z = 1.0
        cube_marker.pose.position.x = 0.0
        cube_marker.pose.position.y = 0.0
        cube_marker.pose.position.z = -4.5
        cube_marker.color.r = 1.0
        cube_marker.color.g = 0.99
        cube_marker.color.b = 0.816
        cube_marker.color.a = 1.0 

        marker_array.markers.append(cube_marker)

        return marker_array

    def timer_callback(self):
        # Update header timestamp and publish the static markers
        now = self.get_clock().now().to_msg()
        for marker in self.marker_array.markers:
            marker.header.stamp = now
        self.publisher_.publish(self.marker_array)
        self.get_logger().info('Published static lane lines')

def main(args=None):
    rclpy.init(args=args)
    node = StaticLaneLinesPublisher()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
