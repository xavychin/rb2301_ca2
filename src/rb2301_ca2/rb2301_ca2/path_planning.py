import argparse
import configparser
import json
import math
import os
import time

import numpy as np
import heapq
import rclpy
from rclpy.node import Node
from rclpy.logging import set_logger_level, LoggingSeverity

from rclpy.qos import (
    ReliabilityPolicy,
    QoSProfile,
)
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from PIL import Image
from geometry_msgs.msg import Twist


np.set_printoptions(
    2, suppress=True, threshold=np.inf
)  # Print numpy arrays to specified d.p., suppress scientific notation (e.g. 1e-5), and do not truncate

set_logger_level("waypoint", level=LoggingSeverity.DEBUG) # Configure to either LoggingSeverity.INFO or LoggingSeverity.DEBUG

occupancy_grid_resolution = 0.2 # Sim grid resolution, in metres per cell (the real maze uses the same 0.2m cells, see ca2_irl_layout.json)
max_translate_velocity = 1.4 # Overwritten in main() based on sim vs real-life; 0.4m/s cap for real life, please keep that in place

_PACKAGE_DIR = os.path.dirname(os.path.realpath(__file__))


# --- Coordinate conversion --------------------------------------------------
# A grid index (i, j) represents a CELL, not a point. That cell's world
# coordinate is its CENTER, e.g. cell [0, 0] is centred half a resolution-step
# away from the grid's origin corner, not exactly on it. This matches how the
# Gazebo world and the real maze are physically laid out (goal tape/markers
# sit in the middle of a cell, not on its boundary line).
def grid_to_world(i:int, j:int, origin:tuple, resolution:float=occupancy_grid_resolution) -> tuple:
    '''Convert grid index (i, j) to the world (x, y) coordinate of that cell's centre.'''
    return (origin[0] + (i + 0.5) * resolution, origin[1] + (j + 0.5) * resolution)

def world_to_grid(x:float, y:float, origin:tuple, resolution:float=occupancy_grid_resolution) -> tuple:
    '''Convert a world (x, y) coordinate to the grid index (i, j) of the cell containing it.'''
    return (int(np.floor((x - origin[0]) / resolution)), int(np.floor((y - origin[1]) / resolution)))


# --- Sim / real-life profiles -----------------------------------------------
# Run without arguments for the simulation. For the real maze pass
# --run test1 | test2 | full  (see README). Everything that differs between
# sim and real -- which map to load, the cell resolution, where the grid's
# [0, 0] corner sits in the frame your pose is reported in, the goal points,
# the speed cap -- lives in one of these profiles.
sim_config = {
    "map_file": "ca2_sim_map.npy",
    "origin": (-1.0, -5.0),
    "resolution": occupancy_grid_resolution,
    "goal_list": [(3.5, -3.5), (3.3, 0.3), (2.5, -3.5), (-0.3, -3.7)],
    "max_translate_velocity": 1.4,
}

IRL_RUNS = ("test1", "test2", "full")


# --- Real maze <-> Optitrack frame -------------------------------------------
# The real maze is a fixed layout (ca2_irl_map.npy + ca2_irl_layout.json, in
# "maze-local" metres: (0, 0) is the outer corner of array cell [0, 0], +x runs
# along array axis 0, +y along axis 1 -- exactly like the sim grid). Optitrack
# reports poses in ITS OWN frame, wherever the maze happens to be taped down.
# optitrack_variables.config holds the 3 numbers that relate the two:
#   origin_x, origin_y : Optitrack coordinates of the maze's (0, 0) corner
#   rotation_deg       : angle of the maze's +x axis in the Optitrack frame
#                        (counter-clockwise positive, degrees)
# The pose from Optitrack is converted into the maze frame before your code
# sees it, so self.origin is (0, 0) and everything else (world_to_grid,
# goal_list, ...) works exactly like in simulation.
def wrap_deg(angle:float) -> float:
    '''Wrap an angle in degrees into [-180, 180).'''
    return (angle + 180.0) % 360.0 - 180.0

def optitrack_to_maze(x:float, y:float, heading_deg:float, frame:dict) -> tuple:
    '''Optitrack-frame pose (x, y, heading in degrees) -> maze-frame pose.'''
    dx, dy = x - frame["origin_x"], y - frame["origin_y"]
    r = math.radians(frame["rotation_deg"])
    c, s = math.cos(r), math.sin(r)
    return (c * dx + s * dy, -s * dx + c * dy,
            wrap_deg(heading_deg - frame["rotation_deg"] - frame.get("heading_offset_deg", 0.0))) # offset: the Motive rigid body's "forward" vs the robot's front

def maze_to_optitrack(mx:float, my:float, heading_deg:float, frame:dict) -> tuple:
    '''Inverse of optitrack_to_maze (handy for working out where to place the robot).'''
    r = math.radians(frame["rotation_deg"])
    c, s = math.cos(r), math.sin(r)
    return (frame["origin_x"] + c * mx - s * my, frame["origin_y"] + s * mx + c * my,
            wrap_deg(heading_deg + frame["rotation_deg"] + frame.get("heading_offset_deg", 0.0)))

def robot_frame(config:dict, robot_number:int) -> dict:
    '''The frame for one robot: the shared origin/rotation plus that robot's heading offset (per-robot value from the config if listed).'''
    frame = dict(config["frame"])
    frame["heading_offset_deg"] = config["heading_offsets"].get(robot_number, frame["heading_offset_deg"])
    return frame

def load_irl_config(run:str) -> dict:
    '''Build the real-maze profile for run "test1", "test2" or "full".'''
    if run not in IRL_RUNS:
        raise ValueError(f"--run must be one of {IRL_RUNS}, got {run!r}")
    with open(os.path.join(_PACKAGE_DIR, "ca2_irl_layout.json")) as f:
        layout = json.load(f)
    parser = configparser.ConfigParser()
    config_path = os.path.join(_PACKAGE_DIR, "optitrack_variables.config")
    if not parser.read(config_path):
        raise FileNotFoundError(f"Could not read {config_path}")
    frame = {
        "origin_x": parser.getfloat("frame", "origin_x"),
        "origin_y": parser.getfloat("frame", "origin_y"),
        "rotation_deg": parser.getfloat("frame", "rotation_deg"),
    }
    # Rigid bodies are created in Motive with the robot facing the maze's +y direction, so the rigid body's zero heading IS maze +y
    # (maze heading 90 deg): heading_maze = reported + 90 = reported - rotation - offset  =>  offset = -(rotation + 90). "auto" means that.
    raw_offset = parser.get("frame", "heading_offset_deg", fallback="auto").strip().lower()
    frame["heading_offset_deg"] = wrap_deg(-(frame["rotation_deg"] + 90.0)) if raw_offset == "auto" else float(raw_offset)
    heading_offsets = {int(k): float(v) for k, v in parser.items("heading_offsets")} if parser.has_section("heading_offsets") else {}
    return {
        "map_file": "ca2_irl_map.npy",
        "origin": (0.0, 0.0), # pose is converted into the maze frame, so the grid corner is (0, 0)
        "resolution": float(layout["resolution"]),
        "goal_list": [tuple(g) for g in layout["runs"][run]["goals"]],
        "start": tuple(layout["runs"][run]["start"]),
        "frame": frame,
        "robot_number": parser.getint("robot", "number"),
        "heading_offsets": heading_offsets, # per-robot override of frame["heading_offset_deg"]
        "max_translate_velocity": 0.4, # Please keep this in place
    }


class WaypointNode(Node):
    '''Node to calculate path and move robot towards given goal_coordinates, using pose info from either gazebo odometer or optitrack'''
    def __init__(self, map_array:np.array, goal_list:list, is_simulation:bool=True, origin:tuple=(0.0, 0.0), resolution:float=occupancy_grid_resolution,
                 frame:dict=None, robot_number:int=3, expected_start:tuple=None, calibrate:bool=False):
        super().__init__('waypoint')
        self.get_logger().info("Starting WaypointNode")

        self.is_simulation = is_simulation

        # Subscribe to the dynamic_pose topic from Gazebo that publishes ground-truth pose data
        if self.is_simulation:
            self.subscription = self.create_subscription(Odometry, 'odom', self.odometer_callback, 2)
        else:
            qos_profile = QoSProfile(depth=2, reliability=ReliabilityPolicy.BEST_EFFORT)

            self.map_sub = self.create_subscription(
                PoseStamped,
                f'/vrpn_mocap/bingda_{robot_number:03d}/pose',
                self.optitrack_callback,
                qos_profile
                )

        self.publisher_ = self.create_publisher(Twist, 'cmd_vel', 10) # Publish to cmd_vel node
        self.timer = self.create_timer(0.05, self.timer_callback)  # Runs at 20Hz. Can be changed.

        self.goal_list = goal_list
        self.map_array = map_array
        self.origin = origin # World (x, y) coordinate of the grid's [0, 0] corner. Use with grid_to_world()/world_to_grid()
        self.resolution = resolution # Metres per grid cell (0.2). Use with grid_to_world()/world_to_grid()

        self.frame = frame # Optitrack->maze frame (real robot only), see optitrack_to_maze()
        self.expected_start = expected_start # Where this run says the robot should be placed (maze frame), real robot only
        self.calibrate = calibrate # Real robot: only print poses, never publish cmd_vel
        self.raw_pose = None # Latest pose exactly as Optitrack reported it (real robot only)
        self._start_checked = False
        self._calibrate_ticks = 0

        self.pose = None
        self.path = [] # Set this to your planned route (a list of grid-index tuples, in travel order) once you've computed it -- it'll automatically show up in the terminal map print
        self._last_printed_path = None

        ##Variables
        self.goal_index = 0    # Index into self.goal_list of the goal currently being driven to
        self.route = []        # Planned waypoints as world (x, y) points in travel order: corners only, last = the exact goal
        self.path_index = 0    # Index into self.route of the waypoint currently being driven to
        self.seg_start = None  # World (x, y) where the current straight segment starts
        self.rows, self.cols = self.map_array.shape

        # Forward speed
        self.kp = 2.0                 # (m/s) per metre left on the segment -> eases onto corners and the goal
        self.max_drive_speed = 0.4    # m/s (move_2D additionally clips to max_translate_velocity)
        self.min_drive_speed = 0.03   # m/s, so the real robot doesn't stall in its motor deadband just short of a point

        # Heading PID. Error in RADIANS, so the output is angular.z in rad/s
        self.kp_turn = 2.0            # (rad/s) per rad of heading error
        self.ki_turn = 0.0            # Leave at 0 unless the robot settles with a constant heading offset
        self.kd_turn = 0.15           # Damps overshoot when turning on the spot
        self.integral_limit = 0.5     # rad*s, anti-windup clamp on the I term
        self.min_turn_speed = 0.25    # rad/s floor when turning on the spot, so small turns still overcome motor deadband
        self.heading_integral = 0.0
        self.prev_heading_error = None
        self.last_time = time.time()

        # Path following
        self.lookahead = 0.15          # m: steer at a point this far ahead ON the segment, so the robot stays on the line
        self.end_zone = 0.05           # m: in the last few cm of a segment, stop chasing the point (its bearing swings wildly)
        self.steer_mode = None         # 'line', 'hold' or 'point' (see timer_callback)
        self.final_point_mode = False  # Latched for the last few cm to the goal: aim straight at the goal point

        # Turn-in-place vs drive-and-steer, with hysteresis so it doesn't flicker between the two
        self.turn_in_place_enter = math.radians(15)  # Stop and rotate on the spot if heading error grows above this
        self.turn_in_place_exit = math.radians(4)    # ...and keep rotating until it drops below this
        self.turning_in_place = True
        self.reversing = False                       # Final approach only: goal is behind -> back onto it

        # Arrival tolerances (metres)
        self.waypoint_tolerance = 0.005  # Corner counts as reached (5 mm)
        self.goal_tolerance = 0.005      # Goal counts as reached (5 mm)

    def _print_calibration(self):
        '''--calibrate: twice a second, print the raw Optitrack pose next to the maze-frame pose and grid cell.'''
        self._calibrate_ticks += 1
        if self._calibrate_ticks % 10:
            return
        ox, oy, oh = self.raw_pose
        mx, my, mh = self.pose
        i, j = world_to_grid(mx, my, self.origin, self.resolution)
        self.get_logger().info(f"optitrack ({ox:+.3f}, {oy:+.3f}, {oh:+7.1f} deg)  ->  maze ({mx:+.3f}, {my:+.3f}, {mh:+7.1f} deg)  cell [{i}, {j}]")

    def _check_start_once(self):
        '''Once, on the first pose: warn if the robot isn't near this run's start point (usually a placement or config mistake).'''
        if self._start_checked or self.expected_start is None:
            return
        self._start_checked = True
        dist = math.hypot(self.pose[0] - self.expected_start[0], self.pose[1] - self.expected_start[1])
        msg = f"Robot is at maze ({self.pose[0]:.2f}, {self.pose[1]:.2f}); this run starts at {self.expected_start} -- {dist:.2f} m away"
        if dist > 0.4:
            self.get_logger().warn(msg + ". Check where the robot is placed; if it is on the start point, ask a TA.")
        else:
            self.get_logger().info(msg + " (ok)")

    def print_map(self):
        '''Prints the occupancy grid to the terminal: walls, your current position ('S'), all goal points ('W'/'G'),
        and your planned route (self.path) if you've set one ('*'). Safe to call anytime pose is known; does nothing
        useful before then. Called automatically from timer_callback() whenever self.path changes.'''
        if self.pose is None:
            return
        shape = self.map_array.shape
        clip = lambda cell: (int(np.clip(cell[0], 0, shape[0]-1)), int(np.clip(cell[1], 0, shape[1]-1)))
        current_cell = clip(world_to_grid(self.pose[0], self.pose[1], self.origin, self.resolution))
        goal_cells = [clip(world_to_grid(gx, gy, self.origin, self.resolution)) for gx, gy in self.goal_list]
        grid = Grid(self.map_array, starting_position=current_cell, goal_position=goal_cells[-1])
        grid.print_grid_map(waypoints=goal_cells, path=self.path)

    def yaw_from_quaternion(self, q):
        '''Returns yaw angle (in rad) for orientation based on given quaternion input q'''
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        return np.arctan2(siny_cosp, cosy_cosp)

    def optitrack_callback(self, msg:PoseStamped):
        '''Callback to calculate 2D pose info from Optitrack node. Pose info includes x and y coordinates, as well as heading in degrees.
        This callback will run everytime the rclpy executor spins'''
        x, y = msg.pose.position.x, msg.pose.position.y
        heading = np.rad2deg(self.yaw_from_quaternion(msg.pose.orientation))
        self.raw_pose = np.array((x,y,heading))
        if self.frame is not None:
            x, y, heading = optitrack_to_maze(x, y, heading, self.frame) # Your code works in the maze frame, like in simulation
        self.pose = np.array((x,y,heading))
        return self.pose

    def odometer_callback(self, msg):
        '''Callback to calculate 2D pose info from Gazebo odomoter. Pose info includes x and y coordinates, as well as heading in degrees.
        This callback will run everytime the rclpy executor spins'''
        latest_pose_msg = msg.pose.pose
        heading = np.rad2deg(self.yaw_from_quaternion(latest_pose_msg.orientation))
        self.pose = np.array((latest_pose_msg.position.x, latest_pose_msg.position.y, heading))
        return self.pose

    def move_2D(self, x:float=0.0, y:float=0.0, turn:float=0.0):
        '''Publishes a Twist message to ROS to move a robot. Inputs are x and y linear velocities, as well as turn (z-axis yaw) angular velocity.'''
        twist_msg = Twist()
        x = np.clip(x, -max_translate_velocity, max_translate_velocity)
        y = np.clip(y, -max_translate_velocity, max_translate_velocity)
        turn = np.clip(turn, -max_translate_velocity*2, max_translate_velocity*2)
        twist_msg.linear.x, twist_msg.linear.y, twist_msg.linear.z = float(x), float(y), 0.0
        twist_msg.angular.x, twist_msg.angular.y, twist_msg.angular.z = 0.0, 0.0, float(turn)
        self.publisher_.publish(twist_msg)

    def set_waypoints(self, waypoints:list):
        '''Set new waypoints when a goal has been reached'''
        self.goal_reached = False
        self.waypoints = waypoints
        self.current_waypoint_idx = 0

    def timer_callback(self):
        """Controller loop. Insert path planning and PID control logic here"""
        if self.pose is None:
            return # Does not run if no pose received from Odom or Optitrack
        now = time.time()
        if now - getattr(self, "_last_pose_log", 0.0) >= 1.0: # at most once a second, so it does not bury the map / warnings below
            self._last_pose_log = now
            self.get_logger().debug(f"Pose: {self.pose}")

        if self.calibrate:
            self._print_calibration()
            return # Calibration mode: look, don't drive

        self._check_start_once()

        if self.path != self._last_printed_path: # Prints once immediately (map + start + goals), then again each time self.path changes
            self.print_map()
            self._last_printed_path = list(self.path)

        ###### INSERT CODE HERE ######
        now = time.time()
        dt = now - self.last_time
        self.last_time = now

        def reset_pid():
            '''Clear PID memory. Called whenever the target or steering mode changes, so the D term doesn't spike.'''
            self.heading_integral = 0.0
            self.prev_heading_error = None

        def wrap_rad(angle):
            '''Wrap an angle in radians into [-pi, pi) so the robot always turns the shorter way.'''
            return (angle + math.pi) % (2 * math.pi) - math.pi

        def route_cells(route, start_xy, goal_xy):
            '''Grid cells the route passes through, in order (start cell left out). Only used for the terminal map print.'''
            cells = []
            ax, ay = start_xy
            for bx, by in route:
                steps = max(1, int(math.hypot(bx - ax, by - ay) / (self.resolution / 4)))
                for k in range(1, steps + 1):
                    cell = world_to_grid(ax + (bx - ax) * k / steps, ay + (by - ay) * k / steps, self.origin, self.resolution)
                    if cell not in cells:
                        cells.append(cell)
                ax, ay = bx, by
            start_cell = world_to_grid(start_xy[0], start_xy[1], self.origin, self.resolution)
            cells = [c for c in cells if c != start_cell]
            return cells or [world_to_grid(goal_xy[0], goal_xy[1], self.origin, self.resolution)]

        # =========================
        # PLANNING: no route -> plan one to the current goal
        # =========================
        if not self.route:
            if self.goal_index >= len(self.goal_list):
                # All goals visited: stop the robot and end the node
                self.move_2D(0, 0, 0)
                self.get_logger().info("All goals reached")
                raise SystemExit

            goal = self.goal_list[self.goal_index]
            self.route = self.aStar(self.pose, goal)

            # No route to this goal. Goals must be visited in order, so stop.
            if not self.route:
                self.get_logger().error(f"No path to goal {self.goal_index}")
                self.move_2D(0, 0, 0)
                raise SystemExit

            robot_xy = (float(self.pose[0]), float(self.pose[1]))
            self.path = route_cells(self.route, robot_xy, goal)  # Shown in the terminal map print
            self.path_index = 0
            self.seg_start = robot_xy       # First segment runs from where the robot is now
            self.turning_in_place = True    # Line up with the first segment before driving
            self.reversing = False
            self.final_point_mode = False
            self.steer_mode = None
            reset_pid()
            self.move_2D(0, 0, 0)
            return

        # =========================
        # TRACKING: follow the route one straight segment at a time
        # =========================
        x, y, heading_deg = self.pose
        heading = math.radians(heading_deg)
        gx, gy = self.goal_list[self.goal_index]

        # GOAL CHECK: on the goal point -> stop, next goal, re-plan next tick
        if math.hypot(gx - x, gy - y) < self.goal_tolerance:
            self.get_logger().info(f"Reached goal {self.goal_index} ({gx}, {gy})")
            self.move_2D(0, 0, 0)
            self.goal_index += 1
            self.route = []
            self.path = []
            self.path_index = 0
            reset_pid()
            return

        # Current segment: from seg_start (previous waypoint) to the target waypoint
        self.path_index = min(self.path_index, len(self.route) - 1)
        is_last = self.path_index == len(self.route) - 1
        tx, ty = self.route[self.path_index]
        ax, ay = self.seg_start
        seg_len = math.hypot(tx - ax, ty - ay)
        if seg_len > 1e-6:
            ux, uy = (tx - ax) / seg_len, (ty - ay) / seg_len  # Unit vector along the segment
        else:
            ux, uy = math.cos(heading), math.sin(heading)
        along = (x - ax) * ux + (y - ay) * uy   # How far along the segment the robot is
        remaining = seg_len - along             # How far is left (negative = past the end)

        # CORNER CHECK: reached (or driven past) the corner -> next segment starts exactly at this corner
        if not is_last and remaining <= self.waypoint_tolerance:
            self.seg_start = (tx, ty)
            self.path_index += 1
            reset_pid()
            return

        # STEERING MODE
        #   line  : steer at a point `lookahead` ahead on the segment -> pulls the robot back onto the line
        #   hold  : last few cm before a corner -> keep the segment's heading, just close the distance
        #   point : last few cm before the goal -> aim straight at the goal point (reverse onto it if overshot)
        if is_last and remaining < self.end_zone:
            self.final_point_mode = True  # Latched, so a small overshoot doesn't flip it back to line mode
        if self.final_point_mode:
            mode = 'point'
        elif remaining < self.end_zone:
            mode = 'hold'
        else:
            mode = 'line'

        reverse = False
        if mode == 'line':
            look = min(max(along, 0.0) + self.lookahead, seg_len)
            qx, qy = ax + ux * look, ay + uy * look
            heading_error = wrap_rad(math.atan2(qy - y, qx - x) - heading)
            drive_dist = remaining
        elif mode == 'hold':
            heading_error = wrap_rad(math.atan2(uy, ux) - heading)
            drive_dist = remaining
        else:
            heading_error = wrap_rad(math.atan2(gy - y, gx - x) - heading)
            drive_dist = math.hypot(gx - x, gy - y)
            reverse = abs(heading_error) > math.pi / 2  # Goal is behind (overshot) -> back onto it
            if reverse:
                heading_error = wrap_rad(heading_error + math.pi)  # Point the robot's BACK at the goal

        if mode != self.steer_mode or reverse != self.reversing:  # Error target jumps -> clear PID memory
            self.steer_mode = mode
            self.reversing = reverse
            reset_pid()

        # Turn on the spot for big errors, otherwise drive and steer (hysteresis stops flickering)
        if self.turning_in_place:
            if abs(heading_error) < self.turn_in_place_exit:
                self.turning_in_place = False
        elif abs(heading_error) > self.turn_in_place_enter:
            self.turning_in_place = True

        # HEADING PID
        derivative = 0.0
        if 0.0 < dt < 0.5:  # Skip I/D on a bad dt (first tick, or the loop stalled)
            self.heading_integral = float(np.clip(self.heading_integral + heading_error * dt,
                                                  -self.integral_limit, self.integral_limit))
            if self.prev_heading_error is not None:
                derivative = wrap_rad(heading_error - self.prev_heading_error) / dt
        self.prev_heading_error = heading_error

        turn_speed = (self.kp_turn * heading_error
                      + self.ki_turn * self.heading_integral
                      + self.kd_turn * derivative)

        if self.turning_in_place:
            # TURN on the spot, with a minimum speed so small corrections still move the real robot
            if abs(turn_speed) < self.min_turn_speed:
                turn_speed = math.copysign(self.min_turn_speed, heading_error)
            self.move_2D(0.0, 0.0, turn_speed)
        else:
            # DRIVE: speed proportional to the distance left (eases onto the point), with a floor so it
            # doesn't stall, scaled by cos(error) so it moves forward less the more it still has to turn
            speed = float(np.clip(self.kp * drive_dist, self.min_drive_speed, self.max_drive_speed))
            speed *= max(math.cos(heading_error), 0.0)
            self.move_2D(-speed if self.reversing else speed, 0.0, turn_speed)

    def aStar(self, start, goal_xy):
        """A* path planner (8-connected moves, octile distance heuristic).

        Plans on the grid of cell CORNERS: a corner is usable only if all 4 cells around it are free.
        The maze's corridors are 2 cells (0.4 m) wide, so this puts the route on the corridor centreline,
        0.2 m from each wall, instead of through cells that touch a wall (only 0.1 m away).
        If there is no such route (e.g. a 1-cell-wide gap), it plans through cell centres instead.

        start   -- robot pose [x, y, heading] in world coordinates
        goal_xy -- goal (x, y) in world coordinates
        Returns the route as world (x, y) waypoints in travel order -- only the points where the
        direction changes, ending exactly on goal_xy -- or [] if no route exists.
        """
        import itertools

        res = self.resolution
        ox, oy = self.origin
        sx, sy = float(start[0]), float(start[1])
        gx, gy = float(goal_xy[0]), float(goal_xy[1])
        start_cell = world_to_grid(sx, sy, self.origin, res)
        goal_cell = world_to_grid(gx, gy, self.origin, res)

        # Already in the goal cell: drive straight onto the goal
        if start_cell == goal_cell:
            return [(gx, gy)]

        def octile(ax, ay, bx, by):
            '''Octile distance between two world points, in cells: min(dx, dy) diagonal steps (sqrt 2 each),
            then the rest straight (1 each). Never more than the real route length, so A* stays optimal.'''
            dx, dy = abs(ax - bx) / res, abs(ay - by) / res
            return np.sqrt(2) * min(dx, dy) + (max(dx, dy) - min(dx, dy))

        def step_cost(dx, dy):
            '''Cost of one move between neighbouring nodes: sqrt 2 diagonal, 1 straight.'''
            return np.sqrt(2) if (dx != 0 and dy != 0) else 1

        dirs = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]  # 8-connected moves

        def search(usable, to_world, start_nodes, goal_nodes):
            '''A* from the robot's position, through grid nodes, to the goal point.
            usable(n)   -- whether node n can be driven through
            to_world(n) -- world (x, y) of node n
            start_nodes -- nodes the robot can drive to directly from where it is
            goal_nodes  -- nodes it can drive from directly onto the goal
            Returns the nodes' world points in travel order, or None if there is no route.'''
            if not start_nodes or not goal_nodes:
                return None
            h = lambda n: octile(*to_world(n), gx, gy)  # Heuristic: octile distance to the goal
            g_cost, prev_node, visited = {}, {}, set()
            queue, tie = [], itertools.count()  # Heap of (f, h, tie-breaker, node); node None = "the goal itself"
            for n in start_nodes:
                g = octile(sx, sy, *to_world(n))  # Robot -> first node
                if g < g_cost.get(n, float('inf')):
                    g_cost[n], prev_node[n] = g, None
                    heapq.heappush(queue, (g + h(n), h(n), next(tie), n))

            best_goal, last_node = float('inf'), None
            while queue:
                _, _, _, n = heapq.heappop(queue)
                if n is None:
                    # Cheapest way onto the goal found: rebuild the route by following parents back
                    route = []
                    while last_node is not None:
                        route.insert(0, to_world(last_node))
                        last_node = prev_node[last_node]
                    return route
                if n in visited:  # A node can be pushed several times; its first pop is the cheapest
                    continue
                visited.add(n)

                if n in goal_nodes:  # From here the robot can drive straight onto the goal
                    g_end = g_cost[n] + octile(*to_world(n), gx, gy)
                    if g_end < best_goal:
                        best_goal, last_node = g_end, n
                        heapq.heappush(queue, (g_end, 0.0, next(tie), None))

                for dx, dy in dirs:
                    m = (n[0] + dx, n[1] + dy)
                    if not usable(m):
                        continue
                    # Diagonals may not cut a corner: both side nodes must be usable too
                    if dx != 0 and dy != 0 and not (usable((n[0] + dx, n[1])) and usable((n[0], n[1] + dy))):
                        continue
                    new_g = g_cost[n] + step_cost(dx, dy)
                    if new_g < g_cost.get(m, float('inf')):
                        g_cost[m], prev_node[m] = new_g, n
                        heapq.heappush(queue, (new_g + h(m), h(m), next(tie), m))
            return None

        free = self.map_array == 0

        # 1) Corner grid. Corner (a, b) sits at world (ox + a*res, oy + b*res) and touches cells
        #    (a-1..a, b-1..b); it is usable only if all 4 of those cells are free (outside the map counts as wall).
        padded = np.pad(free, 1, constant_values=False)
        corner_free = padded[:-1, :-1] & padded[1:, :-1] & padded[:-1, 1:] & padded[1:, 1:]  # shape (rows+1, cols+1)
        corner_ok = lambda n: 0 <= n[0] <= self.rows and 0 <= n[1] <= self.cols and bool(corner_free[n])
        corner_world = lambda n: (ox + n[0] * res, oy + n[1] * res)
        corners_of = lambda c: [(c[0], c[1]), (c[0] + 1, c[1]), (c[0], c[1] + 1), (c[0] + 1, c[1] + 1)]
        route = search(corner_ok, corner_world,
                       [c for c in corners_of(start_cell) if corner_ok(c)],
                       {c for c in corners_of(goal_cell) if corner_ok(c)})

        # 2) Fallback: cell centres (needed only where a passage is a single cell wide)
        if route is None:
            self.get_logger().warn("No corridor-centre route to this goal, planning through cell centres")
            cell_ok = lambda n: 0 <= n[0] < self.rows and 0 <= n[1] < self.cols and bool(free[n])
            cell_world = lambda n: grid_to_world(n[0], n[1], self.origin, res)
            route = search(cell_ok, cell_world,
                           [start_cell] if cell_ok(start_cell) else [],
                           {goal_cell} if cell_ok(goal_cell) else set())
        if route is None:
            return []

        # Prune: keep only points where the direction changes, and end exactly on the goal
        pts = [(sx, sy)] + route + [(gx, gy)]
        kept = [pts[0]]
        for k in range(1, len(pts) - 1):
            (ax, ay), (bx, by), (cx, cy) = kept[-1], pts[k], pts[k + 1]
            v1x, v1y, v2x, v2y = bx - ax, by - ay, cx - bx, cy - by
            if abs(v1x * v2y - v1y * v2x) < 1e-9 and v1x * v2x + v1y * v2y >= 0:
                continue  # b lies on the straight line from the last kept point to c
            kept.append((bx, by))
        kept.append(pts[-1])
        return kept[1:]
        ###### INSERT CODE HERE ######


class Grid():
    '''
    Grid class to use with occupancy grid. Contains the following functions:
        check_grid_validity : Uses flood fill to check if there's a valid path from start to goal position
        draw_grid_map : Creates a colour image of the grid, as well as waypoints and full solution path if given. Can show it and/or save it to a .png file
        print_grid_map : Prints an ASCII-art version of the same map (grid, waypoints, path) to the terminal
        animate_path : Prints the ASCII-art map frame by frame in the terminal, animating the robot moving along a given path

    __init__ input Args:
        grid_array : 2D numpy array representing the occupancy grid
        starting_position : tuple of starting indices within the numpy array
        goal_position : tuple of goal indices within the numpy array. Works with negative indices as well
    '''
    def __init__(self, grid_array:np.array=np.array([]), starting_position:tuple=(0,0), goal_position:tuple=(-1,-1)):
        self.grid = grid_array
        self.shape = self.grid.shape
        self.starting_position = starting_position

        # If goal position given with negative index, need to convert to +ve
        if goal_position[0] < 0:
            goal_x = self.shape[0] + goal_position[0]
        else:
            goal_x = goal_position[0]
        if goal_position[1] < 0:
            goal_y = self.shape[1] + goal_position[1]
        else:
            goal_y = goal_position[1]

        self.goal_position = (goal_x, goal_y)

    def check_grid_validity(self):
        '''Use flood fill to check if there's a viable path between start and goal positions'''
        grid = self.grid.copy()
        flood_stack = [(self.goal_position[0], self.goal_position[1])]
        while flood_stack:
            tile = flood_stack[0]
            del flood_stack[0]
            try:
                next_tile = (tile[0]+1, tile[1])
                if grid[next_tile] == 0:
                    grid[next_tile] = 1
                    flood_stack.append(next_tile)
            except:pass
            try:
                next_tile = (tile[0]-1, tile[1])
                if grid[next_tile] == 0:
                    grid[next_tile] = 1
                    flood_stack.append(next_tile)
            except:pass
            try:
                next_tile = (tile[0], tile[1]+1)
                if grid[next_tile] == 0:
                    grid[next_tile] = 1
                    flood_stack.append(next_tile)
            except:pass
            try:
                next_tile = (tile[0], tile[1]-1)
                if grid[next_tile] == 0:
                    grid[next_tile] = 1
                    flood_stack.append(next_tile)
            except:pass
        if grid[self.starting_position] == 1: # Means the flood is able to reach starting position from the ending position
            return True
        else:
            return False

    def _colour_grid(self, waypoints:list=(), path:list=(), obstacle_threshold:float=50) -> np.array:
        '''Builds the (H, W, 3) colour image array shared by draw_grid_map. Maze walls in blue, empty space in white, path taken in green and waypoints in red'''
        image_grid = np.ones((self.grid.shape[0],self.grid.shape[1],3), dtype=np.uint8)
        image_grid[self.grid <= obstacle_threshold] = (255,255,255)
        image_grid[self.grid > obstacle_threshold] = (0,0,255)

        for x, y in path:
            image_grid[x][y] = (0,255,0)

        for point in waypoints:
            image_grid[point] = (255,0,0)

        return image_grid

    def draw_grid_map(self, waypoints:list=(), path:list=(), obstacle_threshold:float=50, save_path:str=None, show:bool=True):
        '''Creates an image of the maze and path taken. Maze walls in blue, empty space in white, path taken in green and waypoints in red

        Args:
            waypoints : list (or other iterable) of tuple coordinates representing all the grid indices for the waypoints. Will be represented in red, takes precedence over path
            path : list (or other iterable) of tuple coordinates representing all the grid indices forming the solution path. Will be represented in green
            obstacle_threshold : Optional float to indicate threshhold for whether a grid is considered occupied. Not important for ca2
            save_path : Optional file path (e.g. "path_result.png") to save the image to, in addition to/instead of showing it
            show : Whether to pop up the image in a viewer (default True). Set to False if you only want to save it
        '''
        image_grid = self._colour_grid(waypoints, path, obstacle_threshold)

        image_grid = np.flip(image_grid, axis=1)[::-1]
        img = Image.fromarray(image_grid, 'RGB')

        # Resize image
        base_width = 500
        wpercent = (base_width / float(img.size[0]))
        hsize = int((float(img.size[1]) * float(wpercent)))
        img = img.resize((base_width, hsize), getattr(Image, "Resampling", Image).NEAREST)

        if save_path:
            img.save(save_path)
            print(f"Saved map image to {save_path}")

        if show:
            img.show()

    def print_grid_map(self, waypoints:list=(), path:list=(), obstacle_threshold:float=50):
        '''Prints an ASCII-art version of the maze to the terminal: '#' wall, '.' free, '*' solution path, 'W' waypoint, 'S' start, 'G' goal

        Args:
            waypoints : list (or other iterable) of tuple coordinates representing all the grid indices for the waypoints
            path : list (or other iterable) of tuple coordinates representing all the grid indices forming the solution path
            obstacle_threshold : Optional float to indicate threshhold for whether a grid is considered occupied. Not important for ca2
        '''
        chars = np.where(self.grid > obstacle_threshold, '#', '.').astype('<U1')

        for x, y in path:
            chars[x, y] = '*'
        for point in waypoints:
            chars[point] = 'W'
        chars[self.starting_position] = 'S'
        chars[self.goal_position] = 'G'

        display = np.flip(chars, axis=1)[::-1]
        print('\n'.join(''.join(row) for row in display))

    def animate_path(self, path:list, waypoints:list=(), obstacle_threshold:float=50, delay:float=0.2):
        '''Animates the robot moving along path, one cell at a time, by reprinting the ASCII-art map to the terminal.
        'R' marks the robot's current cell, '*' cells it has already passed through.

        Args:
            path : list of tuple grid indices forming the solution path, in travel order
            waypoints : list (or other iterable) of tuple coordinates representing all the grid indices for the waypoints
            obstacle_threshold : Optional float to indicate threshhold for whether a grid is considered occupied. Not important for ca2
            delay : Seconds to pause between animation frames
        '''
        path = list(path)
        base_chars = np.where(self.grid > obstacle_threshold, '#', '.').astype('<U1')
        for point in waypoints:
            base_chars[point] = 'W'
        base_chars[self.goal_position] = 'G'

        for step in range(len(path)):
            frame = base_chars.copy()
            for x, y in path[:step]:
                frame[x, y] = '*'
            frame[path[step]] = 'R'

            display = np.flip(frame, axis=1)[::-1]
            print("\033c", end="")  # Clear terminal between frames
            print('\n'.join(''.join(row) for row in display))
            time.sleep(delay)


def main(args=None):
    global max_translate_velocity

    arg_parser = argparse.ArgumentParser(description="RB2301 CA2 path planning")
    arg_parser.add_argument(
        '--run', choices=IRL_RUNS, default=None,
        help="Run on the real maze: test1, test2 or full (starts/goals in README). Omit to run in Gazebo simulation."
    )
    arg_parser.add_argument(
        '--robot', type=int, default=None,
        help="Number of your Bingda robot (3 -> /vrpn_mocap/bingda_003/pose). Default: your ROS_DOMAIN_ID (set to the robot number on the robots), else [robot] number in optitrack_variables.config."
    )
    arg_parser.add_argument(
        '--calibrate', action='store_true',
        help="Real robot: only print Optitrack and maze-frame poses (twice a second) and never publish cmd_vel. Use it to set origin/rotation."
    )
    cli_args, ros_args = arg_parser.parse_known_args(args=args)
    run = cli_args.run or ("test1" if cli_args.calibrate else None)
    is_simulation = run is None # Remember: pass --run test1|test2|full to ca2.sh when testing on the real lab setup

    print("Starting path planning")
    rclpy.init(args=ros_args)

    robot_number = 3
    frame = expected_start = None
    if is_simulation:
        config = sim_config
    else:
        config = load_irl_config(run)
        frame, expected_start = config["frame"], config["start"]
        domain = os.environ.get("ROS_DOMAIN_ID", "")
        if cli_args.robot is not None:
            robot_number = cli_args.robot
        elif domain.isdigit() and 1 <= int(domain) <= 99: # on the Bingda robots ROS_DOMAIN_ID is the robot number
            robot_number = int(domain)
        else:
            robot_number = config["robot_number"]
        frame = robot_frame(config, robot_number)
        expected_start = config["start"]
        print(f"Running on the real maze, run={run}, robot=bingda_{robot_number:03d}, resolution={config['resolution']}")
        print(f"  maze origin in Optitrack frame = ({frame['origin_x']}, {frame['origin_y']}), rotation = {frame['rotation_deg']} deg, robot heading offset = {frame['heading_offset_deg']} deg")
        print(f"  start={expected_start}, goals={config['goal_list']}")
        if cli_args.calibrate:
            print("  CALIBRATE MODE: printing poses only, the robot will not be commanded")

    max_translate_velocity = config["max_translate_velocity"]

    map_array = np.load(os.path.join(_PACKAGE_DIR, config["map_file"]), allow_pickle=True)
    waypoint = WaypointNode(map_array, config["goal_list"], is_simulation, config["origin"], config["resolution"],
                            frame=frame, robot_number=robot_number, expected_start=expected_start, calibrate=cli_args.calibrate)

    # Start spinning the waypoint node and only stop once SystemExit error is raised within the node callback
    try:
        rclpy.spin(waypoint)
    except SystemExit:
        print("Shutting down")

    waypoint.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()