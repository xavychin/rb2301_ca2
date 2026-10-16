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
        self.goal_index = 0
        self.rows, self.cols = self.map_array.shape
        self.segments = []      # Straight runs of self.path as ((start_x, start_y), (end_x, end_y)) in world metres
        self.segment_index = 0  # Run currently being driven
        self.turning = True     # True: rotate on the spot to face the run; False: drive along it

        # Controller settings, the same for simulation and the real robot.
        # Everything is in SI units: metres, radians, seconds.
        # Heading PID: heading error (rad) -> turn rate wanted (rad/s)
        self.kp_turn = 4.0
        self.ki_turn = 1.0
        self.kd_turn = 0.1
        self.max_turn_speed = 1.6           # rad/s of actual turning (move_2D also caps the command at 2x max_translate_velocity)
        self.min_turn_speed = 0.2           # rad/s, smallest turn rate asked for while turning on the spot
        self.heading_integral_limit = 0.5   # rad*s, anti wind-up clamp on the I term

        # Speed P controller: distance left along the run (m) -> forward speed (m/s)
        self.kp_drive = 2.0
        self.min_drive_speed = 0.05         # m/s, so the robot never stalls just short of a corner
        self.max_accel = 1.0                # m/s^2, ramps the speed up instead of jumping to full speed

        # Cross-track correction: while driving, angle slightly back onto the run's centre line.
        # On-the-spot turns always go to the exact grid heading (0, 90, 180 or -90 degrees); only while
        # driving is the heading bent a little, and the bend fades out over the last part of the run so
        # the robot arrives at the corner square to the maze.
        self.k_cross_track = 4.0            # rad of heading correction per metre off the line
        self.max_cross_track_correction = np.deg2rad(8)
        self.cross_track_fade_distance = 0.2  # m, the correction fades to 0 over this last stretch of the run

        # Deadband compensation, learned while the robot runs.
        # Friction eats the first part of every command: below it the robot doesn't move at all. In Gazebo
        # that is ~1.2 rad/s of an on-the-spot turn and ~0.13 m/s of driving; the real robot has its own amount.
        # The offsets below are added on top of each command. They start at 0 and are learned from the
        # measured motion: raised while the robot is (nearly) stalled, i.e. moving at under stall_ratio of the
        # speed asked for, and lowered slowly while it moves clearly faster than asked. (A loose "nearly stalled"
        # test is used instead of matching speeds exactly because Gazebo can run slower than real time,
        # which makes every measured speed look too slow.)
        # While driving forward the turn command is followed exactly, so the turn offset is only used for on-the-spot turns.
        self.turn_deadband = 0.0            # rad/s, learned
        self.drive_deadband = 0.0           # m/s, learned
        self.turn_deadband_rise = 1.5       # rad/s per second, how fast the turn offset rises while stalled
        self.drive_deadband_rise = 0.15     # m/s per second, same for driving
        self.deadband_fall = 0.3            # Fraction per second the offsets drop while the robot moves clearly too fast
        self.stall_ratio = 0.3              # Moving at under this fraction of the speed asked for counts as stalled
        self.stall_turn_rate = 0.05         # rad/s, measured turn rate below this always counts as stalled
        self.stall_speed = 0.02             # m/s, measured speed below this always counts as stalled
        self.max_turn_deadband = 2.0        # rad/s
        self.max_drive_deadband = 0.3       # m/s
        self.learn_delay = 0.3              # s, don't learn just after starting a turn / run while the robot is still speeding up

        # Tolerances
        self.align_tolerance = np.deg2rad(3)      # On-the-spot turn is done once within this of the run's heading
        self.realign_threshold = np.deg2rad(25)   # While driving, stop and turn on the spot again past this error
        self.turn_timeout = 5.0                   # s, after this an on-the-spot turn is accepted within the looser tolerance below
        self.turn_timeout_tolerance = np.deg2rad(10)  # (the heading PID keeps correcting while driving)
        self.position_tolerance = 0.02            # m, run is done once the robot is this close to (or past) its end

        # Controller memory
        self.heading_integral = 0.0
        self.prev_pose = None          # Pose on the previous tick, to measure the robot's actual motion
        self.measured_turn_rate = 0.0  # rad/s, filtered
        self.measured_speed = 0.0      # m/s forward, filtered
        self.speed = 0.0               # Last forward speed asked for, for the acceleration limit
        self.last_time = None
        self.turn_start_time = time.time()
        self.drive_start_time = time.time()

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
        # Time since the last tick (`now` is set at the top of this callback), clamped so a stall or the very first tick can't blow up the I/D terms
        dt = 0.05 if self.last_time is None else float(np.clip(now - self.last_time, 1e-3, 0.2))
        self.last_time = now

        x, y, heading_deg = self.pose
        heading = np.deg2rad(heading_deg)

        # Measure the robot's actual turn rate and forward speed from the change in pose since the last tick
        # (low-pass filtered to smooth out pose noise). Used for the D term and for learning the deadbands.
        if self.prev_pose is not None:
            px, py, ph = self.prev_pose
            rate_now = math.remainder(heading - ph, 2 * math.pi) / dt  # remainder wraps the angle into [-pi, pi]
            speed_now = ((x - px) * math.cos(heading) + (y - py) * math.sin(heading)) / dt
            self.measured_turn_rate += 0.5 * (rate_now - self.measured_turn_rate)
            self.measured_speed += 0.5 * (speed_now - self.measured_speed)
        self.prev_pose = (x, y, heading)

        # =========================
        # PLANNING: no runs left -> plan a path to the current goal
        # =========================
        if not self.segments:
            if self.goal_index >= len(self.goal_list):
                # All goals visited: stop the robot and end the node
                self.move_2D(0, 0, 0)
                self.get_logger().info("All goals reached")
                raise SystemExit

            goal_xy = self.goal_list[self.goal_index]
            start_cell = world_to_grid(x, y, self.origin, self.resolution)
            goal_cell = world_to_grid(goal_xy[0], goal_xy[1], self.origin, self.resolution)

            if start_cell == goal_cell:
                # Already in the goal cell: A* has nothing to do, just drive to the goal point
                self.path = []
                points = [(x, y), tuple(goal_xy)]
            else:
                self.path = self.aStar(self.pose, goal_xy)
                # A* returned [] -> no route to this goal.
                # Stop and end the node instead of re-planning every tick,
                # since goals must be visited in order.
                if not self.path:
                    self.get_logger().error(f"No path to goal {self.goal_index}")
                    self.move_2D(0, 0, 0)
                    raise SystemExit
                # Path pruning: keep only the nodes where the direction changes (plus the start and the end)
                pruned = self.prune_path([start_cell] + self.path)
                self.get_logger().info(f"Path to goal {self.goal_index}: {len(self.path)} cells, pruned to {len(pruned)} nodes: {pruned}")
                # Pruned nodes -> world points (cell centres), finishing on the exact goal point (inside the last cell)
                points = [grid_to_world(i, j, self.origin, self.resolution) for i, j in pruned]
                points[-1] = tuple(goal_xy)

            # Consecutive corner points form the straight runs to drive (zero-length ones dropped)
            self.segments = [(a, b) for a, b in zip(points, points[1:]) if math.hypot(b[0] - a[0], b[1] - a[1]) > 1e-6]
            self.segment_index = 0
            self.start_turn()
            if not self.segments:
                self.goal_done()
            return

        # =========================
        # TRACKING: drive along the current straight run
        # =========================
        (sx, sy), (ex, ey) = self.segments[self.segment_index]
        length = math.hypot(ex - sx, ey - sy)
        ux, uy = (ex - sx) / length, (ey - sy) / length  # Unit vector along the run

        # Distance still to go to the end of the run, measured along it (negative once past the end)
        along = (ex - x) * ux + (ey - y) * uy
        # Signed distance from the run's centre line (positive = robot is left of the line)
        cross = ux * (y - sy) - uy * (x - sx)

        # Run finished -> stop, then turn on the spot to face the next run
        if along < self.position_tolerance:
            self.segment_index += 1
            if self.segment_index >= len(self.segments):
                self.goal_done()  # Last run -> goal reached (goal_done stops the robot)
            else:
                self.move_2D(0, 0, 0)
                self.start_turn()
            return

        # The run's direction: exactly along a grid axis (0, 90, 180 or -90 degrees).
        # On-the-spot turns aim for exactly this, so every turn is a clean 90 (or 180) degrees.
        run_heading = math.atan2(uy, ux)
        turn_error = math.remainder(run_heading - heading, 2 * math.pi)

        # While driving: the run's direction, bent slightly towards the centre line if the robot has drifted
        # off it. The bend is capped at a few degrees and fades out near the end of the run.
        fade = min(1.0, along / self.cross_track_fade_distance)
        correction = fade * float(np.clip(-self.k_cross_track * cross, -self.max_cross_track_correction, self.max_cross_track_correction))
        drive_error = math.remainder(run_heading + correction - heading, 2 * math.pi)

        # =========================
        # TURN: rotate on the spot until facing the run
        # =========================
        if not self.turning and abs(turn_error) > self.realign_threshold:
            self.start_turn()  # Knocked badly off course while driving -> stop and re-align

        if self.turning:
            # Done once within tolerance, or within the looser tolerance if the turn is taking too long
            timed_out = now - self.turn_start_time > self.turn_timeout
            if abs(turn_error) < self.align_tolerance or (timed_out and abs(turn_error) < self.turn_timeout_tolerance):
                self.turning = False
                self.heading_integral = 0.0  # Fresh I term for the drive
                self.drive_start_time = now
            else:
                turn_rate = self.heading_pid(turn_error, dt)
                # Never ask for less than the minimum, or the robot can stall a few degrees short
                turn_rate = max(abs(turn_rate), self.min_turn_speed)
                # Don't ask for more than move_2D lets through once the deadband is added
                turn_rate = min(turn_rate, 2 * max_translate_velocity - self.turn_deadband)

                # Learn the turn deadband from the measured turn rate (in the direction asked for)
                if now - self.turn_start_time > self.learn_delay:
                    measured = math.copysign(1, turn_error) * self.measured_turn_rate
                    self.turn_deadband = self.learn_deadband(self.turn_deadband, turn_rate, measured, self.stall_turn_rate,
                                                             self.turn_deadband_rise, self.max_turn_deadband, dt)

                # Add the deadband on top so the robot really turns at turn_rate
                self.move_2D(0, 0, math.copysign(turn_rate + self.turn_deadband, turn_error))
                return

        # =========================
        # DRIVE: forward speed from the distance left, steering from the heading PID
        # =========================
        # P on distance so it slows into the corner, at least min speed so it actually arrives
        target_speed = float(np.clip(self.kp_drive * along, self.min_drive_speed, max_translate_velocity))
        # Ease off while the heading is still being corrected
        target_speed *= math.cos(drive_error)
        # Acceleration limit on the way up (slowing down is already gradual from the P term)
        self.speed = min(target_speed, self.speed + self.max_accel * dt)

        # Learn the drive deadband from the measured forward speed
        if now - self.drive_start_time > self.learn_delay:
            self.drive_deadband = self.learn_deadband(self.drive_deadband, self.speed, self.measured_speed, self.stall_speed,
                                                      self.drive_deadband_rise, self.max_drive_deadband, dt)

        turn_speed = self.heading_pid(drive_error, dt)
        # Add the drive deadband on top so the robot really moves at self.speed
        self.move_2D(self.speed + self.drive_deadband, 0, turn_speed)

    def heading_pid(self, error:float, dt:float) -> float:
        '''PID on the heading error (rad) -> angular velocity (rad/s).
        The D term uses the measured turn rate instead of the change in error, so a jump
        in the desired heading (new run) does not cause a derivative kick.'''
        # Only integrate close to the target (no wind-up during big turns), and clamp it
        if abs(error) < np.deg2rad(15):
            self.heading_integral = float(np.clip(self.heading_integral + error * dt,
                                                  -self.heading_integral_limit, self.heading_integral_limit))

        turn_speed = self.kp_turn * error + self.ki_turn * self.heading_integral - self.kd_turn * self.measured_turn_rate
        return float(np.clip(turn_speed, -self.max_turn_speed, self.max_turn_speed))

    def learn_deadband(self, deadband:float, asked:float, measured:float, stall_floor:float,
                       rise:float, limit:float, dt:float) -> float:
        '''One learning step for a deadband offset (turning or driving); returns the updated offset.
        Stalled (moving at under stall_ratio of the speed asked for, or under stall_floor) -> raise it at `rise`
        per second. Clearly faster than asked -> lower it slowly. Clamped to [0, limit].'''
        if measured < max(stall_floor, self.stall_ratio * asked):
            deadband += rise * dt                               # Stalled: push harder
        elif measured > 1.5 * asked + 2 * stall_floor:
            deadband -= self.deadband_fall * deadband * dt      # Clearly too fast: ease off
        return float(np.clip(deadband, 0.0, limit))

    def start_turn(self):
        '''Switch to turning on the spot, with the speed and I term reset.'''
        self.turning = True
        self.speed = 0.0
        self.heading_integral = 0.0
        self.turn_start_time = time.time()

    def goal_done(self):
        '''Current goal reached: stop and move on to the next one (it gets planned on the next tick).'''
        self.move_2D(0, 0, 0)
        self.get_logger().info(f"Reached goal {self.goal_index}: {self.goal_list[self.goal_index]}")
        self.goal_index += 1
        self.segments = []
        self.path = []

    def prune_path(self, cells:list) -> list:
        '''Path pruning: drop every node where the direction does not change.
        cells is the full route (start cell first), one cell apart. A node is kept only if the step into it
        differs from the step out of it, i.e. the robot has to turn there. The first and last nodes are
        always kept. The robot then drives straight from each kept node to the next.'''
        pruned = [cells[0]]
        for prev, cell, nxt in zip(cells, cells[1:], cells[2:]):
            step_in = (cell[0] - prev[0], cell[1] - prev[1])
            step_out = (nxt[0] - cell[0], nxt[1] - cell[1])
            if step_in != step_out:   # Direction changes here -> keep the node
                pruned.append(cell)
            # Same direction -> prune it
        if len(cells) > 1:
            pruned.append(cells[-1])
        return pruned

    def aStar(self, start, goal_xy):
        """A* search on the occupancy grid (4-connected).

        start   -- robot pose [x, y, heading] in world coordinates
        goal_xy -- goal (x, y) in world coordinates
        Returns every grid cell from the start (start cell excluded) to the goal,
        one cell apart, in travel order, or [] if no route exists.
        """
        # Start and goal: world (metres) -> grid cells
        x, y = world_to_grid(
            start[0], start[1],
            self.origin, self.resolution
        )

        goal = world_to_grid(
            goal_xy[0], goal_xy[1],
            self.origin, self.resolution
        )

        plannedPath = []

        def h(cell):  # Heuristic: Manhattan distance from `cell` to `goal`
            # Ignores walls, so it never overestimates the real cost
            # (admissible) and stays consistent.
            # With only 4-connected moves, dx + dy straight steps (1 each) is the shortest possible route
            return abs(cell[0] - goal[0]) + abs(cell[1] - goal[1])

        visited = set()        # Cells already expanded (closed set)
        prevNode = {}          # cell -> the cell before it on its cheapest known route
        # Min-heap of (f, h, cell), starting with the start cell (g = 0, so f = h).
        # Python compares tuples element by element, so h breaks ties in f.
        queue = [(h((x, y)), h((x, y)), (x, y))]
        dir = [(1,0), (-1,0), (0,1), (0,-1)]  # 4-connected moves (no diagonals)
        g_cost = {(x, y): 0}   # cell -> cheapest known cost from the start (start = 0)

        while queue:
            # Expand the cell with the lowest f = g + h.
            # On a tie in f, the cell with the lower h (closer to the goal) comes first.
            _, _, (cx, cy) = heapq.heappop(queue)

            # A cell can be pushed more than once (each time a cheaper route is found).
            # Its first pop is the cheapest, so skip later copies.
            if (cx, cy) in visited:
                continue
            visited.add((cx, cy))

            if (cx, cy) == goal:
                # Goal reached: rebuild the route by following parents back to the start.
                # Each parent is one cell back, so every cell on the route is included.
                # The start cell has no parent, so it is left out of the path.
                px, py = cx, cy

                while (px, py) in prevNode:
                    plannedPath.insert(0, (px, py))   # Prepend so the path runs start -> goal
                    px, py = prevNode[(px, py)]

                return plannedPath
            else:
                # Try each of the 4 neighbours of the current cell
                for dx, dy in dir:
                    nx, ny = cx + dx, cy + dy

                    # Skip neighbours outside the grid or on a wall
                    if not (0 <= nx < self.rows and 0 <= ny < self.cols) or self.map_array[nx, ny] != 0:
                        continue

                    # Cost of reaching the neighbour by going through the current cell
                    # (every straight move costs 1 cell, no wall penalty)
                    new_g = g_cost[(cx, cy)] + 1

                    # Keep it only if it beats the best route found so far
                    # (unseen cells default to infinity, so their first route always wins)
                    if new_g < g_cost.get((nx, ny), float('inf',)):
                        g_cost[(nx, ny)] = new_g          # New best cost to the neighbour
                        prevNode[(nx, ny)] = (cx, cy)     # Remember where it was reached from
                        # Push (f, h, cell): f = cost so far + estimated cost remaining,
                        # h = estimated cost remaining, used as the tie-breaker
                        heapq.heappush(queue, (new_g + h((nx, ny)), h((nx, ny)), (nx, ny)))

        # Queue empty without reaching the goal: no route exists
        return plannedPath
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