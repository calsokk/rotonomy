#!/usr/bin/env python3

import rclpy                    # ROS2 client library
from rclpy.node import Node     # ROS2 node baseclass
from asl_tb3_lib.navigation import BaseNavigator, TrajectoryPlan
from asl_tb3_lib.math_utils import wrap_angle
from asl_tb3_lib.tf_utils import quaternion_to_yaw
from asl_tb3_msgs.msg import TurtleBotControl, TurtleBotState
from scipy.interpolate import splev, splrep
from asl_tb3_lib.grids import StochOccupancyGrid2D
import typing as T
V_PREV_THRES = 0.0001


import numpy as np

class AStar(object):
    """Represents a motion planning problem to be solved using A*"""

    def __init__(self, statespace_lo, statespace_hi, x_init, x_goal, occupancy, resolution=1):
        self.statespace_lo = statespace_lo         # state space lower bound (e.g., [-5, -5])
        self.statespace_hi = statespace_hi         # state space upper bound (e.g., [5, 5])
        self.occupancy = occupancy                 # occupancy grid (a DetOccupancyGrid2D object)
        self.resolution = resolution               # resolution of the discretization of state space (cell/m)
        self.x_offset = x_init
        self.x_init = self.snap_to_grid(x_init)    # initial state
        self.x_goal = self.snap_to_grid(x_goal)    # goal state

        self.closed_set = set()    # the set containing the states that have been visited
        self.open_set = set()      # the set containing the states that are condidate for future expension

        self.est_cost_through = {}  # dictionary of the estimated cost from start to goal passing through state (often called f score)
        self.cost_to_arrive = {}    # dictionary of the cost-to-arrive at state from start (often called g score)
        self.came_from = {}         # dictionary keeping track of each state's parent to reconstruct the path

        self.open_set.add(self.x_init)
        self.cost_to_arrive[self.x_init] = 0
        self.est_cost_through[self.x_init] = self.distance(self.x_init,self.x_goal)

        self.path = None        # the final path as a list of states

    def is_free(self, x):
        """
        Checks if a give state x is free, meaning it is inside the bounds of the map and
        is not inside any obstacle.
        Inputs:
            x: state tuple
        Output:
            Boolean True/False
        Hint: self.occupancy is a DetOccupancyGrid2D object, take a look at its methods for what might be
              useful here
        """
        ########## Code starts here ##########
        isfree = self.occupancy.is_free(np.asarray(x))
        isin = (x[0]>=self.statespace_lo[0] and x[0]<=self.statespace_hi[0]) and (x[1]>=self.statespace_lo[1] and x[1]<=self.statespace_hi[1])
        return (isfree and isin)
        ########## Code ends here ##########

    def distance(self, x1, x2):
        """
        Computes the Euclidean distance between two states.
        Inputs:
            x1: First state tuple
            x2: Second state tuple
        Output:
            Float Euclidean distance

        HINT: This should take one line. Tuples can be converted to numpy arrays using np.array().
        """
        ########## Code starts here ##########
        return np.linalg.norm(np.array(x1)-np.array(x2))

        ########## Code ends here ##########

    def snap_to_grid(self, x):
        """ Returns the closest point on a discrete state grid
        Input:
            x: tuple state
        Output:
            A tuple that represents the closest point to x on the discrete state grid
        """
        return (
            self.resolution * round((x[0] - self.x_offset[0]) / self.resolution) + self.x_offset[0],
            self.resolution * round((x[1] - self.x_offset[1]) / self.resolution) + self.x_offset[1],
        )

    def get_neighbors(self, x):
        """
        Gets the FREE neighbor states of a given state x. Assumes a motion model
        where we can move up, down, left, right, or along the diagonals by an
        amount equal to self.resolution.
        Input:
            x: tuple state
        Ouput:
            List of neighbors that are free, as a list of TUPLES

        HINTS: Use self.is_free to check whether a given state is indeed free.
               Use self.snap_to_grid (see above) to ensure that the neighbors
               you compute are actually on the discrete grid, i.e., if you were
               to compute neighbors by adding/subtracting self.resolution from x,
               numerical errors could creep in over the course of many additions
               and cause grid point equality checks to fail. To remedy this, you
               should make sure that every neighbor is snapped to the grid as it
               is computed.
        """
        neighbors = []
        ########## Code starts here ##########
        for i in range(-1,2):
            for j in range(-1,2):
                if (i,j)!=(0,0):
                    if i!=0 and j!=0:
                        candidate = self.snap_to_grid((x[0]+(self.resolution*i/1.414213),x[1]+(self.resolution*j/1.414213)))
                        if self.is_free(candidate):
                            neighbors.append(candidate)
                    else:
                        candidate = self.snap_to_grid((x[0]+(self.resolution*i),x[1]+(self.resolution*j)))
                        if self.is_free(candidate):
                            neighbors.append(candidate)
        ########## Code ends here ##########
        return neighbors

    def find_best_est_cost_through(self):
        """
        Gets the state in open_set that has the lowest est_cost_through
        Output: A tuple, the state found in open_set that has the lowest est_cost_through
        """
        return min(self.open_set, key=lambda x: self.est_cost_through[x])

    def reconstruct_path(self):
        """
        Use the came_from map to reconstruct a path from the initial location to
        the goal location
        Output:
            A list of tuples, which is a list of the states that go from start to goal
        """
        path = [self.x_goal]
        current = path[-1]
        while current != self.x_init:
            path.append(self.came_from[current])
            current = path[-1]
        return list(reversed(path))

    # def plot_path(self, fig_num=0, show_init_label=True):
    #     """Plots the path found in self.path and the obstacles"""
    #     if not self.path:
    #         return

    #     self.occupancy.plot(fig_num)

    #     solution_path = np.asarray(self.path)
    #     plt.plot(solution_path[:,0],solution_path[:,1], color="green", linewidth=2, label="A* solution path", zorder=10)
    #     plt.scatter([self.x_init[0], self.x_goal[0]], [self.x_init[1], self.x_goal[1]], color="green", s=30, zorder=10)
    #     if show_init_label:
    #         plt.annotate(r"$x_{init}$", np.array(self.x_init) + np.array([.2, .2]), fontsize=16)
    #     plt.annotate(r"$x_{goal}$", np.array(self.x_goal) + np.array([.2, .2]), fontsize=16)
    #     plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.03), fancybox=True, ncol=3)

    #     plt.axis([0, self.occupancy.width, 0, self.occupancy.height])

    # def plot_tree(self, point_size=15):
    #     plot_line_segments([(x, self.came_from[x]) for x in self.open_set if x != self.x_init], linewidth=1, color="blue", alpha=0.2)
    #     plot_line_segments([(x, self.came_from[x]) for x in self.closed_set if x != self.x_init], linewidth=1, color="blue", alpha=0.2)
    #     px = [x[0] for x in self.open_set | self.closed_set if x != self.x_init and x != self.x_goal]
    #     py = [x[1] for x in self.open_set | self.closed_set if x != self.x_init and x != self.x_goal]
    #     plt.scatter(px, py, color="blue", s=point_size, zorder=10, alpha=0.2)

    def solve(self):
        """
        Solves the planning problem using the A* search algorithm. It places
        the solution as a list of tuples (each representing a state) that go
        from self.x_init to self.x_goal inside the variable self.path
        Input:
            None
        Output:
            Boolean, True if a solution from x_init to x_goal was found

        HINTS:  We're representing the open and closed sets using python's built-in
                set() class. This allows easily adding and removing items using
                .add(item) and .remove(item) respectively, as well as checking for
                set membership efficiently using the syntax "if item in set".
        """
        ########## Code starts here ##########
        while len(self.open_set)>0:
            x_curr = self.find_best_est_cost_through()
            if x_curr == self.x_goal:
                self.path=self.reconstruct_path()
                return True
            self.open_set.remove(x_curr)
            self.closed_set.add(x_curr)
            for neighbor in self.get_neighbors(x_curr):
                if neighbor in self.closed_set:
                    continue
                tentative_cost_to_arrive = self.cost_to_arrive[x_curr]+self.distance(x_curr,neighbor)
                if neighbor not in self.open_set:
                    self.open_set.add(neighbor)
                elif tentative_cost_to_arrive>self.cost_to_arrive[neighbor]:
                    continue
                self.came_from[neighbor]=x_curr
                self.cost_to_arrive[neighbor]=tentative_cost_to_arrive
                self.est_cost_through[neighbor]=tentative_cost_to_arrive+self.distance(neighbor,x_curr)

        return False
        ########## Code ends here ##########

class RRT(object):
    """RRT motion planner"""

    def __init__(
        self,
        statespace_lo,
        statespace_hi,
        x_init,
        x_goal,
        occupancy,
        resolution=1.0,
        step_size=1.0,
        max_iters=5000,
        goal_sample_rate=0.05,
        goal_tolerance=0.5,
        seed=None,
    ):
        self.statespace_lo = tuple(statespace_lo)
        self.statespace_hi = tuple(statespace_hi)
        self.occupancy = occupancy
        self.resolution = resolution
        self.x_offset = x_init
        self.x_init = tuple(self.snap_to_grid(x_init))
        self.x_goal = tuple(self.snap_to_grid(x_goal))

        self.step_size = step_size
        self.max_iters = max_iters
        self.goal_sample_rate = goal_sample_rate  # probability of sampling the goal
        self.goal_tolerance = goal_tolerance

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # RRT data structures
        self.vertices = [self.x_init]     # list of nodes
        self.came_from = {}               # child -> parent mapping (parent for each vertex except start)

        self.path = None

    def is_free(self, x):
        """Check if a single state is inside bounds and free in the occupancy grid."""
        x = tuple(x)
        inside = (
            x[0] >= self.statespace_lo[0]
            and x[0] <= self.statespace_hi[0]
            and x[1] >= self.statespace_lo[1]
            and x[1] <= self.statespace_hi[1]
        )
        if not inside:
            return False
        # occupancy expects numpy array
        try:
            return bool(self.occupancy.is_free(np.asarray(x)))
        except Exception:
            # fallback: if occupancy doesn't provide is_free or fails, assume free
            return False

    def distance(self, x1, x2):
        x1 = np.array(x1)
        x2 = np.array(x2)
        return float(np.linalg.norm(x1 - x2))

    def snap_to_grid(self, x):
        """Snap a tuple x to the discrete grid defined by resolution and x_offset."""
        return (
            self.resolution * round((x[0] - self.x_offset[0]) / self.resolution) + self.x_offset[0],
            self.resolution * round((x[1] - self.x_offset[1]) / self.resolution) + self.x_offset[1],
        )

    def sample_random(self):
        """Return a random sample in the state space (or the goal according to goal_sample_rate)."""
        if random.random() < self.goal_sample_rate:
            return self.x_goal
        rx = random.uniform(self.statespace_lo[0], self.statespace_hi[0])
        ry = random.uniform(self.statespace_lo[1], self.statespace_hi[1])
        return (rx, ry)

    def nearest_vertex(self, x_rand):
        """Return the vertex in self.vertices closest to x_rand (linear search)."""
        best = None
        best_d = float("inf")
        for v in self.vertices:
            d = self.distance(v, x_rand)
            if d < best_d:
                best_d = d
                best = v
        return best

    def steer(self, x_near, x_rand):
        """
        Move from x_near toward x_rand by at most self.step_size.
        Returns the new point (not yet added to the tree).
        """
        x_near = np.array(x_near, dtype=float)
        x_rand = np.array(x_rand, dtype=float)
        vec = x_rand - x_near
        dist = np.linalg.norm(vec)
        if dist == 0.0:
            return tuple(self.snap_to_grid(x_near))
        if dist <= self.step_size:
            x_new = x_rand
        else:
            x_new = x_near + (vec / dist) * self.step_size
        x_new = tuple(self.snap_to_grid((float(x_new[0]), float(x_new[1]))))
        return x_new

    def segment_collision_free(self, a, b):
        """
        Check whether the straight-line segment from a to b lies in C_free.
        We discretize the segment with spacing approx self.resolution*0.5 and
        check each sample point with self.is_free.
        """
        a = np.array(a, dtype=float)
        b = np.array(b, dtype=float)
        dist = np.linalg.norm(b - a)
        if dist == 0.0:
            return self.is_free(tuple(a))
        # choose step smaller than resolution for safety
        step = max(self.resolution * 0.5, 0.01)
        n_steps = max(2, int(np.ceil(dist / step)))
        for t in np.linspace(0.0, 1.0, n_steps + 1):
            p = a + t * (b - a)
            p_snap = tuple(self.snap_to_grid((float(p[0]), float(p[1]))))
            if not self.is_free(p_snap):
                return False
        return True

    def reconstruct_path(self, last_node):
        """Reconstruct path from start to last_node using came_from map."""
        path = [last_node]
        current = last_node
        while current != self.x_init:
            current = self.came_from[current]
            path.append(current)
        return list(reversed(path))

    def solve(self):
        """
        Run RRT. Returns True if a path to the goal region was found (self.path set).
        """
        for it in range(self.max_iters):
            x_rand = self.sample_random()
            x_near = self.nearest_vertex(x_rand)
            x_new = self.steer(x_near, x_rand)

            # If x_new is not free or the segment from x_near -> x_new collides, skip
            if not self.is_free(x_new):
                continue
            if not self.segment_collision_free(x_near, x_new):
                continue

            # Add node and record parent
            self.vertices.append(x_new)
            self.came_from[x_new] = x_near

            # Check goal tolerance
            if self.distance(x_new, self.x_goal) <= self.goal_tolerance:
                # Optionally add the goal itself if the straight segment is collision-free
                if self.segment_collision_free(x_new, self.x_goal):
                    self.came_from[self.x_goal] = x_new
                    self.vertices.append(self.x_goal)
                    self.path = self.reconstruct_path(self.x_goal)
                else:
                    self.path = self.reconstruct_path(x_new)
                return True

        # failed to find within max_iters
        return False

class Navigator(BaseNavigator):
    def __init__(self, planner_type="AStar") -> None:
        """
        planner_type: "AStar" or "RRT"
        """
        super().__init__("Navigator")

        self.planner_type = planner_type       # <── NEW
        self.kp = 2.0
        self.kpx = 2.0
        self.kpy = 2.0
        self.kdx = 2.0
        self.kdy = 2.0

        self.coeffs = np.zeros(8)

    def reset(self) -> None:
        self.V_prev = 0.
        self.om_prev = 0.
        self.t_prev = 0.


    def compute_heading_control(self, h_curr: TurtleBotState, h_des: TurtleBotState) -> TurtleBotControl:
        """
        Takes in the current and desired state of type TurtleBotState,
        and returns control message of type TurtleBotControl.
        """
        err = wrap_angle(h_des.theta - h_curr.theta)

        msg = TurtleBotControl()
        msg.v = 0.
        msg.omega = self.kp * err
        return msg

    def compute_path(self, state, goal, occupancy, resolution, horizon):
        """
        Runs either A* or RRT depending on self.planner_type.
        Returns a list of (x,y) tuples OR None.
        """
        lo = (state.x - horizon, state.y - horizon)
        hi = (state.x + horizon, state.y + horizon)

        if self.planner_type == "AStar":
            planner = AStar(lo, hi, (state.x, state.y), (goal.x, goal.y),
                            occupancy, resolution=resolution)
            success = planner.solve()
            return planner.path if success else None

        elif self.planner_type == "RRT":
            planner = RRT(lo, hi,
                          (state.x, state.y), (goal.x, goal.y),
                          occupancy,
                          resolution=resolution,
                          step_size=0.5,          # tune these as needed
                          max_iters=3000,
                          goal_sample_rate=0.05,
                          goal_tolerance=0.4)
            success = planner.solve()
            return planner.path if success else None

        else:
            print(f"[Navigator] Unknown planner_type: {self.planner_type}")
            return None


    def compute_trajectory_tracking_control(self, state: TurtleBotState, plan: TrajectoryPlan, t: float,) -> TurtleBotControl:
        """ Compute control target using a trajectory tracking controller

        Args:
            state (TurtleBotState): current robot state
            plan (TrajectoryPlan): planned trajectory
            t (float): current timestep

        Returns:
            TurtleBotControl: control command
        """
        dt = t - self.t_prev
        ######## desired state ##########3
        x_d=float(splev(t, plan.path_x_spline, der=0))
        y_d=float(splev(t, plan.path_y_spline, der=0))
        xd_d=float(splev(t, plan.path_x_spline, der=1))
        yd_d=float(splev(t, plan.path_y_spline, der=1))
        xdd_d=float(splev(t, plan.path_x_spline, der=2))
        ydd_d=float(splev(t, plan.path_y_spline, der=2))

        ######## Current State ##########
        x, y, th = state.x, state.y, state.theta

        ########## Code starts here ##########
        self.V_prev = max(self.V_prev, V_PREV_THRES)
        u1 = xdd_d + self.kpx*(x_d-x) + self.kdx*(xd_d-(self.V_prev*np.cos(th)))
        u2 = ydd_d + self.kpy*(y_d-y) + self.kdy*(yd_d-(self.V_prev*np.sin(th)))

        U = np.array([u1, u2])
        M = np.array([[np.cos(th), -self.V_prev*np.sin(th)],
                    [np.sin(th), self.V_prev*np.cos(th)]])
        control = np.linalg.solve(M, U)

        V = self.V_prev + control[0]*dt
        om = control[1]
        ########## Code ends here ##########

        # save the commands that were applied and the time
        self.t_prev = t
        self.V_prev = V
        self.om_prev = om

        msg = TurtleBotControl()
        msg.omega = om
        msg.v = V
        return msg


    def compute_trajectory_plan(self, state: TurtleBotState, goal: TurtleBotState,
                                occupancy: StochOccupancyGrid2D,
                                resolution: float, horizon: float
                                ) -> T.Optional[TrajectoryPlan]:

        print(f"[Navigator] Using planner: {self.planner_type}")

        # -------- get path using selected planner -------- #
        path = self.compute_path(state, goal, occupancy, resolution, horizon)
        if path is None or len(path) < 4:
            print("[Navigator] No valid path found")
            return None

        path = np.asarray(path)
        self.reset()

        # ---- Spline fitting (unchanged) ---- #
        v_desired, spline_alpha = 0.15, 0.05
        ts = [resolution / v_desired * i for i in range(len(path))]
        path_x_spline = splrep(ts, path[:,0], s=spline_alpha)
        path_y_spline = splrep(ts, path[:,1], s=spline_alpha)

        return TrajectoryPlan(
            path=path,
            path_x_spline=path_x_spline,
            path_y_spline=path_y_spline,
            duration=ts[-1]
        )


def main():
    rclpy.init(args=None)
    print("main")
    node = Navigator()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == "__main__":
    main()
