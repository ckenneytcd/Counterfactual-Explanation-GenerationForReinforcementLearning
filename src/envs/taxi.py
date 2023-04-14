from contextlib import closing
from io import StringIO
from os import path
from typing import Optional

import numpy as np

from gym import Env, logger, spaces, utils
from gym.envs.toy_text.utils import categorical_sample
from gym.error import DependencyNotInstalled



WINDOW_SIZE = (550, 350)


class TaxiEnv(Env):
    """

    The Taxi Problem
    from "Hierarchical Reinforcement Learning with the MAXQ Value Function Decomposition"
    by Tom Dietterich

    ### Description
    There are four designated locations in the grid world indicated by R(ed),
    G(reen), Y(ellow), and B(lue). When the episode starts, the taxi starts off
    at a random square and the passenger is at a random location. The taxi
    drives to the passenger's location, picks up the passenger, drives to the
    passenger's destination (another one of the four specified locations), and
    then drops off the passenger. Once the passenger is dropped off, the episode ends.

    Map:

        +---------+
        |R: | : :G|
        | : | : : |
        | : : : : |
        | | : | : |
        |Y| : |B: |
        +---------+

    ### Actions
    There are 6 discrete deterministic actions:
    - 0: move south
    - 1: move north
    - 2: move east
    - 3: move west
    - 4: pickup passenger
    - 5: drop off passenger

    ### Observations
    There are 500 discrete states since there are 25 taxi positions, 5 possible
    locations of the passenger (including the case when the passenger is in the
    taxi), and 4 destination locations.

    Note that there are 400 states that can actually be reached during an
    episode. The missing states correspond to situations in which the passenger
    is at the same location as their destination, as this typically signals the
    end of an episode. Four additional states can be observed right after a
    successful episodes, when both the passenger and the taxi are at the destination.
    This gives a total of 404 reachable discrete states.

    Each state space is represented by the tuple:
    (taxi_row, taxi_col, passenger_location, destination)

    An observation is an integer that encodes the corresponding state.
    The state tuple can then be decoded with the "decode" method.

    Passenger locations:
    - 0: R(ed)
    - 1: G(reen)
    - 2: Y(ellow)
    - 3: B(lue)
    - 4: in taxi

    Destinations:
    - 0: R(ed)
    - 1: G(reen)
    - 2: Y(ellow)
    - 3: B(lue)

    ### Info

    ``step`` and ``reset()`` will return an info dictionary that contains "p" and "action_mask" containing
        the probability that the state is taken and a mask of what actions will result in a change of state to speed up training.

    As Taxi's initial state is a stochastic, the "p" key represents the probability of the
    transition however this value is currently bugged being 1.0, this will be fixed soon.
    As the steps are deterministic, "p" represents the probability of the transition which is always 1.0

    For some cases, taking an action will have no effect on the state of the agent.
    In v0.25.0, ``info["action_mask"]`` contains a np.ndarray for each of the action specifying
    if the action will change the state.

    To sample a modifying action, use ``action = env.action_space.sample(info["action_mask"])``
    Or with a Q-value based algorithm ``action = np.argmax(q_values[obs, np.where(info["action_mask"] == 1)[0]])``.

    ### Rewards
    - -1 per step unless other reward is triggered.
    - +20 delivering passenger.
    - -10  executing "pickup" and "drop-off" actions illegally.

    ### Arguments

    ```
    gym.make('Taxi-v3')
    ```

    ### Version History
    * v3: Map Correction + Cleaner Domain Description, v0.25.0 action masking added to the reset and step information
    * v2: Disallow Taxi start location = goal location, Update Taxi observations in the rollout, Update Taxi reward threshold.
    * v1: Remove (3,2) from locs, add passidx<4 check
    * v0: Initial versions release
    """

    metadata = {
        "render_modes": ["human", "ansi", "rgb_array"],
        "render_fps": 4,
    }

    def __init__(self, render_mode: Optional[str] = None):
        self.MAP = [
            "+---------+",
            "|R: | : :G|",
            "| : | : : |",
            "| : : : : |",
            "| | : | : |",
            "|Y| : |B: |",
            "+---------+",
        ]
        self.desc = np.asarray(self.MAP, dtype="c")

        self.locs = locs = [(0, 0), (0, 4), (4, 0), (4, 3)]
        self.locs_colors = [(255, 0, 0), (0, 255, 0), (255, 255, 0), (0, 0, 255)]

        self.s = 0

        self.state_dim = 4

        self.lows = np.array([0]*self.state_dim)
        self.highs = np.array([3]*self.state_dim)

        self.max_steps = 1000
        num_states = 500
        self.num_rows = self.state_dim+1
        self.num_columns = self.state_dim+1
        self.max_row = self.state_dim
        self.max_col = self.state_dim
        self.initial_state_distrib = np.zeros(num_states)
        num_actions = 6
        self.P = {
            state: {action: [] for action in range(num_actions)}
            for state in range(num_states)
        }
        for row in range(self.num_rows):
            for col in range(self.num_columns):
                for pass_idx in range(len(locs) + 1):  # +1 for being inside taxi
                    for dest_idx in range(len(locs)):
                        state = self.encode(row, col, pass_idx, dest_idx)
                        if pass_idx < 4 and pass_idx != dest_idx:
                            self.initial_state_distrib[state] += 1
                        for action in range(num_actions):
                            # defaults
                            new_row, new_col, new_pass_idx = row, col, pass_idx
                            reward = (
                                -1
                            )  # default reward when there is no pickup/dropoff
                            terminated = False
                            taxi_loc = (row, col)

                            if action == 0:
                                new_row = min(row + 1, self.max_row)
                            elif action == 1:
                                new_row = max(row - 1, 0)
                            if action == 2 and self.desc[1 + row, 2 * col + 2] == b":":
                                new_col = min(col + 1, self.max_col)
                            elif action == 3 and self.desc[1 + row, 2 * col] == b":":
                                new_col = max(col - 1, 0)
                            elif action == 4:  # pickup
                                if pass_idx < 4 and taxi_loc == locs[pass_idx]:
                                    new_pass_idx = 4
                                else:  # passenger not at location
                                    reward = -10
                            elif action == 5:  # dropoff
                                if (taxi_loc == locs[dest_idx]) and pass_idx == 4:
                                    new_pass_idx = dest_idx
                                    terminated = True
                                    reward = 20
                                elif (taxi_loc in locs) and pass_idx == 4:
                                    new_pass_idx = locs.index(taxi_loc)
                                else:  # dropoff at wrong location
                                    reward = -10
                            new_state = self.encode(
                                new_row, new_col, new_pass_idx, dest_idx
                            )
                            self.P[state][action].append(
                                (1.0, new_state, reward, terminated)
                            )
        self.initial_state_distrib /= self.initial_state_distrib.sum()
        self.action_space = spaces.Discrete(num_actions)
        self.observation_space = spaces.Discrete(num_states)

        self.render_mode = render_mode

        # pygame utils
        self.window = None
        self.clock = None
        self.cell_size = (
            WINDOW_SIZE[0] / self.desc.shape[1],
            WINDOW_SIZE[1] / self.desc.shape[0],
        )
        self.taxi_imgs = None
        self.taxi_orientation = 0
        self.passenger_img = None
        self.destination_img = None
        self.median_horiz = None
        self.median_vert = None
        self.background_img = None

    def encode(self, taxi_row, taxi_col, pass_loc, dest_idx):
        # (5) 5, 5, 4
        i = taxi_row
        i *= 5
        i += taxi_col
        i *= 5
        i += pass_loc
        i *= 4
        i += dest_idx
        return i

    def decode(self, i):
        out = []
        out.append(i % 4)
        i = i // 4
        out.append(i % 5)
        i = i // 5
        out.append(i % 5)
        i = i // 5
        out.append(i)
        assert 0 <= i < 5
        return reversed(out)

    def action_mask(self, state: int):
        """Computes an action mask for the action space using the state information."""
        mask = np.zeros(6, dtype=np.int8)
        taxi_row, taxi_col, pass_loc, dest_idx = self.decode(state)
        if taxi_row < 4:
            mask[0] = 1
        if taxi_row > 0:
            mask[1] = 1
        if taxi_col < 4 and self.desc[taxi_row + 1, 2 * taxi_col + 2] == b":":
            mask[2] = 1
        if taxi_col > 0 and self.desc[taxi_row + 1, 2 * taxi_col] == b":":
            mask[3] = 1
        if pass_loc < 4 and (taxi_row, taxi_col) == self.locs[pass_loc]:
            mask[4] = 1
        if pass_loc == 4 and (
            (taxi_row, taxi_col) == self.locs[dest_idx]
            or (taxi_row, taxi_col) in self.locs
        ):
            mask[5] = 1
        return mask

    def step(self, a):
        transitions = self.P[self.s][a]
        i = categorical_sample([t[0] for t in transitions], self.np_random)
        p, s, r, t = transitions[i]
        self.s = s
        self.lastaction = a

        if self.render_mode == "human":
            self.render()
        #state number, reward, done, info
        return (list(self.decode(s)), r, t, {"prob": p, "action_mask": self.action_mask(s)})
        #return (int(s), r, t, False, {"prob": p, "action_mask": self.action_mask(s)})
        

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        self.s = categorical_sample(self.initial_state_distrib, self.np_random)
        self.lastaction = None
        self.taxi_orientation = 0

        if self.render_mode == "human":
            self.render()
        return list(self.decode(self.s))
        #return int(self.s), {"prob": 1.0, "action_mask": self.action_mask(self.s)}

    def render(self):
        if self.render_mode is None:
            logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym("{self.spec.id}", render_mode="rgb_array")'
            )
        if self.render_mode == "ansi":
            return self._render_text()
        else:  # self.render_mode in {"human", "rgb_array"}:
            return self._render_gui(self.render_mode)

    def _render_gui(self, mode):
        try:
            import pygame  # dependency to pygame only if rendering with human
        except ImportError:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gym[toy_text]`"
            )

        if self.window is None:
            pygame.init()
            pygame.display.set_caption("Taxi")
            if mode == "human":
                self.window = pygame.display.set_mode(WINDOW_SIZE)
            elif mode == "rgb_array":
                self.window = pygame.Surface(WINDOW_SIZE)

        assert (
            self.window is not None
        ), "Something went wrong with pygame. This should never happen."
        if self.clock is None:
            self.clock = pygame.time.Clock()
        if self.taxi_imgs is None:
            file_names = [
                path.join(path.dirname(__file__), "img/cab_front.png"),
                path.join(path.dirname(__file__), "img/cab_rear.png"),
                path.join(path.dirname(__file__), "img/cab_right.png"),
                path.join(path.dirname(__file__), "img/cab_left.png"),
            ]
            self.taxi_imgs = [
                pygame.transform.scale(pygame.image.load(file_name), self.cell_size)
                for file_name in file_names
            ]
        if self.passenger_img is None:
            file_name = path.join(path.dirname(__file__), "img/passenger.png")
            self.passenger_img = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_size
            )
        if self.destination_img is None:
            file_name = path.join(path.dirname(__file__), "img/hotel.png")
            self.destination_img = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_size
            )
            self.destination_img.set_alpha(170)
        if self.median_horiz is None:
            file_names = [
                path.join(path.dirname(__file__), "img/gridworld_median_left.png"),
                path.join(path.dirname(__file__), "img/gridworld_median_horiz.png"),
                path.join(path.dirname(__file__), "img/gridworld_median_right.png"),
            ]
            self.median_horiz = [
                pygame.transform.scale(pygame.image.load(file_name), self.cell_size)
                for file_name in file_names
            ]
        if self.median_vert is None:
            file_names = [
                path.join(path.dirname(__file__), "img/gridworld_median_top.png"),
                path.join(path.dirname(__file__), "img/gridworld_median_vert.png"),
                path.join(path.dirname(__file__), "img/gridworld_median_bottom.png"),
            ]
            self.median_vert = [
                pygame.transform.scale(pygame.image.load(file_name), self.cell_size)
                for file_name in file_names
            ]
        if self.background_img is None:
            file_name = path.join(path.dirname(__file__), "img/taxi_background.png")
            self.background_img = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_size
            )

        desc = self.desc

        for y in range(0, desc.shape[0]):
            for x in range(0, desc.shape[1]):
                cell = (x * self.cell_size[0], y * self.cell_size[1])
                self.window.blit(self.background_img, cell)
                if desc[y][x] == b"|" and (y == 0 or desc[y - 1][x] != b"|"):
                    self.window.blit(self.median_vert[0], cell)
                elif desc[y][x] == b"|" and (
                    y == desc.shape[0] - 1 or desc[y + 1][x] != b"|"
                ):
                    self.window.blit(self.median_vert[2], cell)
                elif desc[y][x] == b"|":
                    self.window.blit(self.median_vert[1], cell)
                elif desc[y][x] == b"-" and (x == 0 or desc[y][x - 1] != b"-"):
                    self.window.blit(self.median_horiz[0], cell)
                elif desc[y][x] == b"-" and (
                    x == desc.shape[1] - 1 or desc[y][x + 1] != b"-"
                ):
                    self.window.blit(self.median_horiz[2], cell)
                elif desc[y][x] == b"-":
                    self.window.blit(self.median_horiz[1], cell)

        for cell, color in zip(self.locs, self.locs_colors):
            color_cell = pygame.Surface(self.cell_size)
            color_cell.set_alpha(128)
            color_cell.fill(color)
            loc = self.get_surf_loc(cell)
            self.window.blit(color_cell, (loc[0], loc[1] + 10))

        taxi_row, taxi_col, pass_idx, dest_idx = self.decode(self.s)

        if pass_idx < 4:
            self.window.blit(self.passenger_img, self.get_surf_loc(self.locs[pass_idx]))

        if self.lastaction in [0, 1, 2, 3]:
            self.taxi_orientation = self.lastaction
        dest_loc = self.get_surf_loc(self.locs[dest_idx])
        taxi_location = self.get_surf_loc((taxi_row, taxi_col))

        if dest_loc[1] <= taxi_location[1]:
            self.window.blit(
                self.destination_img,
                (dest_loc[0], dest_loc[1] - self.cell_size[1] // 2),
            )
            self.window.blit(self.taxi_imgs[self.taxi_orientation], taxi_location)
        else:  # change blit order for overlapping appearance
            self.window.blit(self.taxi_imgs[self.taxi_orientation], taxi_location)
            self.window.blit(
                self.destination_img,
                (dest_loc[0], dest_loc[1] - self.cell_size[1] // 2),
            )

        if mode == "human":
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        elif mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.window)), axes=(1, 0, 2)
            )

    def get_surf_loc(self, map_loc):
        return (map_loc[1] * 2 + 1) * self.cell_size[0], (
            map_loc[0] + 1
        ) * self.cell_size[1]

    def _render_text(self):
        desc = self.desc.copy().tolist()
        outfile = StringIO()

        out = [[c.decode("utf-8") for c in line] for line in desc]
        taxi_row, taxi_col, pass_idx, dest_idx = self.decode(self.s)

        def ul(x):
            return "_" if x == " " else x

        if pass_idx < 4:
            out[1 + taxi_row][2 * taxi_col + 1] = utils.colorize(
                out[1 + taxi_row][2 * taxi_col + 1], "yellow", highlight=True
            )
            pi, pj = self.locs[pass_idx]
            out[1 + pi][2 * pj + 1] = utils.colorize(
                out[1 + pi][2 * pj + 1], "blue", bold=True
            )
        else:  # passenger in taxi
            out[1 + taxi_row][2 * taxi_col + 1] = utils.colorize(
                ul(out[1 + taxi_row][2 * taxi_col + 1]), "green", highlight=True
            )

        di, dj = self.locs[dest_idx]
        out[1 + di][2 * dj + 1] = utils.colorize(out[1 + di][2 * dj + 1], "magenta")
        outfile.write("\n".join(["".join(row) for row in out]) + "\n")
        if self.lastaction is not None:
            outfile.write(
                f"  ({['South', 'North', 'East', 'West', 'Pickup', 'Dropoff'][self.lastaction]})\n"
            )
        else:
            outfile.write("\n")

        with closing(outfile):
            return outfile.getvalue()

    def close(self):
        if self.window is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()

    def get_actions(self, state):
        
        possible = []
        taxirow = state[0]
        taxicol = state[1]
        passloc = state[2]
        destloc = state[3]

        #Move South
        if(taxirow < self.max_row):
            possible += [0]
        #Move North
        if(taxirow > 0):
            possible += [1]
        safemove = True
        #Move East (baracades included)
        if(taxicol < self.max_col):
            if(taxicol == 1 and (taxirow == 0 or taxirow == 1)):
                safemove = False
            if(taxicol == 0 and (taxirow == self.max_col or taxirow == self.max_col-1)):
                safemove = False
            if(taxicol == 2 and (taxirow == self.max_col or taxirow == self.max_col-1)):
                safemove = False
            if(safemove):
                possible += [2]   
        safemove = True         
        #Move West (baracades included)
        if(taxicol > 0):
            if(taxicol == 2 and (taxirow == 0 or taxirow == 1)):
                safemove = False
            if(taxicol == 1 and (taxirow == self.max_col or taxirow == self.max_col-1)):
                    safemove = False
            if(taxicol == 3 and (taxirow == self.max_col or taxirow == self.max_col-1)):
                    safemove = False
            if(safemove):
                possible += [3]

        #Pick up
        if(passloc == 0 and taxirow == 0 and taxicol == 0):
            possible += [4]
        if(passloc == 1 and taxirow == 0 and taxicol == self.max_col):
            possible += [4]
        if(passloc == 2 and taxirow == self.max_row and taxicol == 0):
            possible += [4]
        if(passloc == 3 and taxirow == self.max_row and taxicol == self.max_col-1):
            possible += [4]
        #Drop off
        if(passloc == 4 and destloc == 0 and taxirow == 0 and taxicol == 0):
            possible += [5]
        if(passloc == 4 and destloc == 1 and taxirow == 0 and taxicol == self.max_col):
            possible += [5]
        if(passloc == 4 and destloc == 2 and taxirow == self.max_row and taxicol == 0):
            possible += [5]
        if(passloc == 4 and destloc == 3 and taxirow == self.max_row and taxicol == self.max_col-1):
            possible += [5]

        #Remove the direction that brings it back to the root
                
        return possible
    
    def render_state(self, state):
        desc = self.desc.copy().tolist()
        outfile = StringIO()
        out = [[c.decode("utf-8") for c in line] for line in desc]
        taxi_row, taxi_col, pass_idx, dest_idx = state

        def ul(x):
            return "_" if x == " " else x

        if pass_idx < 4:
            out[1 + taxi_row][2 * taxi_col + 1] = utils.colorize(
                out[1 + taxi_row][2 * taxi_col + 1], "yellow", highlight=True
            )
            pi, pj = self.locs[pass_idx]
            out[1 + pi][2 * pj + 1] = utils.colorize(
                out[1 + pi][2 * pj + 1], "blue", bold=True
            )
        else:  # passenger in taxi
            out[1 + taxi_row][2 * taxi_col + 1] = utils.colorize(
                ul(out[1 + taxi_row][2 * taxi_col + 1]), "green", highlight=True
            )

        di, dj = self.locs[dest_idx]
        out[1 + di][2 * dj + 1] = utils.colorize(out[1 + di][2 * dj + 1], "magenta")
        outfile.write("\n".join(["".join(row) for row in out]) + "\n")

        with closing(outfile):
            print(outfile.getvalue())
            return outfile.getvalue()
        
    def text_render_state(self, state):
        desc = self.desc.copy().tolist()
        outfile = StringIO()
        out = [[c.decode("utf-8") for c in line] for line in desc]
        taxi_row, taxi_col, pass_idx, dest_idx = state

        def ul(x):
            return "_" if x == " " else x

        if pass_idx < 4:
            out[1 + taxi_row][2 * taxi_col + 1] = utils.colorize(
                out[1 + taxi_row][2 * taxi_col + 1], "yellow", highlight=True
            )
            pi, pj = self.locs[pass_idx]
            out[1 + pi][2 * pj + 1] = utils.colorize(
                out[1 + pi][2 * pj + 1], "blue", bold=True
            )
        else:  # passenger in taxi
            out[1 + taxi_row][2 * taxi_col + 1] = utils.colorize(
                ul(out[1 + taxi_row][2 * taxi_col + 1]), "green", highlight=True
            )

        di, dj = self.locs[dest_idx]
        out[1 + di][2 * dj + 1] = utils.colorize(out[1 + di][2 * dj + 1], "magenta")
        outfile.write("\n".join(["".join(row) for row in out]) + "\n")

        with closing(outfile):
            return outfile.getvalue()

    def check_done(self, state):
        if state[2] == state[3]:
            return True
        else:
            return False
        
    def set_state(self,state):
        self.reset()
        self.s = self.encode(state[0], state[1], state[2], state[3])
        return
    
    def equal_states(self, s1, s2):
        return s1[0] == s2[0] and s1[1] == s2[1] and s1[2] == s2[2] and s1[3] == s2[3]
        
    def realistic(self, x, f, prev_actions):
        #Pickup made outside of possible locations 
        if(prev_actions and prev_actions[-1] == 4):
            if not((f[2] == 0 and x[0] == 0 and x[1] == 0)):
                if not(f[2] == 1 and x[0] == 0 and x[1] == 4):
                    if not(f[2] == 2 and x[0] == 4 and x[1] == 0):
                        if not(f[2] == 3 and x[0] == 4 and x[1] == 4-1):
                            #print("illgeal pickup, F:", f, "X: ", x, "possible: ", self.get_actions(x))
                            return False
        #Dropoff made outside of possible locations 
        if(prev_actions and prev_actions[-1] == 5):
            if not((f[3] == 0 and x[0] == 0 and x[1] == 0)):
                if not(f[3] == 1 and x[0] == 0 and x[1] == 4):
                    if not(f[3] == 2 and x[0] == 4 and x[1] == 0):
                        if not(f[3] == 3 and x[0] == 4 and x[1] == 4-1):
                            #print("illgeal dropoff, F:", f, "X: ", x, "possible: ", self.get_actions(x))
                            return False

        #Taxi row out of bounds    
        if x[0] < 0 or x[0] > 4:
            #print("taxirow out of bounds")
            return False
        #Taxi col out of bounds   
        if x[1] < 0 or x[1] > 4:
            #print("taxicol out of bounds")
            return False
        #Passanger loc out of bounds   
        if x[2] < 0 or x[2] > 4:
            #print("pass out of bounds")
            return False
        #Destination out of bounds   
        if x[3] < 0 or x[3] > 3:
            #print("dst out of bounds")
            return False
        
        #Path unnesairily long
        opposites = {0: 1, 1: 0, 2: 3, 3: 2}
        current_action = None
        for action in prev_actions:
            if(current_action != None and current_action < 4 and action < 4):
                if current_action == opposites[action]:
                    return False
            current_action = action
            
        return True
    
    def actionable(self, x, f, cfmode):
        if(cfmode == "ACTION" or cfmode == "NETACTION"):
            #Passanger location changed without being in the taxi
            if x[2] != f[2] and (x[2] != 4 and f[2] != 4):
                #print("passloc changed without pickup or dropoff")
                return False
            #Destination change
            if x[3] != f[3]:
                #print("dest changed")
                return False
        elif(cfmode == "ACTION_CHANGELOC"):
            #Destination change
            if x[3] != f[3]:
                #print("dest changed")
                return False
        elif(cfmode == "ACTION_CHANGEDST"):
            #Passanger location changed without being in the taxi
            if x[2] != f[2] and not(x[2] == 4 or f[2] == 4):
                #print("passloc changed, F:", f, "X: ", x, "possible: ", self.get_actions(x), 'path: ')
                return False
        elif(cfmode == "ACTION_CHANGEBOTH"):
            #Both passanger location and destination changed
            if x[2] != f[2] and x[3] != f[3]:
                #print("cannot change both at same time", x, f)
                return False
        return True

    def writable_state(self, s):
        ws = 'FState =  TaxiPos:({},{}) PassLoc: {} DestLoc: {}'.format(s[0], s[1], s[2], s[3])
        return ws
    def cf_writable_state(self, s, action_path):
        ws = 'CFState =  TaxiPos:({},{}) PassLoc: {} DestLoc: {} Action path: {}'.format(s[0], s[1], s[2], s[3], action_path)
        return ws
    def single_step_nochange(self, state, action):
        current_state = self.s
        self.set_state(list(state))
        outcome, _, _, _ = self.step(action)

        self.set_state(list(self.decode(current_state)))
        
        return outcome



# Taxi rider from https://franuka.itch.io/rpg-asset-pack
# All other assets by Mel Tillery http://www.cyaneus.com/
