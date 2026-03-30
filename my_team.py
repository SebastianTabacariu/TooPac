# baseline_team.py
# ---------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# baseline_team.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import random
import util

from capture_agents import CaptureAgent
from game import Directions
from util import nearest_point, manhattan_distance ## NEW


#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='OffensiveReflexAgent', second='DefensiveReflexAgent', num_training=0):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --redOpts and --blueOpts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """
    return [eval(first)(first_index), eval(second)(second_index)]


##########
# Agents #
##########

class ReflexCaptureAgent(CaptureAgent):
    """
    A base class for reflex agents that choose score-maximizing actions
    """

    ENDGAME_STEPS = 400            # last N ticks count as "endgame"
    COMFORTABLE_LEAD = 8           # score lead where we feel safe
    DESIRED_DISTANCE = 4           # ideal spacing between teammates
    PHASE_TRIGGER = 600            # Phase 1 (defend) -> Phase 2 (attack) transition point

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.start = None

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)

        # Precompute all walkable tiles for belief tracking.
        walls = game_state.get_walls()
        self.legal_positions = []
        for x in range(walls.width):
            for y in range(walls.height):
                if not walls[x][y]:
                    self.legal_positions.append((x, y))

        self._legal_set = set(self.legal_positions)

        # Each enemy starts at their known spawn.
        self.enemy_beliefs = {}
        for opp in self.get_opponents(game_state):
            spawn = game_state.get_initial_agent_position(opp)
            self.enemy_beliefs[opp] = [spawn]

  

    def _update_beliefs(self, game_state):
        """
        Update belief distributions for each opponent using:
        1. Exact position if the enemy is visible (within Manhattan distance 5).
        2. Noisy distance readings to narrow down possible positions.
        3. Food disappearance to pin enemy location (very high confidence).

        We call this at the start of every choose_action to keep beliefs fresh.
        """
        my_pos = game_state.get_agent_state(self.index).get_position()
        noisy_distances = game_state.get_agent_distances()

        # If an enemy was pinned nearby but now has a huge noisy distance, they probably respawned.
        for opp in self.get_opponents(game_state):
            opp_state = game_state.get_agent_state(opp)
            prev_beliefs = self.enemy_beliefs.get(opp, [])
            if len(prev_beliefs) <= 5 and game_state.get_agent_position(opp) is None:
                noisy_d = noisy_distances[opp]
                if prev_beliefs:
                    avg_x = sum(p[0] for p in prev_beliefs) / len(prev_beliefs)
                    avg_y = sum(p[1] for p in prev_beliefs) / len(prev_beliefs)
                    prev_est_dist = manhattan_distance(my_pos, (avg_x, avg_y))
                    if noisy_d > prev_est_dist + 8:
                        spawn = game_state.get_initial_agent_position(opp)
                        self.enemy_beliefs[opp] = [spawn]

        # If our food just vanished, pin the closest invisible enemy near that spot.
        current_defended = set(self.get_food_you_are_defending(game_state).as_list())
        if hasattr(self, '_prev_belief_food'):
            missing_food = self._prev_belief_food - current_defended
        else:
            missing_food = set()
        self._prev_belief_food = current_defended

        food_pinned_opponents = set()
        for food_pos in missing_food:
            best_opp = None
            best_dist = float('inf')
            for opp in self.get_opponents(game_state):
                if game_state.get_agent_position(opp) is not None:
                    continue
                if opp in food_pinned_opponents:
                    continue
                for bpos in self.enemy_beliefs.get(opp, []):
                    d = manhattan_distance(bpos, food_pos)
                    if d < best_dist:
                        best_dist = d
                        best_opp = opp
            if best_opp is not None:
                pinned = [pos for pos in self.legal_positions
                          if manhattan_distance(pos, food_pos) <= 2]
                if pinned:
                    self.enemy_beliefs[best_opp] = pinned
                    food_pinned_opponents.add(best_opp)

        for opp in self.get_opponents(game_state):
            if opp in food_pinned_opponents:
                continue

            exact_pos = game_state.get_agent_position(opp)

            # Visible ,just store exact position.
            if exact_pos is not None:
                self.enemy_beliefs[opp] = [exact_pos]
                continue

            # Invisible, expand previous beliefs by 1 step, then filter by noisy distance.
            noisy_dist = noisy_distances[opp]
            low = max(0, noisy_dist - 6)
            high = noisy_dist + 6

            prev_positions = self.enemy_beliefs.get(opp, self.legal_positions)

            expanded = set()
            for pos in prev_positions:
                expanded.add(pos)
                px, py = int(pos[0]), int(pos[1])
                for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                    neighbor = (px + dx, py + dy)
                    if neighbor in self._legal_set:
                        expanded.add(neighbor)

            filtered = []
            for pos in expanded:
                d = manhattan_distance(my_pos, pos)
                if low <= d <= high:
                    if d > 5:  # we'd see them if they were within 5
                        filtered.append(pos)

            if not filtered:
                filtered = [pos for pos in self.legal_positions
                            if low <= manhattan_distance(my_pos, pos) <= high and
                            manhattan_distance(my_pos, pos) > 5]

            if filtered:
                self.enemy_beliefs[opp] = filtered



    def _get_likely_enemy_position(self, game_state, opp_index):
        """
        Returns the most likely position of an invisible enemy.
        If the enemy is visible, returns the exact position.
        """
        exact = game_state.get_agent_position(opp_index)
        if exact is not None:
            return exact

        beliefs = self.enemy_beliefs.get(opp_index, [])
        if not beliefs:
            return None

        defended_food = self.get_food_you_are_defending(game_state).as_list()
        if defended_food:
            best_pos = None
            best_dist = float('inf')
            for pos in beliefs:
                for food in defended_food:
                    d = manhattan_distance(pos, food)
                    if d < best_dist:
                        best_dist = d
                        best_pos = pos
            return best_pos
        else:
            avg_x = sum(p[0] for p in beliefs) / len(beliefs)
            avg_y = sum(p[1] for p in beliefs) / len(beliefs)
            return min(beliefs, key=lambda p: abs(p[0] - avg_x) + abs(p[1] - avg_y))




    def _get_likely_invader_entry(self, game_state):
        """
        Predict where invisible enemies are most likely to cross into our territory.
        Returns the boundary point closest to the most threatening estimated enemy position.
        Only returns a prediction when beliefs are concentrated enough to be meaningful
        (fewer than 30 possible positions), otherwise returns None to avoid misleading patrol.
        """
        opponents = self.get_opponents(game_state)
        best_boundary = None
        min_dist = float('inf')

        for opp in opponents:
            beliefs = self.enemy_beliefs.get(opp, [])
            if len(beliefs) > 30 or len(beliefs) == 0:
                continue

            likely_pos = self._get_likely_enemy_position(game_state, opp)
            if likely_pos is None:
                continue
            opp_state = game_state.get_agent_state(opp)
            if opp_state.is_pacman:
                continue
            if hasattr(self, 'home_boundary'):
                boundary = self.home_boundary
            elif hasattr(self, 'red_boundaries'):
                boundary = self.red_boundaries
            else:
                continue
            for b in boundary:
                d = manhattan_distance(likely_pos, b)
                if d < min_dist:
                    min_dist = d
                    best_boundary = b

        return best_boundary




    def choose_action(self, game_state):
        """
        Picks among the actions with the highest Q(s,a).
        """
        self._update_beliefs(game_state)

        actions = game_state.get_legal_actions(self.index)
        values = [self.evaluate(game_state, a) for a in actions]

        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]

        food_left = len(self.get_food(game_state).as_list())

        if food_left <= 2:
            best_dist = 9999
            best_action = None
            for action in actions:
                successor = self.get_successor(game_state, action)
                pos2 = successor.get_agent_position(self.index)
                dist = self.get_maze_distance(self.start, pos2)
                if dist < best_dist:
                    best_action = action
                    best_dist = dist
            return best_action

        return random.choice(best_actions)



    def get_successor(self, game_state, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearest_point(pos):
            return successor.generate_successor(self.index, action)
        else:
            return successor



    def evaluate(self, game_state, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.get_features(game_state, action)
        weights = self.get_weights(game_state, action)
        return features * weights



    def get_features(self, game_state, action):
        """
        Returns a counter of features for the state
        """
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        features['successor_score'] = self.get_score(successor)
        return features



    def get_weights(self, game_state, action):
        """
        Normally, weights do not depend on the game state.  They can be either
        a counter or a dictionary.
        """
        return {'successor_score': 1.0}

    # ── Game state helpers ──

    def _time_left(self, game_state):
        return game_state.data.timeleft

    def _in_endgame(self, game_state):
        return self._time_left(game_state) <= self.ENDGAME_STEPS

    def _we_are_winning(self, game_state):
        return self.get_score(game_state) > 0

    def _we_are_winning_comfortably(self, game_state):
        return self.get_score(game_state) >= self.COMFORTABLE_LEAD

    def _in_late_game(self, game_state):
        return self._time_left(game_state) <= self.PHASE_TRIGGER

    # ── Teammate coordination helpers ──

    def _get_teammate_index(self, game_state):
        if self.red:
            team = game_state.get_red_team_indices()
        else:
            team = game_state.get_blue_team_indices()
        for i in team:
            if i != self.index:
                return i
        return None

    def _dist_to_teammate(self, game_state, my_pos):
        """what's the distance between our agents?"""
        teammate = self._get_teammate_index(game_state)
        if teammate is None or my_pos is None:
            return None
        teammate_pos = game_state.get_agent_position(teammate)
        if teammate_pos is None:
            return None
        return self.get_maze_distance(my_pos, teammate_pos)

    def _open_neighbors_count(self, game_state, pos):
        """How many walkable tiles are adjacent to pos?"""
        if pos is None:
            return 0
        x, y = int(pos[0]), int(pos[1])
        walls = game_state.get_walls()
        count = 0
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < walls.width and 0 <= ny < walls.height and not walls[nx][ny]:
                count += 1
        return count



    def _is_tight_space(self, game_state, pos):
        """Dead-end or corridor (<=2 exits)."""
        return self._open_neighbors_count(game_state, pos) <= 2



    def _teammate_spacing_penalty(self, game_state, my_pos):
        """Penalize being too close to teammate (skip in tight spaces where queuing is needed)."""
        teammate = self._get_teammate_index(game_state)
        if teammate is None or my_pos is None:
            return 0
        teammate_pos = game_state.get_agent_position(teammate)
        if teammate_pos is None:
            return 0
        if self._is_tight_space(game_state, my_pos):
            return 0
        if self._is_tight_space(game_state, teammate_pos):
            return 0
        teammate_dist = self.get_maze_distance(my_pos, teammate_pos)
        return max(0, self.DESIRED_DISTANCE - teammate_dist)



class OffensiveReflexAgent(ReflexCaptureAgent):
    """
  A reflex agent that seeks food. This is an agent
  we give you to get an idea of what an offensive agent might look like,
  but it is by no means the best or only way to build an offensive agent.
  """

    def register_initial_state(self, game_state):
        super().register_initial_state(game_state)

        walls = game_state.get_walls()
        width, height = walls.width, walls.height
        area = width * height

        # Dynamic carry threshold based on map size.
        self.base_carry_threshold = max(3, min(9, int(area / 120)))

        # Home boundary — the column of tiles right on our side of the border.
        if self.red:
            x_red = (width // 2) - 1
        else: x_red = width // 2

        self.red_boundaries= []
        for y in range(height):
            if not walls[x_red][y]:
                self.red_boundaries.append((x_red,y))

        self.mid_y = height // 2

        # Best 3 entry points to cross into enemy territory (closest to vertical center).
        self.entry_points = sorted(
            self.red_boundaries,
            key=lambda p: abs(p[1] - self.mid_y)
        )[:3]
        if not self.entry_points:
            self.entry_points = list(self.red_boundaries)

        self.recent_positions = []

        # Phase 1 patrol spots on the boundary.
        self.border_patrol_points = sorted(
            self.red_boundaries,
            key=lambda p: abs(p[1] - self.mid_y)
        )[:4]
        if not self.border_patrol_points:
            self.border_patrol_points = list(self.red_boundaries)

        # Stolen-food tracking for Phase 1 sentinel reactions.
        self.prev_defended_food_off = self.get_food_you_are_defending(game_state).as_list()
        self.last_stolen_pos_off = None
        self.last_stolen_step_off = -float('inf')
        self.step_count_off = 0



    def _carry_threshold(self, game_state):
        """How many dots to carry before heading home (winning = play safe, losing = carry more)."""
        carry = self.base_carry_threshold
        if self._we_are_winning(game_state):
            carry -= 2
        else:
            carry += 2
        return max(2, min(12, carry))


    def _recent_stolen_food_active_off(self):
        """True if food was stolen on our side in the last 5 steps (Phase 1 sentinel react)."""
        return (self.last_stolen_pos_off is not None and
                (self.step_count_off - self.last_stolen_step_off) <= 5)



    def choose_action(self, game_state):
        self._update_beliefs(game_state)

        # Track stolen food so the Phase 1 sentinel can react.
        current_food = self.get_food_you_are_defending(game_state).as_list()
        missing = set(self.prev_defended_food_off) - set(current_food)
        if missing:
            my_pos = game_state.get_agent_state(self.index).get_position()
            if my_pos is not None:
                self.last_stolen_pos_off = min(missing, key=lambda m: self.get_maze_distance(my_pos, m))
            else:
                self.last_stolen_pos_off = next(iter(missing))
            self.last_stolen_step_off = self.step_count_off
        self.prev_defended_food_off = current_food
        self.step_count_off += 1

        actions = game_state.get_legal_actions(self.index)
        values = [self.evaluate(game_state, a) for a in actions]

        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]

        food_left = len(self.get_food(game_state).as_list())
        current_state = game_state.get_agent_state(self.index)

        # Almost no enemy food left — just go home.
        if food_left <= 2:
            best_dist = 9999
            best_action = None
            for action in actions:
                successor = self.get_successor(game_state, action)
                pos2 = successor.get_agent_position(self.index)
                dist = self.get_maze_distance(self.start, pos2)
                if dist < best_dist:
                    best_action = action
                    best_dist = dist
            chosen_action = best_action
        else:
            comfortable_endgame = self._in_endgame(game_state) and self._we_are_winning_comfortably(game_state)

            # In Phase 2, prefer actions that cross into enemy territory.
            in_late = self._in_late_game(game_state)
            if (not current_state.is_pacman) and (not comfortable_endgame) and in_late:
                crossing_actions = []
                for action in best_actions:
                    successor = self.get_successor(game_state, action)
                    if successor.get_agent_state(self.index).is_pacman:
                        crossing_actions.append(action)
                if crossing_actions:
                    best_actions = crossing_actions

            chosen_action = random.choice(best_actions)

        # Track recent positions for loop detection.
        successor = self.get_successor(game_state, chosen_action)
        chosen_pos = successor.get_agent_state(self.index).get_position()
        if chosen_pos is not None:
            self.recent_positions.append(chosen_pos)
            self.recent_positions = self.recent_positions[-6:]

        return chosen_action



    def _min_dist_enemy_ghost(self, successor):
        """Distance to the nearest visible, non-scared enemy ghost (None if we're a ghost or none visible)."""
        my_state = successor.get_agent_state(self.index)
        if not my_state.is_pacman:
            return None
        my_pos = my_state.get_position()
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]

        ghost_positions = []
        for e in enemies:
            pos = e.get_position()
            if pos is None or e.is_pacman or e.scared_timer > 0:
                continue
            ghost_positions.append(pos)

        if not ghost_positions:
            return None
        return min(self.get_maze_distance(my_pos, g) for g in ghost_positions)



    def _min_capsule_distance_(self, game_state, pos):
        """Distance to the nearest enemy capsule (None if no capsules left)."""
        capsules = self.get_capsules(game_state)
        if not capsules:
            return None
        return min(self.get_maze_distance(pos, c) for c in capsules)




    def _should_return_home(self, game_state, successor):
        """Decides if the attacker should head home (carrying enough, in danger, endgame, etc.)."""
        current_state = game_state.get_agent_state(self.index)
        current_pos = current_state.get_position()
        carrying = game_state.get_agent_state(self.index).num_carrying

        # Already on our side — no need to "return".
        if not current_state.is_pacman:
            return False

        danger_dist = 5
        min_ghost_distance = self._min_dist_enemy_ghost(successor)
        danger = (min_ghost_distance is not None and min_ghost_distance <= danger_dist)

        # Belief-based retreat causes too many false positives — only use visible ghosts here.

        # If a capsule is nearby, we can power through the danger.
        if danger:
            my_pos = successor.get_agent_state(self.index).get_position()
            cap_dist = self._min_capsule_distance_(game_state, my_pos)
            if cap_dist is not None and cap_dist <= 5:
                danger = False

        threshold = self._carry_threshold(game_state)

        # Winning big in endgame — get home and defend.
        if self._in_endgame(game_state) and self._we_are_winning_comfortably(game_state):
            return True

        # Close to boundary with a few dots — just bank them.
        dist_to_home = min(self.get_maze_distance(current_pos, b) for b in self.red_boundaries)
        if carrying >= 2 and dist_to_home <= 2:
            return True

        # Almost no time left and we're not winning — bank what we have.
        if self._time_left(game_state) <= 100:
            if (not self._we_are_winning(game_state)) and carrying > 0:
                return True

        # Endgame: winning = go home, carrying = go home, otherwise flee if in danger.
        if self._in_endgame(game_state):
            if self._we_are_winning(game_state):
                return True
            if carrying > 0:
                return True
            return danger

        # Hit the carry threshold — time to bank.
        if carrying >= threshold:
            return True

        return danger




    def get_features(self, game_state, action):
        features = util.Counter()

        # ── PHASE 1: Border sentinel (defend until PHASE_TRIGGER) ──
        if not self._in_late_game(game_state):
            successor = self.get_successor(game_state, action)
            my_state = successor.get_agent_state(self.index)
            my_pos = my_state.get_position()
            scared = (my_state.scared_timer > 0)

            features['on_defense'] = 1
            if my_state.is_pacman:
                features['on_defense'] = 0

            enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
            invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
            features['num_invaders'] = len(invaders)

            if len(invaders) > 0 and not scared:
                dists = [self.get_maze_distance(my_pos, a.get_position()) for a in invaders]
                features['invader_distance'] = min(dists)
            elif len(invaders) > 0 and scared:
                # Scared — back away so the invader can't eat us.
                dists = [self.get_maze_distance(my_pos, a.get_position()) for a in invaders]
                features['invader_distance'] = -min(dists)
            else:
                # Chase invisible invaders using belief estimates.
                if not scared:
                    phantom_dist = None
                    for opp in self.get_opponents(game_state):
                        opp_state = game_state.get_agent_state(opp)
                        if opp_state.get_position() is not None:
                            continue
                        if not opp_state.is_pacman:
                            continue
                        beliefs = self.enemy_beliefs.get(opp, [])
                        if 0 < len(beliefs) <= 15:
                            likely_pos = self._get_likely_enemy_position(game_state, opp)
                            if likely_pos is not None:
                                d = self.get_maze_distance(my_pos, likely_pos)
                                if phantom_dist is None or d < phantom_dist:
                                    phantom_dist = d
                    if phantom_dist is not None:
                        features['phantom_invader_distance'] = phantom_dist

                # No invaders at all — patrol the boundary or chase stolen food.
                if 'phantom_invader_distance' not in features and not scared and self._recent_stolen_food_active_off():
                    features['stolen_food_distance'] = self.get_maze_distance(
                        my_pos, self.last_stolen_pos_off)
                elif 'phantom_invader_distance' not in features:
                    # Bias patrol toward predicted enemy entry if beliefs are concentrated.
                    predicted_entry = self._get_likely_invader_entry(game_state)
                    if predicted_entry is not None:
                        best_patrol_dist = float('inf')
                        for p in self.border_patrol_points:
                            d = self.get_maze_distance(my_pos, p) + 2 * self.get_maze_distance(p, predicted_entry)
                            if d < best_patrol_dist:
                                best_patrol_dist = d
                        features['distance_to_patrol'] = best_patrol_dist
                    else:
                        features['distance_to_patrol'] = min(
                            self.get_maze_distance(my_pos, p) for p in self.border_patrol_points)

            # Spread out from teammate when patrolling calmly.
            if len(invaders) == 0 and not scared and not self._recent_stolen_food_active_off():
                spacing = self._teammate_spacing_penalty(successor, my_pos)
                if spacing > 0:
                    features['team_spacing_penalty'] = spacing

            if action == Directions.STOP:
                features['stop'] = 1
            rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
            if action == rev:
                features['reverse'] = 1
            return features

        # ── PHASE 2: Attacker (cross border, grab food, return) ──
        successor = self.get_successor(game_state, action)
        food_list = self.get_food(successor).as_list()

        prev_state = game_state.get_agent_state(self.index)
        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()
        comfortable_endgame = self._in_endgame(game_state) and self._we_are_winning_comfortably(game_state)

        returning = self._should_return_home(game_state, successor)

        # ── Comfortable endgame: go home and switch to defender ──
        if comfortable_endgame:
            if my_state.is_pacman:
                features['distance_to_home'] = min(self.get_maze_distance(my_pos, b) for b in self.red_boundaries)
                if prev_state.is_pacman and (not my_state.is_pacman) and prev_state.num_carrying > 0:
                    features['bank_now'] = 1
            else:
                # Already home — act as a second defender.
                features['on_defense'] = 1
                enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
                invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
                features['num_invaders'] = len(invaders)
                if len(invaders) > 0:
                    dists = [self.get_maze_distance(my_pos, a.get_position()) for a in invaders]
                    features['invader_distance'] = min(dists)
                else:
                    features['distance_to_hold'] = min(self.get_maze_distance(my_pos, p) for p in self.entry_points)

            if my_pos in self.recent_positions:
                features['loop_penalty'] = 1

            # Spread out from teammate once home.
            if not my_state.is_pacman:
                spacing_penalty = self._teammate_spacing_penalty(successor, my_pos)
                if spacing_penalty > 0:
                    features['team_spacing_penalty'] = spacing_penalty

            if action == Directions.STOP:
                features['stop'] = 1
            rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
            if action == rev:
                features['reverse'] = 1
            return features

        # ── Normal attack mode ──
        features['successor_score'] = -len(food_list)

        d = self._min_dist_enemy_ghost(successor)

        # Ghost distance feature (capped at 10).
        if d is None:
            features['min_enemy_ghost_distance'] = 10
        else:
            features['min_enemy_ghost_distance'] = min(d, 10)

        # Punish stepping right next to a ghost.
        if d is not None and d <= 1:
            features['danger'] = 1
            if my_state.is_pacman and d is not None and d <= 4:
                if self._is_tight_space(successor, my_pos):
                    features['risk_trap']= 1

        # If a ghost is close and a capsule is reachable, go for the capsule.
        if d is not None and d <= 5:
            cap_dist = self._min_capsule_distance_(game_state, my_pos)
            if cap_dist is not None and cap_dist <= 6:
                features['distance_to_capsule'] = cap_dist

        # Returning home — minimize distance to boundary.
        if returning:
            features['distance_to_home'] = min(self.get_maze_distance(my_pos, b) for b in self.red_boundaries)
            # Big bonus for the step that actually banks our carried food.
            if prev_state.is_pacman and (not my_state.is_pacman) and prev_state.num_carrying >0:
                features['bank_now'] = 1

        # On home side and not returning — head toward an entry point.
        if (not returning) and (not my_state.is_pacman) and self.entry_points:
            features['distance_to_entry'] = min(self.get_maze_distance(my_pos, p) for p in self.entry_points)

        # Reward the moment we cross into enemy territory.
        if (not prev_state.is_pacman) and my_state.is_pacman and (not returning):
            features['cross_border'] = 1

        # Hunt scared ghosts when we're on offense.
        if my_state.is_pacman:
            enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
            scared_ghost_positions = []
            for e in enemies:
               if (not e.is_pacman) and e.get_position() is not None and e.scared_timer > 0:
                   scared_ghost_positions.append(e.get_position())
            if scared_ghost_positions:
                enemies_dist = min(self.get_maze_distance(my_pos, g) for g in scared_ghost_positions)
                features['hunt_scared_ghost'] = 10 - min(enemies_dist, 10)

        # Food cluster targeting: prefer food near other food dots so we collect more per trip.
        if not returning and len(food_list) > 0:
            best_food_score = float('-inf')
            target_food = None
            for food in food_list:
                cluster_value = sum(1 for f2 in food_list
                                    if manhattan_distance(food, f2) <= 4 and f2 != food)
                maze_dist = self.get_maze_distance(my_pos, food)
                score = 3 * cluster_value - maze_dist
                if score > best_food_score:
                    best_food_score = score
                    target_food = food
            features['distance_to_food'] = self.get_maze_distance(my_pos, target_food)

        # Penalize revisiting recent positions (anti-loop).
        if my_pos in self.recent_positions:
            features['loop_penalty'] = 1

        # Spread out from teammate (skip when escaping or in immediate danger).
        apply_spacing = (not returning) and (d is None or d > 2)
        if apply_spacing:
            spacing_penalty = self._teammate_spacing_penalty(successor, my_pos)
            if spacing_penalty > 0:
                features['team_spacing_penalty'] = spacing_penalty

        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev:
            features['reverse'] = 1

        return features



    def get_weights(self, game_state, action):
        # Phase 1: border sentinel weights.
        if not self._in_late_game(game_state):
            successor = self.get_successor(game_state, action)
            my_state = successor.get_agent_state(self.index)
            if my_state.scared_timer > 0:
                return {'on_defense': 100, 'num_invaders': -1000, 'invader_distance': 10,
                        'phantom_invader_distance': 0,
                        'stolen_food_distance': 0, 'distance_to_patrol': -8,
                        'team_spacing_penalty': 0, 'stop': -100, 'reverse': -2}
            return {'on_defense': 100, 'num_invaders': -1000, 'invader_distance': -15,
                    'phantom_invader_distance': -12,
                    'stolen_food_distance': -13, 'distance_to_patrol': -8,
                    'team_spacing_penalty': -10, 'stop': -100, 'reverse': -2}

        # Phase 2: attack weights.
        successor = self.get_successor(game_state, action)
        returning = self._should_return_home(game_state, successor)
        comfortable_endgame = self._in_endgame(game_state) and self._we_are_winning_comfortably(game_state)

        if comfortable_endgame:
            return {'successor_score': 0,
                    'distance_to_food': 0,
                    'distance_to_home': -300,
                    'distance_to_entry': 0,
                    'distance_to_capsule': 0,
                    'min_enemy_ghost_distance': 0,
                    'hunt_scared_ghost': 0,
                    'danger': -400,
                    'stop': -100,
                    'reverse': -20,
                    'loop_penalty': -100,
                    'team_spacing_penalty': -10,
                    'cross_border': 0,
                    'bank_now': 1000,
                    'on_defense': 100,
                    'num_invaders': -1200,
                    'invader_distance': -20,
                    'distance_to_hold': -10,
                    'risk_trap': -300}

        if returning:
            return {'successor_score': 0,
                    'distance_to_food': 0,
                    'distance_to_home': -300,
                    'distance_to_entry': 0,
                    'distance_to_capsule': 0,
                    'min_enemy_ghost_distance': 1,
                    'hunt_scared_ghost': 0,
                    'danger': -400,
                    'stop': -100,
                    'reverse': -20,
                    'loop_penalty': -100,
                    'team_spacing_penalty': 0,
                    'cross_border': 0 ,
                    'bank_now': 1000,
                    'risk_trap': -300}

        # If ghost is close and capsule is reachable, prioritize the capsule.
        d = self._min_dist_enemy_ghost(successor)
        capsule_mode = False
        if d is not None and d <= 5:
            my_pos = successor.get_agent_state(self.index).get_position()
            cap_dist = self._min_capsule_distance_(game_state, my_pos)
            if cap_dist is not None and cap_dist <= 6:
                capsule_mode = True

        if capsule_mode:
            return {'successor_score': 100,
                    'distance_to_food': -10,
                    'distance_to_home': 0,
                    'distance_to_entry': -4,
                    'distance_to_capsule': -8,
                    'min_enemy_ghost_distance': 1,
                    'hunt_scared_ghost': 1,
                    'danger': -20,
                    'stop': -100,
                    'reverse': -4,
                    'loop_penalty': -20,
                    'team_spacing_penalty': 0,
                    'cross_border': 25,
                    'risk_trap': -150 }

        # Default attack weights.
        return {'successor_score': 100,
                    'distance_to_food': -5,
                    'distance_to_home': 0,
                    'distance_to_entry': -15,
                    'distance_to_capsule': 0,
                    'min_enemy_ghost_distance': 1,
                    'hunt_scared_ghost': 1,
                    'danger': -500,
                    'stop': -100,
                    'reverse': -4,
                    'loop_penalty': -20,
                    'team_spacing_penalty': -12,
                    'cross_border': 40,
                    'risk_trap': -300 }




class DefensiveReflexAgent(ReflexCaptureAgent):
    """
    A reflex agent that keeps its side Pacman-free. Again,
    this is to give you an idea of what a defensive agent
    could be like.  It is not the best or only way to make
    such an agent.
    """

    def register_initial_state(self, game_state):
        super().register_initial_state(game_state)

        self.prev_defended_food = self.get_food_you_are_defending(game_state).as_list()
        self.last_stolen_pos = None
        self.last_stolen_step = -float('inf')
        self.step_count = 0

        walls = game_state.get_walls()
        width, height = walls.width, walls.height
        self.mid_y = height // 2

        # Home boundary column.
        if self.red:
            boundary_x = (width //2) - 1
        else:
            boundary_x = width // 2
        self.home_boundary = []
        for y in range(height):
            if not walls[boundary_x][y]:
                self.home_boundary.append((boundary_x, y))

        # Precompute patrol points scored by: center bias + food proximity + capsule proximity.
        defended_food = self.get_food_you_are_defending(game_state).as_list()
        defended_capsules = self.get_capsules_you_are_defending(game_state)

        scored_boundary = []
        for b in self.home_boundary:
            score = abs(b[1] - self.mid_y)
            if defended_food:
                score += 2 * min(self.get_maze_distance(b, food) for food in defended_food)
            if defended_capsules:
                score += 3 * min(self.get_maze_distance(b, cap) for cap in defended_capsules)
            scored_boundary.append((score, b))
        scored_boundary.sort()

        self.patrol_points = [pos for _, pos in scored_boundary[:3]]
        if not self.patrol_points:
            self.patrol_points = list(self.home_boundary)

        # Phase 1 camp points: 3-5 tiles behind the border, near food/capsules.
        if self.red:
            camp_x_range = range(max(1, boundary_x - 5), boundary_x)
        else:
            camp_x_range = range(boundary_x + 1, min(width - 1, boundary_x + 6))

        camp_candidates = []
        for x in camp_x_range:
            for y in range(height):
                if not walls[x][y]:
                    camp_candidates.append((abs(y - self.mid_y), (x, y)))
        camp_candidates.sort()

        scored_camp = []
        for _, pos in camp_candidates[:15]:
            score = abs(pos[1] - self.mid_y)
            if defended_food:
                score += 2 * min(self.get_maze_distance(pos, f) for f in defended_food)
            if defended_capsules:
                score += 3 * min(self.get_maze_distance(pos, c) for c in defended_capsules)
            scored_camp.append((score, pos))
        scored_camp.sort()
        self.camp_patrol_points = [p for _, p in scored_camp[:3]]
        if not self.camp_patrol_points:
            self.camp_patrol_points = [self.start]




    def choose_action(self, game_state):
        self._update_beliefs(game_state)

        # Detect stolen food and remember where it happened.
        current_food = self.get_food_you_are_defending(game_state).as_list()
        missing = set(self.prev_defended_food) - set(current_food)
        if missing:
            my_pos = game_state.get_agent_state(self.index).get_position()
            if my_pos is not None:
                self.last_stolen_pos = min(missing, key=lambda m: self.get_maze_distance(my_pos, m))
            else:
                self.last_stolen_pos = next(iter(missing))
            self.last_stolen_step = self.step_count
        self.prev_defended_food = current_food
        self.step_count += 1

        return super().choose_action(game_state)



    def _recent_stolen_food_active(self):
        """True if food was stolen on our side in the last 5 steps."""
        return (
            self.last_stolen_pos is not None and
            (self.step_count - self.last_stolen_step) <= 5
        )



    def _get_camp_patrol_target(self, game_state):
        """Phase 1: pick the best interior camp point (biased toward food, capsules, predicted entry)."""
        defended_food = self.get_food_you_are_defending(game_state).as_list()
        defended_capsules = self.get_capsules_you_are_defending(game_state)
        predicted_entry = self._get_likely_invader_entry(game_state)

        best_target = None
        best_score = float('inf')
        for p in self.camp_patrol_points:
            score = abs(p[1] - self.mid_y)
            if defended_food:
                score += 2 * min(self.get_maze_distance(p, f) for f in defended_food)
            if defended_capsules:
                score += 3 * min(self.get_maze_distance(p, c) for c in defended_capsules)
            if self._recent_stolen_food_active() and self.last_stolen_pos is not None:
                score += 2 * self.get_maze_distance(p, self.last_stolen_pos)
            if predicted_entry is not None:
                score += 2 * self.get_maze_distance(p, predicted_entry)
            if score < best_score:
                best_score = score
                best_target = p
        return best_target if best_target else self.start



    def _get_patrol_target(self, game_state):
        """Phase 2: pick the best boundary patrol point (biased toward food, capsules, predicted entry)."""
        defended_food = self.get_food_you_are_defending(game_state).as_list()
        defended_capsules = self.get_capsules_you_are_defending(game_state)
        predicted_entry = self._get_likely_invader_entry(game_state)

        best_target = None
        best_score = float('inf')
        for p in self.patrol_points:
            score = abs(p[1] - self.mid_y)
            if defended_food:
                score += 2 * min(self.get_maze_distance(p, food) for food in defended_food)
            if defended_capsules:
                score += 3 * min(self.get_maze_distance(p, cap) for cap in defended_capsules)
            if self._recent_stolen_food_active() and self.last_stolen_pos is not None:
                score += 2 * self.get_maze_distance(p, self.last_stolen_pos)
            if predicted_entry is not None:
                score += 3 * self.get_maze_distance(p, predicted_entry)
            if score < best_score:
               best_score = score
               best_target = p

        return best_target if best_target else self.start



    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)

        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()
        scared = (my_state.scared_timer > 0)

        endgame = self._in_endgame(game_state)
        winning = self._we_are_winning(game_state)

        # Endgame and losing — switch to full attack mode.
        if endgame and (not winning):
            food_list = self.get_food(successor).as_list()
            features['successor_score'] = -len(food_list)
            if len(food_list) > 0:
                features['distance_to_food'] = min(
                    self.get_maze_distance(my_pos, food) for food in food_list)

        features['on_defense'] = 1
        if my_state.is_pacman: features['on_defense'] = 0

        # Chase visible invaders.
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        features['num_invaders'] = len(invaders)
        if len(invaders) > 0:
            dists = [self.get_maze_distance(my_pos, a.get_position()) for a in invaders]
            features['invader_distance'] = min(dists)

        # Chase invisible invaders using belief estimates.
        elif not scared:
            phantom_dist = None
            for opp in self.get_opponents(game_state):
                opp_state = game_state.get_agent_state(opp)
                if opp_state.get_position() is not None:
                    continue
                if not opp_state.is_pacman:
                    continue
                beliefs = self.enemy_beliefs.get(opp, [])
                if 0 < len(beliefs) <= 15:
                    likely_pos = self._get_likely_enemy_position(game_state, opp)
                    if likely_pos is not None:
                        d = self.get_maze_distance(my_pos, likely_pos)
                        if phantom_dist is None or d < phantom_dist:
                            phantom_dist = d
            if phantom_dist is not None:
                features['phantom_invader_distance'] = phantom_dist

        # No invaders — patrol or chase stolen food.
        if len(invaders) == 0 and 'phantom_invader_distance' not in features:
            if (not scared) and self._recent_stolen_food_active():
                features['stolen_food_distance'] = self.get_maze_distance(my_pos, self.last_stolen_pos)
            else:
                # Phase 1 = camp interior, Phase 2 = boundary patrol.
                if not self._in_late_game(game_state):
                    patrol_target = self._get_camp_patrol_target(game_state)
                else:
                    patrol_target = self._get_patrol_target(successor)
                features['distance_to_patrol'] = self.get_maze_distance(my_pos, patrol_target)

        # Spread out from teammate during calm patrol.
        if len(invaders) == 0 and (not scared) and (not self._recent_stolen_food_active()):
            spacing_penalty = self._teammate_spacing_penalty(successor, my_pos)
            if spacing_penalty > 0:
                features['team_spacing_penalty'] = spacing_penalty

        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1

        return features



    def get_weights(self, game_state, action):
        successor = self.get_successor(game_state, action)
        my_state = successor.get_agent_state(self.index)

        endgame = self._in_endgame(game_state)
        winning = self._we_are_winning(game_state)

        # Endgame losing — full attack weights.
        if endgame and (not winning):
            return  {'num_invaders': 0,
                     'on_defense': -50,
                     'invader_distance': 0,
                     'phantom_invader_distance': 0,
                     'successor_score': 100,
                     'distance_to_food': -3,
                     'stop': -100, 'reverse': -2,
                     'stolen_food_distance': 0,
                     'distance_to_patrol': 0,
                     'team_spacing_penalty': 0}

        # Scared — avoid invaders instead of chasing.
        if my_state.scared_timer > 0:
            return {'num_invaders': -1000,
                    'on_defense': 100,
                    'invader_distance': 10,
                    'phantom_invader_distance': 0,
                    'stop': -100,
                    'reverse': -2,
                    'stolen_food_distance': 0,
                    'distance_to_patrol': -8,
                    'team_spacing_penalty': 0}

        # Default — chase invaders aggressively.
        return {'num_invaders': -1000,
                'on_defense': 100,
                'invader_distance': -15,
                'phantom_invader_distance': -12,
                'stop': -100,
                'reverse': -2,
                'stolen_food_distance': -13,
                'distance_to_patrol': -6,
                'team_spacing_penalty': -10}




"""
IDEAS TO IMPLEMENT

1) Return home when carrying enough dots or in danger -> IMPLEMENTED
2) Punish moves toward visible unscared ghosts -> Half implemented
3) Chase enemies that stole our food -> IMPLEMENTED
3b) Don't chase if enemy ate a capsule (we're scared) -> IMPLEMENTED
4) Endgame: winning = defend, losing = full attack -> IMPLEMENTED
5) STOP + reverse penalties on offense -> IMPLEMENTED
6) Dynamic carry threshold based on layout size and score -> IMPLEMENTED
7) Ghost nearby + capsule reachable = go for capsule -> IMPLEMENTED
8) Avoid dead-ends when unscared ghost is near -> IMPLEMENTED
9) Active front-line defense, not spawn camping -> IMPLEMENTED
10) Return to boundary instead of spawn -> IMPLEMENTED
11) Target food clusters over isolated dots -> IMPLEMENTED
12) Camp middle boundary on defense -> IMPLEMENTED
13) Team spacing — agents spread out to cover more ground -> IMPLEMENTED
14) Two-phase strategy (PHASE_TRIGGER=600) -> IMPLEMENTED
    Phase 1 (>600): both agents defend (sentinel + camp)
    Phase 2 (<=600): offensive attacks, defensive patrols boundary
15) Belief tracking for invisible enemy positions -> IMPLEMENTED
"""
