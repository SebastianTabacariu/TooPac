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
from util import nearest_point


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
    # New: Last 'N' moves which we can later change to our preference. The use of this will later be explained in the code.
    ENDGAME_STEPS = 400
    # New: points needed to score to be comfortable.
    COMFORTABLE_LEAD = 8
    # desired sitance between the two teammates.
    DESIRED_DISTANCE = 4
    # Idea 14: timeleft threshold that separates Phase 1 (defense) from Phase 2 (attack).
    PHASE_TRIGGER = 300

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.start = None

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)
     

    def choose_action(self, game_state):
        """
        Picks among the actions with the highest Q(s,a).
        """
        actions = game_state.get_legal_actions(self.index)

        # You can profile your evaluation time by uncommenting these lines
        # start = time.time()
        values = [self.evaluate(game_state, a) for a in actions]
        # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

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
            # Only half a grid position was covered
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
    
    #New: Implementation in which the remaining moves are stored, which give us the last N moves, and an implementation which states if were currently winning or not.   
    # These are used as part of our nr.4 implementation.

    def _time_left(self, game_state):
        return game_state.data.timeleft 
    
    def _in_endgame(self, game_state):
        return self._time_left(game_state) <= self.ENDGAME_STEPS
    
    def _we_are_winning(self, game_state):
        return self.get_score(game_state) > 0
    
    def _we_are_winning_comfortably(self, game_state):
        return self.get_score(game_state) >= self.COMFORTABLE_LEAD

    # Idea 14: True once the phase transition fires (timeleft <= PHASE_TRIGGER).
    def _in_late_game(self, game_state):
        return self._time_left(game_state) <= self.PHASE_TRIGGER

    #New: idea 13

    def _get_teammate_index(self, game_state):   # funtion to find the idex of teammate
        if self.red:
            team = game_state.get_red_team_indices()
        else: 
            team = game_state.get_blue_team_indices()
        
        for i in team: 
            if i != self.index:
                return i
        return None  # fallback just in case
    
    def _dist_to_teammate(self, game_state, my_pos):  #function to find the maze distance between the agent and its teammate
        teammate = self._get_teammate_index(game_state)
        if teammate is None or my_pos is None: 
            return None
        
        teammate_pos = game_state.get_agent_position(teammate)
        if teammate_pos is None:
            return None
        
        return self.get_maze_distance(my_pos, teammate_pos)
    
        # 
    def _open_neighbors_count(self, game_state, pos):
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
    # corridor (2 exits), dead-end (1), corner pocket, ...
        return self._open_neighbors_count(game_state, pos) <= 2

      
    def _teammate_spacing_penalty(self, game_state, my_pos):
        teammate = self._get_teammate_index(game_state)
        if teammate is None or my_pos is None:
            return 0

        teammate_pos = game_state.get_agent_position(teammate)
        if teammate_pos is None:
            return 0

    # do not force separation.
    # Agents sometimes need to queue behind each other to escape.
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

        #NEW
        walls = game_state.get_walls()
        width, height = walls.width, walls.height
        # map size
        area = width * height

        #threshold to carry from layout
        self.base_carry_threshold = max(3, min(9, int(area / 120)))

        if self.red:
            x_red = (width // 2) - 1
        else: x_red = width // 2

        self.red_boundaries= []

        for y in range(height):
            if not walls[x_red][y]:
                self.red_boundaries.append((x_red,y))


        self.mid_y = height // 2
        self.entry_points = sorted(
            self.red_boundaries,
            key=lambda p: abs(p[1] - self.mid_y)
        )[:3]

        if not self.entry_points:
            self.entry_points = list(self.red_boundaries)

        self.recent_positions = []

        # Idea 14 : patrol points on the boundary, biased toward center.
        self.border_patrol_points = sorted(
            self.red_boundaries,
            key=lambda p: abs(p[1] - self.mid_y)
        )[:4]
        if not self.border_patrol_points:
            self.border_patrol_points = list(self.red_boundaries)

        # Track stolen food for Phase 1 sentinel reactions.
        self.prev_defended_food_off = self.get_food_you_are_defending(game_state).as_list()
        self.last_stolen_pos_off = None
        self.last_stolen_step_off = -float('inf')
        self.step_count_off = 0



    #NEW:  dynamic carry threshold
    def _carry_threshold(self, game_state):

        carry = self.base_carry_threshold

        if self._we_are_winning(game_state):
            #play safe
            carry -= 2

        else:  
            #risky, we are losing
            carry += 2


        return max(2, min(12, carry))

    def _recent_stolen_food_active_off(self):
        # Idea 14: used in Phase 1 border-sentinel mode.
        return (self.last_stolen_pos_off is not None and
                (self.step_count_off - self.last_stolen_step_off) <= 5)

    def choose_action(self, game_state):
        # Idea 14: track stolen food so Phase 1 (border sentinel) can react.
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

            # Idea 14: only cross into enemy territory in Phase 2 .
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

        successor = self.get_successor(game_state, chosen_action)
        chosen_pos = successor.get_agent_state(self.index).get_position()
        if chosen_pos is not None:
            self.recent_positions.append(chosen_pos)
            self.recent_positions = self.recent_positions[-6:]

        return chosen_action





    def _min_dist_enemy_ghost(self, successor):

        my_state = successor.get_agent_state(self.index)

        if not my_state.is_pacman:
            return None
        
        my_pos = my_state.get_position()
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]

        ghost_positions = []
        for e in enemies:
            pos = e.get_position()
            if pos is None: # enemy not visible
                continue
            if e.is_pacman:
                continue
            if e.scared_timer > 0:
                continue
            ghost_positions.append(pos)

        if not ghost_positions:
            return None # no dangerous ghosts found
        
        return min(self.get_maze_distance(my_pos, danger_ghost) for danger_ghost in ghost_positions)
    
    #NEW idea 7: distance to the nearest capsule
    # we use a basic loop to find the minimal distance

    def _min_capsule_distance_(self, game_state, pos):
        capsules = self.get_capsules(game_state)
        if not capsules:
            return None
        best = None
        for c in capsules:
           d = self.get_maze_distance(pos, c) 
           if best is None or d < best:
               best = d
        return best
    


    
      #NEW
    def _should_return_home(self, game_state, successor):
        # return if carrying more than N dots or if it is dangerous
        # return if we are in the last N moves and are currently winning, or else continue attacking
        # NEW:  if we are winning, we should stay/go home and defend

        current_state = game_state.get_agent_state(self.index)
        current_pos = current_state.get_position()
        carrying = game_state.get_agent_state(self.index).num_carrying
        # If we are already on our own side, we should not enter "return home" mode.
        if not current_state.is_pacman:
            return False

        danger_dist = 5
        min_ghost_distance = self._min_dist_enemy_ghost(successor)
        
        danger = (min_ghost_distance is not None and min_ghost_distance <= danger_dist)

        #New: if danger is close but we are near a capsule, don't retreat

        if danger: 
            my_pos = successor.get_agent_state(self.index).get_position()
            cap_dist = self._min_capsule_distance_(game_state, my_pos)
            if cap_dist is not None and cap_dist <= 5: #can be adjusted if necessary
                danger = False 

        #NEW
        threshold = self._carry_threshold(game_state)

        if self._in_endgame(game_state) and self._we_are_winning_comfortably(game_state):
            return True

        # NEW:
        # carry a few dots and  VERY close to boundary -> secure the points
        dist_to_home = min(self.get_maze_distance(current_pos, b) for b in self.red_boundaries)
        if carrying >= 2 and dist_to_home <= 2:
            return True
        
        #return if there is almost no time left
        if self._time_left(game_state) <= 100:
            if (not self._we_are_winning(game_state)) and carrying > 0:
                return True

        if self._in_endgame(game_state):
            if self._we_are_winning(game_state):
                return True
            if carrying > 0:
                return True
            return danger

        
        if carrying >= threshold:
            return True 
        
        
        return danger
            
            


    def get_features(self, game_state, action):
        features = util.Counter()

        # Idea 14 – Phase 1: act as border sentinel.
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
                # Scared: back away so enemy cannot eat us.
                dists = [self.get_maze_distance(my_pos, a.get_position()) for a in invaders]
                features['invader_distance'] = -min(dists)
            else:
                # No visible invaders: patrol boundary or chase stolen food.
                if not scared and self._recent_stolen_food_active_off():
                    features['stolen_food_distance'] = self.get_maze_distance(
                        my_pos, self.last_stolen_pos_off)
                else:
                    features['distance_to_patrol'] = min(
                        self.get_maze_distance(my_pos, p) for p in self.border_patrol_points)

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

        # Phase 2: late game.
        successor = self.get_successor(game_state, action)
        food_list = self.get_food(successor).as_list()

        prev_state = game_state.get_agent_state(self.index)
        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()
        comfortable_endgame = self._in_endgame(game_state) and self._we_are_winning_comfortably(game_state)

        returning = self._should_return_home(game_state, successor)

        if comfortable_endgame:
    # If we are still in enemy territory, go home immediately.
            if my_state.is_pacman:
                features['distance_to_home'] = min(self.get_maze_distance(my_pos, b) for b in self.red_boundaries)
        

                if prev_state.is_pacman and (not my_state.is_pacman) and prev_state.num_carrying > 0:
                    features['bank_now'] = 1

    # Once home, become a second defender instead of crossing again.
            else:
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

            #New: idea 13, checks how far the offensive agent would be from his teammate after taking this action and give a penalty

            # Only spread out once we are already home and defending.
            # If still Pacman in endgame, getting home matters more than spacing.
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


        features['successor_score'] = -len(food_list)  # self.get_score(successor)

        d = self._min_dist_enemy_ghost(successor)

        #NEW: avoid DANGER
        if d is None:
            features['min_enemy_ghost_distance'] = 10
        else:
            features['min_enemy_ghost_distance'] = min(d, 10)

         # NEW: strongly punish stepping into immediate danger range
        if d is not None and d <= 1:
            features['danger'] = 1

            if my_state.is_pacman and d is not None and d <= 4:
                if self._is_tight_space(successor, my_pos):
                    features['risk_trap']= 1

        # New: if danger is close and capsule is nearby, move to capsule  
        if d is not None and d <= 5:
            cap_dist = self._min_capsule_distance_(game_state, my_pos)
            if cap_dist is not None and cap_dist <= 6:
                features['distance_to_capsule'] = cap_dist
        


        #NEW: if we return, keep the distance to home
        if returning:
            features['distance_to_home'] = min(self.get_maze_distance(my_pos, b) for b in self.red_boundaries) 

            # big bonus for move that brings points
            if prev_state.is_pacman and (not my_state.is_pacman) and prev_state.num_carrying >0:
                features['bank_now'] = 1


        #NEW: encourage crossing the border when we are stil on home side and we are not returning
        if (not returning) and (not my_state.is_pacman) and self.entry_points:
            features['distance_to_entry'] = min(self.get_maze_distance(my_pos, p) for p in self.entry_points)    

        #NEW: reward when crossing into enemy territory (to encourage this behaviour)

        if (not prev_state.is_pacman) and my_state.is_pacman and (not returning):
            features['cross_border'] = 1


        #NEW: hunt scared ghost when on offense
        if my_state.is_pacman:
            enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
            scared_ghost_positions = []
            for e in enemies:
               if (not e.is_pacman) and e.get_position() is not None and e.scared_timer > 0:
                   enemy_position = e.get_position() 
                   scared_ghost_positions.append(enemy_position)

            if scared_ghost_positions:
                enemies_dist = min(self.get_maze_distance(my_pos, ghost_pos) for ghost_pos in scared_ghost_positions)
                # the closer, the better
                features['hunt_scared_ghost'] = 10 - min(enemies_dist, 10)       

  
        if not returning and  len(food_list) > 0:  # his should always be True,  but better safe than sorry
                features['distance_to_food'] = min(self.get_maze_distance(my_pos, food) for food in food_list)

        # punish revisiting recent positions
        if my_pos in self.recent_positions:
            features['loop_penalty'] = 1

        #idea 13, same principle as above

        # Do not apply spacing while escaping / returning / in immediate danger.
        apply_spacing = (not returning) and (d is None or d > 2)

        if apply_spacing:
            spacing_penalty = self._teammate_spacing_penalty(successor, my_pos)
            if spacing_penalty > 0:
                features['team_spacing_penalty'] = spacing_penalty


       # NEW: STOP + reverse penalties
        if action == Directions.STOP: features['stop'] = 1


        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev: 
            features['reverse'] = 1
                

        return features

    def get_weights(self, game_state, action):
        # Idea 14 – Phase 1: border sentinel weights.
        if not self._in_late_game(game_state):
            successor = self.get_successor(game_state, action)
            my_state = successor.get_agent_state(self.index)
            if my_state.scared_timer > 0:
                return {'on_defense': 100, 'num_invaders': -1000, 'invader_distance': 10,
                        'stolen_food_distance': 0, 'distance_to_patrol': -8,
                        'team_spacing_penalty': 0, 'stop': -100, 'reverse': -2}
            return {'on_defense': 100, 'num_invaders': -1000, 'invader_distance': -15,
                    'stolen_food_distance': -13, 'distance_to_patrol': -8,
                    'team_spacing_penalty': -10, 'stop': -100, 'reverse': -2}

        # Phase 2: late game,  existing attack weights.
        successor = self.get_successor(game_state, action)
        # NEW
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
        
        #New: idea 7: if a dangerous ghost is close and a capsule is reachable, prefer the capsule. -> weights updated
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
        
        return {'successor_score': 100, 
                    'distance_to_food': -5, 
                    'distance_to_home': 0,
                    'distance_to_entry': -4,
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
  # NEW
    def register_initial_state(self, game_state):
        super().register_initial_state(game_state)

        self.prev_defended_food = self.get_food_you_are_defending(game_state).as_list()

        self.last_stolen_pos = None
        self.last_stolen_step = -float('inf')
       # local step counter
        self.step_count = 0

        walls = game_state.get_walls()
        width, height = walls.width, walls.height
        self.mid_y = height // 2

        #Home boundary
        if self.red:
            boundary_x = (width //2) - 1
        else:
            boundary_x = width // 2
        self.home_boundary = []
        for y in range(height):
            if not walls[boundary_x][y]:
                self.home_boundary.append((boundary_x, y))   

        # places that we precompute to defend when nothing is urgent/visible
        defended_food = self.get_food_you_are_defending(game_state).as_list()
        defended_capsules = self.get_capsules_you_are_defending(game_state)

        scored_boundary = []
        for b in self.home_boundary:
            score = 0

             #prefer center tiles
            score += abs(b[1] - self.mid_y)

        # prefer tilesclose to defended food
            if defended_food:
                food_dist = min(self.get_maze_distance(b, food) for food in defended_food)
                score += 2 * food_dist

        # Prefer boundary tiles that are close to defended capsules even more
            if defended_capsules:
                cap_dist = min(self.get_maze_distance(b, cap) for cap in defended_capsules)
                score += 3 * cap_dist

            scored_boundary.append((score, b))

        scored_boundary.sort()

    # Keep only a few best patrol points to reduce noise
        self.patrol_points = [pos for _, pos in scored_boundary[:3]]

    # Safety fallback
        if not self.patrol_points:
            self.patrol_points = list(self.home_boundary)

        # Idea 14 (Phase 1 – camp defender): precompute patrol points 3-5 tiles behind the border.
        if self.red:
            camp_x_range = range(max(1, boundary_x - 5), boundary_x)
        else:
            camp_x_range = range(boundary_x + 1, min(width - 1, boundary_x + 6))

        camp_candidates = []
        for x in camp_x_range:
            for y in range(height):
                if not walls[x][y]:
                    score = abs(y - self.mid_y)
                    camp_candidates.append((score, (x, y)))
        camp_candidates.sort()

        # re-score top candidates by proximity to defended food/capsules.
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



      #NEW
    def choose_action(self, game_state):
        current_food = self.get_food_you_are_defending(game_state).as_list()

        # Detect missing food
        missing = set(self.prev_defended_food) - set(current_food)

        if missing:
            # multiple eaten
            my_pos = game_state.get_agent_state(self.index).get_position()
            if my_pos is not None:
                best_pos = None
                min_dist = float('inf')

                for m in missing:
                    d = self.get_maze_distance(my_pos, m)
                    if d < min_dist:
                        min_dist = d
                        best_pos = m

                self.last_stolen_pos = best_pos
             # can't compute distance (NONE)
            else:   
                self.last_stolen_pos = next(iter(missing))

            self.last_stolen_step = self.step_count            

        self.prev_defended_food = current_food
        self.step_count += 1

        # not missing: normal behaviour          
        return super().choose_action(game_state)    
    

    def _recent_stolen_food_active(self):
        recent_stolen_step_count = 5
        return (
            self.last_stolen_pos is not None and
            (self.step_count - self.last_stolen_step) <= recent_stolen_step_count
    )


    ## Idea 14: pick best interior patrol target for Phase 1 (camp defender).
    def _get_camp_patrol_target(self, game_state):
        defended_food = self.get_food_you_are_defending(game_state).as_list()
        defended_capsules = self.get_capsules_you_are_defending(game_state)
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
            if score < best_score:
                best_score = score
                best_target = p

        return best_target if best_target else self.start

    def _get_patrol_target(self, game_state):

        defended_food = self.get_food_you_are_defending(game_state).as_list()
        defended_capsules = self.get_capsules_you_are_defending(game_state)

        best_target = None
        best_score = float('inf')

        for p in self.patrol_points:
            score = 0

            #bias towards middle lanes
            score += abs(p[1] - self.mid_y)

            # stay close to areas we want to defend
            if defended_food:
                score += 2 * min(self.get_maze_distance(p,food) for food in defended_food)

            if defended_capsules:
                score += 3 * min(self.get_maze_distance(p, cap) for cap in defended_capsules)

            # go to recently stolen stuff
            if self._recent_stolen_food_active() and self.last_stolen_pos is not None:
                score += 2 * self.get_maze_distance(p, self.last_stolen_pos)

            if score < best_score:
               best_score = score
               best_target = p


        if best_target is None:
            return self.start

        return best_target
                       


    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)

        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        # New: If enemy recently ate a capsule, ghost become scared. While scared, do NOT chase the enemy to capture them. Instead, keep distance. 
        scared = (my_state.scared_timer > 0)

        # New: Part of idea 4: If we are losing or tied near the end, go for full attack. Move toward enemy food.
        # if winning, we keep on defending. 

        endgame = self._in_endgame(game_state)
        winning = self._we_are_winning(game_state)

        if endgame and (not winning):
            food_list = self.get_food(successor).as_list()
            features['successor_score'] = -len(food_list)

            if len(food_list) > 0:
                min_distance = 9999
                for food in food_list:
                    dist = self.get_maze_distance(my_pos, food)
                    if dist < min_distance:
                        min_distance = dist
                features['distance_to_food'] = min_distance


        # Computes whether we're on defense (1) or offense (0)
        features['on_defense'] = 1
        if my_state.is_pacman: features['on_defense'] = 0

        # Computes distance to invaders we can see
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        features['num_invaders'] = len(invaders)
        if len(invaders) > 0:
            dists = [self.get_maze_distance(my_pos, a.get_position()) for a in invaders]
            features['invader_distance'] = min(dists)

        # New: when invisible invader
        else:
            if (not scared) and self._recent_stolen_food_active():
                features['stolen_food_distance'] = self.get_maze_distance(my_pos, self.last_stolen_pos)

            else:
                # Idea 14: Phase 1 = roam camp interior; Phase 2 = patrol boundary.
                if not self._in_late_game(game_state):
                    patrol_target = self._get_camp_patrol_target(game_state)
                else:
                    patrol_target = self._get_patrol_target(successor)
                features['distance_to_patrol'] = self.get_maze_distance(my_pos, patrol_target)

        #idea 13, same as with the offensive agent
        
       # Use spacing only during calm patroling.
        # If there is an urgent target, both defenders may come together.
        if len(invaders) == 0 and (not scared) and (not self._recent_stolen_food_active()):
            spacing_penalty = self._teammate_spacing_penalty(successor, my_pos)
            if spacing_penalty > 0:
                features['team_spacing_penalty'] = spacing_penalty

        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1


        return features
   #Updated
    def get_weights(self, game_state, action):
        
        successor = self.get_successor(game_state, action)
        my_state = successor.get_agent_state(self.index)

        endgame = self._in_endgame(game_state)
        winning = self._we_are_winning(game_state)

        if endgame and (not winning):
            return  {'num_invaders': 0, 
                     'on_defense': -50, 
                     'invader_distance': 0, 
                     'successor_score': 100,
                     'distance_to_food': -3, 
                     'stop': -100, 'reverse': -2, 
                     'stolen_food_distance': 0, 
                     'distance_to_patrol': 0,
                     'team_spacing_penalty': 0}

        #If we are scared, avoid invaders instead of chasing. 
        if my_state.scared_timer > 0:
            return {'num_invaders': -1000, 
                    'on_defense': 100, 
                    'invader_distance': 10, 
                    'stop': -100, 
                    'reverse': -2, 
                    'stolen_food_distance': 0, 
                    'distance_to_patrol': -8,
                    'team_spacing_penalty': 0}
        
        #Non scared behaviour, chase invaders and stolen food. 
        return {'num_invaders': -1000, 
                'on_defense': 100, 
                'invader_distance': -15, 
                'stop': -100, 
                'reverse': -2, 
                'stolen_food_distance': -13,  
                'distance_to_patrol': -6,
                'team_spacing_penalty': -10}




"""
ideas to implement

1) Go back home when carrying more than N dots 
or when danger is high. -> IMPLEMENTED

2) punish the moves that bring you closer to visible unscared ghosts,
we should safer routes > shorter routes -> Half implemented.

3) defense when food dots get stolen, try and catch the enemy that ate it/them -> IMPLEMENTED

3.b) IMPORTANT: if the enemy has recently eaten a capsule, do not try and capture him -> IMPLEMENTED

4) last N moves of the game, if we are winning, we should stay home and defend,
if we are losing, go full attack. -> IMPLEMENTED 


5) Add STOP + reverse penalties on offense -> IMPLEMENTED

6) the return-home threshold is static (N = 5), we should make a dynamic one depending on the layout and also If winning: return earlier / play safer, If losing: carry more before returning (take calculated risks). 
-> IMPLEMENTED


7) when a dangerous ghost is close and a capsule is reachable, prefer pathing to a capsule, we convert DANGER -> potential points  IMPLEMENTED

8) avoid dead-end situations especially when an unscared ghost is nearby. Maybe even encourage a dead-end situation if the opponents are scared? IMPLEMENTED

9) when winning DON'T camp in the spawntube, defending has to be active and at the front -> IMPLEMENTED

10) Instead of returning to spawnpoint when having a lot of dots, return to red boundaries -> IMPLEMENTED

11) target food-dense spots

12) camp middle boundary in defense -> Implemented

13) team coordination agents can't be close, they will target same things -> Implemented

14) Two-phase role strategy (timeleft trigger = 300) -> IMPLEMENTED
  Phase 1 (timeleft > 300):
    - OffensiveReflexAgent acts as BORDER SENTINEL: patrols home boundary,
      chases enemies that cross, reacts to stolen food and does NOT attack.
    - DefensiveReflexAgent acts as CAMP DEFENDER: walks in the interior of our
      territory (3-5 tiles behind boundary) protecting food clusters.
  Phase 2 (timeleft <= 300):
    - OffensiveReflexAgent switches to SAFE ATTACKER: crosses border,
      grabs a few dots, returns home quickly. Avoids ghosts aggressively.
    - DefensiveReflexAgent switches to BORDER PATROL: moves to boundary
      line (existing patrol_points), intercepts remaining enemy attacks.
"""