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


class OffensiveReflexAgent(ReflexCaptureAgent):
    """
  A reflex agent that seeks food. This is an agent
  we give you to get an idea of what an offensive agent might look like,
  but it is by no means the best or only way to build an offensive agent.
  """
    # NEW implementation: 1) Go back home when carrying more than N dots 
    #   or when danger is high
    carrying_dots = 5  # NEW: after 5 dots, go back home
    danger_dist = 3 # NEW: 5 steps far away, pacman is in danger
          #NEW
    def _min_dist_enemy_ghost(self, successor):

        my_pos = successor.get_agent_state(self.index).get_position()
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
    
      #NEW
    def _should_return_home(self, game_state, successor):
        # return if carrying more than N dots or if it is dangerous

        carrying = game_state.get_agent_state(self.index).num_carrying
        if carrying >= self.carrying_dots:
            return True 
        
        min_ghost_distance = self._min_dist_enemy_ghost(successor)
        return (min_ghost_distance is not None and min_ghost_distance <= self.danger_dist)
            
            


    def get_features(self, game_state, action):
        features = util.Counter()
        
        successor = self.get_successor(game_state, action)
        food_list = self.get_food(successor).as_list()
        my_pos = successor.get_agent_state(self.index).get_position()
        returning = self._should_return_home(game_state, successor)
        features['successor_score'] = -len(food_list)  # self.get_score(successor)
        d = self._min_dist_enemy_ghost(successor)

        #NEW: avoid DANGER
        if d is None:
            features['min_enemy_ghost_distance'] = 10
        else:
            features['min_enemy_ghost_distance'] = min(d, 10)
        #NEW: if we return, keep the distance to home
        if returning:
            features['distance_to_home'] = self.get_maze_distance(my_pos, self.start) 
        #NEW: hunt scared ghost when on offense
        my_state = successor.get_agent_state(self.index)
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

                

        return features

    def get_weights(self, game_state, action):
        successor = self.get_successor(game_state, action)
        # NEW
        returning = self._should_return_home(game_state, successor)
    

        if returning:
            return {'successor_score': 0, 'distance_to_food': 0, 'distance_to_home': -50, 'min_enemy_ghost_distance': 8, 'hunt_scared_ghost': 0}
    
        else:
            return {'successor_score': 100, 'distance_to_food': -3, 'distance_to_home': 0, 'min_enemy_ghost_distance': 1, 'hunt_scared_ghost': 1}
    
    





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
    



    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)

        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        # New: If enemy recently ate a capsule, ghost become scared. While scared, do NOT chase the enemy to capture them. Instead, keep distance. 
        scared = (my_state.scared_timer > 0)


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
            recent_stolen_step_count = 5 #how many steps we care about
            if (not scared) and self.last_stolen_pos is not None and (self.step_count -self.last_stolen_step) <= recent_stolen_step_count:
                features['stolen_food_distance'] = self.get_maze_distance(my_pos, self.last_stolen_pos)    

        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1


        return features
   #Updated
    def get_weights(self, game_state, action):
        
        successor = self.get_successor(game_state, action)
        my_state = successor.get_agent_state(self.index)

        #If we are scared, avoid invaders instead of chasing. 
        if my_state.scared_timer > 0:
            return {'num_invaders': -1000, 'on_defense': 100, 'invader_distance': 10, 'stop': -100, 'reverse': -2, 'stolen_food_distance': 0}
        
        #Non scared behaviour, chase invaders and stolen food. 
        return {'num_invaders': -1000, 'on_defense': 100, 'invader_distance': -10, 'stop': -100, 'reverse': -2, 'stolen_food_distance': -10}




"""
ideas to implement

1) Go back home when carrying more than N dots 
or when danger is high. -> IMPLEMENTED

2) punish the moves that bring you closer to visible unscared ghosts,
we should safer routes > shorter routes -> Half implemented.

3) defense when food dots get stolen, try and catch the enemy that ate it/them -> IMPLEMENTED

3.b) IMPORTANT: if the enemy has recently eaten a capsule, do not try and capture him -> IMPLEMENTED

4) last N moves of the game, if we are winning, we should stay home and defend,
if we are losing, go full attack. -> NOT YET IMPLEMENTED

"""