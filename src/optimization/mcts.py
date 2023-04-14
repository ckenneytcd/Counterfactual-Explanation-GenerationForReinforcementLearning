import copy

import numpy as np
import math
import random
import sys 

class MCTSNode:

    def __init__(self, state, mainroot, parent, action, rew, env, bb_model,  obj, fact, target_action, cfmode, state_path):
        self.mainroot = mainroot
        self.state = state
        self.parent = parent
        self.env = env
        self.bb_model = bb_model
        self.obj = obj
        self.fact = fact
        self.target_action = target_action
        self.cfmode = cfmode
        self.action = action
        self.rew = rew
        self.n_visits = 0
        self.N_a = {}

        self.children = {}
        
        if self.parent:
            self.prev_actions = self.parent.prev_actions + [action] 
        elif action:
            self.prev_actions = [action] 
        else:
            self.prev_actions = [] 
        
        self.state_path = state_path
 
        self.cumulative_reward = self.parent.cumulative_reward + rew if self.parent else 0
        
        self.expanded_actions = []
        self.Q_values = {}

        self.level = self.parent.level + 1 if self.parent is not None else 0

    def available_actions(self):
        return self.env.get_actions(self.state)

    def is_terminal(self):

        if(len(self.available_actions()) == 1):
            outcome = self.env.single_step_nochange(self.state, self.available_actions()[0])
            if(list(outcome) == list(self.state)):
               return True
            if(self.parent and list(outcome) == list(self.parent.state)):
                return True

        cf_predicted_action = self.bb_model.predict(self.state)
        f_predicted_action = self.bb_model.predict(self.fact)

        if(self.cfmode == 'NETACTION'):
            if(self.env.check_done(self.state)):
                return True
            if(self.state[1] == self.fact[1] and self.state[0] == self.fact[0]):
                return False
            #Check if the position with the same action choice is paraell to the fact
            if(f_predicted_action == 0 and self.state[1] != self.fact[1] and cf_predicted_action == f_predicted_action):
                #print("South: parell would be column")
                return True
            if(f_predicted_action == 1 and self.state[1] != self.fact[1] and cf_predicted_action == f_predicted_action):
                #print("North: parell would be column")
                return True
            if(f_predicted_action == 2 and self.state[0] != self.fact[0] and cf_predicted_action == f_predicted_action):
                #print("East: parell would be row")
                return True
            if(f_predicted_action == 3 and self.state[0] != self.fact[0] and cf_predicted_action == f_predicted_action):
                #print("East: parell would be row")
                return True
                
            return False 
        elif (self.cfmode == 'ACTION_CHANGEDST'):
                
                if(self.env.check_done(self.state) or cf_predicted_action == self.target_action):
                    return True
                
                # Check all valid states with the same taxi and passanger location
                for i in range(4):
                    newstate = copy.deepcopy(self.state)
                    newstate[3] = i
                    if(self.bb_model.predict(newstate) != None and i != newstate[2]):
                        if(self.env.check_done(newstate) or self.bb_model.predict(newstate) == self.target_action):
                            self.state = newstate
                            return True
                
                return False
        elif (self.cfmode == 'ACTION_CHANGELOC'):
            if(self.env.check_done(self.state) or cf_predicted_action == self.target_action):
                return True
            # Check all valid states with the same taxi and destination location while not in the taxi or at destination
            if(self.state[2] != 4 and self.state[2] != self.state[3]):
                for i in range(4):
                    newstate = copy.deepcopy(self.state)
                    newstate[2] = i
                    if(self.bb_model.predict(newstate) != None and i != newstate[3]):
                        if(self.env.check_done(newstate) or self.bb_model.predict(newstate) == self.target_action):
                            self.state = newstate
                            return True
            return False
        elif(self.cfmode == 'ACTION_CHANGEBOTH'):
            if(self.env.check_done(self.state) or cf_predicted_action == self.target_action):
                return True
            #iterate through and check the location or the destination
            listpass = [0,1,2,3]
            listdst = [0,1,2,3]
            while listpass or listdst:
                list_choice = random.randint(0,1)
                if(self.state[2] == 4):
                    listpass = []
                    list_choice = 1
                if(list_choice == 0):
                    
                    if(len(listpass) > 0):
                        i = listpass.pop()
                        newstate = copy.deepcopy(self.state)
                        newstate[2] = i

                        if(self.bb_model.predict(newstate) != None and i != newstate[3]):
                            if(self.env.check_done(newstate) or self.bb_model.predict(newstate) == self.target_action):
                                self.state = newstate
                                return True
                else:
                    if((len(listdst) > 0)):
                        i = listdst.pop()
                        newstate = copy.deepcopy(self.state)
                        newstate[3] = i
                        if(self.bb_model.predict(newstate) != None and i != newstate[2]):
                            if(self.env.check_done(newstate) or self.bb_model.predict(newstate) == self.target_action):
                                self.state = newstate
                                return True
            return False
        else:
            if(self.env.check_done(self.state)):
                return True
            return cf_predicted_action == self.target_action
        

    def take_action(self, action, n_expand, expand=True):
        nns = []
        rewards = []
        s = n_expand if expand else 1
        for i in range(s):
            
            self.env.reset()
            self.env.set_state(self.state)

            obs, rew, done, _ = self.env.step(action)
            found = False
            for nn in nns:
                if self.env.equal_states(obs, nn.state):
                    found = True
                    break
            
            if not found:
                nn = MCTSNode(obs, self.mainroot, self, action, rew, self.env, self.bb_model, self.obj, self.fact, self.target_action, self.cfmode, self.state_path+[self.env.encode(obs[0],obs[1],obs[2],obs[3])])
                nns.append(nn)
                rewards.append(rew)



        return nns, rewards

    def get_reward(self):
        return self.obj.get_reward(self.fact, self.state, self.target_action, self.prev_actions, self.cumulative_reward)

    def clone(self):
        clone = MCTSNode(self.state, None, None, None, self.env, self.bb_model,  self.obj, self.fact, self.target_action, self.state_path)
        clone.prev_actions = self.prev_actions
        clone.cumulative_reward = self.cumulative_reward

        return clone

    def is_valid(self, fact, cfmode):
        return self.env.realistic(self.state,fact, self.prev_actions)

class MCTS:

    def __init__(self, env, bb_model, obj, fact, target_action, cfmode, max_level, n_expand, c=None):
        self.max_level = max_level
        self.c = c if c is not None else 1 / math.sqrt(2)
        self.env = env
        self.bb_model = bb_model
        self.obj = obj
        self.fact = fact
        self.target_action = target_action
        self.n_expand = n_expand
        self.cfmode = cfmode
        self.searched = []

        self.tree_size = 0

    def search(self, init_state, num_iter=200):
        self.root = MCTSNode(init_state, self.fact, None, None, 0, self.env, self.bb_model, self.obj, self.fact, self.target_action, self.cfmode, [self.env.encode(self.fact[0],self.fact[1],self.fact[2],self.fact[3])])
        i = 0
        while i < num_iter:
            
            node = self.select(self.root)

            i += 1
            if (not node.is_terminal()) and (node.level < self.max_level):
                
                new_nodes, action = self.expand(node)
                
                for n in new_nodes:
                    n.value = n.get_reward()

                if len(new_nodes):
                    self.backpropagate(new_nodes[0].parent)

        return self.tree_size, 0

    def select(self, root):
        node = root

        while (not node.is_terminal()) and (len(node.children) > 0):
            
            action_vals = {}
            
            for a in node.available_actions():
                if a in node.available_actions():
                    try:
                        n_a = node.N_a[a]
                        Q_val = node.Q_values[a]
                        action_value = Q_val + self.c * math.sqrt((math.log(node.n_visits) / n_a))
                        action_vals[a] = action_value

                    except KeyError:
                        action_value = 0

            best_action = max(action_vals, key=action_vals.get)
            #print(action_vals)
            max_val = max(action_vals.values())
            best_actions = [k for k, v in action_vals.items() if v == max_val]
            if len(best_actions) == 1:
                best_action = best_actions[0]
            elif len(best_actions) > 2 and ((best_actions[0] == 0 and best_actions[1] == 1) or (best_actions[0] == 1 and best_actions[1] == 0)):
                best_action = 1

            try:
                node.N_a[best_action] += 1
            except KeyError:
                node.N_a[best_action] = 1

            child = np.random.choice(node.children[best_action])

            node = child

        return node

    def expand(self, node):
        nns = []

        allchecked = True
        for aa in node.available_actions():
            if(not(aa in node.expanded_actions)):
                allchecked = False

        if allchecked:
            return [], None

        if node.is_terminal():
            return [], None
        
        for action in node.available_actions():
            
            if action not in node.expanded_actions and action in node.available_actions():
                new_states, new_rewards = node.take_action(action, n_expand=self.n_expand)
                
                try:
                    node.N_a[action] += 1
                except KeyError:
                    node.N_a[action] = 1

                node.expanded_actions.append(action)

                for i, ns in enumerate(new_states):
                    if ns.is_valid(self.fact, self.cfmode):
                        try:
                            node.children[action].append(ns)
                        except KeyError:
                            node.children[action] = [ns]

                        nns.append(ns)

                    else:    
                        
                        self.tree_size += 1

        return nns, action

    def simulate(self, node):
        node = node.clone()
        n_sim = 1
        evals = []

        for i in range(n_sim):
            l = 0
            evaluation = 0.0
            start_node = node.clone()
            while (not start_node.is_terminal()) and (l < 5):
                l += 1

                rand_action = np.random.choice(start_node.available_actions())
                start_node = start_node.take_action(rand_action, n_expand=self.n_expand, expand=False)[0][0]

                e = start_node.get_reward()
                evaluation = e.item()

            evals.append(evaluation)

        return np.mean(evals)

    def backpropagate(self, node):
        while node is not None:
            node.n_visits += 1

            for a in node.expanded_actions:
                try:
                    node.Q_values[a] = np.mean([n.value for n in node.children[a]])
                except KeyError:
                    node.Q_values[a] = -1000

            node = node.parent