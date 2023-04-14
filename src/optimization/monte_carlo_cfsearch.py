import numpy as np
import math
from src.models.counterfactual import CF
from src.optimization.mcts import MCTS
import sys

class MCTSSearch:

    def __init__(self, env, bb_model, dataset, obj, params, cfmode, c):
        self.env = env
        self.bb_model = bb_model
        self.dataset = dataset
        self.n_var = env.state_dim
        self.obj = obj

        self.n_iter = params['ts_n_iter']
        self.n_expand = params['ts_n_expand']
        self.max_cf_path_len = params['max_cf_path_len']
        self.cfmode = cfmode

        self.c = c
        self.action_dic = {0: 'SOUTH', 1: 'NORTH', 2: 'EAST', 3: 'WEST', 4:'PICKUP', 5: 'DROPOFF'}
        
    def generate_counterfactuals(self, fact, target, nbhd=None):
        mcts_solver = MCTS(self.env, self.bb_model, self.obj, fact, target, self.cfmode, c=self.c, max_level=self.max_cf_path_len, n_expand=self.n_expand)
        found = False

        tree_size, time = mcts_solver.search(init_state=fact, num_iter=self.n_iter)

        all_nodes = self.traverse(mcts_solver.root)

        potential_cf = []
        illegal = 0
        
        for n in all_nodes: 
            if n.is_terminal():
                if(not(self.env.check_done(n.state)) and self.bb_model.predict(n.state) == target and self.cfmode != "NETACTION"):
                    n.state_path = [self.env.encode(fact[0],fact[1],fact[2],fact[3])] + n.state_path
                    action_path = self.getActionPath(n.prev_actions)
                    if(self.cfmode != 'ACTION_CHANGELOC' and self.cfmode != 'ACTION_CHANGEBOTH'):
                        if(self.pickuplocchange(n.state_path) == False):
                            potential_cf += [CF(n.state, True, n.prev_actions, n.cumulative_reward, n.get_reward(), tree_size, time, n.state_path, len(action_path), action_path)]
                    else:
                        potential_cf += [CF(n.state, True, n.prev_actions, n.cumulative_reward, n.get_reward(), tree_size, time, n.state_path, len(action_path), action_path)]
                elif(self.cfmode == "NETACTION" and self.bb_model.predict(n.state) == self.bb_model.predict(fact) and not(self.env.check_done(n.state))):
                    n.state_path = [self.env.encode(fact[0],fact[1],fact[2],fact[3])] + n.state_path
                    action_path = self.getActionPath(n.prev_actions)
                    potential_cf += [CF(n.state, True, n.prev_actions, n.cumulative_reward, n.get_reward(), tree_size, time, n.state_path, len(action_path), action_path)]

       # return only the best one
        print('Found {} counterfactuals'.format(len(potential_cf)))

        if len(potential_cf):
            best_cf_ind = np.argmax([cf.value for cf in potential_cf])
            try:
                best_cf = potential_cf[best_cf_ind]
            except IndexError:
                return None
        else:
            return None

        
        return best_cf
    
    def traverse(self, root, nodes=None):
        ''' Returns all nodes in the tree '''
        if nodes is None:
            nodes = set()

        nodes.add(root)
        
        
        if root.children is not None and len(root.children):
            children = []
            for action in root.children.keys():
                children += root.children[action]

            for c in children:
                self.traverse(c, nodes)
        return nodes
    
    def getPath(self, child, root, nodes, length):
        length += [self.env.encode(child.state[0],child.state[1],child.state[2],child.state[3])]
        if(child == root):
            return length
        else:
            return self.getPath(child.parent, root, nodes, length)
    
    def getActionPath(self, path):
        action_path = []
        for a in path:
            action_path += [self.action_dic[a]]
        return action_path

    def pickuplocchange(self, path):
        path2 = path.copy()
        if(len(path2) > 1):
            s1 = list(self.env.decode(path2.pop(0)))
            s2 = list(self.env.decode(path2.pop(0)))
            done = False
            while not(done):
                #Changed passanger state wihtout being in taxi
                if (s1[2] != s2[2] and s1[2] != 4 and s2[2] != 4):
                    print("PickupChanged: ", s1, s2)
                    return True
                if len(path2) == 0:
                    done = True
                else:
                    s1 = s2
                    s2 = list(self.env.decode(path2.pop(0)))
        return False