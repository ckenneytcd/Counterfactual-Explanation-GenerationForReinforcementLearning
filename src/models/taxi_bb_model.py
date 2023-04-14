from collections import defaultdict
import csv
import pickle
import random
import gym
import numpy as np

class TaxiBBModel():

    def __init__(self, env, model_path):
        self.num_episodes = 100000

        self.alpha = 0.9
        self.gamma = 0.6
        self.epsilon = 0.1

        self.model_path = model_path
        self.env = env

        self.model = self.load_model(model_path, self.env)

    def load_model(self, model_path, env):
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            print('Loaded bb model')
        except FileNotFoundError:
            print('Training bb model')
            
            #model is the q-table
            model = defaultdict(int, {})
            model = self.train_agent(model, env, self.num_episodes)
            #save model
            with open(model_path, "wb") as f:
                pickle.dump(dict(model), f)
        return model

    def predict(self, state):
        q_table = self.model
        if(len(state) > 1):
            action = self.select_optimal_action(q_table, state)
        else:
            action = self.select_optimal_action(q_table, state)

        return action
    

    def update(self, q_table, env, slist):
        statenum = env.encode(slist[0], slist[1], slist[2], slist[3])
        if random.uniform(0, 1) < self.epsilon:
            action = env.action_space.sample()
        else:
            action = self.select_optimal_action(q_table, slist)

        next_slist, reward, _, _ = env.step(action)
        old_q_value = q_table[statenum][action]
        next_statenum = env.encode(next_slist[0], next_slist[1], next_slist[2], next_slist[3])

        # Check if next_state has q values already
        if not q_table[next_statenum]:
            q_table[next_statenum] = {action: 0 for action in range(env.action_space.n)}

        # Maximum q_value for the actions in next state
        next_max = max(q_table[next_statenum].values())

        # Calculate the new q_value
        new_q_value = (1 - self.alpha) * old_q_value + self.alpha * (reward + self.gamma * next_max)

        # Finally, update the q_value
        q_table[statenum][action] = new_q_value

        return next_slist, reward
    
    def select_optimal_action(self, q_table, slist):
        statenum = self.env.encode(slist[0], slist[1], slist[2], slist[3])
        max_q_value_action = None
        max_q_value = -100000
        try:
            if q_table[statenum]:
                for action, action_q_value in q_table[statenum].items():
                    if action_q_value >= max_q_value:
                        max_q_value = action_q_value
                        max_q_value_action = action
        except:
            return None

        return max_q_value_action

    def train_agent(self, q_table, env, num_episodes):
        for i in range(num_episodes):
            slist = env.reset()
            statenum = env.encode(slist[0], slist[1], slist[2], slist[3])
            if not q_table[statenum]:
                q_table[statenum] = {
                    action: 0 for action in range(env.action_space.n)}

            epochs = 0
            num_penalties, reward, total_reward = 0, 0, 0
            while reward != 20:
                slist, reward = self.update(q_table, env, slist)
                total_reward += reward

                if reward == -10:
                    num_penalties += 1

                epochs += 1
            print("\nTraining episode {}".format(i + 1))
            print("Time steps: {}, Penalties: {}, Reward: {}".format(epochs,
                                                                    num_penalties,
                                                                    total_reward))

        print("Training finished.\n")

        return q_table
    

if __name__ == "__main__":
    main()
