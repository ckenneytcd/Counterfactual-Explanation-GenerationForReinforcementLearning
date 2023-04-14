import re

import torch
import pandas as pd

from src.envs.gridworld import Gridworld
from src.models.dataset import Dataset
from src.models.gridworld_bb_model import GridworldBBModel
from src.objectives.baseline_objs import BaselineObjectives
from src.optimization.autoenc import AutoEncoder


def main():
    metod_names = ['RL_MCTS']

    model_path = 'trained_models/{}'.format('gridworld')
    dataset_path = 'datasets/{}/dataset.csv'.format('gridworld')

    env = Gridworld()
    bb_model = GridworldBBModel(env, model_path)
    enc_layers = [env.state_dim, 128, 16]

    dataset = Dataset(env, bb_model, dataset_path)
    train_dataset, test_dataset = dataset.split_dataset(frac=0.8)
    vae = AutoEncoder(layers=enc_layers)
    vae.fit(train_dataset, test_dataset)
    enc_data = vae.encode(torch.tensor(dataset._dataset.values))[0]

    baseline_obj = BaselineObjectives(env, bb_model, vae, enc_data, env.state_dim)

    for m in metod_names:
        eval_df_path = 'gridworld_{}_refactored.csv'.format(m)
        baseline_rews = {}
        df = pd.read_csv(eval_df_path, header=0)

        # for i in range(len(df)):
        #     fact = df['fact'].loc[i]
        #     cf = df['cf'].loc[i]
        #     target = int(df['target'].loc[i])
        #
        #     if cf != '0':
        #         fact_json = from_res_to_json(fact)
        #         cf_json = from_res_to_json(cf)
        #
        #         fact_array = env.generate_state_from_json(fact_json)
        #         cf_array = env.generate_state_from_json(cf_json)
        #
        #         ind_rews = baseline_obj.get_ind_rews(fact_array, cf_array, target_action=target)[0]
        #     else:
        #         ind_rews = {'proximity': -1, 'sparsity': -1, 'dmc': -1, 'validity': 0, 'actionable': 0, 'realistic':0}
        #
        #     if len(baseline_rews) == 0:
        #         baseline_rews = {k: [v] for k, v in ind_rews.items()}
        #     else:
        #         baseline_rews = {k: baseline_rews[k] + [nv] for k, nv in ind_rews.items()}
        #
        # for k, v in baseline_rews.items():
        #     df[k] = v

        df['cost'] = df['cost'].apply(lambda x: -float(re.findall(r'[-+]?(?:\d*\.*\d+)', str(x))[0]))
        df['reachability'] = df['reachability'].apply(lambda x: float(re.findall(r'[-+]?(?:\d*\.*\d+)', str(x))[0]))
        df['stochasticity'] = df['stochasticity'].apply(lambda x: float(re.findall(r'[-+]?(?:\d*\.*\d+)', str(x))[0]))

        df['l_bo'] = df['proximity'] + df['sparsity'] + df['dmc']
        df['l_rl'] = df['cost'] + df['reachability'] + df['stochasticity']

        df.to_csv('gridworld_{}_refactored_2.csv'.format(m), index=None)


def from_res_to_json(res):
    locations = re.findall(r'\d+', res)
    agent = locations[0]
    monster = locations[1]

    trees = [{int(locations[i]): int(locations[i + 1])} for i in range(2, len(locations)) if i % 2 == 0]

    json = {
        'agent': int(agent),
        'monster': int(monster),
        'trees': trees
    }

    return json

if __name__ == '__main__':
    main()