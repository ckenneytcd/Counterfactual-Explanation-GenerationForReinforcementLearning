import json
import math

import chess
import chess.engine
import torch
import pandas as pd
import gym
import numpy as np
import pstats


from src.models.taxi_bb_model import TaxiBBModel
from src.optimization.autoenc import AutoEncoder
from src.optimization.genetic_baseline import GeneticBaseline
from src.models.dataset import Dataset
from src.envs.taxi import TaxiEnv
from src.envs.taxi8x8 import TaxiEnv as TaxiEnv8x8
from src.objectives.baseline_objs import BaselineObjectives
from src.objectives.rl_objs import RLObjs
from src.tasks.task import Task
from src.optimization.monte_carlo_cfsearch import MCTSSearch
from src.utils.utils import seed_everything, load_fact


def main():
    seed_everything(seed=1)

    task_name = 'taxi8x8'

    # define paths
    model_path = 'trained_models/{}'.format(task_name)
    dataset_path = 'datasets/{}/dataset.csv'.format(task_name)
    fact_file = 'fact/{}.json'.format(task_name)
    param_file = 'params/{}.json'.format(task_name)

    # load parameters
    with open(param_file, 'r') as f:
        params = json.load(f)
        print('Task = {} Parameters = {}'.format(task_name, params))

    if task_name == 'taxi':
        env = TaxiEnv()
        model_path = model_path + ".pickle"
        bb_model = TaxiBBModel(env, model_path)
        enc_layers = [env.state_dim, 128, 16]
        max_actions = params['max_cf_path_len']
    elif task_name == 'taxi8x8':
        env = TaxiEnv8x8()
        model_path = model_path + ".pickle"
        bb_model = TaxiBBModel(env, model_path)
        statelength = 4
        enc_layers = [statelength, 128, 16]
        max_actions = params['max_cf_path_len']



    # define models
    dataset = Dataset(env, bb_model, dataset_path)
    train_dataset, test_dataset = dataset.split_dataset(frac=0.8)
    vae = AutoEncoder(layers=enc_layers)
    vae.fit(train_dataset, test_dataset)
    enc_data = vae.encode(torch.tensor(dataset._dataset.values))[0]

    # define counterfactual type and amount of facts to test
    test_mode_names = ['SAMPLE_FACTS', 'ALL_STATES']
    test_mode = 'SAMPLE_FACTS'
    cftype_names = ['ACTION', 'NETACTION', 'ACTION_CHANGELOC', 'ACTION_CHANGEDST', 'ACTION_CHANGEBOTH']
    cftype = 'ACTION'

    # # define objectives
    baseline_obj = BaselineObjectives(env, bb_model, vae, enc_data, env.state_dim,  cftype)
    rl_obj = RLObjs(env, bb_model, params, cftype, max_actions=max_actions)

    # get facts
    if task_name == 'taxi':
        if(test_mode == "SAMPLE_FACTS"):
            try:
                dataset_path = 'datasets/{}/facts.csv'.format(task_name)
                facts = pd.read_csv(dataset_path).values
            except FileNotFoundError:
                n_facts = 100
                facts = dataset._dataset.sample(n=n_facts)
                facts.to_csv(dataset_path, index=False)
            targets = None
        elif(test_mode == "ALL_STATES"):
            try:
                dataset_path = 'datasets/{}/allstates.csv'.format(task_name)
                facts = pd.read_csv(dataset_path).values
            except FileNotFoundError:
                n_facts = 100
                facts = dataset._dataset.sample(n=n_facts)
                facts.to_csv(dataset_path, index=False)
            targets = None
    elif task_name == 'taxi8x8':
        try:
            dataset_path = 'datasets/{}/facts.csv'.format(task_name)
            facts = pd.read_csv(dataset_path).values
        except FileNotFoundError:
            n_facts = 100
            facts = dataset._dataset.sample(n=n_facts)
            facts.to_csv(dataset_path, index=False)
        targets = None
    
    

    # define methods n
    BO_GEN = GeneticBaseline(env, bb_model, dataset._dataset, baseline_obj, params)
    BO_MCTS = MCTSSearch(env, bb_model, dataset._dataset, baseline_obj, params, cftype, c=1)
    RL_MCTS = MCTSSearch(env, bb_model, dataset._dataset, rl_obj, params, cftype, c=1/math.sqrt(2))

    # method names
    #methods = [RL_MCTS, BO_MCTS, BO_GEN]
    #method_names = ['RL_MCTS', 'BO_MCTS', 'BO_GEN']
    methods = [RL_MCTS]
    method_names = ['RL_MCTS']
    

    for i, m in enumerate(methods):
        print('\n------------------------ {} ---------------------------------------\n'.format(method_names[i]))
        #eval_path = 'eval/{}/{}/rl_obj_results'.format(task_name, method_names[i])
        eval_path = 'eval/{}/{}/rl_obj_{}_results'.format(task_name, method_names[i],cftype)
        visual_path = 'eval/{}/{}/visual_{}_results.docx'.format(task_name, method_names[i], cftype)
        gen_nbhd = True if method_names[i] == 'BO_GEN' else False

        task = Task(task_name, env, bb_model, dataset, m, method_names[i], rl_obj, [rl_obj, baseline_obj], eval_path, visual_path, nbhd=gen_nbhd)

        task.run_experiment(cftype, facts, targets)



if __name__ == '__main__':
    main()
