import os
from model import *
from tqdm import tqdm
import generate_random_graphs as gg

#import copy
#import heapq
#import scipy
import traceback
import seaborn as sns
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib
import warnings

matplotlib.rcParams.update({'font.size': 30})
warnings.filterwarnings("ignore", category=UserWarning)
sns.axes_style("white")

# ------------------------------------------------------
# Logging
# ------------------------------------------------------

import logging

logger = logging.getLogger('LumpingLogger')
logger.setLevel(logging.INFO)
logpath = 'LogOutput.log'
fh = logging.FileHandler(logpath, mode='w')  # change to a to overwrite
fh.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(process)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)
logger.info('-------------------------------------------------')
logger.info('                 Start Logging                   ')
logger.info('-------------------------------------------------')
# config
BLUE = (59 / 255.0, 117 / 255.0, 175 / 255.0)
RED = (197 / 255.0, 58 / 255.0, 50 / 255.0)

# plots degree distribution of graph
def plot_degree_distribution(G, outpath):
    degrees = [G.degree(n) for n in G.nodes()]
    x_values = range(np.max(degrees) + 1)
    y_values = [degrees.count(d) for d in x_values if degrees.count(d) > 0]
    x_values = [d for d in x_values if degrees.count(d) > 0]
    plt.clf()
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.locator_params(axis='y', nbins=6)
    plt.locator_params(axis='x', nbins=6)
    plt.scatter(x_values, y_values, alpha=0.8, facecolor='None', edgecolors=(76 / 255, 114 / 255, 176 / 255),
                linewidths=2)
    plt.scatter([0], [0], alpha=0)
    (plt.gca()).tick_params(axis='both', which='both', length=0)
    plt.xlabel('Degree', fontsize=30)
    plt.ylabel('Number of Nodes', fontsize=30)
    (plt.gca()).tick_params(axis='both', which='both', length=0)

    plt.savefig(outpath.replace('.gml', 'degrees.pdf'), bbox_inches="tight")
    df = pd.DataFrame({'Time': x_values, 'NumInfected': y_values})
    df.to_csv(outpath.replace('.gml', 'degrees.csv'))

# write graph as .gml and egdgelist file
def write_graph(G, outpath):
    assert ('.gml' in outpath)
    plot_degree_distribution(G, outpath)
    try:  # sometimes fails somehow
        nx.write_gml(G, outpath)
    except Exception as e:
        logger.info('graph output failed')
    nx.write_edgelist(G, outpath.replace('.gml', '.edgelist'))
    if G.number_of_nodes() > 201:
        # too slow
        return
    pos = nx.kamada_kawai_layout(G)
    plt.clf()
    plt.figure()
    nx.draw(G, pos, edge_color='black', width=1, linewidths=0.5,
            node_size=50, node_color='black', alpha=0.8)
    plt.axis('off')
    (plt.gca()).tick_params(axis='both', which='both', length=0)
    plt.savefig(outpath.replace('.gml', 'plot.pdf'))




# numerically determine the disired infection rate for a given R_0 + k_mean using 1D grid search
def get_infection_rate(desired_r_0, k_mean, i1_to_i2=0.033, i2_to_i3=0.042, i1_to_r=0.133, i2_to_r=0.125,
                       i3_to_r=0.075 + 0.05, dumping_factor=5.0):
    def compute_r0(inf_rate, k_mean, i1_to_i2, i2_to_i3, i1_to_r, i2_to_r, i3_to_r, dumping_factor):
        # return k_mean * (inf_rate/(0.033+inf_rate+0.133) + 0.033/(0.033+inf_rate+0.133) * (inf_rate/dumping_factor)/(inf_rate/dumping_factor+0.042+0.125) + 0.033/(0.033+inf_rate+0.133) * 0.125/( 0.125+inf_rate/dumping_factor ) * (inf_rate/dumping_factor)/((inf_rate/dumping_factor) + 0.125))
        prob_i1_inf = inf_rate / (i1_to_i2 + inf_rate + i1_to_r)
        prob_i2_inf = i1_to_i2 / (i1_to_i2 + inf_rate + i1_to_r) * (
                (inf_rate / dumping_factor) / (inf_rate / dumping_factor + i2_to_i3 + i2_to_r))
        prob_i3_inf = i1_to_i2 / (i1_to_i2 + inf_rate + i1_to_r) * i2_to_i3 / (
                inf_rate / dumping_factor + i2_to_i3 + i2_to_r) * (
                              (inf_rate / dumping_factor) / (inf_rate / dumping_factor + i3_to_r))
        prob_edge = prob_i1_inf + prob_i2_inf + prob_i3_inf
        r_0 = k_mean * prob_edge
        return r_0

    values = list()
    for inf_rate in np.linspace(0.0001, 2.0, 100000):
        computed_r_0 = compute_r0(inf_rate, k_mean, i1_to_i2, i2_to_i3, i1_to_r, i2_to_r, i3_to_r, dumping_factor)
        diff_r_0 = np.abs(computed_r_0 - desired_r_0)
        values.append((inf_rate, computed_r_0, diff_r_0))
    values = sorted(values, key=lambda x: x[-1])
    return values[0][0]


def multi_simulation(name, G_gen, sim_num=1000, model_parameters=None, horizon=10, SpreadingModel=CoronaHill):
    os.system('mkdir output_dynamics')
    folderpath = 'output_dynamics/{name}/'.format(name=name)
    os.system('mkdir ' + folderpath)

    plt.clf()
    plt.cla()
    plt.close()

    df_evolution_summary = None
    df_rvalues_summary = None

    # save experiment to make replication easier
    #for pythonfile in glob.glob('*.py'):
    #    os.system('cp {} {}'.format(pythonfile, folderpath + pythonfile))

    if model_parameters is None:
        model_parameters = dict()

    # check if exists
    sim_start = 0
    try:
        if len(glob.glob(folderpath + 'snapshot_df_evolution_summary.csv')) > 0:
            epochpath_df_evolution_summary = sorted(glob.glob(folderpath + 'snapshot_df_evolution_summary.csv'))[-1]
            epochpath_df_rvalues_summary = sorted(glob.glob(folderpath + 'snapshot_df_rvalues_summary.csv'))[-1]
            df_evolution_summary = pd.read_csv(epochpath_df_evolution_summary)
            df_rvalues_summary = pd.read_csv(epochpath_df_rvalues_summary)
            sim_start = np.max(df_rvalues_summary['sim_index']) + 1
            logger.info('found: ' + epochpath_df_evolution_summary)
    except:
        logger.info(traceback.format_exc())
        time.sleep(0.5)

    # ------------------------------------------------------
    # perform simulations
    # ------------------------------------------------------
    for sim_index in range(sim_num):

        if sim_index == 0:
            logger.info('Start simulation of ' + name)
            pbar = tqdm(total=sim_num)
            pbar.set_description('Simulations')

        if sim_index < sim_start:
            pbar.update(1)
            continue
        if sim_index == sim_start and sim_index > 0:
            logger.info('start simulation from here ' + str(sim_index))

        G = G_gen()
        model = SpreadingModel(G, np.linspace(0.0, horizon, 201), **model_parameters)

        # solve ODE (do this only once)
        if sim_index == 0 or sim_index == sim_start:
            try:
                model.solve_ode(folderpath + 'ode.pdf')
            except:
                logger.info(traceback.format_exc())

        for _ in range(1000000):
            # here the actual simulation step is happening
            time_passed, changing_node, new_state = model.step()
            if time_passed is None:
                break
        df_e, df_r = model.get_summary()
        df_e['sim_index'] = str(sim_index)
        df_r['sim_index'] = str(sim_index)
        if df_evolution_summary is None:
            df_evolution_summary = df_e
            df_rvalues_summary = df_r
        else:
            df_evolution_summary = pd.concat([df_evolution_summary, df_e], sort=True)
            df_rvalues_summary = pd.concat([df_rvalues_summary, df_r], sort=True)

        if (sim_index % 10 == 0 and sim_index > 0) or sim_index == sim_num - 1 or (
                sim_index == sim_start and sim_index > 10):
            # ------------------------------------------------------
            # visualize simulations
            # ------------------------------------------------------
            df_evolution_summary.to_csv(folderpath + 'snapshot_df_evolution_summary.csv', index=False)
            df_rvalues_summary.to_csv(folderpath + 'snapshot_df_rvalues_summary.csv', index=False)

            plt.clf()
            palette = None
            try:
                palette = model.colors()
            except:
                pass
            df_evolution_summary['fraccount'] = df_evolution_summary['count'] / G.number_of_nodes()
            df_evolution_summary.to_csv(folderpath + 'summary_evolution.csv', index=False)
            g = sns.lineplot(data=df_evolution_summary, x='time', y='fraccount', hue='state', palette=palette)
            g.legend(loc='center left', bbox_to_anchor=(0.0, 1.4), ncol=3)
            plt.ylim(0.0, 1.0)
            plt.xlim(0.0, horizon)
            plt.xlabel('Time', fontsize=42)
            plt.ylabel('Prevalence', fontsize=42)
            plt.xticks(fontsize=33)
            plt.yticks(fontsize=33)
            (plt.gca()).tick_params(axis='both', which='both', length=0)
            plt.savefig(folderpath + 'evolution_summary.pdf', bbox_inches="tight")

            plt.clf()
            g = sns.lineplot(data=df_evolution_summary, x='time', y='fraccount', hue='state', ci='sd', palette=palette)
            g.legend(loc='center left', bbox_to_anchor=(0.0, 1.4), ncol=3)
            plt.ylim(0.0, 1.0)
            plt.xlim(0.0, horizon)
            plt.xlabel('Time', fontsize=42)
            plt.ylabel('Prevalence', fontsize=42)
            plt.xticks(fontsize=33)
            plt.yticks(fontsize=33)
            (plt.gca()).tick_params(axis='both', which='both', length=0)
            plt.savefig(folderpath + 'evolution_summarySD.pdf', bbox_inches="tight")

        pbar.update(1)

    pbar.close()

    return df_evolution_summary, df_rvalues_summary


def start_experiments(num_nodes=1000, sim_num=1000, r_0=2.5, nameprefix='', horizon=200, mean_degree=8, init_inf=3):
    plt.clf()
    plt.cla()
    plt.close()  # to avoid some plotting artefacts

    def init_state_generator():
        init_infected = list(np.random.choice(range(G.number_of_nodes()), init_inf))
        return ['I1' if i in init_infected else 'S' for i in range(G.number_of_nodes())]

    graphnames = [(lambda: gg.barabasi_mean(num_nodes, mean_degree), 'BA'),
                  (lambda: gg.household_simpleGeom(num_nodes, 4, 5), 'household'),
                  (lambda: gg.newman(num_nodes, mean_degree, 0.05), 'newman05'),
                  (lambda: gg.newman(num_nodes, mean_degree, 0.20), 'newman20'),
                  (lambda: gg.geom(num_nodes, mean_degree=mean_degree), 'geom'),
                  (lambda: gg.power_law_meandegree(num_nodes, mean_degree=mean_degree), 'powerlaw'),
                  (lambda: nx.path_graph(num_nodes), 'complete')]  # important: if name is "complete" use different solver, don't build actual graph (line_graph is ok here)

    for G_gen, graphname in graphnames:
        time.sleep(0.5)  # makes it easier to stop process with ctrl+c

        #if 'comp' not in graphname:
        #    continue

        # Rates and corresponding mean number of days in each compartment
        e_to_i1 = 0.2  # 1.0/5.0
        i1_to_i2 = 0.033  # 0.2 / 6.0
        i2_to_i3 = 0.042  # 0.25 / 6.0 (approximated)
        i1_to_r = 0.133  # 0.8 / 6
        i2_to_r = 0.125  # 0.75 / 6.0
        i3_to_r = 0.075 + 0.05  # 1.0 / 5.0  0.125 = 1/8

        if graphname ==  'complete':
            k_mean = num_nodes-1
            mean_i_rate = get_infection_rate(r_0, k_mean)
        else:
            k_mean = mean_degree
            mean_i_rate = get_infection_rate(r_0, k_mean)
        logger.info('mean_i_rate: '+str(mean_i_rate))


        try:
            name = nameprefix + 'EXP1_' + str(mean_degree) + '_' + graphname

            # ---------------
            # Homog. infection rate
            # ---------------
            G = G_gen()
            logger.info(nx.info(G))
            logger.info('Avg shortest path: ' + str(nx.average_shortest_path_length(G)))
            logger.info('radius: ' + str(nx.radius(G)))
            os.system('mkdir output_graphs')
            write_graph(G, 'output_graphs/example_graph_{}.gml'.format(name))

            infection_rate_generator = lambda: np.ones(G.number_of_nodes()) * mean_i_rate

            model_parameters = {'infection_rate_generator': infection_rate_generator,
                                'init_state_generator': init_state_generator}
            model_parameters['is_complete_graph'] = 'complete' in graphname
            model_parameters['e_to_i1'] = e_to_i1
            model_parameters['i1_to_i2'] = i1_to_i2
            model_parameters['i2_to_i3'] = i2_to_i3
            model_parameters['i1_to_r'] = i1_to_r
            model_parameters['i2_to_r'] = i2_to_r
            model_parameters['i3_to_r'] = i3_to_r
            multi_simulation(name, G_gen, model_parameters=model_parameters, horizon=horizon, sim_num=sim_num,
                             SpreadingModel=CoronaHill)
        except:
            logger.info(traceback.format_exc())

        try:
            name = nameprefix + 'EXP2_' + str(mean_degree) + '_' + graphname

            # ---------------
            # Exponentialy distributed infection rate
            # ---------------
            infection_rate_generator = lambda: np.random.exponential(mean_i_rate, G.number_of_nodes())

            model_parameters = {'infection_rate_generator': infection_rate_generator,
                                'init_state_generator': init_state_generator}
            model_parameters['is_complete_graph'] = 'complete' in graphname
            model_parameters['e_to_i1'] = e_to_i1
            model_parameters['i1_to_i2'] = i1_to_i2
            model_parameters['i2_to_i3'] = i2_to_i3
            model_parameters['i1_to_r'] = i1_to_r
            model_parameters['i2_to_r'] = i2_to_r
            model_parameters['i3_to_r'] = i3_to_r
            multi_simulation(name, G_gen, model_parameters=model_parameters, horizon=horizon, sim_num=sim_num,
                             SpreadingModel=CoronaHill)
        except:
            logger.info(traceback.format_exc())


# ---------------
# Start Exp 1 and Exp 2
# ---------------
num_nodes = 1000
sim_num = 10  # this should be 1000
start_experiments(num_nodes=num_nodes, sim_num=sim_num, mean_degree=8, r_0=2.5)



