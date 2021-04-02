import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
import time, random, os
import seaborn as sns
import scipy
import glob
import pickle
import collections
import heapq
import copy







class CoronaHill:
    def __init__(self, G, savepoints, infection_rate_generator, e_to_i1, i1_to_i2, i2_to_i3, i1_to_r, i2_to_r, i3_to_r, init_state_generator, rate_damper=5.0, variance_reduction=False, is_complete_graph=False):
        G = copy.deepcopy(G)
        G = nx.convert_node_labels_to_integers(G)
        self.eventlist = None
        self.current_state = G
        self.record = list() # (timepoint, G)
        self.local_states = sorted(['S', 'E', 'I1','I2','I3', 'R'])
        self.time = 0.0
        self.savepoints = sorted(savepoints, key=lambda x: -x)
        self.summary_statistics = {'time': list(), 'state': list(), 'count': list()}
        self.variance_reduction = variance_reduction
        self.is_complete_graph = is_complete_graph
        if is_complete_graph:
            print(' Use complete graph solver ')

        self.e_to_i1 = e_to_i1
        self.i1_to_i2 = i1_to_i2
        self.i2_to_i3 = i2_to_i3
        self.i1_to_r = i1_to_r
        self.i2_to_r = i2_to_r
        self.i3_to_r = i3_to_r
        self.rate_damper = rate_damper


        infection_rate_1 = infection_rate_generator()
        infection_rate_2 = infection_rate_generator()/rate_damper
        infection_rate_3 = infection_rate_generator()/rate_damper

        self.infection_rate_1 = infection_rate_1
        self.infection_rate_2 = infection_rate_2
        self.infection_rate_3 = infection_rate_3

        init_states = init_state_generator()

        for node in G:
            G.nodes[node]['became_infected_at'] = None
            G.nodes[node]['num_infected_neighbors'] = 0.0
            G.nodes[node]['infection_rate_1'] = infection_rate_1[node]
            G.nodes[node]['infection_rate_2'] = infection_rate_2[node]
            G.nodes[node]['infection_rate_3'] = infection_rate_3[node]
            assert(init_states[node] in self.local_states)
            G.nodes[node]['state'] = init_states[node]
            if G.nodes[node]['state'] in ['E', 'I1', 'I2', 'I3']:
                G.nodes[node]['became_infected_at'] = 0.0

    def states(self):
        return self.local_states

    def colors(self):
        colors = {'S': sns.xkcd_rgb['denim blue'], 'E':sns.xkcd_rgb['dark magenta'], 'I1': sns.xkcd_rgb['deep pink'],
                  'I2': (239/ 255.0, 152/ 255.0, 164/ 255.0), 'I3':(249/ 255.0, 214/ 255.0, 219/ 255.0), 'R': sns.xkcd_rgb['medium green'],
                  'D': sns.xkcd_rgb['black'], 'I_total': 'gray', 'I': (197/255.0,58/255.0,50/255.0)}
        return colors


    def get_changeevent(self, node_i):
        G = self.current_state
        new_state = 'R'
        fire_time = 1000000000.0   #dummy
        current_state = G.nodes[node_i]['state']
        assert(current_state in ['S','E','I1','I2','I3','R'])

        if current_state == 'S':
            rate_i = 0.0
            for neig_i in G.neighbors(node_i):
                if G.nodes[neig_i]['state'] == 'I1':
                    rate_i += G.nodes[neig_i]['infection_rate_1']
                elif G.nodes[neig_i]['state'] == 'I2':
                    rate_i += G.nodes[neig_i]['infection_rate_2']
                elif G.nodes[neig_i]['state'] == 'I3':
                    rate_i += G.nodes[neig_i]['infection_rate_3']
            if rate_i>0.00000000001:
                fire_time_i = -np.log(random.random()) / rate_i
                if fire_time is None or fire_time_i < fire_time:
                    fire_time = fire_time_i
                    new_state = 'E'
        elif current_state == 'E':
            fire_time_i = -np.log(random.random()) / self.e_to_i1
            if fire_time_i < fire_time:
                fire_time = fire_time_i
                new_state = 'I1'
        elif current_state == 'I1':
            fire_time_i = -np.log(random.random()) / self.i1_to_i2
            if  fire_time_i < fire_time:
                fire_time = fire_time_i
                new_state = 'I2'
        if current_state == 'I1':
            fire_time_i = -np.log(random.random()) / self.i1_to_r
            if fire_time_i < fire_time:
                fire_time = fire_time_i
                new_state = 'R'
        if current_state == 'I2':
            fire_time_i = -np.log(random.random()) / self.i2_to_i3
            if fire_time_i < fire_time:
                fire_time = fire_time_i
                new_state = 'I3'
        if current_state == 'I2':
            fire_time_i = -np.log(random.random()) / self.i2_to_r
            if fire_time_i < fire_time:
                fire_time = fire_time_i
                new_state = 'R'
        if current_state == 'I3':
            fire_time_i = -np.log(random.random()) / self.i3_to_r
            if fire_time_i < fire_time:
                fire_time = fire_time_i
                new_state = 'R'


        return fire_time+ self.time, new_state

    def create_event(self, changing_node):
        if self.is_complete_graph:
            return
        G = self.current_state
        event_time, new_state = self.get_changeevent(changing_node)
        G.nodes[changing_node]['changing_time'] = int(event_time*1000000)
        e = (event_time, changing_node, new_state)
        heapq.heappush(self.eventlist, e)
        #if new_state == 'E':
        #    neig_states = [G.nodes[n]['state'] for n in G.neighbors(node_i)]
        #    assert('I1' in neig_states or 'I2' in neig_states or 'I3' in neig_states)

    def fill_eventlist(self):
        G = self.current_state
        self.eventlist = list()
        for node_i in G.nodes():
            self.create_event(node_i)

    def simulation_step_queue(self):
        G = self.current_state
        if self.eventlist is None or self.time == 0.0:
            assert(self.time == 0.0)
            self.fill_eventlist()

        while True:
            if len(self.eventlist) == 0:
                return 100000000.0, 0, 'R'   #dummy to end simulation
            current_event = heapq.heappop(self.eventlist)
            new_time, changing_node, new_state = current_event
            if G.nodes[changing_node]['changing_time'] == int(new_time*1000000): # this is a removed event
                break # reject this event

        return new_time, changing_node, new_state


    def simulation_step_complete(self):
        G = self.current_state
        s_nodes = [n for n in G.nodes() if G.nodes[n]['state'] == 'S']
        s_count = len(s_nodes)
        new_state = 'R'
        changing_node = None
        fire_time = 1000000000.0   #dummy

        for n_i in G.nodes():
            current_state = G.nodes[n_i]['state']
            if current_state == 'I1':
                fire_rate_i = G.nodes[n_i]['infection_rate_1']* s_count
                if fire_rate_i > 0.0:
                    fire_time_i = -np.log(random.random()) / fire_rate_i
                    if fire_time_i < fire_time:
                        fire_time = fire_time_i
                        new_state = 'E'
                        changing_node = random.choice(s_nodes)
            if current_state == 'I2':
                fire_rate_i = G.nodes[n_i]['infection_rate_2']* s_count
                if fire_rate_i > 0.0:
                    fire_time_i = -np.log(random.random()) / fire_rate_i
                    if fire_time_i < fire_time:
                        fire_time = fire_time_i
                        new_state = 'E'
                        changing_node = random.choice(s_nodes)
            if current_state == 'I3':
                fire_rate_i = G.nodes[n_i]['infection_rate_3'] * s_count
                if fire_rate_i > 0.0:
                    fire_time_i = -np.log(random.random()) / fire_rate_i
                    if fire_time_i < fire_time:
                        fire_time = fire_time_i
                        new_state = 'E'
                        changing_node = random.choice(s_nodes)
            if current_state == 'E':
                fire_time_i = -np.log(random.random()) / self.e_to_i1
                if fire_time_i < fire_time:
                    fire_time = fire_time_i
                    new_state = 'I1'
                    changing_node = n_i
            if current_state == 'I1':
                fire_time_i = -np.log(random.random()) / self.i1_to_i2
                if fire_time_i < fire_time:
                    fire_time = fire_time_i
                    new_state = 'I2'
                    changing_node = n_i
            if current_state == 'I1':
                fire_time_i = -np.log(random.random()) / self.i1_to_r
                if fire_time_i < fire_time:
                    fire_time = fire_time_i
                    new_state = 'R'
                    changing_node = n_i
            if current_state == 'I2':
                fire_time_i = -np.log(random.random()) / self.i2_to_i3
                if fire_time_i < fire_time:
                    fire_time = fire_time_i
                    new_state = 'I3'
                    changing_node = n_i
            if current_state == 'I2':
                fire_time_i = -np.log(random.random()) / self.i2_to_r
                if fire_time_i < fire_time:
                    fire_time = fire_time_i
                    new_state = 'R'
                    changing_node = n_i
            if current_state == 'I3':
                fire_time_i = -np.log(random.random()) / self.i3_to_r
                if fire_time_i < fire_time:
                    fire_time = fire_time_i
                    new_state = 'R'
                    changing_node = n_i
        return fire_time+self.time, changing_node, new_state

    def step(self):
        if self.is_complete_graph:
            fire_time, changing_node, new_state = self.simulation_step_complete()
        else:
            fire_time, changing_node, new_state = self.simulation_step_queue()
        if fire_time is  None:
            return None, None, None
        self.time = fire_time

        if len(self.savepoints) == 0:
            raise ValueError('no more timesteps')

        while self.time > self.savepoints[-1]:
            t = self.savepoints.pop()
            self.save(t)
            if len(self.savepoints) == 0:
                return None, None, None

        if len(self.savepoints) == 0:
            return None, None, None

        G = self.current_state
        #apply
        G.nodes[changing_node]['state'] = new_state
        self.create_event(changing_node)
        #if new_state in ['I1', 'I2', 'I3', 'R']:
        for neig_i in G.neighbors(changing_node):
            self.create_event(neig_i)

        if new_state == 'E':
            G.nodes[changing_node]['became_infected_at'] = self.time
            neighbors = list() # works due to Markov property
            weights = list()
            all_neighbors =  G.neighbors(changing_node)
            if self.is_complete_graph:
                all_neighbors = [n for n in G.nodes() if n!= changing_node]
            for neig in all_neighbors:
                if G.nodes[neig]['state']=='I1':
                    neighbors.append(neig)
                    weights.append(G.nodes[neig]['infection_rate_1'])
                elif G.nodes[neig]['state']=='I2':
                    neighbors.append(neig)
                    weights.append(G.nodes[neig]['infection_rate_2'])
                elif G.nodes[neig]['state']=='I3':
                    neighbors.append(neig)
                    weights.append(G.nodes[neig]['infection_rate_3'])
            assert(len(weights) > 0)
            weights_sum = np.sum(weights)
            weights = [w_i/weights_sum for w_i in weights]
            # choose random neighbor or do variance reduction
            if self.variance_reduction:
                for i, neig_i in enumerate(neighbors):
                    G.nodes[neig_i]['num_infected_neighbors'] += weights[i]
            else:
                rand_neig = np.random.choice(neighbors, p=weights)
                G.nodes[rand_neig]['num_infected_neighbors'] += 1

        return fire_time, changing_node, new_state

    def save(self, t=None):
        if t is None:
            t = self.time
        self.record.append((t, copy.deepcopy(self.current_state)))
        G = self.current_state
        for state in self.states():
            self.summary_statistics['time'].append(t)
            self.summary_statistics['state'].append(state)
            self.summary_statistics['count'].append(len([n for n in G.nodes() if G.nodes[n]['state'] == state]))


    def get_summary(self):
        G = self.current_state
        inf_time = [G.nodes[n]['became_infected_at'] for n in G.nodes() if G.nodes[n]['became_infected_at']  is not None]
        inf_count = [G.nodes[n]['num_infected_neighbors'] for n in G.nodes() if G.nodes[n]['became_infected_at']  is not None]
        df_rvalues = pd.DataFrame({'became_infected_at': inf_time, 'num_infected_neighbors': inf_count})
        return pd.DataFrame(self.summary_statistics), df_rvalues


    # ODE

    # has to be a vector in the order of models.states()
    def ode_init(self, init_infected = 3/1000.0):
        # ['E', 'I1', 'I2', 'I3', 'R', 'S']
        init = [0.0, init_infected, 0.0, 0.0, 0.0, 1.0 - init_infected]
        return init

    def ode_func(self, population_vector, t):
        e = population_vector[0]
        i1 = population_vector[1]
        i2 = population_vector[2]
        i3 = population_vector[3]
        r = population_vector[4]
        s = population_vector[5]

        s_to_e_dueto_i1 = 0.394 # for 2.5  0.378 # for 2.4  0.362652 #for 2.3  0.331117 # for 2.1  0.551862 # for 3.50.346884 #for r= 2.2 #0.5# self.infection_rate_1 # needs conversion
        s_to_e_dueto_i2 = s_to_e_dueto_i1/self.rate_damper  # 0.1 #self.infection_rate_2
        s_to_e_dueto_i3 = s_to_e_dueto_i1/self.rate_damper #self.infection_rate_3
        e_to_i1 = self.e_to_i1
        i1_to_r = self.i1_to_r
        i1_to_i2 = self.i1_to_i2
        i2_to_r = self.i2_to_r
        i2_to_i3 = self.i2_to_i3
        i3_to_r = self.i3_to_r


        s_grad = -(
                s_to_e_dueto_i1 * i1 + s_to_e_dueto_i2  * i3 + s_to_e_dueto_i3  * i3) * s
        e_grad = (
                         s_to_e_dueto_i1  * i1 + s_to_e_dueto_i2  * i3 + s_to_e_dueto_i3  * i3) * s - e_to_i1 * e
        i1_grad = e_to_i1 * e - (i1_to_r + i1_to_i2) * i1
        i2_grad = i1_to_i2 * i1 - (i2_to_r + i2_to_i3) * i2
        i3_grad = i2_to_i3 * i2 - i3_to_r * i3
        r_grad = i1_to_r * i1 + i2_to_r * i2 + i3_to_r * i3

        grad = [e_grad, i1_grad, i2_grad, i3_grad, r_grad, s_grad]

        return grad

    def solve_ode(self, outpath):
        plt.clf()
        time_point_samples = sorted(self.savepoints)
        assert(len(time_point_samples) > 0) # call solve ODE before solving network
        from scipy.integrate import odeint
        init = self.ode_init()
        f = self.ode_func
        sol = odeint(f, init, time_point_samples)
        # try:
        #sol = model.aggregate_ode(sol)
        # except:
        #    pass
        np.savetxt(outpath + '.csv', sol)
        traj_map = dict()
        for state_i, state in enumerate(self.states()):
            sol_i = sol[:, state_i]
            try:
                c = self.colors()[state]
            except Exception as e:
                c = None
            traj_map[state] = sol_i
            plt.plot(time_point_samples, sol_i, label=state, c=c, alpha=0.8, lw=2)
        plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
        plt.ylim([0, 1])
        plt.xlim([0, time_point_samples[-1]])
        plt.xlabel('Time', fontsize=42)
        plt.ylabel('Prevalence', fontsize=42)
        plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left", prop={'size': 17})
        plt.xticks(fontsize=33)
        plt.yticks(fontsize=33)
        (plt.gca()).tick_params(axis='both', which='both', length=0)
        plt.savefig(outpath, bbox_inches="tight")
        # plt.show(block=False)
        print('final values of ODE: ', {self.states()[
              i]: sol[-1, i] for i in range(len(self.states()))})

        final_state = {s: v[-1] for s, v in traj_map.items()}
        #max_y = np.max(traj_map['I_total'])
        #max_x = list(traj_map['I_total']).index(max_y)

        #return sol, max_x, max_y, final_state