import numpy as np
import sys
from automata_util import *


###############################################################################
############################# SYSTEM RULES ####################################
###############################################################################

def infect(node, system):
    # If state is susceptible (1) and at least 1 neighbor is infected do 1 -p-> 2
    # Here p is 100%
    p = np.random.uniform(0.0, 1.0)
    neighbors = get_neighbors(node, system)
    infected_neighbors = neighbors[np.where(neighbors == 1)]
    if (len(infected_neighbors) > 0) & (p <= 1-gamma): return 1
    else: return 0


def recover():
    # If infected (2) recover with prob. p=gamma, otherwise stay infected
    p = np.random.uniform(0.0, 1.0)
    if p <= gamma: return 0
    else: return 1


def init_system(nodes, random=0):
    if random != 0:
        p = random
        system = np.random.uniform(0.0, 1.0, nodes)
        system[np.where(system <= p)] = 1.0
        system[np.where(system != 1.0)] = 0
        return system
    elif random == 0 :
        system = np.zeros(nodes)
        midpoint = (int(np.floor(nodes[0]/2)), int(nodes[1]/2) )
        system[midpoint] = 1
        return system


def update_system(new, old, slice):
    stats = [0,0]
    for i in slice[0]:
        for j in slice[1]:
            state = old[i, j]
            if state == 0:
                new[i, j] = infect((i, j), old)
                stats[0] += 1
            elif state == 1:
                new[i, j] = recover()
                stats[1] += 1
    return new, stats


###############################################################################
############################# INPUT AND EXECUTION #############################
###############################################################################

if len(sys.argv) <= 1:
    print("You need to specify what you want to do:")
    print("*.py population grid_x grid_y time_steps nproc animate save_animation show_stats gamma random")
    sys.exit(0)

N = int(sys.argv[1]) # brains
nodes = (int(sys.argv[2]), int(sys.argv[3])) # grid_x, grid_y
T = int(sys.argv[4]) #time_steps
nproc = int(sys.argv[5]) # nproc
animate = int(sys.argv[6])
save = int(sys.argv[7])
show_stats = int(sys.argv[8])
gammas = [float(sys.argv[9])] # gamma
prange = [float(sys.argv[10])] # random

#gammas = np.linspace(0.3, 0.6, 20)
#gammas = [0.3, 0.4, 0.5, 0.6]
#prange = np.linspace(0.001, 0.05, 10)
#prange = [0.3]
prob = np.zeros((len(prange), len(gammas)))

nstats = 2
state_info = [[0,1], ['Susceptible', 'Infected'], ['black', 'gold']]

infected_vs_gamma = np.zeros(len(gammas))
for p in range(prob.shape[0]):
    if p%1 == 0:
        print(f'p {prange[p]}')
    for g in range(len(gammas)):
        if g%1 == 0:
            print(f'gamma {gammas[g]}')
        gamma = gammas[g]
        stats = np.zeros((N, T, nstats))
        for i in range(N):
            # if i%1 == 0:
            #     print(f'POPULATION {i}')
            system = np.zeros((T, nodes[0], nodes[1]))
            system[0] = init_system(nodes, random=prange[p])
            evolve(update_system, system, stats[i], T, nproc)
            if animate == 1 or save == 1:
                animate_and_save(system, state_info, T, show_animation=animate, save=save)

            avg, var = calculate_stats(stats)
            if show_stats == 1:
                plot_stats([avg, var], state_info[1], 'State dynamics')

        infected_vs_gamma[g] = np.mean(avg[int(T/2):, 1])
    prob[p] = infected_vs_gamma

plt.figure()
for i in range(prob.shape[0]):
    plt.plot(gammas, prob[i]/(nodes[0]*nodes[1])*100, '-*', label=f'p={np.round(prange[i], 3)}')
plt.xlabel('Gamma')
plt.ylabel('% Infected after steady state')
plt.title(f'{N} simulations, {T} time_steps, {nodes[0]}x{nodes[1]} population')
plt.legend()
plt.show()
