import numpy as np
import sys
from automata_util import *


###############################################################################
############################# SYSTEM RULES ####################################
###############################################################################

def fire(node, system):
    # If state is ready (1) and 2 neighbors in state firing (2), do 1 -> 2
    neighbors = get_neighbors(node, system)
    firing_neighbors = neighbors[np.where(neighbors == 2)]
    if len(firing_neighbors) == 2: return 2
    else: return 1

def rest():
    # If firing (2) -> resting (0)
    return 0

def ready():
    # If resting (0) -> ready (1)
    return 1

def init_system(nodes, random=0):
    if random != 0:
        p = random
        system = np.random.uniform(0.0, 1.0, nodes)
        system[np.where(system > p)] = 1
        system[np.where(system <= p)] = 2
        return system
    elif random == 0:
        system = np.ones(nodes)
        midpoint = (int(np.floor(nodes[0]/2)), int(nodes[1]/2) )
        system[midpoint] = 2
        return system


def update_system(new, old, slice):
    stats = [0,0,0]
    for i in slice[0]:
        for j in slice[1]:
            state = old[i, j]
            if state == 0:
                new[i, j] = ready()
                stats[0] += 1
            elif state == 1:
                new[i, j] = fire((i, j), old)
                stats[1] += 1
            elif state == 2:
                new[i, j] = rest()
                stats[2] += 1
    return new, stats


###############################################################################
############################# INPUT AND EXECUTION #############################
###############################################################################

print("How to use:")
print("*.py brains grid_y grid_x time_steps nproc animate save_animation save_stats p_init_fire")
if len(sys.argv) <= 1:
    N = 1; nodes=(40,40); T=100; nproc=2; animate=1; save=0; show_stats=1; p = [0.3]
else:
    N = int(sys.argv[1]) # brains
    nodes = (int(sys.argv[2]), int(sys.argv[3])) # grid_x, grid_y
    T = int(sys.argv[4]) #time_steps
    nproc = int(sys.argv[5]) # nproc
    animate = int(sys.argv[6])
    save = int(sys.argv[7])
    show_stats = int(sys.argv[8])
    p = [float(sys.argv[9])] # random

nstats = 3
stats = np.zeros((N, T, nstats))
state_info = [[0,1,2], ['Rest', 'Ready', 'Fire'], ['black', 'gold', 'crimson']]

for i in range(N):
    if i%1 == 0:
        print(f'BRAIN {i}')
    system = np.zeros((T, nodes[0], nodes[1]))
    system[0] = init_system(nodes, random=p[0])
    evolve(update_system, system, stats[i], T, nproc)
    if animate == 1 or save == 1:
        animate_and_save(system, state_info, T, show_animation=animate, save=save)

if show_stats == 1:
    avg, var = calculate_stats(stats)
    plot_stats([avg, var], state_info[1], y_label='State dynamics', plot_title=f'p(init_state=fire)={p[0]}')

plt.show()
