import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib as mpl
import multiprocessing as mp
import os, sys, time, warnings
from datetime import datetime
# ffmpeg to save animations
plt.rcParams['animation.ffmpeg_path'] = '/usr/local/bin/ffmpeg'
#warnings.simplefilter('ignore')


def get_neighbors(node, system):
    r = 1 #neighborhood size/radius
    grid_size = system.shape
    neighbors = []
    if system.shape[0]==1:
        for i in range(-r, r+1):
            if (i != 0):
                neighbors.append(system[0][(node[1]+i)%grid_size[1]])
    else:
        for i in range(-r, r+1):
            for j in range(-r, r+1):
                if (i != 0) | (j != 0):
                    neighbors.append(system[(node[0]+i)%grid_size[0]][(node[1]+j)%grid_size[1]])
    return np.array(neighbors)


def evolve(update_func, system, stats, time_steps, nproc=1):
    nodes = system[0].shape
    T = time_steps
    if np.sqrt(nproc)%1 == 0 and np.sqrt(nproc) > nodes[0]:
        chunks_per_dim = int(np.sqrt(nproc))
        rows_per_proc = int(nodes[0]/np.sqrt(nproc))
        cols_per_proc = int(nodes[1]/np.sqrt(nproc))
        slices = []
        rr = nodes[0]%chunks_per_dim
        rc = nodes[1]%chunks_per_dim
        for i in range(chunks_per_dim):
            for j in range(chunks_per_dim):
                if i == chunks_per_dim-1 & j == chunks_per_dim-1:
                    slices.append( [range(rows_per_proc*i, rows_per_proc*(i+1)+rr), range(cols_per_proc*j, cols_per_proc*(j+1)+rc)] )
                elif i == chunks_per_dim-1:
                    slices.append( [range(rows_per_proc*i, rows_per_proc*(i+1)+rr), range(cols_per_proc*j, cols_per_proc*(j+1))] )
                elif j == chunks_per_dim-1:
                    slices.append( [range(rows_per_proc*i, rows_per_proc*(i+1)), range(cols_per_proc*j, cols_per_proc*(j+1)+rc)] )
                else:
                    slices.append( [range(rows_per_proc*i, rows_per_proc*(i+1)), range(cols_per_proc*j, cols_per_proc*(j+1))] )
    else:
        cols_per_proc = int(nodes[1]/nproc)
        slices = []
        rc = nodes[1]%nproc
        for i in range(nproc):
            dt = datetime.now()
            np.random.seed(i+dt.second+dt.microsecond)
            if i == nproc-1:
                slices.append( [range(0, nodes[0]), range(cols_per_proc*i, cols_per_proc*(i+1)+rc)]  )
            else:
                slices.append( [range(0, nodes[0]), range(cols_per_proc*i, cols_per_proc*(i+1))]  )

    if __name__ == '__main__':
        pool = mp.Pool(nproc)
        for t in range(1, T):
            # if t%50 == 0:
            #     print('t =', t)

            res = []
            for i in range(nproc):
                res.append(pool.apply_async(update_func, (system[t], system[t-1], slices[i])))

            for i in range(nproc):
                system[t] += res[i].get(timeout=10)[0]
                stats[t-1] += res[i].get(timeout=10)[1]

        pool.close()
        pool.join()


def animate_and_save(system, state_info, time_steps, show_animation = 0, save=0):
    states = state_info[0]
    state_labels = state_info[1]
    state_colors = state_info[2]
    plt.rcParams["figure.figsize"] = [4.5, 5] #[w,h]
    fig, (ax, cax) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [40, 1]})
    ax.axes.set_title(f'{system.shape[1]}x{system.shape[2]}')
    ax.axes.xaxis.set_ticks([])
    ax.axes.yaxis.set_ticks([])
    fig.subplots_adjust(hspace = 0.001)
    cmap = mpl.colors.ListedColormap(state_colors)
    bounds = states.copy()
    bounds.append(max(states)+1)
    ticks = bounds[1:]
    for i in range(len(ticks)):
        ticks[i] -= 0.5
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    im = ax.imshow(system[0], cmap=cmap, vmin=bounds[0], vmax=bounds[-1])
    cbar = fig.colorbar(im, cax=cax, cmap=cmap, norm=norm, boundaries=bounds, ticks=ticks, orientation='horizontal')
    cbar.ax.set_xticklabels(state_labels)

    T = time_steps-1
    ims = []
    for t in range(T):
        im = ax.imshow(system[t], cmap=cmap, vmin=bounds[0], vmax=bounds[-1])
        ims.append([im])
    ani = animation.ArtistAnimation(fig, ims, interval=100, blit=True, repeat_delay=1500)
    if show_animation == 1:
        plt.show()
    if save == 1:
        dt = datetime.now()
        filename = 'automata-'+str(dt.year)+str(dt.month)+str(dt.day)+'_'+str(dt.hour)+str(dt.minute)+str(dt.second)+'.mp4'
        writer = animation.FFMpegWriter(fps=20, metadata=dict(artist='jonasOlsson'), bitrate=400)
        ani.save(filename, writer=writer)
        print('Output animation: '+filename)
        plt.close(fig)

def calculate_stats(stats):
    T = stats.shape[1]
    nstats = stats.shape[2]
    stats_avg = np.zeros((T, nstats))
    stats_var = np.zeros((T, nstats))
    for i in range(T):
        stats_avg[i] = np.mean(stats[:, i], axis=0)
        stats_var[i] = np.var(stats[:, i], axis=0)
    return stats_avg, stats_var


def plot_stats(stats, state_labels, y_label='y', plot_title='automata_state_dynamics'):
    stats_avg = stats[0]
    stats_var = stats[1]
    T = stats_avg.shape[0]
    nstats = stats_avg.shape[1]

    #mpl.rcParams.update(mpl.rcParamsDefault)
    plt.rcParams["figure.figsize"] = [10, 4] #[w,h]
    fig, (ax1, ax2) = plt.subplots(1, 2)
    for i in range(nstats):
        ax1.plot(np.linspace(0, T, T-1), stats_avg[0:-1, i], label=state_labels[i])
        ax2.plot(np.linspace(0, T, T-1), np.sqrt(stats_var[0:-1, i]), label=state_labels[i])
    ax1.set_title(plot_title)
    ax1.set_xlabel('Time')
    ax1.set_ylabel(y_label)
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Std. deviation')
    ax1.legend()
    ax2.legend()
    dt = datetime.now()
    filename = 'automata-stats-'+str(dt.year)+str(dt.month)+str(dt.day)+'_'+str(dt.hour)+str(dt.minute)+str(dt.second)+'.png'
    plt.savefig(filename)
    print('Output stat_plot: '+filename)
