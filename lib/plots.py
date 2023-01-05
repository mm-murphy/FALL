import matplotlib.pyplot as plt
import os

def plot_0_a(data, path, show=False):
    
    figfile = path+'/broadband_lc.png'
    
    # -- plot broadband
    fig, ax = plt.subplots(figsize=(8,5), nrows=2)
    #      top panel: raw BB
    ax[0].scatter(data.times, data.raw_BBflux, s=2, c='black')
    ax[0].set(xlabel='time', ylabel='image units')
    #      bottom panel: normalized BB
    ax[1].scatter(data.times, data.norm_BBflux, s=2, c='black')
    ax[1].set(xlabel='time', ylabel='rel. brightness')
    plt.savefig(figfile, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()
    
def plot_0_b(data, path, show=False):
    
    for i_wav, wave in enumerate(data.wavelengths):
        figfile = path+'/spectral_lc_'+str(wave)+'.png'
        # -- plot spectral LC
        fig, ax = plt.subplots(figsize=(8,5), nrows=2)
        #      top panel: raw BB
        ax[0].scatter(data.times, data.raw_TSflux[i_wav,:], s=2, c='black')
        ax[0].set(xlabel='time', ylabel='image units')
        #      bottom panel: normalized BB
        ax[1].scatter(data.times, data.norm_TSflux[i_wav,:], s=2, c='black')
        ax[1].set(xlabel='time', ylabel='rel. brightness')
        plt.savefig(figfile, bbox_inches='tight')
        if show:
            plt.show()
        plt.close()