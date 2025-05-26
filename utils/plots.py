import matplotlib.pylab as plt
import matplotlib
matplotlib.use("Agg")


def plot_spectrogram(spectrogram):
    fig, ax = plt.subplots(figsize=(10, 2))
    im = ax.imshow(spectrogram.T, aspect="auto", origin="lower",
                   interpolation='none')
    plt.colorbar(im, ax=ax)
    fig.canvas.draw()
    plt.close()
    return fig


def plot_attn(sw, step, attns, prefix, figsize):
    # Attns: Sequence of layers of size (nheads, lq, lk)
    nheads = attns[0].size(0)
    fig, axs = plt.subplots(len(attns), nheads,
                            constrained_layout=True, figsize=figsize)
    if len(attns) == 1 and nheads == 1:
        axs = [[axs]]
    elif len(attns) == 1 or nheads == 1:
        axs = [axs]
    for i, attn in enumerate(attns):  # Each layers
        attn = attn.float().cpu().numpy()
        for j, head_attn in enumerate(attn):
            axs[i][j].matshow(head_attn, aspect="auto",
                              origin="lower", interpolation='none',
                              vmin=0.0)
            if i != 0 or j != 0:
                axs[i][j].axis('off')
    sw.add_figure(prefix, fig, step)
    plt.close()
