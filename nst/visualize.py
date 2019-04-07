import os
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

import calmap  # for making GitHub-style calendar plots of time-series
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters  # Plot using Pandas datatime objects
register_matplotlib_converters()
rc_fonts = {'figure.figsize': (15, 8),
            'axes.labelsize': 16,
            'xtick.labelsize': 14,
            'ytick.labelsize': 14,
            'legend.fontsize': 16}
plt.rcParams.update(rc_fonts)
plt.style.use('ggplot')

def make_dirs(dirpath: str) -> None:
    """Make directories for output if necessary"""
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)

def get_filepath(path: str, name_query: str, suffix: str, filetype: str='csv') -> str:
    "Obtain relative path to data for specific target indicated by the 'name_query' variable"
    suffix_string = "_{}.{}".format(suffix, filetype)
    filepath = os.path.join(path, '_'.join(name_query.split()).lower() + suffix_string)
    return filepath

def bar_timeline(file_path: str, name_query: str, image_path: str, write_: bool) -> pd.DataFrame:
    """Bar chart showing the timeline of average positive and negative scores and deviation
    """
    daily = pd.read_csv(file_path, index_col=0, parse_dates=True)
    # Reindex data daily
    idx = pd.date_range('1/1/2014', '7/5/2017')
    daily = daily.reindex(idx, fill_value=0.0)
    # Generate plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8))
    ax1.fill_between(daily.index, daily['mean_score'], step='mid', color='black', alpha=0.6, linewidth=4)
    ax1.set_ylabel('Mean Score')
    # ax1.set_title('Sentiment scores and deviations with time for "{}"'.format(name_query), size=15)
    ax2.fill_between(daily.index, daily['mean_dev'], step='mid', color='black', alpha=0.6, linewidth=4)
    ax2.set_ylabel('Mean Deviation')
    ax2.set_xlabel('Date')
    fig.patch.set_facecolor('ghostwhite')
    plt.tight_layout()
    if write_:
        plt.savefig(os.path.join(image_path, "bt_{}.png".format('_'.join(name_query.split()).lower())))
    return daily


def calendar_map(daily_data: pd.DataFrame, name_query: str, image_path: str, write_: bool) -> None:
    """Generate a calendar heatmap of sentiment scores on each calendar day
    See: https://pythonhosted.org/calmap/
    """
    fig, axes = calmap.calendarplot(daily_data['mean_score'],
                                    vmin=-1.0,
                                    vmax=1.0,
                                    daylabels='MTWTFSS',
                                    dayticks=[0, 2, 4, 6],
                                    fig_kws=dict(figsize=(12.5, 9)),
                                    linewidth=1,
                                    fillcolor='lightgrey',
                                    cmap='coolwarm_r',
                                    )
    # fig.suptitle("Calendar map of aggregated sentiment for {}".format(name_query), fontsize=18)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.patch.set_facecolor('ghostwhite')
    if write_:
        plt.savefig(os.path.join(image_path, "calmap_{}.png".format('_'.join(name_query.split()).lower())), 
                    facecolor=fig.get_facecolor())


def bar_breakdown(file_path: str, name_query: str, image_path: str, write_: bool) -> None:
    "Plot breakdown of positive and negative articles towards the article per publication"
    bd = pd.read_csv(file_path, index_col=0)
    # Make bar chart
    fig, ax = plt.subplots(1, 1, figsize=(12.5, 9))
    bd.plot(kind='barh')
    fig.patch.set_facecolor('ghostwhite')
    plt.tight_layout()
    if write_:
        plt.savefig(os.path.join(image_path, "breakdown_{}.png".format('_'.join(name_query.split()).lower())), 
                    facecolor=fig.get_facecolor())


def scatter_cosine_dist(file_path: str, name_query: str, image_path: str, write_: bool) -> None:
    "Plot mean cosine distance per publication from the multi-dimensional scaling approach"
    groups = pd.read_csv(file_path, index_col=0)
    fig, ax = plt.subplots(figsize=(12, 9))
    # Pick color indices
    colors = [i for i in range(len(groups.index))]
    ax.scatter(groups['x'], groups['y'], c=colors,
               s=groups['count']*100, linewidths=1.5, alpha=0.7,
               edgecolors='k', cmap=plt.cm.gist_rainbow,
               )
    # Annotate points
    for i, txt in enumerate(groups.index):
        ax.annotate(txt, (groups['x'][i], groups['y'][i]),
                    fontsize=18, alpha=0.7)
    ax.set_xticklabels([''])
    ax.set_yticklabels([''])
    plt.tight_layout()
    if write_:
        plt.savefig(os.path.join(image_path, "cosine_dist_{}.png".format('_'.join(name_query.split()).lower())), 
                    facecolor=fig.get_facecolor())


def make_plots(result_path: str, name_query: str, image_path: str, write_: bool=True) -> None:
    "Write all plots as PNG files"
    make_dirs(image_path)    # Create image output directory

    fpath = get_filepath(result_path, name_query, 'data', 'csv')
    # Get daily bar timeline of pos/neg scores and deviations
    daily = bar_timeline(fpath, name_query, image_path, write_)
    # Get calendar map of sentiment scores
    calendar_map(daily, name_query, image_path, write_)
    # Get breakdown of pos/neg articles per publication
    fpath = get_filepath(result_path, name_query, 'breakdown', 'csv')
    bar_breakdown(fpath, name_query, image_path, write_)
    # Get breakdown of pos/neg articles per publication
    fpath = get_filepath(result_path, name_query, 'cosine_dist', 'csv')
    scatter_cosine_dist(fpath, name_query, image_path, write_)
