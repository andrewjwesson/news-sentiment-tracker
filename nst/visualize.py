import os
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

import calmap  # for making GitHub-style calendar plots of time-series
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters  # Plot using Pandas datatime objects
register_matplotlib_converters()
rc_fonts = {'figure.figsize': (15, 8),
            'axes.labelsize': 18,
            'xtick.labelsize': 18,
            'ytick.labelsize': 18,
            'legend.fontsize': 16}
plt.rcParams.update(rc_fonts)
plt.style.use('ggplot')


def bar_timeline(daily_data: pd.DataFrame, name_query: str, image_path: str,
                 window: tuple, write_: bool) -> None:
    """Bar chart showing the timeline of average positive and negative scores and deviation
    """
    normalized_query = '_'.join(name_query.split()).lower()
    daily_data = daily_data[daily_data['query'] == normalized_query]  # Run one query at a time
    # Resample data daily and fill non-existent values with zeroes
    idx = pd.date_range(window[0], window[1])
    daily_data = daily_data.reindex(idx, fill_value=0.0)
    # Generate plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 9))
    ax1.fill_between(daily_data.index, daily_data['mean_score'], step='mid', color='black', alpha=0.6, linewidth=4)
    ax1.set_ylabel('Mean Score')
    # Initiate a second y-axis with a shared x-axis for the article counts
    ax2_2 = ax2.twinx()
    ax2_2.plot(daily_data.index, daily_data['count'], 'r--', alpha=0.6, linewidth=2)
    ax2_2.grid(False)
    ax2_2.set_ylabel('Article Count')
    # ax1.set_title('Sentiment scores and deviations with time for "{}"'.format(name_query), size=15)
    ax2.fill_between(daily_data.index, daily_data['mean_dev'], step='mid', color='black', alpha=0.6, linewidth=4)
    ax2.set_ylabel('Mean Deviation')
    ax2.set_xlabel('Date')
    fig.patch.set_facecolor('ghostwhite')
    plt.tight_layout()
    if write_:
        plt.savefig(os.path.join(image_path, "bt_{}.png".format('_'.join(name_query.split()).lower())))


def heatmap_calendar(daily_data: pd.DataFrame, name_query: str, image_path: str,
                     window: tuple, write_: bool) -> None:
    """Generate a calendar heatmap of sentiment scores on each calendar day
    See: https://pythonhosted.org/calmap/
    """
    normalized_query = '_'.join(name_query.split()).lower()
    daily_data = daily_data[daily_data['query'] == normalized_query]  # Run one query at a time
    # Resample data daily and fill non-existent values with zeroes
    idx = pd.date_range(window[0], window[1])
    daily_data = daily_data.reindex(idx, fill_value=0.0)
    fig, axes = calmap.calendarplot(daily_data['mean_score'],
                                    vmin=-1.0,
                                    vmax=1.0,
                                    daylabels='MTWTFSS',
                                    dayticks=[0, 2, 4, 6],
                                    fig_kws=dict(figsize=(15, 9)),
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


def bar_breakdown(polarity_df: pd.DataFrame, name_query: str, image_path: str, write_: bool) -> None:
    "Plot breakdown of positive and negative articles towards the article per publication"
    normalized_query = '_'.join(name_query.split()).lower()
    polarity_df = polarity_df[polarity_df['query'] == normalized_query]
    # Make bar chart
    fig, ax = plt.subplots(figsize=(15, 9))
    polarity_df.plot(kind='barh')
    fig.patch.set_facecolor('ghostwhite')
    plt.tight_layout()
    plt.ylabel('')
    if write_:
        plt.savefig(os.path.join(image_path, "breakdown_{}.png".format('_'.join(name_query.split()).lower())), 
                    facecolor=fig.get_facecolor())


def scatter_cosine_dist(cosine_df: pd.DataFrame, name_query: str, image_path: str, write_: bool) -> None:
    "Plot mean cosine distance per publication from the multi-dimensional scaling approach"
    normalized_query = '_'.join(name_query.split()).lower()
    cosine_df = cosine_df[cosine_df['query'] == normalized_query]
    fig, ax = plt.subplots(figsize=(15, 9))
    # Pick color indices
    colors = [i for i in range(len(cosine_df.index))]
    ax.scatter(cosine_df['x'], cosine_df['y'], c=colors,
               s=cosine_df['count']*100, linewidths=1.5, alpha=0.7,
               edgecolors='k', cmap=plt.cm.gist_rainbow,
               )
    # Annotate points
    for i, txt in enumerate(cosine_df.index):
        ax.annotate(txt, (cosine_df['x'][i], cosine_df['y'][i]),
                    fontsize=18, alpha=0.7)
    ax.set_xticklabels([''])
    ax.set_yticklabels([''])
    plt.tight_layout()
    if write_:
        plt.savefig(os.path.join(image_path, "cosine_dist_{}.png".format('_'.join(name_query.split()).lower())), 
                    facecolor=fig.get_facecolor())