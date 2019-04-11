"Sentiment Analysis script to calculate sentiment scores over time and output visualizations"
from preprocess import IdentifyNews, ExtractContent
from sentiment import TextBlobSentiment, FastTextSentiment, FlairSentiment, SentimentAnalyzer
from visualize import bar_timeline, heatmap_calendar, bar_breakdown, scatter_cosine_dist

import argparse
import os
import pandas as pd
from typing import Tuple


def make_dirs(dirpath: str) -> None:
    """Make directories for output if necessary"""
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)

def clean_dirs(dirpath: str) -> None:
    "Clean up results directory before run"
    existing = [os.path.join(dirpath, f) for f in os.listdir(dirpath)]
    for f in existing:
        if os.path.isfile(f):
            os.remove(f)

def get_filepath(path: str, suffix: str, filetype: str='csv') -> str:
    "Obtain relative path to data based on specified suffix string"
    suffix_string = "{}.{}".format(suffix, filetype)
    filepath = os.path.join(path, suffix_string)
    return filepath

def str2bool(v):
    "Parse boolean inputs from argparse correctly"
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def all_the_news(data_path: str) -> pd.DataFrame:
    """Read in "All The News" dataset (downloaded from this source:
    https://components.one/datasets/all-the-news-articles-dataset/)
    """
    colnames = ['title', 'author', 'date', 'content', 'year', 'month', 'publication', 'length']
    news = pd.read_csv(data_path, usecols=colnames, parse_dates=['date'])
    news['author'] = news['author'].str.strip()
    news = news.dropna(subset=['date', 'title'])    # Drop empty rows from these columns
    print("Retrieved {} news articles between the dates {} and {}.".format(news.shape[0],
                                                                           news['date'].min(),
                                                                           news['date'].max()))
    return news

def reduce_news(data_path: str, name_query: str) -> pd.DataFrame:
    "Reduce the full news dataset to only those articles that mention our target name query"
    nw = IdentifyNews(data_path, name_query)
    news = nw.get()
    print("Extracted {} articles that mention {}".format(news.shape[0], name_query))
    return news

def extract_content(news: pd.DataFrame, name_query: str) -> pd.DataFrame:
    "Extract only those sentences from each article that explicitly mention our target name query"
    ex = ExtractContent(news, name_query)
    news_relevant = ex.extract()
    print("Removed duplicates and extracted relevant sentences from {} articles".format(news_relevant.shape[0], name_query))
    return news_relevant

def analyze(news: pd.DataFrame, name_query: str, model_path: str, result_path: str, method: str,
            date_window: tuple) -> None:
    "Perform sentiment analysis on the extracted content using a specific method"
    if method == 'textblob':
        tb = TextBlobSentiment(news)
        df_scores = tb.score()
    elif method == 'fasttext':
        ft = FastTextSentiment(news, model_path)
        df_scores = ft.score()
    elif method == 'flair':
        fl = FlairSentiment(news, model_path)
        df_scores = fl.score()
    else:
        raise Exception("The requested method for sentiment analysis has not yet been implemented!") 

    make_dirs(result_path)  # Create result output directory
    # Analyze sentiment
    data = SentimentAnalyzer(df_scores, name_query, result_path, method,
                             date_window)
    data.get_all()

def read_df(news_path: str, polarity_path: str,
            cosine_dist_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    "Read data from sentiment analysis and store in memory for plotting"
    data = pd.read_csv(news_path, header=None, index_col=0, parse_dates=True)
    data.columns = ['title', 'publication', 'relevant', 'mean_score', 'mean_dev', 'count', 'query']
    # Read in polarity breakdown (pos/neg) per publication
    polarity_df = pd.read_csv(polarity_path, header=None, index_col=0)
    polarity_df.columns = ['Negative', 'Positive', 'query']
    # Read in mean cosine distance per publication
    cosine_df = pd.read_csv(cosine_dist_path, header=None, index_col=0)
    cosine_df.columns = ['count', 'x', 'y', 'query']
    return data, polarity_df, cosine_df

def make_plots(data: pd.DataFrame, polarity_df: pd.DataFrame, cosine_df: pd.DataFrame,
               out_path: str, name_query: str, method: str, date_window: tuple, write_: bool):
    "Generate plots showing the sentiment over time for each target"
    # Generate paths
    image_path = os.path.join(out_path, "plots")
    make_dirs(image_path)    # Create image output directory
    # Call plotting functions
    bar_timeline(data, name_query, image_path, date_window, write_)
    heatmap_calendar(data, name_query, image_path, date_window, write_)
    bar_breakdown(polarity_df, name_query, image_path, write_)
    scatter_cosine_dist(cosine_df, name_query, image_path, write_)

def run_analysis(news: pd.DataFrame, name_query: str, method: str, model_path: str, 
                 result_path: str, date_window: tuple) -> None:
    news = reduce_news(news, name_query)
    news_relevant = extract_content(news, name_query)
    analyze(news_relevant, name_query, model_path, result_path, method, date_window)
    # Generate paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Perform targeted sentiment analysis on a \
                                     particular entity in a large news dataset using NLP.')
    parser.add_argument('-i', '--inp', type=str, help='Path to news dataset (csv or similar)',
                        default='../data/all_the_news_v2.csv')
    parser.add_argument('-m', '--method', type=str, help='Sentiment analysis model (textblob, fasttext or flair)',
                        default="fasttext")
    parser.add_argument('-n', '--name', type=str, help='Name query (e.g. name of a person/organization)',
                        required=True, nargs='+')
    parser.add_argument('-f', '--modelfile', type=str, help='Path to trained classifier model for fasttext or flair',
                        default='./models/fasttext_yelp_review_full.ftz')
    parser.add_argument('-r', '--results', type=str, help='Path to output result data and plots',
                        default='./results')
    parser.add_argument('-w', '--write', type=str2bool, help="Boolean flag in ('yes', 'true', 't', 'y', '1') \
                        and its correponding negation to choose whether or not to write out image files",
                        default=True)
    args = parser.parse_args()
    input_data = args.inp
    method = args.method
    name_query = args.name
    result_path = args.results
    write_ = bool(args.write)
    model_path = args.modelfile

    # Read in dataset
    news = all_the_news(input_data)
    make_dirs(os.path.join(result_path, method))
    clean_dirs(os.path.join(result_path, method))
    # Specify date window to resample data on a daily basis
    date_window = ('1/1/2014', '7/5/2017')

    # Run
    for query in name_query:
        run_analysis(news, query, method, model_path, result_path, date_window)
        print("Success!! Wrote out positive, negative, daily aggregate and publication breakdown data for {}.\n".format(query))
    # Once analysis results are generated, read results for plotting
    out_path = os.path.join(result_path, method)
    news_path = get_filepath(out_path, 'data', 'csv')
    polarity_path = get_filepath(out_path, 'breakdown', 'csv')
    cosine_dist_path = get_filepath(out_path, 'cosine_dist', 'csv')
    data, polarity_df, cosine_df = read_df(news_path, polarity_path, cosine_dist_path)
    # Generate plots for each case
    for query in name_query:
        make_plots(data, polarity_df, cosine_df, out_path, query, method, date_window, write_)
        if write_:
            print("Output plots for {}.".format(query))
    
    print("\nDone... Completed analysis.")
