"Sentiment Analysis script to calculate sentiment scores over time and output visualizations"
from preprocess import IdentifyNews, ExtractContent
from sentiment import TextBlobSentiment, FastTextSentiment, FlairSentiment, SentimentAnalyzer
from visualize import make_plots

import argparse
import os
import pandas as pd

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

def analyze(news: pd.DataFrame, name_query: str, model_path: str, method: str) -> None:
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

    # Analyze sentiment
    # TODO: Make the date window an argument to be able to query different time periods
    date_window = ('1/1/2014', '7/5/2017')
    data = SentimentAnalyzer(df_scores, name_query, method=method,
                             window=date_window, write_data=True)
    data.get_all()

def plot_results(result_path: str, name_query: str, method: str, write_: bool):
    "Generate plots showing the sentiment over time for each target"
    out_path = os.path.join(result_path, method)
    write_ = True
    image_path = os.path.join(out_path, "plots")
    # make plots
    make_plots(out_path, name_query, image_path, write_)

def main(news: pd.DataFrame, name_query: str, method: str, model_path: str, 
         write_: bool, result_path: str) -> None:
    news = reduce_news(news, name_query)
    news_relevant = extract_content(news, name_query)
    analyze(news_relevant, name_query, model_path, method)
    plot_results(result_path, name_query, method, write_)


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
    parser.add_argument('-w', '--write', type=bool, help='Boolean flag to choose whether or not to write output data',
                        default=True)
    args = parser.parse_args()
    input_data = args.inp
    method = args.method
    name_query = args.name
    result_path = args.results
    write_ = args.write
    model_path = args.modelfile

    # Read in dataset
    news = all_the_news(input_data)

    # Run
    for query in name_query:
        main(news, query, method, model_path, write_, result_path)
        print("Success!! Wrote out positive, negative, daily aggregate and publication breakdown data for {}.\n".format(query))
    print("Done... Completed analysis.")
