import os
import re
from typing import List, Tuple

import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None

import fastText
# from flair.models import TextClassifier
# from flair.data import Sentence
# from textblob import TextBlob

from preprocess import IdentifyNews, ExtractContent

# Paths to pretrained sentiment models
fasttext_model = os.path.join('./models', 'fasttext_yelp_review_full.ftz')
flair_model = os.path.join('./models', 'final-model.pt')

def make_dirs(dirpath: str) -> None:
    """Make directories for output if necessary"""
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)

class FastTextSentiment:
    """Class to predict aggregated sentiment on content towards a particular target
    using FastText. The FastText library for Python can be installed using pip
    through this GitHub reporitory: https://github.com/facebookresearch/fastText/tree/master/python
    """
    def __init__(self, news: pd.DataFrame, model_path: str) -> None:
        self.news = news
        try:
            fastText.load_model(model_path)
        except ValueError:
            raise Exception("Could not find FastText model in {}".format(model_path))
        self.model = fastText.load_model(model_path)

    def tokenize(self, sentence: str) -> str:
        "Lowercase sentence and tokenize according to FastText requirements"
        sentence = sentence.lower()
        tokenized_sentence = re.sub(r"([.!?,'/()])", r" \1 ", sentence)
        return tokenized_sentence

    def get_sentiment(self, sentences: List[str]) -> Tuple[float, float]:
        "Calculate the mean sentiment score and standatd deviation for a list of sentences"
        tokenized = list(map(self.tokenize, sentences))
        labels, probabilities = self.model.predict(tokenized, 1)   # Predict just the top label, hence 1
        # The trained model predicts score in [1, 5] - we convert it to a scale [-1, +1]
        sentiment_scores = [(int(l[0][-1]) - 3)/2 if l else 0 for l in labels]
        score = np.mean(sentiment_scores)
        deviation = np.std(sentiment_scores)
        return score, deviation

    def score(self) -> pd.DataFrame:
        "Return a DataFrame with sentiment score and standard deviation columns added"
        self.news['score'], self.news['deviation'] = zip(*self.news['relevant'].map(self.get_sentiment))
        return self.news


class PostProcess:
    "Class to perform data organization and postprocessing on the sentiment analysis DataFrame"
    def __init__(self, news: pd.DataFrame, name: str, write_data: bool=True) -> None:
        self.news = news
        self.name = name
        self.write_ = write_data

    def polarity(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        "Return positive and negative sentiment scoring content on the existing DataFrame"
        df = self.news
        pos = df[df['score'] > 0.0].sort_values(by=['score'], ascending=False).reset_index(drop=True)
        neg = df[df['score'] < 0.0].sort_values(by=['score']).reset_index(drop=True)
        # Write data (optional)
        if self.write_:
            out_path = "./results/fasttext"
            make_dirs(out_path)
            file_prefix = '_'.join(self.name.split()).lower()
            # Write positive content
            pos_filename = file_prefix + '_pos.csv'
            pos_path = os.path.join(out_path, pos_filename)
            pos.sort_values(by='date').to_csv(pos_path, index=False, header=True)
            # Write negative content
            neg_filename = file_prefix + '_neg.csv'
            neg_path = os.path.join(out_path, neg_filename)
            neg.sort_values(by='date').to_csv(neg_path, index=False, header=True)
        return pos, neg

    def peak_polar(self) -> pd.DataFrame:
        "Store the peak absolute polarity (to identify most polar content, positive or negative)"
        self.news['abs'] = self.news['score'].abs()
        news_peak_polar = self.news.groupby('date').max()[['title', 'publication', 'relevant']]
        # Extract just the first 3 relevant sentences from the article and convert to single string
        news_peak_polar['relevant'] = news_peak_polar['relevant'].apply(lambda x: x[:3]).str.join(' ')
        # Concatenate scores/counts DataFrame with most polar news content for that day
        return news_peak_polar

    def aggregate(self) -> pd.DataFrame:
        "Get combined average scores, deviations and article counts per day"
        news_avg_score = self.news.groupby('date')['score'].mean()
        news_avg_dev = self.news.groupby('date')['deviation'].mean()
        news_count = self.news.groupby(['date']).count()['title']
        aggregated = pd.concat((news_avg_score, news_avg_dev, 
                                news_count), axis=1).sort_values(by=['date'])
        aggregated.columns = ['mean_score', 'mean_dev', 'count']
        return aggregated

    def daily_data(self) -> pd.DataFrame:
        """Return a DataFrame that contains just the following columns (indexed by date)
        ['title', 'publication', 'relevant', 'mean_score', 'mean_dev', 'count']
        """
        peak_polar = self.peak_polar()
        aggregated = self.aggregate()
        data = pd.concat((peak_polar, aggregated), axis=1).sort_index()
        # Reindex to get daily data (fill non-existent rows with zeros)
        idx = pd.date_range('1/1/2014', '7/5/2017')
        daily = data.reindex(idx, fill_value=0.0)
        if self.write_:
            out_path = "./results/fasttext"
            make_dirs(out_path)
            file_prefix = '_'.join(self.name.split()).lower()
            out_filename = file_prefix + '_data.csv'
            out_path = os.path.join(out_path, out_filename)
            daily[~daily['relevant'].eq(0)].to_csv(out_path, header=True)
        return daily

    def breakdown(self) -> pd.DataFrame:
        "Get counts of positive and negative mentions based on Publication"
        brk = self.news.groupby('publication').apply(lambda x: x['score'] >= 0.0)
        brk = brk.groupby('publication').value_counts().to_frame()
        brk = brk.unstack().fillna(0.0)
        brk.columns = ['negative', 'positive']
        brk = brk.sort_values(by='negative')
        if self.write_:
            out_path = "./results/fasttext"
            make_dirs(out_path)
            file_prefix = '_'.join(self.name.split()).lower()
            out_filename = file_prefix + '_breakdown.csv'
            out_path = os.path.join(out_path, out_filename)
            brk.to_csv(out_path, header=True)
        return brk

    def get_all(self) -> None:
        "Run all utilities to write out sentiment data to csv"
        _ = self.polarity()
        _ = self.daily_data()
        _ = self.breakdown()


if __name__ == "__main__":
    data_path = os.path.join('../data', 'all_the_news_v2.csv')
    name = "Ryan Lochte"
    nw = IdentifyNews(data_path, name)
    news = nw.get()
    # print(news.head(3))
    print("Extracted {} articles that mention {}".format(news.shape[0], name))
    ex = ExtractContent(news, name)
    news_relevant = ex.extract()
    # print(news_relevant[['relevant', 'lemmas']].head(3))
    print("Removed duplicates and extracted relevant sentences from {} articles".format(news_relevant.shape[0], name))

    ft = FastTextSentiment(news_relevant, fasttext_model)
    df_scores = ft.score()
    data = PostProcess(df_scores, name, write_data=True)
    data.get_all()
    print("Success!! Wrote out positive, negative, daily aggregate and publication breakdown data for {}.".format(name))
