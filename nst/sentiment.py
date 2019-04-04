import os
import re
from typing import List, Tuple

import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None

import fastText
from flair.models import TextClassifier
from flair.data import Sentence
from textblob import TextBlob
from preprocess import IdentifyNews, ExtractContent

# Paths to pretrained sentiment models
fasttext_model = os.path.join('./models', 'fasttext_yelp_review_full.ftz')
flair_model = os.path.join('./models', 'final-model.pt')

def make_dirs(dirpath: str) -> None:
    """Make directories for output if necessary"""
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)


class TextBlobSentiment:
    """Class to predict aggregated sentiment on content towards a particular target
    using TextBlob. Unlike other approaches, this approach does not require a trained 
    model - TextBlob calculates a 'polarity metric' on each individual word, which is 
    used to calculate the overall sentiment of a sequence.
    More details here: https://planspace.org/20150607-textblob_sentiment/
    """
    def __init__(self, news: pd.DataFrame) -> None:
        self.news = news

    def get_sentiment(self, sentences: List[str]) -> Tuple[float, float]:
        "Calculate the mean sentiment score and standatd deviation for a list of sentences"
        sentiment_scores = [round(TextBlob(sentence).sentiment.polarity, 4) for sentence in sentences]
        score = np.mean(sentiment_scores)
        deviation = np.std(sentiment_scores)
        return score, deviation

    def score(self) -> pd.DataFrame:
        "Return a DataFrame with sentiment score and standard deviation columns added"
        self.news['score'], self.news['deviation'] = zip(*self.news['relevant'].map(self.get_sentiment))
        return self.news


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
        sentiment_scores = [(int(label[0][-1]) - 3)/2 if label else 0 for label in labels]
        score = np.mean(sentiment_scores)
        deviation = np.std(sentiment_scores)
        return score, deviation

    def score(self) -> pd.DataFrame:
        "Return a DataFrame with sentiment score and standard deviation columns added"
        self.news['score'], self.news['deviation'] = zip(*self.news['relevant'].map(self.get_sentiment))
        return self.news


class FlairSentiment:
    """Class to predict aggregated sentiment on content towards a particular target
    using the Flair NLP library. Flair is a powerful PyTorch-based NLP framework that 
    utilizes contextualized string embeddings (i.e. 'Flair' embeddings) to perform NLP 
    tasks, including classification. 
    More information here: https://github.com/zalandoresearch/flair
    """
    def __init__(self, news: pd.DataFrame, model_path: str) -> None:
        self.news = news
        try:
            TextClassifier.load_from_file(model_path)
        except ValueError:
            raise Exception("Could not find Flair classification model in {}".format(model_path))
        self.model = TextClassifier.load_from_file(model_path)

    def get_sentiment(self, sentences: List[str]) -> Tuple[float, float]:
        "Calculate the mean sentiment score and standatd deviation for a list of sentences"
        scores = []
        for item in sentences:
            sentence = Sentence(item)
            self.model.predict(sentence)
            scores.append(int(sentence.labels[0].value))
        sentiment_list = [(score-3)/2 if score else 0 for score in scores]
        score = np.mean(sentiment_list)
        deviation = np.std(sentiment_list)
        return score, deviation

    def score(self) -> pd.DataFrame:
        "Return a DataFrame with sentiment score and standard deviation columns added"
        self.news['score'], self.news['deviation'] = zip(*self.news['relevant'].map(self.get_sentiment))
        return self.news


class SentimentAnalyzer:
    "Class to perform data organization and postprocessing on the sentiment analysis DataFrame"
    def __init__(self, news: pd.DataFrame, name: str, method: str, write_data: bool=True) -> None:
        self.news = news
        self.name = name
        self.method = method
        self.write_ = write_data

    def polarity(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        "Return positive and negative sentiment scoring content on the existing DataFrame"
        df = self.news
        pos = df[df['score'] > 0.0].sort_values(by=['score'], ascending=False).reset_index(drop=True)
        neg = df[df['score'] < 0.0].sort_values(by=['score']).reset_index(drop=True)
        # Write data (optional)
        if self.write_:
            out_path = os.path.join("./results/", self.method)
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
            out_path = os.path.join("./results/", self.method)
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
            out_path = os.path.join("./results/", self.method)
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