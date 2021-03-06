import html
import re
from string import punctuation
from typing import List

import pandas as pd
import spacy

class IdentifyNews:
    "Class to perform initial filtering of the full news dataset"
    def __init__(self, news: pd.DataFrame, name: str) -> None:
        self.news = news
        self.name = name

    def check_name(self, content: str, name: str) -> bool:
        "Check if target (person/organization or other entity) exists in the news content"
        flag = False
        if self.name in content:
            flag = True
        return flag

    def filter_df(self, df: pd.DataFrame, name: str) -> pd.DataFrame:
        "Only return those rows in the DataFrame whose content match flag is true"
        df['match'] = df['content'].apply(lambda x: self.check_name(x, name))
        df_relevant = df.loc[df['match'].eq(True)]
        return df_relevant.drop(['match'], axis=1)

    def get(self):
        "Check name, then filter to return relevant articles in a Pandas DataFrame"
        news_relevant = self.filter_df(self.news, self.name)
        return news_relevant


class ExtractContent:
    "Class to extract relevant sentences and their lemmas from news content pertaining to the target"
    def __init__(self, news: pd.DataFrame, name: str) -> None:
        self.name = name
        self.news = news
        # Include spaCy language model for fast sentence segmentation of news content
        sentencizer = spacy.blank('en')
        sentencizer.add_pipe(sentencizer.create_pipe('sentencizer'))
        self.sentencizer = sentencizer

    def clean_text(self, x: str) -> str:
        """Text cleaning function inspired by the cleanup utility function in fastai.text.transform:
        https://github.com/fastai/fastai/blob/2c5eb1e219f55773330d6aa1649f3841e39603ff/fastai/text/transform.py#L58
        """
        re1 = re.compile(r'  +')
        x = x.replace('#39;', "'").replace('amp;', '&').replace('#146;', "'").replace(
            'nbsp;', ' ').replace('#36;', '$').replace('\\n', " ").replace('\n', " ").replace(
            'quot;', "'").replace('<br />', "\n").replace('\\"', '"').replace('\\xa0', ' ').replace(
            ' @.@ ', '.').replace('\xa0', ' ').replace(' @-@ ', '-').replace('\\', ' \\ ')
        return re1.sub(' ', html.unescape(x))

    def sentencize(self, document: str, name: str) -> List[str]:
        "Extract individual sentences from the relevant content"
        doc = self.sentencizer(document)
        relevant = []
        for sent in doc.sents:
            for n in name.split():
                if n in sent.text:
                    clean = self.clean_text(sent.text)
                    relevant.append(clean)
        # Remove duplicates
        relevant = list(dict.fromkeys(relevant))
        return relevant

    def lemmatize(self, sentences: List[str]) -> str:
        "Lemmatize the content for downstream comparison and/or similarity detection"
        document = ' '.join(sentences)
        add_removed_words = {n for n in self.name.split()}
        # For similarity, we don't want the target name in every article
        # If the target is mentioned many times, it might skew the similarity results, so we remove it. 
        stopwords = self.sentencizer.Defaults.stop_words
        stopwords = stopwords.union(add_removed_words)
        doc = self.sentencizer(document)
        lemmas = [str(tok.lemma_).lower() for tok in doc if tok.text not in stopwords \
                  and tok.text not in punctuation]
        return ' '.join(lemmas)

    def extract(self) -> pd.DataFrame:
        """Extract only those sentences that explicitly mention the target entity
        Note: This assumes that the news DataFrame has a column named 'content' obtained
        from the earlier data reduction step.
        We return a DataFrame with two added columns: 'relevant' and 'lemmas'
        """
        self.news['relevant'] = self.news['content'].apply(lambda x: self.sentencize(x, self.name))
        self.news['lemmas'] = self.news['relevant'].apply(self.lemmatize)
        # Drop duplicates before returning
        news_relevant = self.news.drop_duplicates(subset=['lemmas'])
        return news_relevant
