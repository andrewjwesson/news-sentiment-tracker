"Predict classes on the Yelp 5-class sentiment test set and judge classifier performance"
import argparse
import re
from typing import Tuple
import fastText
from flair.models import TextClassifier
from flair.data import Sentence
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score


class FastTextPredict:
    "Class that uses FastText trained model to make predictions on Yelp 5-class review test set"
    def __init__(self, test_path: str, model_path: str) -> None:
        self.test = pd.read_csv(test_path, sep='\t', header=None, names=['label', 'text'],
                                lineterminator='\n')    
        try:
            self.model = fastText.load_model(model_path)
        except ValueError:
            raise Exception("Could not find fastText classification model in {}".format(model_path))

    def fasttext_tokenize(self, string: str) -> str:
        "Tokenize text as per fastText requirements"
        string = string.lower()
        string = re.sub(r"([.!?,'/()])", r" \1 ", string)
        return string

    def fasttext_predict(self, text: str) -> str:
        "Make prediction using trained fastText model"
        preprocessed = self.fasttext_tokenize(text)
        label, probability = self.model.predict(preprocessed, 1)
        return label[0]

    def get_prediction(self, test: pd.DataFrame) -> None:
        self.test['pred'] = self.test['text'].map(self.fasttext_predict)
        self.test = self.test[['label', 'pred', 'text']]
        print(self.test.head(3), '\n')

    def f1_measure(self, test: pd.DataFrame) -> Tuple[float, float, float]:
        "Return Macro, micro and weighted f1-scores for the predicted classes"
        f1_macro = f1_score(self.test['label'], self.test['pred'], average='macro')
        f1_micro = f1_score(self.test['label'], self.test['pred'], average='micro')
        f1_weighted = f1_score(test['label'], self.test['pred'], average='weighted')
        print("Macro-F1: {:.4f}\nMicro-F1: {:.4f}\nWeighted-F1: {:.4f}".format(f1_macro, f1_micro, f1_weighted))
        return f1_macro, f1_micro, f1_weighted

    def accuracy_measure(self, test: pd.DataFrame) -> float:
        "Return accuracy score metric for the predicted classes"
        acc = accuracy_score(self.test['label'], self.test['pred'])*100
        print("Test accuracy: {:.2f}%".format(acc))
        return acc

    def predict(self) -> None:
        self.get_prediction(self.test)
        _ = self.f1_measure(self.test)
        _ = self.accuracy_measure(self.test)
        print("\nDone...\n")


class FlairPredict:
    "Class that uses FastText trained model to make predictions on Yelp 5-class review test set"
    def __init__(self, test_path: str, model_path: str) -> None:
        self.test = pd.read_csv(test_path, sep='\t', header=None, names=['label', 'text'],
                                lineterminator='\n')
        try:
            self.model = TextClassifier.load_from_file(model_path)
        except ValueError:
            raise Exception("Could not find Flair classification model in {}".format(model_path))

    def flair_predict(self, text: str) -> str:
        "Make prediction using trained fastText model"
        doc = Sentence(text)
        self.model.predict(doc)
        score = doc.labels[0].value
        label = '__label__' + score
        return label

    def get_prediction(self, test: pd.DataFrame) -> None:
        self.test['pred'] = self.test['text'].map(self.flair_predict)
        self.test = self.test[['label', 'pred', 'text']]
        print(self.test.head(3))

    def f1_measure(self, test: pd.DataFrame) -> Tuple[float, float, float]:
        "Return Macro, micro and weighted f1-scores for the predicted classes"
        f1_macro = f1_score(self.test['label'], self.test['pred'], average='macro')
        f1_micro = f1_score(self.test['label'], self.test['pred'], average='micro')
        f1_weighted = f1_score(test['label'], self.test['pred'], average='weighted')
        print("Macro-F1: {:.4f}\nMicro-F1: {:.4f}\nWeighted-F1: {:.4f}".format(f1_macro, f1_micro, f1_weighted))
        return f1_macro, f1_micro, f1_weighted

    def accuracy_measure(self, test: pd.DataFrame) -> float:
        "Return accuracy score metric for the predicted classes"
        acc = accuracy_score(self.test['label'], self.test['pred'])*100
        print("Test accuracy: {:.2f}%".format(acc))
        return acc

    def predict(self) -> None:
        self.get_prediction(self.test)
        _ = self.f1_measure(self.test)
        _ = self.accuracy_measure(self.test)
        print("\nDone...\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Make predictions on yelp 5-class review \
                                     test set using a trained classifier model.')
    parser.add_argument('-t', '--test', type=str, help='Path to test set (csv or similar)',
                        default='yelp_data/test.csv')
    parser.add_argument('-m', '--method', type=str, help='Sentiment analysis model (textblob, fasttext or flair)',
                        default="fasttext")
    parser.add_argument('-f', '--modelfile', type=str, help='Path to trained classifier model for fasttext or flair',
                        default='../models/fasttext_yelp_review_full.ftz')
    args = parser.parse_args()
    test_path = args.test
    method = args.method
    model_path = args.modelfile
    # Run prediction
    if method == 'fasttext':
        tb = FastTextPredict(test_path, model_path)
        tb.predict()
    elif method == 'flair':
        ft = FlairPredict(test_path, model_path)
        ft.predict()
    else:
        raise Exception("The requested method for sentiment prediction has not yet been implemented!") 
