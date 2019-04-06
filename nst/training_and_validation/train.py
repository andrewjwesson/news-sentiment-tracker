"""
Train a sentiment classification model using the Flair NLP library
https://github.com/zalandoresearch/flair
"""
from flair.data_fetcher import NLPTaskDataFetcher
from flair.embeddings import WordEmbeddings, FlairEmbeddings, DocumentRNNEmbeddings, ELMoEmbeddings
from flair.models import TextClassifier
from flair.trainers import ModelTrainer
from pathlib import Path

# Specify path to train and test data
file_path = Path('../') / 'data/sentiment'

train = 'train.csv'
test = 'test.csv'

# Load corpus
corpus = NLPTaskDataFetcher.load_classification_corpus(file_path, 
                                                       train_file=train, 
                                                       dev_file=None, 
                                                       test_file=test,
                                                       )

# Create label dictionary
label_dict = corpus.make_label_dictionary()

# Specify word embeddings - in this case we use BERT with Flair's contextualized string embeddings
word_embeddings = [ELMoEmbeddings('original'), 
                   FlairEmbeddings('news-forward'), 
                   FlairEmbeddings('news-backward')]

# Initialize document embedding by passing list of word embeddings
# Can choose between many RNN types (GRU by default, to change use rnn_type parameter)
document_embeddings = DocumentRNNEmbeddings(word_embeddings, hidden_size=512, 
                                            reproject_words=True, reproject_words_dimension=256,
                                            rnn_type='LSTM')


# Define classifier
classifier = TextClassifier(document_embeddings, 
                            label_dictionary=label_dict, 
                            multi_label=False)
trainer = ModelTrainer(classifier, corpus)

# Begin training
NUM_EPOCHS = 10
trainer.train(file_path, max_epochs=NUM_EPOCHS)