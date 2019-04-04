# news-sentiment-tracker
Track changes in sentiment towards an entity using news article data.

# Install modules

First, set up virtual environment and install from ```requirements.txt```:

    python3 -m venv venv
    source venv/bin/activate
    pip3 install -r requirements.txt

For further development, simply activate the existing virtual environment.

    source venv/bin/activate

For tokenization, sentence segmentation and named entity recognition, we use the SpaCy library's English language models. 
These have to be downloaded manually:

    python3 -m spacy download en
    python3 -m spacy download en_core_web_md 

## fastText for Sentiment Analysis
If using [FastText](https://fasttext.cc/) for sentiment analysis, install the Python module from Facebook Research's 
repository using ```pip``` as follows.

    $ git clone https://github.com/facebookresearch/fastText.git
    $ cd fastText
    $ pip install .
    
The fastText library can then be imported into Python using the regular command:

    import fastText
