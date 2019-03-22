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
    python3 -m spacy download en_core_web_sm
    python3 -m spacy download en_core_web_md 
