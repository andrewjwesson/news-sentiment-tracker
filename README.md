# News Sentiment Tracker
## A Targeted Opinion Mining Interface

In this project, we showcase an end-to-end NLP-based application that automatically detects
fine-grained sentiment towards a specific target query (such as a person, event, product or 
organization) in news articles. We apply novel combinations of techniques from big data, NLP
and time series visualization to provide the end user targeted insights into press coverage on a
specific entity. Our system is shown to identify large-scale shifts in sentiment in news coverage
towards a target reliably, as can be seen in the showcased real-world examples from past events
covered by US news publications. 

## Our Data Product

[See the web-based app on Heroku.](https://nlp-733-dash.herokuapp.com)
Example usage of the web UI is at the bottom of this page. 

## Install modules

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

### fastText for Sentiment Analysis

If using [FastText](https://fasttext.cc/) for sentiment analysis, install the Python module from Facebook Research's 
repository using ```pip``` as follows.

    $ git clone https://github.com/facebookresearch/fastText.git
    $ cd fastText
    $ pip3 install .
    
The fastText library can then be imported into Python using the regular command:

    import fastText
    
## Sentiment Models

The below sentiment classifier models have been implemented thus far:
 - **TextBlob**: Does not require a trained model (uses the internal "polarity" metric from TextBlob)
 - **fastText**: Requires the 5-classes [fastText supervised model](https://fasttext.cc/docs/en/supervised-models.html) 
 (already present in ```./models```)
 - **Flair**: Requires a GPU-enabled machine to run, and a trained Flair classifier model (to be added soon)
 
 ## Usage
 
 Run the file ```analysis.py```. An example case is shown below.
 
    python3 analysis.py --method fasttext -name "United Airlines"
    
 To get a full list of options for ```analysis.py```, type ```python3 analysis -h```

## User Interface Demo

Below is some [example usage of our UI](https://nlp-733-dash.herokuapp.com) for specific target queries.

Update the calendar heat maps by selecting targets of interest from the dropdown.

![](./assets/gif/calmaps.gif)
  
Inspect the sentiment over time by zooming in on periods of interest. Hover over the bars to see the article count for the day 
and the most polar headline for that day.

![](./assets/gif/timeseries.gif)

Narrow down on the relevant content about each target by using the time slider bar above the data table. Sort the table by 
publication or sentiment score to track the reasons for a large shift in trends.

![](./assets/gif/data_table.gif)
