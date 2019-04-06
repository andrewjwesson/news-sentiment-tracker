# Training sentiment models on the Yelp 5-class review dataset
The provided code performs training and validation for the Yelp 5-class review dataset. 

## Approach 1: fastText

### fastText training (command line utility)

The training step uses the command line utility provided by fastText. Follow the build instructions (might require
a gcc/g++ compiler update) [as per the documentation](https://fasttext.cc/docs/en/support.html). 

    $ git clone https://github.com/facebookresearch/fastText.git
    $ cd fastText
    $ make

Once the installation completes, the fastText executable can be run from the command line. Training the model is done as per 
the [tutorials for classification](https://fasttext.cc/docs/en/supervised-tutorial.html).

To train a bigram model with a learning rate of 0.5 for 5 epochs, the following command is used

    fasttext supervised -input yelp_data/train.csv -output model_bigram -lr 0.5 -epoch 5 -wordNgrams 2

For a full set of tunable hyperparameters to improve the training accuracy, use the following command

    fasttext supervised -h


### fastText prediction (Python)

If not done already, download the [FastText](https://fasttext.cc/) library for sentiment analysis. Install the Python 
module for fastText from Facebook Research's 
repository using ```pip``` as follows.

    $ git clone https://github.com/facebookresearch/fastText.git
    $ cd fastText
    $ pip3 install .
    
The fastText library can then be imported into Python using the regular command:

    import fastText

To get the prediction accuracy on the Yelp test set, run the script ```predict.py``` with the following arguments:

    python3 predict.py -t yelp_data/test.csv -m fasttext -f ../models/fasttext_yelp_review_full.ftz

    
## Approach 2: Flair NLP 

*Note that the current implementation is limited in terms of how large a training set can be read in and loaded into GPU memory*
Currently, a subset of the full Yelp training set (approximately 40k training samples) are used for training with Flair. 
The test set is similarly downsized. Stratified sampling is used on both the training and test sets (using the provided notebook
```resample.ipynb```.

To train the Flair NLP model on the Yelp dataset, run the script ```train.py``` on a GPU-enabled machine:

    python train.py

This will generate a model file ```final-model.pt``` which can then be used for test predictions. 

To get the prediction accuracy on the Yelp test set, run the script ```predict.py``` with the following arguments:

    python3 predict.py -t yelp_data/test.csv -m flair -f ../models/final-model.pt






  
