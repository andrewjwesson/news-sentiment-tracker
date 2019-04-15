import glob
import pandas as pd

csv_files = sorted(glob.glob("articles*.csv"))

li = []

for filename in csv_files:
    df = pd.read_csv(filename, index_col=0, header=0)
    li.append(df)

data = pd.concat(li, axis=0)
data.columns = ['id', 'title', 'publication', 'author', 'date',
                'year', 'month', 'url', 'content']
data.to_csv('all_the_news.csv')
