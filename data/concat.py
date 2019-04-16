import glob
import pandas as pd

csv_files = sorted(glob.glob("articles*.csv"))

li = []

for filename in csv_files:
    df = pd.read_csv(filename, index_col=0, header=0)
    li.append(df)

data = pd.concat(li, axis=0)
data.columns = ['idx', 'title', 'publication', 'author', 'date',
                'year', 'month', 'url', 'content']
data = data.drop('idx', axis=1)
data['length'] = data['content'].str.len()
# Reorder columns as per the components v2 data
reordered = data[['title', 'author', 'date', 'content',
                  'year', 'month', 'publication', 'length']]
reordered.to_csv('all_the_news.csv')
