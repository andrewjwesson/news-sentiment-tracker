import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
import os
import glob
import time
import base64
import dash
import dash_table
from dash_table.Format import Format
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html

# Path to csv data
data_path = "./data/"
# Path to static image data (for calendar maps)
img_path = os.path.join(data_path, 'img')

def collect_calendar_maps(img_path, prefix='calmap_', suffix='.png'):
    # Obtain all png files in the directory
    paths = glob.glob(os.path.join(img_path, "*.png"))
    images = dict()
    names = []
    for path in paths:
        fname = os.path.basename(path)
        query = fname[len(prefix):-len(suffix)]
        images[query] = path
        names.append(query)
    return images, names


# Read in data
data_cols = ['title', 'publication', 'relevant', 'mean_score', 'mean_dev', 'count', 'query']
polarity_cols = ['date', 'title', 'publication', 'relevant', 'score', 'deviation', 'query']
bd_cols = ['publication', 'Negative', 'Positive', 'query']
cosine_cols = ['publication', 'count', 'x', 'y', 'query']
data = pd.read_csv(os.path.join(data_path, 'data.csv'), index_col=0, parse_dates=True, names=data_cols)
bd = pd.read_csv(os.path.join(data_path, 'breakdown.csv'), names=bd_cols)
polarity = pd.read_csv(os.path.join(data_path, 'polarity.csv'), parse_dates=['date'], names=polarity_cols)
cosine_dist = pd.read_csv(os.path.join(data_path, 'cosine_dist.csv'), names=cosine_cols)
calmap_images, names = collect_calendar_maps(img_path)

# Get date range for the full data
daterange = pd.date_range(polarity['date'].min(), polarity['date'].max(), freq='m')

# Initialize app
app = dash.Dash(__name__)
# For Heroku deployment.
server = app.server
server.secret_key = os.environ.get("SECRET_KEY", "secret")

# External files
dcc._js_dist[0]['external_url'] = 'https://cdn.plot.ly/plotly-basic-latest.min.js'
external_css = ["https://cdnjs.cloudflare.com/ajax/libs/skeleton/2.0.4/skeleton.min.css",
                "https://fonts.googleapis.com/css?family=Roboto:400,400i,700,700i",
                ]

for css in external_css:
    app.css.append_css({'external_url': css})


"""As of April 2019, Plotly-dash cannot handle Pandas DateTime objects
The below slightly hacky helper functions are adapted from here: 
https://stackoverflow.com/questions/51063191/date-slider-with-plotly-dash-does-not-work
"""
def unixTimeMillis(dt):
    "Convert datetime to unix timestamp"
    return int(time.mktime(dt.timetuple()))

def unixToDatetime(unix):
    "Convert unix timestamp to datetime"
    return pd.to_datetime(unix, unit='s')

def getMarks(start, end, Nth=3):
    """Returns the marks for labeling. 
        Every Nth value will be used.
    """
    result = {}
    for i, date in enumerate(daterange):
        if i % Nth == 1:
            result[unixTimeMillis(date)] = str(date.strftime('%Y-%m'))
    return result


# App Layout
app.layout = html.Div([
    html.Div([
        html.H1('News Sentiment Tracker'),
        html.P('''This is a dashboard to help study the sentiment of news articles pertaining to
        a specific topic: i.e. a person, organization or any other entity towards which there can
        specifically be positive or negative coverage. Multiple visualization techniques
        are utilized to showcase the sentiment as aggregated values, over 16 different news publications
        as a function of time.
        '''),
        html.P('Choose an entity for news sentiment tracking:'),
        dcc.Dropdown(
            id='entity-dropdown',
            options=[
                {'label': 'United Airlines', 'value': 'united_airlines'},
                {'label': 'Liberia', 'value': 'liberia'},
                {'label': 'Martin Shkreli', 'value': 'martin_shkreli'},
                {'label': 'Ryan Lochte', 'value': 'ryan_lochte'}
            ],
            value='united_airlines',
            className="threeColumns",
            style={'text-align': 'left', 'font-size': 22, 'height': '30px', 'width': '600px'}
        ),
        html.Hr(),
        html.H3('Calendar Map of Aggregated Sentiment', className='Title'),
        html.Div([
            html.P('''The below chart shows a calendar heat map, which is a 2D representation of aggregated 
            sentiment scores over a multi-year period. Blue indicates strongly positive mean sentiment
            while red indicates strongly negative mean sentiment.
            '''),
            html.Div([
                html.Img(id='calmap-image')
            ]),
        ]),
    ]),
    html.Hr(),
    html.Div([
        html.H3('Breakdown per publication', className='Title'),
        html.P('''The total number of articles written by each publication (positive or negative
        in mean sentiment) towards a targer are listed below. The publication that wrote the most
        negative content towards the target entity is placed on top.
        '''),
        html.Div([
            dcc.Graph(id='bar-breakdown'),
        ]),
    ]),
    html.Hr(),
    html.Div([
        html.H3('Sentiment Bar Graph Time Series', className='Title'),
        html.P('''In the below chart, the height of each bar shows the aggregated (mean) sentiment score
        for all articles on a particular target for each day. The hover box shows the headline of the most
        polar article (positive or negative) for each day, as well as the name of the publication
        that wrote it. In addition, a red line plot is overlaid, showing the article counts for that particular day.
        '''),
        html.Div([
            dcc.Graph(id='bar-score-timeline')
        ]),
    ]),
    html.Hr(),
    html.Div([
        html.H3('Euclidean Distance of Cosine Similarities', className='Title'),
        html.P('''The below section shows a Multi-dimensional Scaling (MDS) plot
        of the mean cosine distances per publication. The cosine distance matrix was generated
        using the TF-IDF vectors of each article's relevant content towards the target.
        The further apart two points are, the more different the vocabulary or key words used by the 
        publication towards the target. The size of the bubbles represents the number of articles
        for that publication.
        '''),
        html.Div([
            dcc.Graph(id='count-scatter')
        ]),
    ]),
    html.Hr(),
    html.Div([
        html.H3('Data Table', className='Title'),
        html.P('''The below table separates the positive and negative content (per the dropdown),
        ordered by date. If there is a sudden change in sentiment trend, inspecting the content
        directly from the table can yield insights into who wrote the content and the reasons for
        the change. By default, the table shows content with negative scores.
        '''),
        dcc.Dropdown(
            id='polarity-dropdown',
            options=[
                {'label': 'Positive Score', 'value': 'pos'},
                {'label': 'Negative Score', 'value': 'neg'},
            ],
            value='neg',
            className="threeColumns",
            style={'text-align': 'left', 'font-size': 20, 'height': '25px', 'width': '500px'},
        ),
        html.Hr(),
        html.H5("Specify Date Range to Filter News Content"),
        html.P("Drag the below slider bar's pointers left or right to narrow down on specific date ranges."),
        html.Div([
            dcc.RangeSlider(
                id='date-slider',
                min=unixTimeMillis(daterange.min()),
                max=unixTimeMillis(daterange.max()),
                value=[unixTimeMillis(daterange.min()),
                       unixTimeMillis(daterange.max())],
                marks=getMarks(daterange.min(),
                               daterange.max()),
            ),
        ]), 
        html.Hr(),
        html.Div([dash_table.DataTable(
                 id='test-table',
                 columns=[
                     {"name": 'date', "id": 'date'},
                     {"name": 'title', "id": 'title'},
                     {"name": 'publication', "id": 'publication'},
                     {"name": 'relevant', "id": 'relevant'},
                     {'name': 'score',
                         'id': 'score',
                         'type': 'numeric',
                         'format': Format(precision=2),
                      },
                     {'name': 'deviation',
                         'id': 'deviation',
                         'type': 'numeric',
                         'format': Format(precision=2),
                      },
                 ],
                 sorting=True,
                 style_data={'whiteSpace': 'normal'},
                 style_table={'overflowX': 'scroll', 'maxHeight': '600'},
                 style_cell={
                     'backgroundColor': 'rgb(248, 248, 255)',
                     'textAlign': 'left',
                     'font_family': 'roboto',
                 },
                 css=[{
                     'selector': '.dash-cell div.dash-cell-value',
                     'rule': 'display: inline; white-space: inherit; overflow: inherit; text-overflow: inherit;'
                     }]
                 )
        ]),
    ])
], style={'padding': '0px 10px 15px 10px',
          'marginLeft': 'auto', 'marginRight': 'auto', "width": "900px",
          'boxShadow': '0px 0px 5px 5px rgba(204,204,204,0.4)'})

# Interaction callbacks

@app.callback(Output('calmap-image', 'src'), [Input('entity-dropdown', 'value')])
def update_image_src(selected_dropdown_value):
    # print the image_path to confirm the selection is as expected
    image = calmap_images[selected_dropdown_value]
    encoded_image = base64.b64encode(open(image, 'rb').read())
    return 'data:image/png;base64,{}'.format(encoded_image.decode())

@app.callback(Output("bar-breakdown", "figure"), [Input('entity-dropdown', 'value')])
def update_bars(selected_dropdown_value):  
    bdd = bd[bd['query'] == selected_dropdown_value].sort_values(by='Negative')
    return {
        'data': [
            {'x': bdd['Negative'],
             'y': bdd['publication'],
             'type': 'bar',
             'orientation': 'h',
             'marker': {'color': 'rgb(220, 80, 80)'},
             'name': 'Negative'},

            {'x': bdd['Positive'],
             'y': bdd['publication'],
             'type': 'bar',
             'orientation': 'h',
             'marker': {'color': 'rgb(45, 85, 200)'},
             'name': 'Positive'},
        ],
        'layout': {
            'title': 'Breakdown of coverage by publication',
            'width': 900,
            'height': 500,
            'plot_bgcolor': 'rgb(248, 248, 255)',
            'paper_bgcolor': 'rgb(248, 248, 255)',
            'xaxis': {'dtick': 10, 'title': 'Article Count'},
            'yaxis': {'automargin': True},
            'legend': {'orientation': 'h', 'x': 0.0, 'y': -0.1},
        }
    }


@app.callback(Output("bar-score-timeline", "figure"), [Input('entity-dropdown', 'value')])
def update_score_timeline(selected_dropdown_value):  
    df = data[data['query'] == selected_dropdown_value]
    df['display'] = df['publication'].astype(str) + "<br>" + df['title']
    return {
        'data': [
            {'x': df.index,
             'y': df['mean_score'],
             'type': 'bar',
             'text': df['display'],
             'hoverlabel': {'namelength': -1},
             'opacity': 0.8,
             'marker': {'color': 'rgba(45, 85, 200, 0.8)',
                        'line': {'color': 'rgba(45, 85, 200, 0.8)', 'width': 3}},
             'name': 'Mean sentiment score',
             },
            {'x': df.index,
             'y': df['count'],
             'type': 'scatter',
             'opacity': 0.8,
             'mode': 'lines',
             'line': {'dash': 'line',
                      'color': 'rgba(220, 80, 80, 0.6)',
                      'size': 3,
                      },
             'name': 'Article count',
             'yaxis': 'y2',
             },  
        ],
        'layout': {
            'title': 'Time series of sentiment scores with range slider',
            'width': 900,
            'height': 600,
            'plot_bgcolor': 'rgb(248, 248, 255)',
            'paper_bgcolor': 'rgb(248, 248, 255)',
            'showlegend': True,
            'legend': {'orientation': 'h', 'x': 0.0, 'y': -0.45},
            'yaxis': {'title': 'Mean Sentiment Score'},
            'yaxis2': {'title': 'Article Count', 'overlaying': 'y', 'side': 'right'},
            'xaxis': {'range': [df.index.min() - pd.DateOffset(30), df.index.max() + pd.DateOffset(30)], 
                      'rangeselector': {'buttons': [
                          {'count': 2, 'label': '2m', 'step': 'month', 'stepmode': 'backward'},
                          {'count': 6, 'label': '6m', 'step': 'month', 'stepmode': 'backward'},
                          {'count': 1, 'label': '1y', 'step': 'year', 'stepmode': 'backward'},
                          {'count': 2, 'label': '2y', 'step': 'year', 'stepmode': 'backward'},
                          {'step': 'all'}]},
                      'rangeslider': {'visible': True},
                      'type': 'date',
                      }
                }
    }


@app.callback(Output("count-scatter", "figure"), [Input('entity-dropdown', 'value')])
def update_score_timeline(selected_dropdown_value):  
    df = cosine_dist[cosine_dist['query'] == selected_dropdown_value]
    np.random.seed(5)   # Random seed
    colors = ['rgb({}, {}, {})'.format(np.random.randint(0, 256), 
                                       np.random.randint(0, 256),
                                       np.random.randint(0, 256)) for _ in range(len(df['publication']))]
    return {
        'data': [
            {'x': df['x'],
             'y': df['y'],
             'type': 'scatter',
             'opacity': 0.9,
             'mode': 'markers+text',
             'marker': {'line': {'color': 'rgba(25, 25, 25, 0.8)', 'width': 1.5},
                        'size': np.sqrt(df['count'])*10,
                        'color': colors,
                        },
             'text': df['publication'].astype(str),
             'textfont': {'size': 16},
             'hovertext': df['count'].astype(str) + " article(s)",
             'textposition': 'bottom',
             'cliponaxis': False,
             'color': np.random.randint(len(df['publication'])),
             },          
        ],
        'layout': {
            'title': 'Multi-Dimensional Scaling of Mean Cosine Distance (per Publication)',
            'width': 900,
            'height': 900,
            'plot_bgcolor': 'rgb(248, 248, 255)',
            'paper_bgcolor': 'rgb(248, 248, 255)',
            'margin': {'l': 100, 'r': 100, 't': 100, 'b': 100},
            'autorange': True,
            'xaxis': {'showline': True, 'zeroline': False, 'showgrid': False,
                      'mirror': True, 'showticklabels': False},
            'yaxis': {'showline': True, 'zeroline': False, 'showgrid': False,
                      'mirror': True, 'showticklabels': False},
            }
    }

# Display data table of extracted positive and negative news content
@app.callback(Output("test-table", "data"), [
              Input('entity-dropdown', 'value'),
              Input('polarity-dropdown', 'value'),
              Input('date-slider', 'value')
              ])
def update_table(selected_dropdown_value, polarity_value, daterange_value):
    """Uses a unixtoDateTime helper function to translate Pandas DateTime stamps
    to unix timestamps and back. This is because Plotly Dash cannot handle
    Pandas DateTime objects as of April 2019
    """
    date_min = unixToDatetime(daterange_value[0])
    date_max = unixToDatetime(daterange_value[1])
    if polarity_value == 'neg':
        df = polarity[(polarity['query'] == selected_dropdown_value) & (polarity['score'] < 0.0)]
        df = df.loc[(df['date'] >= date_min) & (df['date'] <= date_max)]
    else:
        df = polarity[(polarity['query'] == selected_dropdown_value) & (polarity['score'] > 0.0)]
        df = df.loc[(df['date'] >= date_min) & (df['date'] <= date_max)]
    return df.to_dict("rows")


if __name__ == "__main__":
    app.run_server(debug=True)
