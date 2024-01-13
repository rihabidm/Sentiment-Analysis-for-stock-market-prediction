import pandas as pd
from datetime import datetime, date
from urllib.request import urlopen, Request
from bs4 import BeautifulSoup
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
from flask import Flask, request, render_template

# Initialiser l'application Flask
app = Flask(__name__)
app.static_folder = 'templates/static'
@app.route('/')
def index():
    tickers = ['AMZN', 'AMD']
    return render_template('page.html', tickers=tickers)

@app.route('/ajouter_ticker', methods=['POST'])
def ajouter_ticker():
    global df
    # Téléchargez les ressources nécessaires pour l'analyse de sentiment VADER
    finviz_url='https://finviz.com/quote.ashx?t='
    tickers=['AMZN','AMD']
    new_ticker = request.form['search']
    tickers.append(new_ticker)
    news_tables = {}
    for ticker in tickers:
        url = finviz_url + ticker

        req = Request(url=url, headers={'user-agent': 'my-app'})
        response = urlopen(req)
        html=BeautifulSoup(response,'html')
        news_table = html.find(id='news-table')
        news_tables[ticker] = news_table

    parsed_data=[]
    for ticker,news_table in news_tables.items():
        for row in news_table.findAll('tr'):
            if row.a:
                title= row.a.text
                date_data = row.td.text.strip()
                date_data=date_data.split(' ')
                if len(date_data)==1:
                    time_str=date_data[0]

                else:
                    date=date_data[0]
                    if date_data[0]=='Today':
                        current_time = datetime.now()
                        date=current_time.date()
                    time_str = date_data[1] # Assuming date_data[1] contains the time string
                time = datetime.strptime(time_str, '%I:%M%p').time()

                parsed_data.append([ticker,date,time,title])

    df=pd.DataFrame(parsed_data,columns=['ticker','date','time','title'])
    vader=SentimentIntensityAnalyzer()
    f=lambda title: vader.polarity_scores(title)['compound']
    df['compound']=df['title'].apply(f)
    

    return render_template('page2.html', tickers=tickers, df=df)
@app.route('/show_histogram', methods=['POST'])
def show_histogram():
    global df
    plt.figure(figsize=(10,8))
    mean_df=df.groupby(['ticker','date']).mean()
    mean_df=mean_df.unstack()
    mean_df=mean_df.xs('compound',axis="columns").transpose()
    mean_df.plot(kind='bar')
    plt.show()

    return "L'histogramme a été affiché."

if __name__ == '__main__':
    app.run(debug=True)

