#Importing the libraries to be used
#import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import itertools
import collections
import re, os
from datetime import datetime
import requests
import asyncio

#import panel as pn
#pn.extension('tabulator')
#import hvplot.pandas
import dataframe_image as dfi

import tweepy
import tweepy as tw
from textblob import TextBlob
#import pygal
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer 
from nltk import bigrams as bg
from textblob import TextBlob
import networkx # for creating networknodes
import networkx as nx
#from pandas.io.json import json_normalize

import warnings
warnings.filterwarnings("ignore")

sns.set(font_scale=1.5)
sns.set_style("whitegrid")

# api_key = "CWW5rvfohMPrJaf4KuACiE0FN"
# api_key_secret = "r6iRyGGXzY9iwQaxjAjA64WweBIUFZnPM6tVpVfUMIrx3zlGtq"

# access_token = "3403682451-rwoiYIUkNtEeNDZ97ZoRzbI5daYm9vOKRnzXlan"
# access_token_secret = "QdrHp0YVqfIdVpyZVhTgIUSnGsOnL24sWEQUqUdbhUxVK"

api_key = "sIziTRjWI8ggmaIj67HTe327B"
api_key_secret = "HsmH8vTVgQ06I6EMHd7dobD06E9pGbrqKEjNs0nBhL4ibk5aNe"
bearer_token = """AAAAAAAAAAAAAAAAAAAAANi9jQEAAAAAkx%2Frh3pRKvIU1veXKSEI%2FGvm98A%3DcwfVZMG8nLpdIKyrcS3X6eDjJQCksTkCANCufhzhJXlHQe6dPz"""


data_folder = os.path.abspath("static/assets/images")
most_eng_html = os.path.abspath("templates/most_engagement.html")
# All Function -----------------------
## Text Processing
#A Function for cleaning the file (The text column in it)
def text_clean(df_tweets):
    #Lowercasing all the letters
    df_tweets['text'] = df_tweets['text'].str.lower() 

    #Removes mentions containing rt word
    df_tweets['text'] = df_tweets['text'].str.replace(r'rt @[A-Za-z0-9_]+:', '', regex=True) 
    #Removes mention just containing @word only
    df_tweets['text'] = df_tweets['text'].str.replace(r'@[A-Za-z0-9_]+', '', regex=True) 
    #Removing #tags 
    #df_tweets['text'] = df_tweets['text'].str.replace(r'#[A-Za-z0-9_]+', '', regex=True)  

    #Removing links
    df_tweets['text'] = df_tweets['text'].str.replace(r'http\S+', '', regex=True)
    df_tweets['text'] = df_tweets['text'].str.replace(r'www.\S+', '', regex=True) 

    #Removing punctuations and replacing with a single space
    df_tweets['text'] = df_tweets['text'].str.replace(r'[()!?]', ' ', regex=True)  
    df_tweets['text'] = df_tweets['text'].str.replace(r'\[.*?\]', ' ', regex=True)

    #Filtering non-alphanumeric characters
    df_tweets['text'] = df_tweets['text'].str.replace(r'[^a-z0-9]', ' ', regex=True) 

    #Removing Stoping words + keywords_to_hear
    stop = stopwords.words(['english', 'spanish', 'portuguese']) + ['l','pez', 'n', 'andr','p', 'si','est', 'c', 'qu']
    df_tweets['tweet_without_stopwords'] = df_tweets['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))


#Function to get the subjectivity Subjectivity refers to an individual's feelings, opinions, or preferences.
def getSubjectivity(text):
    return TextBlob(text).sentiment.subjectivity

#Create a function to get the polarity (Tells how positive or negative the text is)
def getPolarity(text):
    return TextBlob(text).sentiment.polarity

#Create fxn to compute negative , neutral and positive analysis
def getAnalysis(score):
    if score < 0:
        return 'Negative'
    elif score == 0:
        return 'Neutral'
    else:
        return 'Positive'

#Calculating the percentages
def percentage_polarity(part, whole_data):
    percentage = 100 * float(part) / float(whole_data)
    return round(percentage, 1)

def word_cloud(wd_list):
    # print(len(wd_list))
    stopwords = set(STOPWORDS)
    all_words = ' '.join([text for text in wd_list])
    wordcloud = WordCloud(
        background_color='snow',
        stopwords=stopwords,
        width=1600,
        height=800,
        random_state=1,
        colormap='jet',
        max_words=80,
        max_font_size=200).generate(all_words)
    plt.figure(figsize=(12, 10))
    plt.axis('off')
    plt.imshow(wordcloud, interpolation="bilinear")


#  program starts here -----------------------
def run_twitter_analysis(tweet_keywords, tweet_location, tweet_date):
    tweet_location = tweet_location.strip()

    #authentication
    # auth = tweepy.OAuthHandler(api_key, api_key_secret)
    # auth.set_access_token(access_token, access_token_secret)
    # api = tweepy.API(auth)

    # Authentication bearer token 
    auth = tweepy.OAuth2BearerHandler(bearer_token)
    auth.apply_auth()
    api = tweepy.API(auth)

    # keywords = input("Enter the keyword to get tweets :: ")
    # Specify keywords to search for
    # here we pass it from frontend 
    keywords = tweet_keywords 
    limit = 1000 #Number of tweets to obtain

    #limit at a time , we get 200.To solve this issue run code below.
    tweets = tweepy.Cursor(api.search_tweets, q=keywords, count=200, until=tweet_date, tweet_mode = 'extended').items(limit)

    #Create DataFrame
    columns = ['created_at', 'text', 'User', 'retweet_count', 'user_location', 'user_followers_count']
    data = []

    # for tweet in tweets:
    for tweet in tweets:
        data.append([ tweet.created_at, tweet.full_text, tweet.user.screen_name, tweet.retweet_count, tweet.user.location, tweet.user.followers_count])

    df = pd.DataFrame(data, columns=columns)
    df.rename(columns={"full_text": "text"}, inplace=True)
    df.to_csv("tweet_data.csv",index=False)
    df.head(5)

    df = pd.read_csv("tweet_data.csv",index_col=False)
    df.head(4)

    #Removing the duplicates
    df = df.drop_duplicates()

    #Convert the created_at column to datetime datatype
    df['created_at'] = df['created_at'].astype('datetime64[ns]')

    #Checking date ranges and hours
    #Creating a column for hour
    df['hour'] = df['created_at'].dt.hour
    #Creating a column for days
    # df['date'] = df['created_at'].dt.strftime('%d/%m/%Y')
    df['date'] = df['created_at'].dt.date
    #Creating a column for month
    df['month'] = df['created_at'].dt.month
    # df.head()
    df.head(4)

    df = df.dropna()
    filter_by_location = df['user_location'].str.lower().str.strip().str.contains(tweet_location, regex= False)
    df = df[filter_by_location]

    # table of most engagement---------
    sorted_by_retweet_count = df.sort_values(by=['retweet_count'], ascending=False)
    table_texts = sorted_by_retweet_count[['date', 'User', 'retweet_count', 'text']]
    most_engagement = table_texts.to_html(index=False)
    # print(most_engagement)
    with open(f"{most_eng_html}", "w", encoding='utf-8') as f:
        f.writelines('{% extends "results.html" %}\n\n')
        f.writelines("{% block table %}\n\n")
        f.writelines(most_engagement)
        f.writelines("\n\n{% endblock %}")
    # -------------------------------


    # time series showing when the tweets for this analysis was created
    reactions = df.groupby(['date']).count()
    fig = plt.figure(figsize=(15,6))
    ax = reactions.text.plot(figsize=(15,6), ls='--', c='blue')
    # print(reactions)
    ax.plot(reactions, 'r^')
    plt.ylabel('The Count of tweets collected')
    plt.title('A Trend on the counts of tweets and the dates created' , fontsize=15, color= 'brown', fontweight='bold')
    ax.xaxis.grid(True)
    ax.yaxis.grid(True)
    save_path = os.path.join(data_folder, "trends_date.png")
    fig.savefig(save_path)

    # time series plot for the most active hours for tweeting
    reactions = df.groupby(['hour']).count().sort_values(by='created_at',ascending=0)
    reactions = df.groupby(['hour']).count()
    # print(reactions)
    fig = plt.figure(figsize=(15,6))
    ax = reactions.text.plot(figsize=(15,6),ls='--',c='green')
    ax.plot(reactions, 'go')
    plt.ylabel('The Count of tweets collected')
    plt.title('A Trend on the counts of tweets and the hours created',  fontsize=15, color= 'green', fontweight='bold')
    ax.xaxis.grid(True)
    ax.yaxis.grid(True)
    save_path = os.path.join(data_folder, "trends_hour.png")
    fig.savefig(save_path)


    ##EXPLORATORY DATA ANALYSIS
    #Creating a copy for the text column This will enable us work with the text column solely
    df_tweets = df[['text']].copy()
    #Dropping the duplicates
    df_tweets = df_tweets.drop_duplicates()

    text_clean(df_tweets)
    # tokenize the tweets

  # print(df_tweets)
    if not df_tweets.empty:
        df_tweets['tokenized_sents'] = df_tweets.apply(lambda row: nltk.word_tokenize(row['tweet_without_stopwords']), axis=1)
        ### Visualizing/InfoGraphics the text column (Unigram)
        # Create a list of lists containing words for each tweet
        words_in_tweet = list(df_tweets['tokenized_sents'])

        #Calculate word frequencies
        # List of all words across tweets
        all_words = list(itertools.chain(*words_in_tweet))

        # Create counter
        counts_words = collections.Counter(all_words)
        # print(counts_words)

        # transform the list into a pandas dataframe
        df_counts_words = pd.DataFrame(counts_words.most_common(15), columns=['words', 'count'])
        # print(df_counts_words)
        #A horizontal bar graph to visualize the most common words
        # fig = plt.figure(figsize=(10, 6))
        fig = df_counts_words.plot(kind='bar', x='words', y='count', title='Most common words').get_figure()
        plt.tight_layout()
        save_path = os.path.join(data_folder, "most_common_words.png")
        fig.savefig(save_path)

        # Plot horizontal bar graph
        fig = df_counts_words.sort_values(by='count').plot.barh(x='words', y='count',color="green").get_figure()
        # ax.set_title("Common Words Found in Tweets",  fontsize=15, color= 'violet', fontweight='bold')
        plt.tight_layout()
        save_path = os.path.join(data_folder, "count_unigram.png")
        fig.savefig(save_path)


        wordcloud2 = WordCloud(background_color="white", max_words=100,
                            height=3000, width=3000,
                            colormap='Set2',
                            collocations=False,
                            repeat=True).generate(' '.join(df_counts_words["words"]))
        # Generate plot
        plt.figure(figsize=(10,7), facecolor='k')
        plt.tight_layout(pad=0)
        plt.imshow(wordcloud2, interpolation='bilinear')
        plt.axis("off")
        save_path = os.path.join(data_folder, "cloud_uni.png")
        plt.savefig(save_path)
        # plt.show()


        ## Collection of Words – Bigrams
        #Create a list of tokenized_sents
        tweets_words = list(df_tweets['tokenized_sents'])
        #Remove any empty lists
        tweets_words_new = [x for x in tweets_words if x != []]
        # Create list of lists containing bigrams in tweets
        # print(tweets_words_new)

        # print(bigrams)
        # for tweet in tweets_words_new:
        #     d =  bigrams([tweet])
        terms_bigram = [list(bg(tweet)) for tweet in tweets_words_new]

        # # Flatten list of bigrams in clean tweets
        bigrams = list(itertools.chain(*terms_bigram))
        # # Create counter of words in clean bigrams
        bigram_counts = collections.Counter(bigrams)
        # #Creating a dataframe of the most common bigrams
        bigram_df = pd.DataFrame(bigram_counts.most_common(20),columns=['bigram', 'count'])


        ##Visualize Networks of Bigrams
        # Create dictionary of bigrams and their counts
        d = bigram_df.set_index('bigram').T.to_dict('records')
        # Create network plot 
        G = nx.Graph()
        # Create connections between nodes
        for k, v in d[0].items():
            G.add_edge(k[0], k[1], weight=(v * 10))

        fig, ax = plt.subplots(figsize=(20, 15))
        pos = nx.spring_layout(G, k=2)
        # Plot networks
        nx.draw_networkx(G, pos,font_size=16,width=3,edge_color='red',node_color='black',with_labels = False,ax=ax)
        # Create offset labels
        for key, value in pos.items():
            x, y = value[0]+.135, value[1]+.045
            ax.text(x, y,
                    s=key,
                    bbox=dict(facecolor='aqua', alpha=0.55),
                    horizontalalignment='center', fontsize=20)
        plt.title('Visualize Networks of Bigrams',  fontsize=15, color= 'indigo', fontweight='bold')  
        save_path = os.path.join(data_folder, "bigrams_network.png")
        plt.savefig(save_path)
        # plt.show() 

        ##Polarity
        
        #Create two new columns
        df['Subjectivity'] = df['text'].apply(getSubjectivity)
        df['Polarity'] = df['text'].apply(getPolarity)

        #plot the WordCloud
        allwords  = ' '.join([txts for txts in df['text']])
        wordCloud3 = WordCloud(background_color="white",
                            width = 5000, height = 3000, 
                            collocations=False,
                            colormap='Set2',
                            random_state = 1, repeat=True).generate(allwords)

        plt.figure(figsize=(10,7), facecolor='k')
        plt.imshow(wordCloud3, interpolation='bilinear')
        plt.axis('off')
        save_path = os.path.join(data_folder, "cloud_all_pol.png")
        plt.savefig(save_path)


        
        df['Analysis'] = df['Polarity'].apply(getAnalysis)
        sortedDF = df.sort_values(by='Polarity')

        #Plot the polarity and subjectivity
        fig = plt.figure(figsize=(28,10))
        upper_limit = sortedDF.shape[0]
        # print(upper_limit, df)

        for i in range(0, upper_limit):
            # print(df['Polarity'])
            # The range is the number of rows in our dataset
            plt.scatter(df['Polarity'].iloc[i], df['Subjectivity'].iloc[i], color='black')
            
        plt.title("Sentiment Analysis Distribution",  fontsize=15, color= 'grey', fontweight='bold')
        plt.xlabel('Polarity')
        plt.ylabel('Subjectivity')
        save_path = os.path.join(data_folder, "sentiment_analysis_distribution.png")
        fig.savefig(save_path)
        # plt.show()

        #Show the Value counts
        fig = plt.figure()
        sns.countplot(x='Analysis', data=df)
        #plot and visualize the counts
        sns.set(rc={'figure.figsize':(5,5)})
        plt.title('Sentiment Analysis', fontsize=15, color= 'orange', fontweight='bold')
        plt.xlabel('Sentiment')
        plt.ylabel('Counts')
        save_path = os.path.join(data_folder, "sentiment_analysis_graph.png")
        fig.savefig(save_path)
        # plt.show()

        ## polarity ( positive, negative , and neutral scores for each tweet)
        '''using polarity_scores() we, 
        will find all the positive, negative, and neutral scores for each tweet.'''
        analyzer = SentimentIntensityAnalyzer()

        scores = []
        # Declare variables for scores
        compound_list = []
        positive_list = []
        negative_list = []
        neutral_list = []
        for i in range(df['text'].shape[0]):
        #print(analyser.polarity_scores(sentiments_pd['text'][i]))
            compound = analyzer.polarity_scores(df['text'].iloc[i])["compound"]
            pos = analyzer.polarity_scores(df['text'].iloc[i])["pos"]
            neu = analyzer.polarity_scores(df['text'].iloc[i])["neu"]
            neg = analyzer.polarity_scores(df['text'].iloc[i])["neg"]
            
            scores.append({"Compound": compound,
                            "Positive": pos,
                            "Negative": neg,
                            "Neutral": neu
                        })
        
        #Converting the scores dictionary containing the scores into the data frame, then join the sentiments_score data frame with the df data frame.
        sentiments_score = pd.DataFrame.from_dict(scores)
        df = df.join(sentiments_score)

        # sentiment_analysis_polarity_score
        fig, ax = plt.subplots(figsize=(8, 6))
        df.hist(column='Polarity', bins=[-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1],
                ax=ax, color="maroon")
        ax.set_title("Sentiments from Tweets(Using Polarity Score) on the tweets", fontsize=15, color= 'blue', fontweight='bold')
        save_path = os.path.join(data_folder, "sentiment_analysis_polarity_score.png")
        plt.savefig(save_path)
        # plt.show()

        #Finding the percentages of +ve, -ve and neutral
        

        negative = 0
        positive = 0
        neutral = 0

        for index, row in df.iterrows():
            neg = row['Negative']
            pos = row['Positive']
            if neg > pos :
                negative += 1
                negative_list.append(df.text)
            elif pos > neg :
                positive += 1
            elif pos == neg:
                neutral += 1

        positive_percentage = percentage_polarity(positive, df.shape[0])
        negative_percentage = percentage_polarity(negative, df.shape[0])
        neutral_percentage = percentage_polarity(neutral, df.shape[0])

        print(" positive, negative and neutral %")
        print(positive_percentage, negative_percentage, neutral_percentage)
        
        if positive_percentage == 0.0 and negative_percentage == 0.0 and neutral_percentage == 0.0 :
            print("No Pie chart because all percentanges are 0.0")
            fig = plt.figure()
            plt.title("No Pie chart Formed")
            save_path = os.path.join(data_folder, "sentiment_analysis_doughnut.png")
            fig.savefig(save_path)
        else:

            #Creating PieCart for percentages
            labels = ['Positive ['+str(positive_percentage)+'%]' , 'Neutral ['+str(neutral_percentage)+'%]','Negative ['+str(negative_percentage)+'%]']

            sizes = [positive_percentage, neutral_percentage, negative_percentage]
            colors = ['darkblue', '#33BFFF','lightblue']

            fig = plt.figure()

            my_circle=plt.Circle( (0,0), 0.5, color='white')

            # print(sizes)
            patches, texts = plt.pie(sizes, colors=colors, startangle=90)
            # print(patches)

            p=plt.gcf()
            p.gca().add_artist(my_circle)
            plt.style.use('default')
            plt.legend(labels)
            plt.title("Sentiment Analysis Result ", fontsize=15, color= 'blue', fontweight='bold' )
            plt.axis('equal')
            # plt.show()
            save_path = os.path.join(data_folder, "sentiment_analysis_doughnut.png")
            fig.savefig(save_path)

        try:
            fig=plt.figure()
            plt.title("Cloud All Polarity (2)")
            word_cloud(df['text'],)
            save_path = os.path.join(data_folder, "cloud_all_2_pol.png")
            plt.savefig(save_path)
        except:
            fig=plt.figure()
            plt.title("No Cloud All Polarity (2)")
            save_path = os.path.join(data_folder, "cloud_all_2_pol.png")
            fig.savefig(save_path)
            print("No cloud because no text data")

        try:
            fig=plt.figure()
            plt.title("Negative Polarity")
            # Negative sentiment word cloud
            word_cloud(df['text'][df['Positive'] < df['Negative']])
            save_path = os.path.join(data_folder, "cloud_neg_pol.png")
            plt.savefig(save_path)
        except:
            fig = plt.figure()
            plt.title("No Negative polarity chart Formed")
            save_path = os.path.join(data_folder, "cloud_neg_pol.png")
            fig.savefig(save_path)
            print("No cloud because no Positive and Negative values present")

        try:
            fig = plt.figure()
            plt.title("Positive Polarity")
            #Positive sentiment word cloud
            word_cloud(df['text'][df['Positive'] > df['Negative']])
            save_path = os.path.join(data_folder, "cloud_pos_pol.png")
            plt.savefig(save_path)
        except:
            fig = plt.figure()
            plt.title("No Positive polarity chart Formed")
            save_path = os.path.join(data_folder, "cloud_pos_pol.png")
            fig.savefig(save_path)
            print("No cloud because no positive and negative values present")

        try:
            fig=plt.figure()
            #Neutral cloud
            plt.title("Neutral Polarity")
            word_cloud(df['text'][df['Positive'] == df['Negative']])
            save_path = os.path.join(data_folder, "cloud_neu_pol.png")
            plt.savefig(save_path)
        except:
            fig=plt.figure()
            plt.title("No Neutral polarity chart Formed")
            save_path = os.path.join(data_folder, "cloud_neu_pol.png")
            fig.savefig(save_path)
            print("No cloud because no positive and negative values present")

        return 'success'
    else:
        fig = plt.figure()
        plt.title("Error occured")
        save_path = os.path.join(data_folder, "error.png")
        fig.savefig(save_path)
        print("Cannot get Tokenized sents since dataframe empty")
        return 'error'

def run_analysis_check():

    auth = tweepy.OAuth2BearerHandler(bearer_token)
    auth.apply_auth()
    api = tweepy.API(auth)
    tweets = tweepy.Cursor(api.search_tweets, q="elon", count=200, tweet_mode = 'extended').items(10)
    # api = tweepy.API(auth)
    for tweet in tweets:
        print(tweet)
