# MrWhosethebossShorts Youtube Analysis
This is an EDA of MrWhosethebossShorts, one of my favourite Youtubers. In this analysis, I employed youtube API service to gather data from the channel.Amidst the rise of short video clips, Youtube short is proving to be a force to reckon with.

This is MrWhosetheboss....https://www.youtube.com/user/Mrwhosetheboss
### Importing the neccessary libraries needed
# data analysis packages
from googleapiclient.discovery import build
import pandas as pd
import seaborn as sns
from IPython.display import JSON
import numpy as np
from dateutil import parser

# Data viz packages
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib as mpl

# NLP
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
nltk.download('punkt')
from wordcloud import WordCloud

# Duration converter
import isodate
### Assessing Youtube API
Youtube API requires a Google account. Youtube Developer API 
# Youtube API Key can be assessed from Youtube Developer Console
api_key = '***********************'
# this list can be used for other youtube channel also
channel_ids = ['UCZSlfzadjnw7G419c_OJ9eg'
              ]
api_service_name = "youtube"
api_version = "v3"


# Get credentials and create an API client
youtube = build(api_service_name, api_version, developerKey=api_key)
# JSON display to understand the root path of the file

request = youtube.channels().list(
        part="snippet,contentDetails,statistics",
        id=','.join(channel_ids)
    )

response = request.execute()

JSON(response)
# define a function to assess the channel stats

def get_channel_stats(youtube, channel_ids):
    
    all_data = []
    
    request = youtube.channels().list(
        part="snippet,contentDetails,statistics",
        id=','.join(channel_ids)
    )
    response = request.execute()

    # loop through items
    for item in response['items']:
        data = {'channelName': item['snippet']['title'],
                'subscribers': item['statistics']['subscriberCount'],
                'views': item['statistics']['viewCount'],
                'totalVideos': item['statistics']['videoCount'],
                'playlistId': item['contentDetails']['relatedPlaylists']['uploads'],
                'dateCreated': item['snippet']['publishedAt']
               }
        
        all_data.append(data)
        
    return(pd.DataFrame(all_data))
channel_stats = get_channel_stats(youtube, channel_ids)
channel_stats
##### As of 16th of June, 2022, MrWhosethebossShorts has 144 videos
#### Overall Stats of MrWhosetheboss Shorts channel
numeric_cols = ['views', 'subscribers', 'totalVideos']
channel_stats[numeric_cols] = channel_stats[numeric_cols].apply(pd.to_numeric, errors = 'coerce', axis = 1)
channel_stats['viewsPerSubscriber'] = np.where(channel_stats['subscribers'] != 0, channel_stats['views']//channel_stats['subscribers'],0)
channel_stats
# Playlist ID of MRWhosethebossShorts to identify the videoIDS

playlist_id="UUZSlfzadjnw7G419c_OJ9eg"
request = youtube.playlistItems().list(
        part="contentDetails,snippet,status",
        playlistId=playlist_id,
        maxResults = 50
    )
response = request.execute()
JSON(response)
def get_video_ids(youtube, playlist_id):
    
    video_ids = []
    
    request = youtube.playlistItems().list(
        part="snippet,contentDetails",
        playlistId=playlist_id,
        maxResults = 50
    )
    response = request.execute()
    
    for item in response['items']:
        video_ids.append(item['contentDetails']['videoId'])
        
    next_page_token = response.get('nextPageToken')
    
    while next_page_token is not None:
        request = youtube.playlistItems().list(
                    part='contentDetails',
                    playlistId = playlist_id,
                    maxResults = 50,
                    pageToken = next_page_token)
        response = request.execute()

        for item in response['items']:
            video_ids.append(item['contentDetails']['videoId'])

        next_page_token = response.get('nextPageToken')
        
    return video_ids

def get_video_details(youtube, video_ids):

    all_video_info = []
    
    for i in range(0, len(video_ids), 50):
        request = youtube.videos().list(
            part="snippet,contentDetails,statistics",
            id=','.join(video_ids[i:i+50])
        )
        response = request.execute() 

        for video in response['items']:
            stats_to_keep = {'snippet': ['channelTitle', 'title', 'description', 'tags', 'publishedAt'],
                             'statistics': ['viewCount', 'likeCount', 'favouriteCount', 'commentCount'],
                             'contentDetails': ['duration', 'definition', 'caption']
                            }
            video_info = {}
            video_info['video_id'] = video['id']

            for k in stats_to_keep.keys():
                for v in stats_to_keep[k]:
                    try:
                        video_info[v] = video[k][v]
                    except:
                        video_info[v] = None

            all_video_info.append(video_info)
    
    return pd.DataFrame(all_video_info)
video_ids = get_video_ids(youtube, playlist_id)
len(video_ids)
# Get video details
video_df = get_video_details(youtube, video_ids)
video_df.tail()
video_df.info()
### Pre-processing the dataframe for analysis
numeric_cols = ['viewCount', 'likeCount', 'favouriteCount','commentCount']
video_df[numeric_cols] = video_df[numeric_cols].apply(pd.to_numeric, errors = 'coerce', axis = 1)
video_df.info()
# convert duration to seconds
video_df['durationSecs'] = video_df['duration'].apply(lambda x: isodate.parse_duration(x))
video_df['durationSecs'] = video_df['durationSecs'].astype('timedelta64[s]')
# Add tag count
video_df['tagCount'] = video_df['tags'].apply(lambda x: 0 if x is None else len(x))
# include a date column
video_df['date'] = pd.to_datetime(video_df['publishedAt'])
video_df['date'] = video_df['date'].apply(lambda x: x.strftime('%Y-%m-%d'))
video_df['date'] = pd.to_datetime(video_df['date'])
# Publish day in the week
video_df['publishedAt'] = video_df['publishedAt'].apply(lambda x: parser.parse(x)) 
video_df['publishDayName'] = video_df['publishedAt'].apply(lambda x: x.strftime("%A")) 
# sort dataFrame by date, from the oldest
video_df = video_df.sort_values(by='date', ascending=True)
video_df.sample(2)
### Best Performing short videos of MrWhosethebossShorts
ax = sns.barplot(x = 'title', y = 'viewCount', data = video_df.sort_values('viewCount', ascending=False)[0:14])
plot = ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
plot = ax.set_title('Best performing MrWhosetheboss Shorts', size=16)
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos:'{:,.0f}'.format(x/1000000) + 'M'))
ax = sns.barplot(x = 'title', y = 'viewCount', data = video_df.sort_values('viewCount', ascending=True)[0:9])
plot = ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
plot = ax.set_title('MrWhosetheboss bottom 10 shorts', size=16)
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos:'{:,.0f}'.format(x/1000000) + 'M'))
### Most used words by MrWhosetheboss
stop_words = set(stopwords.words('english'))
video_df['title_no_stopwords'] = video_df['title'].apply(lambda x: [item for item in str(x).split() if item not in stop_words])

all_words = list([a for b in video_df['title_no_stopwords'].tolist() for a in b])
all_words_str = ' '.join(all_words) 

def plot_cloud(wordcloud):
    plt.figure(figsize=(30, 20))
    plt.imshow(wordcloud) 
    plt.axis("off");

wordcloud = WordCloud(width = 500, height = 250, random_state=1, background_color='white', 
                      colormap='viridis', collocations=False).generate(all_words_str)
plot_cloud(wordcloud)
### Upload Schedule 
day_df = pd.DataFrame(video_df['publishDayName'].value_counts())
weekdays = [ 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
day_df = day_df.reindex(weekdays)
### Duration of MrWhosetheboss Shorts
sns.histplot(data = video_df, x = 'durationSecs', bins=30)
### Violin Chart of MrWhosetheboss shorts view count
sns.violinplot(video_df['channelTitle'], video_df['viewCount'])
### Scatter plot on the relationship between View counts, like and comments
fig, ax = plt.subplots(1,2)
sns.scatterplot(data = video_df, x = 'commentCount', y = 'viewCount', ax = ax[0])
sns.scatterplot(data = video_df, x = 'likeCount', y = 'viewCount', ax = ax[1])
### Monthly distribution of Views and Like
# Prepare data

video_df['month'] = [d.strftime('%b') for d in video_df.date]
months = video_df['month'].unique()

# Draw Plot
ax = sns.boxplot(x='month', y='viewCount', data=video_df)


# Set Title
ax.set_title('Monthly View Performance', fontsize=18)
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos:'{:,.0f}'.format(x/1000000) + 'M'))

plt.show()
# Prepare data

video_df['month'] = [d.strftime('%b') for d in video_df.date]
months = video_df['month'].unique()

# Draw Plot
ax = sns.boxplot(x='month', y='likeCount', data=video_df)


# Set Title
ax.set_title('Monthly Likes Performance', fontsize=18)
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos:'{:,.0f}'.format(x/1000000) + 'M'))

plt.show()
### Trend Analysis of Views and Like on MrWhosethebossShorts
# Draw Plot
def plot_df(video_df, x, y, title="", xlabel='Date', ylabel='viewCount', dpi=100):
    plt.figure(figsize=(16,5), dpi=dpi)
    plt.plot(x, y, color='tab:red')
    plt.gca().set(title=title, xlabel=xlabel, ylabel=ylabel)
    plt.show()

plot_df(video_df, x=video_df.date, y=video_df.viewCount, title='Time Series of Views on MrWhosetheboss Shorts') 
# Draw Plot
def plot_df(video_df, x, y, title="", xlabel='Date', ylabel='likeCount', dpi=100):
    plt.figure(figsize=(16,5), dpi=dpi)
    plt.plot(x, y, color='tab:blue')
    plt.gca().set(title=title, xlabel=xlabel, ylabel=ylabel)
    plt.show()

plot_df(video_df, x=video_df.date, y=video_df.likeCount, title='Time Series of Likes on MrWhosetheboss Shorts') 
video_df['likesPerView'] = np.where(video_df['viewCount'] != 0, video_df['likeCount']/video_df['viewCount'],0)
# Draw Plot
def plot_df(video_df, x, y, title="", xlabel='Date', ylabel='likesPerView', dpi=100):
    plt.figure(figsize=(16,5), dpi=dpi)
    plt.plot(x, y, color='tab:blue')
    plt.gca().set(title=title, xlabel=xlabel, ylabel=ylabel)
    plt.show()

plot_df(video_df, x=video_df.date, y=video_df.likesPerView, title='Trend of Likes Per View on MrWhosetheboss Shorts') 
### There has been a steady rise in likes per view of MrWhostheBossShorts.
Contact Info:
    
### LinkedIn: Adeoti Sheriffdeen 
### Twitter: @SheriffHolla
### Contact me at s.adeoti86@gmail.com

