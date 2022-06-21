{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# MrWhosethebossShorts Youtube Analysis"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "This is an EDA of MrWhosethebossShorts, one of my favourite Youtubers. In this analysis, I employed youtube API service to gather data from the channel.Amidst the rise of short video clips, Youtube short is proving to be a force to reckon with.\n",
    "\n",
    "This is MrWhosetheboss....https://www.youtube.com/user/Mrwhosetheboss"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Importing the neccessary libraries needed"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# data analysis packages\n",
    "from googleapiclient.discovery import build\n",
    "import pandas as pd\n",
    "import seaborn as sns\n",
    "from IPython.display import JSON\n",
    "import numpy as np\n",
    "from dateutil import parser\n",
    "\n",
    "# Data viz packages\n",
    "import seaborn as sns\n",
    "import matplotlib.pyplot as plt\n",
    "import matplotlib.ticker as ticker\n",
    "import matplotlib as mpl\n",
    "\n",
    "# NLP\n",
    "import nltk\n",
    "from nltk.corpus import stopwords\n",
    "from nltk.tokenize import word_tokenize\n",
    "nltk.download('stopwords')\n",
    "nltk.download('punkt')\n",
    "from wordcloud import WordCloud\n",
    "\n",
    "# Duration converter\n",
    "import isodate"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Assessing Youtube API"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Youtube API requires a Google account. Youtube Developer API "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Youtube API Key can be assessed from Youtube Developer Console\n",
    "api_key = '***********************'"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# this list can be used for other youtube channel also\n",
    "channel_ids = ['UCZSlfzadjnw7G419c_OJ9eg'\n",
    "              ]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "api_service_name = \"youtube\"\n",
    "api_version = \"v3\"\n",
    "\n",
    "\n",
    "# Get credentials and create an API client\n",
    "youtube = build(api_service_name, api_version, developerKey=api_key)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# JSON display to understand the root path of the file\n",
    "\n",
    "request = youtube.channels().list(\n",
    "        part=\"snippet,contentDetails,statistics\",\n",
    "        id=','.join(channel_ids)\n",
    "    )\n",
    "\n",
    "response = request.execute()\n",
    "\n",
    "JSON(response)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# define a function to assess the channel stats\n",
    "\n",
    "def get_channel_stats(youtube, channel_ids):\n",
    "    \n",
    "    all_data = []\n",
    "    \n",
    "    request = youtube.channels().list(\n",
    "        part=\"snippet,contentDetails,statistics\",\n",
    "        id=','.join(channel_ids)\n",
    "    )\n",
    "    response = request.execute()\n",
    "\n",
    "    # loop through items\n",
    "    for item in response['items']:\n",
    "        data = {'channelName': item['snippet']['title'],\n",
    "                'subscribers': item['statistics']['subscriberCount'],\n",
    "                'views': item['statistics']['viewCount'],\n",
    "                'totalVideos': item['statistics']['videoCount'],\n",
    "                'playlistId': item['contentDetails']['relatedPlaylists']['uploads'],\n",
    "                'dateCreated': item['snippet']['publishedAt']\n",
    "               }\n",
    "        \n",
    "        all_data.append(data)\n",
    "        \n",
    "    return(pd.DataFrame(all_data))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "channel_stats = get_channel_stats(youtube, channel_ids)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "channel_stats"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "##### As of 16th of June, 2022, MrWhosethebossShorts has 144 videos"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Overall Stats of MrWhosetheboss Shorts channel"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "numeric_cols = ['views', 'subscribers', 'totalVideos']\n",
    "channel_stats[numeric_cols] = channel_stats[numeric_cols].apply(pd.to_numeric, errors = 'coerce', axis = 1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "channel_stats['viewsPerSubscriber'] = np.where(channel_stats['subscribers'] != 0, channel_stats['views']//channel_stats['subscribers'],0)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "channel_stats"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Playlist ID of MRWhosethebossShorts to identify the videoIDS\n",
    "\n",
    "playlist_id=\"UUZSlfzadjnw7G419c_OJ9eg\""
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "request = youtube.playlistItems().list(\n",
    "        part=\"contentDetails,snippet,status\",\n",
    "        playlistId=playlist_id,\n",
    "        maxResults = 50\n",
    "    )\n",
    "response = request.execute()\n",
    "JSON(response)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "def get_video_ids(youtube, playlist_id):\n",
    "    \n",
    "    video_ids = []\n",
    "    \n",
    "    request = youtube.playlistItems().list(\n",
    "        part=\"snippet,contentDetails\",\n",
    "        playlistId=playlist_id,\n",
    "        maxResults = 50\n",
    "    )\n",
    "    response = request.execute()\n",
    "    \n",
    "    for item in response['items']:\n",
    "        video_ids.append(item['contentDetails']['videoId'])\n",
    "        \n",
    "    next_page_token = response.get('nextPageToken')\n",
    "    \n",
    "    while next_page_token is not None:\n",
    "        request = youtube.playlistItems().list(\n",
    "                    part='contentDetails',\n",
    "                    playlistId = playlist_id,\n",
    "                    maxResults = 50,\n",
    "                    pageToken = next_page_token)\n",
    "        response = request.execute()\n",
    "\n",
    "        for item in response['items']:\n",
    "            video_ids.append(item['contentDetails']['videoId'])\n",
    "\n",
    "        next_page_token = response.get('nextPageToken')\n",
    "        \n",
    "    return video_ids\n",
    "\n",
    "def get_video_details(youtube, video_ids):\n",
    "\n",
    "    all_video_info = []\n",
    "    \n",
    "    for i in range(0, len(video_ids), 50):\n",
    "        request = youtube.videos().list(\n",
    "            part=\"snippet,contentDetails,statistics\",\n",
    "            id=','.join(video_ids[i:i+50])\n",
    "        )\n",
    "        response = request.execute() \n",
    "\n",
    "        for video in response['items']:\n",
    "            stats_to_keep = {'snippet': ['channelTitle', 'title', 'description', 'tags', 'publishedAt'],\n",
    "                             'statistics': ['viewCount', 'likeCount', 'favouriteCount', 'commentCount'],\n",
    "                             'contentDetails': ['duration', 'definition', 'caption']\n",
    "                            }\n",
    "            video_info = {}\n",
    "            video_info['video_id'] = video['id']\n",
    "\n",
    "            for k in stats_to_keep.keys():\n",
    "                for v in stats_to_keep[k]:\n",
    "                    try:\n",
    "                        video_info[v] = video[k][v]\n",
    "                    except:\n",
    "                        video_info[v] = None\n",
    "\n",
    "            all_video_info.append(video_info)\n",
    "    \n",
    "    return pd.DataFrame(all_video_info)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "video_ids = get_video_ids(youtube, playlist_id)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "len(video_ids)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Get video details\n",
    "video_df = get_video_details(youtube, video_ids)\n",
    "video_df.tail()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "video_df.info()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Pre-processing the dataframe for analysis"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "numeric_cols = ['viewCount', 'likeCount', 'favouriteCount','commentCount']\n",
    "video_df[numeric_cols] = video_df[numeric_cols].apply(pd.to_numeric, errors = 'coerce', axis = 1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "video_df.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# convert duration to seconds\n",
    "video_df['durationSecs'] = video_df['duration'].apply(lambda x: isodate.parse_duration(x))\n",
    "video_df['durationSecs'] = video_df['durationSecs'].astype('timedelta64[s]')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Add tag count\n",
    "video_df['tagCount'] = video_df['tags'].apply(lambda x: 0 if x is None else len(x))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# include a date column\n",
    "video_df['date'] = pd.to_datetime(video_df['publishedAt'])\n",
    "video_df['date'] = video_df['date'].apply(lambda x: x.strftime('%Y-%m-%d'))\n",
    "video_df['date'] = pd.to_datetime(video_df['date'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Publish day in the week\n",
    "video_df['publishedAt'] = video_df['publishedAt'].apply(lambda x: parser.parse(x)) \n",
    "video_df['publishDayName'] = video_df['publishedAt'].apply(lambda x: x.strftime(\"%A\")) "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# sort dataFrame by date, from the oldest\n",
    "video_df = video_df.sort_values(by='date', ascending=True)\n",
    "video_df.sample(2)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Best Performing short videos of MrWhosethebossShorts"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "ax = sns.barplot(x = 'title', y = 'viewCount', data = video_df.sort_values('viewCount', ascending=False)[0:14])\n",
    "plot = ax.set_xticklabels(ax.get_xticklabels(), rotation=90)\n",
    "plot = ax.set_title('Best performing MrWhosetheboss Shorts', size=16)\n",
    "ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos:'{:,.0f}'.format(x/1000000) + 'M'))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "ax = sns.barplot(x = 'title', y = 'viewCount', data = video_df.sort_values('viewCount', ascending=True)[0:9])\n",
    "plot = ax.set_xticklabels(ax.get_xticklabels(), rotation=90)\n",
    "plot = ax.set_title('MrWhosetheboss bottom 10 shorts', size=16)\n",
    "ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos:'{:,.0f}'.format(x/1000000) + 'M'))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Most used words by MrWhosetheboss"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "stop_words = set(stopwords.words('english'))\n",
    "video_df['title_no_stopwords'] = video_df['title'].apply(lambda x: [item for item in str(x).split() if item not in stop_words])\n",
    "\n",
    "all_words = list([a for b in video_df['title_no_stopwords'].tolist() for a in b])\n",
    "all_words_str = ' '.join(all_words) \n",
    "\n",
    "def plot_cloud(wordcloud):\n",
    "    plt.figure(figsize=(30, 20))\n",
    "    plt.imshow(wordcloud) \n",
    "    plt.axis(\"off\");\n",
    "\n",
    "wordcloud = WordCloud(width = 500, height = 250, random_state=1, background_color='white', \n",
    "                      colormap='viridis', collocations=False).generate(all_words_str)\n",
    "plot_cloud(wordcloud)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Upload Schedule "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "scrolled": true
   },
   "outputs": [],
   "source": [
    "day_df = pd.DataFrame(video_df['publishDayName'].value_counts())\n",
    "weekdays = [ 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']\n",
    "day_df = day_df.reindex(weekdays)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Duration of MrWhosetheboss Shorts"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "sns.histplot(data = video_df, x = 'durationSecs', bins=30)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Violin Chart of MrWhosetheboss shorts view count"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "scrolled": true
   },
   "outputs": [],
   "source": [
    "sns.violinplot(video_df['channelTitle'], video_df['viewCount'])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Scatter plot on the relationship between View counts, like and comments"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "scrolled": true
   },
   "outputs": [],
   "source": [
    "fig, ax = plt.subplots(1,2)\n",
    "sns.scatterplot(data = video_df, x = 'commentCount', y = 'viewCount', ax = ax[0])\n",
    "sns.scatterplot(data = video_df, x = 'likeCount', y = 'viewCount', ax = ax[1])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Monthly distribution of Views and Like"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Prepare data\n",
    "\n",
    "video_df['month'] = [d.strftime('%b') for d in video_df.date]\n",
    "months = video_df['month'].unique()\n",
    "\n",
    "# Draw Plot\n",
    "ax = sns.boxplot(x='month', y='viewCount', data=video_df)\n",
    "\n",
    "\n",
    "# Set Title\n",
    "ax.set_title('Monthly View Performance', fontsize=18)\n",
    "ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos:'{:,.0f}'.format(x/1000000) + 'M'))\n",
    "\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Prepare data\n",
    "\n",
    "video_df['month'] = [d.strftime('%b') for d in video_df.date]\n",
    "months = video_df['month'].unique()\n",
    "\n",
    "# Draw Plot\n",
    "ax = sns.boxplot(x='month', y='likeCount', data=video_df)\n",
    "\n",
    "\n",
    "# Set Title\n",
    "ax.set_title('Monthly Likes Performance', fontsize=18)\n",
    "ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos:'{:,.0f}'.format(x/1000000) + 'M'))\n",
    "\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Trend Analysis of Views and Like on MrWhosethebossShorts"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Draw Plot\n",
    "def plot_df(video_df, x, y, title=\"\", xlabel='Date', ylabel='viewCount', dpi=100):\n",
    "    plt.figure(figsize=(16,5), dpi=dpi)\n",
    "    plt.plot(x, y, color='tab:red')\n",
    "    plt.gca().set(title=title, xlabel=xlabel, ylabel=ylabel)\n",
    "    plt.show()\n",
    "\n",
    "plot_df(video_df, x=video_df.date, y=video_df.viewCount, title='Time Series of Views on MrWhosetheboss Shorts') "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Draw Plot\n",
    "def plot_df(video_df, x, y, title=\"\", xlabel='Date', ylabel='likeCount', dpi=100):\n",
    "    plt.figure(figsize=(16,5), dpi=dpi)\n",
    "    plt.plot(x, y, color='tab:blue')\n",
    "    plt.gca().set(title=title, xlabel=xlabel, ylabel=ylabel)\n",
    "    plt.show()\n",
    "\n",
    "plot_df(video_df, x=video_df.date, y=video_df.likeCount, title='Time Series of Likes on MrWhosetheboss Shorts') "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "video_df['likesPerView'] = np.where(video_df['viewCount'] != 0, video_df['likeCount']/video_df['viewCount'],0)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Draw Plot\n",
    "def plot_df(video_df, x, y, title=\"\", xlabel='Date', ylabel='likesPerView', dpi=100):\n",
    "    plt.figure(figsize=(16,5), dpi=dpi)\n",
    "    plt.plot(x, y, color='tab:blue')\n",
    "    plt.gca().set(title=title, xlabel=xlabel, ylabel=ylabel)\n",
    "    plt.show()\n",
    "\n",
    "plot_df(video_df, x=video_df.date, y=video_df.likesPerView, title='Trend of Likes Per View on MrWhosetheboss Shorts') "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "### There has been a steady rise in likes per view of MrWhostheBossShorts."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Contact Info: \n",
    "    \n",
    "### LinkedIn: Adeoti Sheriffdeen \n",
    "### Twitter: @SheriffHolla\n",
    "### Contact me at s.adeoti86@gmail.com\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "interpreter": {
   "hash": "4233e328bbec369313ec7d1501781540994d17256ee39186eefe039e78eb09ef"
  },
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
