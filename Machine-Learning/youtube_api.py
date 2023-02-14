#%%

import pandas as pd
import requests
import json

# Define API key
api_key = 'AIzaSyCbJyx1zklwXS0bIXbvUnGtAV8-aZw036M'

# Define video id
video_id = 'lW508pBeih8'

# Construct API request URL
url = f'https://www.googleapis.com/youtube/v3/videos?part=snippet&id={video_id}&key={api_key}'

# Make API request
response = requests.get(url)

# Parse API response
data = json.loads(response.text)

# Extract relevant data into a dictionary
video_data = {}
video_data['title'] = data['items'][0]['snippet']['title']
video_data['description'] = data['items'][0]['snippet']['description']
video_data['published_at'] = data['items'][0]['snippet']['publishedAt']

# Convert dictionary to a Pandas DataFrame
df = pd.DataFrame(video_data, index=[0])

# Preview DataFrame
print(df.head())

# %%
import requests
import json

# Define API key
api_key = 'AIzaSyCbJyx1zklwXS0bIXbvUnGtAV8-aZw036M'

# Define the channel id
channel_id = 'UCXD5aOFDPO1W264sJvYc85Q'

# Define the maxResults parameter to limit the number of results returned
max_results = 100

# Construct API request URL to search for videos
url = f'https://www.googleapis.com/youtube/v3/search?part=id&channelId={channel_id}&type=video&maxResults={max_results}&key={api_key}'

# Make API request
response = requests.get(url)

# Check if the request was successful
if response.status_code == 200:
    # Parse API response
    data = json.loads(response.text)

    # Extract the video IDs from the response
    video_ids = [item['id']['videoId'] for item in data['items']]

    # Join the video IDs into a string, separated by commas
    video_id_string = ','.join(video_ids)

    # Construct another API request URL to retrieve the video details
    url = f'https://www.googleapis.com/youtube/v3/videos?part=snippet,statistics&id={video_id_string}&key={api_key}'

    # Make another API request
    response = requests.get(url)

    # Check if the request was successful
    if response.status_code == 200:
        # Parse API response
        data = json.loads(response.text)

        # Extract relevant data into a list of dictionaries
        videos = []
        for item in data['items']:
            video = {}
            video['title'] = item['snippet']['title']
            video['description'] = item['snippet']['description']
            video['published_at'] = item['snippet']['publishedAt']
            video['views'] = item['statistics']['viewCount']
            video['likes'] = item['statistics']['likeCount']
            video['comments'] = item['statistics']['commentCount']
            videos.append(video)

        # Convert list of dictionaries to a Pandas DataFrame
        import pandas as pd
        df = pd.DataFrame(videos)

        # Preview DataFrame
        print(df.head())
    else:
        print("Error: Failed to retrieve video details")
else:
    print("Error: Failed to search for videos")
# %%

import matplotlib.pyplot as plt


df.published_at = pd.to_datetime(df.published_at)
df.likes = df.likes.astype(int)
df.views = df.views.astype(int)


plt.plot(df.index, df.views)

# %%
