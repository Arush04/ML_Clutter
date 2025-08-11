import requests
from bs4 import BeautifulSoup
from yt_dlp import YoutubeDL

def download_youtube_vid(url, filename):
  # Path to drive folder
  gdrive_path = "/content/drive/My Drive/TTS_Data/audio"
  try:
    audio_opts = {
        'format': 'ba[language=hi]/bestaudio/best',
        'outtmpl': f'{gdrive_path}/{filename}.%(ext)s',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'keepvideo': False,
    }
    with YoutubeDL(audio_opts) as ydl:
        ydl.download([url])
    return "Downloaded"
  except Exception as e:
    return e

def get_data(request_data, flag):
    if flag == "song":
        soup_per_song = BeautifulSoup(request_data.text, "html.parser")
        for script in soup_per_song(["script", "style"]):
            script.extract()

        ###################################
        # Getting data for youtube search #
        ###################################
        tbody = soup_per_song.find("tbody")
        rows = tbody.find_all("tr")
        data = {}

        for row in rows:
            cols = row.find_all("td")
            if len(cols) == 2:
                key = cols[0].get_text(strip=True)
                value = cols[1].get_text(strip=True)

                # Normalize the key
                normalized_key = key.replace(" ", "")
                if normalized_key == "📌SongTitle":
                    data["📌Song Title"] = value  # store in consistent format
                elif normalized_key == "🎤Singer":
                    data["🎤Singer"] = value

        # Combine into a single string
        print(f"data>>>>{data}")
        search_string = f"{data['📌Song Title']} {data['🎤Singer']}"

        ###################################
        #       Getting hindi lyrics      #
        ###################################
        x = soup_per_song.find("main")
        lyrics = x.find("div", id="hindilyrics")
        return data['📌Song Title'], search_string, lyrics.text

    elif flag == "song_list":
        #########################################
        # Getting links for each page of lyrics #
        #########################################
        soup = BeautifulSoup(request_data.text, "html.parser")
        for script in soup(["script", "style"]):
            script.extract()
        x = soup.find("main")
        articles = x.find_all("a")
        links = list(set(
            a.get("href") for a in articles
            if a.get("href") and ("/tag/" or "/category/") not in a.get("href")
        ))
        return links

    else:
        return "invalid flag"

def get_top_video_url(search_query):
    try:
        request = youtube.search().list(
            q=search_query,
            part="snippet",
            type="video",
            maxResults=1
        )
        response = request.execute()
        if response["items"]:
            video_id = response["items"][0]["id"]["videoId"]
            return f"https://www.youtube.com/watch?v={video_id}"
    except Exception as e:
        print(f"❌ YouTube search failed for '{search_query}': {e}")
    return None


# === Main execution with error handling ===
url_data = requests.get("https://sonylyrics.com/category/lyrics/hindi-songs-lyrics/page/5/")
songs_lists = get_data(url_data, "song_list")

lyrics_path = '/content/drive/My Drive/TTS_Data/lyrics'
os.makedirs(lyrics_path, exist_ok=True)

for song_url in songs_lists:
    try:
        song_name, search_string, lyrics = get_data(requests.get(song_url), flag="song")
        print(f"🎵 Processing: {song_name}")

        # Save lyrics (even if video fails)
        try:
            with open(os.path.join(lyrics_path, f'{song_name}.txt'), 'w', encoding="utf-8") as f:
                f.write(lyrics)
            print(f"✅ Lyrics saved for: {song_name}")
        except Exception as e:
            print(f"❌ Failed to save lyrics for '{song_name}': {e}")

        # Download video (even if lyrics fail)
        try:
            video_url = get_top_video_url(search_string)
            if video_url:
                download_youtube_vid(video_url, song_name)
                print(f"✅ Video downloaded for: {song_name}")
            else:
                print(f"⚠️ No YouTube results for: {search_string}")
        except Exception as e:
            print(f"❌ Failed to download video for '{song_name}': {e}")

    except Exception as e:
        print(f"❌ Skipping '{song_url}' due to error: {e}")
