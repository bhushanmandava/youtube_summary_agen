from openai import OpenAI
from youtube_transcript_api import YouTubeTranscriptApi
import re
from agents import Agent, function_tool, Runner, ItemHelpers, RunContextWrapper
from openai.types.responses import ResponseTextDeltaEvent
from dotenv import load_dotenv
import asyncio

# import environment variables from .env file
load_dotenv()

# OpenRouter API client
client = OpenAI(
  base_url="<your api base url>",
  api_key="<your Api key>",
)

# define instructions for the agent
instructions = "You provide help with tasks related to YouTube videos."

# function to fetch YouTube video transcript and format it
# @function_tool
def fetch_youtube_transcript(url: str) -> str:
    """
    Extract transcript with timestamps from a YouTube video URL and format it for LLM consumption
    
    Args:
        url (str): YouTube video URL
        
    Returns:
        str: Formatted transcript with timestamps, where each entry is on a new line
             in the format: "[MM:SS] Text"
    """
    # Extract video ID from URL
    video_id_pattern = r'(?:v=|\/)([0-9A-Za-z_-]{11}).*'
    video_id_match = re.search(video_id_pattern, url)
    
    if not video_id_match:
        raise ValueError("Invalid YouTube URL")
    
    video_id = video_id_match.group(1)
    
    try:
        # Fetch transcript from YouTube
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        
        if not transcript:
            raise ValueError("No transcript available for this video.")
        
        # Format each entry with timestamp and text
        formatted_entries = []
        for entry in transcript:
            minutes = int(entry['start'] // 60)
            seconds = int(entry['start'] % 60)
            timestamp = f"[{minutes:02d}:{seconds:02d}]"
            
            formatted_entry = f"{timestamp} {entry['text']}"
            formatted_entries.append(formatted_entry)
        
        # Join all entries with newlines
        return "\n".join(formatted_entries)
    
    except Exception as e:
        print(f"Error fetching transcript: {str(e)}")
        return f"Error: {str(e)}. This might be due to no available transcript for the video."


# Define the agent
agent = Agent(
    name="YouTube Transcript Agent",
    instructions=instructions,
    tools=[fetch_youtube_transcript],
)

# Function to interact with the LLM using the transcript content
async def main():
    input_items = []

    print("=== YouTube Transcript Agent ===")
    print("Type 'exit' to end the conversation")
    print("Ask me anything about YouTube videos!")

    while True:
        # Get user input (YouTube URL)
        user_input = input("\nYou: ").strip()
        
        # If 'exit', stop the conversation
        if user_input.lower() in ['exit', 'quit', 'bye']:
            print("\nGoodbye!")
            break
        
        # Check if the input is a valid YouTube URL
        if "youtube.com" not in user_input:
            print("\nPlease provide a valid YouTube URL.")
            continue
        
        input_items.append({"content": user_input, "role": "user"})
        
        print("\nFetching transcript...")

        # The tool fetches the transcript as part of the agent's workflow
        try:
            # Directly use the agent's tool to fetch the transcript
            result = fetch_youtube_transcript(user_input)
            # print(result)
            completion = client.chat.completions.create(
            extra_headers={
                "HTTP-Referer": "<YOUR_SITE_URL>", # Optional. Site URL for rankings on openrouter.ai.
                "X-Title": "<YOUR_SITE_NAME>", # Optional. Site title for rankings on openrouter.ai.
            },
            extra_body={},
            model="nvidia/llama-3.1-nemotron-nano-8b-v1:free",
            messages=[
                {
                "role": "user",
                "content": "summary of the transcript: " + result
                }
            ]
            )
            print(completion.choices[0].message.content)
        except Exception as e:
            print(f"Error: {e}")
            continue

        print("\n")  # Add a newline after each response

if __name__ == "__main__":
    asyncio.run(main())


# (video_summary) bhushanchowdary@bhushans-Mac-mini video_Summary % python maini.py
# === YouTube Transcript Agent ===
# Type 'exit' to end the conversation
# Ask me anything about YouTube videos!

# You: summary "https://www.youtube.com/watch?v=-BUs1CPHKfU&t=1s"       

# Fetching transcript...
# This transcript summarizes discussions around introducing AI agents as platform-assisted tools designed to reduce the manual effort in programmatically managing complex query routes through multiple stages. Interested readers will likely find information on:

# 1. Agentic systems and their key feature: enabling LLMs. (No specific feature listed here as it starts explaining this feature later as leverage in detail.)
# 2. The ABC type of software systems for customer support.
# 3. The role of AI agents in simplifying complex query routes.
# 4. The process of using LLMs with tool calls, integrating tools, and implementing instructions.

# The transcript also walks through a practical example of creating a YouTube Video Transcription Agent using OpenAI's Anthropic and Obloido Antimistral tools. The summary focuses on the agent's behavior, including handling asynchronous interactions, tool calls, and user input, and how these tools enable more straightforward, flexible solution design. 

# The summary provides value as it:
# - Logs the main qualifications of advanced users or people interested in integrating platform-assisted tools.
# - Highlights the novel approach in reducing expensive manual work or designing update content.
# - Lets other users or students know. 
# - Helps users allocate time based on what the video covers. (eg., if they are teaching this). 

# For easier understanding, clustered elements that refer to similar in-depth details have been provided by sections appropriately separated them.



# You: https://www.youtube.com/watch?v=kffacxfA7G4

# Fetching transcript...
# Here is a summary and analysis of the transcript in sections:

# ---

# ### **Opening Songs**

# * **[00:00-02:09]** **"Baby"**
#   - Released: 2001 (The Black Dot completist remix includes a slight cut from "Don't Know" from the-video "Don't Know the Feeling" but it's more likely the remix of the "Don't Know" from the "Days Go On Forever" but the actual song's start is here)

# ### **First Verse of "Ooh Whoa"**

# * **[00:02-05]** and **[00:09-14]**
#   - Mode of Address: Narrative (Storytelling) as speaking of one's own or someone's romantic feelings.
#   - **Content**: Expressing a strong emotional attachment ("You know you love me, I know you care", "Just shout whenever and I'll be there) and a profound connection (e.g., "You are my love, you are my heart" emphasizing emotional dependence.

# ### **Verse 2 of "Ooh Whoa"**

# * **[00:21-28]**
#   - Mode of Address: Narrative
#   - **Content**: Reiterating romantic feelings ("Are we an item?", "We're just friends,") and setting boundaries ("Are we an item?"). 

# ### **Verse 1: "Girl Quit Playin'"**

# * **[00:30-39]**
#   - Mode of Address: Narrative (Storytelling, possibly with a slight shift to poetry)
#   - **Content**: A contemplative introspection ("I thought you'd always be mine, mine"). This suggests a mature, realistic view of the relationship, acknowledging the possibility of separation ("their heart is breakin'").

# ### **Verse 2: "Like Baby, Baby, Baby..."**

# * **[00:44-54] and [01:02-50] and [02:00-02:53] and **[03:00-3:54]**
#   - Mode of Address: Narrative (maintained, possibly with a slight lyrical twist in the remix, but primarily consistent)
#   - **Content**: Repetitions of "Like baby, baby, baby..." suggest longing and nostalgia over a lost or fading relationship, and a longer conversation. It's quite repetitive and seems to be the core of what the song is getting at: a long and strong attachment that is now waning.

# ### **Verse 3 of "Ooh Whoa" and Repetitions:**

# * **[00:55-59], [01:05-9], [1:21-25], [02:11-49], [02:13-49], and [03:00-3:54]**
#   - Mode of Address: Narrative; **Content**: Referencing the expectation of the relationship ("I thought you'd always be mine") and setting high emotional expectations. While the content is somewhat similar, it's noted for its dreamlike quality, suggesting a lasting commitment.

# ### **Bridge and the Second Verse:**

# * **[01:16-20] and [02:38-42, 5] and [03:06-49]**
#   - Mode of Address: Narrative
#   - **Content**: Directly addressing the long-term impact of the relationship ("Oh for you, I would have done whatever", "And I just can't believe we ain't together"), focusing on the urgency and sadness over separation. The feeling of longing and the desire to be reunited are clear.

# ### **Verse of "My First Love Broke My Heart...":**

# * **[00:39-59]** (before the repetition)
#   - **Content**: This suggests an emotional depth where the first love had a significant impact on the speaker's heart for the first time, showcasing a learning curve in emotional attachment.

# **Overall Summary:**

# - The song appears structured around a **longer conversation** between two individuals who have a deep, emotional connection but are facing challenges or a desire to separate.
# - **Repeating themes**: 
#   - **Deep emotional attachment**: various forms of declaring undying love for one person ("You are my love...", "baby, baby, baby...", "my first love broke my heart").
#   - **Practical consideration**: expressions of setting boundaries, expectations for long-term commitment.
#   - **Separation anxiety and longing**: multiple verses highlighting the speaker's sadness over being apart, either directly ("And I just can't believe we ain't together") or implicitly through boundaries and repeated longing statements.
# - **Remix and sampling aspects**:
#   - Repetition and repetition of the same line structures.
#   - The song is made more melodic with drums and restructured to sound more polished and emotive, possibly with a further reduction in the song length and alteration of the tone to produce a more catchy and memorable track.

# **Conclusion**: This song is meant to impress listeners with its melodic flair while the lyrics provide a narrative of long deep-felt connections and the challenges of separating. The remix streamlines the song and makes it more catchy for a broader audience. The song might have a specific social context where relationships are considered with lasting commitment, but not always in the habit of making commitment. **Note**: Detailed breakdown with the remix of a specific version would be beneficial as different songs and renditions can have unique aspects. The provided transcript is likely sourced from "The Days Go On Forever" (though slightly modified) and the remix might integrate other elements or reuse parts of the original. The emphasis in the remix could be to craft a more succinct and memorable version. 

# ---

# **Please Note**:
# - The summary relies on the broad assumption of the original song's context (specific identity, sample, impression of the relationship in the original material). Inferring musical notes, production elements, or additional inspired responses is beyond the provided text. 
# - The Will Testament has not been identified based on the transcript clip, which only contains an extract of the third verse's remix. Recognizing the song as "My First Love Broke My Heart" is an assumption. 

# For a more precise analysis, please provide the original song if possible or use additional tactics such as social media context if the Original Song Appears on Sources.



# You: 