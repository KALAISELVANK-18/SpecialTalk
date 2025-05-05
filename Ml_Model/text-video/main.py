import uvicorn
from fastapi import FastAPI, HTTPException, Form
from fastapi.responses import StreamingResponse
from moviepy.editor import TextClip, CompositeVideoClip
from moviepy.video.compositing.concatenate import concatenate_videoclips
from moviepy.video.io.VideoFileClip import VideoFileClip

app = FastAPI()

from pydantic import BaseModel


from fuzzywuzzy import fuzz

video_names = [
    "0.mp4", "1.mp4", "2.mp4", "3.mp4", "4.mp4", "5.mp4", "6.mp4", "7.mp4", "8.mp4", "9.mp4",
    "A.mp4", "After.mp4", "Again.mp4", "Against.mp4", "Age.mp4", "All.mp4", "Alone.mp4", "Also.mp4", "And.mp4", "Ask.mp4",
    "At.mp4", "B.mp4", "Be.mp4", "Beautiful.mp4", "Before.mp4", "Best.mp4", "Better.mp4", "Busy.mp4", "But.mp4", "Bye.mp4",
    "C.mp4", "Can.mp4", "Cannot.mp4", "Change.mp4", "College.mp4", "Come.mp4", "Computer.mp4", "D.mp4", "Day.mp4", "Distance.mp4",
    "Do Not.mp4", "Do.mp4", "Does Not.mp4", "E.mp4", "Eat.mp4", "Engineer.mp4", "F.mp4", "Fight.mp4", "Finish.mp4", "From.mp4",
    "G.mp4", "Glitter.mp4", "Go.mp4", "God.mp4", "Gold.mp4", "Good.mp4", "Great.mp4", "H.mp4", "Hand.mp4", "Hands.mp4",
    "Happy.mp4", "Hello.mp4", "Help.mp4", "Her.mp4", "Here.mp4", "His.mp4", "Home.mp4", "Homepage.mp4", "How.mp4", "I.mp4",
    "Invent.mp4", "It.mp4", "J.mp4", "K.mp4", "Keep.mp4", "L.mp4", "Language.mp4", "Laugh.mp4", "Learn.mp4", "M.mp4", "ME.mp4",
    "More.mp4", "My.mp4", "N.mp4", "Name.mp4", "Next.mp4", "Not.mp4", "Now.mp4", "O.mp4", "Of.mp4", "On.mp4", "Our.mp4",
    "Out.mp4", "P.mp4", "Pretty.mp4", "Q.mp4", "R.mp4", "Right.mp4", "S.mp4", "Sad.mp4", "Safe.mp4", "See.mp4", "Self.mp4",
    "Sign.mp4","So.mp4", "Sound.mp4", "Stay.mp4", "Study.mp4", "T.mp4", "Talk.mp4", "Television.mp4", "Thank You.mp4",
    "Thank.mp4", "That.mp4", "They.mp4", "This.mp4", "Those.mp4", "Time.mp4", "To.mp4", "Type.mp4", "U.mp4", "Us.mp4", "V.mp4",
    "W.mp4", "Walk.mp4", "Wash.mp4", "Way.mp4", "We.mp4", "Welcome.mp4", "What.mp4", "When.mp4", "Where.mp4", "Which.mp4",
    "Who.mp4", "Whole.mp4", "Whose.mp4", "Why.mp4", "Will.mp4", "With.mp4", "Without.mp4", "Words.mp4", "Work.mp4", "World.mp4",
    "Wrong.mp4", "X.mp4", "Y.mp4", "You.mp4", "Your.mp4", "Yourself.mp4", "Z.mp4"
]
video_urls = ["video/" + name for name in video_names]

# Remove the ".mp4" extension and get only the name
video_names_without_extension = [name.split('.')[0] for name in video_names]

def convert(target_word):

    # Compute similarity scores for each word
    similarity_scores = {word: fuzz.partial_ratio(target_word, word) for word in video_names_without_extension}

    # Sort words by similarity score (highest to lowest)
    sorted_words = sorted(similarity_scores.items(), key=lambda x: x[1], reverse=True)

    # Print sorted words and their similarity scores
    for word, score in sorted_words:
        print(f"{word}: {score}%")
    return  True,sorted_words[0]

def capitalize_first_letters(input_string):
    words = input_string.split()


    capitalized_words = []

    for word in words:
        if("video/"+word.capitalize()+".mp4" in video_urls):
            capitalized_words.append("video/"+word.capitalize()+".mp4")

            for letter in word:
                capitalized_words.append("video/"+letter.capitalize()+".mp4")



    return list(capitalized_words)

def merge_videos(video_paths, output_path):
    # Load all video clips
    clips = [VideoFileClip(video_path) for video_path in video_paths]

    # Concatenate video clips
    final_clip = concatenate_videoclips(clips)

    # Write final clip to output file
    final_clip.write_videofile(output_path)

    # Close all clips
    for clip in clips:
        clip.close()

class VideoRequest(BaseModel):
    text: str



@app.post("/generate_video")
async def generate_video(text: VideoRequest):
    texts = text.text

    ans=capitalize_first_letters(texts)
    print(ans)
    # Example usage
    video_paths = ans
    output_path = "merged_vide.mp4"
    merge_videos(video_paths, output_path)

    output_path = "merged_vide.mp4"


    # Return the video file as a response
    return StreamingResponse(open(output_path, "rb"), media_type="video/mp4")
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)