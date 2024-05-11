from openai import OpenAI


STT_API_KEY = "api key" # api key

client = OpenAI(api_key = STT_API_KEY)

audio_file_path = "audio file directory" # directory
audio_file = open(audio_file_path, "rb")

transcript = client.audio.transcriptions.create(
    file = audio_file,
    model="whisper-1",
    language="ko",
    response_format="text",
)

print(transcript)