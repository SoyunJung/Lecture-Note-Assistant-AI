from openai import OpenAI


STT_API_KEY = "" # OpenAI API KEY

client = OpenAI(api_key = STT_API_KEY)

audio_file_path = "C:/Users/SOYUN/Desktop/whisper_test/audio.mp3"
audio_file = open(audio_file_path, "rb")

transcript = client.audio.transcriptions.create(
    file = audio_file,
    model="whisper-1",
    language="ko",
    response_format="text",
)

print(transcript)
