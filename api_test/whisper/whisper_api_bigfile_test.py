from pydub import AudioSegment
from openai import OpenAI
import time
import os
import math


STT_API_KEY = "" # OpenAI API key

client = OpenAI(api_key=STT_API_KEY)

audio_file_path = ""  # 용량 큰 mp3 경로
song = AudioSegment.from_mp3(audio_file_path)

# PyDub은 milliseconds 단위
ten_minutes = 10 * 60 * 1000
segments = []

# 파일 업로드 경로 설정
upload_dir = "stt/"
if not os.path.exists(upload_dir):
    os.makedirs(upload_dir)

# 분할된 오디오 파일 저장
for i in range(0, len(song), ten_minutes):
    segment = song[i:i + ten_minutes]
    segment_path = f"stt/segment_{i // ten_minutes}.mp3"
    segment.export(segment_path, format="mp3")
    segments.append(segment_path)

# 최종 transcript를 저장할 파일 경로
transcript_file_path = "stt/transcript.txt"

# 빈 파일 생성
with open(transcript_file_path, "w", encoding="utf-8") as transcript_file:
    pass

# Whisper API로 분할된 파일들 처리
for segment_path in segments:
    with open(segment_path, "rb") as audio_file:
        try:
            response = client.audio.transcriptions.create(
                file=audio_file,
                model="whisper-1",
                language="ko",
                response_format="text",
            )
            # 응답 확인
            print(response)
            if response:
                with open(transcript_file_path, "a", encoding="utf-8") as transcript_file:
                    transcript_file.write(response + "\n")
            else:
                print("Empty response received.")
        except Exception as e:
            print(f"Error occurred: {e}")

        # API 오류 방지_잠시 대기
        time.sleep(1)

# 최종 transcript 출력
def split_txtfile(file_path, min_chunk_size=5000): # 5000자로 나눔
    with open(file_path, "r", encoding="utf-8") as txt_file:
        text = txt_file.read()
    
    total_length = len(text)
    num_chunks = math.ceil(total_length / min_chunk_size)
    
    # txt file을 chunk들로 쪼갬
    chunk_size = math.ceil(total_length / num_chunks)
    chunks = tuple(text[i:i + chunk_size] for i in range(0, total_length, chunk_size))
    
    return chunks

# 사용
transcript_file_path = "stt/transcript.txt"
transcript = split_txtfile(transcript_file_path)

# 모든 chunk를 print
for i, chunk in enumerate(transcript):
    print(chunk)
