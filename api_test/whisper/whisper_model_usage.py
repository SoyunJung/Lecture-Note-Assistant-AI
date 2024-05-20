import whisper
import re

def transcribe_audio(file_path, model_size='large', language='Korean'):
    model = whisper.load_model(model_size)

    # 오디오 파일에서 음성 인식
    result = model.transcribe(file_path, language=language)
    return result['text']

def remove_timestamps(text):
    # [시간 --> 시간] 패턴 제거
    pattern = r"\[\d{2}:\d{2}:\d{2} --> \d{2}:\d{2}:\d{2}\]"
    return re.sub(pattern, "", text)

# 사용 예제
audio_file = "" # 오디오 파일 경로 삽입
transcription = transcribe_audio(audio_file)
cleaned_transcription = remove_timestamps(transcription)

# 결과를 텍스트 파일에 저장
with open("", "w", encoding="utf-8") as file: # 텍스트 파일 경로 삽입
    file.write(cleaned_transcription)
