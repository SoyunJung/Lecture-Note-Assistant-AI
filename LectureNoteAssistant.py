import gradio as gr
from PyPDF2 import PdfReader
import re
import fitz  # PyMuPDF
import os

from openai import OpenAI

from pydub import AudioSegment # 오디오 슬라이싱
import math

import time
import json
from PIL import Image
import requests
import uuid

from pathlib import Path

def speech_to_text(audio_file, openai_api_key): # raw_transcript 텍스트를 튜플로 return함

    STT_API_KEY = openai_api_key
    client = OpenAI(api_key=STT_API_KEY)

    song = AudioSegment.from_mp3(audio_file)

    # PyDub에서 시간은 ms단위로 카운트
    ten_minutes = 10 * 60 * 1000
    segments = []

    # 파일 업로드 경로 설정
    upload_dir = "/mnt/data/stt/"
    if not os.path.exists(upload_dir):
        os.makedirs(upload_dir)

    # 분할된 오디오 파일 저장
    for i in range(0, len(song), ten_minutes):
        segment = song[i:i + ten_minutes]
        segment_path = f"/mnt/data/stt/segment_{i // ten_minutes}.mp3"
        segment.export(segment_path, format="mp3")
        segments.append(segment_path)

    # 최종 raw_transcript를 저장할 파일 경로
    raw_transcript_file_path = "/mnt/data/stt/raw_transcript.txt"

    # 빈 파일 생성
    with open(raw_transcript_file_path, "w", encoding="utf-8") as raw_transcript_file:
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
                    with open(raw_transcript_file_path, "a", encoding="utf-8") as raw_transcript_file:
                        raw_transcript_file.write(response + "\n")
                else:
                    print("Empty response received.")
            except Exception as e:
                print(f"Error occurred: {e}")

            # API 오류 방지를 위해 1초 대기
            time.sleep(1)

    # raw_transcript 출력
    def split_txtfile(file_path, min_chunk_size=4000): # 4000자로 나눔
        with open(file_path, "r", encoding="utf-8") as txt_file:
            text = txt_file.read()
        
        total_length = len(text)
        num_chunks = math.ceil(total_length / min_chunk_size)
        
        # txt file을 chunk들로 쪼갬
        chunk_size = math.ceil(total_length / num_chunks)
        chunks = tuple(text[i:i + chunk_size] for i in range(0, total_length, chunk_size))
        
        return chunks

    raw_transcript = split_txtfile(raw_transcript_file_path)

    # 모든 chunk를 print
    for i, chunk in enumerate(raw_transcript):
        print(f"{i}번째 STT 내용: {chunk}")
    print(f"raw_transcript 원소 개수: {len(raw_transcript)}")

    return raw_transcript, "\n".join(raw_transcript)


def ocr_slide_text(pdf_file, ocr_api_key): # num_pages와 page_texts(dictionary)를 튜플로 return함
    secret_key_ocr = ocr_api_key
    api_url_ocr = 'https://6pfb41u4zq.apigw.ntruss.com/custom/v1/30851/1891e1fe857cbe3bd2c4f29f5fc24ef11956164d7d5ef1135925f3d227a8b617/general'
    
    # PDF 파일 열기
    pdf = fitz.open(pdf_file)
    num_pages = pdf.page_count

    # 각 페이지에서 텍스트 추출
    page_texts = {}

    # 페이지를 10개씩 나누기
    for i in range(0, num_pages, 10):
        chunk_end = min(i + 10, num_pages)
        for page_num in range(i, chunk_end):
            # 페이지를 이미지로 변환
            page = pdf.load_page(page_num)
            pix = page.get_pixmap()
            image_path = f'page_{page_num+1}.png'
            pix.save(image_path)

            with open(image_path, 'rb') as image_file:
                request_json = {
                    'images': [
                        {
                            'format': 'png',
                            'name': f'page_{page_num+1}'
                        }
                    ],
                    'requestId': str(uuid.uuid4()),
                    'version': 'V2',
                    'timestamp': int(round(time.time() * 1000))
                }

                payload = {'message': json.dumps(request_json).encode('UTF-8')}
                files = [
                    ('file', image_file)
                ]
                headers = {
                    'X-OCR-SECRET': secret_key_ocr
                }

                response = requests.request("POST", api_url_ocr, headers=headers, data=payload, files=files)
                response_json = response.json()

                page_text = ''
                if 'images' in response_json:
                    for image in response_json['images']:
                        for field in image['fields']:
                            page_text += field['inferText'] + ' '
                    page_texts[page_num+1] = page_text.strip()  # 페이지 번호를 1부터 시작
                else:
                    print(f"Error in OCR for page {page_num+1}: {response_json}")

            # 이미지 파일 삭제
            os.remove(image_path)

    return num_pages, page_texts

# 체크박스의 상태에 따라, 텍스트를 긁어오는 방식을 위 두 가지 (ocr, extract) 중 선택한 후
# 아래의 녹취록 다듬기 함수를 호출함
def refine_transcript_wrapper(raw_transcript, pdf_file, use_ocr, openai_api_key, ocr_api_key):
    if use_ocr:
        slide_tuple = ocr_slide_text(pdf_file, ocr_api_key)
    else:
        slide_tuple = extract_slide_text(pdf_file)
        
    transcript = refine_transcript(raw_transcript, slide_tuple, openai_api_key)
    transcript_string = ""
    for string in transcript:
        transcript_string += string
    return transcript_string, transcript # 가독성 처리에 표시할 string 변수, 튜플타입의 transcript 변수

def download_ocr_result(ocr_result):
    return ocr_result


# 파일 업로드 경로 설정
upload_dir = "/mnt/data/uploads/"
if not os.path.exists(upload_dir):
    os.makedirs(upload_dir)


def pdf_length(pdf_path):# pdf 경로 입력 시 페이지 수를 반환
    # PDF 파일 열기
    pdf = PdfReader(pdf_path)
    # PDF 파일의 페이지 수 가져오기
    num_pages = len(pdf.pages)
    return num_pages


def extract_slide_text(pdf_file):
    # PDF/PPT 파일에서 텍스트를 추출하는 함수
    pdf_reader = PdfReader(pdf_file)
    num_pages = len(pdf_reader.pages)
    page_texts = {}

    for page_num in range(num_pages):
        page_text = pdf_reader.pages[page_num].extract_text()
        page_texts[page_num+1] = page_text.strip()

    return num_pages, page_texts

# 녹취록 다듬기
def refine_transcript(raw_transcript, slide_tuple, openai_api_key):

    #test print
    print(f"slide_tuple 원소 개수: {len(slide_tuple)}")
    print(f"slide_tuple 안의 딕셔너리의 원소 개수: {len(slide_tuple[1])}")

    num_pages, page_dict = slide_tuple
    result = ""
    for page_num in range(1, num_pages + 1):
        if page_num in page_dict:
            page_text = page_dict[page_num]
            result += f"({page_num}page)\n{page_text}\n\n"
    slide_text_converted = result.strip()

    NLP_API_KEY = openai_api_key
    client = OpenAI(api_key=NLP_API_KEY)

    transcript = [None] * len(raw_transcript)  # 미리 크기 설정
    for i, chunk in enumerate(raw_transcript):
        try:
            completion = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": """당신은 맞춤법을 교정하는 데 도움이 되는 어시스턴트입니다.
당신의 임무는 Speech-to-Text로 받아쓰기한 텍스트에서 맞춤법의 불일치를 수정하는 것입니다.
잘못 받아쓰기한 단어가 있다면 옳은 단어로 수정하십시오. 맞춤법 오류를 수정하십시오. 그리고 주어진 맥락만을 사용하여 마침표, 쉼표, 대문자 표기와 같은 필수적인 구두점을 추가하십시오. 
이 강의에서 사용한 교과서 텍스트도 제공해드릴테니, 받아쓰기한 텍스트를 수정할 때 교과서의 내용을 참고해도 좋습니다. 둘은 유기적으로 연결된 강의록과 교과서 관계니까요.
답변은 교정된 강의록만 제공하십시오. 교재, 인삿말, 대답 모두 금지합니다. 오로지 교정된 강의록의 내용만 제공하십시오."""},
                    {"role": "user", "content": "강의록: " + chunk + "\n\n교재: " + slide_text_converted}
                ]
            )

            transcript[i] = completion.choices[0].message.content
            print(f"gpt refine {i}번째 결과:\n{completion.choices[0].message.content}")

        except Exception as e:
            print(f"강의록 refine 중 오류 발생: {e}")

    return tuple(transcript)


# 녹취록과 슬라이드 텍스트 매칭
def match_transcript_and_slides(transcript, slide_tuple, openai_api_key): # string타입의 대본 # ocr_slide_text와 extract_slide text의 return 결과인 튜플타입의 slide_tuple
    # nlp 사용하기 전, 튜플로 들어온 slide_tuple 형식을 nlp 프롬프트에 작성한 양식의 string으로 변환
    num_pages, page_dict = slide_tuple
    result = ""
    for page_num in range(1, num_pages + 1):
        if page_num in page_dict:
            page_text = page_dict[page_num]
            result += f"({page_num}page)\n{page_text}\n\n"
    slide_text_converted = result.strip()

    NLP_API_KEY = openai_api_key
    client = OpenAI(api_key = NLP_API_KEY)

    matched_text = [None] * len(transcript) # 크기 설정
    for i, transcript_sliced in enumerate(transcript):
        try:
            completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": """당신은 강의 녹취록을 해당 교과서에 맞춰주는 일을 맡은 유용한 어시스턴트입니다.
            당신의 임무는 다음과 같습니다.
            주어진 자료:
            강의 녹취록을 포함한 텍스트
            교과서 텍스트를 포함한 텍스트 (각 페이지의 내용은 페이지 번호로 시작함)
            목표는 강의 녹취록의 각 문단을 교과서 텍스트의 해당 페이지 번호와 일치시키는 것입니다. 이를 위해 다음 단계를 따르세요:
            강의 녹취록의 각 문단을 읽으세요.
            내용을 비교하여 교과서 텍스트에서 가장 관련성 높은 문단을 찾으세요.
            강의 녹취록의 일치하는 문단 앞에 교과서 텍스트의 해당 페이지 번호를 붙이세요.
            예를 들어, 강의 녹취록의 한 문단이 교과서 42페이지의 내용과 일치하면 해당 문단 앞에 "(42page)"를 붙입니다.
            절대 \(\d+page\)의 양식을 벗어나서는 안됩니다.
            모든 문단의 앞에는 \(\d+page\)이 할당되어있어야합니다.
            출력 결과는 강의 녹취록 문단이 포함된 새로운 텍스트여야 하며, 각 문단 앞에는 교과서의 해당 페이지 번호가 붙어 있어야 합니다. 강의 녹취록과 교과서 텍스트 사이에 정확히 일치하는 단어가 항상 있는 것은 아니므로, 문단을 정확하게 매칭하려면 최선의 판단을 사용해야 할 수 있습니다.
            이외의 모든 잡담은 첨언하지 마십시오.
            인삿말, 대답 모두 금지합니다. 오로지 내용만 제공하십시오."""},
                {"role": "user", "content": "강의록: " + transcript_sliced + "\n\n교재: " + slide_text_converted}
            ]
            )

            matched_text[i] = completion.choices[0].message.content
            print("=============%d번째 매칭 수행 중============" %(i+1))
            print(f"gpt matched 결과:\n{completion.choices[0].message.content}")

        except Exception as e:
            print(f"matching 중 오류 발생: {e}")

    return tuple(matched_text)


# 매칭된 결과 표시
def display_matched_results(matched_text, pdf_file):
    images, textboxes = [], []
    doc = fitz.open(pdf_file)
    
    for i, matched_text_string in enumerate(matched_text):
        for text in matched_text_string.strip().split("\n\n"):
            if re.match(r"\(\d+page\)", text):
                page_num = int(re.search(r"\((\d+)page\)", text).group(1))
                print("page_num : ", page_num)
                content = re.sub(r"\(\d+page\)", "", text).strip()
                
                # PDF 페이지를 이미지로 변환
                page = doc.load_page(page_num - 1)  # 페이지 인덱스는 0부터 시작
                img_path = os.path.join(upload_dir, f"page_{page_num}.png")
                pix = page.get_pixmap()
                pix.save(img_path)
                
                img = gr.Image(value=img_path, label=f"{page_num}번째 페이지")
                txt = gr.Textbox(value=content.strip(), max_lines=10)
                
                images.append(img)
                textboxes.append(txt)
    
    # 필요한 개수만큼 빈 이미지와 텍스트박스를 추가
    for _ in range(50 - len(images)):
        images.append(gr.Image(label="Empty"))
    for _ in range(50 - len(textboxes)):
        textboxes.append(gr.Textbox(label="Empty"))

    return images + textboxes

# 매칭 수행
def match_only(transcript, pdf_file, use_ocr, openai_api_key, ocr_api_key):
    # 테스트용 transcript string삭제
    if use_ocr:
        print("use ocr")
        result_tuple = ocr_slide_text(pdf_file, ocr_api_key)
    else:
        print("use extract")
        result_tuple = extract_slide_text(pdf_file)
    print("\nreulst_tuple => ", result_tuple)
    print("\nmatch_only에 파라미터로 들어온 강의대본 : ", transcript)
    matched_text = match_transcript_and_slides(transcript, result_tuple, openai_api_key) # 반환값 : (Npage) + 띄어쓰기 한 칸 + stt내용. + \n\n
    matching_results = display_matched_results(matched_text, pdf_file)
    return matching_results + [matched_text]


# 파일 업로드 경로 업데이트
def update_uploaded_file(pdf_file):
    if pdf_file is not None:
        uploaded_file_path = pdf_file.name
        uploaded_file_name = os.path.basename(uploaded_file_path)
        return uploaded_file_path, uploaded_file_name
    return None, None

# 다운로드
def save_text_file(content, filename):
    file_path = os.path.join(upload_dir, filename)

    # content가 튜플일 경우 하나의 문자열로 변환
    if isinstance(content, tuple):
        content = "\n".join(content)

    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(content)

    return file_path

# 다운로드
def download_files(download_lecture_txt, download_refined_txt, download_matched_txt, raw_transcript, refined_transcript, matched_text):
    files = []

    # tuple일 경우 string으로 변환
    if isinstance(raw_transcript, tuple):
        raw_transcript = "\n".join(raw_transcript)
    if isinstance(refined_transcript, tuple):
        refined_transcript = "\n".join(refined_transcript)
    if isinstance(matched_text, tuple):
        matched_text = "\n".join(matched_text)

    if download_lecture_txt:
        lecture_file_path = save_text_file(raw_transcript, "강의록.txt")
        files.append(lecture_file_path)
    if download_refined_txt:
        refined_file_path = save_text_file(refined_transcript, "다듬은 강의록.txt")
        files.append(refined_file_path)
    if download_matched_txt:
        matched_file_path = save_text_file(matched_text, "문단과 매칭된 강의록.txt")
        files.append(matched_file_path)
    return files


def main():

    with gr.Blocks() as web:
        with gr.Row():
            gr.Markdown("# Lecture Note Assistant AI")
            openai_api_key = gr.Textbox(label="OpenAI API Key")
            ocr_api_key = gr.Textbox(label="OCR API Key")

        with gr.Tabs() as tabs:
            # 전체 작업 수행
            with gr.TabItem("모든 작업 수행"):
                with gr.Row():
                    with gr.Column():
                        audio_file = gr.Audio(label="강의 녹음 파일 업로드", type="filepath")
                        pdf_file = gr.File(label="강의 자료 PDF 업로드", file_types=[".pdf"])
                        pdf_file.upload(fn=update_uploaded_file, inputs=pdf_file, outputs=None)
                        use_ocr = gr.Checkbox(label="OCR 사용 (이미지 기반 PDF인 경우 권장)")
                        submit_btn = gr.Button("제출")
                        

                    with gr.Column():
                        tabs_inner = gr.Tabs()
                        with tabs_inner:
                            with gr.TabItem("Whisper STT 결과"):
                                stt_textbox = gr.Textbox(label="STT 결과")
                                next_btn1 = gr.Button("다음 단계")

                            with gr.TabItem("NLP 가독성 처리 결과"):
                                stt_claude_textbox = gr.Textbox(label="가독성 처리 결과")
                                next_btn2 = gr.Button("다음 단계")

                            with gr.TabItem("PDF 매칭 결과"):
                                matching_results = gr.Group()
                                images, textboxes = [], []
                                for i in range(50): #임의로50
                                    with gr.Row():
                                        img = gr.Image(label=f"{i+1}번째 페이지", interactive=False)
                                        txt = gr.Textbox(label=f"{i+1}번째 페이지 텍스트")
                                        images.append(img)
                                        textboxes.append(txt)
                                        img
                                        txt
                                download_lecture_txt = gr.Checkbox(label="강의록.txt")
                                download_refined_txt = gr.Checkbox(label="다듬은 강의록.txt")
                                download_matched_txt = gr.Checkbox(label="문단과 매칭된 강의록.txt")
                                download_btn = gr.Button("다운로드")
                
                matched_text = gr.State()
                stt_result = gr.State()
                claude_output = gr.State()


                submit_btn.click(fn= speech_to_text, inputs=[audio_file, openai_api_key], outputs=[stt_result, stt_textbox])
                next_btn1.click(fn= refine_transcript_wrapper, inputs=[stt_result, pdf_file, use_ocr, openai_api_key, ocr_api_key], outputs=[stt_claude_textbox, claude_output])
                next_btn2.click(fn= match_only, inputs=[claude_output, pdf_file, use_ocr, openai_api_key, ocr_api_key], outputs=images + textboxes + [matched_text])
                download_btn.click(fn= download_files, inputs=[download_lecture_txt, download_refined_txt, download_matched_txt, stt_textbox, claude_output, matched_text], outputs=gr.File(label="다운로드된 파일"))

            #only_tab 가져와서 붙여넣기

        gr.Markdown(
            """
            """
        )
    
    web.launch()

if __name__ == "__main__":
    main()
