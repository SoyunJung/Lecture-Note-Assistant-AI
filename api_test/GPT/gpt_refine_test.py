from openai import OpenAI

NLP_API_KEY = "" # OpenAI API Key
client = OpenAI(api_key = NLP_API_KEY)


completion = client.chat.completions.create(
  model="gpt-4o",
  messages=[
    {"role": "system", "content": """당신은 맞춤법을 교정하는 데 도움이 되는 어시스턴트입니다.
당신의 임무는 Speech-to-Text로 받아쓰기한 텍스트에서 맞춤법의 불일치를 수정하는 것입니다.
잘못 받아쓰기한 단어가 있다면 옳은 단어로 수정하십시오. 맞춤법 오류를 수정하십시오. 그리고 주어진 맥락만을 사용하여 마침표, 쉼표, 대문자 표기와 같은 필수적인 구두점을 추가하십시오. 
이 강의에서 사용한 교과서 텍스트도 제공해드릴테니, 받아쓰기한 텍스트를 수정할 때 교과서의 내용을 참고해도 좋습니다. 둘은 유기적으로 연결된 강의록과 교과서 관계니까요.
"""},
    {"role": "user", "content": "강의록: " + raw_transcript + "\n\n교재: " + slide_text_converted}
  ]
)

print(completion.choices[0].message)
