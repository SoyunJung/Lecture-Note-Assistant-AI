from openai import OpenAI

STT_API_KEY = "" # OpenAI API key
client = OpenAI(api_key = STT_API_KEY)

transcript = """""" # 강의록

slide_text_converted = """""" # 교재

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
    {"role": "user", "content": "강의록: " + transcript + "\n\n교재: " + slide_text_converted}
    ]
)

print(completion.choices[0].message.content)