import time
import json

from PIL import Image # import Image

import requests
import uuid

secret_key = 'secret_key' # SECRET_KEY
api_url = 'api_url'

pdf_file = 'pdf_test.pdf'

request_json = {
    'images': [
        {
            'format': 'jpg',
            'name': 'demo'
        }
    ],
    'requestId': str(uuid.uuid4()),
    'version': 'V2',
    'timestamp': int(round(time.time() * 1000))
}

payload = {'message': json.dumps(request_json).encode('UTF-8')}
files = [
  ('file', open(pdf_file,'rb'))
]
headers = {
  'X-OCR-SECRET': secret_key
}

response = requests.request("POST", api_url, headers=headers, data = payload, files = files)
# print(response.text.encode('utf8')) # checking

# JSON -> korean
response_json = response.json()
for image in response_json['images']:
    print(f"이미지 이름: {image['name']}")
    print(f"인식 결과: {image['inferResult']}")
    print("인식된 텍스트:")
    for field in image['fields']:
        print(f"- {field['inferText']}")
    print("-" * 20)