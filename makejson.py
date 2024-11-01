from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime
import io
import base64
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import requests
import numpy as np
from sklearn.linear_model import LinearRegression
app = Flask(__name__)
CORS(app)

@app.route('/api/data', methods=['POST'])
def get_data():
    data = request.json
    arr = []
    if not data:
        return jsonify({"error": "No data provided"}), 400
    print(data)
    Largevalue = data.get('Largevalue', 'Unknown')
    midvalue = data.get('midvalue', 'Unknown')

    for i in range(10):
        # url = f"https://at.agromarket.kr/openApi/price/dateWhsalPumSale.do?serviceKey=D5C1EC499CB04B1295FBB05434FB3592&apiType=json&pageNo=1&strDate={int(datetime.today().strftime('%Y%m%d')) - i - 4}&endDate={int(datetime.today().strftime('%Y%m%d')) - i - 3}&large={Largevalue}&mid={midvalue}"
        url = f"https://at.agromarket.kr/openApi/price/dateWhsalPumSale.do?serviceKey=BE89EA5AFF424DF4B2F73E4774EC242B&apiType=json&pageNo=1&strDate={20240918 - i - 1}&endDate={20240918 - i}&large={Largevalue}&mid={midvalue}"
        response = requests.get(url)
        contents = response.text
        json_ob = json.loads(contents)
        print(url)
        for i in range(len(json_ob['data'])):
            if json_ob['data'][i]['whsalname'] == "서울가락":
                arr.append(json_ob["data"][i]["totamt"] / json_ob["data"][i]["totqty"])
    print(arr[0])
    # 입력 데이터(X)와 출력 데이터(y) 준비
    X = np.array(range(len(arr))).reshape(-1, 1)  # 인덱스를 입력으로 사용
    y = np.array(arr)  # 리스트 값을 출력으로 사용
    font_path = 'C:/Windows/Fonts/malgun.ttf'

    # 폰트 이름 가져오기
    font_name = fm.FontProperties(fname=font_path).get_name()

    # 폰트 설정
    plt.rc('font', family=font_name)

    # 선형 회귀 모델 학습
    model = LinearRegression()
    model.fit(X, y)

    # 다음 값 예측
    next_index = len(arr)
    predicted_value = model.predict(np.array([[next_index]]))

    print(f"다음 예측 값: {predicted_value[0]}")

    prearr = arr.copy()
    prearr.append(predicted_value[0])

    print(prearr)

    # 마지막 10개의 값 선택
    subset = prearr[-10:]

    # 전체 꺾은선 그리기 (모든 점을 파란색으로)
    plt.plot(subset, marker='o', color='blue')
    
    for i, value in enumerate(subset):
        plt.text(i, value + 500, f'{int(value)}', ha='center', fontsize=9)

    # 마지막 부분 강조 (마지막 점만 빨간색으로 표시)
    plt.plot(len(subset) - 1, subset[-1], marker='o', color='red', markersize=10)
    plt.text(len(subset) - 1, subset[-1] - 500, "예측값", ha='center', fontsize=9)

    # y축 범위 설정
    plt.ylim(min(prearr) - 1000, max(prearr) + 1000)

    # 그래프 제목 및 축 레이블 설정
    plt.title('최근 9일의 데이터와 다음날 예측 값')
    plt.xlabel('일차')
    plt.ylabel('원 / kg')
    # plt.xticks(rotation=90)
    # 이미지를 메모리로 저장
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    
    # 이미지를 base64로 인코딩
    img_base64 = base64.b64encode(img.getvalue()).decode('utf-8')
    print(arr)
    # 서버에서 간단한 처리 후 데이터 반환
    response = {
        'price' : f'{arr[0]}',
        'message': f'{Largevalue, midvalue}',
        'image' : img_base64,
        'status': 'success'
    }
    # print(response)
    return jsonify(response), 200

if __name__ == '__main__':
    app.run(debug=True)