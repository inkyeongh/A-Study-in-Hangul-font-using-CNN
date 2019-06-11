# A-Study-in-Hangul-font-using-CNN
A study in Hangul font characteristics using convolutional neural networks

**A study in Hangul font characteristics using convolutional neural networks**의 데이터 생성 및 예제 코드 공유를 위한 자료


### 폴더의 구성
1. data
    - *commonly_used_hangul.csv* : 국립국어 연구원이 발표한 자주 쓰이는 한국어 5,888개의 기초 낱말 모음로 관련 자료는 https://ko.wiktionary.org/wiki/%EB%B6%80%EB%A1%9D:%EC%9E%90%EC%A3%BC_%EC%93%B0%EC%9D%B4%EB%8A%94_%ED%95%9C%EA%B5%AD%EC%96%B4_%EB%82%B1%EB%A7%90_5800 에서 다운로드 가능
1. font
    - 논문에 사용된 한글 서체 모음
1. code
    1. *data image check.ipynb* : 학습 데이터 생성 전 문자 이미지를 어떻게 구성하는지 확인할 수 있음
    1. *generate_function.py* : 학습 데이터 생성에 필요한 내용을 함수화한 코드
    1. *generate_dataset.ipynb, .py* : 학습 데이터를 생성하는 코드
    1. *hangul_model.py* : cnn 모형을 함수화한 코드(lasagne version code)
    1. *hangul_cnn_setting.py* : cnn 모형을 학습하는데 필요한 코드
    1. *hangul_cnn_lasagne_code.py* : cnn 모형 학습 코드
