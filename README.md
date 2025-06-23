# capstone-filter-model

커뮤니티 게시글 내 **비방** 또는 **혐오 표현** 여부를 판별하는 텍스트 분류 모델입니다.

- 실제 서비스 환경과 유사하다고 판단되는 [2runo/Curse-detection-data](https://github.com/2runo/Curse-detection-data.git)에 포함된 커뮤니티형 데이터셋을 활용하여 학습하였습니다.

- **검증용 데이터셋**은 직접 에브리타임 키워드 검색을 통해 게시글(제목 + 본문)을 수집한 뒤, 라벨링을 진행하였습니다.  
  전체 학습 데이터의 약 **10% 규모**로 구성되어 있습니다.

- 지역명 및 특정 대학교명 등 민감 정보는 모두 OO으로 처리하였습니다

검증용 데이터셋은 작성자가 수작업으로 라벨링한 결과이므로, 사용자에 따라 "혐오" 혹은 "비방"의 판단 기준과 다를 수 있으며, 잘못 라벨링 된 데이터가 존재할 수 있습니다.

이 모델은 **커뮤니티 운영 자동화**, **신고 시스템 보조**, **사전 차단 시스템 구축** 등의 목적에 활용될 수 있습니다.

## Directory structure
```
├── data
│   ├── dataset.txt
│   └── validation.json
├── trained_model # 학습된 모델 및 관련 파일 저장 디렉토리
│   ├── config.json
│   ├── model.safetensors
│   ├── special_tokens_map.json
│   ├── tokenizer_config.json
│   ├── tokenizer.json
│   ├── training_args.bin
│   └── vocab.txt
├── utils
│   └── s3_util.py
├── train.py
├── predict.py
├── preprocess.py
├── settings.py
├── .gitattributes
├── .gitignore
└── README.md