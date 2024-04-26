## 디렉토리 및 데이터셋 설정
1. data폴더는 경로 설정에 따라 바뀔 수 있음.
2. 단 train 폴더는 디렉토리 구조와 폴더명을 그대로 사용해야 하며, a 폴더에는 target image(GT), b 폴더에는 filtered(마스킹된) 이미지가 들어가야 함. (sample이미지를 삭제하고 사용할 것)
3. a, b 폴더 내의 pair이미지(필터 적용 여부 이외엔 서로 동일한 이미지) 간 파일명은 서로 일치해야 함.
4. 학습된 모델 가중치 .pt파일이 model_weight안에 있어야 실행됨.
5. .py파일들은 모두 같은 경로에 존재해야 함.

## 코드 경로 설정
1. dataset.py, model.py, optimizer.py 에서는 경로 설정 불필요
2. train.py 파일에서 train set 경로, 가중치 저장 폴더 경로가 지정되어야 함.
3. restore.py 파일에서 model_wieght 파일 경로, input이미지 폴더 경로, output이미지 폴더 경로가 지정되어야 함.

## 사용
0. 코드 내에서 경로 설정(코드 경로 설정 참고)
1. optimzer.py 에서 모델 학습 파라미터 조정
2. train.py 실행하여 모델 학습
3. restore.py 에서 model_weight 파일 경로 지정 후 실행

## 이미지 파일 확장자
train 및 input 이미지로는 아래와 같은 형식을 지원함(PIL.open() 함수에서 지원하는 형식)

- BMP (Windows Bitmap)
- EPS (Encapsulated Postscript)
- GIF (Graphics Interchange Format)
- ICO (Windows Icon)
- JPEG (Joint Photographic Experts Group)
- PNG (Portable Network Graphics)
- PPM (Portable Pixmap)
- TIFF (Tagged Image File Format)
- WebP (Google WebP Image Format)

% output 형식은 jpg로 지정해두었으나 restore.py에서 주석 처리된 코드를 바꾸는 것으로 확장자 제한을 간단하게 없앨 수 있게 되어있음.
