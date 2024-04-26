## 디렉토리 설정
1. 디렉토리는 pix2pix 폴더와 동일한 구조로 이루어져 있어야 함. 다만 data폴더는 경로 설정에 따라 바뀔 수 있음.
2. 단 train 폴더는 디렉토리 구조와 폴더명을 그대로 사용해야 함. a에는 target image(GT), b에는 filtered(마스킹된) 이미지가 들어가야 함. (코드 실행 여부 확인을 위해 sample이미지는 아무렇게나 넣어두었음.)
3. a, b 폴더 내의 pair 이미지의 파일명은 서로 일치해야 함.
4. 학습된 모델 가중치 .pt파일은 model_weight에 넣어야 함.
5. scripts 내의 .py파일들은 모두 같은 폴더에 있어야 함.

## 코드 경로 설정
1. dataset.py, model.py, optimizer.py 에서는 경로 설정 불필요
2. train.py 파일에서 train set 경로, 가중치 저장 폴더 경로가 지정되어야 함.
3. restore.py 파일에서 model_wieght 파일 경로, input이미지 폴더 경로, output이미지 폴더 경로가 지정되어야 함.

## 사용
0. 코드 내에서 경로 설정
1. optimzer.py 에서 모델 학습 파라미터 조정
2. train.py 실행하여 모델 학습
3. restore.py 에서 model_weight 파일 경로 지정 후 실행

## 이미지 파일 확장자
train 및 input 이미지로는 아래와 같은 형식을 지원함(PIL.open() 함수에서 지원하는 형식)

BMP (Windows Bitmap)
EPS (Encapsulated Postscript)
GIF (Graphics Interchange Format)
ICO (Windows Icon)
JPEG (Joint Photographic Experts Group)
PNG (Portable Network Graphics)
PPM (Portable Pixmap)
TIFF (Tagged Image File Format)
WebP (Google WebP Image Format)

output 형식은 jpg로 지정해두었으나 restore.py에서 주석 처리된 코드를 바꾸는 것으로 확장자 제한을 간단하게 없앨 수 있게 되어있음.