import cv2
import mediapipe as mp
import imutils
import random
from math import hypot
import os
import numpy

class Image_Filter_Generator():
    def __init__(self):
        self.cap = cv2.VideoCapture(0)

    def overlay(self, image, x, y, w, h, overlay_image): # 대상 이미지, x, y 좌표, width, height, 덮어씌울 이미지
        try:
            alpha = overlay_image[:, :, 3] # BGRA
            mask_image = alpha / 255 # 0 ~ 255 => 0 ~ 1(1: 불투명, 0: 투명)

            for c in range(0, 3):
                image[y-h:y+h, x-w:x+w, c] = (overlay_image[:,:,c] * mask_image) + (image[y-h:y+h, x-w:x+w, c] * (1 - mask_image))
        except:
            mask_image = 0.5
            for c in range(0, 3):
                image[y-h:y+h, x-w:x+w, c] = (overlay_image[:,:,c] * mask_image) + (image[y-h:y+h, x-w:x+w, c] * (1 - mask_image))

    def full_filter(self, Cam_Option, File_path, Image_width, Image_height):
        # 얼굴을 찾고, 찾은 얼굴에 표시를 해주기 위한 변수 정의
        mp_face_detection = mp.solutions.face_detection # 얼굴 검출을 위한 face_detection 모듈
        mp_drawing = mp.solutions.drawing_utils # 얼굴의 특징을 그리기 위한 drawing_utils 모듈

        pwd = os.path.dirname(__file__) # 현재 스크립트 파일의 디렉토리 경로 가져오기
        # 이미지 불러오기
        image_left_eye = cv2.imread(pwd +"\\samples\\left_eye_1.png", cv2.IMREAD_UNCHANGED)
        image_right_eye = cv2.imread(pwd +"\\samples\\right_eye_1.png", cv2.IMREAD_UNCHANGED)
        image_nose_tip = cv2.imread(pwd +"\\samples\\nose_tip_1.png", cv2.IMREAD_UNCHANGED)
        with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
            while Cam_Option.isOpened():
                success, image = Cam_Option.read()
                success, image2 = Cam_Option.read()
                image2 = imutils.resize(image2, width = Image_width, height = Image_height)
                image = imutils.resize(image, width = Image_width, height = Image_height)
                if not success:
                    break

                # To improve performance, optionally mark the image as nots writeable to pass by reference
                image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = face_detection.process(image)

                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                if results.detections:
                    # 6개 특징 : 오른쪽 눈, 왼쪽 눈, 코 끝부분, 입 중심, 오른쪽 귀, 왼쪽 귀
                    for detection in results.detections:
                        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        # mp_drawing.draw_detection(image, detection)

                        keypoints = detection.location_data.relative_keypoints
                        right_eye = keypoints[0]
                        left_eye = keypoints[1]
                        nose_tip = keypoints[2]

                        h, w, _ = image.shape # height, width, channel
                        right_eye = (int(right_eye.x * w) - int(abs((keypoints[0].x - keypoints[1].x) * 0.5) * w), int(right_eye.y * h) - int(abs((keypoints[0].y - keypoints[3].y) * 1) * h))
                        left_eye = (int(left_eye.x * w) + int(abs((keypoints[0].x - keypoints[1].x) * 0.5) * w), int(left_eye.y * h) - int(abs((keypoints[0].y - keypoints[3].y) * 1) * h))
                        nose_tip = (int(nose_tip.x * w), int(nose_tip.y * h))

                        # 양 눈에 동그라미 그리기
                        # cv2.circle(image, right_eye, 50, (255, 0, 0), 10, cv2.LINE_AA) # 파란색
                        # cv2.circle(image, right_eye, 50, (0, 255, 0), 10, cv2.LINE_AA) # 초록색
                        # cv2.circle(image, nose_tip, 75, (0, 255, 255), 10, cv2.LINE_AA) # 노란색
                        ear_size = int(abs(keypoints[0].x * w - keypoints[1].x * w) * 1)
                        nose_size = int(abs(keypoints[0].x * w - keypoints[1].x * w) * 1)
                        if ear_size != ear_size // 2 * 2 :
                            ear_input_size = ear_size // 2
                            ear_size = ear_input_size * 2
                        else :
                            ear_input_size = ear_size // 2
                        
                        if nose_size != nose_size // 2 * 2 :
                            nose_input_size = nose_size // 2
                            nose_size = nose_input_size * 2
                        else :
                            nose_input_size = nose_size // 2
                        image_left_eye = cv2.resize(image_left_eye, dsize = (ear_size, ear_size))
                        image_right_eye = cv2.resize(image_right_eye, dsize = (ear_size, ear_size))
                        image_nose_tip = cv2.resize(image_nose_tip, dsize = (nose_size, nose_size))

                        # image[right_eye[1] - 50 : right_eye[1] + 50, right_eye[0] - 50 : right_eye[0] + 50] = image_right_eye
                        # image[left_eye[1] - 50 : left_eye[1] + 50, left_eye[0] -50 : left_eye[0] + 50] = image_left_eye
                        # image[nose_tip[1] - 75 : nose_tip[1] + 75, nose_tip[0] - 75 : nose_tip[0] + 75] = image_nose_tip
                        self.overlay(image, *right_eye, ear_input_size, ear_input_size, image_left_eye)
                        self.overlay(image, *left_eye, ear_input_size, ear_input_size, image_right_eye)
                        self.overlay(image, *nose_tip, nose_input_size, nose_input_size, image_nose_tip)
                # Flip the image horizontally for a selfie-view display
                image = cv2.resize(image, None, fx = 1, fy = 1)
                image2 = cv2.resize(image2, None, fx = 1, fy = 1)
                cv2.imshow("Full Face Filter Generator", image)

                if cv2.waitKey(1) == ord('q'):
                    cv2.imwrite(File_path + "_gt.png", image2)
                    cv2.imwrite(File_path + "full_filter.png", image)
                    break
        Cam_Option.release()
        cv2.destroyAllWindows()

    def glitter_filter(self, Cam_Option, File_path, Image_width, Image_height, filter_level):
        temp_idx = 0
        frequency = filter_level # 0 보다 큰 정수
        eye_list = [242, 238, 94, 370, 362, 458, 462, 289, 455, 439, 235, 219, 94, 407, 408, 292, 306, 324, 318, 95, 77, 183, 191, 184, 76, 62, 81, 41, 38, 82, 12, 13, 312, 268, 271, 310, 272, 7, 246, 247, 30, 29, 46, 53, 52, 65, 55, 124, 113, 225, 247, 151, 33, 150, 7, 25, 110, 24, 23, 22, 26, 226, 35, 31, 156, 124, 113, 150, 140, 130, 120, 110, 180, 160, 190, 362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 463, 341, 256, 252, 253, 254, 339, 255, 359, 467, 260, 259, 257, 258, 286, 414, 413, 441, 442, 443, 444, 445, 342, 446, 261, 448, 449, 450, 451, 452, 453, 464, 465, 357, 350, 349, 348, 449, 448, 261, 265, 133, 173, 157, 158, 159, 160, 161, 162, 33, 7, 163, 144, 145, 153, 154, 155, 112, 243, 190, 56, 28, 27, 29, 30, 247, 226, 31, 228, 229, 230, 231, 232, 233, 244, 189, 221, 222, 223, 224, 226, 124, 35, 31, 228, 229, 230, 231, 232, 233, 245, 55, 53, 46, 156, 143, 111, 117, 118, 119, 120, 121, 128, 188]
        # video load : 웹캠 비디오를 캡처합니다.
        image_gliter = cv2.imread(pwd + "\\samples\\gliter.png", cv2.IMREAD_UNCHANGED)
        pwd = os.path.dirname(__file__) 

        # mediapipe function
        mpDraw = mp.solutions.drawing_utils
        mpFaceMesh = mp.solutions.face_mesh
        faceMesh = mpFaceMesh.FaceMesh(max_num_faces=2)
        drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=1)

        # detect landmark fram video
        while Cam_Option.isOpened():
            # read Video
            ret, frame = Cam_Option.read()
            ret, frame2 = Cam_Option.read()

            # Frame Resizing
            frame = imutils.resize(frame, width = Image_width, height = Image_height)
            frame2 = imutils.resize(frame2, width = Image_width, height = Image_height)

            # To improve performance, optionally mark the image as nots writeable to pass by reference
            frame.flags.writeable = False
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = faceMesh.process(frame)

            frame.flags.writeable = True
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            gliter_size = int(max(numpy.round(len(frame) * 0.005,0), 1))
            
            if gliter_size % 2 != 0 :
                input_gliter_size = gliter_size // 2
                gliter_size = input_gliter_size * 2
            else :
                input_gliter_size = gliter_size // 2

            image_gliter = cv2.resize(image_gliter, dsize = (gliter_size, gliter_size))

            if results.multi_face_landmarks:
                # Overlay Landmarks on Face
                for face_landmarks in results.multi_face_landmarks:
                    # mpDraw.draw_landmarks(frame, face_landmarks, mpFaceMesh.FACEMESH_CONTOURS, drawSpec, drawSpec)
                    # Coord of Landmark
                    for id, lm in enumerate(face_landmarks.landmark):
                        if temp_idx % 50 == 0 :
                            temp_list = []
                            for i in range(1, len(face_landmarks.landmark) + 1):
                                temp_list.append(random.randint(0, frequency))
                        ih, iw, ic = frame.shape
                        x, y = int(lm.x*iw), int(lm.y*ih)
                        if temp_list[id] == 0 and id not in eye_list:
                            self.overlay(frame, x, y, input_gliter_size, input_gliter_size, image_gliter)
                    temp_idx += 1
                    # print(temp_idx)
                        # print nums
                        # cv2.putText(frame, str(id), (x, y), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
            # Video
            frame = cv2.resize(frame, None, fx = 1, fy = 1)
            frame2 = cv2.resize(frame2, None, fx = 1, fy = 1)
            cv2.imshow("Glitter Filter Generator", frame)

            # q 입력시 종료
            if cv2.waitKey(1) == ord('q'):
                cv2.imwrite(File_path + "_gt.png", frame2)
                cv2.imwrite(File_path + "_glitter.png", frame)
                break
        Cam_Option.release()
        cv2.destroyAllWindows()

    def nose_filter(self, Cam_Option, File_path, Image_width, Image_height):
        # 영상 및 돼지코 이미지 로드
        pwd = os.path.dirname(__file__) 
        nose_img = cv2.imread(pwd + '/samples/pig_nose.png') 

        # 5개의 center nose landmark point
        nose_landmarks = [49,279,197,2,5] 

        # mediapipe 호출
        mpFaceMesh = mp.solutions.face_mesh 
        faceMesh = mpFaceMesh.FaceMesh(max_num_faces=4) # max_num_faces로 영상에서 인식할 얼굴 개수 설정

        # 특정 조건이 충족될때까지 영상 재생 esc버튼 종료시까지
        while True:
            # 영상 읽기
            ret, frame = Cam_Option.read()
            ret, frame2 = Cam_Option.read()
            # frame 크기 조정
            frame = imutils.resize(frame, width = Image_width, height = Image_height)
            frame2 = imutils.resize(frame2, width = Image_width, height = Image_height)

            #frame에서 facemesh 검출
            results = faceMesh.process(frame)
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # nose landmarks
                    left_nose_x = 0
                    left_nose_y = 0
                    right_nose_x = 0
                    right_nose_y = 0
                    center_nose_x = 0
                    center_nose_y = 0
                    
                    # 각 랜드마크의 정보 가져오기
                    for id, lm in enumerate(face_landmarks.landmark):
                        # frame의 height, width, channel
                        h, w, c = frame.shape
                        # 랜드마크의 x좌표와 frame의 width를 곱해주어 x로 설정,랜드마크의 y좌표와 frame의 height을 곱하여 y좌표
                        x, y = int(lm.x * w), int(lm.y * h)
                        
                        # 앞서 설정한 nose_landmark와 일치하는 랜드마크 넘버에 대해 x,y좌표 부여
                        if id == nose_landmarks[0]:
                            left_nose_x, left_nose_y = x, y
                        if id == nose_landmarks[1]:
                            right_nose_x, right_nose_y = x, y
                        if id == nose_landmarks[4]:
                            center_nose_x, center_nose_y = x, y
                
                    # nose_width 계산
                    nose_width = int(hypot(left_nose_x-right_nose_x, left_nose_y-right_nose_y*1.2))
                    nose_height = int(nose_width*0.77)
                    
                    # nose_width와 nose_height가 0이 아닐 때 돼지코 이미지를 해당 크기에 맞게 resize
                    if (nose_width and nose_height) != 0:
                        pig_nose = cv2.resize(nose_img, (nose_width, nose_height))
                    
                    # nose_area 구하기
                    top_left = (int(center_nose_x-nose_width/2),int(center_nose_y-nose_height/2))
                    bottom_right = (int(center_nose_x+nose_width/2),int(center_nose_y+nose_height/2))

                    nose_area = frame[
                        top_left[1]: top_left[1]+nose_height,
                        top_left[0]: top_left[0]+nose_width
                    ]

                    # nose mask 생성
                    pig_nose_gray = cv2.cvtColor(pig_nose, cv2.COLOR_BGR2GRAY)
                    _, nose_mask = cv2.threshold(pig_nose_gray, 25, 255, cv2.THRESH_BINARY_INV)
                    no_nose = cv2.bitwise_and(nose_area, nose_area, mask=nose_mask)
                    
                    # no_nose에 pig nose 중첩
                    final_nose = cv2.add(no_nose, pig_nose)
                    # pig nose filter를 영상에 적용
                    frame[
                        top_left[1]: top_left[1]+nose_height,
                        top_left[0]: top_left[0]+nose_width
                    ] = final_nose

            # 변경된 이미지 출력
            frame = cv2.resize(frame, None, fx = 1, fy = 1)
            frame2 = cv2.resize(frame2, None, fx = 1, fy = 1)
            cv2.imshow("Nose Filter Generator", cv2.resize(frame, None, fx = 1, fy = 1))
            # q 입력시 종료
            if cv2.waitKey(1) == ord('q'):
                cv2.imwrite(File_path + "_nose_filter.png", frame)
                cv2.imwrite(File_path + "_gt.png", frame2)
                break
        Cam_Option.release()
        cv2.destroyAllWindows()

# 테스트
if __name__ == "__main__":
    filter_gen = Image_Filter_Generator()
    filter_gen.full_filter(filter_gen.cap, "test", 500, 500)
    #filter_gen.glitter_filter(filter_gen.cap, "glitter_test", 500, 500, 1)
    #filter_gen.nose_filter(filter_gen.cap, "nose_filter_output.png", 500, 500)