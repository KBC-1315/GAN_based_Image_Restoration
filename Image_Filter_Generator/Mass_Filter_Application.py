import cv2
import mediapipe as mp
import imutils
import random
from math import hypot
import os
import numpy as np

class Image_Filter_Generator():
    def __init__(self):
        pass

    def overlay(self, image, x, y, w, h, overlay_image): 
        try:
            alpha = overlay_image[:, :, 3] 
            mask_image = alpha / 255 

            for c in range(0, 3):
                image[y-h:y+h, x-w:x+w, c] = (overlay_image[:,:,c] * mask_image) + (image[y-h:y+h, x-w:x+w, c] * (1 - mask_image))
        except:
            mask_image = 0.5
            for c in range(0, 3):
                image[y-h:y+h, x-w:x+w, c] = (overlay_image[:,:,c] * mask_image) + (image[y-h:y+h, x-w:x+w, c] * (1 - mask_image))

    def full_filter(self, folder_path, output_path, image_width, image_height):
        mp_face_detection = mp.solutions.face_detection 
        mp_drawing = mp.solutions.drawing_utils 

        image_left_eye = cv2.imread("./Image_Filter_Generator/samples/left_eye_1.png", cv2.IMREAD_UNCHANGED)
        image_right_eye = cv2.imread("./Image_Filter_Generator/samples/right_eye_1.png", cv2.IMREAD_UNCHANGED)
        image_nose_tip = cv2.imread("./Image_Filter_Generator/samples/nose_tip_1.png", cv2.IMREAD_UNCHANGED)

        with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
            for filename in os.listdir(folder_path):
                if filename.endswith(".png") or filename.endswith(".jpg"):
                    image = cv2.imread(os.path.join(folder_path, filename))
                    if image is not None:
                        image = imutils.resize(image, width=image_width, height=image_height)

                        image.flags.writeable = False
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        results = face_detection.process(image)

                        image.flags.writeable = True
                        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                        
                        if results.detections:
                            for detection in results.detections:
                                keypoints = detection.location_data.relative_keypoints
                                right_eye = keypoints[0]
                                left_eye = keypoints[1]
                                nose_tip = keypoints[2]

                                h, w, _ = image.shape 
                                right_eye = (int(right_eye.x * w) - int(abs((keypoints[0].x - keypoints[1].x) * 0.5) * w), int(right_eye.y * h) - int(abs((keypoints[0].y - keypoints[3].y) * 1) * h))
                                left_eye = (int(left_eye.x * w) + int(abs((keypoints[0].x - keypoints[1].x) * 0.5) * w), int(left_eye.y * h) - int(abs((keypoints[0].y - keypoints[3].y) * 1) * h))
                                nose_tip = (int(nose_tip.x * w), int(nose_tip.y * h))

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

                                self.overlay(image, *right_eye, ear_input_size, ear_input_size, image_left_eye)
                                self.overlay(image, *left_eye, ear_input_size, ear_input_size, image_right_eye)
                                self.overlay(image, *nose_tip, nose_input_size, nose_input_size, image_nose_tip)

                        cv2.imwrite(os.path.join(output_path, filename[:-4] + "_full_filter.png"), image)
                        print(filename, "full_filter : complete")

    def glitter_filter(self, folder_path, output_path, image_width, image_height, filter_level):
        eye_list = [242, 238, 94, 370, 362, 458, 462, 289, 455, 439, 235, 219, 94, 407, 408, 292, 306, 324, 318, 95, 77, 183, 191, 184, 76, 62, 81, 41, 38, 82, 12, 13, 312, 268, 271, 310, 272, 7, 246, 247, 30, 29, 46, 53, 52, 65, 55, 124, 113, 225, 247, 151, 33, 150, 7, 25, 110, 24, 23, 22, 26, 226, 35, 31, 156, 124, 113, 150, 140, 130, 120, 110, 180, 160, 190, 362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 463, 341, 256, 252, 253, 254, 339, 255, 359, 467, 260, 259, 257, 258, 286, 414, 413, 441, 442, 443, 444, 445, 342, 446, 261, 448, 449, 450, 451, 452, 453, 464, 465, 357, 350, 349, 348, 449, 448, 261, 265, 133, 173, 157, 158, 159, 160, 161, 162, 33, 7, 163, 144, 145, 153, 154, 155, 112, 243, 190, 56, 28, 27, 29, 30, 247, 226, 31, 228, 229, 230, 231, 232, 233, 244, 189, 221, 222, 223, 224, 226, 124, 35, 31, 228, 229, 230, 231, 232, 233, 245, 55, 53, 46, 156, 143, 111, 117, 118, 119, 120, 121, 128, 188]
        mpDraw = mp.solutions.drawing_utils
        mpFaceMesh = mp.solutions.face_mesh
        faceMesh = mpFaceMesh.FaceMesh(max_num_faces=2)
        drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=1)

        image_gliter = cv2.imread(".\\Image_Filter_Generator\\samples\\gliter.png", cv2.IMREAD_UNCHANGED)

        for filename in os.listdir(folder_path):
            if filename.endswith(".png") or filename.endswith(".jpg"):
                frame = cv2.imread(os.path.join(folder_path, filename))
                frame = imutils.resize(frame, width=image_width, height=image_height)

                frame.flags.writeable = False
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = faceMesh.process(frame)

                frame.flags.writeable = True
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                gliter_size = int(max(np.round(len(frame) * 0.005,0), 1))
                
                if gliter_size % 2 != 0 :
                    input_gliter_size = gliter_size // 2
                    gliter_size = input_gliter_size * 2
                else :
                    input_gliter_size = gliter_size // 2

                image_gliter = cv2.resize(image_gliter, dsize = (gliter_size, gliter_size))

                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        for id, lm in enumerate(face_landmarks.landmark):
                            temp_list = []
                            for i in range(1, len(face_landmarks.landmark) + 1):
                                temp_list.append(random.randint(0, filter_level))
                            ih, iw, ic = frame.shape
                            x, y = int(lm.x*iw), int(lm.y*ih)
                            if temp_list[id] == 0 and id not in eye_list:
                                self.overlay(frame, x, y, input_gliter_size, input_gliter_size, image_gliter)

                cv2.imwrite(os.path.join(output_path, filename[:-4] + "_glitter_filter.png"), frame)
                print(filename, "glitter_filter : complete")

    def nose_filter(self, folder_path, output_path, image_width, image_height):
        mpFaceMesh = mp.solutions.face_mesh
        faceMesh = mpFaceMesh.FaceMesh(max_num_faces=2)
        drawSpec = mp.solutions.drawing_utils.DrawingSpec(thickness=1, circle_radius=1)

        image_nose = cv2.imread(".\\Image_Filter_Generator\\samples\\pig_nose.png", cv2.IMREAD_UNCHANGED)

        nose_landmarks = [49, 279, 197, 2, 5]

        for filename in os.listdir(folder_path):
            if filename.endswith(".png") or filename.endswith(".jpg"):
                frame = cv2.imread(os.path.join(folder_path, filename))
                frame = imutils.resize(frame, width=image_width, height=image_height)

                frame.flags.writeable = False
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = faceMesh.process(frame)

                frame.flags.writeable = True
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                nose_size = int(max(np.round(len(frame) * 0.15,0), 1))
                
                if nose_size % 2 != 0 :
                    input_nose_size = nose_size // 2
                    nose_size = input_nose_size * 2
                else :
                    input_nose_size = nose_size // 2
                
                image_nose = cv2.resize(image_nose, dsize = (nose_size, nose_size))

                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        for id, lm in enumerate(face_landmarks.landmark):
                            if id == 1 :
                                ih, iw, ic = frame.shape
                                x, y = int(lm.x*iw), int(lm.y*ih)
                                self.overlay(frame, x, y, input_nose_size, input_nose_size, image_nose)

                cv2.imwrite(os.path.join(output_path, filename[:-4] + "_nose_filter.png"), frame)
                print(filename, "nose_filter : complete")

# 테스트
if __name__ == "__main__":
    filter_gen = Image_Filter_Generator()
    filter_gen.full_filter("./Image_Filter_Generator/input_images", "./Image_Filter_Generator/output_images", 500, 500)
    filter_gen.glitter_filter("./Image_Filter_Generator/input_images", "./Image_Filter_Generator/output_images", 500, 500, 1)
    filter_gen.nose_filter("./Image_Filter_Generator/input_images", "./Image_Filter_Generator/output_images", 500, 500)
