#dataset.py
from os import listdir
from os.path import join
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import face_recognition

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    transforms.Resize((256, 256))
])

class Dataset(Dataset):
    def __init__(self, path2img, direction='b2a', transform=transform):
        super().__init__()
        self.direction = direction
        self.path2a = join(path2img, 'a')
        self.path2b = join(path2img, 'b')
        self.img_filenames = [x for x in listdir(self.path2a)]
        self.transform = transform

    def __getitem__(self, index):
        a = Image.open(join(self.path2a, self.img_filenames[index])).convert('RGB')
        b = Image.open(join(self.path2b, self.img_filenames[index])).convert('RGB')

        # b에서 얼굴 인식 및 크롭
        a_array = face_recognition.load_image_file(join(self.path2a, self.img_filenames[index]))
        face_locations = face_recognition.face_locations(a_array)

        if len(face_locations) > 0:
            # 발견된 첫 번째 얼굴에 대해서만 처리
            top, right, bottom, left = face_locations[0]
            a_cropped = a.crop((left, top, right, bottom))
            b_cropped = b.crop((left, top, right, bottom))

            if self.transform:
                a_cropped = self.transform(a_cropped)
                b_cropped = self.transform(b_cropped)

            if self.direction == 'b2a':
                return b_cropped, a_cropped
            else:
                return a_cropped, b_cropped
        else:
            # 얼굴이 감지되지 않은 경우 원본 이미지 반환
            if self.transform:
                a = self.transform(a)
                b = self.transform(b)

            if self.direction == 'b2a':
                return b, a
            else:
                return a, b

    def __len__(self):
        return len(self.img_filenames)