#train.py
import torch
import torch.nn as nn
import time
import matplotlib.pyplot as plt
import os
from dataset import Dataset
from torch.utils.data import DataLoader
from model import GeneratorUNet, Discriminator, initialize_weights
from optimizer import get_optimizer
import torchvision.transforms as transforms  # transforms 추가

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#transform 정의
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    transforms.Resize((256, 256))
])

# 손실함수
loss_func_gan = nn.BCELoss()
loss_func_pix = nn.L1Loss()

# loss_func_pix 가중치
lambda_pixel = 100

# patch 수
patch = (1, 256//2**4, 256//2**4)

# 모델 생성 및 가중치 초기화
model_gen = GeneratorUNet().to(device)
model_dis = Discriminator().to(device)
model_gen.apply(initialize_weights)
model_dis.apply(initialize_weights)

# 최적화 파라미터 설정
opt_gen, opt_dis = get_optimizer(model_gen, model_dis)

# 데이터셋 불러오기
path2img = '/content/drive/MyDrive/pix2pix/data/train'
train_ds = Dataset(path2img, transform=transform)
train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)

# 학습
model_gen.train()
model_dis.train()

batch_count = 0
num_epochs = 3
start_time = time.time()

loss_hist = {'gen': [], 'dis': []}

for epoch in range(num_epochs):
    for a, b in train_dl:
        ba_si = a.size(0)

        # real image
        real_a = a.to(device)
        real_b = b.to(device)

        # patch label
        real_label = torch.ones(ba_si, *patch, requires_grad=False).to(device)
        fake_label = torch.zeros(ba_si, *patch, requires_grad=False).to(device)

        # generator
        model_gen.zero_grad()

        fake_b = model_gen(real_a)  # 가짜 이미지 생성
        out_dis = model_dis(fake_b, real_b)  # 가짜 이미지 식별

        gen_loss = loss_func_gan(out_dis, real_label)
        pixel_loss = loss_func_pix(fake_b, real_b)

        g_loss = gen_loss + lambda_pixel * pixel_loss
        g_loss.backward()
        opt_gen.step()

        # discriminator
        model_dis.zero_grad()

        out_dis = model_dis(real_b, real_a)  # 진짜 이미지 식별
        real_loss = loss_func_gan(out_dis, real_label)

        out_dis = model_dis(fake_b.detach(), real_a)  # 가짜 이미지 식별
        fake_loss = loss_func_gan(out_dis, fake_label)

        d_loss = (real_loss + fake_loss) / 2.
        d_loss.backward()
        opt_dis.step()

        loss_hist['gen'].append(g_loss.item())
        loss_hist['dis'].append(d_loss.item())

        batch_count += 1
    print('Epoch: %.0f, G_Loss: %.6f, D_Loss: %.6f, time: %.2f min' % (epoch, g_loss.item(), d_loss.item(), (time.time()-start_time)/60))

# loss history
plt.figure(figsize=(10, 5))
plt.title('Loss Progress')
plt.plot(loss_hist['gen'], label='Gen. Loss')
plt.plot(loss_hist['dis'], label='Dis. Loss')
plt.xlabel('batch count')
plt.ylabel('Loss')
plt.legend()
plt.show()

# 가중치 저장
path2models = '/content/drive/MyDrive/pix2pix/model_weight'
os.makedirs(path2models, exist_ok=True)
path2weights_gen = os.path.join(path2models, 'weights_gen.pt')
path2weights_dis = os.path.join(path2models, 'weights_dis.pt')

torch.save(model_gen.state_dict(), path2weights_gen)
torch.save(model_dis.state_dict(), path2weights_dis)