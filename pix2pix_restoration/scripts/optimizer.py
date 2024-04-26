import torch.optim as optim

# 최적화 파라미터
#lr = 2e-4
lr = 0.001
beta1 = 0.5
beta2 = 0.999

def get_optimizer(model_gen, model_dis):
    opt_dis = optim.Adam(model_dis.parameters(), lr=lr, betas=(beta1, beta2))
    opt_gen = optim.Adam(model_gen.parameters(), lr=lr, betas=(beta1, beta2))
    return opt_gen, opt_dis