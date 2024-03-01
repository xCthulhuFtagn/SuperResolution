# from ESRGAN import deGenerator, Discriminator
# import torch
# import torch.nn as nn
# from torch.optim import Adam
# from torchvision import transforms
# from dataset import SRDataset, SameTransform
# import cv2
# from torch.utils.data import DataLoader
# from skimage.metrics import peak_signal_noise_ratio,  structural_similarity
# import numpy as np

# def initialize_weights(model, scale = 0.1):
#     for m in model.modules():
#         if isinstance(m, nn.Conv2d):
#             nn.init.kaiming_normal(m.weight.data)
#             m.weight.data *= scale
#         elif isinstance(m, nn.Linear):
#             nn.init.kaiming_normal(m.weight.data)
#             m.weight.data *= scale
#         # elif isinstance(m. nn.Module):
#         #     initialize_weights(m)

# class ESRGAN_Trainer():
#     def __init__(self):
#         # устройство на котором идет обучение
#         self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

#         # количество шагов предобучения
#         self.n_presteps = 1e6
#         # количество шагов обучения
#         self.n_steps = 1e10

#         # раз во сколько шагов выводить результаты
#         self.print_interval = 100
        
#         # раз во сколько шагов чекпоинт
#         self.save_interval = 2500

#         self.batch_size = 50
#         self.workers = 8

#         # инициализация модели
#         self.generator = deGenerator(in_channels = 3).to(self.device)
#         self.discriminator = Discriminator(in_channels=3).to(self.device)
#         initialize_weights(self.generator)

#         # конфигурация оптимизатора Adam
#         self.optimizer_gen = Adam(
#             self.generator.parameters(),
#             1e-4
#         )
        
#         self.optimizer_disc = Adam(
#             self.discriminator.parameters(),
#             1e-4
#         )

#         # функция потерь MSE
#         self.pixel_criterion = nn.MSELoss().to(self.device)

#         # разрешение hr изображения в формате (h, w)
#         self.size = (480, 856)
#         self.gcrop = transforms.CenterCrop([480, 856])

#         # # аугментации для обучения и валидации
#         train_transform = SameTransform('train')

#         # путь где хранятся папки lr и hr с изображениями
#         train_prefix = './train_frames'

#         # train датасет
#         trainset = SRDataset(
#             f'{train_prefix}/lr',
#             f'{train_prefix}/hr',
#             train_transform
#         )

#         # даталоадер для обучения батчами
#         self.trainloader = DataLoader(
#             trainset,
#             batch_size=self.batch_size,
#             shuffle=True,
#             drop_last=True,
#             num_workers=self.workers,
#             pin_memory=True
#         )

#         # аугментации для инференса
#         self.resize = transforms.Resize(self.size, antialias=None)
#         self.np2tensor = transforms.ToTensor()

#     def pretrain_step(self, lr, hr):
#         pass 
    
#     def train_step(self, lr, hr):
#         g_hr = self.fsrcnn(lr)
#         g_hr = self.gcrop.forward(g_hr)
#         loss = self.pixel_criterion(g_hr, hr)
#         self.optimizer.zero_grad()
#         loss.backward()
#         self.optimizer.step()
#         return loss.item()
    
#     def pretrain(self):
#         pass

#     def train(self):
#         self.fsrcnn.train()
#         step = 0
        
#         while True:
#             if step >= self.n_steps:
#                 break

#             for batch in self.trainloader:
#                 lr, hr = batch
#                 lr = lr.to(self.device, non_blocking=True)
#                 hr = hr.to(self.device, non_blocking=True)

#                 mse = self.train_step(lr, hr)
#                 step += 1

#                 if step % self.print_interval == 0:
#                     print(f'STEP={step} MSE={mse:.5f}')
                    
#                 if step % self.save_interval == 0:
#                     torch.save({
#                         'epoch': step,
#                         'model_state_dict': self.fsrcnn.state_dict(),
#                         'optimizer_state_dict': self.optimizer.state_dict(),
#                         'loss': mse,
#                         }, "./checkpoint")
        
#         torch.save({
#                 'epoch': step,
#                 'model_state_dict': self.fsrcnn.state_dict(),
#                 'optimizer_state_dict': self.optimizer.state_dict(),
#                 'loss': mse,
#                 }, "./checkpoint")

#     def frame2tensor(self, img):
#         rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         hr = self.np2tensor(rgb)#self.resize(self.np2tensor(rgb))
#         return hr

#     def tensor2frame(self, img):
#         nparr = (img.detach().cpu().numpy() * 255).astype(np.uint8)
#         nparr = np.transpose(nparr, (1, 2, 0))
#         bgr = cv2.cvtColor(nparr, cv2.COLOR_RGB2BGR)
#         return bgr

#     def super_resolution(self, input_video, output_video, test_video = None):
#         crop = transforms.CenterCrop(self.size)
#         self.fsrcnn.eval()

#         cap = cv2.VideoCapture(input_video)
#         fps = cap.get(cv2.CAP_PROP_FPS)

#         fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#         writer = cv2.VideoWriter(
#             output_video,
#             fourcc,
#             fps,
#             (self.size[1], self.size[0])
#         )
        
#         resize_lr = transforms.Resize((120, 214), antialias = True)
        
#         if test_video: 
#             test_cap = cv2.VideoCapture(test_video)
#             psnr = []
#             ssim = []

#         while True:
#             success, frame = cap.read()
#             if test_video: 
#                 t_success, test_frame = test_cap.read()
#                 success = success and t_success
            
#             if not success: break
            
#             if test_video:
#                 psnr.append(peak_signal_noise_ratio(test_frame, frame))
#                 ssim.append(structural_similarity(test_frame, frame))
            
#             tensor = self.frame2tensor(frame).to(self.device).unsqueeze_(0)#lr_crop.forward(self.frame2tensor(frame).to(self.device)).unsqueeze_(0)
#             tensor = resize_lr(tensor)
#             with torch.no_grad(): 
#                 output_tensor = self.fsrcnn(tensor)
#             output_frame = self.tensor2frame(crop.forward(output_tensor[0]))

#             writer.write(output_frame)

#         cap.release()
#         writer.release()
        
#         if test_video: 
#             test_cap.release()
#             print(f"Average PSNR for output video and ground truth is {np.array(psnr).mean()}")
#             print(f"Average SSIM for output video and ground truth is {np.array(ssim).mean()}")
    