from ESRGAN import Generator, Discriminator
import torch
import torch.nn as nn
from torch.optim import Adam
from torchvision import transforms
from dataset import SRDataset, SameTransform
import cv2
from torch.utils.data import DataLoader
from skimage.metrics import peak_signal_noise_ratio,  structural_similarity
import numpy as np

def initialize_weights(model, scale = 0.1):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal(m.weight.data)
            m.weight.data *= scale
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal(m.weight.data)
            m.weight.data *= scale
        # elif isinstance(m. nn.Module):
        #     initialize_weights(m)

class ESRGAN_Trainer():
    def __init__(self, lr = 1e-4, betas = (.9, .99)):
        # устройство на котором идет обучение
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        # количество шагов предобучения
        self.n_presteps = 1e6
        # количество шагов обучения
        self.n_steps = 1e10
        # количество совершенных шагов
        self.cur_steps = 0

        # раз во сколько шагов выводить результаты
        self.print_interval = 100
        
        # раз во сколько шагов чекпоинт
        self.save_interval = 2500

        self.batch_size = 50
        self.workers = 8

        # инициализация модели
        self.generator = Generator(in_channels = 3).to(self.device)
        self.discriminator = Discriminator(in_channels=3).to(self.device)
        initialize_weights(self.generator)
        initialize_weights(self.discriminator)

        # конфигурация оптимизатора Adam
        self.optimizer_G = Adam(
            self.generator.parameters(),
            lr = lr,
            betas = betas
        )
        self.optimizer_D = Adam(
            self.discriminator.parameters(),
            lr = 1e-4,
            betas = (.9, .99)
        )

        # функции потерь
        self.pixel_criterion = nn.MSELoss().to(self.device)
        self.GAN_criterion = nn.BCEWithLogitsLoss().to(self.device)
        self.content_criterion = nn.L1Loss().to(self.device)

        # разрешение hr изображения в формате (h, w)
        self.frame_size = (480, 856)
        # self.gcrop = transforms.CenterCrop([480, 856])

        # # аугментации для обучения и валидации
        train_transform = SameTransform('train')

        # путь где хранятся папки lr и hr с изображениями
        train_prefix = '../train_frames'

        # train датасет
        trainset = SRDataset(
            f'{train_prefix}/lr',
            f'{train_prefix}/hr',
            train_transform
        )

        # даталоадер для обучения батчами
        self.trainloader = DataLoader(
            trainset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.workers,
            pin_memory=True
        )

        # аугментации для инференса
        self.resize = transforms.Resize(self.frame_size, antialias=None)
        self.np2tensor = transforms.ToTensor()
        
    def read_state(self, state):
        self.cur_steps = state['epoch']
        self.generator.load_state_dict(state['generator_state_dict'])
        self.optimizer_G.load_state_dict(state['optimizer_G_state_dict'])
        self.discriminator.load_state_dict(state['generator_state_dict'])
        self.optimizer_D.load_state_dict(state['optimizer_D_state_dict'])

    def save_state(self, loss_G, loss_D):
        torch.save({
            'epoch': self.cur_steps,
            'generator_state_dict': self.generator.state_dict(),
            'optimizer_G_state_dict': self.optimizer_G.state_dict(),
            'loss_G': loss_G,
            'discriminator_state_dict': self.discriminator.state_dict(),
            'optimizer_D_state_dict': self.optimizer_D.state_dict(),
            'loss_D': loss_D,
            }, "./checkpoint")

    def train(self):
        self.generator.train()

        while True:
            if self.cur_steps >= self.n_steps + self.n_presteps:
                break

            for batch in self.trainloader:
                lr, hr = batch
                lr = lr.to(self.device, non_blocking=True)
                hr = hr.to(self.device, non_blocking=True)
                self.cur_steps += 1
                
                valid = torch.Tensor(np.ones((self.batch_size, *self.frame_size)))
                fake = torch.Tensor(np.zeros((self.batch_size, *self.frame_size)))
                
                
                g_hr = self.generator(lr)
                # g_hr = self.gcrop.forward(g_hr)
                
                # training generator
                
                self.optimizer_G.zero_grad()
                
                pixel_loss = self.pixel_criterion(g_hr, hr)
                
                # pretraining
                if self.cur_steps <= self.n_presteps:   
                    pixel_loss.backward()
                    self.optimizer_G.step()
                                        
                    if self.cur_steps % self.print_interval == 0:
                        print(f'STEP={self.cur_steps} Loss_G={pixel_loss:.5f}')
                    if self.cur_steps % self.save_interval == 0:
                        self.save_state(loss_G=pixel_loss, loss_D=None)
                    continue
                
                pred_real = self.discriminator(hr).detach()
                pred_fake = self.discriminator(g_hr)
                
                GAN_loss = self.GAN_criterion(pred_fake - pred_real.mean(0, keepdim=True), valid)

                g_features = self.feature_extractor(g_hr)
                real_features = self.feature_extractor(hr).detach()
                content_loss = self.content_criterion(g_features, real_features)
                
                total_loss_G = content_loss + self.GAN_lambda * GAN_loss + self.pixel_lambda * pixel_loss
                
                total_loss_G.backward()
                self.optimizer_G.step()
                
                # training discriminator
                
                self.discriminator.train()
                
                self.optimizer_D.zero_grad()
                
                loss_real = self.GAN_criterion(pred_real - pred_fake.mean(0, keepdim=True), valid)
                loss_fake = self.GAN_criterion(pred_fake - pred_real.mean(0, keepdim=True), fake)
                
                total_loss_D = (loss_real + loss_fake) / 2

                total_loss_D.backward()
                self.optimizer_D.step()

                if self.cur_steps % self.print_interval == 0:
                    print(f'STEP={self.cur_steps}, Loss_G={total_loss_G:.5f}, Loss_D={total_loss_D:.5f}')
                if self.cur_steps % self.save_interval == 0:
                    self.save_state(loss_G=total_loss_G, loss_D=total_loss_D)
        
        print(f'STEP={self.cur_steps}, Loss_G={total_loss_G:.5f}, Loss_D={total_loss_D:.5f}')              
        self.save_state(loss_G=total_loss_G, loss_D=total_loss_D)

    def frame2tensor(self, img):
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        hr = self.np2tensor(rgb)#self.resize(self.np2tensor(rgb))
        return hr

    def tensor2frame(self, img):
        nparr = (img.detach().cpu().numpy() * 255).astype(np.uint8)
        nparr = np.transpose(nparr, (1, 2, 0))
        bgr = cv2.cvtColor(nparr, cv2.COLOR_RGB2BGR)
        return bgr

    def super_resolution(self, input_video, output_video, test_video = None):
        crop = transforms.CenterCrop(self.frame_size)
        self.fsrcnn.eval()

        cap = cv2.VideoCapture(input_video)
        fps = cap.get(cv2.CAP_PROP_FPS)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(
            output_video,
            fourcc,
            fps,
            (self.frame_size[1], self.frame_size[0])
        )
        
        resize_lr = transforms.Resize((120, 214), antialias = True)
        
        if test_video: 
            test_cap = cv2.VideoCapture(test_video)
            psnr = []
            ssim = []
        
        frame_n = 0

        while True:
            success, frame = cap.read()
            if test_video: 
                t_success, test_frame = test_cap.read()
                success = success and t_success
            
            if not success: break
            
            frame_n += 1
            
            tensor = self.frame2tensor(frame).to(self.device).unsqueeze_(0)#lr_crop.forward(self.frame2tensor(frame).to(self.device)).unsqueeze_(0)
            tensor = resize_lr(tensor)
            with torch.no_grad(): 
                output_tensor = self.generator(tensor)
            output_frame = self.tensor2frame(crop.forward(output_tensor[0]))

            if test_video:
                # print(test_frame.shape, output_frame.shape)
                psnrs = [peak_signal_noise_ratio(test_frame[:,:,i], output_frame[:,:,i]) for i in range(3)]
                psnr.append(np.mean(psnrs))
                ssim.append(structural_similarity(test_frame, output_frame, channel_axis = 2))
                
            writer.write(output_frame)

        cap.release()
        writer.release()
        
        if test_video: 
            test_cap.release()
            print(f"Average PSNR for output video and ground truth is {np.array(psnr).mean()}")
            print(f"Average SSIM for output video and ground truth is {np.array(ssim).mean()}")