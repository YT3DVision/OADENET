import time
import argparse
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from utils import *
from tqdm import tqdm
from model import Net
from torch.utils.tensorboard import SummaryWriter


# Settings
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--parallel', type=bool, default=False)
    parser.add_argument('--num_workers', type=int, default=16) #num_worker不能设置太大，会过度消耗cpu和内存，导致内存不足
    parser.add_argument("--angRes", type=int, default=9, help="angular resolution")
    parser.add_argument('--model_name', type=str, default='OACC_faster-Net_dis8')
    parser.add_argument('--trainset_dir', type=str, default='./dataset/training/')
    parser.add_argument('--validset_dir', type=str, default='./dataset/validation/')
    parser.add_argument('--patchsize', type=int, default=32)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3, help='initial learning rate')
    parser.add_argument('--n_epochs', type=int, default=3500, help='number of epochs to train')
    parser.add_argument('--n_steps', type=int, default=800, help='number of epochs to update learning rate')
    parser.add_argument('--gamma', type=float, default=0.5, help='learning rate decaying factor')
    parser.add_argument('--load_pretrain', type=bool, default=False)
    parser.add_argument('--model_path', type=str, default='./log/OACC-Net.pth.tar')

    return parser.parse_args()

def train(cfg):
    if cfg.parallel:
        cfg.device = 'cuda:0'
    net = Net(cfg.angRes)
    net.to(cfg.device)
    cudnn.benchmark = True
    epoch_state = 0
    writer = SummaryWriter(log_dir='runs/loss')
    if cfg.load_pretrain:
        if os.path.isfile(cfg.model_path):
            model = torch.load(cfg.model_path, map_location={'cuda:0': cfg.device})
            net_dict = net.state_dict()
            #modify
            state_dict = {k:v for k, v in model['state_dict'].items() if k in net_dict.keys()}
            print(state_dict.keys())
            net_dict.update(state_dict)

            net.load_state_dict(net_dict, strict=False)
        else:
            print("=> no model found at '{}'".format(cfg.load_model))

    if cfg.parallel:
        net = torch.nn.DataParallel(net, device_ids=[0, 1])

    criterion_Loss = torch.nn.L1Loss().to(cfg.device)
    optimizer = torch.optim.Adam([paras for paras in net.parameters() if paras.requires_grad == True], lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.n_steps, gamma=cfg.gamma)
    scheduler._step_count = epoch_state

    loss_list = []

    for idx_epoch in range(epoch_state, cfg.n_epochs):
        train_set = TrainSetLoader(cfg)
        train_loader = DataLoader(dataset=train_set, num_workers=cfg.num_workers, batch_size=cfg.batch_size, shuffle=True)
        loss_epoch = []
        for idx_iter, (data, dispGT) in tqdm(enumerate(train_loader), total=len(train_loader)):
            data, dispGT = data.to(cfg.device), dispGT.to(cfg.device)
            #data(16,1,288,288)  dispGT(16,2,32,32)
            disp, maskgt, mask = net(data, dispGT)  # modify
            # loss = criterion_Loss(disp*(1 - mask), dispGT[:, 0, :, :].unsqueeze(1)*(1-mask))  # modify
            mask = rearrange(mask,'b (u v) h w->b (u h) (v w)', u=9, v=9)
            maskgt = rearrange(maskgt,'b (u v) h w->b (u h) (v w)', u=9, v=9)
            loss = (criterion_Loss(disp, dispGT[:, 0, :, :].unsqueeze(1)) + criterion_Loss(mask, maskgt))/2 #ours 输入了86个loss，跟具体输入到网络的patch数目有关
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_epoch.append(loss.data.cpu())

        if idx_epoch % 1 == 0:
            loss_list.append(float(np.array(loss_epoch).mean()))
            print(time.ctime()[4:-5] + ' Epoch----%5d, loss---%f' % (idx_epoch + 1, float(np.array(loss_epoch).mean())))
            writer.add_scalar(tag="loss",  # 可以暂时理解为图像的名字
                              scalar_value=float(np.array(loss_epoch).mean()),  # 纵坐标的值
                              global_step=idx_epoch+1  # 当前是第几次迭代，可以理解为横坐标的值
                              )
            # if cfg.parallel:
            #     save_ckpt({
            #     'epoch': idx_epoch + 1,
            #     'state_dict': net.module.state_dict(),
            # }, save_path='./log/', filename=cfg.model_name + '.pth')
            # else:
            #     save_ckpt({ #每一轮都保存了模型权重，但是模型名称没变，因此只是在更新模型
            #         'epoch': idx_epoch + 1,
            #         'state_dict': net.state_dict(),
            #     }, save_path='./log/', filename=cfg.model_name + '.pth')
        if idx_epoch % 10 == 9: #每10轮保存一次权重，模型名称变化
            if cfg.parallel:
                save_ckpt({
                    'epoch': idx_epoch + 1,
                    'state_dict': net.module.state_dict(),
                }, save_path='./log/', filename=cfg.model_name + str(idx_epoch + 1) + '.pth')
            else:
                save_ckpt({
                'epoch': idx_epoch + 1,
                'state_dict': net.state_dict(),
            }, save_path='./log/', filename=cfg.model_name + str(idx_epoch + 1) + '.pth')

            valid(net, cfg, idx_epoch + 1)

        print("第%d个epoch的学习率：%f" % (idx_epoch, optimizer.param_groups[0]['lr']))
        scheduler.step()


def valid(net, cfg, epoch):

    torch.no_grad()
    scene_list = ['boxes', 'cotton', 'dino', 'sideboard', 'backgammon', 'dots', 'pyramids', 'stripes']
    angRes = cfg.angRes

    txtfile = open(cfg.model_name + '_MSE100.txt', 'a')
    txtfile.write('Epoch={}:\t'.format(epoch))
    txtfile.close()

    for scenes in scene_list:
        lf = np.zeros(shape=(9, 9, 512, 512, 3), dtype=int)
        for i in range(81):
            temp = imageio.imread(cfg.validset_dir + scenes + '/input_Cam0%.2d.png' % i)
            lf[i // 9, i - 9 * (i // 9), :, :, :] = temp
            del temp
        lf = np.mean((1 / 255) * lf.astype('float32'), axis=-1, keepdims=False)
        disp_gt = np.float32(
            read_pfm(cfg.validset_dir + scenes + '/gt_disp_lowres.pfm'))  # load ground truth disparity map
        angBegin = (9 - angRes) // 2

        lf_angCrop = lf[angBegin:  angBegin + angRes, angBegin: angBegin + angRes, :, :]

        patchsize = 64
        stride = patchsize // 2
        mini_batch = 4

        data = torch.from_numpy(lf_angCrop)
        sub_lfs = LFdivide(data.unsqueeze(2), patchsize, stride)
        n1, n2, u, v, c, h, w = sub_lfs.shape
        sub_lfs = rearrange(sub_lfs, 'n1 n2 u v c h w -> (n1 n2) u v c h w')
        num_inference = (n1 * n2) // mini_batch
        with torch.no_grad():
            out_disp = []
            for idx_inference in range(num_inference):
                current_lfs = sub_lfs[idx_inference * mini_batch: (idx_inference + 1) * mini_batch, :, :, :, :, :]
                input_data = rearrange(current_lfs, 'b u v c h w -> b c (u h) (v w)')
                disp, _ = net(input_data.to(cfg.device))
                out_disp.append(disp)

            if (n1 * n2) % mini_batch:
                current_lfs = sub_lfs[(idx_inference + 1) * mini_batch:, :, :, :, :, :]
                input_data = rearrange(current_lfs, 'b u v c h w -> b c (u h) (v w)')
                disp, _ = net(input_data.to(cfg.device))
                out_disp.append(disp)

        out_disps = torch.cat(out_disp, dim=0)
        out_disps = rearrange(out_disps, '(n1 n2) c h w -> n1 n2 c h w', n1=n1, n2=n2)
        disp = LFintegrate(out_disps, patchsize, patchsize // 2)
        disp = disp[0: data.shape[2], 0: data.shape[3]]
        disp = np.float32(disp.data.cpu())

        mse100 = np.mean((disp[11:-11, 11:-11] - disp_gt[11:-11, 11:-11]) ** 2) * 100
        txtfile = open(cfg.model_name + '_MSE100.txt', 'a')
        txtfile.write('mse_{}={:3f}\t'.format(scenes, mse100))
        txtfile.close()

    txtfile = open(cfg.model_name + '_MSE100.txt', 'a')
    txtfile.write('\n')
    txtfile.close()

    return


def save_ckpt(state, save_path='./log', filename='checkpoint.pth.tar'):
    torch.save(state, os.path.join(save_path,filename))



if __name__ == '__main__':
    cfg = parse_args()
    train(cfg)