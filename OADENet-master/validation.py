import argparse
from utils import *
from model_for_consistency import Net
import seaborn as sns
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument("--angRes", type=int, default=9, help="angular resolution")
    parser.add_argument('--model_name', type=str, default='OACC-Net')
    parser.add_argument('--testset_dir', type=str, default='./demo_input/')
    parser.add_argument('--crop', type=bool, default=False)
    parser.add_argument('--patchsize', type=int, default=128)
    parser.add_argument('--minibatch_test', type=int, default=4)
    parser.add_argument('--model_path', type=str, default='./log/OACC_faster-Net2440.pth')
    parser.add_argument('--save_path', type=str, default='./Results/')
    parser.add_argument('--mask_save_path', type=str, default='./mask/')
    parser.add_argument('--maskgt_save_path', type=str, default='./mask_gt/')

    return parser.parse_args()

'''
Note: 1) We crop LFs into overlapping patches to save the CUDA memory during inference. 
      2) When cropping is performed, the inference time will be longer than the one reported in our paper.
'''

def test(cfg):
    scene_list = os.listdir(cfg.testset_dir)
    angRes = cfg.angRes

    net = Net(cfg.angRes)
    net.to(cfg.device)
    model = torch.load(cfg.model_path, map_location={'cuda:0': cfg.device})
    net.load_state_dict(model['state_dict'])

    for scenes in scene_list:
        print('Working on scene: ' + scenes + '...')
        temp = imageio.imread(cfg.testset_dir + scenes + '/input_Cam000.png')  #和valid的类似，只是以一个先打开图像，找尺寸，一个直接是（9，9，512，512，3）
        lf = np.zeros(shape=(9, 9, temp.shape[0], temp.shape[1], 3), dtype=int)
        for i in range(81):
            temp = imageio.imread(cfg.testset_dir + scenes + '/input_Cam0%.2d.png' % i)
            lf[i // 9, i - 9 * (i // 9), :, :, :] = temp
        lf_gray = np.mean((1 / 255) * lf.astype('float32'), axis=-1, keepdims=False)
        disp_gt = np.float32(
            read_pfm(cfg.testset_dir + scenes + '/gt_disp_lowres.pfm'))  # load ground truth disparity map

        angBegin = (9 - angRes) // 2
        lf_angCrop = lf_gray[angBegin:  angBegin + angRes, angBegin: angBegin + angRes, :, :]

        # if cfg.crop == False:
        data = rearrange(lf_angCrop, 'u v h w -> (u h) (v w)')
        data = ToTensor()(data.copy())
        disp_gt = ToTensor()(disp_gt.copy())
        with torch.no_grad():#input data(1,4608,4608)
            #mask_ref是学习后的mask,mask是无学习的mask
            disp, mask_gt, mask_ref, mask = net(data.unsqueeze(0).to(cfg.device), disp_gt.unsqueeze(0).to(cfg.device))

        for i in [0, 8, 72, 80]:

            disp_pre = np.float32(disp[0, 0, :, :].data.cpu())
            # 可以取不同区域的mask值进行比较，查看我们学习的mask和maskgt之间的的差异

            # 可以查看不同的视点的mask，比如用mask[0,8,:,:],查看u=1,v=9的子视点的mask,比如用mask[0,8,:,:],查看u=1,v=9的子视点的mask
            #mask(1,1) 在[0,0],mask(1,9) 在[0,8],mask(9,1)在mask[0,72],mask(9,9)在mask[0,80],
            mask_pre = np.float32(mask[0, i, :, :].data.cpu())
            mask_ref_pre = np.float32(mask_ref[0, i, :, :].data.cpu())

            # 可以查看不同的视点的mask，比如用mask[0,8,:,:],查看u=1,v=9的子视点的mask,比如用mask[0,8,:,:],查看u=1,v=9的子视点的mask
            mask_gt_view = mask_gt[0, i, :, :].data.cpu()
            print('Finished! \n')
            # write_pfm(disp, cfg.save_path + '%s.pfm' % (scenes))
            #显示深度图
            plt.imsave(cfg.save_path + '%s_view_disp.png' % (scenes), disp_pre)
            #需要把小于0和大于1的内容转换一下
            #显示学习出的遮挡热力图
            # write_pfm(mask, cfg.save_path + '%s.pfm' % (scenes))
            mask_ref_pre[mask_ref_pre > 1] = 1
            mask_ref_pre[mask_ref_pre < 0] = 0
            # mask_ref = torch.clamp(mask_ref)
            s1 = sns.heatmap(mask_ref_pre,  xticklabels=[], yticklabels=[], cbar=True, cmap='viridis')
            s1 = s1.get_figure()
            s1.savefig(cfg.mask_save_path + '%s_mask_ref_view%d_HeatMap.png'%(scenes, i), dpi=400, bbox_inches='tight')
            # print(i)
            # plt.imsave(cfg.save_path + '%s_mask.png' % (scenes), mask)


            #显示无学习的遮挡热力图
            # write_pfm(mask, cfg.save_path + '%s.pfm' % (scenes))
            mask_pre[mask_pre > 1] = 1
            mask_pre[mask_pre < 0] = 0
            s1 = sns.heatmap(mask_pre,  xticklabels=[], yticklabels=[], cbar=False, cmap='viridis')
            s1 = s1.get_figure()
            s1.savefig(cfg.mask_save_path + '%s_mask_view%d_HeatMap.png' %(scenes, i), dpi=400, bbox_inches='tight')
            # plt.imsave(cfg.save_path + '%s_mask.png' % (scenes), mask)


            #程序有点bug,对于中心视图，默认没有遮挡，最大最小值均为1，不能/ (max_val - min_val)
            # min_val = torch.min(mask_gt)
            # max_val = torch.max(mask_gt)
            # 将张量归一化到0到1之间
            # mask_gt = (mask_gt - min_val) / (max_val - min_val)
            mask_gt_view = np.float32(mask_gt_view)
            s2 = sns.heatmap(mask_gt_view, xticklabels=[], yticklabels=[], cbar=False, cmap='viridis')
            s2 = s2.get_figure()
            s2.savefig(cfg.maskgt_save_path + '%s_maskgt_view%d_HeatMap.png' %(scenes, i), dpi=400, bbox_inches='tight')
            # plt.imsave(cfg.save_path + '%s_maskgt.png' % (scenes), mask_gt)

    return


if __name__ == '__main__':
    cfg = parse_args()
    test(cfg)
