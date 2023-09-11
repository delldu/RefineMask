import os 
import os.path as osp
import torch 
import torch.nn.functional as F 
import numpy as np
import cv2
import mmcv
from mmcv.ops.nms import nms
from mmcv.ops.roi_align import roi_align
from tqdm import tqdm
from functools import partial
from torch.utils.data import Dataset, DataLoader

from mmcv.runner import load_checkpoint
from mmseg.models import build_segmentor
from mmcv.parallel import MMDataParallel, DataContainer, collate
import pdb

def __output_kv__(prefix, k, v):
    if isinstance(v, torch.Tensor):
        print(f"{prefix}tensor [{k}] size:", list(v.size()), ", min:", v.min(), ", max:", v.max())
    elif isinstance(v, np.ndarray):
        print(f"{prefix}array [{k}] shape:", v.shape, ", min:", v.min(), ", max:", v.max())
    elif isinstance(v, list):
        print(f"{prefix}list [{k}] len:", len(v), ",", v)
    elif isinstance(v, tuple):
        print(f"{prefix}tuple [{k}] len:", len(v), ",", v)
    elif isinstance(v, str):
        print(f"{prefix}[{k}] value:", "'" + v + "'")
    else:
        print(f"{prefix}[{k}] value:", v)

def debug_var(v_name, v_value):
    if isinstance(v_value, dict):
        prefix = "    "
        print(f"{v_name} is dict:")
        for k, v in v_value.items():
            __output_kv__(prefix, k, v)
    else:
        prefix = ""
        __output_kv__(prefix, v_name, v_value)

def find_float_boundary(maskdt, kernel_size=3):
    """Find the boundaries.
    """

    N, H, W = maskdt.shape
    maskdt = maskdt.view(N, 1, H, W)
    kernel = maskdt.new_ones((1, 1, kernel_size, kernel_size))
    # tensor [kernel] size: [1, 1, 3, 3] , min: tensor(1., device='cuda:0') , 
    #   max: tensor(1., device='cuda:0')

    boundary_mask = F.conv2d(maskdt, kernel, stride=1, padding=kernel_size//2)
    # tensor [boundary_mask] size: [1, 1, 1024, 2048] , min: tensor(0., device='cuda:0') , max: tensor(9., device='cuda:0')

    bml = torch.abs(boundary_mask - kernel_size*kernel_size)
    bms = torch.abs(boundary_mask)
    fbmask = torch.min(bml, bms) / (kernel_size*kernel_size/2)

    # tensor [fbmask] size: [1, 1, 1024, 2048] , min: tensor(0., device='cuda:0') , max: tensor(0.8889, device='cuda:0')

    return fbmask.view(N, H, W)


def _force_move_back(sdets, H, W, patch_size):
    # force the out of range patches to move back
    # box.x1
    s = sdets[:, 0] < 0 
    sdets[s, 0] = 0
    sdets[s, 2] = patch_size # x2

    # box.y1
    s = sdets[:, 1] < 0 
    sdets[s, 1] = 0
    sdets[s, 3] = patch_size # y2

    # box.x2
    s = sdets[:, 2] >= W 
    sdets[s, 0] = W - 1 - patch_size # x1
    sdets[s, 2] = W - 1 # x2

    # box.y2
    s = sdets[:, 3] >= H 
    sdets[s, 1] = H - 1 - patch_size # y1
    sdets[s, 3] = H - 1 # y2

    return sdets


def get_dets(fbmask, patch_size=64, iou_thresh=0.25):
    """boundaries of coarse mask -> patch bboxs
    """

    # fbmask.size() -- [1024, 2048]
    # patch_size = 64
    # iou_thresh = 0.25

    ys, xs = torch.nonzero(fbmask, as_tuple=True)
    scores = fbmask[ys,xs]

    ys = ys.float()
    xs = xs.float()
    dets = torch.stack([xs - patch_size//2, ys - patch_size//2, 
            xs + patch_size//2, ys + patch_size//2, scores]).T
    # dets.size() -- [3384, 5]

    _, inds = nms(dets[:,:4].contiguous(), dets[:,4].contiguous(), iou_thresh)
    sdets = dets[inds] #  sdets.size() -- [33, 5]

    H, W = fbmask.shape

    return _force_move_back(sdets, H, W, patch_size)


class PatchDataset(Dataset):
    def __init__(self, img_paths, mask_paths, device, out_size=(512, 512)):
        self.device = device
        self.out_size = out_size
        self.img_mean = np.array([123.675, 116.28, 103.53]).reshape(1,1,3)
        self.img_std = np.array([58.395, 57.12, 57.375]).reshape(1,1,3)
        self._img2dts = list(zip(img_paths, mask_paths))      # list of (img_path, list of coarse_mask_path)
        # self.img_mean/255
        # array([[[0.485, 0.456, 0.406]]])
        # (Pdb) self.img_std/255
        # array([[[0.229, 0.224, 0.225]]])

    def __len__(self):
        return len(self._img2dts)

    def __getitem__(self, i):
        img_path, mask_paths = self._img2dts[i]
        img = cv2.imread(img_path)[:,:,::-1]     # BGR -> RGB
        img = np.ascontiguousarray(img)
        img = (img - self.img_mean) / self.img_std

        valid_dt_paths, valid_maskdt = [], []   # skip empty mask
        for dt_path in mask_paths:
            m = cv2.imread(dt_path, 0) > 0
            if m.any():
                valid_dt_paths.append(dt_path)
                valid_maskdt.append(m)
        if len(valid_dt_paths):
            valid_maskdt = np.stack(valid_maskdt)
        else:
            valid_maskdt = np.zeros((0, 1024, 2048), dtype=np.float32)

        # (Pdb) img.shape, img.min(), img.max()
        # ((1024, 2048, 3), -2.0665296686360133, 2.64)
        # (Pdb) valid_maskdt.shape, valid_maskdt.min(), valid_maskdt.max()
        # ((1, 1024, 2048), False, True)

        return DataContainer([
                valid_dt_paths,
                torch.tensor(img, dtype=torch.float), \
                torch.tensor(valid_maskdt, dtype=torch.float)
            ])


def _build_dataloader(img_paths, mask_paths, device):
    dataset = PatchDataset(img_paths, mask_paths, device)
    return DataLoader(dataset, pin_memory=True, collate_fn=collate)


def _build_model(cfg, ckpt, patch_size=64):
    # ckpt = '../ckpts/hrnet18s_128-24055c80.pth'
    # patch_size = 64

    # build the model and load checkpoint
    cfg = mmcv.Config.fromfile(cfg)
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True
    img_meta = [dict(
        ori_shape=(patch_size, patch_size),
        flip=False)]

    model = build_segmentor(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)

    checkpoint = load_checkpoint(model, ckpt, map_location='cpu')
    model.CLASSES = checkpoint['meta']['CLASSES'] # ('bg', 'fg'
    model.PALETTE = checkpoint['meta']['PALETTE'] # [[128, 64, 128], [244, 35, 232]]

    # model -- EncoderDecoderRefine(...)

    model = MMDataParallel(model, device_ids=[0])
    model.eval()


    return partial(model.module.inference, img_meta=img_meta, rescale=False)


def _to_rois(xyxys):
    inds = xyxys.new_zeros((xyxys.size(0), 1))
    return torch.cat([inds, xyxys], dim=1).float().contiguous()


def split(img, maskdts, boundary_width=3, iou_thresh=0.25, patch_size=64, out_size=128):
    # tensor [img] size: [1024, 2048, 3] , min: tensor(-2.0665, device='cuda:0') , max: tensor(2.6400, device='cuda:0')
    # tensor [maskdts] size: [1, 1024, 2048] , min: tensor(0., device='cuda:0') , max: tensor(1., device='cuda:0')
    # boundary_width = 3
    # iou_thresh = 0.25
    # patch_size = 64
    # out_size = 128
    fbmasks = find_float_boundary(maskdts, boundary_width)

    detss = []
    for i in range(fbmasks.size(0)):
        dets = get_dets(fbmasks[i], patch_size, iou_thresh=iou_thresh)[:,:4]
        detss.append(dets)

    all_dets = torch.cat(detss, dim=0)
    img = img.permute(2,0,1).unsqueeze(0).float().contiguous()   # 1,3,H,W
    img_patches = roi_align(img, _to_rois(all_dets), patch_size)
    # tensor [img_patches] size: [33, 3, 64, 64] , min: tensor(-2.0665, device='cuda:0') , max: tensor(2.4286, device='cuda:0')

    _detss = [torch.cat([i*_.new_ones((_.size(0), 1)), _], dim=1) 
        for i,_ in enumerate(detss)]
    _detss = torch.cat(_detss)
    dt_patches = roi_align(maskdts[:,None,:,:], _detss, patch_size)
    # tensor [dt_patches] size: [33, 1, 64, 64] , min: tensor(0., device='cuda:0') , max: tensor(1., device='cuda:0')

    img_patches = F.interpolate(img_patches, (out_size, out_size), mode='bilinear')
    # tensor [img_patches] size: [33, 3, 128, 128] , min: tensor(-1.9959, device='cuda:0') , max: tensor(2.3585, device='cuda:0')

    dt_patches = F.interpolate(dt_patches, (out_size, out_size), mode='nearest')
    # tensor [dt_patches] size: [33, 1, 128, 128] , min: tensor(0., device='cuda:0') , max: tensor(1., device='cuda:0')

    return detss, torch.cat([img_patches, 2*dt_patches - 1], dim=1)


def merge(maskdts, detss, maskss, patch_size=64):
    # detss: list of dets (Ni,4), x1,y1,x2,y2 format, len K
    # maskdts: (K, H, W)
    # maskss (sum_i Ni, 128, 128)
    # tensor [maskdts] size: [1, 1024, 2048] , min: tensor(0., device='cuda:0') , max: tensor(1., device='cuda:0')

    # list [detss] len: 1 , [tensor([[ 855.,  363.,  919.,  427.],
    #         [1083.,  363., 1147.,  427.],
    #         ...
    #         [1188.,  543., 1252.,  607.],
    #         [ 931.,  633.,  995.,  697.]], device='cuda:0')]
    # detss[0].size() -- [33, 4]

    # tensor [maskss] size: [33, 128, 128] , min: tensor(8.3961e-05, device='cuda:0', grad_fn=<MinBackward1>) , max: tensor(0.9999, device='cuda:0', grad_fn=<MaxBackward1>)

    out = []

    K, H, W = maskdts.shape
    maskdts = maskdts.bool()
    maskss = F.interpolate(maskss.unsqueeze(0), (patch_size, patch_size), 
            mode='bilinear').squeeze(0)
    dt_refined = torch.zeros_like(maskdts[0], dtype=torch.float32)  # H, W
    dt_count = torch.zeros_like(maskdts[0], dtype=torch.float32)    # H, W

    p = 0
    for k in range(K):
        dets = detss[k]
        dets = dets[:, :4].int()    # Ni, 4
        maskdt = maskdts[k]         # H, W
        q = p + dets.size(0)
        masks = maskss[p:q]         # Ni, 64, 64
        p = q

        dt_refined.zero_()
        dt_count.zero_()
        for i in range(dets.size(0)):
            x1, y1, x2, y2 = dets[i]
            dt_refined[y1:y2, x1:x2] += masks[i]
            dt_count[y1:y2, x1:x2] += 1

        s = dt_count > 0
        dt_refined[s] /= dt_count[s]
        maskdt[s] = dt_refined[s] > 0.5

        out.append(maskdt)

    # (Pdb) len(out), out[0].size()
    # (1, torch.Size([1024, 2048]))
    return out


# def inference(cfg, ckpt, img_paths, mask_paths, out_dir, max_ins=32):
#     pdb.set_trace()

#     if not osp.exists(out_dir):
#         os.makedirs(out_dir)

#     model = _build_model(cfg, ckpt)
#     dataloader = _build_dataloader(img_paths, mask_paths, 
#             device=torch.device('cuda:0'))

#     def _inference_one(img, sub_maskdts, sub_dt_paths): # to save GPU memory
#         dets, patches = split(img, sub_maskdts)
#         masks = model(patches)[:,1,:,:]         # N, 128, 128
#         refineds = merge(sub_maskdts, dets, masks)
#         for i, dt_path in enumerate(sub_dt_paths):
#             cv2.imwrite(
#                 osp.join(out_dir, osp.basename(dt_path)),
#                 refineds[i].cpu().numpy().astype(np.uint8) * 255
#             )
#         return refineds[i].cpu().numpy().astype(np.uint8) * 255

#     pdb.set_trace()

#     # inference on each image
#     with tqdm(dataloader) as tloader:
#         for dc in tloader:
#             mask_paths, img, maskdts = dc.data[0][0]
#             if len(mask_paths):
#                 img = img.cuda()             # 3, 1024, 2048
#                 maskdts = maskdts.cuda()     # N, 1024, 2048

#                 p = 0
#                 for sub_maskdts in maskdts.split(max_ins):
#                     q = p + sub_maskdts.size(0)
#                     sub_dt_paths = mask_paths[p:q]
#                     p = q
#                     _inference_one(img, sub_maskdts, sub_dt_paths)

#     pdb.set_trace()

# if __name__=='__main__':
#     cfg = "../configs/bpr/hrnet18s_128.py"
#     ckpt = "../ckpts/hrnet18s_128-24055c80.pth"
#     img_paths = ['lindau_000000_000019_leftImg8bit.png', ]            # image
#     mask_paths = [['lindau_000000_000019_leftImg8bit_15_car.png'], ]    # coarse mask images: 0 for background, >0 for instance
#     inference(cfg, ckpt, img_paths, mask_paths, "./refined", max_ins=32)
