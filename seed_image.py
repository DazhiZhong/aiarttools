import sys
import os

os.system("git clone https://github.com/dazhizhong/MiDaS.git")
os.system("cd MiDaS && git checkout e8bafa9")
os.system("pip install timm==0.4.5")
os.system("wget --output-document MiDaS/weights/dpt_large-midas-2f21e586.pt https://github.com/intel-isl/DPT/releases/download/1_0/dpt_large-midas-2f21e586.pt --continue")
os.system("wget --output-document MiDaS/weights/midas_v21_small-70d6b9c8.pt https://github.com/AlexeyAB/MiDaS/releases/download/midas_dpt/midas_v21_small-70d6b9c8.pt --continue")
os.system("wget --output-document MiDaS/weights/dpt_hybrid-midas-501f0c75.pt https://github.com/intel-isl/DPT/releases/download/1_0/dpt_hybrid-midas-501f0c75.pt --continue")

# !pip install timm
# !wget --output-document MiDaS/weights/dpt_large-midas-2f21e586.pt https://github.com/intel-isl/DPT/releases/download/1_0/dpt_large-midas-2f21e586.pt --continue
# !wget --output-document MiDaS/weights/midas_v21_small-70d6b9c8.pt https://github.com/AlexeyAB/MiDaS/releases/download/midas_dpt/midas_v21_small-70d6b9c8.pt --continue
# !wget --output-document MiDaS/weights/dpt_hybrid-midas-501f0c75.pt https://github.com/intel-isl/DPT/releases/download/1_0/dpt_hybrid-midas-501f0c75.pt --continue

#@title new seed_image setup { form-width: "15%" }
import torch
import torch.nn.functional as F
import kornia
from kornia.geometry.depth import *
from typing import Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from kornia.filters.sobel import spatial_gradient
from kornia.utils import create_meshgrid
from kornia.geometry.camera import cam2pixel, PinholeCamera, pixel2cam, project_points, unproject_points
from kornia.geometry.conversions import normalize_pixel_coordinates
from kornia.geometry.linalg import compose_transformations, convert_points_to_homogeneous, inverse_transformation, transform_points


from torchvision.transforms import functional as TF
from torchvision.utils import save_image
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import sys
sys.path.append("MiDaS")
import midasrun
import importlib
importlib.reload(midasrun)



def my_warp_frame_depth(
    image_src: torch.Tensor,
    depth_dst: torch.Tensor,
    src_trans_dst: torch.Tensor,
    camera_matrix: torch.Tensor,
    normalize_points: bool = False,
) -> torch.Tensor:
    """Warp a tensor from a source to destination frame by the depth in the destination.

    Compute 3d points from the depth, transform them using given transformation, then project the point cloud to an
    image plane.

    Args:
        image_src: image tensor in the source frame with shape :math:`(B,D,H,W)`.
        depth_dst: depth tensor in the destination frame with shape :math:`(B,1,H,W)`.
        src_trans_dst: transformation matrix from destination to source with shape :math:`(B,4,4)`.
        camera_matrix: tensor containing the camera intrinsics with shape :math:`(B,3,3)`.
        normalize_points: whether to normalise the pointcloud. This must be set to ``True`` when the depth
           is represented as the Euclidean ray length from the camera position.

    Return:
        the warped tensor in the source frame with shape :math:`(B,3,H,W)`.
    """
    if not isinstance(image_src, torch.Tensor):
        raise TypeError(f"Input image_src type is not a torch.Tensor. Got {type(image_src)}.")

    if not len(image_src.shape) == 4:
        raise ValueError(f"Input image_src musth have a shape (B, D, H, W). Got: {image_src.shape}")

    if not isinstance(depth_dst, torch.Tensor):
        raise TypeError(f"Input depht_dst type is not a torch.Tensor. Got {type(depth_dst)}.")

    if not (len(depth_dst.shape) == 4 and depth_dst.shape[-3] == 1):
        raise ValueError(f"Input depth_dst musth have a shape (B, 1, H, W). Got: {depth_dst.shape}")

    if not isinstance(src_trans_dst, torch.Tensor):
        raise TypeError(f"Input src_trans_dst type is not a torch.Tensor. " f"Got {type(src_trans_dst)}.")

    if not (len(src_trans_dst.shape) == 3 and src_trans_dst.shape[-2:] == (4, 4)):
        raise ValueError(f"Input src_trans_dst must have a shape (B, 4, 4). " f"Got: {src_trans_dst.shape}.")

    if not isinstance(camera_matrix, torch.Tensor):
        raise TypeError(f"Input camera_matrix type is not a torch.Tensor. " f"Got {type(camera_matrix)}.")

    if not (len(camera_matrix.shape) == 3 and camera_matrix.shape[-2:] == (3, 3)):
        raise ValueError(f"Input camera_matrix must have a shape (B, 3, 3). " f"Got: {camera_matrix.shape}.")
    # unproject source points to camera frame
    points_3d_dst: torch.Tensor = depth_to_3d(depth_dst, camera_matrix, normalize_points)  # Bx3xHxW

    # transform points from source to destination
    points_3d_dst = points_3d_dst.permute(0, 2, 3, 1)  # BxHxWx3

    # apply transformation to the 3d points
    points_3d_src = transform_points(src_trans_dst[:, None], points_3d_dst)  # BxHxWx3

    # project back to pixels
    camera_matrix_tmp: torch.Tensor = camera_matrix[:, None, None]  # Bx1x1xHxW
    points_2d_src: torch.Tensor = project_points(points_3d_src, camera_matrix_tmp)  # BxHxWx2

    # normalize points between [-1 / 1]
    height, width = depth_dst.shape[-2:]
    points_2d_src_norm: torch.Tensor = normalize_pixel_coordinates(points_2d_src, height, width)  # BxHxWx2

    return F.grid_sample(image_src, points_2d_src_norm, align_corners=True,padding_mode="reflection")  # type: ignore



default_models = {
    "midas_v21_small": "MiDaS/weights/midas_v21_small-70d6b9c8.pt",
    "dpt_large": "MiDaS/weights/dpt_large-midas-2f21e586.pt",
    "dpt_hybrid": "MiDaS/weights/dpt_hybrid-midas-501f0c75.pt",
}
# modelpath = default_models[model]
# midasrun.MiDasRun(model_path=modelpath, model_type=)

class SeedImage():
    def __init__(self,model_name):
        modelpath = default_models[model_name]
        self.midas = midasrun.MiDasRun(model_path=modelpath, model_type=model_name)
    def make_seed_img(self, from_fn, mat=None, to_fn='cur_seed.png', rot = 5.0, scale = 1.0, bias = (0,0), f = None, t=None):
        if mat is not None:
            tmp_img_path = "tmp_depth.png"
            depthpath = self.midas.run(from_fn,tmp_img_path)
            im = Image.open(from_fn)
            im_tensor = TF.to_tensor(im).unsqueeze(0)
            depth = Image.open(tmp_img_path)
            depth_tensor = TF.to_tensor(depth).unsqueeze(0)
            depth_tensor = TF.resize(depth_tensor, im_tensor.size()[2:])
            depth_tensor = depth_tensor.type(torch.float)
            depth_tensor -= torch.min(depth_tensor).type(torch.float)
            depth_tensor /= torch.max(depth_tensor).type(torch.float)
            depth_tensor *= 10
            K = torch.eye(3)[None]
            # K = torch.tensor(camera,dtype=torch.float).unsqueeze(0)
            if not isinstance(mat, torch.Tensor):
                mat = torch.tensor(mat,dtype=torch.float).unsqueeze(0)
            # new_im = kornia.geometry.depth.warp_frame_depth(im_tensor, depth_tensor, mat, K)
            new_im = my_warp_frame_depth(im_tensor, depth_tensor, mat, K)
            in_img = new_im.permute(0,2,3,1)[0].cpu().numpy()
        else:
            in_img = imread(from_fn)

        if f == "None":
            f = None
        x,y = bias
        scale = float(scale)
        x = int(x)
        y = int(y)
        W,H,C = in_img.shape
        # print(W,H,C)
        
        tx = ((scale-1.0)*W)/-2.0
        ty = ((scale-1.0)*H)/-2.0
        atrans = transform.AffineTransform(scale = scale, translation=[tx,ty])

        # cropped = crop(in_img, ((enl-x, enl-y), (enl+x, enl+y), (0,0)), copy=False)
        # out_img = atrans(cropped)
        cropped = transform.warp(
                in_img, atrans, mode = 'symmetric'
        )
        # out_img = cropped
        out_img = transform.rotate(cropped, rot, mode = 'symmetric')
        if f!=None:
            out_img = f(out_img, t=t)
        imsave(to_fn,out_img)

        
        return out_img




# def make_seed_img(from_fn, to_fn='cur_seed.png', rot = 5.0, scale = 1.0, bias = (0,0), f = None, t=None):
#     if f == "None":
#         f = None
#     in_img = imread(from_fn)

#     x,y = bias
#     scale = int(scale)
#     x = int(x)
#     y = int(y)
    
#     W,H,C = in_img.shape
#     print(W,H,C)
    
#     tx = ((scale-1.0)*W)/-2.0
#     ty = ((scale-1.0)*H)/-2.0
#     atrans = transform.AffineTransform(scale = scale, translation=[tx,ty])

#     # cropped = crop(in_img, ((enl-x, enl-y), (enl+x, enl+y), (0,0)), copy=False)
#     # out_img = atrans(cropped)
#     cropped = transform.warp(
#             in_img, atrans
#     )
#     # out_img = cropped
#     out_img = transform.rotate(cropped, rot, mode = 'symmetric')
#     if f!=None:
#         out_img = f(out_img, t=t)
#     imsave(to_fn,out_img)

