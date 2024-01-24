# import torchjpeg
# torchjpeg.codec.write_coefficients(args.output, dimensions, quantization, Y_coefficients, CbCr_coefficients)
# dimensions, quantization, Y_coefficients, CbCr_coefficients = torchjpeg.codec.quantize_at_quality(im, args.quality, args.color_samp_factor_vertical, args.color_samp_factor_horizontal)

import argparse

from PIL import Image
import numpy as np
import torchjpeg.codec
import torch
from torchvision.transforms.functional import to_pil_image, to_tensor

parser = argparse.ArgumentParser("Tests the pytorch DCT loader by reading and image, quantizing its pixels, and writing the DCT coefficients to a JPEG")
parser.add_argument("input", help="Input image, should be lossless")
parser.add_argument("output", help="Output image, must be a JPEG")
parser.add_argument("quality", type=int, help="Output quality on the 0-100 scale")
parser.add_argument("color_samp_factor_vertical", type=int, nargs="?", default=2, help="Vertical chroma subsampling factor. Defaults to 2.")
parser.add_argument("color_samp_factor_horizontal", type=int, nargs="?", default=2, help="Horizontal chroma subsampling factor. Defaults to 2.")
args = parser.parse_args()

im = to_tensor(Image.open(args.input))

if im.shape[0] > 3:
    im = im[:3]
dimensions, quantization, Y_coefficients, CbCr_coefficients = torchjpeg.codec.quantize_at_quality(im, args.quality, args.color_samp_factor_vertical, args.color_samp_factor_horizontal)
# quantization = torch.tensor(
#     [[[16, 11, 10, 16, 24, 40, 51, 61], 
#       [12, 12, 14, 19, 26, 58, 60, 55],
#       [14, 13, 16, 24, 40, 57, 69, 56],
#       [14, 17, 22, 29, 51, 87, 80, 62], 
#       [18, 22, 37, 56, 68, 109, 103, 77], 
#       [24, 35, 55, 64, 81, 104, 113, 92],
#       [49, 64, 78, 87, 103, 121, 120, 101], 
#       [72, 92, 95, 98, 112, 100, 103, 99]],
#       [[17, 18, 24, 47, 99, 99, 99, 99],
#         [18, 21, 26, 66, 99, 99, 99, 99],
#         [24, 26, 56, 99, 99, 99, 99, 99],
#         [47, 66, 99, 99, 99, 99, 99, 99],
#         [99, 99, 99, 99, 99, 99, 99, 99],
#         [99, 99, 99, 99, 99, 99, 99, 99],
#         [99, 99, 99, 99, 99, 99, 99, 99],
#         [99, 99, 99, 99, 99, 99, 99, 99]],
#       [[17, 18, 24, 47, 99, 99, 99, 99],
#         [18, 21, 26, 66, 99, 99, 99, 99],
#         [24, 26, 56, 99, 99, 99, 99, 99],
#         [47, 66, 99, 99, 99, 99, 99, 99],
#         [99, 99, 99, 99, 99, 99, 99, 99],
#         [99, 99, 99, 99, 99, 99, 99, 99],
#         [99, 99, 99, 99, 99, 99, 99, 99],
#         [99, 99, 99, 99, 99, 99, 99, 99]]
#      ]).short()
# Y_coefficients = torch.load("/NEW_EDS/JJ_Group/zhuzr/DPS_copy/y.pt").short()
# CbCr_coefficients = torch.load("/NEW_EDS/JJ_Group/zhuzr/DPS_copy/cbcr.pt").short()
# quantization = quantization.short()
torchjpeg.codec.write_coefficients(args.output, dimensions, quantization, Y_coefficients, CbCr_coefficients)
# spatial = torchjpeg.codec.reconstruct_full_image(Y_coefficients.cuda(), quantization.cuda(), CbCr_coefficients.cuda(), dimensions.cuda())
# print("a")
# to_pil_image(spatial).save(args.output)