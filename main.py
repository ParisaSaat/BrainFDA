import argparse
import os

import nibabel as nib
import torch

from dataset import VS
from utils import FDA_source_to_target


def create_parser():
    main_parser = argparse.ArgumentParser(description="Brain FDA App.")
    main_parser.set_defaults(action=main_parser.print_help)
    main_parser.add_argument("-s", "--source-domain", required=True, help="Source domain path")
    main_parser.add_argument("-sl", "--source-domain-labels", required=True, help="Source domain labels path")
    main_parser.add_argument("-t", "--target-domain", required=True, help="Target domain path")
    main_parser.add_argument("-o", "--output-dir", required=True, default="Path to save new data")
    return main_parser


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    source_imgs_dir = args.source_domain
    source_labels_dir = args.source_domain_labels
    target_imgs_dir = args.target_domain
    output_dir = args.output_dir
    source_data = VS(source_imgs_dir, source_labels_dir)
    target_data = VS(target_imgs_dir)

    for src, trg in zip(sorted(source_data.images_list), sorted(target_data.images_list)):
        src_img = nib.load(os.path.join(source_imgs_dir, src))
        affine = src_img.affine
        header = src_img.header
        src_img = nib.load(os.path.join(source_imgs_dir, src)).get_fdata()
        src_img = torch.from_numpy(src_img)
        trg_img = nib.load(os.path.join(target_imgs_dir, trg)).get_fdata()
        trg_img = torch.from_numpy(trg_img)
        src_in_trg = FDA_source_to_target(src_img, trg_img)
        img = nib.Nifti1Image(src_in_trg.numpy(), affine=affine, header=header)
        img.to_filename(os.path.join(output_dir, '{}'.format(src)))
