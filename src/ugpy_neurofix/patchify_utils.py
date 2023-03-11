import os
import numpy as np
import numpy.typing as npt
import tifffile as tif
from pathlib import Path
from typing import Union, List, Tuple
from patchify import patchify, unpatchify


def save_patches(patches: npt.NDArray,
                 save_folder: Union[str, Path],
                 file_tag: Union[str, List[str]] = 'ptch'):
    """
    Save patches to the folder. The patches are saved in the format of
    ``{tag}_{ii}_{jj}_{kk}.tif``.

    Args:
        patches : Patches to save. Shape: (nz,ny,nx, *patch_size).
        save_folder : Folder to save the patches. If the folder does not exist,
            it will be created.
        file_tag : Tag for the filenames.
            If a list is given, the tag will be joined by '_'.
    """
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # add patch shape to the tag
    if isinstance(file_tag, str):
        file_tag = [file_tag]
    file_tag.append(f'patch_shape_{patches.shape[0:3]}')
    tag = '_'.join(file_tag)

    # save the patches
    for ii in range(patches.shape[0]):
        for jj in range(patches.shape[1]):
            for kk in range(patches.shape[2]):
                tif.imwrite(Path(save_folder,
                                 '_'.join(file_tag) + f'_{ii}_{jj}_{kk}.tif'),
                            patches[ii, jj, kk])


def patchify_and_save(img: npt.NDArray,
                      patch_size: int,
                      save_folder: Union[str, Path],
                      file_tag: Union[str, List[str]] = 'ptch',
                      step: int = None,
                      reversible: bool = True
                      ):
    """
    Patchify the image and save to folder.
    Automatically adds image shape to the file_tag.

    Args:
        img : Image to patchify.
        patch_size : Size of the patches in 3D.
        step : Step size of the patches.
        reversible : Whether the patchify should be reversible.
        file_tag : Tag for the filenames.
            If a list is given, the tag will be joined by '_'.
        save_folder : Folder to save the patches.
            If the folder does not exist, it will be created.
    """
    if step is None:
        step = patch_size

    # add image_shape to the file_tag
    if isinstance(file_tag, str):
        file_tag = [file_tag]
    file_tag.append(f'img_shape_{img.shape}')

    if reversible:
        assert np.all(
            np.mod((np.array(img.shape) - patch_size), step) == 0), \
            "The image size is not divisible by the patch size."
    patches = patchify(img, (patch_size, patch_size, patch_size), step=step)
    save_patches(patches, save_folder, file_tag)


def load_patches(patches_folder: Union[str, Path],
                 ) -> npt.NDArray:
    """
    Load patches from the folder. The patches should be saved in the format of
    ``..._patch_shape_(...)_..._{ii}_{jj}_{kk}.tif``.
    """
    # get all tiff files in the folder
    files = [f for f in os.listdir(patches_folder) if f.endswith('.tif')]
    if len(files) == 0:
        raise FileNotFoundError(f'No tiff files found in {patches_folder}')

    # get the original shape of the patches
    try:
        patch_shape = [
            int(s) for s in
            files[0].split('_patch_shape_(')[1].split(')')[0].split(',')
        ]
    except Exception:
        raise ValueError(f'Problem when processing {files[0]}. '
                         f'Check that the file name is in the correct format '
                         f'and has patch shape.')

    # load the patches
    patches = None
    for i, f in enumerate(files):
        patch = tif.imread(Path(patches_folder, f))
        if i == 0:
            one_patch_shape = patch.shape
            patches = np.zeros((*patch_shape, *one_patch_shape),
                               dtype=patch.dtype)
        try:
            # get the indices from the end of the file name and place the patch
            ii, jj, kk = [int(s) for s in f.split('.tif')[0].split('_')[-3:]]
        except Exception:
            raise ValueError(f'Problem when processing {f}. '
                             f'Check that the file name is in the correct '
                             f'format and has patch indices.')
        patches[ii, jj, kk] = patch
    return patches


def load_and_unpatchify(patches_folder: Union[str, Path],
                        image_size: Tuple[int, int, int] = None,
                        ) -> npt.NDArray:
    """
    Load patches from the folder and unpatch them to the original shape.
    The patches should be saved in the format of ``{tag}_{i}.tif``.
    """
    patches = load_patches(patches_folder)

    # get image shape from the file name
    if image_size is None:
        files = [
            f for f in os.listdir(patches_folder) if f.endswith('.tif')
        ]
        image_size = [
            int(s) for s in
            files[0].split('_img_shape_(')[1].split(')')[0].split(',')
        ]
    return unpatchify(patches, image_size)


def patchify_from_config(img: npt.NDArray,
                         output_folder, config, verbose=True):
    """
    Patchifies an image and saves the patches to the output folder using
    parameters from the config file.
    """
    patch_size = config["patch_size"]
    step = config["step"]
    if "tag" in config:
        patch_tag = config["tag"]
    else:
        patch_tag = "patch"
    if verbose:
        print(f"patch size {patch_size}, step {step}")
        print(f"Saving patches to {output_folder} with tag {patch_tag}")
    patchify_and_save(img, patch_size, output_folder, patch_tag, step=step)


def unpatchify_from_config(input_folder, config, verbose=True):
    """
    Loads patches from the input folder using parameters from the config file
    and unpatchifies them.
    """
    image_size = config["image_size"]
    if verbose:
        print(f'Loading patches from {input_folder} '
              f'; unpatchifying into {image_size} image size')
    img = load_and_unpatchify(input_folder, image_size=image_size)
    return img


if __name__ == '__main__':
    # Example:
    # patchify a file and save the patches
    img = tif.imread('path/to/image.tif')
    # 128x128x128 patches with zero overlap
    patchify_and_save(img, 128, 'path/to/save/folder', 'tag')

    # load the patches and unpatchify them
    # (only patches originally with zero overlap
    # can be unpatchified with load_and_unpatchify)
    img = load_and_unpatchify('path/to/save/folder')
