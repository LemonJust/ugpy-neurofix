"""Classes and methods for reprocessing raw image and label data for
training, testing and inference."""

import os
import numpy as np
import numpy.typing as npt
import tifffile as tif
from pathlib import Path
from typing import Union, List, Tuple, Optional, Literal
import scipy.ndimage as ndi
import scipy.spatial as spatial
from skimage import io, morphology
from bm3d import bm3d
from time import time
import wandb
import yaml


def preprocess_from_config(config_file):
    """
    Preprocesses images based on the instructions in
    the config file. The config file is a yaml file with the following
    structure:
    ```yaml
        input:
            folder: path/to/input/folder
            filename: input_filename
            tag: input_tag
        output:
            folder: path/to/output/folder
            tag: output_tag
        preprocessing:
            - name: preprocessing_method_1
                args:
                    arg1: value1
                    arg2: value2
            - name: preprocessing_method_2
                args:
                    arg1: value1
                    arg2: value2
    ```
    """
    # load config file
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    # load input image
    input_folder = Path(config["input"]["folder"])
    input_filename = config["input"]["filename"]
    input_tag = config["input"]["tag"]
    input_path = input_folder / input_filename

    # load output folder
    output_folder = Path(config["output"]["folder"])
    output_tag = config["output"]["tag"]

    # load preprocessing methods
    preprocessing = config["preprocessing"]

    # load image
    img = Image(input_path, input_tag)

    # preprocess image
    tag = input_tag
    for method in preprocessing:
        tag = getattr(img, method["name"])(**method["args"])

    # save image
    img.save(output_folder, output_tag)


class Image:
    """
    Class for preprocessing raw image data.

    Args:
        image : Image data or a Path to the image file.
        tag : Tag for the image.
        info : Dictionary of corresponding information.

    Attributes:
        img : Dictionary of image data.
        info : Dictionary of corresponding information.
    """

    def __init__(self,
                 image: Union[npt.NDArray, str, Path],
                 tag: str,
                 info: Union[dict, str, None] = None):
        self.img: dict = {}
        self.info: dict = {}
        self.add_image(image, tag, info)

    def add_image(self, image: Union[npt.NDArray, str, Path],
                  tag: str,
                  info: Union[dict, str, None] = None):
        """
        Add image data to the object.

        Args:
            image : Image data or a Path to the image file.
            tag : Tag for the image.
            info : Dictionary of corresponding information.
        """
        if isinstance(image, (str, Path)):
            self.img[tag] = tif.imread(image)
        else:
            self.img[tag] = image
        self.info[tag] = info

    def save(self, save_folder: Union[str, Path],
             save_tag: Union[str, List[str]] = '',
             tags: Union[str, List[str]] = None,
             skip: Union[str, List[str]] = []):
        """
        Save the image to the folder. Concatenates all tags into a filename.

        Args:
            save_folder : Folder to save the image.
            save_tag : Extra tag for the filename,
                will be added at the beginning of the filename.
            tags : Tags of the image to save. All images will be saved if None.
            skip : Tags of the image to skip when saving.
                Useful when tags = None
        """
        if isinstance(save_tag, str):
            save_tag = [save_tag]

        if isinstance(skip, str):
            skip = [skip]

        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        if tags is None:
            tags = self.img.keys()
        elif isinstance(tags, str):
            tags = [tags]

        for tag in tags:
            if tag in skip:
                continue
            tif.imwrite(Path(save_folder, '_'.join(save_tag) + tag + '.tif'),
                        self.img[tag])

    def scale(self, tag: str,
              scale: Union[int, Tuple[int, int, int]],
              order: int = 3, keep_original=False):
        """
        Upscale the image by a factor of scale.

        Args:
            scale : Scaling factor in each dimension.
                If int, the same factor will be used in all dimensions.
            order : The order of the spline interpolation, default is 3.
            tag : Tag of the image to scale.

            keep_original : Whether to keep the original image.
                Default is False.
                If True, both the original and scaled images will be kept in
                the img dictionary. If False, only the scaled image will be
                kept in the img dictionary with the updated tag.
        """
        if isinstance(scale, int):
            scale = (scale, scale, scale)

        scaled_img = ndi.zoom(self.img[tag], scale, order=order)

        new_tag = tag + f'_scaled_{scale}_order_{order}'
        self.add_image(scaled_img, new_tag)

        if not keep_original:
            self.img.pop(tag)

        return new_tag

    def crop(self, tag: str,
             shape: Optional[Tuple[int, int, int]] = None,
             mode: Literal['center', '000'] = 'center',
             slices: Optional[Tuple[slice, slice, slice]] = None,
             keep_original=False):
        """
        Crop the image to certain size.
        Using to size the images to the same size.

        Args:
            shape : Shape of the cropped image.
            mode : Mode of cropping.
                'center' will crop the center of the image in all dimensions.
                '000' will crop starting from 0 in all dimensions (ZYX).
            slices : Tuple of slices for cropping.
            tag : Tag of the image to crop.
        """
        if shape is None and slices is None:
            raise ValueError('shape or slices must be specified.')

        if slices is None:
            if mode == 'center':
                slices = tuple(slice((self.img[tag].shape[i] - shape[i]) // 2,
                                     (self.img[tag].shape[i] + shape[i]) // 2)
                               for i in range(3))
            elif mode == '000':
                slices = tuple(slice(0, shape[i]) for i in range(3))
            else:
                raise ValueError('mode must be "center" or "000".')

        cropped_img = self.img[tag][slices]
        self.add_image(cropped_img,
                       tag + f'_cropped_{slices}_shape_{cropped_img.shape}')

        if not keep_original:
            self.img.pop(tag)

        return new_tag

    def gaussian_filter(self, tag: str, sigma: float,keep_original=False):
        """
        Apply Gaussian filter to the image.

        Args:
            sigma : Standard deviation for Gaussian kernel.
            tag : Tag of the image to filter.
        """
        filtered_img = ndi.gaussian_filter(self.img[tag], sigma)

        new_tag = tag + f'_gaussian_{sigma}'
        self.add_image(filtered_img, new_tag)
        return new_tag

    def spacial_std_filter(self, tag: str, size: int, keep_original=False):
        """
        Apply spacial standard deviation filter to the image.

        Args:
            size : Size of the filter.
            tag : Tag of the image to filter.
        """
        filtered_img = ndi.generic_filter(self.img[tag],
                                          np.std,
                                          size=size)

        new_tag = tag + f'_spacial_std_{size}'
        self.add_image(filtered_img, new_tag)
        return new_tag

    def normalize(self, tag: str,
                  mode: Literal['minmax', 'standard', 'prc'] = 'minmax',
                  prc: Tuple[float, float] = (1, 99), keep_original=False):
        """
        Normalize the image.

        Args:
            mode : Mode of normalization.
                'minmax' will normalize to [0, 1].
                'standardise' will standardise using mean and std.
                'prc' will normalize to [0, 1] using percentiles.
            tag : Tag of the image to normalize.
            prc : Percentiles for normalization.
                Only used when mode is 'prc'. Default is (1, 99).
        """
        if mode == 'minmax':
            min_value = self.img[tag].min()
            max_value = self.img[tag].max()
            norm_img = (self.img[tag] - min_value) / (max_value - min_value)
        elif mode == 'standard':
            norm_img = \
                (self.img[tag] - self.img[tag].mean()) / self.img[tag].std()
        elif mode == 'prc':
            min_value = np.percentile(self.img[tag], prc[0])
            max_value = np.percentile(self.img[tag], prc[1])
            norm_img = (self.img[tag] - min_value) / (max_value - min_value)
        else:
            raise ValueError('mode must be "minmax" or "standard" or "prc".')

        new_tag = tag + f'_norm_{mode}'
        if mode == 'prc':
            new_tag += f'_{prc[0]}_{prc[1]}'
        self.add_image(norm_img, new_tag)

        return new_tag

    def closing(self, tag: str, size: int = 3):
        """
        A wrapper for scipy.ndimage.grey_closing:
            Multidimensional grayscale closing.
            A grayscale closing consists in the succession of a grayscale dilation,
            and a grayscale erosion.
            The action of a grayscale closing with a flat structuring element
            amounts to smoothen deep local minima,
            whereas binary closing fills small holes.

            https://en.wikipedia.org/wiki/Mathematical_morphology

        Args:
            size : Size of the ball for closing.
            tag : Tag of the image to filter.
        """
        filtered_img = morphology.closing(self.img[tag],
                                          footprint=morphology.ball(size))
        self.add_image(filtered_img, tag + f'_closing_{size}')

    def opening(self, tag: str, size: int):
        """
        Apply opening to the image.

        Args:
            size : Size of the ball for opening.
            tag : Tag of the image to filter.
        """
        filtered_img = morphology.opening(
            self.img[tag], footprint=morphology.ball(size))
        self.add_image(filtered_img, tag + f'_opening_{size}')

    def dilation(self, tag: str, size: int):
        """
        Apply dilation to the image.

        Args:
            size : Size of the ball for dilation.
            tag : Tag of the image to filter.
        """
        filtered_img = morphology.dilation(
            self.img[tag], footprint=morphology.ball(size))
        self.add_image(filtered_img, tag + f'_dilation_{size}')

    def erosion(self, tag: str, size: int):
        """
        Apply erosion to the image.

        Args:
            size : Size of the ball for erosion.
            tag : Tag of the image to filter.
        """
        filtered_img = morphology.erosion(
            self.img[tag], footprint=morphology.ball(size))
        self.add_image(filtered_img, tag + f'_erosion_{size}')

    def binary_closing(self, tag: str, size: int):
        """
        Apply binary closing to the image.

        Args:
            size : Size of the ball for binary closing.
            tag : Tag of the image to filter.
        """
        filtered_img = morphology.binary_closing(
            self.img[tag], footprint=morphology.ball(size))

        new_tag = tag + f'_binary_closing_{size}'
        self.add_image(filtered_img, new_tag)

        return new_tag

    def binary_opening(self, tag: str, size: int):
        """
        Apply binary opening to the image.

        Args:
            size : Size of the ball for binary opening.
            tag : Tag of the image to filter.
        """
        filtered_img = morphology.binary_opening(
            self.img[tag], footprint=morphology.ball(size))

        new_tag = tag + f'_binary_opening_{size}'
        self.add_image(filtered_img, new_tag)

        return new_tag

    def binary_dilation(self, tag: str, size: int):
        """
        Apply binary dilation to the image.

        Args:
            size : Size of the ball for binary dilation.
            tag : Tag of the image to filter.
        """
        filtered_img = morphology.binary_dilation(
            self.img[tag], footprint=morphology.ball(size))
        self.add_image(filtered_img, tag + f'_binary_dilation_{size}')

    def binary_erosion(self, tag: str, size: int):
        """
        Apply binary erosion to the image.

        Args:
            size : Size of the ball for binary erosion.
            tag : Tag of the image to filter.
        """
        filtered_img = morphology.binary_erosion(
            self.img[tag], footprint=morphology.ball(size))
        self.add_image(filtered_img, tag + f'_binary_erosion_{size}')

    def binary_fill_holes(self, tag: str):
        """
        Apply binary fill holes to the image.

        Args:
            tag : Tag of the image to filter.
        """
        filtered_img = ndi.binary_fill_holes(self.img[tag])
        self.add_image(filtered_img, tag + f'_binary_fill_holes')

    def binary_closing_with_holes(self, tag: str, size: int):
        """
        Apply binary closing to the image.

        Args:
            size : Size of the ball for binary closing.
            tag : Tag of the image to filter.
        """
        filtered_img = morphology.binary_closing(
            self.img[tag], footprint=morphology.ball(size))
        filtered_img = ndi.binary_fill_holes(filtered_img)
        self.add_image(
            filtered_img, tag + f'_binary_closing_with_holes_{size}')

    def binary_opening_with_holes(self, tag: str, size: int):
        """
        Apply binary opening to the image.

        Args:
            size : Size of the ball for binary opening.
            tag : Tag of the image to filter.
        """
        filtered_img = morphology.binary_opening(
            self.img[tag], footprint=morphology.ball(size))
        filtered_img = ndi.binary_fill_holes(filtered_img)
        self.add_image(
            filtered_img, tag + f'_binary_opening_with_holes_{size}')

    def remove_noise(self, tag: str, size: int):
        """
        Remove noise from the image.

        Args:
            size : Size of the ball for binary opening.
            tag : Tag of the image to filter.
        """
        filtered_img = morphology.binary_opening(
            self.img[tag], footprint=morphology.ball(size))
        filtered_img = morphology.binary_closing(
            filtered_img, footprint=morphology.ball(size))
        self.add_image(filtered_img, tag + f'_remove_noise_{size}')

    def get_boundary(self, tag: str):
        """
        Get the boundary of the image.

        Args:
            tag : Tag of the image to filter.
        """
        # TODO : maybe just dilation , not binary?
        filtered_img = morphology.binary_dilation(
            self.img[tag], footprint=morphology.ball(3))
        filtered_img = filtered_img - self.img[tag]
        self.add_image(filtered_img, tag + f'_boundary')

    def flood_fill_hull(self, tag: str):
        """
        copied from
        # 3D Cellpose Extension.
        # Copyright (C) 2021 D. Eschweiler, J. Stegmaier

        Flood fill the convex hull of the image by using the Delaunay
        triangulation of the convex hull. This is a fast way to fill the convex
        hull of a binary image.
        """
        points = np.transpose(np.where(self.img[tag]))
        hull = spatial.ConvexHull(points)
        deln = spatial.Delaunay(points[hull.vertices])
        idx = np.stack(np.indices(self.img[tag].shape), axis=-1)
        out_idx = np.nonzero(deln.find_simplex(idx) + 1)
        out_img = np.zeros(self.img[tag].shape)
        out_img[out_idx] = 1
        self.add_image(out_img, tag + f'_flood_fill_hull')

    def denoise_bm3d(self, tag: str, sigma: float = 0.1):
        """
        Denoise the image with the bd3d algorithm.
        Image should be scaled before running this!
        For more information see: https://pypi.org/project/bm3d/
        Can also see demos here: ugpy-neurofix/extras/bm3d_demos
        """
        filtered_img = bm3d(self.img[tag], sigma)
        self.add_image(filtered_img, tag + f'_bd3d_{sigma}')


if __name__ == '__main__':
    # Apply all the filters and save to file
    # (this is just for testing purposes)
    img = Image(
        'D:/Code/repos/ugpy-neurofix/data/GADbRbC2_HuCDmouseC3-Fish1-1.tif',
        '3D')

    # img.gaussian_filter('3D', 3)

    # somehow it takes forever to run this
    # img.spacial_std_filter('3D', 5)

    # img.normalize('3D', mode='prc')

    # takes about 40 min , doesn't make it look better :(
    # img.denoise_bm3d('3D_norm_prc_1_99', 0.1)

    # img.closing('3D', 3)
    # img.opening('3D', 3)
    # img.dilation('3D', 3)
    # img.erosion('3D', 3)

    # img.binary_closing('3D', 3)
    # img.binary_opening('3D', 3)
    # img.binary_dilation('3D', 3)
    # img.binary_erosion('3D', 3)
    # img.binary_fill_holes('3D')
    # img.binary_closing_with_holes('3D', 3)
    # img.binary_opening_with_holes('3D', 3)

    # almost all 0 image, some 255
    img.remove_noise('3D', 3)

    # funny looking mostly white image
    # maybe not for the grey scale images
    img.get_boundary('3D')

    # returned all 1 image
    img.flood_fill_hull('3D')

    print("saving image")
    img.save('D:/Code/repos/ugpy-neurofix/data/3D',
             skip=['3D', '3D_norm_prc_1_99'])
