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
from skimage import morphology
from bm3d import bm3d
import yaml
from patchify_utils import patchify_from_config, unpatchify_from_config
import warnings


def preprocess_image_from_config(config_file):
    """
    Preprocesses images based on the instructions in
    the config file. If the image will be patchified,
    the patches will be created from the image after preprocessing.
    The config file is a yaml file with the following
    structure:
    ```yaml
        input:
            folder: path/to/input/folder
            filename: input_filename # not needed if unpatchify is used
            tag: input_tag
        output:
            folder: path/to/output/folder
            tag: output_tag
        unpatchify: # optional
            image_size: !!python/tuple [64, 1024, 1024] # in ZYX order
        preprocessing: # optional
            - name: preprocessing_method_1
              args:
                arg1: value1
                arg2: value2
            - name: preprocessing_method_2
              args:
                arg1: value1
                arg2: value2
        patchify: # optional
            patch_size: 64
            step: 64
            tag: patch_tag
    ```
    """
    with open(config_file, "r") as f:
        config = yaml.unsafe_load(f)

    tag = config["input"]["tag"]
    input_folder = Path(config["input"]["folder"])

    # 1. unpatchify or load image
    if "unpatchify" in config:
        print(f"Unpatchifying {tag}")
        img = Image3D(
                    unpatchify_from_config(input_folder, config["unpatchify"]),
                    tag)
    else:
        input_file = input_folder / config["input"]["filename"]
        print(f"Loading {input_file} with tag {tag}")
        img = Image3D(input_file, tag)

    # 2. preprocess image
    if "preprocessing" in config:
        for method in config["preprocessing"]:
            print(f"Preprocessing {tag} with {method['name']}")
            method["args"]["tag"] = tag
            tag = getattr(img, method["name"])(**method["args"])

    # 3. save image or patchify and save patches
    output_folder = Path(config["output"]["folder"])
    if "tag" in config["output"]:
        output_tag = config["output"]["tag"]
    else:
        output_tag = ""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    output_tag = output_tag + tag

    if "patchify" in config:
        print(f"Patchifying {tag}")
        output_folder = output_folder / output_tag
        patchify_from_config(img.img[tag], output_folder, config["patchify"])
    else:
        print(f"Saving {tag} to {output_folder} as {output_tag}")
        img.save(output_folder, tags=tag)


class Image3D:
    """
    Class for preprocessing raw 3D image data.

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
        Add 3D image data to the object.
        If the image is 2D, it will be expanded to 3D.

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

        # check if the image is 3D
        if len(self.img[tag].shape) == 2:
            # show warning using warnings module
            warnings.warn(f"Image {tag} is 2D. Expanding to 3D.")
            self.img[tag] = self.img[tag][np.newaxis, ...]
        elif len(self.img[tag].shape) > 3:
            raise ValueError(f"Image {tag} has more than 3 dimensions.")

    def save(self, save_folder: Union[str, Path],
             save_tag: Union[str, List[str]] = '',
             tags: Union[str, List[str]] = None,
             skip: Union[str, List[str]] = ()):
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

    def pad(self, tag: str,
            padded_shape: Tuple[int, int, int],
            mode: Literal['constant', 'reflect', 'nearest'] = 'constant',
            constant_values: int = 0,
            keep_original=False):
        """
        Pad the image to certain size. Using to size the images to the same
        size. Not symmetrical: Will pad only one side of the image ( bottom,
        right , back)
        Args:
            padded_shape : Shape of the padded image. In zyx order.
            mode : Mode of padding. See numpy.pad for more details.
            constant_values : Used in 'constant' mode.
                The values to set the padded values for each axis.
            tag : Tag of the image to pad.
            keep_original : Whether to keep the original image.
                Default is False.
                If True, both the original and scaled images will be kept in
                the img dictionary. If False, only the scaled image will be
                kept in the img dictionary with the updated tag.
        """
        if padded_shape == self.img[tag].shape:
            return tag

        pad_width = []
        for i in range(3):
            pad_width.append((0, padded_shape[i] - self.img[tag].shape[i]))
        padded_img = np.pad(self.img[tag], pad_width, mode=mode,
                            constant_values=constant_values)

        new_tag = tag + f'_padded_{padded_shape}_{mode}'
        self.add_image(padded_img, new_tag)

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
            keep_original : Whether to keep the original image.
                Default is False.
                If True, both the original and scaled images will be kept in
                the img dictionary. If False, only the scaled image will be
                kept in the img dictionary with the updated tag.
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

        new_tag = tag + f'_cropped_{slices}_to_{cropped_img.shape}'
        self.add_image(cropped_img, new_tag)

        if not keep_original:
            self.img.pop(tag)

        return new_tag

    def gaussian_filter(self, tag: str, sigma: float, keep_original=False):
        """
        Apply Gaussian filter to the image.

        Args:
            sigma : Standard deviation for Gaussian kernel.
            tag : Tag of the image to filter.
            keep_original : Whether to keep the original image.
                Default is False.
                If True, both the original and scaled images will be kept in
                the img dictionary. If False, only the scaled image will be
                kept in the img dictionary with the updated tag.
        """
        filtered_img = ndi.gaussian_filter(self.img[tag], sigma)

        new_tag = tag + f'_gaussian_{sigma}'
        self.add_image(filtered_img, new_tag)

        if not keep_original:
            self.img.pop(tag)

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
            keep_original : Whether to keep the original image.
                Default is False.
                If True, both the original and scaled images will be kept in
                the img dictionary. If False, only the scaled image will be
                kept in the img dictionary with the updated tag.
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

        if not keep_original:
            self.img.pop(tag)

        return new_tag

    def closing(self, tag: str, size: int = 3, keep_original=False):
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
            keep_original : Whether to keep the original image.
                Default is False.
                If True, both the original and scaled images will be kept in
                the img dictionary. If False, only the scaled image will be
                kept in the img dictionary with the updated tag.
        """
        filtered_img = morphology.closing(self.img[tag],
                                          footprint=morphology.ball(size))

        new_tag = tag + f'_closing_{size}'
        self.add_image(filtered_img, new_tag)

        if not keep_original:
            self.img.pop(tag)

        return new_tag

    def opening(self, tag: str, size: int, keep_original=False):
        """
        Apply opening to the image.

        Args:
            size : Size of the ball for opening.
            tag : Tag of the image to filter.
            keep_original : Whether to keep the original image.
                Default is False.
                If True, both the original and scaled images will be kept in
                the img dictionary. If False, only the scaled image will be
                kept in the img dictionary with the updated tag.
        """
        filtered_img = morphology.opening(
            self.img[tag], footprint=morphology.ball(size))

        new_tag = tag + f'_opening_{size}'
        self.add_image(filtered_img, new_tag)

        if not keep_original:
            self.img.pop(tag)

        return new_tag

    def dilation(self, tag: str, size: int, keep_original=False):
        """
        Apply dilation to the image.

        Args:
            size : Size of the ball for dilation.
            tag : Tag of the image to filter.
            keep_original : Whether to keep the original image.
                Default is False.
                If True, both the original and scaled images will be kept in
                the img dictionary. If False, only the scaled image will be
                kept in the img dictionary with the updated tag.
        """
        filtered_img = morphology.dilation(
            self.img[tag], footprint=morphology.ball(size))

        new_tag = tag + f'_dilation_{size}'
        self.add_image(filtered_img, new_tag)

        if not keep_original:
            self.img.pop(tag)

        return new_tag

    def erosion(self, tag: str, size: int, keep_original=False):
        """
        Apply erosion to the image.

        Args:
            size : Size of the ball for erosion.
            tag : Tag of the image to filter.
            keep_original : Whether to keep the original image.
                Default is False.
                If True, both the original and scaled images will be kept in
                the img dictionary. If False, only the scaled image will be
                kept in the img dictionary with the updated tag.
        """
        filtered_img = morphology.erosion(
            self.img[tag], footprint=morphology.ball(size))

        new_tag = tag + f'_erosion_{size}'
        self.add_image(filtered_img, new_tag)

        if not keep_original:
            self.img.pop(tag)

        return new_tag

    def binary_closing(self, tag: str, size: int, keep_original=False):
        """
        Apply binary closing to the image.

        Args:
            size : Size of the ball for binary closing.
            tag : Tag of the image to filter.
            keep_original : Whether to keep the original image.
                Default is False.
                If True, both the original and scaled images will be kept in
                the img dictionary. If False, only the scaled image will be
                kept in the img dictionary with the updated tag.
        """
        filtered_img = morphology.binary_closing(
            self.img[tag], footprint=morphology.ball(size))

        new_tag = tag + f'_binary_closing_{size}'
        self.add_image(filtered_img, new_tag)

        if not keep_original:
            self.img.pop(tag)

        return new_tag

    def binary_opening(self, tag: str, size: int, keep_original=False):
        """
        Apply binary opening to the image.

        Args:
            size : Size of the ball for binary opening.
            tag : Tag of the image to filter.
            keep_original : Whether to keep the original image.
                Default is False.
                If True, both the original and scaled images will be kept in
                the img dictionary. If False, only the scaled image will be
                kept in the img dictionary with the updated tag.
        """
        filtered_img = morphology.binary_opening(
            self.img[tag], footprint=morphology.ball(size))

        new_tag = tag + f'_binary_opening_{size}'
        self.add_image(filtered_img, new_tag)

        if not keep_original:
            self.img.pop(tag)

        return new_tag

    def binary_dilation(self, tag: str, size: int, keep_original=False):
        """
        Apply binary dilation to the image.

        Args:
            size : Size of the ball for binary dilation.
            tag : Tag of the image to filter.
            keep_original : Whether to keep the original image.
                Default is False.
                If True, both the original and scaled images will be kept in
                the img dictionary. If False, only the scaled image will be
                kept in the img dictionary with the updated tag.
        """
        filtered_img = morphology.binary_dilation(
            self.img[tag], footprint=morphology.ball(size))

        new_tag = tag + f'_binary_dilation_{size}'
        self.add_image(filtered_img, new_tag)

        if not keep_original:
            self.img.pop(tag)

        return new_tag

    def binary_erosion(self, tag: str, size: int, keep_original=False):
        """
        Apply binary erosion to the image.

        Args:
            size : Size of the ball for binary erosion.
            tag : Tag of the image to filter.
            keep_original : Whether to keep the original image.
                Default is False.
                If True, both the original and scaled images will be kept in
                the img dictionary. If False, only the scaled image will be
                kept in the img dictionary with the updated tag.
        """
        filtered_img = morphology.binary_erosion(
            self.img[tag], footprint=morphology.ball(size))

        new_tag = tag + f'_binary_erosion_{size}'
        self.add_image(filtered_img, new_tag)

        if not keep_original:
            self.img.pop(tag)

        return new_tag

    def binary_fill_holes(self, tag: str, keep_original=False):
        """
        Apply binary fill holes to the image.

        Args:
            tag : Tag of the image to filter.
            keep_original : Whether to keep the original image.
                Default is False.
                If True, both the original and scaled images will be kept in
                the img dictionary. If False, only the scaled image will be
                kept in the img dictionary with the updated tag.
        """
        filtered_img = ndi.binary_fill_holes(self.img[tag])

        new_tag = tag + f'_binary_fill_holes'
        self.add_image(filtered_img, new_tag)

        if not keep_original:
            self.img.pop(tag)

        return new_tag

    def binary_closing_with_holes(self, tag: str, size: int,
                                  keep_original=False):
        """
        Apply binary closing to the image.

        Args:
            size : Size of the ball for binary closing.
            tag : Tag of the image to filter.
            keep_original : Whether to keep the original image.
                Default is False.
                If True, both the original and scaled images will be kept in
                the img dictionary. If False, only the scaled image will be
                kept in the img dictionary with the updated tag.
        """
        filtered_img = morphology.binary_closing(
            self.img[tag], footprint=morphology.ball(size))
        filtered_img = ndi.binary_fill_holes(filtered_img)

        new_tag = tag + f'_binary_closing_with_holes_{size}'
        self.add_image(filtered_img, new_tag)

        if not keep_original:
            self.img.pop(tag)

        return new_tag

    def binary_opening_with_holes(self, tag: str, size: int,
                                  keep_original=False):
        """
        Apply binary opening to the image.

        Args:
            size : Size of the ball for binary opening.
            tag : Tag of the image to filter.
            keep_original : Whether to keep the original image.
                Default is False.
                If True, both the original and scaled images will be kept in
                the img dictionary. If False, only the scaled image will be
                kept in the img dictionary with the updated tag.
        """
        filtered_img = morphology.binary_opening(
            self.img[tag], footprint=morphology.ball(size))
        filtered_img = ndi.binary_fill_holes(filtered_img)

        new_tag = tag + f'_binary_opening_with_holes_{size}'
        self.add_image(filtered_img, new_tag)

        if not keep_original:
            self.img.pop(tag)

        return new_tag

    def remove_noise(self, tag: str, size: int, keep_original=False):
        """
        Remove noise from the image.

        Args:
            size : Size of the ball for binary opening.
            tag : Tag of the image to filter.
            keep_original : Whether to keep the original image.
                Default is False.
                If True, both the original and scaled images will be kept in
                the img dictionary. If False, only the scaled image will be
                kept in the img dictionary with the updated tag.
        """
        filtered_img = morphology.binary_opening(
            self.img[tag], footprint=morphology.ball(size))
        filtered_img = morphology.binary_closing(
            filtered_img, footprint=morphology.ball(size))

        new_tag = tag + f'_remove_noise_{size}'
        self.add_image(filtered_img, new_tag)

        if not keep_original:
            self.img.pop(tag)

        return new_tag

    def get_boundary(self, tag: str, keep_original=False):
        """
        Get the boundary of the image.

        Args:
            tag : Tag of the image to filter.
            keep_original : Whether to keep the original image.
                Default is False.
                If True, both the original and scaled images will be kept in
                the img dictionary. If False, only the scaled image will be
                kept in the img dictionary with the updated tag.
        """
        # TODO : maybe just dilation , not binary?
        filtered_img = morphology.binary_dilation(
            self.img[tag], footprint=morphology.ball(3))
        filtered_img = filtered_img - self.img[tag]

        new_tag = tag + f'_boundary'
        self.add_image(filtered_img, new_tag)

        if not keep_original:
            self.img.pop(tag)

        return new_tag

    def flood_fill_hull(self, tag: str, keep_original=False):
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

        new_tag = tag + f'_flood_fill_hull'
        self.add_image(out_img, new_tag)

        if not keep_original:
            self.img.pop(tag)

        return new_tag

    def denoise_bm3d(self, tag: str, sigma: float = 0.1, keep_original=False):
        """
        Denoise the image with the bm3d algorithm.
        Image should be scaled before running this!
        For more information see: https://pypi.org/project/bm3d/
        Can also see demos here: ugpy-neurofix/extras/bm3d_demos
        """
        filtered_img = bm3d(self.img[tag], sigma)

        new_tag = tag + f'_bd3d_{sigma}'
        self.add_image(filtered_img, new_tag)

        if not keep_original:
            self.img.pop(tag)

        return new_tag


if __name__ == '__main__':
    # preprocess_image_from_config(
    #     'D:/Code/repos/ugpy-neurofix/configs/preprocess_config.yaml'
    # )
    preprocess_image_from_config(
        'D:/Code/repos/ugpy-neurofix/configs/unpatchify_config.yaml'
    )
