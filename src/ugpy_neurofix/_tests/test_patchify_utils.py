import numpy as np
import matplotlib.pyplot as plt
from skimage import data
from skimage.transform import resize
from ugpy_neurofix.patchify_utils import patchify_and_save, load_and_unpatchify
import pytest


def test_patchify(tmp_path):
    # load the image
    img = data.camera()
    img = resize(img, (128, 128, 128), anti_aliasing=True)
    img = img.astype(np.float32)

    # patchify and save
    patchify_and_save(img, 32, tmp_path, 'test')

    # load and unpatchify
    img_recovered = load_and_unpatchify(tmp_path)
    assert np.allclose(img, img_recovered)

    # assertion error when the image size is not divisible by the patch size
    img = np.zeros((128, 128, 129))
    with pytest.raises(AssertionError) as e_info:
        patchify_and_save(img, 32, tmp_path, 'test')
    assert str(e_info.value) == \
           'The image size is not divisible by the patch size.'



