# 
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

from io import BytesIO
from muna import Dtype
from muna.c import Value
from numpy import allclose, load as npz_load
from numpy.random import randn
from numpy.lib.npyio import NpzFile
from pathlib import Path

def test_serialize_ndarray_to_npz():
    # Create tensor value
    tensor = randn(100, 100, 3)
    tensor_value = Value.from_object(tensor)
    tensor_data = tensor_value.serialize()
    # Deserialize
    npz_file: NpzFile = npz_load(BytesIO(tensor_data))
    encoded_tensor = next(iter(npz_file.values()))
    # Check
    assert allclose(tensor, encoded_tensor)

def test_deserialize_png_to_image():
    # Load image data
    image_data = Path("test/data/cat.jpg").read_bytes()
    image_value = Value.from_bytes(image_data, mime="image/*")
    # Check
    assert image_value.type == Dtype.image
    assert image_value.shape == (224, 224, 3)