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
    assert image_value.dtype == Dtype.image
    assert image_value.shape == (224, 224, 3)

def test_serialize_array_list_to_npz():
    arrays = [randn(16000).astype("float32"), randn(32000).astype("float32")]
    value = Value.from_object(arrays)
    data = value.serialize("application/x-npz")
    npz_file: NpzFile = npz_load(BytesIO(data))
    for original, encoded in zip(arrays, npz_file.values()):
        assert allclose(original, encoded)

def test_deserialize_npz_to_array_list():
    arrays = [randn(16000).astype("float32"), randn(32000).astype("float32")]
    value = Value.from_object(arrays)
    data = value.serialize("application/x-npz")
    deserialized_value = Value.from_bytes(data, "application/x-npz")
    result = deserialized_value.to_object()
    assert isinstance(result, list)
    assert len(result) == len(arrays)
    for original, decoded in zip(arrays, result):
        assert allclose(original, decoded)