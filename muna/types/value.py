# 
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

from enum import StrEnum
from io import BytesIO
from numpy import ndarray
from PIL import Image

class Dtype(StrEnum):
    """
    Value type.
    """
    null        = "null"
    bfloat16    = "bfloat16"
    float16     = "float16"
    float32     = "float32"
    float64     = "float64"
    int8        = "int8"
    int16       = "int16"
    int32       = "int32"
    int64       = "int64"
    uint8       = "uint8"
    uint16      = "uint16"
    uint32      = "uint32"
    uint64      = "uint64"
    complex64   = "complex64"
    complex128  = "complex128"
    bool        = "bool"
    string      = "string"
    list        = "list"
    dict        = "dict"
    image       = "image"
    binary      = "binary"
    array_list  = "array_list"
    image_list  = "image_list"

Value = (
    None                |
    float               |
    int                 |
    bool                |
    complex             |
    ndarray             |
    str                 |
    list[object]        |
    dict[str, object]   |
    Image.Image         |
    bytes               |
    BytesIO
)