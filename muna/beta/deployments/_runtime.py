#
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

FXNC_VERSION = "0.0.46"
MUNA_SERVER_VERSION = "0.0.3"
SERVER_PORT = 8000
TARGET_ARCH = "x86_64-unknown-linux-gnu"

FXNC_LIBRARY_URL = (
    f"https://cdn.fxn.ai/fxnc/{FXNC_VERSION}/"
    "libFunction-linux-x86_64.so"
)
MUNA_SERVER_URL = (
    "https://github.com/muna-ai/muna-server/releases/download/"
    f"{MUNA_SERVER_VERSION}/muna-server-{TARGET_ARCH}"
)
