"""Where the data lives.  By default `<project>/data/`; set the environment variable GS_DATA_DIR to read it from somewhere else
(on Colab the data is extracted to the VM's fast local disk while the code and outputs live on Google Drive)."""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def data_dir(name=""):
    return os.path.join(os.environ.get("GS_DATA_DIR", os.path.join(ROOT, "data")), name)
