"""Optional TensorFlow pip-library discovery fix, confined to this virtualenv.

Based on https://www.tensorflow.org/install/pip#linux .
Run again after recreating the virtualenv if TensorFlow cannot discover CUDA.
"""
from pathlib import Path
import sysconfig

site = Path(sysconfig.get_paths()["purelib"])
tf = site / "tensorflow"
for source in (site / "nvidia").glob("*/lib/*.so*"):
    target = tf / source.name
    if not target.exists():
        target.symlink_to(source)
print("Linked installed NVIDIA libraries into", tf)
