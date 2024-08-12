# Author: Michael Drolet

name = "irl_control"
version_info = (2, 0, 0)  # (major, minor, patch)
dev = False

v = ".".join(str(v) for v in version_info)
dev_v = ".dev" if dev else ""

version = f"{v}{dev_v}"
