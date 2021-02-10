# !/usr/bin/env python
from pathlib import Path
import subprocess
from setuptools import setup, find_packages

PATH_ROOT = Path(__file__).parent
VERSION = "0.3.0rc1"

PACKAGE_NAME = 'yolort'
sha = 'Unknown'

try:
    sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=PATH_ROOT).decode('ascii').strip()
except Exception:
    pass


def write_version_file():
    version_path = PATH_ROOT / PACKAGE_NAME / 'version.py'
    with open(version_path, 'w') as f:
        f.write(f"__version__ = '{VERSION}'\n")
        f.write(f"git_version = {repr(sha)}\n")
        f.write("from torchvision.extension import _check_cuda_version\n")
        f.write("if _check_cuda_version() > 0:\n")
        f.write("    cuda = _check_cuda_version()\n")


def load_requirements(path_dir=PATH_ROOT, file_name='requirements.txt', comment_char='#'):
    with open(path_dir.joinpath(file_name), 'r', encoding="utf-8", errors="ignore") as file:
        lines = [ln.strip() for ln in file.readlines()]
    reqs = []
    for ln in lines:
        if comment_char in ln:  # filer all comments
            ln = ln[:ln.index(comment_char)].strip()
        if ln.startswith('http'):  # skip directly installed dependencies
            continue
        if ln:  # if requirement is not empty
            reqs.append(ln)
    return reqs


if __name__ == "__main__":
    print("Building wheel {}-{}".format(PACKAGE_NAME, VERSION))

    write_version_file()

    with open('README.md', encoding='utf-8') as f:
        readme = f.read()

    setup(
        name=PACKAGE_NAME,
        version=VERSION,
        description="Yet Another YOLOv5 and with Additional Runtime Stack",
        author="Zhiqiang Wang",
        author_email="me@zhiqwang.com",
        long_description=readme,
        long_description_content_type="text/markdown",
        url="https://github.com/zhiqwang/yolov5-rt-stack",
        license="GPL-3.0",
        packages=find_packages(exclude=['test', 'deployment', 'notebooks']),

        zip_safe=False,
        classifiers=[
            "Operating System :: POSIX :: Linux",
            # How mature is this project? Common values are
            #   3 - Alpha, 4 - Beta, 5 - Production/Stable
            "Development Status :: 4 - Beta",
            # Indicate who your project is intended for
            "Intended Audience :: Developers",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
            "Topic :: Scientific/Engineering :: Information Analysis",
            # Pick your license as you wish
            "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
            # Specify the Python versions you support here. In particular, ensure
            # that you indicate whether you support Python 2, Python 3 or both.
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.6",
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
        ],
        install_requires=load_requirements(),
    )
