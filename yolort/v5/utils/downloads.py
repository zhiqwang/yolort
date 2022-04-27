# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Download utils
"""

import os
import platform
import subprocess
import time
import urllib
from pathlib import Path
from zipfile import ZipFile

import requests
from torch.hub import download_url_to_file


def gsutil_getsize(url=""):
    # gs://bucket/file size https://cloud.google.com/storage/docs/gsutil/commands/du
    s = subprocess.check_output(f"gsutil du {url}", shell=True).decode("utf-8")
    return eval(s.split(" ")[0]) if len(s) else 0  # bytes


def safe_download(file, url, url2=None, min_bytes=1e0, error_msg="", hash_prefix=None):
    """
    Attempts to download file from url or url2, checks
    and removes incomplete downloads < min_bytes
    """
    file = Path(file)
    assert_msg = f"Downloaded file '{file}' does not exist or size is < min_bytes={min_bytes}"
    try:  # url1
        print(f"Downloading {url} to {file}...")
        download_url_to_file(url, str(file), hash_prefix=hash_prefix)
        assert file.exists() and file.stat().st_size > min_bytes, assert_msg  # check
    except Exception as e:  # url2
        file.unlink(missing_ok=True)  # remove partial downloads
        print(f"ERROR: {e}\nRe-attempting {url2 or url} to {file}...")
        # curl download, retry and resume on fail
        os.system(f"curl -L '{url2 or url}' -o '{file}' --retry 3 -C -")
    finally:
        if not file.exists() or file.stat().st_size < min_bytes:  # check
            file.unlink(missing_ok=True)  # remove partial downloads
            print(f"ERROR: {assert_msg}\n{error_msg}")
        print("")


def attempt_download(file, repo="ultralytics/yolov5", hash_prefix=None):
    # Attempt file download if does not exist
    file = Path(str(file).strip().replace("'", ""))

    if not file.exists():
        # URL specified
        name = Path(urllib.parse.unquote(str(file))).name  # decode '%2F' to '/' etc.
        if str(file).startswith(("http:/", "https:/")):  # download
            url = str(file).replace(":/", "://")  # Pathlib turns :// -> :/
            name = name.split("?")[0]  # parse authentication https://url.com/file.txt?auth...
            safe_download(file=name, url=url, min_bytes=1e5, hash_prefix=hash_prefix)
            return name

        # GitHub assets
        file.parent.mkdir(parents=True, exist_ok=True)
        try:
            # github api
            response = requests.get(f"https://api.github.com/repos/{repo}/releases/latest").json()
            assets = [x["name"] for x in response["assets"]]
            tag = response["tag_name"]  # i.e. 'v1.0'
        except Exception as e:  # fallback plan
            print(f"Wrong when calling GitHub API: {e}")
            assets = [
                "yolov5n.pt",
                "yolov5s.pt",
                "yolov5m.pt",
                "yolov5l.pt",
                "yolov5x.pt",
                "yolov5n6.pt",
                "yolov5s6.pt",
                "yolov5m6.pt",
                "yolov5l6.pt",
                "yolov5x6.pt",
            ]
            try:
                tag = (
                    subprocess.check_output("git tag", shell=True, stderr=subprocess.STDOUT)
                    .decode()
                    .split()[-1]
                )
            except Exception as e:
                print(f"Wrong when getting GitHub tag: {e}")
                tag = "v6.0"  # current release

        if name in assets:
            safe_download(
                file,
                url=f"https://github.com/{repo}/releases/download/{tag}/{name}",
                min_bytes=1e5,
                error_msg=f"{file} missing, try downloading from https://github.com/{repo}/releases/",
            )

    return str(file)


def gdrive_download(id="16TiPfZj7htmTyhntwcZyEEAejOUxuT6m", file="tmp.zip"):
    # Downloads a file from Google Drive.
    t = time.time()
    file = Path(file)
    cookie = Path("cookie")  # gdrive cookie
    print(f"Downloading https://drive.google.com/uc?export=download&id={id} as {file}... ", end="")
    file.unlink(missing_ok=True)  # remove existing file
    cookie.unlink(missing_ok=True)  # remove existing cookie

    # Attempt file download
    out = "NUL" if platform.system() == "Windows" else "/dev/null"
    os.system(f'curl -c ./cookie -s -L "drive.google.com/uc?export=download&id={id}" > {out}')
    if os.path.exists("cookie"):  # large file
        s = f"curl -Lb ./cookie 'drive.google.com/uc?export=download&confirm={get_token()}&id={id}'"
    else:  # small file
        s = f'curl -s -L "drive.google.com/uc?export=download&id={id}"'
    download_excute = f"{s} -o {file}"
    r = os.system(download_excute)
    cookie.unlink(missing_ok=True)  # remove existing cookie

    # Error check
    if r != 0:
        file.unlink(missing_ok=True)  # remove partial
        print("Download error ")  # raise Exception('Download error')
        return r

    # Unzip if archive
    if file.suffix == ".zip":
        print("unzipping... ", end="")
        ZipFile(file).extractall(path=file.parent)  # unzip
        file.unlink()  # remove zip

    print(f"Done ({time.time() - t:.1f}s)")
    return r


def get_token(cookie="./cookie"):
    with open(cookie) as f:
        for line in f:
            if "download" in line:
                return line.split()[-1]
    return ""
