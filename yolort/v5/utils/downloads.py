# YOLOv5 by Ultralytics, GPL-3.0 license
"""
Download utils
"""

import os
import subprocess
import urllib
from pathlib import Path

import requests
import torch


def safe_download(file, url, url2=None, min_bytes=1e0, error_msg=""):
    # Attempts to download file from url or url2, checks and removes incomplete downloads < min_bytes
    file = Path(file)
    assert_msg = f"Downloaded file '{file}' does not exist or size is < min_bytes={min_bytes}"
    try:  # url1
        print(f"Downloading {url} to {file}...")
        torch.hub.download_url_to_file(url, str(file))
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


def attempt_download(file, repo="ultralytics/yolov5"):
    # Attempt file download if does not exist
    file = Path(str(file).strip().replace("'", ""))

    if not file.exists():
        # URL specified
        name = Path(urllib.parse.unquote(str(file))).name  # decode '%2F' to '/' etc.
        if str(file).startswith(("http:/", "https:/")):  # download
            url = str(file).replace(":/", "://")  # Pathlib turns :// -> :/
            name = name.split("?")[0]  # parse authentication https://url.com/file.txt?auth...
            safe_download(file=name, url=url, min_bytes=1e5)
            return name

        # GitHub assets
        file.parent.mkdir(parents=True, exist_ok=True)  # make parent dir (if required)
        try:
            # github api
            response = requests.get(f"https://api.github.com/repos/{repo}/releases/latest").json()
            # release assets, i.e. ['yolov5s.pt', 'yolov5m.pt', ...]
            assets = [x["name"] for x in response["assets"]]
            tag = response["tag_name"]  # i.e. 'v1.0'
        except requests.exceptions.RequestException as e:  # fallback plan
            print(str(e))
            assets = [
                "yolov5s.pt",
                "yolov5m.pt",
                "yolov5l.pt",
                "yolov5x.pt",
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
            except subprocess.CalledProcessError:
                tag = "v5.0"  # current release

        if name in assets:
            safe_download(
                file,
                url=f"https://github.com/{repo}/releases/download/{tag}/{name}",
                # url2=f'https://storage.googleapis.com/{repo}/ckpt/{name}',  # backup url (optional)
                min_bytes=1e5,
                error_msg=f"{file} missing, try downloading from https://github.com/{repo}/releases/",
            )

    return str(file)


def get_token(cookie="./cookie"):
    with open(cookie) as f:
        for line in f:
            if "download" in line:
                return line.split()[-1]
    return ""
