import argparse
import logging
import os
from pathlib import Path
from typing import List, Tuple, Union

import cv2
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from requests import Session
from tqdm import tqdm

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger()

# Config to download images
root_dir = "./cxr_dataset/"
metadata = "./mimic-cxr-2.0.0-metadata.csv"

# URL of the site to login
site_url = "https://www.physionet.org/login/"
# Credentials
username = "phphuc"
password = "@ABCDE12345@"


def create_file_url(subject_id: str, study_id: str, image_id: str) -> str:
    """
    Create a URL to download a file from PhysioNet.
    """
    return f"https://www.physionet.org/files/mimic-cxr-jpg/2.0.0/files/p{subject_id[0:2]}/p{subject_id}/s{study_id}/{image_id}.jpg"


def create_saved_path(
    root_dir: Union[str, Path], subject_id: str, study_id: str, image_id: str
) -> Path:
    """
    Create a path to save a file from PhysioNet.
    """
    return (
        Path(root_dir)
        / f"p{subject_id[0:2]}/p{subject_id}/s{study_id}/{image_id}.jpg"
    )


def transform_image_from_bytes(image: bytes) -> np.ndarray:
    """
    Transform an image from bytes to a NumPy array.
    """
    nparr = np.frombuffer(image, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    height, width = img.shape[:2]

    # Calculate the new dimensions
    new_width = int(width / 4)
    new_height = int(height / 4)

    # Resize the image
    img = cv2.resize(
        img, (new_width, new_height), interpolation=cv2.INTER_AREA
    )
    return img


def download_images(
    session: Session, images_info: List[Tuple[str, str, str]]
) -> List[Tuple[str, str, str]]:
    """
    Download an image from PhysioNet.
    """
    failed_download = []

    for image_info in tqdm(images_info):
        subject_id, study_id, image_id = image_info
        url = create_file_url(subject_id, study_id, image_id)
        path = create_saved_path(root_dir, subject_id, study_id, image_id)

        os.makedirs(path.parent, exist_ok=True)

        logger.info(
            f"Start downloading {subject_id} - {study_id} - {image_id}\nFrom URL:{url}\nSaved to: {path}"
        )

        response = session.get(url)
        if response.status_code == 200:
            logger.info("Downloaded successfully - Transforming image...")
            img = transform_image_from_bytes(response.content)
            cv2.imwrite(str(path), img)
            logger.info(f"Saved image to {path}")
        else:
            logger.warning(
                f"Failed to download {subject_id} - {study_id} - {image_id}."
            )
            failed_download.append(image_info)


def get_images_info(metadata_path: str):
    """
    Get a list of images info from the metadata file.
    """
    df = pd.read_csv(metadata_path, dtype=str)
    images_info = df[["subject_id", "study_id", "dicom_id"]].values.tolist()
    return images_info


def main(job_id: int, n_jobs: int = 5):
    # Create a session object
    session = requests.Session()

    # Get login CSRF token
    response = session.get(site_url)
    soup = BeautifulSoup(response.text, "html.parser")
    csrfmiddlewaretoken = soup.find("input", dict(name="csrfmiddlewaretoken"))[
        "value"
    ]

    headers = {
        "referer": "https://www.physionet.org/login/",
    }

    # Login
    payload = {
        "username": username,
        "password": password,
        "csrfmiddlewaretoken": csrfmiddlewaretoken,
        "next": "/",
    }
    login_response = session.post(site_url, data=payload, headers=headers)

    # Check if login was successful
    if login_response.status_code == 200:
        logger.debug("Login successful")

        imgs_info = get_images_info(metadata)

        n_images = len(imgs_info)
        n_images_per_job = n_images // n_jobs
        start_index = job_id * n_images_per_job
        end_index = (job_id + 1) * n_images_per_job
        if job_id == n_jobs - 1:
            end_index = n_images
        imgs_info = imgs_info[start_index:end_index]

        failed_download = download_images(session, imgs_info)
        pd.DataFrame(
            data=failed_download,
            columns=["subject_id", "study_id", "dicom_id"],
        ).to_csv(f"failed_download_{job_id}_on_{n_jobs}.csv", index=False)
    else:
        logger.warn("Login failed\n" + login_response.text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download cxr images from physio."
    )
    parser.add_argument("job_id", type=int, help="Job index to run (< n_jobs)")
    parser.add_argument(
        "n_jobs",
        type=int,
        default=5,
        help="Number of jobs are expected to run",
    )

    args = parser.parse_args()

    main(args.job_id, args.n_jobs)
