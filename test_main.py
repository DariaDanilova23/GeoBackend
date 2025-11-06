from fastapi.testclient import TestClient
import io
import os
from geo import app

client = TestClient(app)

# Путь до папки с tif
base_path = os.path.join(os.path.dirname(__file__), "files")

def test_NDVI():
    # Открываем реальные файлы
    with open(os.path.join(base_path, "red.tif"), "rb") as red, \
         open(os.path.join(base_path, "nir.tif"), "rb") as nir:
        response = client.post(
            "/NDVI",
            files={
                "red_file": ("red.tif", red, "image/tiff"),
                "nir_file": ("nir.tif", nir, "image/tiff"),
            }
        )
    assert response.status_code == 200
    assert response.headers["Content-Type"] == "image/tiff"

def test_CVI():
    # Открываем реальные файлы
    with open(os.path.join(base_path, "green.tif"), "rb") as green, \
         open(os.path.join(base_path, "red.tif"), "rb") as red, \
         open(os.path.join(base_path, "nir.tif"), "rb") as nir:
        response = client.post(
            "/CVI",
            files={
                "green_file":("green.tif",green,"image/tiff"),
                "red_file": ("red.tif", red, "image/tiff"),
                "nir_file": ("nir.tif", nir, "image/tiff"),
            }
        )
    assert response.status_code == 200
    assert response.headers["Content-Type"] == "image/tiff"

def test_EVI():
    with open(os.path.join(base_path, "blue.tif"), "rb") as blue, \
         open(os.path.join(base_path, "red.tif"), "rb") as red, \
         open(os.path.join(base_path, "nir.tif"), "rb") as nir:
        response = client.post(
            "/EVI",
            files={
                "blue_file":("blue.tif",blue,"image/tiff"),
                "red_file": ("red.tif", red, "image/tiff"),
                "nir_file": ("nir.tif", nir, "image/tiff"),
            }
        )
    assert response.status_code == 200
    assert response.headers["Content-Type"] == "image/tiff"

def test_NDVI_missing_file():
    # Открываем реальные файлы
    with open(os.path.join(base_path, "red.tif"), "rb") as red:
        response = client.post(
            "/NDVI",
            files={
                "red_file": ("red.tif", red, "image/tiff"),
            }
        )
    assert response.status_code == 422

def test_CVI_missing_file():
    # Открываем реальные файлы
    with open(os.path.join(base_path, "red.tif"), "rb") as red, \
         open(os.path.join(base_path, "nir.tif"), "rb") as nir:
        response = client.post(
            "/CVI",
            files={
                "red_file": ("red.tif", red, "image/tiff"),
                "nir_file": ("nir.tif", nir, "image/tiff"),
            }
        )
    assert response.status_code == 422

def test_EVI_missing_file():
    with open(os.path.join(base_path, "blue.tif"), "rb") as blue, \
         open(os.path.join(base_path, "nir.tif"), "rb") as nir:
        response = client.post(
            "/EVI",
            files={
                "blue_file":("blue.tif",blue,"image/tiff"),
                "nir_file": ("nir.tif", nir, "image/tiff"),
            }
        )
    assert response.status_code == 422


def test_NDVI_wrong_file_format():
    response = client.post(
        "/NDVI",
        files={
            "red_file": ("red.txt", io.BytesIO(b"not tiff"), "text/plain"),
            "nir_file": ("nir.txt", io.BytesIO(b"not tiff"), "text/plain"),
        }
    )
    assert response.status_code == 400  

def test_CVI_wrong_file_format():
    response = client.post(
        "/CVI",
        files={
            "green_file":("green.txt", io.BytesIO(b"not tiff"), "text/plain"),
            "red_file": ("red.txt", io.BytesIO(b"not tiff"), "text/plain"),
            "nir_file": ("nir.txt", io.BytesIO(b"not tiff"), "text/plain"),
        }
    )
    assert response.status_code == 400

def test_EVI_wrong_file_format():
    response = client.post(
        "/EVI",
        files={
            "blue_file":("blue.txt",io.BytesIO(b"not tiff"), "text/plain"),
            "red_file": ("red.txt", io.BytesIO(b"not tiff"), "text/plain"),
            "nir_file": ("nir.txt", io.BytesIO(b"not tiff"), "text/plain"),
        }
    )
    assert response.status_code == 400