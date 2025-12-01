from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import rasterio
from rasterio.io import MemoryFile
from rasterio.crs import CRS
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.enums import ColorInterp
import io
import traceback
import os
from matplotlib.colors import LinearSegmentedColormap
from functools import lru_cache
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if 'PROJ_LIB' in os.environ:
    del os.environ['PROJ_LIB']  # Удаляем проблемную переменную

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Кэшируем создание цветовых карт, так как они неизменны
@lru_cache(maxsize=3)
def get_cmap(name):
    if name == 'NDVI':
        colors = [
            (0.0, (1.0, 0.0, 0.0)), (0.5, (1.0, 0.0, 0.0)),
            (0.57, (1.0, 1.0, 0.0)), (0.665, (0.7, 1.0, 0.2)),
            (0.83, (0.0, 0.5, 0.0)), (1.0, (0.0, 0.2, 0.0))
        ]
    elif name == 'EVI':
        colors = [
            (0.0, (0.5, 0.5, 0.5)), (0.3, (0.8, 0.8, 0.2)),
            (0.6, (0.2, 0.8, 0.2)), (1.0, (0.0, 0.4, 0.0))
        ]
    elif name == 'CVI':
        colors = [
            (0.0, (0.8, 0.8, 0.8)), (0.3, (0.5, 0.8, 0.3)),
            (0.6, (0.2, 0.6, 0.2)), (0.8, (0.8, 0.6, 0.2)),
            (1.0, (0.8, 0.2, 0.2))
        ]
    return LinearSegmentedColormap.from_list(name.lower(), colors)

WGS84_WKT = """
GEOGCS["WGS 84",
    DATUM["WGS_1984",
        SPHEROID["WGS 84",6378137,298.257223563,
            AUTHORITY["EPSG","7030"]],
        AUTHORITY["EPSG","6326"]],
    PRIMEM["Greenwich",0,
        AUTHORITY["EPSG","8901"]],
    UNIT["degree",0.0174532925199433,
        AUTHORITY["EPSG","9122"]],
    AUTHORITY["EPSG","4326"]]
"""
def safe_crs_from_epsg(epsg_code):
    """Безопасное создание CRS с обработкой ошибок PROJ"""
    try:
        return CRS.from_epsg(epsg_code)
    except Exception as e:
        logger.warning(f"Ошибка при создании CRS из EPSG:{epsg_code}: {str(e)}")
        try:
            return CRS.from_wkt(WGS84_WKT)
        except Exception as e:
            logger.error(f"Ошибка при создании CRS из WKT: {str(e)}")
            raise ValueError("Не удалось создать CRS")

def strict_crs_validation(src, default_epsg=4326):
    """Проверка CRS без использования is_valid"""
    if src.crs is None:
        return safe_crs_from_epsg(default_epsg)
    
    try:
        crs = CRS.from_user_input(src.crs)
        # Вместо is_valid просто проверяем, что CRS создан без ошибок
        return crs
    except Exception:
        return safe_crs_from_epsg(default_epsg)

def reproject_with_transform(src, data, target_crs):
    """Оптимизированная репроекция"""
    try:
        transform, width, height = calculate_default_transform(
            src.crs, target_crs, src.width, src.height, *src.bounds)
        
        destination = np.empty((height, width), dtype=data.dtype)
        
        reproject(
            source=data,
            destination=destination,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs=target_crs,
            resampling=Resampling.bilinear
        )
        
        return destination, transform, width, height
    except Exception as e:
        logger.warning(f"Reprojection failed, using original: {str(e)}")
        return data, src.transform, src.width, src.height

async def read_and_validate_files(*files):
    """Чтение и валидация файлов"""
    file_data = []
    for file in files:
        content = await file.read()
        try:
            with MemoryFile(content) as memfile:
                with memfile.open() as src:
                    if src.count < 1:
                        raise ValueError(f"Файл {file.filename} не содержит данных")
                    file_data.append((content, src.profile, src.shape))
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Ошибка обработки файла {file.filename}: {str(e)}"
            )
    return file_data

def create_colored_geotiff(data, src_profile, crs, transform, cmap_name, index_name):
    """Создание GeoTIFF без использования is_valid"""
    cmap = get_cmap(cmap_name)
    
    if index_name == 'NDVI':
        normalized_data = (data + 1) / 2
    else:
        normalized_data = np.clip(data, 0, 1)
    
    colored_data = cmap(normalized_data, bytes=True)[:, :, :3]
    
    profile = {
        'driver': 'GTiff',
        'dtype': 'uint8',
        'count': 3,
        'width': src_profile['width'],
        'height': src_profile['height'],
        'transform': transform,
        'compress': 'lzw',
        'tiled': True,
        'blockxsize': 256,
        'blockysize': 256,
        'photometric': 'rgb'
    }
    
    #Проверяем, что crs не None
    if crs is not None:
        try:
            profile['crs'] = crs
        except Exception as e:
            logger.warning(f"Failed to set CRS: {str(e)}")
    
    with MemoryFile() as memfile:
        with memfile.open(**profile) as dst:
            for i in range(3):
                dst.write(colored_data[:, :, i], i+1)
            
            dst.set_band_description(1, "Red channel")
            dst.set_band_description(2, "Green channel")
            dst.set_band_description(3, "Blue channel")
            
            dst.colorinterp = [ColorInterp.red, ColorInterp.green, ColorInterp.blue]
        
        return memfile.read()
@app.post("/upload-raster-layer")
async def upload_layer(file: UploadFile = File(...)):
    return {"message": "Layer uploaded successfully"}

@app.post("/NDVI")
async def calculate_ndvi(
    red_file: UploadFile = File(..., max_size=200_000_000),
    nir_file: UploadFile = File(..., max_size=200_000_000),
    target_epsg: int = 4326
):
    try:
        red_content, red_profile, red_shape = (await read_and_validate_files(red_file))[0]
        nir_content, nir_profile, nir_shape = (await read_and_validate_files(nir_file))[0]
        
        if red_shape != nir_shape:
            raise HTTPException(status_code=400, detail="Image dimensions mismatch")

        target_crs = safe_crs_from_epsg(target_epsg)
        
        with MemoryFile(red_content) as memfile_red, MemoryFile(nir_content) as memfile_nir:
            with memfile_red.open() as red_src, memfile_nir.open() as nir_src:
                red_crs = strict_crs_validation(red_src, target_epsg)
                nir_crs = strict_crs_validation(nir_src, target_epsg)
                
                red = red_src.read(1).astype('float32')
                nir = nir_src.read(1).astype('float32')

                red, red_transform, _, _ = reproject_with_transform(red_src, red, target_crs)
                nir, _, _, _ = reproject_with_transform(nir_src, nir, target_crs)

                # Расчет NDVI с оптимизацией
                denominator = nir + red
                np.divide(nir - red, denominator, out=denominator, where=denominator!=0)
                ndvi = np.clip(denominator, -1, 1)

                # Создание результата
                ndvi_bytes = create_colored_geotiff(
                    ndvi, red_profile, target_crs, red_transform, 'NDVI', 'NDVI'
                )

                return StreamingResponse(
                    io.BytesIO(ndvi_bytes),
                    media_type="image/tiff",
                    headers={
                        "Content-Disposition": "attachment; filename=ndvi_colored.tif",
                        "CRS-Info": str(target_crs.to_string()),
                        "NDVI-Stats": f"min={np.nanmin(ndvi):.2f},max={np.nanmax(ndvi):.2f},mean={np.nanmean(ndvi):.2f}"
                    }
                )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"NDVI calculation error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/EVI")
async def calculate_evi(
    blue_file: UploadFile = File(..., max_size=200_000_000),
    red_file: UploadFile = File(..., max_size=200_000_000),
    nir_file: UploadFile = File(..., max_size=200_000_000),
    target_epsg: int = 4326,
    L: float = 1.0,
    C1: float = 6.0,
    C2: float = 7.5,
    G: float = 2.5
):
    """Расчет EVI (Enhanced Vegetation Index)"""
    try:
        # Чтение и проверка файлов
        blue_content, blue_profile, blue_shape = (await read_and_validate_files(blue_file))[0]
        red_content, red_profile, red_shape = (await read_and_validate_files(red_file))[0]
        nir_content, nir_profile, nir_shape = (await read_and_validate_files(nir_file))[0]
        
        if not (blue_shape == red_shape == nir_shape):
            raise HTTPException(status_code=400, detail="Размеры изображений не совпадают")

        target_crs = safe_crs_from_epsg(target_epsg)
        
        with MemoryFile(blue_content) as memfile_blue, \
             MemoryFile(red_content) as memfile_red, \
             MemoryFile(nir_content) as memfile_nir:
            
            with memfile_blue.open() as blue_src, \
                 memfile_red.open() as red_src, \
                 memfile_nir.open() as nir_src:
                
                strict_crs_validation  (blue_src, target_epsg)
                strict_crs_validation  (red_src, target_epsg)
                strict_crs_validation  (nir_src, target_epsg)
                
                blue = blue_src.read(1).astype('float32')
                red = red_src.read(1).astype('float32')
                nir = nir_src.read(1).astype('float32')

                blue, blue_transform, _, _ = reproject_with_transform(blue_src, blue, target_crs)
                red, _, _, _ = reproject_with_transform(red_src, red, target_crs)
                nir, _, _, _ = reproject_with_transform(nir_src, nir, target_crs)

                # Расчет EVI
                numerator = nir - red
                denominator = nir + C1 * red - C2 * blue + L
                denominator[denominator == 0] = np.nan
                evi = G * (numerator / denominator)
                evi = np.clip(evi, 0, 1)  # EVI обычно в диапазоне 0-1

                # Создание результата
                output_bytes = create_colored_geotiff(
                    evi, blue_src.profile, target_crs, blue_transform, 'EVI', 'EVI'
                )

                return StreamingResponse(
                    io.BytesIO(output_bytes),
                    media_type="image/tiff",
                    headers={
                        "Content-Disposition": "attachment; filename=evi_colored.tif",
                        "CRS-Info": str(target_crs.to_string()),
                        "EVI-Stats": f"min={np.nanmin(evi):.2f},max={np.nanmax(evi):.2f},mean={np.nanmean(evi):.2f}"
                    }
                )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"EVI calculation error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/CVI")
async def calculate_cvi(
    green_file: UploadFile = File(..., max_size=200_000_000),
    red_file: UploadFile = File(..., max_size=200_000_000),
    nir_file: UploadFile = File(..., max_size=200_000_000),
    target_epsg: int = 4326
):
    """Расчет CVI (Chlorophyll Vegetation Index)"""
    try:
        green_content, green_profile, green_shape = (await read_and_validate_files(green_file))[0]
        red_content, red_profile, red_shape = (await read_and_validate_files(red_file))[0]
        nir_content, nir_profile, nir_shape = (await read_and_validate_files(nir_file))[0]
        
        if not (green_shape == red_shape == nir_shape):
            raise HTTPException(status_code=400, detail="Размеры изображений не совпадают")

        target_crs = safe_crs_from_epsg(target_epsg)
        
        with MemoryFile(green_content) as memfile_green, \
             MemoryFile(red_content) as memfile_red, \
             MemoryFile(nir_content) as memfile_nir:
            
            with memfile_green.open() as green_src, \
                 memfile_red.open() as red_src, \
                 memfile_nir.open() as nir_src:
                
                strict_crs_validation(green_src, target_epsg)
                strict_crs_validation(red_src, target_epsg)
                strict_crs_validation(nir_src, target_epsg)
                
                green = green_src.read(1).astype('float32')
                red = red_src.read(1).astype('float32')
                nir = nir_src.read(1).astype('float32')

                green, green_transform, _, _ = reproject_with_transform(green_src, green, target_crs)
                red, _, _, _ = reproject_with_transform(red_src, red, target_crs)
                nir, _, _, _ = reproject_with_transform(nir_src, nir, target_crs)

                green_squared = green ** 2
                # Заменяем нули на NaN, чтобы избежать деления на ноль
                green_squared[green_squared == 0] = np.nan
                
                # Вычисляем CVI, игнорируя деление на ноль
                with np.errstate(divide='ignore', invalid='ignore'):
                    cvi = (nir * red) / green_squared
                
                cvi = np.nan_to_num(cvi, nan=0.0, posinf=0.0, neginf=0.0)
                
                # Автоматическое определение диапазона
                valid_values = cvi[np.isfinite(cvi)]
                if len(valid_values) > 0:
                    cvi_min = np.percentile(valid_values, 2)  # 2-й перцентиль
                    cvi_max = np.percentile(valid_values, 98) # 98-й перцентиль
                else:
                    cvi_min, cvi_max = 0.0, 1.0
                
                # Нормализация к 0-1 с обрезкой выбросов
                range_diff = cvi_max - cvi_min
                if range_diff > 0:
                    cvi_normalized = (cvi - cvi_min) / range_diff
                else:
                    cvi_normalized = np.zeros_like(cvi)
                cvi_normalized = np.clip(cvi_normalized, 0, 1)

                output_bytes = create_colored_geotiff(
                    cvi_normalized,
                    green_src.profile,
                    target_crs,
                    green_transform,
                    'CVI',
                    'CVI'
                )

                return StreamingResponse(
                    io.BytesIO(output_bytes),
                    media_type="image/tiff",
                    headers={
                        "Content-Disposition": "attachment; filename=cvi_colored.tif",
                        "CRS-Info": str(target_crs.to_string()),
                        "CVI-Stats": f"min={np.nanmin(cvi):.2f},max={np.nanmax(cvi):.2f},mean={np.nanmean(cvi):.2f}",
                        "CVI-Range": f"normalized_range={cvi_min:.2f}-{cvi_max:.2f}"
                    }
                )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"CVI calculation error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")
    
async def shutdown_event():
    print("Shutting down...")

app.add_event_handler("shutdown", shutdown_event)

if __name__ == "__main__":
    config = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="debug",
        # Добавляем graceful shutdown
        timeout_graceful_shutdown=10.0,
    )
    server = uvicorn.Server(config)
    asyncio.run(server.serve())
