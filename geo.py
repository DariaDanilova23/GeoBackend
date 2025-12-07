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
import os
from matplotlib.colors import LinearSegmentedColormap
from functools import lru_cache
import logging
import tempfile
import uvicorn
import asyncio
from typing import List, Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if 'PROJ_LIB' in os.environ:
    del os.environ['PROJ_LIB']  

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Кэширование создания цветовых карт, т.к. они неизменны
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
    if src.crs is None:
        return safe_crs_from_epsg(default_epsg)
    
    try:
        crs = CRS.from_user_input(src.crs)
        return crs
    except Exception:
        return safe_crs_from_epsg(default_epsg)

async def save_uploaded_file_to_temp(file: UploadFile, max_size_mb: int = 200) -> str:
    """Сохраняет загруженный файл во временный файл с потоковой записью"""
    # Проверяем размер файла
    file.file.seek(0, 2)  # Переход в конец
    file_size = file.file.tell()
    file.file.seek(0)  # Возврат в начало
    
    if file_size > max_size_mb * 1024 * 1024:
        raise HTTPException(
            status_code=400,
            detail=f"Файл {file.filename} превышает максимальный размер {max_size_mb}MB"
        )
    
    # Создаем временный файл
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.tif')
    temp_path = temp_file.name
    temp_file.close()
    
    try:
        # Потоковая запись файла
        with open(temp_path, 'wb') as f:
            while True:
                chunk = await file.read(1024 * 1024)  # Читаем по 1MB
                if not chunk:
                    break
                f.write(chunk)
        
        # Валидация файла
        with rasterio.open(temp_path) as src:
            if src.count < 1:
                raise ValueError(f"Файл {file.filename} не содержит данных")
            
            return temp_path
                
    except Exception as e:
        # Очистка в случае ошибки
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        raise HTTPException(
            status_code=400,
            detail=f"Ошибка обработки файла {file.filename}: {str(e)}"
        )

async def read_and_validate_files(*files) -> List[Dict[str, Any]]:
    """Читает и валидирует файлы с сохранением во временные файлы"""
    file_data = []
    temp_files = []
    
    try:
        for file in files:
            # Сохраняем файл на диск
            temp_path = await save_uploaded_file_to_temp(file)
            temp_files.append(temp_path)
            
            # Читаем метаданные
            with rasterio.open(temp_path) as src:
                file_data.append({
                    'path': temp_path,
                    'profile': src.profile,
                    'shape': src.shape,
                    'crs': src.crs,
                    'transform': src.transform,
                    'dtype': src.dtypes[0],
                    'count': src.count
                })
        
        return file_data
        
    except Exception as e:
        # Очистка временных файлов в случае ошибки
        for temp_path in temp_files:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
        raise

def calculate_ndvi_chunked(red_path: str, nir_path: str) -> str:
    """Вычисление NDVI по кускам"""
    temp_output = tempfile.NamedTemporaryFile(delete=False, suffix='.tif')
    output_path = temp_output.name
    temp_output.close()
    
    try:
        with rasterio.open(red_path) as red_src, \
             rasterio.open(nir_path) as nir_src:
       
            profile = red_src.profile.copy()
            
            profile.update({
                'dtype': 'float32',
                'count': 1,
                'driver': 'GTiff',
                'tiled': True,
                'blockxsize': 256,
                'blockysize': 256,
                'compress': 'lzw'
            })
            
            if 'nodata' in profile:
                del profile['nodata']
            
            with rasterio.open(output_path, 'w', **profile) as dst:
                #Обрабатываем по блокам
                for _, window in red_src.block_windows(1):
                    #Читение только текущего блок
                    red_chunk = red_src.read(1, window=window).astype('float32')
                    nir_chunk = nir_src.read(1, window=window).astype('float32')
                    
                    #Вычисление NDVI для блока
                    denominator = nir_chunk + red_chunk
                    mask = denominator != 0
                    
                    ndvi_chunk = np.zeros_like(red_chunk, dtype='float32')
                    
                    if np.any(mask):
                        ndvi_chunk[mask] = (nir_chunk[mask] - red_chunk[mask]) / denominator[mask]

                    ndvi_chunk = np.clip(ndvi_chunk, -1, 1)
                    
                    #Для невалидных пикселей 0
                    ndvi_chunk[~mask] = 0
                    
                    dst.write(ndvi_chunk, 1, window=window)
            
            return output_path
            
    except Exception as e:
        if os.path.exists(output_path):
            os.unlink(output_path)
        raise

def calculate_evi_chunked(blue_path: str, red_path: str, nir_path: str, 
                         L: float = 1.0, C1: float = 6.0, C2: float = 7.5, G: float = 2.5) -> str:
    """Вычисление EVI по кускам"""
    temp_output = tempfile.NamedTemporaryFile(delete=False, suffix='.tif')
    output_path = temp_output.name
    temp_output.close()
    
    try:
        with rasterio.open(blue_path) as blue_src, \
             rasterio.open(red_path) as red_src, \
             rasterio.open(nir_path) as nir_src:
            
            profile = blue_src.profile.copy()
            profile.update({
                'dtype': 'float32',
                'count': 1,
                'driver': 'GTiff',
                'tiled': True,
                'blockxsize': 256,
                'blockysize': 256,
                'compress': 'lzw'
            })
            
            if 'nodata' in profile:
                del profile['nodata']
            
            with rasterio.open(output_path, 'w', **profile) as dst:
                for _, window in blue_src.block_windows(1):
                    blue_chunk = blue_src.read(1, window=window).astype('float32')
                    red_chunk = red_src.read(1, window=window).astype('float32')
                    nir_chunk = nir_src.read(1, window=window).astype('float32')
                    
                    #Вычисление EVI
                    numerator = nir_chunk - red_chunk
                    denominator = nir_chunk + C1 * red_chunk - C2 * blue_chunk + L
                    
                    mask = denominator != 0
                    evi_chunk = np.zeros_like(blue_chunk, dtype='float32')
                    
                    if np.any(mask):
                        evi_chunk[mask] = G * (numerator[mask] / denominator[mask])
                    
                    evi_chunk = np.clip(evi_chunk, 0, 1)
                    evi_chunk[~mask] = 0
                    
                    dst.write(evi_chunk, 1, window=window)
            
            return output_path
            
    except Exception as e:
        if os.path.exists(output_path):
            os.unlink(output_path)
        raise

def calculate_cvi_chunked(green_path: str, red_path: str, nir_path: str) -> str:
    """Вычисление CVI по кускам"""
    temp_output = tempfile.NamedTemporaryFile(delete=False, suffix='.tif')
    output_path = temp_output.name
    temp_output.close()
    
    try:
        with rasterio.open(green_path) as green_src, \
             rasterio.open(red_path) as red_src, \
             rasterio.open(nir_path) as nir_src:
            
            profile = green_src.profile.copy()
            profile.update({
                'dtype': 'float32',
                'count': 1,
                'driver': 'GTiff',
                'tiled': True,
                'blockxsize': 256,
                'blockysize': 256,
                'compress': 'lzw'
            })
            
            if 'nodata' in profile:
                del profile['nodata']
            
            all_values = []
            
            for _, window in green_src.block_windows(1):
                green_chunk = green_src.read(1, window=window).astype('float32')
                red_chunk = red_src.read(1, window=window).astype('float32')
                nir_chunk = nir_src.read(1, window=window).astype('float32')
                
                green_squared = green_chunk ** 2
                mask = green_squared != 0
                
                if np.any(mask):
                    cvi_chunk = np.zeros_like(green_chunk, dtype='float32')
                    cvi_chunk[mask] = (nir_chunk[mask] * red_chunk[mask]) / green_squared[mask]
                    
                    valid_values = cvi_chunk[mask]
                    all_values.extend(valid_values.flatten())
            
            #Вычисление диапазона для нормализации
            if all_values:
                cvi_min = np.percentile(all_values, 2)
                cvi_max = np.percentile(all_values, 98)
                diff = cvi_max - cvi_min
            else:
                cvi_min, cvi_max, diff = 0, 1, 1
            
            with rasterio.open(output_path, 'w', **profile) as dst:
                for _, window in green_src.block_windows(1):
                    green_chunk = green_src.read(1, window=window).astype('float32')
                    red_chunk = red_src.read(1, window=window).astype('float32')
                    nir_chunk = nir_src.read(1, window=window).astype('float32')
                    
                    green_squared = green_chunk ** 2
                    mask = green_squared != 0
                    
                    cvi_chunk = np.zeros_like(green_chunk, dtype='float32')
                    
                    if np.any(mask):
                        cvi_chunk[mask] = (nir_chunk[mask] * red_chunk[mask]) / green_squared[mask]
                        
                        #Нормализация
                        if diff > 0:
                            cvi_norm = (cvi_chunk[mask] - cvi_min) / diff
                            cvi_chunk[mask] = np.clip(cvi_norm, 0, 1)
                        else:
                            cvi_chunk[mask] = 0
                    
                    dst.write(cvi_chunk, 1, window=window)
            
            return output_path
            
    except Exception as e:
        if os.path.exists(output_path):
            os.unlink(output_path)
        raise

def create_colored_geotiff_chunked(index_path: str, cmap_name: str, index_name: str) -> bytes:
    """Создание цветного GeoTIFF по чанкам"""
    cmap = get_cmap(cmap_name)
    
    with rasterio.open(index_path) as src:
        # Читаем профиль исходного файла
        profile = src.profile.copy()
        
        # Обновляем профиль для RGB файла (uint8 без nodata)
        profile.update({
            'dtype': 'uint8',
            'count': 3,
            'driver': 'GTiff',
            'tiled': True,
            'blockxsize': 256,
            'blockysize': 256,
            'compress': 'lzw',
            'photometric': 'rgb'
        })
        
        if 'nodata' in profile:
            del profile['nodata']
        
        with MemoryFile() as memfile:
            with memfile.open(**profile) as dst:
                #Обрабатка по блокам
                for _, window in src.block_windows(1):
                    #Получаем индекс для текущего блока
                    index_chunk = src.read(1, window=window)
                    
                    if index_name == 'NDVI':
                        normalized_chunk = (index_chunk + 1) / 2
                    else:
                        normalized_chunk = np.clip(index_chunk, 0, 1)
                    
                    #Применение цветовой карты
                    colored_chunk = cmap(normalized_chunk, bytes=True)[:, :, :3]

                    for i in range(3):
                        dst.write(colored_chunk[:, :, i], i + 1, window=window)
                
                dst.set_band_description(1, "Red channel")
                dst.set_band_description(2, "Green channel")
                dst.set_band_description(3, "Blue channel")
                
                dst.colorinterp = [ColorInterp.red, ColorInterp.green, ColorInterp.blue]
            
            return memfile.read()

def cleanup_temp_files(*file_paths):
    """Очистка временных файлов"""
    for file_path in file_paths:
        if file_path and os.path.exists(file_path):
            try:
                os.unlink(file_path)
            except Exception as e:
                logger.warning(f"Не удалось удалить временный файл {file_path}: {str(e)}")

@app.post("/upload-raster-layer")
async def upload_layer(file: UploadFile = File(...)):
    return {"message": "Layer uploaded successfully"}

@app.post("/NDVI")
async def calculate_ndvi(
    red_file: UploadFile = File(...),
    nir_file: UploadFile = File(...),
    target_epsg: int = 4326
):
    red_data = None
    nir_data = None
    ndvi_path = None
    temp_files = []
    
    try:
        files_data = await read_and_validate_files(red_file, nir_file)
        
        red_data = files_data[0]
        nir_data = files_data[1]
        temp_files.extend([red_data['path'], nir_data['path']])
        
        if red_data['shape'] != nir_data['shape']:
            raise HTTPException(status_code=400, detail="Image dimensions mismatch")
        
        target_crs = safe_crs_from_epsg(target_epsg)
        
        #Вычисление NDVI по кускам
        ndvi_path = calculate_ndvi_chunked(red_data['path'], nir_data['path'])
        temp_files.append(ndvi_path)
        
        # Создаем цветное изображение
        colored_bytes = create_colored_geotiff_chunked(ndvi_path, "NDVI", "NDVI")
        
        return StreamingResponse(
            io.BytesIO(colored_bytes),
            media_type="image/tiff",
            headers={
                "Content-Disposition": "attachment; filename=ndvi_colored.tif",
                "CRS-Info": str(target_crs.to_string())
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"NDVI calculation error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    finally:
        #Очищаем временные файлы
        cleanup_temp_files(*temp_files)

@app.post("/EVI")
async def calculate_evi(
    blue_file: UploadFile = File(...),
    red_file: UploadFile = File(...),
    nir_file: UploadFile = File(...),
    target_epsg: int = 4326,
    L: float = 1.0, C1: float = 6.0, C2: float = 7.5, G: float = 2.5
):
    blue_data = None
    red_data = None
    nir_data = None
    evi_path = None
    temp_files = []
    
    try:
        files_data = await read_and_validate_files(blue_file, red_file, nir_file)
        
        blue_data = files_data[0]
        red_data = files_data[1]
        nir_data = files_data[2]
        temp_files.extend([blue_data['path'], red_data['path'], nir_data['path']])

        if not (blue_data['shape'] == red_data['shape'] == nir_data['shape']):
            raise HTTPException(status_code=400, detail="Размеры изображений не совпадают")
        
        target_crs = safe_crs_from_epsg(target_epsg)
        
        #Вычисляем EVI по кускам
        evi_path = calculate_evi_chunked(
            blue_data['path'], red_data['path'], nir_data['path'], 
            L, C1, C2, G
        )
        temp_files.append(evi_path)
        
        colored_bytes = create_colored_geotiff_chunked(evi_path, "EVI", "EVI")
        
        return StreamingResponse(
            io.BytesIO(colored_bytes),
            media_type="image/tiff",
            headers={
                "Content-Disposition": "attachment; filename=evi_colored.tif",
                "CRS-Info": str(target_crs.to_string())
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"EVI calculation error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    finally:
        cleanup_temp_files(*temp_files)

@app.post("/CVI")
async def calculate_cvi(
    green_file: UploadFile = File(...),
    red_file: UploadFile = File(...),
    nir_file: UploadFile = File(...),
    target_epsg: int = 4326
):
    green_data = None
    red_data = None
    nir_data = None
    cvi_path = None
    temp_files = []
    
    try:
        files_data = await read_and_validate_files(green_file, red_file, nir_file)
        
        green_data = files_data[0]
        red_data = files_data[1]
        nir_data = files_data[2]
        temp_files.extend([green_data['path'], red_data['path'], nir_data['path']])
        
        if not (green_data['shape'] == red_data['shape'] == nir_data['shape']):
            raise HTTPException(status_code=400, detail="Размеры изображений не совпадают")
        
        target_crs = safe_crs_from_epsg(target_epsg)
        
        #Вычисляем CVI по кускам
        cvi_path = calculate_cvi_chunked(green_data['path'], red_data['path'], nir_data['path'])
        temp_files.append(cvi_path)
        
        colored_bytes = create_colored_geotiff_chunked(cvi_path, "CVI", "CVI")
        
        return StreamingResponse(
            io.BytesIO(colored_bytes),
            media_type="image/tiff",
            headers={
                "Content-Disposition": "attachment; filename=cvi_colored.tif",
                "CRS-Info": str(target_crs.to_string())
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"CVI calculation error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    finally:
        cleanup_temp_files(*temp_files)

if __name__ == "__main__":
    config = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="debug",
        timeout_graceful_shutdown=10.0,
    )
    server = uvicorn.Server(config)
    asyncio.run(server.serve())
