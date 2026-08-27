from __future__ import annotations

import base64
import io
import math
from collections import deque
from dataclasses import dataclass

import numpy as np
from matplotlib import colormaps
from PIL import Image
from rasterio.enums import Resampling
from rasterio.features import geometry_mask
from rasterio.io import MemoryFile
from rasterio.transform import array_bounds
from rasterio.warp import transform, transform_bounds, transform_geom
from shapely.geometry import mapping


@dataclass(frozen=True)
class FlowOverlay:
    image_data_url: str
    bounds_wgs84: tuple[float, float, float, float]
    max_contributing_area_ha: float
    concentrated_area_percent: float
    downslope_direction: str
    downslope_bearing: float
    outlet_lon: float
    outlet_lat: float
    effective_cell_m: float
    display_low_ha: float
    display_high_ha: float


def _cardinal(bearing: float) -> str:
    labels = ["N", "NE", "E", "SE", "S", "SO", "O", "NO"]
    return labels[int((bearing + 22.5) // 45) % 8]


def _rgba_flow(area_ha: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, float, float]:
    positive = area_ha[valid & np.isfinite(area_ha) & (area_ha > 0)]
    low, high = np.nanpercentile(positive, [70, 99.5])
    scaled = np.clip((np.log1p(area_ha) - np.log1p(low)) / max(np.log1p(high) - np.log1p(low), 1e-9), 0, 1)
    pixels = (colormaps["Blues"](scaled) * 255).astype(np.uint8)
    pixels[..., 3] = np.where(valid & (area_ha >= low), 45 + 200 * scaled, 0).astype(np.uint8)
    return pixels, float(low), float(high)


def analyze_flow(content: bytes, boundary_wgs84, display_geometry_wgs84=None, maximum_cells: int = 1_200_000) -> FlowOverlay:
    """Preliminary D8 flow accumulation without depression breaching or stream burning."""
    with MemoryFile(content) as memory:
        with memory.open() as source:
            scale = min(1.0, math.sqrt(maximum_cells / max(source.width * source.height, 1)))
            width = max(2, round(source.width * scale))
            height = max(2, round(source.height * scale))
            elevation = source.read(1, out_shape=(height, width), resampling=Resampling.bilinear).astype("float64")
            affine = source.transform * source.transform.scale(source.width / width, source.height / height)
            valid = np.isfinite(elevation)
            if source.nodata is not None:
                valid &= elevation != source.nodata
            if not valid.any():
                raise ValueError("El DEM no contiene celdas válidas para analizar el flujo.")

            geometry = transform_geom("EPSG:4326", source.crs, mapping(boundary_wgs84))
            inside = geometry_mask([geometry], (height, width), affine, invert=True) & valid
            if not inside.any():
                raise ValueError("El DEM no intersecta el lote seleccionado.")
            display_geometry = transform_geom(
                "EPSG:4326", source.crs, mapping(display_geometry_wgs84 or boundary_wgs84)
            )
            display = geometry_mask([display_geometry], (height, width), affine, invert=True) & valid

            center_lat = boundary_wgs84.centroid.y
            if source.crs.is_geographic:
                dx = abs(affine.a) * 111_320 * math.cos(math.radians(center_lat))
                dy = abs(affine.e) * 110_574
            else:
                dx, dy = abs(affine.a), abs(affine.e)
            cell_area_ha = max(dx * dy / 10_000, 1e-9)

            destinations = np.full((height, width), -1, dtype=np.int64)
            best_drop = np.zeros((height, width), dtype="float64")
            offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
            distances = [math.hypot(dx, dy), dy, math.hypot(dx, dy), dx, dx, math.hypot(dx, dy), dy, math.hypot(dx, dy)]
            rows, cols = np.indices((height, width))
            for (row_shift, col_shift), distance in zip(offsets, distances):
                neighbor = np.full_like(elevation, np.nan)
                row_source = slice(max(0, row_shift), min(height, height + row_shift))
                col_source = slice(max(0, col_shift), min(width, width + col_shift))
                row_target = slice(max(0, -row_shift), min(height, height - row_shift))
                col_target = slice(max(0, -col_shift), min(width, width - col_shift))
                neighbor[row_target, col_target] = elevation[row_source, col_source]
                drop = (elevation - neighbor) / max(distance, 1e-9)
                better = valid & np.isfinite(neighbor) & (drop > best_drop)
                neighbor_rows = rows + row_shift
                neighbor_cols = cols + col_shift
                destinations[better] = (neighbor_rows[better] * width + neighbor_cols[better]).astype(np.int64)
                best_drop[better] = drop[better]

            flat_destinations = destinations.ravel()
            valid_flat = valid.ravel()
            has_destination = valid_flat & (flat_destinations >= 0)
            indegree = np.zeros(height * width, dtype=np.int32)
            np.add.at(indegree, flat_destinations[has_destination], 1)
            accumulation = np.where(valid_flat, 1.0, 0.0)
            queue = deque(np.flatnonzero(valid_flat & (indegree == 0)).tolist())
            while queue:
                cell = queue.popleft()
                destination = flat_destinations[cell]
                if destination >= 0:
                    accumulation[destination] += accumulation[cell]
                    indegree[destination] -= 1
                    if indegree[destination] == 0:
                        queue.append(int(destination))
            area_ha = accumulation.reshape(height, width) * cell_area_ha

            inside_area = area_ha[inside]
            threshold = max(1.0, float(np.nanpercentile(inside_area, 90)))
            concentrated_percent = float(100 * np.mean(inside_area >= threshold))
            masked_area = np.where(inside, area_ha, -np.inf)
            outlet_row, outlet_col = np.unravel_index(int(np.argmax(masked_area)), masked_area.shape)
            outlet_x, outlet_y = affine * (outlet_col + 0.5, outlet_row + 0.5)
            outlet_lon, outlet_lat = transform(source.crs, "EPSG:4326", [outlet_x], [outlet_y])

            inside_rows, inside_cols = np.where(inside)
            x = affine.c + (inside_cols + 0.5) * affine.a
            y = affine.f + (inside_rows + 0.5) * affine.e
            design = np.column_stack((x - x.mean(), y - y.mean(), np.ones_like(x)))
            coefficients, *_ = np.linalg.lstsq(design, elevation[inside], rcond=None)
            east_component, north_component = -coefficients[0], -coefficients[1]
            bearing = float((math.degrees(math.atan2(east_component, north_component)) + 360) % 360)

            pixels, display_low, display_high = _rgba_flow(area_ha, display)
            output = io.BytesIO()
            Image.fromarray(pixels, "RGBA").save(output, "PNG")
            west, south, east, north = array_bounds(height, width, affine)
            if source.crs.to_epsg() != 4326:
                west, south, east, north = transform_bounds(source.crs, "EPSG:4326", west, south, east, north)

    return FlowOverlay(
        image_data_url=f"data:image/png;base64,{base64.b64encode(output.getvalue()).decode('ascii')}",
        bounds_wgs84=(float(west), float(south), float(east), float(north)),
        max_contributing_area_ha=float(np.nanmax(inside_area)),
        concentrated_area_percent=concentrated_percent,
        downslope_direction=_cardinal(bearing),
        downslope_bearing=bearing,
        outlet_lon=float(outlet_lon[0]),
        outlet_lat=float(outlet_lat[0]),
        effective_cell_m=float(math.sqrt(dx * dy)),
        display_low_ha=display_low,
        display_high_ha=display_high,
    )
