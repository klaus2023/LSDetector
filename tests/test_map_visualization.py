import json
from pathlib import Path

import pytest

from lsd_app.core.map_visualization import build_interactive_map_preview


def _write_rgb_geotiff(path: Path) -> None:
    np = pytest.importorskip("numpy")
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_origin

    rows, cols = np.indices((8, 8))
    data = np.stack(
        [
            (cols * 20 + 30).astype("uint8"),
            (rows * 20 + 30).astype("uint8"),
            np.full((8, 8), 100, dtype="uint8"),
        ]
    )
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=8,
        height=8,
        count=3,
        dtype="uint8",
        crs="EPSG:4326",
        transform=from_origin(10.0, 20.0, 0.01, 0.01),
    ) as dst:
        dst.write(data)


def _write_rgb_geotiff_at(path: Path, *, west: float, north: float, value: int) -> None:
    np = pytest.importorskip("numpy")
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_origin

    data = np.stack(
        [
            np.full((8, 8), value, dtype="uint8"),
            np.full((8, 8), max(0, value - 20), dtype="uint8"),
            np.full((8, 8), max(0, value - 40), dtype="uint8"),
        ]
    )
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=8,
        height=8,
        count=3,
        dtype="uint8",
        crs="EPSG:4326",
        transform=from_origin(west, north, 0.01, 0.01),
    ) as dst:
        dst.write(data)


def _write_landslide_geojson(path: Path) -> None:
    payload = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"id": 7, "area_m2": 1250.0, "mean_confidence": 0.82},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [10.02, 19.98],
                            [10.05, 19.98],
                            [10.05, 19.95],
                            [10.02, 19.95],
                            [10.02, 19.98],
                        ]
                    ],
                },
            }
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_interactive_map_preview_writes_map_controls(tmp_path: Path) -> None:
    pytest.importorskip("geopandas")
    pytest.importorskip("folium")
    raster_path = tmp_path / "source.tif"
    vector_path = tmp_path / "landslides.geojson"
    _write_rgb_geotiff(raster_path)
    _write_landslide_geojson(vector_path)

    result = build_interactive_map_preview(
        raster_path=raster_path,
        vector_path=vector_path,
        output_dir=tmp_path / "visualizations",
        RGB_bands="RGB_bands: 1,2,3",
        max_dimension=4,
        simplify_pixels=0.0,
        max_features=10,
        polygon_opacity=0.4,
    )

    assert result.width == 4
    assert result.height == 4
    assert result.feature_count == 1
    assert result.displayed_feature_count == 1
    assert result.base_preview_path.exists()
    assert result.vector_preview_path.exists()
    assert result.map_html_path.exists()
    html = result.map_html_path.read_text(encoding="utf-8")
    assert "GeoTIFF RGB basemap" in html
    assert "Landslide polygons" in html
    assert "Latitude/longitude grid" in html
    assert "north-arrow-control" in html
    assert "L.control.scale" in html
    assert "Mean confidence" in html


def test_interactive_map_preview_accepts_multiple_basemap_geotiffs(tmp_path: Path) -> None:
    pytest.importorskip("geopandas")
    pytest.importorskip("folium")
    raster_a = tmp_path / "tile_0001_mosaic.tif"
    raster_b = tmp_path / "tile_0002_mosaic.tif"
    vector_path = tmp_path / "landslides.geojson"
    _write_rgb_geotiff_at(raster_a, west=10.0, north=20.0, value=120)
    _write_rgb_geotiff_at(raster_b, west=10.08, north=20.0, value=200)
    _write_landslide_geojson(vector_path)

    result = build_interactive_map_preview(
        raster_paths=[raster_a, raster_b],
        vector_path=vector_path,
        output_dir=tmp_path / "visualizations",
        RGB_bands="RGB_bands: 1,2,3",
        max_dimension=16,
        simplify_pixels=0.0,
        max_features=10,
        polygon_opacity=0.4,
    )

    assert result.raster_paths == (raster_a.resolve(), raster_b.resolve())
    assert result.width == 16
    assert result.height == 8
    assert result.feature_count == 1
    assert result.map_html_path.exists()
    assert "Basemap mosaic uses 2 GeoTIFF(s)" in result.message

