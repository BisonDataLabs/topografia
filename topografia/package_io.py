from __future__ import annotations

import hashlib
import io
import json
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any


MAX_FILES = 2_000
MAX_UNCOMPRESSED_BYTES = 1_500_000_000


@dataclass(frozen=True)
class TopographyPackage:
    root: Path
    catalog: dict[str, Any]
    package: dict[str, Any]


def _safe_name(name: str) -> bool:
    path = PurePosixPath(name)
    return bool(name) and not path.is_absolute() and ".." not in path.parts


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_package(content: bytes) -> TopographyPackage:
    package_hash = hashlib.sha256(content).hexdigest()
    root = Path(tempfile.gettempdir()) / "topografia-streamlit" / package_hash
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        members = archive.infolist()
        if len(members) > MAX_FILES:
            raise ValueError(f"El paquete supera el máximo de {MAX_FILES} archivos.")
        if sum(member.file_size for member in members) > MAX_UNCOMPRESSED_BYTES:
            raise ValueError("El paquete supera el tamaño descomprimido admitido.")
        if any(not _safe_name(member.filename) for member in members):
            raise ValueError("El paquete contiene rutas no admitidas.")
        names = {member.filename for member in members}
        if not {"layers.json", "package.json"} <= names:
            raise ValueError("El ZIP no contiene layers.json y package.json.")
        root.mkdir(parents=True, exist_ok=True)
        archive.extractall(root)
    catalog = json.loads((root / "layers.json").read_text(encoding="utf-8"))
    package = json.loads((root / "package.json").read_text(encoding="utf-8"))
    if catalog.get("schema") != "topografia-layer-catalog":
        raise ValueError("layers.json no corresponde al catálogo topográfico.")
    if package.get("schema") != "topografia-streamlit-package":
        raise ValueError("package.json no corresponde al paquete Streamlit.")
    layers = catalog.get("layers")
    if not isinstance(layers, list) or not layers:
        raise ValueError("El paquete no contiene capas.")
    for layer in layers:
        relative = PurePosixPath(str(layer.get("path", "")))
        path = root.joinpath(*relative.parts).resolve()
        if root.resolve() not in path.parents or not path.is_file():
            raise ValueError(f"Falta la capa declarada: {relative}")
    for item in package.get("files", []):
        relative = PurePosixPath(str(item.get("path", "")))
        path = root.joinpath(*relative.parts).resolve()
        if not path.is_file() or _sha256(path) != item.get("sha256"):
            raise ValueError(f"Falló la verificación de integridad: {relative}")
    return TopographyPackage(root=root, catalog=catalog, package=package)
