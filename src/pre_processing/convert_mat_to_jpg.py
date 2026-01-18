#!/usr/bin/env python3
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.io import loadmat

INPUT_DIR = Path("/mnt/projects/tilapia_ia/data/raw_images/imagens_mat")
OUTPUT_DIR = Path("/mnt/projects/tilapia_ia/data/raw_images/imagens_jpg")
FILE_GLOB = "Chip_*.mat"
PREFIX = "imagem"

CHIP_RE = re.compile(r"^Chip_(\d+)\.mat$", re.IGNORECASE)


def _to_uint8(img: np.ndarray) -> np.ndarray:
    x = np.asarray(img)
    if x.dtype.kind == "f":
        if np.nanmax(x) <= 1.0:
            x = x * 255.0
    return np.clip(x, 0, 255).astype(np.uint8)


def _save_jpg(img: np.ndarray, out_path: Path) -> None:
    x = _to_uint8(img)

    # Caso venha como (H,W,1)
    if x.ndim == 3 and x.shape[2] == 1:
        x = x[:, :, 0]

    # Caso venha como (C,H,W)
    if x.ndim == 3 and x.shape[0] in (1, 3, 4) and x.shape[1] > 32 and x.shape[2] > 32:
        x = np.transpose(x, (1, 2, 0))

    Image.fromarray(x).save(out_path, format="JPEG", quality=95)


def _find_key_case_insensitive(d: dict, wanted: str) -> str | None:
    wl = wanted.lower()
    for k in d.keys():
        if k.lower() == wl:
            return k
    return None


def extract_from_mat(mat_path: Path) -> None:
    mat = loadmat(mat_path, squeeze_me=True, struct_as_record=False)

    key = _find_key_case_insensitive(mat, "Icolor")
    if key is None:
        # Mostra chaves úteis pra debug
        keys = [k for k in mat.keys() if not k.startswith("__")]
        raise RuntimeError(f"Variável 'Icolor' não encontrada. Chaves disponíveis: {keys}")

    ic = mat[key]  # cell array -> geralmente vira np.ndarray dtype=object (ou list) após squeeze
    # Normaliza para lista de elementos
    if isinstance(ic, np.ndarray) and ic.dtype == object:
        frames = list(ic.ravel())
    elif isinstance(ic, (list, tuple)):
        frames = list(ic)
    else:
        # Às vezes vem como um único objeto com indexação
        try:
            frames = list(ic)
        except Exception:
            raise RuntimeError(f"Icolor encontrado, mas em formato inesperado: {type(ic)}")

    if len(frames) < 5:
        raise RuntimeError(f"Icolor contém {len(frames)} frame(s), esperado >= 5.")

    m = CHIP_RE.match(mat_path.name)
    chip_id = m.group(1) if m else mat_path.stem

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for i in range(1, 6):
        img = np.asarray(frames[i - 1])
        out_name = f"{PREFIX}_{chip_id}_{i}.jpg"
        _save_jpg(img, OUTPUT_DIR / out_name)


def main() -> None:
    mats = sorted(INPUT_DIR.glob(FILE_GLOB))
    if not mats:
        raise SystemExit(f"Nenhum arquivo .mat encontrado em {INPUT_DIR}")

    total = 0
    for mat_file in mats:
        try:
            extract_from_mat(mat_file)
            total += 5
            print(f"[OK] {mat_file.name} -> 5 imagens extraídas (Icolor)")
        except Exception as e:
            print(f"[ERRO] {mat_file.name}: {e}")

    print(f"\nConcluído. Total extraído: {total} imagens")
    print(f"Saída: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
