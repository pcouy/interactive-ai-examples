#!/usr/bin/env python3
"""Exporte le point de contrôle PyTorch du CNN MNIST en manifeste JSON versionné.

Le manifeste produit (`demos/public/mnist/model-v1.json`) est consommé par
`demos/src/core/mnist-cnn-assets.ts`. Il ne contient que des poids et des
métadonnées : aucune dépendance DOM, aucun code exécutable.

Conventions de conversion (voir `demos/src/core/mnist-cnn.ts`) :
- noyaux de convolution PyTorch OIHW -> HWIO [kernelY][kernelX][entrée][sortie] ;
- noyaux denses PyTorch [sortie, entrée] -> [entrée, sortie] ;
- première couche dense : l'aplatissement PyTorch (`nn.Flatten`) est en ordre
  CHW, le core attend l'ordre NHWC (canal le plus rapide). On réordonne donc
  l'axe des caractéristiques d'entrée avant la transposée.

Le script n'importe jamais `train.py` : ce module instancie
`torchvision.datasets.MNIST(download=True)` au niveau module. Il recharge
seulement le `state_dict` avec `map_location='cpu', weights_only=True` et
reconstruit l'architecture localement pour la vérification de parité.

Usage :
    python export_mnist_cnn_checkpoint.py \
        --checkpoint /chemin/vers/model.pth \
        --output demos/public/mnist/model-v1.json

L'option `--parity` imprime les logits de référence PyTorch pour les entrées
synthétiques déterministes utilisées par le test de parité TypeScript.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch import nn

SCHEMA_VERSION = 1
FILTERS = (3, 3, 3, 3)
EXPECTED_PARAMETERS = 980

# Clés du state_dict PyTorch et formes attendues (OIHW pour les convolutions,
# [sortie, entrée] pour les denses).
EXPECTED_CONV_KEYS = {
    "conv_stack.0.weight": (3, 1, 3, 3),
    "conv_stack.0.bias": (3,),
    "conv_stack.2.weight": (3, 3, 3, 3),
    "conv_stack.2.bias": (3,),
    "conv_stack.4.weight": (3, 3, 3, 3),
    "conv_stack.4.bias": (3,),
    "conv_stack.6.weight": (3, 3, 3, 3),
    "conv_stack.6.bias": (3,),
    "linear.0.weight": (8, 75),
    "linear.0.bias": (8,),
    "linear.2.weight": (10, 8),
    "linear.2.bias": (10,),
}

# Nom de couche du manifeste -> clé PyTorch du noyau et du biais.
LAYER_KEYS = {
    "conv1": ("conv_stack.0.weight", "conv_stack.0.bias"),
    "conv2": ("conv_stack.2.weight", "conv_stack.2.bias"),
    "conv3": ("conv_stack.4.weight", "conv_stack.4.bias"),
    "conv4": ("conv_stack.6.weight", "conv_stack.6.bias"),
    "dense1": ("linear.0.weight", "linear.0.bias"),
    "dense2": ("linear.2.weight", "linear.2.bias"),
}

# Entrées synthétiques du test de parité : valeur entière exacte / 250, donc
# reproductible à l'identique en Python (float64 -> float32) et en TypeScript.
PARITY_IMAGE_COUNT = 3
PARITY_DENOMINATOR = 250
PARITY_MODULUS = 251
PARITY_SEED_STRIDE = 131
PARITY_PIXEL_STRIDE = 17


class NeuralNetwork(nn.Module):
    """Copie locale de l'architecture de `tf-playground/mnist/train.py`."""

    def __init__(self, kernel_size: int = 3) -> None:
        super().__init__()
        self.conv_stack = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size, stride=1),
            nn.Sigmoid(),
            nn.Conv2d(3, 3, kernel_size, stride=2, padding=1),
            nn.Sigmoid(),
            nn.Conv2d(3, 3, kernel_size, stride=1),
            nn.Sigmoid(),
            nn.Conv2d(3, 3, kernel_size, stride=2),
            nn.Sigmoid(),
            nn.Dropout(p=0.2),
        )
        self.flatten = nn.Flatten()
        self.linear = nn.Sequential(nn.Linear(75, 8), nn.Sigmoid(), nn.Linear(8, 10))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_stack(x)
        return self.linear(self.flatten(x))


def load_state_dict(checkpoint: Path) -> dict[str, torch.Tensor]:
    if not checkpoint.is_file():
        raise SystemExit(f"Point de contrôle introuvable : {checkpoint}")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise SystemExit("Le point de contrôle ne contient pas de state_dict (dict attendu).")
    return state


def validate_state_dict(state: dict[str, torch.Tensor]) -> None:
    missing = set(EXPECTED_CONV_KEYS) - set(state)
    if missing:
        raise SystemExit(f"Clés manquantes dans le state_dict : {sorted(missing)}")
    unexpected = set(state) - set(EXPECTED_CONV_KEYS)
    if unexpected:
        raise SystemExit(f"Clés inattendues dans le state_dict : {sorted(unexpected)}")
    total = 0
    for key, shape in EXPECTED_CONV_KEYS.items():
        tensor = state[key]
        if tuple(tensor.shape) != shape:
            raise SystemExit(f"Forme inattendue pour {key} : {tuple(tensor.shape)} != {shape}")
        if tensor.dtype != torch.float32:
            raise SystemExit(f"Type inattendu pour {key} : {tensor.dtype} != float32")
        if not bool(torch.isfinite(tensor).all()):
            raise SystemExit(f"Valeur non finie dans {key}")
        total += tensor.numel()
    if total != EXPECTED_PARAMETERS:
        raise SystemExit(f"Nombre de paramètres {total} != {EXPECTED_PARAMETERS}")


def conv_kernel_to_hwio(weight: torch.Tensor) -> list[float]:
    """PyTorch OIHW [sortie, entrée, ky, kx] -> HWIO [ky, kx, entrée, sortie]."""
    return weight.detach().permute(2, 3, 1, 0).reshape(-1).tolist()


def dense_kernel_to_io(weight: torch.Tensor) -> list[float]:
    """PyTorch [sortie, entrée] -> [entrée, sortie]."""
    return weight.detach().t().reshape(-1).tolist()


def dense1_kernel_to_nhwc_io(weight: torch.Tensor) -> list[float]:
    """Aplatissement PyTorch CHW -> ordre NHWC du core, puis [entrée, sortie]."""
    out_features, in_features = weight.shape
    channels, height, width = FILTERS[3], 5, 5
    if in_features != channels * height * width:
        raise SystemExit(
            f"Première couche dense inattendue : {in_features} != {channels * height * width}"
        )
    reordered = weight.detach().reshape(out_features, channels, height, width).permute(0, 2, 3, 1)
    return reordered.reshape(out_features, in_features).t().reshape(-1).tolist()


def bias_list(bias: torch.Tensor) -> list[float]:
    return bias.detach().reshape(-1).tolist()


def build_manifest(state: dict[str, torch.Tensor]) -> dict[str, object]:
    layers: dict[str, dict[str, list[float]]] = {}
    for name, (kernel_key, bias_key) in LAYER_KEYS.items():
        if name.startswith("conv"):
            kernel = conv_kernel_to_hwio(state[kernel_key])
        elif name == "dense1":
            kernel = dense1_kernel_to_nhwc_io(state[kernel_key])
        else:
            kernel = dense_kernel_to_io(state[kernel_key])
        layers[name] = {"kernel": kernel, "bias": bias_list(state[bias_key])}
    return {
        "schemaVersion": SCHEMA_VERSION,
        "architecture": {"filters": list(FILTERS)},
        "parameterCount": EXPECTED_PARAMETERS,
        "layers": layers,
    }


def parity_inputs() -> list[list[float]]:
    return [
        [
            ((image * PARITY_SEED_STRIDE + pixel * PARITY_PIXEL_STRIDE) % PARITY_MODULUS)
            / PARITY_DENOMINATOR
            for pixel in range(28 * 28)
        ]
        for image in range(PARITY_IMAGE_COUNT)
    ]


def parity_logits(state: dict[str, torch.Tensor]) -> list[list[float]]:
    model = NeuralNetwork(kernel_size=3)
    model.load_state_dict(state)
    model.eval()
    images = parity_inputs()
    tensor = torch.tensor(images, dtype=torch.float32).reshape(-1, 1, 28, 28)
    with torch.no_grad():
        logits = model(tensor)
    return logits.tolist()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=False)
    path.write_text(text + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    default_checkpoint = Path(
        "/home/pierre/code/histoire_IA/tf-playground/mnist/model.pth"
    )
    default_output = Path(__file__).resolve().parents[1] / "public" / "mnist" / "model-v1.json"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=default_checkpoint)
    parser.add_argument("--output", type=Path, default=default_output)
    parser.add_argument(
        "--parity",
        action="store_true",
        help="Imprime les entrées et logits de référence pour le test de parité.",
    )
    args = parser.parse_args(argv)

    state = load_state_dict(args.checkpoint)
    validate_state_dict(state)
    manifest = build_manifest(state)
    write_json(args.output, manifest)
    print(f"Manifeste écrit : {args.output}")

    if args.parity:
        payload = {"inputs": parity_inputs(), "logits": parity_logits(state)}
        print("--- PARITY ---")
        print(json.dumps(payload, ensure_ascii=False))
        print("--- /PARITY ---")

    return 0


if __name__ == "__main__":
    sys.exit(main())
