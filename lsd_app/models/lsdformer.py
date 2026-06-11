"""LSDFormer adapter and hybrid fine-tuning helpers."""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import importlib.util
from importlib import import_module
from pathlib import Path
import sys
from typing import Any, Iterator, Sequence

from lsd_app.core.channels import DEFAULT_RGB_BANDS, IMAGENET_RGB_MEAN, IMAGENET_RGB_STD, extract_rgb_bands, normalize_rgb_bands
from lsd_app.core.paths import resolve_user_path
from lsd_app.models.base import ModelAdapter, ModelAdapterDependencyError, ModelAdapterError, missing_dependency_message


BUNDLED_LSDFORMER_MODEL_ROOT = Path(__file__).resolve().parent / "model_repository" / "LSDFormer"
DEFAULT_LSDFORMER_MODEL_ROOT = BUNDLED_LSDFORMER_MODEL_ROOT
LSDFORMER_TARGET_MODULES = "encoder attention/ffn LoRA + MB3 1x1 Conv-LoRA"


class LSDFormerFastAdapter(ModelAdapter):
    """Adapter for the external LSDFormer segmentation model."""

    name = "LSDFormer"

    def __init__(self) -> None:
        self.model: Any | None = None
        self.torch: Any | None = None
        self.device = "cpu"
        self.image_size = 512
        self.threshold = 0.5
        self.RGB_bands = DEFAULT_RGB_BANDS

    def load(self, config: Any) -> "LSDFormerFastAdapter":
        checkpoint_path = getattr(config, "checkpoint_path", None)
        if checkpoint_path is None:
            raise ModelAdapterError("Model adapter 'LSDFormer' requires model.checkpoint_path before loading.")
        checkpoint = Path(checkpoint_path).expanduser().resolve()
        if not checkpoint.exists():
            raise ModelAdapterError(f"LSDFormer checkpoint does not exist: {checkpoint}")

        options = dict(getattr(config, "adapter_options", {}) or {})
        self.image_size = _positive_int(options.get("image_size", 512), "image_size")
        self.threshold = float(getattr(config, "threshold", 0.5))
        self.RGB_bands = normalize_rgb_bands(getattr(config, "RGB_bands", self.RGB_bands))

        torch = _require_dependency("torch", "Install the training/inference dependencies before using LSDFormer.")
        _require_dependency("cv2", "Install opencv-python before using LSDFormer preprocessing.")
        self.device = _select_device(torch, str(getattr(config, "device", "cpu")))
        self.torch = torch

        model_root = _resolve_model_root(options, config)
        model = build_lsdformer_model(
            torch,
            checkpoint,
            model_root=model_root,
            options=options,
            map_location="cpu",
        )
        self.model = model.to(self.device)
        self.model.eval()
        return self

    def preprocess_patch(self, array: Any) -> Any:
        if self.torch is None:
            raise ModelAdapterError("LSDFormer adapter is not loaded.")
        cv2 = import_module("cv2")
        import numpy as np

        patch = np.asarray(array, dtype=np.float32)
        if patch.ndim != 3:
            raise ModelAdapterError(f"Expected patch array with shape (C, H, W), got {patch.shape}.")
        patch = extract_rgb_bands(patch, self.RGB_bands, layout="chw")
        patch = np.moveaxis(patch, 0, -1)
        if patch.shape[2] < 3:
            patch = np.repeat(patch[:, :, :1], 3, axis=2)
        image = _ensure_unit_range(patch[:, :, :3])
        image = _resize_and_pad(image, self.image_size, cv2)
        image = _imagenet_normalize(image)
        return self.torch.as_tensor(np.moveaxis(image, -1, 0).astype("float32", copy=False), dtype=self.torch.float32)

    def predict_batch(self, batch: Sequence[Any]) -> Sequence[Any]:
        if self.model is None or self.torch is None:
            raise ModelAdapterError("LSDFormer adapter is not loaded.")
        if not batch:
            return []
        with self.torch.no_grad():
            inputs = self.torch.stack(list(batch)).to(self.device).float()
            outputs = self.model(inputs)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            return [item.detach().cpu() for item in outputs]

    def postprocess_logits(self, logits: Any) -> Any:
        import numpy as np

        logits_array = logits.detach().cpu().numpy() if hasattr(logits, "detach") else np.asarray(logits)
        if logits_array.ndim == 3 and logits_array.shape[0] >= 2:
            prediction = np.argmax(logits_array, axis=0)
        elif logits_array.ndim == 3 and logits_array.shape[0] == 1:
            prediction = logits_array[0] >= self.threshold
        elif logits_array.ndim == 2:
            prediction = logits_array >= self.threshold
        else:
            raise ModelAdapterError(f"Unsupported LSDFormer output shape: {logits_array.shape}")
        return prediction.astype(np.uint8) * 255


def _require_dependency(module_name: str, install_hint: str) -> Any:
    try:
        return import_module(module_name)
    except ModuleNotFoundError as exc:
        raise ModelAdapterDependencyError(
            missing_dependency_message(LSDFormerFastAdapter.name, module_name, install_hint)
        ) from exc


def _select_device(torch: Any, requested_device: str) -> str:
    requested = str(requested_device or "auto").strip().lower()
    if requested in {"auto", "gpu"}:
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return requested or "cpu"


def _resolve_model_root(options: dict[str, Any], config: Any | None = None) -> Path:
    if options.get("model_root"):
        return resolve_user_path(str(options["model_root"]))
    candidates: list[Path] = []
    if config is not None and getattr(config, "config_path", None) is not None:
        candidates.append(Path(config.config_path).resolve().parent / "LSDFormer_scripts")
    candidates.extend(
        [
            BUNDLED_LSDFORMER_MODEL_ROOT,
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return DEFAULT_LSDFORMER_MODEL_ROOT.resolve()


@contextmanager
def _temporary_sys_path(path: Path) -> Iterator[None]:
    text = str(path)
    inserted = text not in sys.path
    if inserted:
        sys.path.insert(0, text)
    try:
        yield
    finally:
        if inserted:
            try:
                sys.path.remove(text)
            except ValueError:
                pass


def _import_lsdformer_class(model_root: Path) -> Any:
    resolved = _resolve_lsdformer_import_paths(model_root)
    if len(resolved) == 3:
        _, _, module = resolved
        lsdformer = getattr(module, "LSDFormer", None)
        if lsdformer is None:
            raise ModelAdapterDependencyError("LSDFormer model package does not expose LSDFormer.")
        return lsdformer
    import_root, package_dir = resolved
    package_init = package_dir / "__init__.py"
    module_key = hashlib.sha1(str(package_dir).encode("utf-8")).hexdigest()[:12]
    module_name = f"_lsdapp_lsdformer_model_{module_key}"
    if module_name in sys.modules:
        module = sys.modules[module_name]
    else:
        spec = importlib.util.spec_from_file_location(
            module_name,
            package_init,
            submodule_search_locations=[str(package_dir)],
        )
        if spec is None or spec.loader is None:
            raise ModelAdapterDependencyError(f"Could not load LSDFormer model package from: {package_dir}.")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        with _temporary_sys_path(import_root):
            try:
                spec.loader.exec_module(module)
            except ModuleNotFoundError as exc:
                raise ModelAdapterDependencyError(
                    f"Could not import LSDFormer model package from: {package_dir}."
                ) from exc
    lsdformer = getattr(module, "LSDFormer", None)
    if lsdformer is None:
        raise ModelAdapterDependencyError("LSDFormer model package does not expose LSDFormer.")
    return lsdformer


def _resolve_lsdformer_import_paths(model_root: Path) -> tuple[Path, Path] | tuple[None, None, Any]:
    if not model_root.exists():
        raise ModelAdapterDependencyError(f"LSDFormer model root does not exist: {model_root}. Set model.options.model_root.")
    if (model_root / "model" / "__init__.py").exists():
        return model_root.resolve(), (model_root / "model").resolve()
    if (model_root / "__init__.py").exists() and (model_root / "model.py").exists():
        return model_root.parent.resolve(), model_root.resolve()
    with _temporary_sys_path(model_root):
        try:
            module = import_module("model")
        except ModuleNotFoundError as exc:
            raise ModelAdapterDependencyError(
                f"Could not import LSDFormer model package from: {model_root}."
            ) from exc
    package_file_value = getattr(module, "__file__", None)
    if not package_file_value:
        return None, None, module
    package_file = Path(package_file_value).resolve()
    if not package_file.exists():
        return None, None, module
    package_dir = package_file.parent
    return package_dir.parent, package_dir


def build_lsdformer_model(
    torch: Any,
    checkpoint_path: str | Path,
    *,
    model_root: str | Path | None = None,
    options: dict[str, Any] | None = None,
    map_location: str = "cpu",
) -> Any:
    """Instantiate external LSDFormer and load a checkpoint."""

    opts = dict(options or {})
    root = Path(model_root).expanduser().resolve() if model_root not in {None, ""} else _resolve_model_root(opts)
    model_cls = _import_lsdformer_class(root)
    model_kwargs = dict(opts.get("model_kwargs") or {})
    model_kwargs.setdefault("num_classes", int(opts.get("num_classes", 2)))
    model_kwargs.setdefault("in_chans", int(opts.get("in_chans", 3)))
    model = model_cls(**model_kwargs)

    checkpoint = _torch_load(torch, Path(checkpoint_path).expanduser().resolve(), map_location)
    state_dict = _extract_lsdformer_state_dict(checkpoint)
    load_result = model.load_state_dict(state_dict, strict=False)
    missing = getattr(load_result, "missing_keys", [])
    unexpected = getattr(load_result, "unexpected_keys", [])
    if not state_dict:
        raise ModelAdapterError("LSDFormer checkpoint did not contain model weights.")
    if unexpected and len(unexpected) == len(state_dict) and missing:
        raise ModelAdapterError("LSDFormer checkpoint did not match the model architecture.")
    return model


def _torch_load(torch: Any, path: Path, map_location: str) -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _extract_lsdformer_state_dict(payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict):
        for key in ("model_state_dict", "net", "state_dict"):
            value = payload.get(key)
            if isinstance(value, dict):
                return _strip_state_prefixes(value)
        if payload and all(isinstance(key, str) for key in payload):
            return _strip_state_prefixes(payload)
    raise ModelAdapterError("LSDFormer checkpoint must be a state_dict or contain model_state_dict/net/state_dict.")


def _strip_state_prefixes(state_dict: dict[str, Any]) -> dict[str, Any]:
    state = dict(state_dict)
    for prefix in ("module.", "model."):
        if state and all(key.startswith(prefix) for key in state):
            state = {key[len(prefix):]: value for key, value in state.items()}
    return state


class LSDFormerFeatureWrapper:
    """Thin nn.Module wrapper that exposes encoder final features without editing external code."""

    def __new__(cls, torch: Any, model: Any) -> Any:
        nn = torch.nn

        class _Wrapper(nn.Module):
            def __init__(self, inner: Any) -> None:
                super().__init__()
                self.model = inner

            def forward(self, x: Any, return_feats: bool = False) -> Any:
                output_size = x.shape[-2:]
                features = self.model.encoder(x)
                logits = self.model.decoder(features, output_size)
                if return_feats:
                    return logits, features[-1]
                return logits

            def train(self, mode: bool = True) -> Any:
                super().train(mode)
                if mode:
                    set_lsdformer_frozen_encoder_norms_eval(self.model)
                return self

        return _Wrapper(model)


class LinearLoRA:
    """Factory for a Linear layer with a frozen base path and trainable low-rank branch."""

    def __new__(cls, torch: Any, base_layer: Any, *, r: int, alpha: int, dropout: float) -> Any:
        nn = torch.nn

        class _LinearLoRA(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.base_layer = base_layer
                for param in self.base_layer.parameters():
                    param.requires_grad = False
                self.r = int(r)
                self.lora_alpha = int(alpha)
                self.scaling = float(alpha) / max(1, int(r))
                self.lora_dropout = nn.Dropout(float(dropout))
                self.lora_A = nn.Linear(base_layer.in_features, int(r), bias=False)
                self.lora_B = nn.Linear(int(r), base_layer.out_features, bias=False)
                nn.init.kaiming_uniform_(self.lora_A.weight, a=5**0.5)
                nn.init.zeros_(self.lora_B.weight)

            def forward(self, x: Any) -> Any:
                return self.base_layer(x) + self.lora_B(self.lora_A(self.lora_dropout(x))) * self.scaling

            def merged(self) -> Any:
                merged = nn.Linear(
                    self.base_layer.in_features,
                    self.base_layer.out_features,
                    bias=self.base_layer.bias is not None,
                )
                merged.to(self.base_layer.weight.device)
                merged.weight.data.copy_(self.base_layer.weight.data)
                if self.base_layer.bias is not None:
                    merged.bias.data.copy_(self.base_layer.bias.data)
                delta = self.lora_B.weight.data @ self.lora_A.weight.data
                merged.weight.data.add_(delta * self.scaling)
                return merged

        return _LinearLoRA()


class Conv1x1LoRA:
    """Factory for a 1x1 Conv2d with a frozen base path and trainable low-rank branch."""

    def __new__(cls, torch: Any, base_layer: Any, *, r: int, alpha: int, dropout: float) -> Any:
        nn = torch.nn

        class _Conv1x1LoRA(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                if tuple(base_layer.kernel_size) != (1, 1):
                    raise ModelAdapterError("Conv1x1LoRA only supports 1x1 Conv2d layers.")
                self.base_layer = base_layer
                for param in self.base_layer.parameters():
                    param.requires_grad = False
                self.r = int(r)
                self.lora_alpha = int(alpha)
                self.scaling = float(alpha) / max(1, int(r))
                self.lora_dropout = nn.Dropout2d(float(dropout))
                self.lora_A = nn.Conv2d(
                    base_layer.in_channels,
                    int(r),
                    kernel_size=1,
                    stride=base_layer.stride,
                    padding=base_layer.padding,
                    dilation=base_layer.dilation,
                    bias=False,
                )
                self.lora_B = nn.Conv2d(int(r), base_layer.out_channels, kernel_size=1, bias=False)
                nn.init.kaiming_uniform_(self.lora_A.weight, a=5**0.5)
                nn.init.zeros_(self.lora_B.weight)

            def forward(self, x: Any) -> Any:
                return self.base_layer(x) + self.lora_B(self.lora_A(self.lora_dropout(x))) * self.scaling

            def merged(self) -> Any:
                merged = nn.Conv2d(
                    self.base_layer.in_channels,
                    self.base_layer.out_channels,
                    kernel_size=1,
                    stride=self.base_layer.stride,
                    padding=self.base_layer.padding,
                    dilation=self.base_layer.dilation,
                    groups=self.base_layer.groups,
                    bias=self.base_layer.bias is not None,
                )
                merged.to(self.base_layer.weight.device)
                merged.weight.data.copy_(self.base_layer.weight.data)
                if self.base_layer.bias is not None:
                    merged.bias.data.copy_(self.base_layer.bias.data)
                a = self.lora_A.weight.data.reshape(self.r, self.base_layer.in_channels)
                b = self.lora_B.weight.data.reshape(self.base_layer.out_channels, self.r)
                delta = (b @ a).reshape_as(merged.weight.data)
                merged.weight.data.add_(delta * self.scaling)
                return merged

        return _Conv1x1LoRA()


def apply_lsdformer_hybrid_adapters(
    torch: Any,
    model: Any,
    *,
    linear_r: int = 16,
    linear_alpha: int = 32,
    conv_r: int = 8,
    conv_alpha: int = 16,
    dropout: float = 0.05,
) -> dict[str, int]:
    """Freeze LSDFormer encoder base weights, inject adapters, and unfreeze decoder."""

    for param in model.parameters():
        param.requires_grad = False

    counts = {"linear_lora": 0, "conv_lora": 0, "decoder_params": 0}
    for stage in getattr(getattr(model, "encoder", None), "stages", []):
        for block_group_name in ("efficient_ct_blocks", "enhanced_ct_blocks"):
            for block in getattr(stage, block_group_name, []):
                emsa = getattr(block, "emsa", None)
                if emsa is not None:
                    for attr in ("query", "key_value", "proj"):
                        layer = getattr(emsa, attr, None)
                        if _is_linear(torch, layer):
                            setattr(emsa, attr, LinearLoRA(torch, layer, r=linear_r, alpha=linear_alpha, dropout=dropout))
                            counts["linear_lora"] += 1
                ffn = getattr(block, "ffn", None)
                if ffn is not None:
                    for attr in ("fc1", "fc2"):
                        layer = getattr(ffn, attr, None)
                        if _is_linear(torch, layer):
                            setattr(ffn, attr, LinearLoRA(torch, layer, r=linear_r, alpha=linear_alpha, dropout=dropout))
                            counts["linear_lora"] += 1
                mb3 = getattr(block, "mb3_branch", None)
                if mb3 is not None:
                    for attr in ("expand_conv", "project_conv"):
                        layer = getattr(mb3, attr, None)
                        if _is_conv1x1(torch, layer):
                            setattr(mb3, attr, Conv1x1LoRA(torch, layer, r=conv_r, alpha=conv_alpha, dropout=dropout))
                            counts["conv_lora"] += 1

    decoder = getattr(model, "decoder", None)
    if decoder is not None:
        for param in decoder.parameters():
            param.requires_grad = True
            counts["decoder_params"] += int(param.numel())
    set_lsdformer_frozen_encoder_norms_eval(model)
    return counts


def _is_linear(torch: Any, layer: Any) -> bool:
    return isinstance(layer, torch.nn.Linear)


def _is_conv1x1(torch: Any, layer: Any) -> bool:
    return isinstance(layer, torch.nn.Conv2d) and tuple(layer.kernel_size) == (1, 1)


def set_lsdformer_frozen_encoder_norms_eval(model: Any) -> None:
    encoder = getattr(model, "encoder", None)
    if encoder is None:
        return
    for module in encoder.modules():
        if module.__class__.__name__ in {"BatchNorm2d", "SyncBatchNorm"}:
            module.eval()


def merge_lsdformer_adapters(model: Any) -> Any:
    """Merge in-place Linear/Conv LoRA wrappers into plain LSDFormer layers."""

    _merge_children(model)
    return model


def _merge_children(module: Any) -> None:
    for name, child in list(module.named_children()):
        if hasattr(child, "merged"):
            setattr(module, name, child.merged())
        else:
            _merge_children(child)


def infer_lsdformer_feature_channels(model_or_wrapper: Any) -> int:
    model = getattr(model_or_wrapper, "model", model_or_wrapper)
    decoder = getattr(model, "decoder", None)
    head6 = getattr(decoder, "head6", None)
    channels = getattr(head6, "in_channels", None)
    return int(channels or 512)


def trainable_lsdformer_param_groups(model_or_wrapper: Any, *, base_lr: float, conv_lr_scale: float) -> list[dict[str, Any]]:
    main_params = []
    conv_params = []
    for name, param in model_or_wrapper.named_parameters():
        if not getattr(param, "requires_grad", False):
            continue
        if "mb3_branch" in name and ("expand_conv.lora_" in name or "project_conv.lora_" in name):
            conv_params.append(param)
        else:
            main_params.append(param)
    groups: list[dict[str, Any]] = []
    if main_params:
        groups.append({"params": main_params, "lr": float(base_lr), "lr_scale": 1.0})
    if conv_params:
        groups.append({"params": conv_params, "lr": float(base_lr) * float(conv_lr_scale), "lr_scale": float(conv_lr_scale)})
    return groups


def set_optimizer_group_lrs(optimizer: Any, base_lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = float(base_lr) * float(group.get("lr_scale", 1.0))


def _ensure_unit_range(image: Any) -> Any:
    import numpy as np

    values = np.asarray(image, dtype=np.float32)
    finite = values[np.isfinite(values)]
    if finite.size and float(finite.max()) > 1.0:
        values = values / 255.0
    return np.clip(values, 0.0, 1.0).astype(np.float32, copy=False)


def _resize_and_pad(image: Any, image_size: int, cv2: Any) -> Any:
    import numpy as np

    height, width = image.shape[:2]
    scale = image_size / max(height, width)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    if (new_height, new_width) != (height, width):
        image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    padded = np.zeros((image_size, image_size, image.shape[2]), dtype=np.float32)
    padded[:new_height, :new_width, :] = image
    return padded


def _imagenet_normalize(image: Any) -> Any:
    import numpy as np

    mean = np.array(IMAGENET_RGB_MEAN, dtype=np.float32)
    std = np.array(IMAGENET_RGB_STD, dtype=np.float32)
    return (np.asarray(image, dtype=np.float32) - mean) / std


def _positive_int(value: Any, field_name: str) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ModelAdapterError(f"Invalid LSDFormer {field_name}: expected positive integer, got {value!r}.") from exc
    if number <= 0:
        raise ModelAdapterError(f"Invalid LSDFormer {field_name}: expected positive integer, got {number}.")
    return number
