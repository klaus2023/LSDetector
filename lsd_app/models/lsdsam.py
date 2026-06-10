"""LSDSAM adapter skeleton with delayed heavy imports."""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
from importlib import import_module
import importlib.util
from pathlib import Path
import sys
import warnings
from typing import Any, Iterator, Sequence

from lsd_app.core.channels import DEFAULT_RGB_BANDS, IMAGENET_RGB_MEAN, IMAGENET_RGB_STD, extract_rgb_bands, normalize_rgb_bands
from lsd_app.core.paths import resolve_user_path
from lsd_app.models.base import ModelAdapter, ModelAdapterDependencyError, ModelAdapterError, missing_dependency_message


BUNDLED_LSDSAM_MODEL_ROOT = Path(__file__).resolve().parent / "model_repository" / "LSDSAM"


class LSDSAMFewShotAdapter(ModelAdapter):
    """Adapter for the legacy LSDSAM few-shot/SAM-LoRA inference stack."""

    name = "LSDSAM-B"

    def __init__(self) -> None:
        self.model: Any | None = None
        self.torch: Any | None = None
        self.device = "cpu"
        self.edge: str | None = "wavelets"
        self.image_size = 512
        self.threshold = 0.5
        self.RGB_bands = DEFAULT_RGB_BANDS

    def load(self, config: Any) -> "LSDSAMFewShotAdapter":
        checkpoint_path = getattr(config, "checkpoint_path", None)
        if checkpoint_path is None:
            raise ModelAdapterError(
                "Model adapter 'LSDSAM' requires model.checkpoint_path before it can load weights."
            )

        checkpoint = Path(checkpoint_path).expanduser().resolve()
        if not checkpoint.exists():
            raise ModelAdapterError(f"LSDSAM checkpoint does not exist: {checkpoint}")

        options = dict(getattr(config, "adapter_options", {}) or {})
        if "edge" in options and str(options["edge"]).strip().lower() != "wavelets":
            warnings.warn(
                "LSDSAM edge selection is fixed to 'wavelets'; ignoring model.options.edge.",
                RuntimeWarning,
                stacklevel=2,
            )
        self.edge = "wavelets"
        self.image_size = _positive_int(options.get("image_size", 512), "image_size")
        self.threshold = float(getattr(config, "threshold", 0.5))
        self.RGB_bands = normalize_rgb_bands(getattr(config, "RGB_bands", self.RGB_bands))
        weight_type = str(options.get("weight_type") or options.get("checkpoint_format") or "").strip().lower()

        torch = _require_dependency("torch", "Install the training/inference dependencies before using LSDSAM.")
        _require_dependency("cv2", "Install geospatial/image dependencies before using LSDSAM preprocessing.")
        _require_dependency("pywt", "Install PyWavelets before using LSDSAM wavelet edge features.")

        requested_device = str(getattr(config, "device", "cpu")).lower()
        self.device = _select_device(torch, requested_device)
        self.torch = torch

        legacy_root = _legacy_root(options, config)
        build_module = _import_legacy_build_module(legacy_root)
        build_sam = _resolve_build_sam_fn(build_module, options)
        if build_sam is None:
            raise ModelAdapterDependencyError(
                "Legacy LSDSAM build module does not expose the requested build_sam_* function."
            )

        checkpoint_params = _torch_load(torch, checkpoint, self.device)
        checkpoint_state = _extract_checkpoint_state_dict(checkpoint_params, checkpoint)
        base_checkpoint = _resolve_base_checkpoint_for_state(
            build_module,
            legacy_root,
            options,
            checkpoint_state,
            source=checkpoint,
        )
        if _is_merged_weight_type(weight_type, checkpoint_state):
            model = self._build_base_model(torch, build_sam, base_checkpoint)
            model.load_state_dict(checkpoint_state, strict=True)
            self.model = model.to(self.device)
            self.model.eval()
            return self

        peft = _require_dependency("peft", "Install PEFT before using unmerged LSDSAM LoRA checkpoints.")
        model = build_sam(checkpoint=False, map_location=self.device)
        if base_checkpoint is not None:
            _load_base_sam_checkpoint(torch, model, base_checkpoint, self.device)
        model.image_encoder.resize_pos_embed(new_img_size=self.image_size)
        model = model.to(self.device)

        trainable_param_names = [
            name for name, param in model.named_parameters() if getattr(param, "requires_grad", False)
        ]
        matched_modules = [
            name for name, module in model.named_modules()
            if name.startswith("image_encoder.blocks.") and (
                name.endswith("attn.qkv")
                or name.endswith("attn.proj")
                or name.endswith("mlp.lin1")
                or name.endswith("mlp.lin2")
            )
        ]

        adapter_name = _resolve_lora_adapter_name(
            checkpoint_state,
            preferred=str(options.get("peft_adapter_name") or ""),
        )
        task_rank = _infer_lora_rank(
            checkpoint_state,
            adapter_name=adapter_name,
            fallback=int(options.get("lora_task_r", 16)),
        )
        new_config = peft.LoraConfig(
            r=task_rank,
            lora_alpha=_resolve_lora_alpha(options, adapter_name=adapter_name, fallback=32),
            target_modules=matched_modules,
            lora_dropout=float(options.get("lora_dropout", 0.05)),
            bias="none",
        )
        model = _attach_lora_adapter(peft, model, adapter_name, new_config)

        model.load_state_dict(checkpoint_state)
        base_model = getattr(getattr(model, "base_model", None), "model", None)
        if base_model is not None:
            for name, param in base_model.named_parameters():
                if name in trainable_param_names:
                    param.requires_grad = True

        self.model = model.to(self.device)
        self.model.eval()
        return self

    def _build_base_model(self, torch: Any, build_sam: Any, base_checkpoint: Path | None) -> Any:
        model = build_sam(checkpoint=False, map_location=self.device)
        if base_checkpoint is not None:
            _load_base_sam_checkpoint(torch, model, base_checkpoint, self.device)
        model.image_encoder.resize_pos_embed(new_img_size=self.image_size)
        return model.to(self.device)

    def preprocess_patch(self, array: Any) -> Any:
        if self.torch is None:
            raise ModelAdapterError("LSDSAM adapter is not loaded.")

        import numpy as np

        cv2 = import_module("cv2")
        patch = np.asarray(array, dtype=np.float32)
        if patch.ndim != 3:
            raise ModelAdapterError(f"Expected patch array with shape (C, H, W), got {patch.shape}.")
        patch = extract_rgb_bands(patch, self.RGB_bands, layout="chw")
        patch = np.moveaxis(patch, 0, -1)
        if patch.shape[2] < 3:
            patch = np.repeat(patch[:, :, :1], 3, axis=2)

        image = patch[:, :, :3].astype(np.float32, copy=True)
        image = _ensure_unit_range(image)
        image = _resize_and_pad(image, self.image_size, cv2)
        edge = _build_edge_features(image, self.edge, cv2)
        image[:, :, :3] = _imagenet_normalize(image[:, :, :3])
        model_input = np.concatenate((image[:, :, :3], edge), axis=-1)
        model_input = np.moveaxis(model_input, -1, 0)
        return self.torch.as_tensor(model_input, dtype=self.torch.float32)

    def predict_batch(self, batch: Sequence[Any]) -> Sequence[Any]:
        if self.model is None or self.torch is None:
            raise ModelAdapterError("LSDSAM adapter is not loaded.")
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

        if self.torch is not None and hasattr(logits, "detach"):
            logits_array = logits.detach().cpu().numpy()
        else:
            logits_array = np.asarray(logits)
        if logits_array.ndim == 3 and logits_array.shape[0] >= 2:
            prediction = np.argmax(logits_array, axis=0)
        elif logits_array.ndim == 3 and logits_array.shape[0] == 1:
            prediction = logits_array[0] >= self.threshold
        elif logits_array.ndim == 2:
            prediction = logits_array >= self.threshold
        else:
            raise ModelAdapterError(f"Unsupported LSDSAM output shape: {logits_array.shape}")
        return prediction.astype(np.uint8) * 255


_PIXEL_MEAN: Any
_PIXEL_STD: Any


def _init_constants() -> None:
    global _PIXEL_MEAN, _PIXEL_STD
    import numpy as np

    _PIXEL_MEAN = np.array(IMAGENET_RGB_MEAN, dtype=np.float32)
    _PIXEL_STD = np.array(IMAGENET_RGB_STD, dtype=np.float32)


_init_constants()


def _require_dependency(module_name: str, install_hint: str) -> Any:
    try:
        return import_module(module_name)
    except ModuleNotFoundError as exc:
        raise ModelAdapterDependencyError(
            missing_dependency_message(LSDSAMFewShotAdapter.name, module_name, install_hint)
        ) from exc


def _select_device(torch: Any, requested_device: str) -> str:
    requested = str(requested_device or "auto").strip().lower()
    if requested in {"auto", "gpu"}:
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        warnings.warn(
            "CUDA was requested for LSDSAM but is not available. Falling back to CPU; inference will be very slow.",
            RuntimeWarning,
            stacklevel=2,
        )
        return "cpu"
    if requested == "cuda":
        return "cuda"
    return requested


def _legacy_root(options: dict[str, Any], config: Any) -> Path:
    if options.get("legacy_root"):
        return _resolve_legacy_root_path(str(options["legacy_root"]))
    candidates: list[Path] = []
    if getattr(config, "config_path", None) is not None:
        candidates.append(Path(config.config_path).resolve().parent / "LSDSAM_finetune")
    candidates.extend(
        [
            BUNDLED_LSDSAM_MODEL_ROOT,
        ]
    )
    for candidate in candidates:
        resolved = _resolve_legacy_root_path(candidate)
        if resolved.exists():
            return resolved
    return _resolve_legacy_root_path(candidates[-1])


def _resolve_legacy_root_path(value: str | Path) -> Path:
    return resolve_user_path(value)


def _import_legacy_build_module(legacy_root: Path) -> Any:
    if not legacy_root.exists():
        raise ModelAdapterDependencyError(
            f"Legacy LSDSAM directory does not exist: {legacy_root}. Set model.options.legacy_root."
        )
    if (legacy_root / "build_sam_diy.py").exists():
        module_key = hashlib.sha1(str(legacy_root).encode("utf-8")).hexdigest()[:12]
        module_name = f"_lsdapp_lsdsam_build_sam_diy_{module_key}"
        if module_name in sys.modules:
            return sys.modules[module_name]
        spec = importlib.util.spec_from_file_location(
            module_name,
            legacy_root / "build_sam_diy.py",
            submodule_search_locations=[str(legacy_root)],
        )
        if spec is None or spec.loader is None:
            raise ModelAdapterDependencyError(f"Could not load build_sam_diy from LSDSAM directory: {legacy_root}.")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        with _temporary_sys_path(legacy_root):
            try:
                spec.loader.exec_module(module)
            except ModuleNotFoundError as exc:
                raise ModelAdapterDependencyError(
                    f"Could not import build_sam_diy from LSDSAM directory: {legacy_root}."
                ) from exc
        return module
    with _temporary_sys_path(legacy_root.parent):
        try:
            return import_module(
                "LSDSAM_finetune.SAM_edge_4_after512_crossfusion_multihead_LoRA.build_sam_diy"
            )
        except ModuleNotFoundError as exc:
            raise ModelAdapterDependencyError(
                "Could not import legacy LSDSAM build module. Keep LSDSAM_finetune available "
                "or set model.options.legacy_root to the legacy directory."
            ) from exc


def _resolve_sam_variant(options: dict[str, Any]) -> str:
    value = str(options.get("sam_variant") or options.get("variant") or "B").strip().upper()
    return value if value in {"H", "L", "B"} else "B"


def _resolve_build_sam_fn(build_module: Any, options: dict[str, Any]) -> Any:
    variant = _resolve_sam_variant(options).lower()
    return getattr(build_module, f"build_sam_{variant}", None)


@contextmanager
def _temporary_sys_path(path: Path) -> Iterator[None]:
    path_text = str(path)
    inserted = path_text not in sys.path
    if inserted:
        sys.path.insert(0, path_text)
    try:
        yield
    finally:
        if inserted:
            try:
                sys.path.remove(path_text)
            except ValueError:
                pass


def _resolve_base_checkpoint(build_module: Any, legacy_root: Path, options: dict[str, Any]) -> Path:
    if options.get("sam_checkpoint_path"):
        return Path(str(options["sam_checkpoint_path"])).expanduser().resolve()
    raise ModelAdapterError(
        "LSDSAM no longer uses bundled or legacy SAM initialization checkpoints. "
        "Use a registered full/merged base_weight, or set base_checkpoint_path/sam_checkpoint_path "
        "in the weight metadata when loading a pure LoRA delta."
    )


def _resolve_base_checkpoint_for_state(
    build_module: Any,
    legacy_root: Path,
    options: dict[str, Any],
    state_dict: dict[str, Any],
    *,
    source: Path,
) -> Path | None:
    if _state_dict_contains_base_layers(state_dict):
        return None
    if not options.get("sam_checkpoint_path"):
        raise ModelAdapterError(
            "LSDSAM weight contains only adapter/delta parameters, so a registered base SAM checkpoint is required. "
            "Register a full/merged base_weight, or set base_checkpoint_path/sam_checkpoint_path in the weight metadata. "
            f"Weight: {source}"
        )
    base_checkpoint = _resolve_base_checkpoint(build_module, legacy_root, options)
    if not base_checkpoint.exists():
        raise ModelAdapterError(f"LSDSAM base SAM checkpoint does not exist: {base_checkpoint}")
    return base_checkpoint


def _state_dict_contains_base_layers(state_dict: dict[str, Any]) -> bool:
    for key in state_dict:
        if ".base_layer." in key:
            return True
        if key.startswith("base_model.model.") and ".lora_" not in key:
            return True
        if key.startswith(("image_encoder.", "mask_decoder.", "prompt_encoder.", "seg_head.")) and ".lora_" not in key:
            return True
    return False


def _load_base_sam_checkpoint(torch: Any, model: Any, checkpoint_path: Path, device: str) -> None:
    checkpoint = _torch_load(torch, checkpoint_path, device)
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    if not isinstance(state_dict, dict):
        raise ModelAdapterError(f"Invalid SAM base checkpoint format: {checkpoint_path}")
    if "mask_decoder.mask_tokens.weight" in state_dict:
        mask_tokens_weight = state_dict["mask_decoder.mask_tokens.weight"]
        if mask_tokens_weight.size(0) > 2:
            state_dict["mask_decoder.mask_tokens.weight"] = mask_tokens_weight[:2, :]
    model.load_state_dict(state_dict, strict=False)


def _torch_load(torch: Any, path: Path, device: str) -> Any:
    map_location = _checkpoint_map_location(device)
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _checkpoint_map_location(device: str) -> str:
    requested = str(device or "cpu").strip().lower()
    return "cpu" if requested.startswith("cuda") else requested or "cpu"


def _extract_checkpoint_state_dict(checkpoint_params: Any, checkpoint_path: Path) -> dict[str, Any]:
    if isinstance(checkpoint_params, dict) and isinstance(checkpoint_params.get("model_state_dict"), dict):
        return checkpoint_params["model_state_dict"]
    if isinstance(checkpoint_params, dict) and _looks_like_state_dict(checkpoint_params):
        return checkpoint_params
    raise ModelAdapterError(
        "LSDSAM checkpoint must be a state_dict or contain a 'model_state_dict' entry: "
        f"{checkpoint_path}"
    )


def _looks_like_state_dict(value: dict[str, Any]) -> bool:
    return bool(value) and all(isinstance(key, str) for key in value)


def _is_merged_weight_type(weight_type: str, state_dict: dict[str, Any]) -> bool:
    if weight_type in {"merged", "merged_checkpoint", "inference-ready", "inference_ready", "full", "full_checkpoint"}:
        return True
    return bool(state_dict) and not any(".lora_A." in key or ".lora_B." in key for key in state_dict)


def _resolve_lora_adapter_name(state_dict: dict[str, Any], *, preferred: str = "") -> str:
    names = _lora_adapter_names(state_dict)
    preferred = preferred.strip()
    if preferred and preferred in names:
        return preferred
    if "default" in names:
        return "default"
    if preferred:
        return preferred
    if len(names) == 1:
        return next(iter(names))
    return "default"


def _lora_adapter_names(state_dict: dict[str, Any]) -> set[str]:
    names: set[str] = set()
    for key in state_dict:
        parts = key.split(".")
        for index, part in enumerate(parts[:-1]):
            if part in {"lora_A", "lora_B"}:
                names.add(parts[index + 1])
    return names


def _infer_lora_rank(state_dict: dict[str, Any], *, adapter_name: str, fallback: int) -> int:
    suffix = f".lora_A.{adapter_name}.weight"
    for key, value in state_dict.items():
        if key.endswith(suffix) and hasattr(value, "shape") and len(value.shape) >= 1:
            return int(value.shape[0])
    return fallback


def _resolve_lora_alpha(options: dict[str, Any], *, adapter_name: str, fallback: int) -> int:
    if options.get("lora_alpha") not in {None, ""}:
        return int(options["lora_alpha"])
    if adapter_name == "default" and options.get("lora_base_alpha") not in {None, ""}:
        return int(options["lora_base_alpha"])
    if options.get("lora_task_alpha") not in {None, ""}:
        return int(options["lora_task_alpha"])
    if options.get("lora_base_alpha") not in {None, ""}:
        return int(options["lora_base_alpha"])
    return int(fallback)


def _attach_lora_adapter(peft: Any, model: Any, adapter_name: str, lora_config: Any) -> Any:
    if hasattr(model, "add_adapter"):
        model.add_adapter(adapter_name, lora_config)
        if hasattr(model, "set_adapter"):
            model.set_adapter(adapter_name)
        return model
    wrapped = peft.get_peft_model(model, lora_config, adapter_name=adapter_name)
    if hasattr(wrapped, "set_adapter"):
        wrapped.set_adapter(adapter_name)
    return wrapped


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


def _build_edge_features(image: Any, edge: str | None, cv2: Any) -> Any:
    import numpy as np

    if edge is None:
        return np.zeros((image.shape[0], image.shape[1], 3), dtype=np.float32)
    if edge == "sobel":
        return _ensure_three_channels(_get_sobel(image, cv2))
    if edge == "wavelets":
        return _ensure_three_channels(_get_wavelets(image, cv2))
    if edge == "fft":
        return _ensure_three_channels(_get_fft(image, cv2))
    raise ModelAdapterError(f"Invalid LSDSAM edge mode: {edge}")


def _imagenet_normalize(image: Any) -> Any:
    """Normalize an RGB float image with ImageNet mean/std."""

    import numpy as np

    values = np.asarray(image, dtype=np.float32)
    return (values - _PIXEL_MEAN) / _PIXEL_STD


def _ensure_unit_range(image: Any) -> Any:
    """Scale common uint-style image ranges to 0..1 before ImageNet normalization."""

    import numpy as np

    values = np.asarray(image, dtype=np.float32)
    finite = values[np.isfinite(values)]
    if finite.size and float(finite.max()) > 1.0:
        values = values / 255.0
    return np.clip(values, 0.0, 1.0).astype(np.float32, copy=False)


def _ensure_three_channels(edge: Any) -> Any:
    import numpy as np

    if edge.ndim == 2:
        edge = edge[..., np.newaxis]
    if edge.shape[2] == 3:
        return edge.astype(np.float32, copy=False)
    if edge.shape[2] == 1:
        return np.repeat(edge, 3, axis=2).astype(np.float32, copy=False)
    return edge[:, :, :3].astype(np.float32, copy=False)


def _get_sobel(image: Any, cv2: Any) -> Any:
    import numpy as np

    image_uint8 = (np.clip(image, 0, 1) * 255).astype(np.uint8)
    gray = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2GRAY)
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(grad_x, grad_y)
    magnitude_norm = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min() + 1e-8)
    return magnitude_norm.astype(np.float32)[..., np.newaxis]


def _get_wavelets(image: Any, cv2: Any, wavelet_type: str = "bior1.3") -> Any:
    import numpy as np

    pywt = import_module("pywt")
    image_uint8 = (np.clip(image, 0, 1) * 255).astype(np.uint8)
    gray = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2GRAY).astype(np.float32)
    _, (lh, hl, hh) = pywt.dwt2(gray, wavelet_type)
    high_freq = np.stack([lh, hl, hh], axis=-1).astype(np.float32)
    input_h, input_w = image.shape[:2]
    if high_freq.shape[:2] != (input_h, input_w):
        high_freq = cv2.resize(high_freq, (input_w, input_h), interpolation=cv2.INTER_LINEAR)
    for i in range(3):
        channel = high_freq[..., i]
        min_value = channel.min()
        max_value = channel.max()
        if max_value > min_value:
            high_freq[..., i] = (channel - min_value) / (max_value - min_value)
        else:
            high_freq[..., i] = 0.0
    return high_freq


def _get_fft(image: Any, cv2: Any, radius: int = 20) -> Any:
    import numpy as np

    image_uint8 = (np.clip(image, 0, 1) * 255).astype(np.uint8)
    gray = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2GRAY).astype(np.float32)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    height, width = gray.shape
    mask = np.ones((height, width), dtype=np.uint8)
    center_y, center_x = height // 2, width // 2
    mask[center_y - radius:center_y + radius, center_x - radius:center_x + radius] = 0
    high_freq = np.abs(np.fft.ifft2(np.fft.ifftshift(fshift * mask))).astype(np.float32)
    high_freq = high_freq / (high_freq.max() + 1e-8)
    return high_freq[..., np.newaxis]


def _validate_choice(value: str, allowed: set[str], field_name: str) -> str:
    if value not in allowed:
        choices = ", ".join(sorted(allowed))
        raise ModelAdapterError(f"Invalid LSDSAM {field_name}: {value!r}. Expected one of: {choices}.")
    return value


def _positive_int(value: Any, field_name: str) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ModelAdapterError(f"Invalid LSDSAM {field_name}: expected positive integer, got {value!r}.") from exc
    if number <= 0:
        raise ModelAdapterError(f"Invalid LSDSAM {field_name}: expected positive integer, got {number}.")
    return number
