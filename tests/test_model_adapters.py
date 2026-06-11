from types import SimpleNamespace

import pytest

from lsd_app.models import ModelAdapterDependencyError, ModelAdapterError, available_model_names, get_adapter


def test_lsdsam_default_legacy_root_uses_bundled_repository(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    from lsd_app.models import lsdsam as lsdsam_module

    app_dir = tmp_path / "LSD__APP"
    legacy_root = tmp_path / "LSDSAM_finetune"
    app_dir.mkdir()
    legacy_root.mkdir()
    monkeypatch.chdir(app_dir)

    resolved = lsdsam_module._legacy_root({}, SimpleNamespace(config_path=None))

    assert resolved == lsdsam_module.BUNDLED_LSDSAM_MODEL_ROOT.resolve()


def test_lsdsam_explicit_legacy_root_can_use_external_repository(tmp_path) -> None:
    from lsd_app.models import lsdsam as lsdsam_module

    legacy_root = tmp_path / "LSDSAM_finetune"
    legacy_root.mkdir()

    resolved = lsdsam_module._legacy_root({"legacy_root": str(legacy_root)}, SimpleNamespace(config_path=None))

    assert resolved == legacy_root.resolve()


def test_registry_exposes_required_adapters() -> None:
    names = available_model_names()

    assert "mock" in names
    assert "lsdsam-fewshot" in names
    assert "lsdformer-fast" in names
    assert "LSDFormer" in names
    assert "LSDSAM-H" in names
    assert "LSDSAM-L" in names
    assert "LSDSAM-B" in names
    assert get_adapter("lsdsam").name == "LSDSAM-B"
    assert get_adapter("lsdformer").name == "LSDFormer"


def test_mock_adapter_does_not_import_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    import builtins
    import numpy as np

    real_import = builtins.__import__

    def guarded_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "torch" or name.startswith("torch."):
            raise AssertionError("mock adapter should not import torch")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    adapter = get_adapter("mock").load(SimpleNamespace(threshold=0.5))
    patch = np.ones((3, 4, 4), dtype=np.float32)
    logits = adapter.predict_batch([adapter.preprocess_patch(patch)])
    prediction = adapter.postprocess_logits(logits[0])

    assert prediction.shape == (4, 4)
    assert prediction.dtype == np.uint8
    assert prediction.max() == 255


def test_lsdsam_adapter_delays_dependency_errors_until_load(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "checkpoint.pth"
    checkpoint.write_bytes(b"placeholder")
    from lsd_app.models import lsdsam as lsdsam_module

    def fake_import_module(name: str) -> object:
        if name == "torch":
            raise ModuleNotFoundError("No module named 'torch'")
        raise AssertionError(f"unexpected import before torch dependency check: {name}")

    monkeypatch.setattr(lsdsam_module, "import_module", fake_import_module)

    adapter = get_adapter("lsdsam-fewshot")

    with pytest.raises(ModelAdapterDependencyError, match="torch"):
        adapter.load(SimpleNamespace(checkpoint_path=checkpoint))


def test_lsdsam_adapter_reports_missing_checkpoint_before_dependencies(tmp_path) -> None:
    adapter = get_adapter("lsdsam-fewshot")

    with pytest.raises(ModelAdapterError, match="LSDSAM checkpoint does not exist") as exc:
        adapter.load(SimpleNamespace(checkpoint_path=tmp_path / "missing.pth"))

    assert "torch" not in str(exc.value).lower()


def test_lsdsam_adapter_loads_legacy_lora_chain_with_fake_dependencies(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from lsd_app.models import lsdsam as lsdsam_module

    lora_checkpoint = tmp_path / "lora.pth"
    base_checkpoint = tmp_path / "sam_base.pth"
    lora_checkpoint.write_bytes(b"fake")
    base_checkpoint.write_bytes(b"fake")
    calls: list[object] = []

    class FakeTensor:
        shape = (16, 4)

    lora_state = {
        "base_model.model.image_encoder.blocks.0.attn.qkv.lora_A.Task-Specific.weight": FakeTensor(),
        "base_model.model.image_encoder.blocks.0.attn.qkv.lora_B.Task-Specific.weight": FakeTensor(),
    }

    class FakeParam:
        def __init__(self, requires_grad: bool = True) -> None:
            self.requires_grad = requires_grad

    class FakeImageEncoder:
        def resize_pos_embed(self, new_img_size: int) -> None:
            calls.append(("resize_pos_embed", new_img_size))

    class FakeInnerModel:
        def named_parameters(self):
            return [("image_encoder.blocks.0.attn.qkv.weight", FakeParam())]

    class FakeBaseModel:
        model = FakeInnerModel()

    class FakeModel:
        image_encoder = FakeImageEncoder()
        base_model = FakeBaseModel()

        def named_parameters(self):
            return [
                ("image_encoder.blocks.0.attn.qkv.weight", FakeParam()),
                ("other.weight", FakeParam(False)),
            ]

        def named_modules(self):
            return [
                ("image_encoder.blocks.0.attn.qkv", object()),
                ("image_encoder.blocks.0.attn.proj", object()),
                ("image_encoder.blocks.0.mlp.lin1", object()),
                ("image_encoder.blocks.0.mlp.lin2", object()),
            ]

        def load_state_dict(self, state_dict, strict=True):
            calls.append(("load_state_dict", state_dict, strict))

        def to(self, device: str):
            calls.append(("to", device))
            return self

        def eval(self):
            calls.append("eval")

        def add_adapter(self, name, config):
            calls.append(("add_adapter", name, config.r, config.lora_alpha))

        def set_adapter(self, name):
            calls.append(("set_adapter", name))

    class FakeBuildModule:
        __file__ = str(tmp_path / "build_sam_diy.py")
        b_params = {"checkpoint": str(base_checkpoint)}

        @staticmethod
        def build_sam_b(checkpoint, map_location="cuda"):
            calls.append(("build_sam_b", checkpoint, map_location))
            return FakeModel()

    class FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return True

    class FakeTorch:
        cuda = FakeCuda()
        float32 = "float32"

        @staticmethod
        def load(path, map_location=None, weights_only=False):
            path = str(path)
            calls.append(("torch.load", path, map_location, weights_only))
            if path.endswith("lora.pth"):
                return {"model_state_dict": lora_state}
            return {"model": {"base": "state"}}

    class FakeLoraConfig:
        def __init__(self, r, lora_alpha, target_modules, lora_dropout, bias):
            self.r = r
            self.lora_alpha = lora_alpha
            self.target_modules = target_modules
            self.lora_dropout = lora_dropout
            self.bias = bias

    class FakePeft:
        LoraConfig = FakeLoraConfig

    def fake_import_module(name: str):
        if name == "torch":
            return FakeTorch
        if name == "peft":
            return FakePeft
        if name == "cv2":
            return object()
        if name == "pywt":
            return object()
        if name.endswith("build_sam_diy"):
            return FakeBuildModule
        if "Predicter1" in name:
            raise AssertionError("adapter should not import Predicter1.py")
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(lsdsam_module, "import_module", fake_import_module)

    adapter = get_adapter("lsdsam-fewshot").load(
        SimpleNamespace(
            checkpoint_path=lora_checkpoint,
            threshold=0.5,
            device="cuda",
            config_path=None,
            adapter_options={
                "legacy_root": str(tmp_path),
                "sam_checkpoint_path": str(base_checkpoint),
                "edge": "none",
                "image_size": 512,
            },
        )
    )

    assert adapter.name == "LSDSAM-B"
    assert adapter.edge == "wavelets"
    assert ("build_sam_b", False, "cuda") in calls
    assert ("resize_pos_embed", 512) in calls
    assert not any(
        call == "merge_and_unload" or (isinstance(call, tuple) and call[0] == "get_peft_model")
        for call in calls
    )
    assert ("add_adapter", "Task-Specific", 16, 32) in calls
    assert ("set_adapter", "Task-Specific") in calls
    assert ("load_state_dict", lora_state, True) in calls
    assert "eval" in calls


def test_lsdsam_adapter_uses_registered_full_peft_weight_without_default_sam_checkpoint(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from lsd_app.models import lsdsam as lsdsam_module

    checkpoint = tmp_path / "registered_taft.pth"
    checkpoint.write_bytes(b"fake")
    calls: list[object] = []

    class FakeTensor:
        shape = (16, 4)

    state = {
        "base_model.model.image_encoder.blocks.0.attn.qkv.base_layer.weight": FakeTensor(),
        "base_model.model.image_encoder.blocks.0.attn.qkv.lora_A.default.weight": FakeTensor(),
        "base_model.model.image_encoder.blocks.0.attn.qkv.lora_B.default.weight": FakeTensor(),
    }

    class FakeImageEncoder:
        def resize_pos_embed(self, new_img_size: int) -> None:
            calls.append(("resize_pos_embed", new_img_size))

    class FakeModel:
        image_encoder = FakeImageEncoder()

        def named_parameters(self):
            return []

        def named_modules(self):
            return [("image_encoder.blocks.0.attn.qkv", object())]

        def load_state_dict(self, state_dict, strict=True):
            calls.append(("load_state_dict", state_dict, strict))

        def to(self, device: str):
            calls.append(("to", device))
            return self

        def eval(self):
            calls.append("eval")

    class FakeBuildModule:
        __file__ = str(tmp_path / "build_sam_diy.py")
        b_params = {"checkpoint": "check_points/sam_vit_b_01ec64.pth"}

        @staticmethod
        def build_sam_b(checkpoint, map_location="cuda"):
            calls.append(("build_sam_b", checkpoint, map_location))
            return FakeModel()

    class FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return True

    class FakeTorch:
        cuda = FakeCuda()

        @staticmethod
        def load(path, map_location=None, weights_only=False):
            calls.append(("torch.load", str(path), map_location, weights_only))
            if str(path).endswith("sam_vit_b_01ec64.pth"):
                raise AssertionError("registered full PEFT weight should not load default SAM checkpoint")
            return {"model_state_dict": state}

    class FakeLoraConfig:
        def __init__(self, r, lora_alpha, target_modules, lora_dropout, bias):
            self.r = r
            self.lora_alpha = lora_alpha
            self.target_modules = target_modules
            self.lora_dropout = lora_dropout
            self.bias = bias

    class FakePeftModel:
        def __init__(self, model):
            self.model = model

        def load_state_dict(self, state_dict, strict=True):
            calls.append(("peft_load_state_dict", state_dict, strict))

        def merge_and_unload(self):
            calls.append("merge_and_unload")
            return self.model

        def set_adapter(self, name):
            calls.append(("set_adapter", name))

        def to(self, device):
            return self.model.to(device)

        def eval(self):
            return self.model.eval()

    class FakePeft:
        LoraConfig = FakeLoraConfig

        @staticmethod
        def get_peft_model(model, config, adapter_name="default"):
            calls.append(("get_peft_model", adapter_name, config.r, config.lora_alpha))
            return FakePeftModel(model)

    def fake_import_module(name: str):
        if name == "torch":
            return FakeTorch
        if name == "peft":
            return FakePeft
        if name == "cv2":
            return object()
        if name == "pywt":
            return object()
        if name.endswith("build_sam_diy"):
            return FakeBuildModule
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(lsdsam_module, "import_module", fake_import_module)

    adapter = get_adapter("lsdsam-fewshot").load(
        SimpleNamespace(
            checkpoint_path=checkpoint,
            threshold=0.5,
            device="cuda",
            config_path=None,
            adapter_options={"legacy_root": str(tmp_path), "image_size": 512},
        )
    )

    assert adapter.name == "LSDSAM-B"
    assert ("get_peft_model", "default", 16, 32) in calls
    assert ("build_sam_b", False, "cuda") in calls
    assert not any("sam_vit_b_01ec64.pth" in str(call) for call in calls)


def test_lsdsam_cuda_request_falls_back_to_cpu_with_warning() -> None:
    from lsd_app.models import lsdsam as lsdsam_module

    class FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return False

    class FakeTorch:
        cuda = FakeCuda()

    with pytest.warns(RuntimeWarning, match="Falling back to CPU"):
        assert lsdsam_module._select_device(FakeTorch, "cuda") == "cpu"

    assert lsdsam_module._select_device(FakeTorch, "gpu") == "cpu"
    assert lsdsam_module._select_device(FakeTorch, "auto") == "cpu"


def test_lsdsam_imagenet_preprocessing_scales_uint_ranges() -> None:
    import numpy as np
    from lsd_app.models import lsdsam as lsdsam_module

    white = np.full((1, 1, 3), 255.0, dtype=np.float32)
    unit = lsdsam_module._ensure_unit_range(white)
    normalized = lsdsam_module._imagenet_normalize(unit)

    expected = (np.ones(3, dtype=np.float32) - lsdsam_module._PIXEL_MEAN) / lsdsam_module._PIXEL_STD
    np.testing.assert_allclose(normalized[0, 0, :], expected, rtol=1e-6)


def test_lsdsam_adapter_applies_RGB_bands_inside_preprocess(monkeypatch: pytest.MonkeyPatch) -> None:
    import numpy as np
    from lsd_app.models import lsdsam as lsdsam_module

    class FakeTorch:
        float32 = np.float32

        @staticmethod
        def as_tensor(values, dtype=None):
            return np.asarray(values, dtype=np.float32)

    class FakeCv2:
        COLOR_RGB2GRAY = 1

        @staticmethod
        def resize(image, size, interpolation=None):
            return image

        @staticmethod
        def cvtColor(image, code):
            return image[:, :, 0]

    def fake_import_module(name: str):
        if name == "cv2":
            return FakeCv2
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(lsdsam_module, "import_module", fake_import_module)
    monkeypatch.setattr(lsdsam_module, "_build_edge_features", lambda image, edge, cv2: np.zeros_like(image))

    adapter = get_adapter("lsdsam-fewshot")
    adapter.torch = FakeTorch
    adapter.image_size = 2
    adapter.RGB_bands = "RGB_bands: 3,2,1"

    patch = np.zeros((3, 2, 2), dtype=np.float32)
    patch[0, :, :] = 0.1
    patch[1, :, :] = 0.2
    patch[2, :, :] = 0.3

    model_input = adapter.preprocess_patch(patch)
    expected_rgb = np.array([0.3, 0.2, 0.1], dtype=np.float32)
    expected = (expected_rgb - lsdsam_module._PIXEL_MEAN) / lsdsam_module._PIXEL_STD

    np.testing.assert_allclose(model_input[:3, 0, 0], expected, rtol=1e-6)



def test_lsdformer_adapter_reports_missing_checkpoint_before_dependencies(tmp_path) -> None:
    adapter = get_adapter("lsdformer-fast")

    with pytest.raises(ModelAdapterError, match="LSDFormer checkpoint does not exist") as exc:
        adapter.load(SimpleNamespace(checkpoint_path=tmp_path / "missing.pth"))

    assert "torch" not in str(exc.value).lower()


def test_lsdformer_adapter_loads_fake_checkpoint(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    from lsd_app.models import lsdformer as lsdformer_module

    checkpoint = tmp_path / "lsdformer.pth"
    checkpoint.write_bytes(b"fake")
    calls: list[object] = []

    class FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return False

    class FakeTorch:
        cuda = FakeCuda()
        float32 = "float32"

        @staticmethod
        def load(path, map_location=None, weights_only=False):
            calls.append(("torch.load", str(path), map_location, weights_only))
            return {"model_state_dict": {"encoder.weight": object()}}

    class FakeLSDFormer:
        def __init__(self, **kwargs):
            calls.append(("LSDFormer", kwargs))

        def load_state_dict(self, state_dict, strict=False):
            calls.append(("load_state_dict", state_dict, strict))
            return SimpleNamespace(missing_keys=[], unexpected_keys=[])

        def to(self, device):
            calls.append(("to", device))
            return self

        def eval(self):
            calls.append("eval")
            return self

    def fake_import_module(name: str):
        if name == "torch":
            return FakeTorch
        if name == "cv2":
            return object()
        if name == "model":
            return SimpleNamespace(LSDFormer=FakeLSDFormer)
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(lsdformer_module, "import_module", fake_import_module)

    adapter = get_adapter("lsdformer-fast").load(
        SimpleNamespace(
            checkpoint_path=checkpoint,
            threshold=0.5,
            device="cpu",
            config_path=None,
            RGB_bands="RGB_bands: 3,2,1",
            adapter_options={"model_root": str(tmp_path), "image_size": 256},
        )
    )

    assert adapter.name == "LSDFormer"
    assert ("to", "cpu") in calls
    assert "eval" in calls


def test_lsdformer_hybrid_adapter_injection_shape_smoke() -> None:
    torch = pytest.importorskip("torch")
    from lsd_app.models.lsdformer import (
        BUNDLED_LSDFORMER_MODEL_ROOT,
        LSDFormerFeatureWrapper,
        apply_lsdformer_hybrid_adapters,
        infer_lsdformer_feature_channels,
    )

    import sys

    model_root = str(BUNDLED_LSDFORMER_MODEL_ROOT)
    if model_root not in sys.path:
        sys.path.insert(0, model_root)
    from model import LSDFormer

    model = LSDFormer(in_chans=3, num_classes=2)
    counts = apply_lsdformer_hybrid_adapters(torch, model)
    wrapper = LSDFormerFeatureWrapper(torch, model)
    logits, feats = wrapper(torch.zeros(1, 3, 64, 64), return_feats=True)

    assert counts["linear_lora"] > 0
    assert counts["conv_lora"] > 0
    assert logits.shape[-2:] == (64, 64)
    assert feats.shape[1] == infer_lsdformer_feature_channels(wrapper)
    assert not any(param.requires_grad for param in model.encoder.stem.parameters())
    assert any(param.requires_grad for param in model.decoder.parameters())
