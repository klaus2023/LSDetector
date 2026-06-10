import pytest

np = pytest.importorskip("numpy")

from lsd_app.models.lsdsam import (
    _attach_lora_adapter,
    _infer_lora_rank,
    _is_merged_weight_type,
    _lora_adapter_names,
    _resolve_lora_alpha,
    _resolve_lora_adapter_name,
)


def test_lora_adapter_name_is_detected_from_checkpoint_keys() -> None:
    state_dict = {
        "base_model.model.image_encoder.blocks.0.attn.qkv.lora_A.default.weight": np.zeros((32, 768)),
        "base_model.model.image_encoder.blocks.0.attn.qkv.lora_B.default.weight": np.zeros((2304, 32)),
    }

    assert _lora_adapter_names(state_dict) == {"default"}
    assert _resolve_lora_adapter_name(state_dict, preferred="Task-Specific") == "default"
    assert _infer_lora_rank(state_dict, adapter_name="default", fallback=16) == 32


def test_merged_state_dict_is_detected_when_lora_keys_are_absent() -> None:
    assert _is_merged_weight_type("merged", {"image_encoder.blocks.0.attn.qkv.weight": object()})
    assert _is_merged_weight_type("", {"image_encoder.blocks.0.attn.qkv.weight": object()})
    assert not _is_merged_weight_type(
        "",
        {"base_model.model.image_encoder.blocks.0.attn.qkv.lora_A.default.weight": object()},
    )


def test_raw_sam_model_is_wrapped_before_adding_lora_adapter() -> None:
    class RawSam:
        pass

    class FakeWrapped:
        def __init__(self) -> None:
            self.selected_adapter = ""

        def set_adapter(self, adapter_name: str) -> None:
            self.selected_adapter = adapter_name

    class FakePeft:
        def __init__(self) -> None:
            self.calls = []
            self.wrapped = FakeWrapped()

        def get_peft_model(self, model, lora_config, adapter_name: str):
            self.calls.append((model, lora_config, adapter_name))
            return self.wrapped

    raw_model = RawSam()
    fake_peft = FakePeft()
    config = object()

    wrapped = _attach_lora_adapter(fake_peft, raw_model, "default", config)

    assert wrapped is fake_peft.wrapped
    assert fake_peft.calls == [(raw_model, config, "default")]
    assert wrapped.selected_adapter == "default"


def test_default_lora_uses_base_alpha_when_available() -> None:
    assert _resolve_lora_alpha({"lora_base_alpha": 64, "lora_task_alpha": 32}, adapter_name="default", fallback=16) == 64
    assert _resolve_lora_alpha({"lora_task_alpha": 32}, adapter_name="Task-Specific", fallback=16) == 32
