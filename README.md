# LSD_APP

LSD_APP 是一个面向滑坡遥感识别、制图、评估和训练管理的本地工作台。当前工程把旧科研脚本保留为 legacy 资产，把稳定能力沉淀到 `lsd_app/`：模型仓库、权重仓库、数据仓库、端到端推理、独立评估、训练入口和 Streamlit 操作台。

## 模块定位

Streamlit 工作台按以下模块组织：

1. **Model Repository**：展示 Net/Adapter 的基本信息，不存权重。
2. **Weight Repository**：登记不同版本 checkpoint，阶段包括 `TAFT`、`DAFT`、`TSFT`、`inference-ready`，并挂到 `model_id`。
3. **Dataset Repository**：登记 `TAFT` 预训练数据、`DAFT` 域适应数据、`TSFT` 小样本数据。用户视角用 `version` 作为唯一标识，`dataset_id` 只作为内部 registry 键。
4. **DAFT Training**：预留域适应训练 dry-run 接口，不启动真实训练。
5. **TSFT Training**：预留小样本训练 dry-run 接口，不启动真实训练。
6. **Inference**：只做部署推理，输出滑坡多边形。
7. **Evaluation**：选择带 GT 的数据集和某个权重进行独立评估。

## 安装与启动

推荐使用项目启动脚本。首次启动会自动创建名为 `lsdector` 的 conda 环境并安装依赖；之后再次启动会检测并复用这个环境。

```bash
cd /path/to/LSD__APP
bash tools/start_lsd_app.sh
```

Windows PowerShell 可使用：

```powershell
cd C:\path\to\LSD__APP
.\tools\start_lsd_app.ps1
```

默认启动地址为 `0.0.0.0:8501`。可以用环境变量覆盖，例如：

```bash
LSD_APP_PORT=8502 LSD_APP_ADDRESS=127.0.0.1 bash tools/start_lsd_app.sh
```

打开 [http://127.0.0.1:8501](http://127.0.0.1:8501)。

CLI 可直接使用模块入口：

```bash
conda run -n lsdector env PYTHONPATH=/path/to/LSD__APP python -m lsd_app.cli.main --help
```

## 仓库结构

默认 registry 存在 `outputs/registry/`：

```text
outputs/registry/
  models.json
  weights.json
  datasets.json
  prediction_runs.json
  dataset_store/
    TAFT/
      v1/
        20260505/
          train/images/
          train/labels/
          test/images/
          test/labels/
        manifest.json
    DAFT/
      target-v1/
        images/
        manifest.json
    TSFT/
      few-shot-v1/
        train/images/
        train/labels/
        test/images/
        test/labels/
        manifest.json
```

模型仓库字段：

```text
model_id, net_name, adapter, model_family, input_mode, channel_rule, status, description
```

权重仓库字段：

```text
weight_id, model_id, stage, version, path, exists, created_at, metrics_summary, notes,
weight_type, base_checkpoint_path, adapter_profile, adapter_name, image_size, input_mode,
channel_rule, lora_base_r, lora_base_alpha, lora_task_r, lora_task_alpha,
lora_dropout, peft_adapter_name, target_modules
```

数据仓库字段：

```text
dataset_id, dataset_class, name, imagery_path, label_path, has_gt, RGB_bands, RGB_bands_class,
tile_schema, source, created_at, notes, version, repository_path, manifest_path,
sample_shape_layout, sample_dtype, sample_count, train_count, test_count, subset_count
```

数据类型固定为：

- `TAFT`：预训练数据。
- `DAFT`：域适应/评估相关数据。
- `TSFT`：用户标注小样本训练数据。

## 数据仓库标准化

数据仓库的最小样本语义统一为 `(H, W, C)`，数值统一为 `float32 0..1`。用户上传 `.tif/.tiff/.png` 时，标准化入口会读取为 HWC，保留原始通道数量和原始通道顺序，只把 `RGB_bands` 作为数据集元数据写入 manifest。若样本不是 `float32 0..1`，就对每个样本内部、每个通道单独执行 linear stretch，并以标准化 GeoTIFF 写入 `dataset_store/`。

三类数据仓库的落盘规则：

- `TAFT/<version>/<time-subset>/train|test/images|labels/`：预训练数据版本可以包含多个时间子集。TAFT tree 只保留 `version` 作为对外标识，不再要求 tree name/preset。
- `DAFT/<version>/images/`：域适应数据先保留 images-only 形式，不要求 label。支持两种导入方式：`packaged images` 和 `Tiff Slicing`。
- `TSFT/<version>/<subset>/train|test/images|labels/`：小样本数据版本可以包含多个子集。`train` 必须提供，`test` 可选预留。

每个标准化版本都会写 `manifest.json`，记录 `sample_count`、`train_count`、`test_count`、`has_gt`、`RGB_bands`、`sample_shape_layout=(H, W, C)`、`sample_dtype=float32` 和 `concat_ready=true`。这些统计会同步进入 `datasets.json`，并在 Streamlit Dataset Repository 表格中展示。

## CLI 用法

查看模型仓库：

```bash
python -m lsd_app.cli.main models list --project-root /Volumes/WD_BLACK/LSD__APP
```

登记权重：

```bash
python -m lsd_app.cli.main weights add \
  --registry-root outputs \
  --model-id lsdsam \
  --stage TSFT \
  --version v1 \
  --path /path/to/checkpoint.pth \
  --metrics-summary "f1=0.82" \
  --notes "customer few-shot set 1"
```

登记数据集：

```bash
python -m lsd_app.cli.main datasets add \
  --registry-root outputs \
  --name "customer-tsft-round-1" \
  --class TSFT \
  --imagery /path/to/images_or_raster \
  --label /path/to/labels_or_gt \
  --has-gt \
  --rgb-bands "RGB_bands: 3,2,1" \
  --tile-schema "tile_size=512;stride=256" \
  --source customer
```

上传并标准化入数据仓库：

```bash
python -m lsd_app.cli.main datasets add \
  --registry-root outputs \
  --name "taft-hunan-v1" \
  --class TAFT \
  --version v1 \
  --subset 20260505 \
  --imagery /path/to/raw/images \
  --label /path/to/raw/labels \
  --rgb-bands "RGB_bands: 3,2,1" \
  --standardize
```

查看标准化数据集/子集统计：

```bash
python -m lsd_app.cli.main datasets summary --registry-root outputs
```

导入 Landslide70K/93K 这类层级 TAFT tree：

```bash
python -m lsd_app.cli.main datasets import-taft-tree \
  --registry-root outputs \
  --version Landslide93K \
  --root /Volumes/WD_BLACK/LSD__APP/LSDSAM_finetune/Landslide70K \
  --standardize \
  --resume
```

这里 `--resume` 是续传语义：只有当已有版本的 source/config 和本次一致时才继续；已完整且通过完整性校验的 image/label pair 会被跳过，缺失或损坏的 pair 会重写。`--force-rebuild` 才是删除 `outputs/registry/dataset_store/TAFT/<version>/` 并从头重建；二者不要同时使用。

端到端推理只做部署：

```bash
python -m lsd_app.cli.main run --config path/to/task.yaml --model mock
```

推理链路固定为：

```text
prepare -> infer -> mosaic -> vectorize
```

输出重点文件：

```text
outputs/.../rasters/probability.tif
outputs/.../rasters/mask.tif
outputs/.../vectors/landslides.shp
outputs/.../vectors/landslides.gpkg
outputs/.../vectors/landslides.geojson
```

可视化预览是独立可选步骤，Streamlit 里可点击 `Generate preview`，CLI 里可单独运行 `visualize`。

## 权重与 LSDSAM LoRA 调用逻辑

权重仓库只登记本地 checkpoint 路径和模型对齐元数据，不复制大文件。当前已经登记了 legacy 权重：

```text
weight_id: weight-lsdsam-fewshot-inference-ready-ls70k-sam-b-lora-edge4-after512-cf-mh
model_id: lsdsam-fewshot
path: /Volumes/WD_BLACK/LSD__APP/LSDSAM_finetune/models_save/LS70k/SAM_b_LoRA_edge4_after512_cf_mh.pth
weight_type: lora_task
base_checkpoint_path: /Volumes/WD_BLACK/LSD__APP/LSDSAM_finetune/SAM_edge_4_after512_crossfusion_multihead_LoRA/check_points/sam_vit_b_01ec64.pth
adapter_profile: lsdsam-sam-b-lora-edge4-after512-cf-mh
image_size: 512
lora_base_r/lora_base_alpha: 32/64
lora_task_r/lora_task_alpha: 16/32
peft_adapter_name: Task-Specific
target_modules: image_encoder.blocks.*.(attn.qkv|attn.proj|mlp.lin1|mlp.lin2)
```

LSDSAM 推理时，`lsd_app/models/lsdsam.py::LSDSAMFewShotAdapter.load` 会按 legacy `finetune_SAMLoRA_Hunan_UAV_3.py` 的结构组网：

1. 读取 `model.checkpoint_path` 作为 task LoRA checkpoint。
2. 导入 legacy `SAM_edge_4_after512_crossfusion_multihead_LoRA.build_sam_diy::build_sam_b`。
3. 加载 SAM-B base checkpoint；如无法从 legacy build module 解析，需要在 config 的 `model.options.sam_checkpoint_path` 显式传入。
4. `resize_pos_embed(new_img_size=512)`。
5. 对 `image_encoder.blocks.*` 中的 `attn.qkv/attn.proj/mlp.lin1/mlp.lin2` 建 base LoRA：`r=32, alpha=64`，随后 merge。
6. 新增 `Task-Specific` LoRA adapter：`r=16, alpha=32`。
7. 从 checkpoint 的 `model_state_dict` 加载权重，并进入 `eval()` 推理。

推理配置中的关键字段：

```yaml
model:
  name: lsdsam-fewshot
  checkpoint_path: /Volumes/WD_BLACK/LSD__APP/LSDSAM_finetune/models_save/LS70k/SAM_b_LoRA_edge4_after512_cf_mh.pth
  options:
    legacy_root: /Volumes/WD_BLACK/LSD__APP/LSDSAM_finetune
    sam_checkpoint_path: /Volumes/WD_BLACK/LSD__APP/LSDSAM_finetune/SAM_edge_4_after512_crossfusion_multihead_LoRA/check_points/sam_vit_b_01ec64.pth
    image_size: 512
    lora_base_r: 32
    lora_base_alpha: 64
    lora_task_r: 16
    lora_task_alpha: 32
    peft_adapter_name: Task-Specific
input:
  RGB_bands: RGB_bands: 3,2,1
```

评估独立运行，必须配置 truth raster 或 truth vector：

```bash
python -m lsd_app.cli.main evaluate --config path/to/evaluation_task.yaml
```

## Mock 验收

生成本地小样例并跑通推理、评估和仓库登记：

```bash
tools/test_lsd_tool_smoke.sh --work-dir outputs/tool_smoke_latest
```

手动最小链路：

```bash
python examples/generate_mock_dataset.py --output /tmp/lsd_app_demo --with-truth --overwrite
python -m lsd_app.cli.main run --config /tmp/lsd_app_demo/config.yaml --model mock
python -m lsd_app.cli.main evaluate --config /tmp/lsd_app_demo/config.yaml
```

测试：

```bash
python -m compileall -x '(^|/)\._' lsd_app
pytest -q
```

## 输入规则

- LSDSAM 固定使用 `RGB + wavelets`，edge 分支不再暴露选择。
- 其他模型统一使用 `RGB`。
- 输入数据必须声明 `RGB_bands`，表示模型 RGB 三通道分别读取原始 GeoTIFF 的哪三个 1-based band。数据仓库会保留原始全部 band，具体模型 adapter 在消费数据时按 `RGB_bands` 抽取三通道。常用示例：
  - `RGB_bands: 1,2,3`：普通 RGB。
  - `RGB_bands: 3,2,1`：普通 BGR 存储。
  - `RGB_bands: 4,3,2`：Sentinel-2 true color。
  - `RGB_bands: 6,4,2`：PlanetScope true color。
- `RGB_bands` 是全项目唯一通道语义入口；LSDSAM/LSDFormer adapter 内部按该元数据抽取模型可见 RGB 三通道，不再暴露 legacy `mode` 通道开关。
- LSDSAM 使用 ImageNet RGB mean/std：`mean=(0.485, 0.456, 0.406)`，`std=(0.229, 0.224, 0.225)`。这一步在 `lsd_app/models/lsdsam.py` 的 adapter 预处理里执行，先生成 wavelets，再只对 RGB 三通道做 ImageNet normalization。

## Legacy Dataloader 与增强

旧代码 `LSDSAM_finetune/dataloader_SAM.py::CustomDataset` 暂不直接并入推理链路。它的能力被拆成两类：

- 数据读取和通道语义：迁到 `lsd_app/core/channels.py`、`lsd_app/config.py`、`lsd_app/core/inference.py`，由 `RGB_bands` 统一控制。
- 训练期增强 profile：登记在 `lsd_app/training/augmentations.py`，会出现在 TAFT/DAFT/TSFT dry-run plan 中，后续接真实训练 DataModule 时调用。

legacy 训练增强包括 `LongestMaxSize`、`PadIfNeeded`、水平/垂直翻转、`RandomResizedCrop`、旋转、噪声、模糊、亮度对比度扰动；测试/评估只保留 resize/pad。推理和 evaluation 不做随机增强。

## Legacy 边界

以下目录只作为迁移参考，本轮优化不移动、不删除、不重命名：

- `LSDSAM_finetune/`
- `preprocess/`

真实大权重、完整数据集、大幅遥感影像和实验输出不进入代码仓库，只通过本地 registry 记录路径和元数据。
