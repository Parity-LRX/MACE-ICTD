# MACE-ICTD

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20690950.svg)](https://doi.org/10.5281/zenodo.20690950)

MACE-ICTD is a standalone implementation of MACE in an Irreducible Cartesian
Tensor Decomposition (ICTD) basis. It keeps the MACE interaction/readout
semantics, but stores equivariant angular features in compact Cartesian
irreducible blocks and uses fixed basis-change operators to communicate with
the original e3nn/MACE convention.

The repository is intended for three workflows:

- train MACE-style force fields directly in the ICTD basis;
- convert compatible native `mace-torch` checkpoints into MACE-ICTD checkpoints;
- export trained or converted models for ASE, AOTInductor, and LAMMPS deployment.

Full manuals:

- English: [docs/USER_MANUAL.md](docs/USER_MANUAL.md)
- 中文: [docs/USER_MANUAL.zh-CN.md](docs/USER_MANUAL.zh-CN.md)

## What Is Included

- `PureCartesianICTDFix`, a MACE-compatible model using compact ICTD angular
  blocks with dimension `2l + 1` per degree.
- Fixed `Q` and `U` operators for ICTD/e3nn basis correspondence and MACE
  symmetric-contraction compatibility.
- Native `mace-torch` conversion for supported `ScaleShiftMACE` checkpoints.
- Training from H5 datasets with energy, force, optional stress/virial losses,
  Stage-Two/SWA schedules, EMA, LR schedulers, checkpoint resume, and
  `make_fx`/Inductor training compilation.
- Optional cuEquivariance product backend for high-performance training and
  inference.
- AOTInductor `.pt2` export and a LAMMPS `USER-MFFTORCH` package.
- A learned scalar reciprocal long-range correction path
  (`reciprocal-spectral-v1`).
- Paper benchmark scripts, selected source CSV/JSON records, validation logs,
  and SVG figures under [benchmarks/paper](benchmarks/paper).

## Installation

```bash
cd /path/to/MACE-ICTD
pip install -e .
```

Optional extras:

```bash
pip install -e ".[pyg]"    # torch-scatter / torch-cluster acceleration
pip install -e ".[cue]"    # cuEquivariance product backend
pip install -e ".[e0]"     # pandas support for fitted E0 CSV files
pip install -e ".[full]"   # all optional Python extras
```

Runtime expectations:

- Python >= 3.9
- PyTorch >= 2.4
- PyTorch >= 2.7 recommended for AOTInductor and `make_fx` workflows
- `e3nn >= 0.4.4, < 0.6`
- CUDA for cuEquivariance and serious Inductor/AOTI benchmarking

An optional compiled ICTD tensor-product extension can be built with:

```bash
MFF_BUILD_ICTD_TP_EXT=1 pip install -e .
```

The pure-Python/PyTorch path works without this extension.

After installation, the console scripts `mff-convert-mace`, `mff-export-aoti`,
`mff-export-core`, and `mff-lammps` are available. The examples below use
`python -m ...` where possible so they also work from a source checkout before
the shell has refreshed its script PATH.

## Quick Check

```python
import torch
from mace_ictd.synthetic import build_model, make_fixed_graph, compute_energy_forces

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = build_model(
    channels=64,
    lmax=2,
    num_interaction=2,
    correlation=2,
    route="baseline",
    product_backend="ictd-bridge-u",
    dtype=torch.float64,
    device=device,
)

graph = make_fixed_graph(
    num_nodes=128,
    avg_degree=24,
    dtype=torch.float64,
    device=device,
)

energy, forces, atomic_energy = compute_energy_forces(
    model, graph, create_graph=False
)
```

The model forward signature is:

```python
model(pos, A, batch, edge_src, edge_dst, edge_shifts, cell)
```

where `A` is a tensor of atomic numbers, not species indices.

## Choosing a Runtime Mode

| Goal | Recommended mode |
| --- | --- |
| Conservative MACE parity or native MACE conversion | `product_backend=ictd-bridge-u`, `angular_basis=ictd` |
| Fast training | `product_backend=cueq`, `angular_basis=e3nn`, `--train-makefx-compile` |
| Fast AOTI inference with cuEq available | `python -m mace_ictd.cli.export_aoti_core --cueq-product --angular-basis e3nn` |
| Strictest deployment path | export the checkpoint without product replacement |
| Debug/reference comparisons | `native-mace` or `ictd-pure-u`, depending on the question being tested |

Important distinction:

- `ictd-bridge-u` is the main MACE-correspondence path. It preserves the source
  MACE product convention through the ICTD/e3nn basis bridge.
- `cueq + angular_basis=e3nn` is the main performance path. It folds compatible
  fixed angular operators once and uses cuEquivariance for the product block.
- `ictd-pure-u` is useful for diagnostics and ICTD-native experiments, but it is
  not the default exact native-MACE conversion path.

## Training From Scratch

The trainer consumes preprocessed H5 files such as:

```text
DATA/
  processed_train.h5
  processed_val.h5
```

Canonical bridge-U training:

```bash
python -m mace_ictd.cli.train \
  --data-dir DATA \
  --train-prefix train \
  --val-prefix val \
  --seed 123 \
  --channels 64 \
  --lmax 2 \
  --max-ell 2 \
  --num-interaction 2 \
  --correlation 2 \
  --function-type bessel \
  --product-backend ictd-bridge-u \
  --scaling rms_forces_scaling \
  --epochs 300 \
  --batch-size 4 \
  --loss smooth_l1 \
  --energy-weight 1.0 \
  --force-weight 10.0 \
  --stress-weight 0.0 \
  --optimizer adamw \
  --lr 1e-3 \
  --min-lr 1e-6 \
  --weight-decay 0.0 \
  --lr-scheduler plateau \
  --warmup-batches 1000 \
  --swa \
  --start-swa 225 \
  --swa-lr 1e-4 \
  --swa-energy-weight 1000.0 \
  --swa-force-weight 100.0 \
  --device cuda \
  --dtype float64 \
  --checkpoint model_bridge_u.pth
```

High-performance cuEq/product-compiled training:

```bash
python -m mace_ictd.cli.train \
  --data-dir DATA \
  --train-prefix train \
  --val-prefix val \
  --seed 123 \
  --channels 64 \
  --lmax 2 \
  --max-ell 2 \
  --num-interaction 2 \
  --correlation 2 \
  --function-type bessel \
  --product-backend cueq \
  --angular-basis e3nn \
  --train-makefx-compile \
  --makefx-buckets 6 \
  --scaling rms_forces_scaling \
  --epochs 300 \
  --batch-size 8 \
  --loss smooth_l1 \
  --energy-weight 1.0 \
  --force-weight 10.0 \
  --stress-weight 0.0 \
  --optimizer adamw \
  --lr 1e-3 \
  --min-lr 1e-6 \
  --lr-scheduler cosine \
  --warmup-batches 1000 \
  --swa \
  --start-swa 225 \
  --swa-lr 1e-4 \
  --swa-energy-weight 1000.0 \
  --swa-force-weight 100.0 \
  --ema-decay 0.999 \
  --ema-start-step 1000 \
  --device cuda \
  --dtype float32 \
  --checkpoint model_cueq_e3nn_makefx.pth
```

Training notes:

- `--epochs` is an epoch limit; `--max-steps` is an optional optimizer-step cap.
  If both are set, training stops when either limit is reached.
- Stage Two/SWA changes the loss weights and LR at the SWA boundary, matching
  the relevant `mace-torch` behavior used by this repository.
- LR schedules include `plateau`, `exp`, `cosine`, `step`, and `none`; the LR is
  clamped to `[--min-lr, --lr]`.
- MACE-style ScaleShift is enabled by default through
  `--scaling rms_forces_scaling`; use `--scaling no_scaling` only when that is
  intentionally part of the experiment.
- Checkpoints store model hyperparameters, scaling, E0 metadata, optimizer
  state, EMA/SWA state when enabled, and enough metadata for deployment reloads.

See [docs/USER_MANUAL.md](docs/USER_MANUAL.md) for all CLI options.

## Convert a Native MACE Checkpoint

MACE-ICTD can import compatible native `mace-torch` `ScaleShiftMACE` objects:

```bash
python -m mace_ictd.cli.convert_mace \
  --mace-model mace.model \
  --out mace_ictd.pth \
  --product-backend ictd-bridge-u \
  --dtype float64 \
  --device cpu
```

The input must be a `torch.save(model)` object, not a raw `state_dict`.

Conservative AOTI export from the converted checkpoint:

```bash
python -m mace_ictd.cli.export_aoti_core \
  --checkpoint mace_ictd.pth \
  --elements H,C,N,O \
  --out mace_ictd.pt2 \
  --dynamic \
  --embed-e0
```

Performance-oriented cuEq export:

```bash
python -m mace_ictd.cli.export_aoti_core \
  --checkpoint mace_ictd.pth \
  --elements H,C,N,O \
  --out mace_ictd_cueq_e3nn.pt2 \
  --dynamic \
  --cueq-product \
  --angular-basis e3nn
```

Supported conversion is intentionally structure-based and strict. The current
converter targets the `ScaleShiftMACE` layout used in this repository's tests
and benchmarks, validated with `mace==0.3.16` and `e3nn<0.6`. Newer MACE
releases may work when the saved object layout remains compatible, but arbitrary
research forks, raw state dicts, pair-repulsion variants, and changed readout or
radial layouts are rejected rather than silently converted.

For exact MACE correspondence, the converted ICTD model must preserve the source
model's structural options, including `use_reduced_cg`.

## Long-Range Correction

The supported long-range path is a learned scalar reciprocal correction:

```bash
python -m mace_ictd.cli.train \
  --data-dir DATA \
  --channels 64 \
  --lmax 2 \
  --num-interaction 2 \
  --long-range-mode reciprocal-spectral-v1 \
  --long-range-boundary periodic \
  --long-range-reciprocal-backend direct_kspace \
  --long-range-kmax 4 \
  --long-range-source-channels 1 \
  --checkpoint model_lr.pth
```

`--long-range-source-channels` is the number of learned scalar source channels
predicted from the final invariant descriptor. The module is initialized near
zero, so enabling it starts close to the short-range model and learns the
correction from the same energy/force/stress losses. This is not a fixed-charge
analytic Ewald term.

Native MACE conversion preserves the source model; it does not add long-range
terms to an already trained MACE checkpoint.

## ASE, AOTI, and LAMMPS

ASE:

```python
from mace_ictd.evaluation.calculator import MyE3NNCalculator

atoms.calc = MyE3NNCalculator(checkpoint="model.pth", device="cuda")
```

AOTInductor:

```bash
python -m mace_ictd.cli.export_aoti_core \
  --checkpoint model.pth \
  --elements H,C,N,O \
  --out model.pt2 \
  --dynamic
```

LAMMPS support is in [lammps_user_mfftorch](lammps_user_mfftorch). After building
LAMMPS with the provided `USER-MFFTORCH` package and LibTorch/AOTI support:

```text
pair_style   mff/torch  model.pt2
pair_coeff   * *
```

Read [lammps_user_mfftorch/README.md](lammps_user_mfftorch/README.md) and
[lammps_user_mfftorch/docs/BUILD_AND_RUN.md](lammps_user_mfftorch/docs/BUILD_AND_RUN.md)
for build details.

## Benchmarks and Reproducibility

The curated benchmark archive lives in [benchmarks/paper](benchmarks/paper). It
contains lightweight scripts, CSV/JSON summaries, validation logs, and SVG
figures used by the technical report. Large generated binaries such as
checkpoint snapshots, AOTI packages, PNG/PDF figure exports, and MD trajectory
arrays are kept out of Git and should be distributed through GitHub Releases.

Replot the archived figures from the repository root:

```bash
python benchmarks/paper/scripts/plot_benchmark_figures.py
python benchmarks/paper/scripts/training/plot_paper_training_figures.py
```

Selected RTX 4090 throughput figures:

![Backend throughput benchmark](docs/figures/backend_throughput_benchmark_channels64.png)

![Backend speedup benchmark](docs/figures/backend_speedup_benchmark_channels64.png)

These benchmarks are controlled computational records, not universal
chemical-accuracy leaderboards. The synthetic whole-model runs fix graph
workloads to expose backend throughput; the matched training runs fix small
protocols to compare parameterization/backend behavior under the same data and
optimizer settings.

## Validation Scope

The repository contains tests and benchmark records for:

- ICTD/e3nn frame correspondence;
- converted native-MACE energy and force agreement for supported models;
- rotation invariance/equivariance checks;
- fused contraction agreement against reference contraction paths;
- representation-layer SO(2) and spin-coupled double-cover prototype checks;
- long-MD checkpoint-correspondence diagnostics for selected models.

The strongest exactness statement is for the supported MACE correspondence path:
compatible native MACE checkpoints can be represented in MACE-ICTD with the same
energy/force behavior up to expected floating-point tolerances. This should not
be read as a guarantee for arbitrary MACE forks, unsupported model structures,
or every possible ICTD-native experimental backend.

## Repository Layout

```text
MACE-ICTD/
  mace_ictd/
    cli/                 # training, conversion, export, LAMMPS helpers
    data/                # preprocessing, H5 datasets, batching
    evaluation/          # ASE calculator wrappers
    interfaces/          # checkpoint and deployment wrappers
    models/              # ICTD model, irreps, products, radial, long-range
    training/            # ForceTrainer and make_fx compile helpers
    utils/               # graph, scatter, config, checkpoint metadata
  lammps_user_mfftorch/  # LAMMPS USER-MFFTORCH package
  benchmarks/paper/      # curated paper benchmark records and scripts
  docs/                  # user manuals and figures
```

## Citation

If you use MACE-ICTD, cite the Zenodo DOI:

```text
MACE-ICTD DOI: 10.5281/zenodo.20690950
```
