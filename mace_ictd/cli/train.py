r"""Train a baseline ICTD-MACE model (energy + force) from preprocessed H5.

Standalone training entry point so MACE-ICTD can *train*, not only deploy. Supports
the make_fx-compiled second-order force step (``--train-makefx-compile``) and the
size-bucketing that makes it cheap on variable-size data (``--makefx-buckets``).

The model is constructed with the SAME ``ModelConfig`` -> kwarg mapping that the
deploy-side ``LAMMPS_MLIAP_MFF.from_checkpoint`` uses, and every architecture choice
is written into the checkpoint's ``model_hyperparameters``, so a model trained here
reloads bit-for-bit for AOTI / LAMMPS export.

Example
-------
    python -m mace_ictd.cli.train \
        --data-dir /path/to/data \           # holds processed_train.h5 (+ processed_val.h5)
        --channels 64 --lmax 2 --num-interaction 2 \
        --epochs 50 --batch-size 4 --device cuda --dtype float64 \
        --train-makefx-compile --makefx-buckets 6 \
        --checkpoint model.pth
"""

from __future__ import annotations

import argparse
import logging
import os

import h5py
import torch
from torch.utils.data import DataLoader

from mace_ictd.data import H5Dataset, collate_fn_h5, BucketBatchSampler
from mace_ictd.models.pure_cartesian_ictd_fix import PureCartesianICTDFix
from mace_ictd.utils.config import ModelConfig
from mace_ictd.training.train_loop import ForceTrainer


def _avg_num_neighbors_from_h5(path: str) -> float:
    """Total directed edges / total nodes over the whole training file.

    This is the message-normalization constant the ictd-fix model divides by; it is a
    plain float (not a state_dict buffer), so it must be recorded in the checkpoint or
    the deployed forces are silently wrong. Mirrors the FSCETP cli computation."""
    tot_e = tot_n = 0
    with h5py.File(path, "r") as f:
        for k in f.keys():
            tot_e += int(f[k]["edge_src"].shape[0])
            tot_n += int(f[k]["pos"].shape[0])
    return float(tot_e) / max(tot_n, 1)


def build_baseline_model(
    cfg: ModelConfig,
    *,
    avg_num_neighbors: float,
    num_interaction: int,
    route: str,
    product_backend: str,
    correlation: int,
    radial_sqrt_num_basis: bool,
    attn_heads: int,
    atomic_numbers,
    ictd_save_tp_mode: str,
    invariant_channels: int,
    device,
    dtype,
) -> PureCartesianICTDFix:
    """Construct the model exactly the way from_checkpoint rebuilds it (so the saved
    weights reload into an identical module). All structural choices come from ``cfg``
    (which is saved verbatim into ``model_hyperparameters``) plus the explicit args."""
    return PureCartesianICTDFix(
        max_embed_radius=cfg.max_radius,
        main_max_radius=cfg.max_radius_main,
        main_number_of_basis=cfg.number_of_basis_main,
        hidden_dim_conv=cfg.channel_in,
        hidden_dim_sh=cfg.get_hidden_dim_sh(),
        hidden_dim=cfg.emb_number_main_2,
        channel_in2=cfg.channel_in2,
        embedding_dim=cfg.embedding_dim,
        max_atomvalue=cfg.max_atomvalue,
        atomic_numbers=atomic_numbers,
        output_size=cfg.output_size,
        embed_size=cfg.embed_size,
        main_hidden_sizes3=cfg.main_hidden_sizes3,
        num_layers=cfg.num_layers,
        num_interaction=num_interaction,
        invariant_channels=invariant_channels,
        function_type_main=cfg.function_type,
        lmax=cfg.lmax,
        ictd_save_tp_mode=ictd_save_tp_mode,
        ictd_fix_route=route,
        ictd_fix_product_backend=product_backend,
        ictd_fix_interaction_attn_heads=attn_heads,
        save_contraction_order=correlation,
        radial_sqrt_num_basis=radial_sqrt_num_basis,
        avg_num_neighbors=avg_num_neighbors,
        internal_compute_dtype=cfg.internal_compute_dtype,
        device=device,
    ).to(device=device, dtype=dtype)


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # data
    ap.add_argument("--data-dir", required=True,
                    help="Directory holding processed_<prefix>.h5 (built by preprocessing).")
    ap.add_argument("--train-prefix", default="train")
    ap.add_argument("--val-prefix", default="val")
    # architecture
    ap.add_argument("--channels", type=int, default=64)
    ap.add_argument("--lmax", type=int, default=2)
    ap.add_argument("--num-interaction", type=int, default=2)
    ap.add_argument("--num-layers", type=int, default=1)
    ap.add_argument("--correlation", type=int, default=2, help="save_contraction_order (body order - 1).")
    ap.add_argument("--route", default="baseline")
    ap.add_argument("--product-backend", default="ictd-pure-u")
    ap.add_argument("--invariant-channels", type=int, default=32)
    ap.add_argument("--ictd-save-tp-mode", default="fully-connected")
    ap.add_argument("--function-type", default="gaussian", choices=["gaussian", "bessel"])
    ap.add_argument("--max-radius", type=float, default=5.0)
    ap.add_argument("--attn-heads", type=int, default=0)
    ap.add_argument("--radial-sqrt-num-basis", action="store_true",
                    help="Use the sqrt(num_basis) radial scale (default OFF = byte-literal MACE radial).")
    ap.add_argument("--avg-num-neighbors", type=float, default=None,
                    help="Override the message normalizer (default: auto-computed from the train H5).")
    # atomic-energy E0 offset
    ap.add_argument("--atomic-energy-keys", default=None, help='e.g. "1,6,7,8"')
    ap.add_argument("--atomic-energy-values", default=None, help='e.g. "-430.5,-821.0,-1488.2,-2044.4"')
    # optimization
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--min-lr", type=float, default=1e-6)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--optimizer", default="adamw", choices=["adamw", "adam"])
    ap.add_argument("--lr-scheduler", default="cosine", choices=["cosine", "step", "none"])
    ap.add_argument("--warmup-batches", type=int, default=0)
    ap.add_argument("--energy-weight", type=float, default=1.0)
    ap.add_argument("--force-weight", type=float, default=10.0)
    ap.add_argument("--stress-weight", type=float, default=0.0,
                    help="Weight of the stress/virial loss (0 = energy+force only; >0 enables it).")
    ap.add_argument("--force-shift-value", type=float, default=1.0)
    ap.add_argument("--max-grad-norm", type=float, default=None)
    ap.add_argument("--no-shuffle", dest="shuffle", action="store_false")
    ap.set_defaults(shuffle=True)
    # make_fx + bucketing
    ap.add_argument("--train-makefx-compile", action="store_true",
                    help="Flatten forward+force-autograd and torch.compile it (faster force-loss training).")
    ap.add_argument("--makefx-buckets", default=None,
                    help='Size-bucket samples for make_fx: an int K (quantile buckets) or a comma '
                         'list of atom-count bounds (e.g. "64,128,256"). One compile per bucket.')
    ap.add_argument("--makefx-max-slots", type=int, default=8)
    ap.add_argument("--pad-nodes-to-max", action="store_true")
    ap.add_argument("--pad-edges-to-max", action="store_true")
    # misc
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float64", choices=["float32", "float64"])
    ap.add_argument("--checkpoint", default="model.pth")
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--log-interval", type=int, default=10)
    return ap


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_arg_parser().parse_args(argv)

    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    device = torch.device(args.device)

    # parse make_fx bucket spec: "6" -> int K, "64,128,256" -> explicit bounds
    makefx_buckets = None
    if args.makefx_buckets:
        mb = str(args.makefx_buckets).strip()
        makefx_buckets = [int(x) for x in mb.split(",") if x.strip()] if "," in mb else int(mb)

    # atomic-energy E0 (None -> trainer's H/C/N/O default)
    aek = aev = None
    if args.atomic_energy_keys and args.atomic_energy_values:
        aek = [int(x) for x in args.atomic_energy_keys.split(",") if x.strip()]
        aev = [float(x) for x in args.atomic_energy_values.split(",") if x.strip()]

    # datasets (bucketing forces fixed-shape padding inside H5Dataset)
    train_ds = H5Dataset(
        prefix=args.train_prefix, data_dir=args.data_dir,
        makefx_buckets=makefx_buckets,
        pad_nodes_to_max=args.pad_nodes_to_max,
        pad_edges_to_max=args.pad_edges_to_max,
    )
    val_h5 = os.path.join(args.data_dir, f"processed_{args.val_prefix}.h5")
    val_ds = H5Dataset(prefix=args.val_prefix, data_dir=args.data_dir) if os.path.exists(val_h5) else None

    # avg_num_neighbors (the message normalizer baked into the weights)
    train_h5 = os.path.join(args.data_dir, f"processed_{args.train_prefix}.h5")
    ann = args.avg_num_neighbors if args.avg_num_neighbors is not None else _avg_num_neighbors_from_h5(train_h5)
    logging.info("avg_num_neighbors = %.4f", ann)

    # model config (everything from_checkpoint reads back is captured here or in extra_hparams)
    cfg = ModelConfig(dtype=dtype)
    cfg.channel_in = args.channels
    cfg.irreps_output_conv_channels = args.channels
    cfg.lmax = args.lmax
    cfg.num_layers = args.num_layers
    cfg.max_radius = args.max_radius
    cfg.max_radius_main = args.max_radius
    cfg.function_type = args.function_type
    cfg.internal_compute_dtype = dtype

    atomic_numbers = aek if aek is not None else [1, 6, 7, 8]
    model = build_baseline_model(
        cfg, avg_num_neighbors=ann, num_interaction=args.num_interaction,
        route=args.route, product_backend=args.product_backend, correlation=args.correlation,
        radial_sqrt_num_basis=args.radial_sqrt_num_basis, attn_heads=args.attn_heads,
        atomic_numbers=atomic_numbers, ictd_save_tp_mode=args.ictd_save_tp_mode,
        invariant_channels=args.invariant_channels, device=device, dtype=dtype,
    )
    n_params = sum(p.numel() for p in model.parameters())
    logging.info("model: %s route=%s channels=%d lmax=%d num_interaction=%d params=%d",
                 type(model).__name__, args.route, args.channels, args.lmax, args.num_interaction, n_params)

    # dataloaders
    if makefx_buckets is not None:
        sampler = BucketBatchSampler(
            train_ds.sample_bucket, batch_size=args.batch_size,
            shuffle=args.shuffle, drop_last=True, seed=0)
        train_loader = DataLoader(train_ds, batch_sampler=sampler,
                                  collate_fn=collate_fn_h5, num_workers=args.num_workers)
        logging.info("bucketing ON: %d buckets, bounds=%s",
                     len(set(train_ds.sample_bucket)), getattr(train_ds, "_bucket_bounds", None))
    else:
        sampler = None
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=args.shuffle,
                                  drop_last=False, collate_fn=collate_fn_h5, num_workers=args.num_workers)
    val_loader = (DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             collate_fn=collate_fn_h5, num_workers=args.num_workers)
                  if val_ds is not None else None)

    # the architecture choices from_checkpoint reads that are NOT ModelConfig fields
    extra_hparams = dict(
        num_interaction=args.num_interaction,
        invariant_channels=args.invariant_channels,
        ictd_fix_route=args.route,
        ictd_fix_product_backend=args.product_backend,
        save_contraction_order=args.correlation,
        ictd_save_tp_mode=args.ictd_save_tp_mode,
        ictd_fix_interaction_attn_heads=args.attn_heads,
        radial_sqrt_num_basis=bool(args.radial_sqrt_num_basis),
        avg_num_neighbors=float(ann),
    )

    trainer = ForceTrainer(
        model, train_loader, val_loader=val_loader, device=device, config=cfg,
        dtype=dtype, max_radius=args.max_radius,
        energy_weight=args.energy_weight, force_weight=args.force_weight,
        stress_weight=args.stress_weight, force_shift_value=args.force_shift_value,
        atomic_energy_keys=aek, atomic_energy_values=aev,
        learning_rate=args.lr, min_learning_rate=args.min_lr, weight_decay=args.weight_decay,
        optimizer_type=args.optimizer, lr_scheduler=args.lr_scheduler,
        warmup_batches=args.warmup_batches, epochs=args.epochs, max_grad_norm=args.max_grad_norm,
        train_makefx_compile=args.train_makefx_compile, makefx_max_slots=args.makefx_max_slots,
        train_sampler=sampler, checkpoint_path=args.checkpoint, log_interval=args.log_interval,
        extra_hparams=extra_hparams,
    )
    best = trainer.fit()
    logging.info("done. best loss = %.6f. checkpoint -> %s", best, args.checkpoint)


if __name__ == "__main__":
    main()
