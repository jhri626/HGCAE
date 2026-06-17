# Repository Guidelines

## Project Structure & Module Organization

This PyTorch repository implements HGCAE for link prediction, node clustering, and image clustering. Main entry points are `train_solver.py`, `config.py`, and `solver.py`.

- `models/`: encoder, decoder, and base models.
- `layers/`: Euclidean, hyperbolic, and attention layers.
- `manifolds/`: manifold implementations such as `PoincareBall` and `Euclidean`.
- `optimizers/`: custom optimizers, including Riemannian Adam support.
- `utils/`: data, math, evaluation, and training helpers.
- `script/`: dataset-specific shell commands for reproducible experiments.
- `data/`: bundled datasets and feature files used by the scripts.
- `images/`: figures used by the README.
- `hyperbolic_representation_learning/`: Git submodule for point-based hyperbolic representation learning.

## Build, Test, and Development Commands

Use the Docker image from `README.md`:

```bash
docker run --gpus all -it --rm --shm-size 100G -v $PWD:/workspace junhocho/hyperbolicgraphnn:8 bash
```

Inside the container, run a script:

```bash
sh script/train_cora_lp.sh      # Cora link prediction
sh script/train_cora_nc.sh      # Cora node clustering
sh script/train_ImageNet10.sh   # Image clustering
```

For manual runs, set `DATAPATH` first:

```bash
export HGCAE_HOME=$(pwd)
export DATAPATH="$HGCAE_HOME/data"
python train_solver.py --dataset cora --model HGCAE --manifold PoincareBall --cuda 0
```

After cloning, initialize the submodule:

```bash
git submodule update --init --recursive
```

## Coding Style & Naming Conventions

Follow the existing Python style: 4-space indentation, snake_case functions and variables, CamelCase classes, and grouped imports. Keep CLI flags in `config.py` hyphenated; `argparse` converts them to underscore attributes. Preserve tensor shapes, logging patterns, and dataset names.

## Testing Guidelines

There is no dedicated test suite. Validate changes with the smallest relevant script, usually `sh script/train_cora_lp.sh` or `sh script/train_cora_nc.sh`. For data loading or metric changes, also run one image or larger graph script when GPU memory allows. Record command, dataset, seed, and metrics.

## Commit & Pull Request Guidelines

Recent history uses short messages such as `Update README.md` and `citation added.` Keep commits focused. Pull requests should describe motivation, commands run, changed metrics, and related issues or papers. Include screenshots only when README figures or visual outputs change.

## Submodule Guidelines

The `hyperbolic_representation_learning` directory is pinned by the parent repository. Do not edit it during ordinary HGCAE changes unless the PR updates that submodule. To update it, commit inside the submodule first, then commit the changed pointer and `.gitmodules` updates here.

## Security & Configuration Tips

Do not commit caches, checkpoints, logs, or local environment files. Keep large new datasets out of Git unless required for reproducibility and documented in `README.md`.
