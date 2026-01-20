"""
Prefetch PyTDC (TDC) oracle *.pkl files needed by our GRPO benchmark rewards.

Why
---
Our GRPO correctness reward for mol_opt calls `tdc.Oracle(...)` (via ChemCoTBench's `mol_opt_evaluater`).
PyTDC stores oracle model pickles in a *relative* folder `./oracle/` (relative to the current working directory).
If your training working directory is `Bio-LatentCOT/code_train_sft`, then the required files should live in:

    Bio-LatentCOT/code_train_sft/oracle/

This script downloads (or re-downloads) the required pkls into that folder *before* training,
and can optionally smoke-test that the oracles can be instantiated.

Typical usage (run with the SAME python env as training):
    cd Bio-LatentCOT/code_train_sft
    python ../utils/prefetch_tdc_oracles.py --force --verify
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path
from typing import Iterable


def _sklearn_is_current_variant() -> bool:
    from packaging import version
    import sklearn  # type: ignore

    v = version.parse(getattr(sklearn, "__version__", "0.0.0"))
    return v >= version.parse("0.24.0")


def _download_oracle(name: str, oracle_dir: Path, force: bool) -> Path:
    import tdc  # type: ignore
    from tdc import metadata  # type: ignore
    from tdc.utils import oracle_load  # type: ignore

    if name not in metadata.oracle2type:
        raise ValueError(f"Unknown oracle name for this PyTDC: {name!r}")
    ext = str(metadata.oracle2type[name])
    dst = oracle_dir / f"{name}.{ext}"

    if force and dst.exists():
        dst.unlink()

    # Important: oracle_load's default path is './oracle', but we pass the absolute path to be explicit.
    oracle_load(name, path=str(oracle_dir))

    if not dst.exists():
        raise FileNotFoundError(f"Expected oracle file was not created: {dst}")
    return dst


_TREE_DTYPE_ERR_SUBSTR = "node array from the pickle has an incompatible dtype"


def _load_pickle_tree_dtype_compatible(path: Path) -> object:
    """
    Load a sklearn pickle even if it was created with an older Tree node dtype (missing `missing_go_to_left`).

    This patches `sklearn.tree._tree.Tree` at import-time so unpickling uses a subclass that upgrades the state.
    """
    import pickle
    import numpy as np
    import sklearn.tree._tree as _tree  # type: ignore

    expected_dtype = np.dtype(_tree.NODE_DTYPE)
    orig_tree = _tree.Tree

    class PatchedTree(orig_tree):  # type: ignore[misc, valid-type]
        def __setstate__(self, state):  # type: ignore[override]
            nodes = None
            try:
                nodes = state.get("nodes") if isinstance(state, dict) else None
            except Exception:
                nodes = None

            if isinstance(nodes, np.ndarray) and nodes.dtype.fields is not None:
                names = nodes.dtype.names or ()
                if "missing_go_to_left" not in names and "missing_go_to_left" in expected_dtype.names:
                    upgraded = np.zeros(nodes.shape, dtype=expected_dtype)
                    for name in names:
                        if name in upgraded.dtype.names:
                            upgraded[name] = nodes[name]
                    upgraded["missing_go_to_left"] = 0  # default: route missing to right
                    if isinstance(state, dict):
                        state = dict(state)
                        state["nodes"] = upgraded

            return super().__setstate__(state)

    _tree.Tree = PatchedTree  # type: ignore[assignment]
    try:
        with path.open("rb") as f:
            return pickle.load(f)
    finally:
        _tree.Tree = orig_tree  # type: ignore[assignment]


def _maybe_resave_incompatible_sklearn_pickle(path: Path) -> bool:
    """
    If `path` is a sklearn pickle failing due to Tree node dtype mismatch, load it with a compatibility patch and
    rewrite it (in place) so that normal sklearn unpickling works.

    Returns:
        True if the file was rewritten, else False.
    """
    import pickle

    try:
        with path.open("rb") as f:
            pickle.load(f)
        return False
    except ValueError as e:
        if _TREE_DTYPE_ERR_SUBSTR not in str(e):
            raise

    obj = _load_pickle_tree_dtype_compatible(path)

    backup = path.with_suffix(path.suffix + ".bak")
    if not backup.exists():
        backup.write_bytes(path.read_bytes())
        print(f"[info] backup created: {backup}")

    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as f:
        pickle.dump(obj, f, protocol=4)
    os.replace(tmp, path)

    # Verify can be loaded normally now
    with path.open("rb") as f:
        pickle.load(f)
    return True


def _ensure_oracle_dir(code_train_sft_dir: Path) -> Path:
    oracle_dir = code_train_sft_dir / "oracle"
    oracle_dir.mkdir(parents=True, exist_ok=True)
    return oracle_dir


def _iter_required_names(include_cyp: bool) -> list[str]:
    names = ["fpscores", "drd2", "gsk3b", "jnk3"]
    if include_cyp:
        names.append("cyp3a4_veith")
    return names


def _resolve_download_names(base_names: Iterable[str]) -> list[str]:
    use_current = _sklearn_is_current_variant()
    out: list[str] = []
    for base in base_names:
        if base in {"drd2", "gsk3b", "jnk3"} and use_current:
            out.append(f"{base}_current")
        else:
            out.append(base)
    return out


def _verify_oracles(smiles: str) -> None:
    import tdc  # type: ignore

    # These will load the pkls from ./oracle/*.pkl (relative to cwd).
    for name in ["drd2", "gsk3b", "jnk3"]:
        o = tdc.Oracle(name=name)
        _ = o(smiles)

    # fpscores.pkl is used by SA score; call the loader directly to ensure it's readable.
    from tdc.chem_utils.oracle.oracle import readFragmentScores  # type: ignore

    readFragmentScores()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--code-train-sft-dir",
        type=str,
        default=None,
        help="Path to `Bio-LatentCOT/code_train_sft`. Default: current working directory.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete existing oracle pkls first to force re-download.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Instantiate key oracles after download to ensure pkls are loadable.",
    )
    parser.add_argument(
        "--resave-incompatible",
        type=lambda x: (str(x).lower() == "true"),
        default=True,
        help=(
            "If true, automatically rewrite sklearn oracle pkls that fail to load due to Tree dtype mismatch "
            "(e.g. missing_go_to_left). Default: true."
        ),
    )
    parser.add_argument(
        "--include-cyp",
        action="store_true",
        help="Also download `cyp3a4_veith` oracle (not used by our current ChemCoTBench rewards).",
    )
    parser.add_argument(
        "--smiles",
        type=str,
        default="CC",
        help="SMILES used for verification calls (default: CC).",
    )
    args = parser.parse_args()

    code_train_sft_dir = (
        Path(args.code_train_sft_dir).expanduser().resolve() if args.code_train_sft_dir else Path.cwd().resolve()
    )
    oracle_dir = _ensure_oracle_dir(code_train_sft_dir)

    # Ensure we download into the same relative location the training code will use.
    os.chdir(code_train_sft_dir)

    base_names = _iter_required_names(include_cyp=bool(args.include_cyp))
    names = _resolve_download_names(base_names)

    print(f"[info] python={sys.executable}")
    print(f"[info] cwd={Path.cwd()}")
    print(f"[info] oracle_dir={oracle_dir}")
    print(f"[info] force={bool(args.force)} verify={bool(args.verify)}")
    print(f"[info] will_download={names}")

    created: list[Path] = []
    for n in names:
        p = _download_oracle(n, oracle_dir=oracle_dir, force=bool(args.force))
        created.append(p)
        print(f"[ok] {p}")

    if bool(args.resave_incompatible):
        rewritten = 0
        for p in created:
            if p.suffix.lower() != ".pkl":
                continue
            try:
                if _maybe_resave_incompatible_sklearn_pickle(p):
                    rewritten += 1
                    print(f"[ok] resaved sklearn oracle pickle: {p}")
            except Exception as e:
                # Only surface as warning here; `--verify` will still catch runtime issues. This helps keep the script
                # usable even when some pkls are not sklearn models (e.g. fpscores).
                print(f"[warn] failed to resave {p}: {type(e).__name__}: {e}")
        if rewritten:
            print(f"[info] resaved {rewritten} incompatible oracle pkls")

    if bool(args.verify):
        print("[info] verifying oracles...")
        try:
            _verify_oracles(smiles=str(args.smiles))
        except Exception as e:
            print("[error] oracle verification failed")
            print(f"exception={type(e).__name__}: {e}")
            traceback.print_exc()
            return 2
        print("[ok] verification passed")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
