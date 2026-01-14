"""
Re-save PyTDC (TDC) oracle sklearn pickles to be compatible with a newer scikit-learn.

Background
----------
Some PyTDC oracle models (e.g. jnk3/gsk3b/drd2) are shipped as sklearn pickles under `./oracle/*.pkl`.
When scikit-learn's internal tree node dtype changes (e.g. adding `missing_go_to_left`),
loading an older pickle in a newer sklearn may fail with:

    ValueError: node array from the pickle has an incompatible dtype ...

This script loads the existing pickle *in the newer sklearn* by patching the unpickling path to
auto-upgrade the tree node dtype, then re-saves the model back to disk.

Usage (run in the SAME python env as your training, i.e. the "high" sklearn version):

    python Bio-LatentCOT/utils/resave_tdc_oracle_pkls.py --oracle-dir Bio-LatentCOT/code_train_sft/oracle --inplace

By default it targets:
    drd2_current.pkl, gsk3b_current.pkl, jnk3_current.pkl
"""

from __future__ import annotations

import argparse
import os
import pickle
import shutil
from pathlib import Path
from typing import Any, Iterable


def _patch_sklearn_tree_unpickling() -> tuple[Any, Any]:
    """
    Patch `sklearn.tree._tree.Tree` so unpickling older node dtypes can succeed on newer sklearn.

    Returns:
        (module, original_tree_class) so caller can restore after loading.
    """
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
                    # Default: route missing values to the right (0). This matches sklearn's default.
                    upgraded["missing_go_to_left"] = 0
                    if isinstance(state, dict):
                        state = dict(state)
                        state["nodes"] = upgraded

            return super().__setstate__(state)

    _tree.Tree = PatchedTree  # type: ignore[assignment]
    return _tree, orig_tree


def _restore_sklearn_tree(module: Any, orig_tree: Any) -> None:
    try:
        module.Tree = orig_tree
    except Exception:
        pass


def _load_with_patch(path: Path) -> object:
    _tree_mod, orig_tree = _patch_sklearn_tree_unpickling()
    try:
        with path.open("rb") as f:
            return pickle.load(f)
    finally:
        _restore_sklearn_tree(_tree_mod, orig_tree)


def _verify_load(path: Path) -> None:
    # Verify output can be loaded without any patch (normal sklearn path).
    with path.open("rb") as f:
        _ = pickle.load(f)


def _iter_targets(oracle_dir: Path, filenames: Iterable[str]) -> list[Path]:
    out: list[Path] = []
    for name in filenames:
        p = oracle_dir / name
        if p.exists():
            out.append(p)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--oracle-dir",
        type=str,
        default="oracle",
        help="Directory that contains the oracle *.pkl files (default: ./oracle).",
    )
    parser.add_argument(
        "--files",
        type=str,
        nargs="*",
        default=["drd2_current.pkl", "gsk3b_current.pkl", "jnk3_current.pkl"],
        help="Which pickle filenames to resave (default: the three *_current.pkl files).",
    )
    parser.add_argument(
        "--inplace",
        action="store_true",
        help="Overwrite the original file (creates a .bak backup if absent).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="If set, write converted pkls to this directory instead of overwriting.",
    )
    args = parser.parse_args()

    oracle_dir = Path(args.oracle_dir).expanduser().resolve()
    if not oracle_dir.exists():
        raise FileNotFoundError(f"oracle-dir not found: {oracle_dir}")

    targets = _iter_targets(oracle_dir, args.files)
    if not targets:
        raise FileNotFoundError(f"No target pkls found under {oracle_dir} for files={args.files!r}")

    out_dir: Path | None = None
    if args.out_dir is not None:
        out_dir = Path(args.out_dir).expanduser().resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[info] oracle_dir={oracle_dir}")
    print(f"[info] files={', '.join([p.name for p in targets])}")
    print(f"[info] inplace={bool(args.inplace)} out_dir={out_dir}")

    for src in targets:
        print(f"\n[info] processing: {src}")
        obj = _load_with_patch(src)

        if out_dir is not None:
            dst = out_dir / src.name
        elif args.inplace:
            dst = src
        else:
            dst = src.with_name(src.stem + ".resaved.pkl")

        tmp = dst.with_suffix(dst.suffix + ".tmp")

        if dst == src and args.inplace:
            backup = src.with_suffix(src.suffix + ".bak")
            if not backup.exists():
                shutil.copy2(src, backup)
                print(f"[info] backup created: {backup}")

        with tmp.open("wb") as f:
            pickle.dump(obj, f, protocol=4)
        os.replace(tmp, dst)
        print(f"[info] wrote: {dst}")

        try:
            _verify_load(dst)
            print("[info] verify: OK (loads without patch)")
        except Exception as e:
            print(f"[warn] verify: FAILED ({type(e).__name__}: {e})")
            print("       The file was written, but sklearn may still not be able to load it in normal mode.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

