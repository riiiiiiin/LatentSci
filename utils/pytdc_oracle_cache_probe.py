import argparse
import os
import sys
import traceback
from pathlib import Path


def _print_kv(key: str, value) -> None:
    print(f"{key}={value}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Probe where PyTDC (tdc.Oracle) downloads oracle *.pkl files. "
            "This reproduces the GRPO mol_opt/jnk3 oracle init path and then prints all *.pkl under ./oracle."
        )
    )
    parser.add_argument(
        "--cwd",
        type=str,
        default=None,
        help=(
            "Working directory to run the probe in (controls where relative path ./oracle resolves). "
            "Default: this script's directory."
        ),
    )
    parser.add_argument(
        "--set-env",
        action="append",
        default=[],
        help=(
            "Set environment variables before importing tdc, format KEY=VALUE. "
            "Example: --set-env TDC_HOME=/path/to/cache"
        ),
    )
    parser.add_argument(
        "--prop",
        type=str,
        default="jnk3",
        help="Oracle property name passed to `mol_opt_evaluater(prop=...)` (default: jnk3).",
    )
    args = parser.parse_args()

    # Apply env overrides before importing tdc / benchmark utils.
    for item in args.set_env:
        if "=" not in item:
            raise ValueError(f"--set-env expects KEY=VALUE, got: {item!r}")
        k, v = item.split("=", 1)
        os.environ[str(k).strip()] = str(v)

    script_dir = Path(__file__).resolve().parent
    cwd = Path(args.cwd).expanduser().resolve() if args.cwd else script_dir
    cwd.mkdir(parents=True, exist_ok=True)
    os.chdir(cwd)

    print("=== Environment ===")
    _print_kv("python", sys.executable)
    _print_kv("version", sys.version.replace("\n", " "))
    _print_kv("cwd", os.getcwd())
    _print_kv("HOME", os.environ.get("HOME"))
    _print_kv("XDG_CACHE_HOME", os.environ.get("XDG_CACHE_HOME"))
    _print_kv("TDC_HOME", os.environ.get("TDC_HOME"))
    _print_kv("TDC_DATA_DIR", os.environ.get("TDC_DATA_DIR"))
    print("")

    # Import tdc
    import tdc  # type: ignore

    print("=== PyTDC ===")
    _print_kv("tdc.__file__", getattr(tdc, "__file__", None))
    _print_kv("tdc.__version__", getattr(tdc, "__version__", None))
    print("")

    # Match our GRPO reward path: reward_utils/ChemCoTBench/core/eval_metric.py
    # TODO:S
    repo_root = script_dir.parent
    reward_utils_dir = repo_root / "code_train_sft" / "reward_utils"
    sys.path.insert(0, str(reward_utils_dir))

    print("=== Probe (mol_opt_evaluater) ===")
    print(f"reward_utils_dir={reward_utils_dir}")
    print(f"prop={args.prop}")
    print(f"expected_oracle_dir={Path('oracle').resolve()}")
    print(f"expected_oracle_pkl_legacy={Path('oracle') / f'{args.prop}.pkl'}")
    print(f"expected_oracle_pkl_current={Path('oracle') / f'{args.prop}_current.pkl'}")
    print("")

    try:
        from ChemCoTBench.core.eval_metric import mol_opt_evaluater  # type: ignore

        _ = mol_opt_evaluater(prop=str(args.prop))
        print("mol_opt_evaluater init: OK")
    except Exception as e:
        print("mol_opt_evaluater init: FAILED (this is expected if sklearn/pickle is incompatible)")
        print(f"exception={type(e).__name__}: {e}")
        traceback.print_exc()

    print("\n=== Files under ./oracle ===")
    oracle_dir = Path("oracle")
    if not oracle_dir.exists():
        print("(no ./oracle directory found)")
        return 0

    pkls = sorted(oracle_dir.rglob("*.pkl"))
    if not pkls:
        print("(no *.pkl found under ./oracle)")
        return 0

    for p in pkls:
        try:
            size = p.stat().st_size
        except Exception:
            size = None
        print(f"- {p.resolve()}  size={size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

