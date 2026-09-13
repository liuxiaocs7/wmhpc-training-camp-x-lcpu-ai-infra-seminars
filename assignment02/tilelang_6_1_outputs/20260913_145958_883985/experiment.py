#!/usr/bin/env python3
"""Export TileLang GEMM lowering and cross-compile; never launch a GPU kernel.

Target construction and scoping were checked with TileLang 0.1.13 / TVM.
Full CUDA lowering and NVCC compilation still require the Linux environment.
"""

import argparse
from datetime import datetime
import json
from pathlib import Path
import platform
import re
import shlex
import shutil
import subprocess
import sys
import traceback


def make_gemm(T):
    # Identical program and tuning parameters for both targets.
    M = N = K = 1024
    BM = BN = 128
    BK = 64

    @T.prim_func
    def gemm(
        A: T.Tensor((M, K), "bfloat16"),
        B: T.Tensor((K, N), "bfloat16"),
        C: T.Tensor((M, N), "float32"),
    ):
        with T.Kernel(T.ceildiv(N, BN), T.ceildiv(M, BM), threads=128) as (bx, by):
            A_shared = T.alloc_shared((BM, BK), "bfloat16")
            B_shared = T.alloc_shared((BK, BN), "bfloat16")
            C_local = T.alloc_fragment((BM, BN), "float32")
            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, BK), num_stages=3):
                T.copy(A[by * BM, k * BK], A_shared)
                T.copy(B[k * BK, bx * BN], B_shared)
                T.gemm(A_shared, B_shared, C_local)
            T.copy(C_local, C[by * BM, bx * BN])

    return gemm


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def compile_cuda(nvcc, arch, directory):
    # NVCC still accepts -arch=sm_XX; TVM Target uses a config dict below.
    compiler = nvcc.get_nvcc_compiler()
    options = nvcc.default_compile_options()
    for output_format in ("ptx", "cubin"):
        destination = directory / f"kernel.{output_format}"
        command = [
            compiler, f"--{output_format}", "-O3", f"-arch={arch}",
            *options, str(directory / "kernel.cu"), "-o", str(destination),
        ]
        result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        (directory / f"nvcc_{output_format}.log").write_text(
            shlex.join(command) + "\n\n" + result.stdout
            + f"\nexit_code={result.returncode}\n", encoding="utf-8",
        )
        if result.returncode != 0:
            raise RuntimeError(f"NVCC failed: inspect {directory / f'nvcc_{output_format}.log'}")
        if not destination.is_file() or destination.stat().st_size == 0:
            raise RuntimeError(f"NVCC produced no usable {output_format} file")


def collect_evidence(directory):
    # Search hints, not an automatic conclusion about the selected instruction.
    pattern = re.compile(
        r"wgmma|tcgen05|tcgen5|mma\.sync|gemm_ss|gemm_rs|ldmatrix|"
        r"cp\.async|tma|tensor.?map|descriptor|desc_|swizzle|mbarrier|"
        r"shared\.tmem|tmem|fence\.proxy", re.IGNORECASE,
    )
    matches = []
    for name in ("kernel.cu", "kernel.ptx", "host_ir.txt", "device_ir.txt"):
        path = directory / name
        if path.exists():
            for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if pattern.search(line):
                    matches.append(f"{name}:{number}: {line}")
    (directory / "evidence.txt").write_text("\n".join(matches) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, help="New output directory; existing directories are refused")
    parser.add_argument(
        "--source-only", action="store_true",
        help="Only export CUDA and lowering IR; does NOT verify NVCC compilation",
    )
    args = parser.parse_args()

    try:
        import tilelang
        import tilelang.language as T
        from tilelang import tvm
        from tilelang.engine import lower
        from tilelang.contrib import nvcc
    except ImportError as error:
        parser.exit(2, f"Missing TileLang dependency: {error}\nRun in the course's TileLang/CUDA environment.\n")

    root = (args.out or Path("tilelang_6_1_outputs") / datetime.now().strftime("%Y%m%d_%H%M%S_%f")).resolve()
    root.mkdir(parents=True, exist_ok=False)
    shutil.copy2(Path(__file__).resolve(), root / "experiment.py")
    metadata = {
        "platform": platform.platform(), "python": sys.version,
        "tilelang_version": getattr(tilelang, "__version__", "unknown"),
        "tilelang_path": str(tilelang.__file__),
        "shape_MNK": [1024, 1024, 1024], "tile_MNK": [128, 128, 64],
        "input_dtype": "bfloat16", "accumulator_and_output_dtype": "float32",
        "accumulator_scope": "local.fragment", "threads": 128, "num_stages": 3,
        "source_only": args.source_only, "kernel_launched": False,
        "targets": {},
    }
    if not args.source_only:
        try:
            compiler = nvcc.get_nvcc_compiler()
            version = subprocess.run(
                [compiler, "--version"], text=True,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True,
            )
            metadata["nvcc"] = compiler
            metadata["nvcc_version"] = version.stdout
        except Exception:
            metadata["nvcc_discovery_error"] = traceback.format_exc()
    save_json(root / "environment.json", metadata)

    @tvm.instrument.pass_instrument
    class DumpPasses:
        def __init__(self, directory):
            self.directory = directory
            self.index = 0

        def run_after_pass(self, mod, info):
            name = re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(info.name))
            path = self.directory / f"{self.index:03d}_{name}.txt"
            path.write_text(mod.script(), encoding="utf-8")
            self.index += 1

    failed = False
    for arch in ("sm_90a", "sm_100a"):
        directory = root / arch
        passes = directory / "passes"
        passes.mkdir(parents=True)
        status = {"target": {"kind": "cuda", "arch": arch}, "lowering_ok": False, "nvcc_ok": False}
        metadata["targets"][arch] = status
        try:
            prim_func = make_gemm(T)
            (directory / "input_ir.txt").write_text(prim_func.script(), encoding="utf-8")
            target = tvm.target.Target({"kind": "cuda", "arch": arch})
            # lower(target=...) does not establish Target.current() in 0.1.13.
            # The vectorizer used by T.clear needs an active target scope.
            with tvm.transform.PassContext(opt_level=3, instruments=[DumpPasses(passes)]), target:
                artifact = lower(
                    prim_func,
                    target=target,
                    enable_host_codegen=False,
                    enable_device_compile=False,
                )
            (directory / "kernel.cu").write_text(artifact.kernel_source, encoding="utf-8")
            (directory / "host_ir.txt").write_text(artifact.host_mod.script(), encoding="utf-8")
            (directory / "device_ir.txt").write_text(artifact.device_mod.script(), encoding="utf-8")
            status["lowering_ok"] = True
            if not args.source_only:
                compile_cuda(nvcc, arch, directory)
                status["nvcc_ok"] = True
            print(f"{arch}: {'CUDA/IR exported only' if args.source_only else 'CUDA/IR/PTX/CUBIN complete'}", flush=True)
        except Exception:
            failed = True
            status["error"] = traceback.format_exc()
            (directory / "error.txt").write_text(status["error"], encoding="utf-8")
            print(f"{arch}: FAILED; see {directory / 'error.txt'}", file=sys.stderr, flush=True)
        finally:
            collect_evidence(directory)
            save_json(root / "environment.json", metadata)

    print(f"Output: {root}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
