#!/usr/bin/env python
r"""TJ1 补实验批量 runner: 按序 A1b → A1c → A2a → A2b, 进度追加写 ..\TJ1_batch_progress.md.

单实验崩溃不拖垮整批 (逐段 try/except); --stage 可单跑一段。已有 run 由各启动器
自行跳过 (ensure 检查产物), 因此中断后重跑同一条命令即可续批。

用法:
  python scripts/tj_batch_runner.py                # 全批按序
  python scripts/tj_batch_runner.py --stage a1b    # 只跑一段
"""
from __future__ import annotations

import argparse
import contextlib
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(CODE / "scripts"))

PROGRESS = CODE.parent / "TJ1_batch_progress.md"


def _append(lines: list[str]) -> None:
    with PROGRESS.open("a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


@contextlib.contextmanager
def _tee_log(out_root: Path, stage: str):
    """把启动器的 stdout 同步落一份到 results/<stage 目录>/batch_stdout.log."""
    out_root.mkdir(parents=True, exist_ok=True)
    fh = (out_root / "batch_stdout.log").open("a", encoding="utf-8")

    class _Tee:
        def write(self, s):
            sys.__stdout__.write(s)
            fh.write(s)

        def flush(self):
            sys.__stdout__.flush()
            fh.flush()

    sys.__stdout__.flush()
    old = sys.stdout
    sys.stdout = _Tee()
    try:
        yield
    finally:
        sys.stdout = old
        fh.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="TJ1 补实验批量 runner")
    ap.add_argument("--stage", choices=["a1b", "a1c", "a2a", "a2b"], default=None,
                    help="只跑一段; 缺省按 a1b→a1c→a2a→a2b 全跑")
    args = ap.parse_args()

    import tj_run_a1b_long
    import tj_run_a1c_hardcut
    import tj_run_a2a_freeze
    import tj_run_a2b_narrow

    all_stages = [
        ("a1b", tj_run_a1b_long),
        ("a1c", tj_run_a1c_hardcut),
        ("a2a", tj_run_a2a_freeze),
        ("a2b", tj_run_a2b_narrow),
    ]
    stages = all_stages if args.stage is None else [
        (n, m) for n, m in all_stages if n == args.stage
    ]

    _append([
        "",
        f"## 批启动 {datetime.now():%Y-%m-%d %H:%M:%S}",
        f"范围: {'全批 a1b→a1c→a2a→a2b' if args.stage is None else args.stage}",
    ])
    print(f"[batch] 进度文件: {PROGRESS}", flush=True)

    failed = []
    for name, mod in stages:
        _append([f"- [{datetime.now():%Y-%m-%d %H:%M:%S}] {name} 开始"])
        t0 = time.time()
        try:
            with _tee_log(Path(mod.OUT), name):
                res = mod.main()
            wall = round((time.time() - t0) / 60, 1)
            _append([
                f"- [{datetime.now():%Y-%m-%d %H:%M:%S}] {name} 完成: "
                f"新跑 {len(res['done'])} 个 ({', '.join(res['done']) or '-'}), "
                f"跳过 {len(res['skipped'])} 个 ({', '.join(res['skipped']) or '-'}), "
                f"wall {wall} min, verdict = {res['verdict']}",
            ])
            print(f"[batch] {name} 完成, verdict = {res['verdict']}", flush=True)
        except Exception:
            failed.append(name)
            _append([
                f"- [{datetime.now():%Y-%m-%d %H:%M:%S}] {name} 崩溃 (批继续):",
                "```",
                traceback.format_exc(limit=8).rstrip(),
                "```",
            ])
            print(f"[batch] {name} 崩溃, 继续下一段 (详见进度文件)", file=sys.stderr)

    tail = f"失败段: {', '.join(failed)}" if failed else "全部段正常返回"
    _append([f"- [{datetime.now():%Y-%m-%d %H:%M:%S}] 批结束 —— {tail}"])
    print(f"[batch] 批结束 —— {tail}", flush=True)


if __name__ == "__main__":
    main()
