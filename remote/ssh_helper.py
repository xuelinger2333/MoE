"""SSH helper for 算家云 (Suanjiayun) menu-gated instance.

The gateway shows a Chinese workspace menu first; selecting `0` drops you
into the actual VM shell (~5-15s wait). Once a key is uploaded via
``copy-key`` you can use plain ssh/rsync without paramiko.

Password is read from env var SSH_PWD only — never from CLI args.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import paramiko

HOST = "xn-b.suanjiayun.com"
PORT = 2020
USER = "18851577893"
MENU_CHOICE = "0\r\n"
MENU_WAIT_S = 15.0  # gateway → VM dropdown


def _connect() -> paramiko.SSHClient:
    pwd = os.environ.get("SSH_PWD")
    if not pwd:
        sys.stderr.write("ERROR: set SSH_PWD env var first\n")
        sys.exit(2)
    cli = paramiko.SSHClient()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    cli.connect(HOST, port=PORT, username=USER, password=pwd, timeout=30,
                look_for_keys=False, allow_agent=False)
    return cli


def _open_shell(cli):
    """Open interactive shell, navigate workspace menu, return shell channel."""
    t = cli.get_transport()
    chan = t.open_session()
    chan.get_pty(width=200, height=60, term="xterm-256color")
    chan.invoke_shell()
    # Drain menu banner
    time.sleep(2.0)
    while chan.recv_ready():
        chan.recv(32768)
        time.sleep(0.2)
    # Select workspace 0
    chan.send(MENU_CHOICE)
    time.sleep(MENU_WAIT_S)
    # Drain workspace banner
    while chan.recv_ready():
        chan.recv(32768)
        time.sleep(0.2)
    return chan


def _recv_until_count(chan, marker: str, count: int, timeout: float = 60.0) -> str:
    """Read until ``marker`` has appeared ``count`` times (PTY echoes the command line)."""
    buf = b""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if chan.recv_ready():
            buf += chan.recv(32768)
            try:
                txt = buf.decode("utf-8")
            except UnicodeDecodeError:
                txt = buf.decode("gbk", errors="replace")
            if txt.count(marker) >= count:
                return txt
        else:
            time.sleep(0.3)
    return buf.decode("utf-8", errors="replace")


def run(chan, command: str, timeout: float = 60.0) -> str:
    """Run a command inside the established shell, return only its real output."""
    marker = f"___END_{int(time.time() * 1000)}___"
    # Use printf so the marker text doesn't appear in the command-echo as a literal `echo`.
    full = f"{command}; printf '\\n%s\\n' {marker}\n"
    chan.send(full)
    raw = _recv_until_count(chan, marker, count=2, timeout=timeout)
    # Strip everything up to (and including) the first marker occurrence (the echo),
    # then take what's before the second occurrence.
    after_first = raw.split(marker, 1)[1] if marker in raw else raw
    real = after_first.split(marker, 1)[0]
    return real.strip("\r\n")


def cmd_exec(args) -> int:
    cli = _connect()
    try:
        chan = _open_shell(cli)
        out = run(chan, args.command, timeout=args.timeout)
        sys.stdout.write(out)
        return 0
    finally:
        cli.close()


def cmd_copy_key(args) -> int:
    pub_path = Path(args.pub_key).expanduser()
    if not pub_path.exists():
        sys.stderr.write(f"ERROR: pub key not found at {pub_path}\n")
        return 1
    pub = pub_path.read_text().strip().replace("'", "")
    cli = _connect()
    try:
        chan = _open_shell(cli)
        # Need ~/.ssh writable & put pub key
        prog = (
            "mkdir -p ~/.ssh && chmod 700 ~/.ssh && "
            f"(grep -qxF '{pub}' ~/.ssh/authorized_keys 2>/dev/null || "
            f"echo '{pub}' >> ~/.ssh/authorized_keys) && "
            "chmod 600 ~/.ssh/authorized_keys && echo COPY_KEY_OK"
        )
        out = run(chan, prog, timeout=30)
        sys.stdout.write(out)
        return 0 if "COPY_KEY_OK" in out else 1
    finally:
        cli.close()


def cmd_put(args) -> int:
    """Direct SFTP upload — bypasses the menu shell entirely (auth still required)."""
    cli = _connect()
    try:
        sftp = cli.open_sftp()
        try:
            sftp.put(args.local, args.remote)
            print(f"uploaded {args.local} -> {args.remote}")
            return 0
        finally:
            sftp.close()
    finally:
        cli.close()


def cmd_put_dir(args) -> int:
    """Recursive SFTP upload of a local directory (skips heavy files)."""
    cli = _connect()
    try:
        sftp = cli.open_sftp()
        local = Path(args.local)
        remote = args.remote.rstrip("/")
        skip_ext = {".pyc", ".parquet", ".pt", ".pth", ".bin", ".safetensors"}
        skip_dirs = {"__pycache__", ".git", ".venv", "outputs", "wandb", "node_modules"}
        n = 0
        for path in local.rglob("*"):
            if any(p in skip_dirs for p in path.parts):
                continue
            if path.is_dir():
                continue
            if path.suffix in skip_ext:
                continue
            rel = path.relative_to(local).as_posix()
            rpath = f"{remote}/{rel}"
            try:
                sftp.stat(rpath.rsplit("/", 1)[0])
            except FileNotFoundError:
                # mkdir -p the parent
                parts = rpath.rsplit("/", 1)[0].split("/")
                cur = ""
                for p in parts:
                    if not p:
                        cur = "/"; continue
                    cur = (cur + "/" + p) if cur != "/" else "/" + p
                    try:
                        sftp.mkdir(cur)
                    except IOError:
                        pass
            sftp.put(str(path), rpath)
            n += 1
            if n % 20 == 0:
                print(f"  uploaded {n} files...", flush=True)
        sftp.close()
        print(f"DONE: uploaded {n} files to {remote}")
        return 0
    finally:
        cli.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="op", required=True)

    e = sub.add_parser("exec")
    e.add_argument("command")
    e.add_argument("--timeout", type=int, default=60)
    e.set_defaults(func=cmd_exec)

    k = sub.add_parser("copy-key")
    k.add_argument("--pub_key", default="~/.ssh/id_ed25519.pub")
    k.set_defaults(func=cmd_copy_key)

    p = sub.add_parser("put")
    p.add_argument("local")
    p.add_argument("remote")
    p.set_defaults(func=cmd_put)

    d = sub.add_parser("put-dir")
    d.add_argument("local")
    d.add_argument("remote")
    d.set_defaults(func=cmd_put_dir)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
