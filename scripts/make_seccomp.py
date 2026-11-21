"""Write a classic-BPF seccomp filter that makes socket()/socketpair() with AF_VSOCK fail with EPERM.

Usage: python3 make_seccomp.py OUT.bpf   (bubblewrap loads it with `--seccomp FD`).
Why: before Linux 7.0, AF_VSOCK sockets were not isolated per network namespace, and WSL2's host services use
Hyper-V sockets, so `bwrap --unshare-net` alone may not close that path. Hand-assembled to avoid needing libseccomp.
Sources: struct seccomp_data and return values — Linux seccomp(2) man page
(https://man7.org/linux/man-pages/man2/seccomp.2.html); classic BPF opcodes — linux/filter.h / bpf_common.h;
x86_64 syscall numbers socket = 41, socketpair = 53 (arch/x86/entry/syscalls/syscall_64.tbl); AF_VSOCK = 40;
AUDIT_ARCH_X86_64 = 0xC000003E; vsock/netns note — https://stefano-garzarella.github.io/posts/2026-02-11-vsock-netns/.
"""

import struct
import sys

LD_W_ABS = 0x20  # BPF_LD | BPF_W | BPF_ABS: A = *(u32 *)(seccomp_data + k)
JEQ_K = 0x15  # BPF_JMP | BPF_JEQ | BPF_K: if A == k goto jt else goto jf (offsets from next instruction)
RET_K = 0x06  # BPF_RET | BPF_K: return k
ALLOW = 0x7FFF0000  # SECCOMP_RET_ALLOW
ERRNO_EPERM = 0x00050000 | 1  # SECCOMP_RET_ERRNO | EPERM

PROGRAM = [
    (LD_W_ABS, 0, 0, 4),  # 0: A = arch
    (JEQ_K, 0, 6, 0xC000003E),  # 1: not x86_64 → allow (8)
    (LD_W_ABS, 0, 0, 0),  # 2: A = syscall number
    (JEQ_K, 1, 0, 41),  # 3: socket → check family (5)
    (JEQ_K, 0, 3, 53),  # 4: socketpair → check family (5), else allow (8)
    (LD_W_ABS, 0, 0, 16),  # 5: A = low 32 bits of args[0] (address family)
    (JEQ_K, 0, 1, 40),  # 6: AF_VSOCK → deny (7), else allow (8)
    (RET_K, 0, 0, ERRNO_EPERM),  # 7
    (RET_K, 0, 0, ALLOW),  # 8
]

if __name__ == "__main__":
    with open(sys.argv[1], "wb") as f:
        for code, jt, jf, k in PROGRAM:
            f.write(struct.pack("<HBBI", code, jt, jf, k))  # struct sock_filter {u16 code; u8 jt; u8 jf; u32 k}
