# GPU Workstation Setup (NVIDIA + AI/DL + Robotics on Ubuntu LTS)

Install guide and utilities for Linux workstations with NVIDIA GPUs, targeting AI/DL/NN
workloads (TensorFlow, PyTorch) and robotics environments (ROS 2, Isaac ROS/Gym, Omniverse).

Targets Linux/UNIX only — Ubuntu LTS (22.04 / 24.04 / 26.04) unless stated otherwise.

> Version numbers below were verified against NVIDIA's documentation in **November 2026**.
> Always double-check the linked NVIDIA pages before installing — releases move fast.

---

## ⚠ Version ceilings for Pascal & Volta GPUs

If your workstation has a **Volta** card (TITAN V, compute capability 7.0) or **Pascal**
(GTX 10-series / TITAN Xp, compute capability 6.0/6.1) — this repo's workstation has a
TITAN V alongside an RTX 3080 Ti — the current stack has hard version ceilings. **Do not
blindly upgrade to the latest versions**: CUDA 13, TensorRT 10/11, and post-R580 drivers
all dropped these architectures.

| Component | Ceiling for Pascal/Volta | Notes |
|-----------|--------------------------|-------|
| NVIDIA driver | **R580 branch is the last** | The Linux 580.* series is the last to support Maxwell, Pascal, and Volta. It receives critical security updates only, through **October 2028**. Newer branches (R590+) support Turing and newer only — the TITAN V is explicitly on the affected list. |
| CUDA Toolkit | **CUDA 12.x (12.9.2 newest)** | CUDA 13.0 removed offline compilation and library support for Pascal/Volta ("CUDA 13.0 supports all NVIDIA architectures from Turing through Grace Blackwell"). The 12.x nvcc still compiles `sm_60`/`sm_61`/`sm_70`; 13.x does not. |
| cuDNN | **Turing (7.5) is the official minimum in cuDNN 9.x** | The cuDNN 9.28 support matrix lists only Turing and newer. In practice the cuDNN 9.x *CUDA 12* wheels still run convolutions on Volta (verified on the TITAN V of this workstation with cuDNN 9.10/CUDA 12.8) — but NVIDIA does not guarantee it, so pin a version you have tested. |
| TensorRT | **TensorRT 8.6.1 (legacy)** | TensorRT 10.x and 11.x require **SM 7.5+ (Turing)**. The entire 10.x line never supported Volta/Pascal. On these GPUs use TensorRT 8.6.1 or skip TensorRT entirely. |
| PyTorch | **CUDA 12.x wheels only (`cu126`/`cu128`)** | Official CUDA 12.x wheels ship `sm_50`/`sm_60`/`sm_70` kernels (verified: `torch 2.14.0+cu126` → `['sm_50', 'sm_60', 'sm_70', 'sm_75', 'sm_80', 'sm_86', 'sm_90']`). CUDA 13 wheels dropped `sm_50/60/70`. Never install PyTorch from a `cu130+` index on these GPUs. |
| TensorFlow | **`[and-cuda]` wheels (CUDA 12.x builds)** | Official docs list compute capabilities 6.0/7.0 as supported; the wheels are built with CUDA 12.x + cuDNN 9.x, matching the ceilings above. |

**Recommended legacy stack (as running on this workstation):** driver 580.x, CUDA 12.8–12.9,
`torch` from the `whl/cu126`/`cu128` index, `tensorflow[and-cuda]`, and no TensorRT on the
Volta card (or TensorRT 8.6.1 if needed).

> **Mixed-GPU note:** with an Ampere card (RTX 30xx) alongside a Volta/Pascal card, CUDA 12.x
> serves both. Moving the system to CUDA 13 would drop the legacy card entirely — the newer
> card's extra features are not worth losing the TITAN V.

Sources: [CUDA 13.0 release notes (architecture removal)](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-toolkit-release-notes/index.html) ·
[NVIDIA Unix driver EOL notice (a_id/3142)](https://nvidia.custhelp.com/app/answers/detail/a_id/3142) ·
[NVIDIA driver support matrix](https://docs.nvidia.com/datacenter/tesla/drivers/cuda-toolkit-driver-and-architecture-matrix.html) ·
[cuDNN support matrix](https://docs.nvidia.com/deeplearning/cudnn/backend/latest/reference/support-matrix.html) ·
[TensorRT support matrix](https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/support-matrix.html)

---

## Repository contents

| File | Purpose |
|------|---------|
| `utils/gpu-fans-max.py` | Set every controllable NVIDIA GPU fan to a fixed speed (default 100%). |
| `utils/gpu-fans-auto.py` | Return all GPU fans to the driver's automatic control. |
| `wipeNVStuffs.sh` | Complete removal of NVIDIA drivers, CUDA, cuDNN, TensorRT and Secure Boot MOK keys (Ubuntu 24.04). |

### GPU fan control

Both scripts use NVML (`pynvml`) and require root. Install the bindings first — directly on
the system, no venv:

```bash
sudo pip3 install --break-system-packages nvidia-ml-py
```

If you keep this repo at `~/GPU-Workstation-Setup`, symlink the utils so the aliases work:

```bash
ln -s ~/GPU-Workstation-Setup/utils ~/utils
```

Add these aliases to your `~/.bashrc`:

```bash
alias nvFanMax='sudo python3 ~/utils/gpu-fans-max.py'
alias nvFanAuto='sudo python3 ~/utils/gpu-fans-auto.py'
alias nvFanSet='sudo python3 ~/utils/gpu-fans-max.py --speed'   # e.g. nvFanSet 70
```

- `nvFanMax` — sets all fans to 100% (speed persists after the script exits).
- `nvFanSet <n>` — set a specific speed (0–100).
- `nvFanAuto` — hand control back to the driver. Run this to undo manual speeds.

Note: on cards without controllable fans (many data-center/blower cards report none), the
scripts print a warning and skip them. Consumer cards (e.g. RTX series) work as expected.

### NVIDIA wipe-out (`wipeNVStuffs.sh`)

Removes every NVIDIA package, repository, config, `libcuda*` library, Secure Boot MOK keys,
and leftover `/usr/local/cuda*` directories, then rebuilds initramfs and GRUB. Useful before a
clean driver/CUDA reinstall. Logs to `/var/log/nvidia-wipeout-<timestamp>.log`.

```bash
sudo bash wipeNVStuffs.sh
```

**Read it before running** — it is destructive by design and assumes Ubuntu 24.04. After it
finishes: reboot, confirm MOK key deletion in the MOK Manager if prompted, then reinstall
the driver and toolkit, and re-enroll your MOK keys if Secure Boot is enabled:

```bash
ls -la /var/lib/shim-signed/mok/
sudo mokutil --import /var/lib/shim-signed/mok/MOK.der   # prompts enrollment at next boot
```

---

## Fresh install guide

### 1) Blacklist nouveau

The open-source nouveau driver conflicts with the proprietary NVIDIA driver. Create
`/etc/modprobe.d/blacklist-nouveau.conf` (as root):

```
blacklist nouveau
options nouveau modeset=0
```

Then rebuild the initramfs and reboot:

```bash
sudo update-initramfs -u
sudo reboot
```

### 2) Install the NVIDIA driver

Check the latest branch at <https://www.nvidia.com/en-us/drivers/> — **620.70.06** (R620)
was the latest stable Linux driver at the time of writing. On Ubuntu the easiest route:

```bash
sudo apt-get install nvidia-driver-620   # replace with the branch available/latest for your release
```

or use `sudo ubuntu-drivers autoinstall`. Reboot and verify:

```bash
nvidia-smi
```

**Important:** since CUDA 13.4 the Linux CUDA Toolkit no longer bundles a driver — install
the driver separately as above. CUDA 13.x binaries run on any driver ≥ 580 via minor-version
compatibility; the newest CUDA 13.5 features need an R620+ driver.

> **Pascal/Volta:** R580 is the **last** driver branch for these GPUs — stay on it (it gets
> security fixes through Oct 2028). See the legacy GPU ceilings section above.

> **Framework compatibility:** if you build TensorFlow/PyTorch from source, check which
> driver/CUDA version your target framework expects first — mismatched versions cause build
> failures or runtime issues. For plain Python usage, the pip wheels below bundle their own
> CUDA dependencies and only need a working driver.

### 3) CUDA Toolkit

Latest: **CUDA 13.5.0** (<https://developer.nvidia.com/cuda-downloads>). Install the network
repo (replace `ubuntu2404` with `ubuntu2204`/`ubuntu2604` to match your release):

> **Pascal/Volta:** CUDA 13 dropped these architectures. Use **CUDA 12.9.x** (or 12.8)
> instead — same keyring method, then `apt-get -y install cuda-toolkit-12.9`. When building
> for these GPUs pass the arch explicitly, e.g.
> `nvcc -arch=sm_70` (TITAN V) or `-arch=sm_61` (GTX 10-series).

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit
```

Add it to your `PATH` in `~/.bashrc` (very important — adjust the version to what you
installed):

```bash
export PATH=/usr/local/cuda-13.5/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-13.5/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
```

```bash
source ~/.bashrc
nvcc --version   # must print the toolkit version
```

### 4) cuDNN

Latest: **cuDNN 9.28.1** (<https://developer.nvidia.com/cudnn-downloads>, NVIDIA developer
login required). Use the **cuDNN 9.28.1 for CUDA 13.x** package (built against CUDA 13.5,
needs Linux driver ≥ 615.71.09) — or the CUDA 12.x package if you stayed on a CUDA 12
toolkit. Pick the `.deb` for your Ubuntu release on the downloads page and follow the
commands it shows (the `dpkg -i` local-repo + keyring + `apt-get install cudnn` pattern).

> **Pascal/Volta:** the cuDNN 9.x support matrix officially lists Turing (7.5)+ only. The
> CUDA 12.x wheels have been verified to still run on Volta in practice (this workstation),
> but pin a tested version rather than upgrading freely.

Verify the libraries are visible to your builds:

```bash
sudo find / -name libcudnn*
```

### 5) TensorRT

Latest: **TensorRT 11.5.0** (<https://developer.nvidia.com/tensorrt>, downloads behind an
NVIDIA login at <https://developer.nvidia.com/tensorrt/download>). The `.deb` packages
require a CUDA 13.x toolkit (there are CUDA 12.x builds as well); the Python bindings
support Python 3.11–3.14.

> **Pascal/Volta:** TensorRT 10.x and 11.x require **SM 7.5+ (Turing)** — neither will run
> on these GPUs. The last supporting release is **TensorRT 8.6.1** (listed under "Legacy
> Versions" on the TensorRT docs page). On a Volta/Pascal workstation, skip TensorRT or use
> 8.6.1.

Easiest route is the pip wheel — `cuda-python` first, then TensorRT (use `tensorrt-cu13`
for CUDA 13 stacks or `tensorrt-cu12` for CUDA 12 to pin the CUDA major version):

```bash
pip install cuda-python
pip install tensorrt
```

Check the installation:

```bash
python3 -c "import tensorrt; print(tensorrt.__version__)"
```

For the full C++/Python SDK, download the `.deb` local repo from the TensorRT download
page and:

```bash
sudo dpkg -i nv-tensorrt-local-repo-*.deb
sudo cp /var/nv-tensorrt-local-repo-*/nv-tensorrt-local-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get install tensorrt
dpkg-query -W tensorrt
```

### 6) Python frameworks (pip wheels — no source build needed)

Both TensorFlow and PyTorch ship GPU wheels that bundle their CUDA dependencies, so these
satisfy the requirement for Python usage. Only build from source if you need source-level
control.

**TensorFlow** (2.21+):

```bash
python3 -m pip install 'tensorflow[and-cuda]'
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

**PyTorch** — installed from PyTorch's own index; pick the CUDA build matching your driver
(`cu128` shown; see <https://pytorch.org/get-started/locally/> for current options):

> **Pascal/Volta:** only the CUDA 12.x wheel lines (`cu126`, `cu128`, `cu129`) ship
> `sm_60`/`sm_70` kernels. Never install from a `cu130+` index on these GPUs. Check any
> install with `python3 -c "import torch; print(torch.cuda.get_arch_list())"`.

```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu128
python3 -c "import torch; print(torch.cuda.is_available())"
```

---

## References

- NVIDIA driver downloads: <https://www.nvidia.com/en-us/drivers/>
- CUDA Toolkit: <https://developer.nvidia.com/cuda-downloads> · [Install Guide (Linux)](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)
- cuDNN: <https://developer.nvidia.com/cudnn-downloads> · [Support matrix](https://docs.nvidia.com/deeplearning/cudnn/backend/latest/reference/support-matrix.html)
- TensorRT: <https://developer.nvidia.com/tensorrt> · [Support matrix](https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/support-matrix.html) · [Install guide](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/)
- NVML Python bindings: <https://pypi.org/project/nvidia-ml-py/>
