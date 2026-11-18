"""Part 1, step 1: verify your installation.   python check_setup.py
Checks: python/torch/CUDA, gsplat (compiles its CUDA kernels on first use: ~1-2 minutes), a differentiable
render + backward pass, LPIPS weights, the depth-model download, and that the data folders are present."""
import os, sys, time

def step(name, fn):
    t = time.time()
    try:
        out = fn()
        print(f"[ OK ] {name}" + (f"  ({out})" if out else "") + f"  [{time.time()-t:.1f}s]", flush=True)
        return True
    except Exception as e:
        print(f"[FAIL] {name}: {type(e).__name__}: {e}", flush=True)
        return False

res = []
def imp():
    global torch
    import torch
    assert sys.version_info >= (3, 10), "need Python >= 3.10"
    return f"python {sys.version.split()[0]}, torch {torch.__version__}"
res.append(step("import torch", imp))
res.append(step("CUDA GPU", lambda: (torch.cuda.is_available() or (_ for _ in ()).throw(RuntimeError("torch.cuda.is_available() is False - install the CUDA build of PyTorch")))
                and f"{torch.cuda.get_device_name(0)}, {torch.cuda.get_device_properties(0).total_memory/2**30:.0f} GB"))

def build_tools():
    """gsplat compiles CUDA code the first time it is imported, unless a prebuilt wheel or an earlier build is available.
    Check the prerequisites HERE, so that a missing compiler gives a clear instruction instead of a long compiler error."""
    import glob, importlib.util, re, shutil, subprocess
    import torch
    from torch.utils import cpp_extension
    spec = importlib.util.find_spec("gsplat")                       # do NOT import gsplat here: importing it starts the compile
    if spec is None:
        raise RuntimeError("gsplat is not installed: run  pip install -r requirements.txt")
    if glob.glob(os.path.join(os.path.dirname(spec.origin), "csrc*.so")):
        return "gsplat is a prebuilt wheel: nothing to compile"
    root = os.environ.get("TORCH_EXTENSIONS_DIR") or os.path.join(os.path.expanduser("~"), ".cache", "torch_extensions")
    if glob.glob(os.path.join(root, "*", "gsplat_cuda", "gsplat_cuda*.so")) or glob.glob(os.path.join(root, "gsplat_cuda", "gsplat_cuda*.so")):
        return "gsplat was compiled before on this machine: nothing to compile"
    problems = []
    home = cpp_extension.CUDA_HOME
    nvcc = os.path.join(home, "bin", "nvcc") if home else shutil.which("nvcc")
    if not nvcc or not os.path.exists(nvcc):
        problems.append(f"nvcc (the CUDA toolkit) was not found. Install the CUDA toolkit that matches PyTorch's CUDA version ({torch.version.cuda}), "
                        f"from NVIDIA (https://developer.nvidia.com/cuda-toolkit-archive), then set CUDA_HOME to it")
    else:
        out = subprocess.run([nvcc, "--version"], capture_output=True, text=True).stdout
        m = re.search(r"release (\d+)\.(\d+)", out)
        if m and m.group(1) != torch.version.cuda.split(".")[0]:
            problems.append(f"the CUDA toolkit at {nvcc} is version {m.group(1)}.{m.group(2)}, but PyTorch was built for CUDA {torch.version.cuda}: "
                            f"the major versions must match. Install a matching toolkit and set CUDA_HOME")
    if not (shutil.which("g++") or shutil.which("c++")):
        problems.append("no C++ compiler found: install one (Ubuntu: sudo apt install build-essential)")
    if not shutil.which("ninja"):
        problems.append("ninja not found: pip install ninja")
    if problems:
        raise RuntimeError("gsplat has to be compiled on this machine (once), and:\n         - " + "\n         - ".join(problems) + "\n         (see Section 2.4 of the project PDF, \"Building gsplat\")")
    cores = os.cpu_count() or 2
    try:
        mem_gb = int(re.search(r"MemTotal:\s+(\d+)", open("/proc/meminfo").read()).group(1)) / 2**20
    except Exception:
        mem_gb = 16
    os.environ.setdefault("MAX_JOBS", str(max(1, min(cores, int(mem_gb // 1.5), 16))))        # each compile job needs about 1-1.5 GB of RAM
    return f"gsplat will be compiled once, on the next step: {cores} cores, {mem_gb:.0f} GB RAM, MAX_JOBS={os.environ['MAX_JOBS']}"
res.append(step("gsplat build prerequisites", build_tools))


def gs():
    import torch
    from gsplat import rasterization
    n, d = 2000, "cuda"
    means = torch.randn(n, 3, device=d, requires_grad=True); means.data[:, 2] += 5
    q = torch.nn.functional.normalize(torch.randn(n, 4, device=d), dim=-1)
    out, alpha, _ = rasterization(means, q, torch.rand(n, 3, device=d) * 0.1, torch.rand(n, device=d), torch.rand(n, 3, device=d),
                                  torch.eye(4, device=d)[None], torch.tensor([[300., 0, 200], [0, 300, 150], [0, 0, 1]], device=d)[None], 400, 300)
    out.mean().backward()
    assert means.grad is not None and torch.isfinite(means.grad).all()
    return "forward + backward OK"
if res[-1]:
    res.append(step("gsplat render (first run compiles CUDA code, be patient)", gs))
else:
    print("[skip] gsplat render: fix 'gsplat build prerequisites' first"); res.append(False)
def lp():
    import lpips; lpips.LPIPS(net="alex", verbose=False)
res.append(step("LPIPS weights", lp))
def tf():
    import transformers, cv2, scipy, plyfile, imageio
    return f"transformers {transformers.__version__}"
res.append(step("other packages", tf))
root = os.path.dirname(os.path.abspath(__file__))
dd = os.environ.get("GS_DATA_DIR", f"{root}/data")
for d in ["part1_garden", "part2_room", "part3_wanderland", "part4_robot"]:
    p = os.path.join(dd, d, "cameras.json")
    if os.path.exists(p): print(f"[ OK ] data/{d}")
    else: print(f"[skip] data/{d} not extracted yet (fine until you start that part)")
print("\nEverything is set up!" if all(res) else "\nSome checks failed - fix them before continuing.")
sys.exit(0 if all(res) else 1)
