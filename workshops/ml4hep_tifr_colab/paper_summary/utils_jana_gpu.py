"""GPU setup and launch plumbing, outside the fingerprinted JANA driver.

This module is importable in both notebook Python and the legacy interpreter.
Only ``check_tensorflow_gpu`` imports TensorFlow. CUDA library search paths
are attached to JANA subprocesses, never to the modern notebook's environment.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import shlex
import subprocess
import sys


def subprocess_environment(environment=None):
    environment = dict(os.environ if environment is None else environment)
    libraries = environment.get("PAPER_SUMMARY_JANA_CUDA_LIBRARY_PATH", "")
    if libraries:
        paths = libraries.split(os.pathsep)
        paths += environment.get("LD_LIBRARY_PATH", "").split(os.pathsep)
        environment["LD_LIBRARY_PATH"] = os.pathsep.join(dict.fromkeys(p for p in paths if p))
    cuda_root = environment.get("PAPER_SUMMARY_JANA_CUDA_DATA_DIR")
    if cuda_root:
        flags = [flag for flag in shlex.split(environment.get("XLA_FLAGS", ""))
                 if not flag.startswith("--xla_gpu_cuda_data_dir=")]
        flags.append("--xla_gpu_cuda_data_dir=" + cuda_root)
        environment["XLA_FLAGS"] = shlex.join(flags)
        paths = [str(Path(cuda_root) / "bin"), *environment.get("PATH", "").split(os.pathsep)]
        environment["PATH"] = os.pathsep.join(dict.fromkeys(p for p in paths if p))
    environment["PYTHONUNBUFFERED"] = "1"
    environment["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    return environment


def check_tensorflow_gpu():
    """Require GPU matmul/gradient and the default Adam/XLA update path."""
    import tensorflow as tf

    if tf.__version__ != "2.12.0":
        raise RuntimeError(f"Expected TensorFlow 2.12.0, found {tf.__version__}.")
    devices = tf.config.list_physical_devices("GPU")
    if not devices:
        raise RuntimeError(
            "Exact JANA has no TensorFlow GPU. Select a GPU runtime in Colab "
            "and rerun notebook 02's environment cell. Refusing CPU training."
        )
    for device in devices:
        tf.config.experimental.set_memory_growth(device, True)
    # Keep the scientific float32 calculation; do not silently enable TF32
    # or mixed precision on newer GPUs.
    tf.config.experimental.enable_tensor_float_32_execution(False)
    previous_soft_placement = tf.config.get_soft_device_placement()
    tf.config.set_soft_device_placement(False)
    try:
        with tf.device("/GPU:0"):
            values = tf.Variable(tf.ones((16, 16)))
            optimizer = tf.keras.optimizers.Adam(5e-4, global_clipnorm=1.0)

            @tf.function
            def step():
                with tf.GradientTape() as tape:
                    product = tf.matmul(values, values)
                    loss = tf.reduce_sum(product)
                gradient = tape.gradient(loss, values)
                optimizer.apply_gradients([(gradient, values)])
                return product, gradient

            product, gradient = step()
            finite = bool(tf.reduce_all(tf.math.is_finite(gradient)).numpy())
        if "GPU:0" not in product.device or "GPU:0" not in gradient.device or not finite:
            raise RuntimeError("The exact-JANA GPU computation check failed.")
    finally:
        tf.config.set_soft_device_placement(previous_soft_placement)
    details = tf.config.experimental.get_device_details(devices[0])
    result = {
        "tensorflow": tf.__version__,
        "device": product.device,
        "device_name": details.get("device_name", devices[0].name),
        "compute_capability": details.get("compute_capability"),
        "cuda_build": tf.sysconfig.get_build_info().get("cuda_version"),
        "tf32": False,
        "adam_step": int(optimizer.iterations.numpy()),
    }
    print("[exact JANA GPU] " + json.dumps(result), flush=True)
    return result


def prepare_gpu_environment(interpreter, *, install_if_missing=True):
    import utils_jana as jana

    # Resolving a venv's bin/python symlink would select the BASE interpreter
    # and lose the isolated site-packages (and pip). Keep the venv entry path.
    interpreter = Path(interpreter).expanduser().absolute()
    environment = jana._isolated_jana_subprocess_env()
    # Fast hardware check BEFORE downloading the large CUDA wheels.
    if not os.environ.get("CUDA_VISIBLE_DEVICES", "all").strip() or os.environ.get("CUDA_VISIBLE_DEVICES") == "-1":
        raise RuntimeError("CUDA_VISIBLE_DEVICES hides all GPUs from exact JANA.")
    try:
        hardware = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            text=True, capture_output=True, timeout=30,
        )
    except FileNotFoundError as error:
        raise RuntimeError("Select a GPU in Colab (Runtime > Change runtime type), then rerun 02.") from error
    if hardware.returncode or not hardware.stdout.strip():
        raise RuntimeError("Colab has no usable NVIDIA GPU: " + hardware.stderr[-2000:])
    print("[exact JANA GPU] Colab hardware: " + hardware.stdout.strip(), flush=True)
    if install_if_missing:
        print("[exact JANA GPU] Installing/checking CUDA 11.8 and cuDNN 8.6 in the isolated environment.", flush=True)
        subprocess.run(
            [str(interpreter), "-m", "pip", "--isolated", "install", "--no-input",
             "--disable-pip-version-check", "--no-user", "--no-deps", "--retries", "8",
             "--timeout", "120", "-r", str(Path(__file__).with_name("requirements_jana_gpu.txt"))],
            check=True, env=environment,
        )
    # Ask the selected interpreter, not the notebook interpreter, for its
    # site-packages location. No guessed Python-version directory.
    probe = subprocess.run(
        [str(interpreter), "-c", "import json, pathlib, sysconfig; p=pathlib.Path(sysconfig.get_paths()['purelib'])/'nvidia'; print(json.dumps({'libraries': [str(x) for x in sorted(p.glob('*/lib')) if x.is_dir()], 'cuda_root': str(p/'cuda_nvcc')}))"],
        check=True, capture_output=True, text=True, env=environment,
    )
    payload = json.loads(probe.stdout)
    libraries = payload["libraries"]
    if not libraries:
        raise RuntimeError("No isolated NVIDIA libraries found. Enable installation in the 02 environment cell.")
    # Only our dedicated setting changes in the notebook process.
    os.environ["PAPER_SUMMARY_JANA_CUDA_LIBRARY_PATH"] = os.pathsep.join(libraries)
    cuda_root = Path(payload["cuda_root"])
    if not (cuda_root / "nvvm" / "libdevice" / "libdevice.10.bc").is_file() or not (cuda_root / "bin" / "ptxas").is_file():
        raise RuntimeError("Missing isolated CUDA 11.8 ptxas/libdevice. Rerun the 02 environment cell with installation enabled.")
    os.environ["PAPER_SUMMARY_JANA_CUDA_DATA_DIR"] = str(cuda_root)
    os.environ["PAPER_SUMMARY_JANA_REQUIRE_GPU"] = "1"
    subprocess.run(
        [str(interpreter), "-u", str(Path(__file__).resolve()), "check"],
        check=True, env=subprocess_environment(jana._isolated_jana_subprocess_env()),
    )
    activate_runtime_hooks(jana)
    return interpreter


def launch_gpu_campaign(python_executable, **kwargs):
    import utils_jana as jana

    command = [str(python_executable), "-u", str(Path(__file__).with_name("utils_jana_training.py")), "campaign"]
    for key in ("artifact_root", "master_bank_path", "shape_bank_path", "pilot_bank_path", "validation_bank_path"):
        flag = key.removesuffix("_path").replace("_", "-")
        command += ["--" + flag, str(Path(kwargs[key]).expanduser().resolve())]
    for key in ("budgets", "seeds"):
        command += ["--" + key, *[str(int(value)) for value in kwargs[key]]]
    command += ["--profile", str(kwargs.get("profile", "PAPER"))]
    if not kwargs.get("load_if_available", True):
        command += ["--no-load-if-available"]
    if kwargs.get("force", False):
        command += ["--force"]
    subprocess.run(command, check=True, env=subprocess_environment(jana._isolated_jana_subprocess_env()))
    path = Path(kwargs["artifact_root"]).expanduser().resolve() / "jana_paper" / "campaign_manifest.json"
    return json.loads(path.read_text())


def activate_runtime_hooks(jana):
    """Idempotent, opt-in hooks; reloading utils_jana is safe."""
    if os.environ.get("PAPER_SUMMARY_JANA_REQUIRE_GPU") != "1":
        return
    original = jana._isolated_jana_subprocess_env
    if not getattr(original, "_jana_gpu_hook", False):
        def gpu_environment():
            return subprocess_environment(original())
        gpu_environment._jana_gpu_hook = True
        jana._isolated_jana_subprocess_env = gpu_environment
    jana.launch_isolated_campaign = launch_gpu_campaign


if __name__ == "__main__":
    if sys.argv[1:] != ["check"]:
        raise SystemExit("Usage: utils_jana_gpu.py check")
    check_tensorflow_gpu()
