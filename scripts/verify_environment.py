# Environment verification for the EfficientViT crack-segmentation workflow.
# Run after installation: python scripts/verify_environment.py
#
# It reports each module's load path and flags anything loaded from OUTSIDE the
# active environment (e.g. a stale ~/.local user-site torch shadowing the env).
import importlib
import os
import platform
import sys

REQUIRED = ["numpy", "PIL", "cv2", "torch", "torchvision", "einops", "timm", "tqdm", "efficientvit"]


def _inside_env(path):
    if not path or not path.startswith(("/", "\\")):
        return True
    prefix = os.path.realpath(sys.prefix)
    return os.path.realpath(path).startswith(prefix)


def main():
    print("Python:", sys.version.split()[0], "|", platform.platform())
    print("prefix:", sys.prefix, "| PYTHONNOUSERSITE:", os.environ.get("PYTHONNOUSERSITE", "0"))
    failed, outside = [], []
    for name in REQUIRED:
        try:
            module = importlib.import_module(name)
            version = getattr(module, "__version__", "?")
            path = getattr(module, "__file__", "?") or "?"
            if not _inside_env(path):
                print("  WARN {:<13} {:<12} {}  <-- OUTSIDE the env (user site?)".format(name, str(version), path))
                outside.append(name)
            else:
                print("  OK   {:<13} {:<12} {}".format(name, str(version), path))
        except Exception as error:  # noqa: BLE001
            message = str(error)
            print("  FAIL {:<13} {}".format(name, message))
            if "libtorch" in message or "cannot open shared object" in message:
                print("       hint: torch shared libraries are broken/missing; "
                      "run `ldd {}` and reinstall torch inside this env".format(path if 'path' in dir() else "<torch/lib/*.so>"))
            failed.append(name)

    try:
        import torch

        print("  CUDA available:", torch.cuda.is_available(), "| device count:", torch.cuda.device_count())
        if torch.cuda.is_available():
            print("  Device 0:", torch.cuda.get_device_name(0))
    except Exception as error:  # noqa: BLE001
        print("  CUDA check failed:", error)

    # import the crack-segmentation helpers end-to-end
    try:
        import torch
        from efficientvit.seg_model_zoo import create_seg_model

        model = create_seg_model("b0", "cityscapes", pretrained=False, n_classes=2)
        out = model(torch.randn(1, 3, 128, 128))
        print("  OK   seg model forward:", tuple(out.shape))
    except Exception as error:  # noqa: BLE001
        print("  FAIL seg model:", error)
        failed.append("seg-model")

    if outside:
        print("\nWARNING: modules loaded from outside the environment: {}.".format(", ".join(outside)))
        print('Set PYTHONNOUSERSITE=1 (or run python -s) and reinstall so only this env is used.')

    if failed:
        print("\nEnvironment verification FAILED:", ", ".join(failed))
        return 1
    print("\nEnvironment verification passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
