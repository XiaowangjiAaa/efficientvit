# Environment verification for the EfficientViT crack-segmentation workflow.
# Run after installation: python scripts/verify_environment.py
import importlib
import platform
import sys

REQUIRED = ["numpy", "PIL", "cv2", "torch", "torchvision", "einops", "timm", "tqdm", "efficientvit"]


def main():
    print("Python:", sys.version.split()[0], "|", platform.platform())
    failed = []
    for name in REQUIRED:
        try:
            module = importlib.import_module(name)
            version = getattr(module, "__version__", "?")
            print("  OK  {:<14} {}".format(name, version))
        except Exception as error:  # noqa: BLE001
            print("  FAIL {:<13} {}".format(name, error))
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
        from efficientvit.seg_model_zoo import create_seg_model

        model = create_seg_model("b0", "cityscapes", pretrained=False, n_classes=2)
        out = model(torch.randn(1, 3, 128, 128))
        print("  OK  seg model forward:", tuple(out.shape))
    except Exception as error:  # noqa: BLE001
        print("  FAIL seg model:", error)
        failed.append("seg-model")

    if failed:
        print("\nEnvironment verification FAILED:", ", ".join(failed))
        return 1
    print("\nEnvironment verification passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
