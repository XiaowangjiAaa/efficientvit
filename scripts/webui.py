# Launch TensorBoard for the crack-segmentation runs.
#   python scripts/webui.py --logdir work_dirs --port 6006
# Then open http://127.0.0.1:6006
import argparse
import os


def main():
    parser = argparse.ArgumentParser(description="Launch TensorBoard for EfficientViT segmentation runs")
    parser.add_argument("--logdir", default="work_dirs", help="directory that contains the run folders")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6006)
    args = parser.parse_args()

    logdir = os.path.abspath(args.logdir)
    os.makedirs(logdir, exist_ok=True)
    try:
        from tensorboard import program
    except ImportError:
        raise SystemExit('TensorBoard is not installed. Install it with: pip install -e ".[dev]" or pip install tensorboard')

    tb = program.TensorBoard()
    tb.configure(argv=[None, "--logdir", logdir, "--host", args.host, "--port", str(args.port)])
    url = tb.launch()
    print("TensorBoard running at {} (logdir={})".format(url, logdir))


if __name__ == "__main__":
    main()
