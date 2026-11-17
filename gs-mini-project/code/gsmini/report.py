"""Helpers to look at results (used by the notebook; also usable from a terminal / your own scripts)."""
import glob
import json
import os


def load_metrics(run):
    """A run directory -> its metrics.json as a dict (or None if the run has not finished)."""
    p = os.path.join(run, "metrics.json")
    return json.load(open(p)) if os.path.exists(p) else None


def results_table(runs, metric="psnr", splits=None, extra=("lpips",)):
    """Print one row per run directory: <metric> (and the `extra` metrics) for every evaluated split.

        results_table(["runs/p3_base", "runs/p3_best"])
    """
    rows = []
    for r in runs:
        m = load_metrics(r)
        rows.append((os.path.basename(r.rstrip("/")), m))
    sp = splits or sorted({k for _, m in rows if m for k, v in m.items() if isinstance(v, dict) and "psnr" in v})
    head = ["run"] + [f"{s}:{metric}" + "".join(f"/{e}" for e in extra) for s in sp] + ["#gaussians"]
    out = [head]
    for name, m in rows:
        if m is None:
            out.append([name] + ["(not finished)"] * len(sp) + [""])
            continue
        cells = []
        for s in sp:
            v = m.get(s)
            cells.append("-" if not v else f"{v[metric]:.2f}" + "".join(f"/{v[e]:.3f}" for e in extra))
        out.append([name] + cells + [str(m.get("num_gaussians", ""))])
    w = [max(len(str(r[i])) for r in out) for i in range(len(head))]
    for k, r in enumerate(out):
        print("  ".join(str(c).ljust(w[i]) for i, c in enumerate(r)))
        if k == 0:
            print("  ".join("-" * w[i] for i in range(len(head))))


def show_renders(run, split, n=3, start=0):
    """Show n side-by-side images (ground truth | render) of a split, in a notebook."""
    from PIL import Image
    files = sorted(glob.glob(os.path.join(run, "renders", split, "*.jpg")))[start::max(1, len(glob.glob(os.path.join(run, "renders", split, "*.jpg"))) // n)][:n]
    try:
        from IPython.display import display
    except ImportError:
        display = lambda im: im.show()
    for f in files:
        print(os.path.basename(f))
        display(Image.open(f))


def compare_renders(runs, split, n=3, labels=None, width=320, start=0):
    """Compare several runs on the same held-out views, in a notebook.  One row per view; the columns are the ground truth followed by the
    render of every run:

        compare_renders(["runs/p3_base", "runs/p3_best"], "val_offtraj", n=3, labels=["baseline", "your best"])
    """
    from PIL import Image, ImageDraw
    labels = labels or [os.path.basename(os.path.normpath(r)) for r in runs]
    names = sorted(os.path.basename(f) for f in glob.glob(os.path.join(runs[0], "renders", split, "*.jpg")))
    names = names[start::max(1, len(names) // n)][:n]
    first = None
    rows = []
    for nm in names:
        panels = []
        for k, r in enumerate(runs):
            p = os.path.join(r, "renders", split, nm)
            if not os.path.exists(p):
                panels.append(None); continue
            im = Image.open(p); w = im.width // 2                       # the saved image is [ground truth | render]
            if k == 0:
                first = im.crop((0, 0, w, im.height))                    # the ground truth is the same in every run
            panels.append(im.crop((w, 0, 2 * w, im.height)))
        rows.append((nm, [first] + panels))
    if not rows:
        print("no rendered images found (is the run finished, and is the split name right?)"); return
    pw = width; ph = round(rows[0][1][0].height * width / rows[0][1][0].width); head = 18; cols = len(runs) + 1
    sheet = Image.new("RGB", (pw * cols, (ph + head) * len(rows) + head), "white"); d = ImageDraw.Draw(sheet)
    for j, t in enumerate(["ground truth"] + list(labels)):
        d.text((j * pw + 4, 3), t, fill="black")
    for i, (nm, ims) in enumerate(rows):
        for j, im in enumerate(ims):
            if im is not None:
                sheet.paste(im.resize((pw, ph)), (j * pw, head + i * (ph + head) + head - 2))
        d.text((4, head + i * (ph + head) + 2), os.path.splitext(nm)[0], fill="black")
    try:
        from IPython.display import display
        display(sheet)
    except ImportError:
        sheet.show()
