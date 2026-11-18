"""Step 10 - computational profile (Section 4.7.3, Tables 14 and 16).

* parameters per component (counted from the instantiated modules)
* multiply-accumulate operations per image (fvcore; one fused multiply-add = 1 MAC)
* FP32 latency, batch size 1: 10 warm-up passes excluded, mean over 100 timed passes with
  torch.cuda.synchronize() before and after each pass; stage-wise and end-to-end
  (preprocessing = resize + Reinhard + tensor conversion on the CPU)
* batched throughput (images/min) on GPU and CPU
for the proposed pipeline and the two baseline pipelines.
"""
from __future__ import annotations

import time

import numpy as np
import torch

from _common import base_parser, ensure_dir, load_json, load_manifest, norm_stats, processed, save_json, setup
from onconet.data import ReinhardNormalizer, imread_rgb, split_frames
from onconet.engine import build_classifier
from onconet.models import AEMLP, build_seg_model, count_params
from onconet.pipeline import EndToEnd, _to_tensor


def _sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize()


def timeit(fn, dev, warmup, reps):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(reps):
        _sync(dev)
        t0 = time.perf_counter()
        fn()
        _sync(dev)
        ts.append(time.perf_counter() - t0)
    return float(np.mean(ts)), float(np.std(ts))


def macs(model, inputs):
    try:
        from fvcore.nn import FlopCountAnalysis
        fca = FlopCountAnalysis(model, inputs)
        fca.unsupported_ops_warnings(False)
        fca.uncalled_modules_warnings(False)
        return float(fca.total())
    except Exception as e:  # pragma: no cover
        return f"unavailable: {e}"


def build_all(cfg, out, dev):
    names = cfg["dataset"]["class_names"]
    C = len(names)
    seg = build_seg_model("unetpp_ag_msf", cfg["segmentation"]["channels"])
    unet = build_seg_model("unet", cfg["segmentation"]["channels"])
    deit = build_classifier(cfg, "deit_mr", C)
    mlp = AEMLP(deit.embed_dim, tuple(cfg["aemlp"]["hidden"]), C, cfg["aemlp"]["dropout"])
    res = build_classifier(cfg, "resnet50", C)
    vit = build_classifier(cfg, "vit_small", C)
    for m, path in ((seg, "segmentation/unetpp_ag_msf/best.pt"), (unet, "segmentation/unet/best.pt"),
                    (deit, "proposed/deit/best.pt"), (mlp, "proposed/aemlp_final.pt"),
                    (res, "baselines/resnet50/best.pt"), (vit, "baselines/vit_small/best.pt")):
        p = out / path
        if p.exists():
            m.load_state_dict(torch.load(p, map_location="cpu")["state_dict"])
    return {k: v.to(dev).eval() for k, v in dict(seg=seg, unet=unet, deit=deit, mlp=mlp, res=res, vit=vit).items()}


def profile(cfg, out, dev, log, reps, raws, rn, mean, std):
    M = build_all(cfg, out, dev)
    S, V = cfg["preprocess"]["seg_size"], cfg["preprocess"]["vit_size"]
    b = cfg["benchmark"]
    xs = torch.randn(1, 3, S, S, device=dev)
    xv = torch.randn(1, 3, V, V, device=dev)
    z = torch.randn(1, M["deit"].embed_dim, device=dev)
    pipes = {
        "proposed": EndToEnd(cfg, M["seg"], lambda xl, xg: M["mlp"](M["deit"].embed(xl, xg)), mean, std, rn),
        "cnn_baseline": EndToEnd(cfg, M["unet"], lambda xl, xg: M["res"](xg), mean, std, rn),
        "vit_baseline": EndToEnd(cfg, M["unet"], lambda xl, xg: M["vit"](xg), mean, std, rn),
    }
    raw = raws[0]
    r = {"device": str(dev), "gpu": torch.cuda.get_device_name(0) if dev.type == "cuda" else None}
    with torch.no_grad():
        img = pipes["proposed"].preprocess(raw)
        masks, _ = pipes["proposed"].segment([img])
        xl, xg = pipes["proposed"].roi_tensors([img], masks)
        stages = {
            "preprocessing": lambda: _to_tensor(pipes["proposed"].preprocess(raw), mean, std).to(dev),
            "segmentation_unetpp": lambda: M["seg"](xs),
            "roi_extraction": lambda: pipes["proposed"].roi_tensors([img], pipes["proposed"].segment([img])[0]),
            "deit_two_views": lambda: M["deit"].embed(xl, xg),
            "aemlp": lambda: M["mlp"](z),
        }
        r["stage_latency_s"] = {k: timeit(f, dev, b["warmup"], reps)[0] for k, f in stages.items()}
        r["end_to_end_latency_s"] = {}
        r["throughput_images_per_min"] = {}
        for name, pipe in pipes.items():
            mu, sd = timeit(lambda: pipe.predict([raw]), dev, b["warmup"], reps)
            r["end_to_end_latency_s"][name] = {"mean": mu, "sd": sd}
            batch = (raws * ((b["throughput_batch"] // len(raws)) + 1))[:b["throughput_batch"]]
            mu_b, _ = timeit(lambda: pipe.predict(batch), dev, 2, max(3, reps // 10))
            r["throughput_images_per_min"][name] = 60.0 * len(batch) / mu_b
            log(f"[{dev.type}] {name}: latency {mu * 1000:.1f} ms/image, throughput "
                f"{r['throughput_images_per_min'][name]:.0f} images/min (batch {len(batch)})")
    if dev.type == "cpu":
        r["params"] = {"unetpp_ag_msf": count_params(M["seg"]), "deit_small_shared": count_params(M["deit"].backbone),
                       "deit_heads_training_only": count_params(M["deit"]) - count_params(M["deit"].backbone),
                       "aemlp": count_params(M["mlp"]), "unet": count_params(M["unet"]),
                       "resnet50": count_params(M["res"]), "vit_small": count_params(M["vit"])}
        r["params"]["proposed_inference_total"] = (r["params"]["unetpp_ag_msf"] + r["params"]["deit_small_shared"]
                                                   + 2 + r["params"]["aemlp"])
        cpu = torch.device("cpu")
        Mc = {k: v.to(cpu) for k, v in M.items()}
        xs_c, xv_c = xs.cpu(), xv.cpu()

        class _Embed(torch.nn.Module):
            def __init__(s, d):
                super().__init__()
                s.d = d

            def forward(s, a, b_):
                return s.d.embed(a, b_)

        r["macs_per_image"] = {"unetpp_ag_msf": macs(Mc["seg"], (xs_c,)),
                               "deit_two_views": macs(_Embed(Mc["deit"]), (xv_c, xv_c)),
                               "aemlp": macs(Mc["mlp"], (z.cpu(),)), "unet": macs(Mc["unet"], (xs_c,)),
                               "resnet50": macs(Mc["res"], (xv_c,)), "vit_small": macs(Mc["vit"], (xv_c,))}
        for k, v in M.items():
            v.to(dev)
    return r


def main():
    args = base_parser(__doc__).parse_args()
    cfg, out, log = setup(args, "10_benchmark")
    df = load_manifest(cfg)
    _, _, te = split_frames(df)
    P = processed(cfg)
    meta = load_json(P / "meta.json")
    rn = ReinhardNormalizer(meta["reinhard_lab"]["mean"], meta["reinhard_lab"]["std"])
    mean, std = norm_stats(cfg)
    raws = [imread_rgb(P / "images_256_raw" / f"{i}.png") for i in te.image_id[:16]]
    torch.set_grad_enabled(False)
    res = {}
    if torch.cuda.is_available():
        res["gpu"] = profile(cfg, out, torch.device("cuda"), log, cfg["benchmark"]["repetitions"], raws, rn, mean, std)
    res["cpu"] = profile(cfg, out, torch.device("cpu"), log, cfg["benchmark"]["cpu_repetitions"], raws, rn, mean, std)
    res["precision"] = "fp32"
    save_json(res, ensure_dir(out / "benchmark") / "benchmark.json")
    log("params:", res["cpu"]["params"])
    log("MACs:", res["cpu"]["macs_per_image"])


if __name__ == "__main__":
    main()
