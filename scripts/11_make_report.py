"""Step 11 - collect every JSON/CSV produced by steps 01-10 into results/RESULTS.md (tables
laid out like the manuscript) and draw the figures (training curves, confusion matrix, ROC).
Nothing is computed here except formatting. If docs/manuscript_reported_values.json is present,
its values are shown in the comparison columns; otherwise those columns show "-".
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

from _common import REPO, base_parser, ensure_dir, load_json, setup  # noqa: E402
from onconet.metrics import mcnemar_exact  # noqa: E402


class _NA:
    """Placeholder used when no reference values file is present."""

    def __getitem__(self, k):
        return self

    def get(self, k, default="-"):
        return default

    def __iter__(self):
        return iter(("-", "-"))

    def __contains__(self, k):
        return False

    def __bool__(self):
        return False

    def __str__(self):
        return "-"

    __format__ = lambda self, spec: "-"  # noqa: E731


def j(path):
    return load_json(path) if Path(path).exists() else None


def f(x, d=3, pct=False):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    return f"{100 * x:.1f}" if pct else f"{x:.{d}f}"


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def main():
    args = base_parser(__doc__).parse_args()
    cfg, out, log = setup(args, "11_make_report")
    rep = ensure_dir(REPO / "results")
    figs = ensure_dir(rep / "figures")
    ref = REPO / "docs" / "manuscript_reported_values.json"
    ms = load_json(ref) if ref.exists() else _NA()
    names = cfg["dataset"]["class_names"]
    L = ["# OncoNet - results", "",
         "All numbers below were produced by the scripts in this repository (see `outputs/` for the raw "
         "JSON/CSV files and `outputs/logs/` for the run logs). Columns marked *manuscript* repeat the "
         "values printed in the published article for comparison.", ""]

    ds = j(out / "data" / "dataset_summary.json")
    if ds:
        L += ["## Dataset and partitions", "",
              md_table(["Partition"] + names, [[s] + [ds["split_counts"][s][c] for c in names] for s in ("train", "val", "test")]),
              "", "Class weights (training partition, w_c = N/(C n_c)): " +
              ", ".join(f"{c} {w:.3f}" for c, w in zip(names, ds["class_weights"])),
              f"; folder mapping: `{json.dumps(ds['mapping'])}`", ""]

    # ---- Table 2
    rows = []
    for v, label in (("otsu", "Otsu threshold (classical)"), ("unet", "UNet (baseline)"), ("unetpp", "UNet++"),
                     ("unetpp_ag", "UNet++ + Attention Gates"), ("unetpp_msf", "UNet++ + Multi-Scale Fusion"),
                     ("unetpp_ag_msf", "UNet++ + AG + MSF (proposed)")):
        m = j(out / "segmentation" / v / "test_metrics.json")
        if m:
            s, msv = m["mean_per_image"], ms["table2_segmentation_test"].get(v, {})
            rows.append([label, f(s["dice"]), f(s["iou"]), f(s["precision"]), f(s["recall"]), f(s["accuracy"]),
                         f(s["boundary_f1"]), m["best_epoch"] or "-",
                         f"{msv.get('dice', '-')}/{msv.get('iou', '-')}" if msv else "-"])
    if rows:
        L += ["## Table 2 - segmentation (held-out test, n = 75, mean of per-image values)", "",
              md_table(["Model", "Dice", "IoU", "Precision", "Recall", "Pixel acc.", "Boundary F1", "Best epoch",
                        "Dice/IoU (manuscript)"], rows), ""]

    # ---- Table 8 / training curves
    for stage, path in (("DeiT fine-tuning", out / "proposed" / "deit" / "history.csv"),
                        ("AE-MLP", out / "proposed" / "aemlp_history.csv")):
        if Path(path).exists():
            h = pd.read_csv(path)
            sel = [h[h.epoch == e].iloc[0] for e in (10, 20, 30, 40, 50) if (h.epoch == e).any()]
            if not sel or sel[-1].epoch != h.epoch.max():
                sel.append(h.iloc[-1])
            best = h[h.best].iloc[-1]
            L += [f"## Table 8 - {stage}: metrics over epochs (validation, n = 75)", "",
                  md_table(["Epoch", "Train acc", "Val acc", "Precision", "Recall", "F1", "AUC"],
                           [[int(r.epoch), f(r.train_acc), f(r.val_acc), f(r.val_precision), f(r.val_recall),
                             f(r.val_f1), f(r.val_auc)] for r in sel]),
                  f"\nSelected checkpoint: epoch {int(best.epoch)} (highest validation macro-F1); "
                  f"training stopped after epoch {int(h.epoch.max())}.", ""]
    _curves(out, figs)

    # ---- Table 9
    cv = j(out / "cv" / "cv_summary.json")
    if cv:
        rows = [[r["fold"] + 1, f(r["dice"]), f(r["iou"]), f(r["accuracy"]), f(r["f1_macro"])] for r in cv["folds"]]
        rows.append(["Mean ± SD"] + [f"{f(cv[m]['mean'])} ± {f(cv[m]['sd'])}" for m in ("dice", "iou", "accuracy", "f1_macro")])
        rows.append(["95% CI (t, 4 df)"] + [f"{f(cv[m]['ci_low'])}-{f(cv[m]['ci_high'])}" for m in ("dice", "iou", "accuracy", "f1_macro")])
        rows.append(["Manuscript"] + [f"{a} ± {b}" for a, b in (ms["table9_cv"][m] for m in ("dice", "iou", "accuracy", "f1_macro"))])
        L += ["## Table 9 - five-fold CV on the 425-image development pool", "",
              md_table(["Fold", "Dice", "IoU", "Accuracy", "Macro-F1"], rows), ""]

    # ---- proposed / baselines classification (Tables 11, 16)
    P = j(out / "proposed" / "test_metrics.json")
    B = {k: j(out / "baselines" / k / "test_metrics.json") for k in ("resnet50", "vit_small")}
    bench = j(out / "benchmark" / "benchmark.json")
    if P:
        pc = P["per_class"]
        L += ["## Table 11 - class-specific performance of the proposed framework (held-out test)", "",
              md_table(["Metric", "Re-run", "Manuscript"], [
                  ["AML sensitivity (%)", f(pc["AML"]["sensitivity"], pct=True), ms["table11_per_class"]["AML_sensitivity"]],
                  ["ALL sensitivity (%)", f(pc["ALL"]["sensitivity"], pct=True), ms["table11_per_class"]["ALL_sensitivity"]],
                  ["Healthy sensitivity (%)", f(pc["Healthy"]["sensitivity"], pct=True), ms["table11_per_class"]["Healthy_sensitivity"]],
                  ["AML false-negative rate (%)", f(pc["AML"]["false_negative_rate"], pct=True), ms["table11_per_class"]["AML_fnr"]],
                  ["ALL false-positive rate (%)", f(pc["ALL"]["false_positive_rate"], pct=True), ms["table11_per_class"]["ALL_fpr"]],
                  ["Overall accuracy (%) [95% bootstrap CI]",
                   f"{f(P['accuracy'], pct=True)} [{f(P['bootstrap_95ci']['accuracy'][0], pct=True)}-{f(P['bootstrap_95ci']['accuracy'][1], pct=True)}]",
                   ms["abstract"]["accuracy"]],
                  ["Macro-F1", f(P["f1_macro"]), "-"],
                  ["AUC-ROC (macro, one-vs-rest)", f(P["auc_macro_ovr"]), ms["abstract"]["auc"]],
              ]), "", "Confusion matrix (rows = true, columns = predicted, order " + ", ".join(names) + "): `"
              + json.dumps(P["confusion_matrix"]) + "`", ""]
        _cm_roc(out, figs, names)
        rows = []
        for label, m, key in (("Proposed", P, "proposed"), ("CNN (U-Net + ResNet-50)", B["resnet50"], "cnn_baseline"),
                              ("ViT (U-Net + ViT-S/16)", B["vit_small"], "vit_baseline")):
            if m is None:
                continue
            lat = bench["gpu"]["end_to_end_latency_s"][key]["mean"] if bench and "gpu" in bench else None
            rows.append([label, f(m["blast_sensitivity_exact"], pct=True), f(m["blast_sensitivity_any_malignant"], pct=True),
                         f(m["minority_blast_sensitivity"], pct=True), f(m["accuracy"], pct=True), f(m["f1_macro"]),
                         f(m["auc_macro_ovr"]), f(lat, 4) if lat is not None else "n/a"])
        L += ["## Table 16 - comparison under the common protocol (held-out test, n = 75)", "",
              md_table(["Pipeline", "Blast sens. exact (%)", "Blast sens. any-malignant (%)", "Minority-blast (ALL) sens. (%)",
                        "Accuracy (%)", "Macro-F1", "AUC", "GPU latency (s/image)"], rows),
              "", "Blast-cell sensitivity is reported under three definitions: "
              "*exact* = AML+ALL images assigned to their own class; *any-malignant* = AML+ALL images assigned to AML or ALL; "
              "*minority* = sensitivity of the less frequent blast class.", ""]
        pp = out / "proposed" / "test_predictions.csv"
        tests = []
        for k in ("resnet50", "vit_small"):
            bp = out / "baselines" / k / "test_predictions.csv"
            if pp.exists() and bp.exists():
                a = pd.read_csv(pp).set_index("image_id")
                b = pd.read_csv(bp).set_index("image_id").reindex(a.index)
                t = mcnemar_exact(a.y_true == a.y_pred, b.y_true == b.y_pred)
                tests.append([f"Proposed vs {k}", t["a_right_b_wrong"], t["a_wrong_b_right"], f(t["p_value"], 4)])
        if tests:
            L += ["Paired exact McNemar tests on the 75 test images:", "",
                  md_table(["Comparison", "Proposed right / other wrong", "Proposed wrong / other right", "p"], tests), ""]

    # ---- Table 15
    eg = j(out / "proposed" / "error_guided_iterations.json")
    if eg:
        L += ["## Table 15 - error-guided fine-tuning", "",
              md_table(["Iteration", "Hard training images", "Val misclass. (%)", "Test misclass. (%)"],
                       [[r["iteration"], r["n_hard"], f(r["val"]["misclassification_rate"], pct=True),
                         f(r["test"]["misclassification_rate"], pct=True)] for r in eg]),
              f"\nManuscript: {ms['table15_error_guided']['before']}% -> {ms['table15_error_guided']['after']}%. "
              "Hard examples were mined only inside the training partition; the test column is reported, never used.", ""]

    # ---- Tables 10, 12
    it = j(out / "interpretability" / "summary.json")
    if it:
        rows = []
        for k, label in (("proposed", "Proposed"), ("resnet50", "ResNet-50"), ("vit_small", "ViT-S/16")):
            if k in it:
                s = it[k]
                rows.append([label, f(s["afs_mean"], 1), f(s["afs_fused_mean"], 1), f(s["peak_in_nucleus_pct"], 1),
                             f(s["peak_in_cell_pct"], 1), f(s["background_fraction_mean"], 1),
                             f(s["aml_vs_all_accuracy"], pct=True)])
        L += ["## Table 10 - interpretability (held-out test, Grad-CAM, predicted class)", "",
              md_table(["Model", "AFS (%)", "AFS fused views (%)", "Peak in nucleus (%)", "Peak in cell (%)",
                        "Background attention (%)", "AML-vs-ALL acc. (%)"], rows),
              f"\nManuscript AFS: proposed {ms['table10_interpretability']['afs_proposed']}%, "
              f"ResNet-50 {ms['table10_interpretability']['afs_resnet50']}%."
              + (f" Wilcoxon signed-rank p (proposed vs ResNet-50 AFS) = {f(it['afs_proposed_vs_resnet50_wilcoxon_p'], 4)}."
                 if "afs_proposed_vs_resnet50_wilcoxon_p" in it else ""), ""]
        reg = it["proposed"]["regional_by_predicted_class"]
        L += ["## Table 12 - regional attention by predicted class (proposed)", "",
              md_table(["Predicted class", "n", "Nucleus (%)", "Cytoplasm (%)", "Manuscript nucleus/cytoplasm"],
                       [[c, v["n"], f(v["nucleus_share"], 2), f(v["cytoplasm_share"], 2),
                         (f"{ms['table12_regional'][c]['nucleus']}/{ms['table12_regional'][c]['cytoplasm']}"
                          if c in ms["table12_regional"] else "-")] for c, v in reg.items()]),
              f"\nAE-MLP |grad x input| attribution share of the 128x128 (local) view: "
              f"{f(it['aemlp_attribution_local_share_pct'], 1)}%.", ""]

    # ---- Table 13
    rb = j(out / "robustness" / "robustness.json")
    if rb:
        rows = []
        for c, r in rb["results"].items():
            m = ms["table13_robustness"].get(c, {})
            rows.append([c, f(r["proposed_accuracy"], pct=True), f(r["cnn_baseline_accuracy"], pct=True),
                         f(r.get("seg_error_reduction_pct"), 1), f(r["proposed_seg_dice"]),
                         f"{m.get('proposed', '-')}/{m.get('traditional', '-')}"])
        L += ["## Table 13 - controlled robustness (held-out test)", "",
              md_table(["Condition", "Proposed acc. (%)", "CNN pipeline acc. (%)", "Seg. error reduction by stain norm (%)",
                        "Seg. Dice", "Manuscript proposed/traditional"], rows), ""]

    # ---- Table 14
    if bench:
        c = bench["cpu"]
        pr, mc = c["params"], c["macs_per_image"]
        tot_macs = sum(v for k, v in mc.items() if k in ("unetpp_ag_msf", "deit_two_views", "aemlp") and isinstance(v, float))
        rows = [["UNet++-AG-MSF parameters", f"{pr['unetpp_ag_msf'] / 1e6:.2f} M"],
                ["DeiT-Small/16 (shared) parameters", f"{pr['deit_small_shared'] / 1e6:.2f} M"],
                ["AE-MLP parameters", f"{pr['aemlp'] / 1e6:.2f} M"],
                ["Total inference parameters", f"{pr['proposed_inference_total'] / 1e6:.2f} M"],
                ["FP32 / FP16 weight storage", f"{pr['proposed_inference_total'] * 4 / 1e6:.0f} MB / {pr['proposed_inference_total'] * 2 / 1e6:.0f} MB"],
                ["MACs per image (UNet++ / DeiT x2 / AE-MLP)",
                 " / ".join(f"{mc[k] / 1e9:.2f} G" if isinstance(mc[k], float) else str(mc[k]) for k in ("unetpp_ag_msf", "deit_two_views", "aemlp"))],
                ["Total MACs per image", f"{tot_macs / 1e9:.2f} G"]]
        for dev in ("gpu", "cpu"):
            if dev in bench:
                b = bench[dev]
                rows += [[f"{dev.upper()} end-to-end latency (batch 1)", f"{b['end_to_end_latency_s']['proposed']['mean'] * 1000:.1f} ms"],
                         [f"{dev.upper()} throughput (batch {cfg['benchmark']['throughput_batch']})",
                          f"{b['throughput_images_per_min']['proposed']:.0f} images/min"]]
                rows += [[f"{dev.upper()} stage: {k}", f"{v * 1000:.2f} ms"] for k, v in b["stage_latency_s"].items()]
        L += ["## Table 14 - computational profile", "", md_table(["Characteristic", "Value"], rows),
              f"\nGPU: {bench.get('gpu', {}).get('gpu', 'n/a')}; FP32; 10 warm-up + {cfg['benchmark']['repetitions']} timed passes.", ""]

    (rep / "RESULTS.md").write_text("\n".join(L), encoding="utf-8")
    log(f"wrote {rep / 'RESULTS.md'}")


def _curves(out, figs):
    seg = out / "segmentation" / "unetpp_ag_msf" / "history.csv"
    if seg.exists():
        h = pd.read_csv(seg)
        fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
        ax[0].plot(h.epoch, h.train_loss, label="train")
        ax[0].plot(h.epoch, h.val_loss, label="validation")
        ax[0].set(xlabel="Epoch", ylabel="0.5 Dice + 0.5 BCE loss", title="UNet++-AG-MSF loss")
        ax[0].legend()
        ax[1].plot(h.epoch, h.val_dice)
        ax[1].set(xlabel="Epoch", ylabel="Validation Dice", title="UNet++-AG-MSF Dice")
        fig.tight_layout()
        fig.savefig(figs / "fig15_unetpp_training.png", dpi=200)
        plt.close(fig)
    for name, path, fn in (("DeiT", out / "proposed" / "deit" / "history.csv", "fig16_deit_training.png"),
                           ("AE-MLP", out / "proposed" / "aemlp_history.csv", "fig17_aemlp_training.png")):
        if path.exists():
            h = pd.read_csv(path)
            fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
            ax[0].plot(h.epoch, h.train_loss, label="train")
            ax[0].plot(h.epoch, h.val_loss, label="validation")
            ax[0].set(xlabel="Epoch", ylabel="Loss", title=f"{name} loss")
            ax[0].legend()
            if name == "DeiT":
                ax[1].plot(h.epoch, h.lr)
                ax[1].set(xlabel="Epoch", ylabel="Learning rate", title="Warm-up + cosine schedule")
            else:
                ax[1].plot(h.epoch, h.train_acc, label="train")
                ax[1].plot(h.epoch, h.val_acc, label="validation")
                ax[1].set(xlabel="Epoch", ylabel="Accuracy", title="AE-MLP accuracy")
                ax[1].legend()
            fig.tight_layout()
            fig.savefig(figs / fn, dpi=200)
            plt.close(fig)


def _cm_roc(out, figs, names):
    p = out / "proposed" / "test_predictions.csv"
    if not p.exists():
        return
    d = pd.read_csv(p)
    from sklearn.metrics import ConfusionMatrixDisplay
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    ConfusionMatrixDisplay.from_predictions(d.y_true, d.y_pred, display_labels=names, ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Proposed - held-out test")
    fig.tight_layout()
    fig.savefig(figs / "fig23_confusion_matrix.png", dpi=200)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    for i, c in enumerate(names):
        fpr, tpr, _ = roc_curve(d.y_true == i, d[f"p_{c}"])
        ax.plot(fpr, tpr, label=c)
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set(xlabel="False-positive rate", ylabel="True-positive rate", title="One-vs-rest ROC (test)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figs / "roc_test.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
