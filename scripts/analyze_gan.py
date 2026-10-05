#!/usr/bin/env python3
"""Run the ngspice I-V and C-V testbenches for the behavioral GaN HEMT, then
plot the curves and extract datasheet-style figures of merit.

Usage (from anywhere):
    python3 scripts/analyze_gan.py              # simulate + analyze
    python3 scripts/analyze_gan.py --skip-sim   # re-analyze existing raw data

Outputs:
    results/raw/*            raw ngspice data
    results/summary.csv      extracted metrics
    docs/images/gan_iv.png   I-V plots
    docs/images/gan_cv.png   C-V plots (+ Qoss / Eoss)
"""
import argparse
import subprocess
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "results" / "raw"
IMAGES = ROOT / "docs" / "images"
trapz = getattr(np, "trapezoid", None) or np.trapz

VGS_FAMILY = [2, 3, 4, 5]
V_RATED_EVAL = 50.0           # voltage at which Coss/Qoss/Eoss are reported
I_REV = 10.0                  # current for the reverse-conduction voltage drop
I_VTH = 1e-3                  # constant-current threshold definition


def run_ngspice():
    RAW.mkdir(parents=True, exist_ok=True)
    for net in ("gan_iv.spice", "gan_cv.spice"):
        print(f"Running ngspice on {net} ...")
        try:
            subprocess.run(["ngspice", "-b", str(ROOT / "netlists" / net)], cwd=ROOT,
                           check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as err:
            raise SystemExit(f"ngspice failed on {net}:\n{err.stdout}\n{err.stderr}")


def load_dc(name):
    d = np.loadtxt(RAW / name, skiprows=1)
    return d[:, 0], d[:, 1]


def load_cv():
    a = np.genfromtxt(RAW / "cv_ciss.csv", delimiter=",", names=True)
    b = np.genfromtxt(RAW / "cv_cossrss.csv", delimiter=",", names=True)
    return a["vds"], a["ciss"], b["coss"], b["crss"]


def threshold_voltages(vgs, ids):
    """Constant-current Vth and gm-max linear-extrapolation Vth."""
    on = ids > 1e-9
    vth_cc = np.interp(I_VTH, ids[on], vgs[on])
    gm = np.gradient(ids, vgs)
    i = int(np.argmax(gm))
    vth_ex = vgs[i] - ids[i] / gm[i]
    return vth_cc, vth_ex


def ron_mohm(vds, ids, v_probe=0.1):
    i = int(np.argmin(np.abs(vds - v_probe)))
    return vds[i] / ids[i] * 1e3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-sim", action="store_true", help="reuse existing raw data")
    args = ap.parse_args()
    if not args.skip_sim:
        run_ngspice()
    IMAGES.mkdir(parents=True, exist_ok=True)

    # ---------------- I-V ----------------
    vg, id_lin = load_dc("transfer_vds_0.1_.dat")
    _, id_hi = load_dc("transfer_vds_5_.dat")
    vth_cc, vth_ex = threshold_voltages(vg, id_lin)

    out = {v: load_dc(f"output_vgs_{v}_.dat") for v in VGS_FAMILY}
    ron = {v: ron_mohm(*out[v]) for v in VGS_FAMILY}
    vds5, id5 = out[5]
    idsat5 = id5[int(np.argmin(np.abs(vds5 - 5.0)))]
    vrev, irev = load_dc("reverse_vgs0.dat")
    vsd_rev = -np.interp(-I_REV, irev, vrev)          # irev increases toward 0

    fig, ax = plt.subplots(2, 2, figsize=(10, 7.5))
    a = ax[0, 0]
    a.semilogy(vg, np.maximum(id_lin, 1e-12), label="Vds = 0.1 V")
    a.semilogy(vg, np.maximum(id_hi, 1e-12), label="Vds = 5 V")
    a.axvline(vth_cc, color="k", ls=":", lw=0.8)
    a.set(xlabel="Vgs (V)", ylabel="Id (A)", ylim=(1e-9, 1e3), title="Transfer curve")
    a.text(vth_cc + 0.05, 2e-9, f"Vth(1 mA) = {vth_cc:.2f} V", fontsize=8)
    a = ax[0, 1]
    for v in VGS_FAMILY:
        a.plot(*out[v], label=f"Vgs = {v} V")
    a.set(xlabel="Vds (V)", ylabel="Id (A)", title="Output characteristics")
    a = ax[1, 0]
    a.plot(vrev, -irev)
    a.axhline(I_REV, color="k", ls=":", lw=0.8)
    a.set(xlabel="Vds (V)", ylabel="-Id (A)", title="Reverse conduction, Vgs = 0 V")
    a = ax[1, 1]
    a.plot(VGS_FAMILY, [ron[v] for v in VGS_FAMILY], "o-")
    a.set(xlabel="Vgs (V)", ylabel="Rds(on) (mohm)",
          title="On-resistance (Vds = 0.1 V)")
    for x in ax.flat:
        x.grid(True, alpha=0.3)
    for x in (ax[0, 0], ax[0, 1]):
        x.legend()
    fig.tight_layout()
    fig.savefig(IMAGES / "gan_iv.png", dpi=200)

    # ---------------- C-V ----------------
    v, ciss, coss, crss = load_cv()
    vv = np.concatenate([[0.0], v])                  # extend flat to 0 V
    cc = np.concatenate([[coss[0]], coss])
    grid = np.linspace(0, vv[-1], 5001)
    c_grid = np.interp(grid, vv, cc)
    q_cum = np.concatenate([[0], np.cumsum(0.5 * (c_grid[1:] + c_grid[:-1]) * np.diff(grid))])
    e_cum = np.concatenate([[0], np.cumsum(0.5 * (c_grid[1:] * grid[1:] + c_grid[:-1] * grid[:-1])
                                           * np.diff(grid))])
    at = lambda arr_x, arr_y: float(np.interp(V_RATED_EVAL, arr_x, arr_y))
    qoss = at(grid, q_cum)
    eoss = at(grid, e_cum)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    a = ax[0]
    a.semilogy(v, ciss * 1e12, label="Ciss")
    a.semilogy(v, coss * 1e12, label="Coss")
    a.semilogy(v, crss * 1e12, label="Crss")
    a.set(xlabel="Vds (V)", ylabel="Capacitance (pF)", title="C-V at 1 MHz, Vgs = 0 V")
    a.legend()
    a = ax[1]
    a.plot(grid, q_cum * 1e9, color="C0")
    a.set(xlabel="Vds (V)", ylabel="Qoss (nC)", title="Output charge and energy")
    b = a.twinx()
    b.plot(grid, e_cum * 1e6, color="C3")
    b.set_ylabel("Eoss (uJ)", color="C3")
    for x in ax:
        x.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(IMAGES / "gan_cv.png", dpi=200)

    # ---------------- summary ----------------
    rows = [
        ("Vth_constant_current_1mA_V", vth_cc),
        ("Vth_gm_max_extrapolation_V", vth_ex),
        ("Rdson_Vgs5V_mohm", ron[5]),
        ("Rdson_Vgs3V_mohm", ron[3]),
        ("Idsat_Vgs5V_Vds5V_A", idsat5),
        (f"Vsd_reverse_at_{I_REV:g}A_Vgs0_V", vsd_rev),
        (f"Ciss_at_{V_RATED_EVAL:g}V_pF", at(v, ciss) * 1e12),
        (f"Coss_at_{V_RATED_EVAL:g}V_pF", at(v, coss) * 1e12),
        (f"Crss_at_{V_RATED_EVAL:g}V_pF", at(v, crss) * 1e12),
        (f"Qoss_0_to_{V_RATED_EVAL:g}V_nC", qoss * 1e9),
        (f"Eoss_0_to_{V_RATED_EVAL:g}V_uJ", eoss * 1e6),
    ]
    with open(ROOT / "results" / "summary.csv", "w") as f:
        f.write("metric,value\n")
        for name, val in rows:
            f.write(f"{name},{val:.4g}\n")
    print("\nmetric,value")
    for name, val in rows:
        print(f"{name},{val:.4g}")


if __name__ == "__main__":
    main()
