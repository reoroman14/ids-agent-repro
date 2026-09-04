"""Full-scale CIC-IoT-2023 baseline: all six classifiers, all 63 files."""
import os, sys, time, warnings
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import loader as L
from src.data import preprocessing as P
from src.classifiers import train_eval as TE
from src.evaluation import baseline_report as BR

OUT = "experiments/phase1_cic_iot"
started = time.time()

cfg = L.load_config()
mcfg = TE.load_models_config()
print(f"mlp.max_iter = {mcfg['mlp']['max_iter']}")
print(f"caps         = {mcfg['training']['max_train_rows']}")
print(f"holdout      = {cfg['split']['holdout_classes']} "
      f"(empty = standard baseline)\n", flush=True)

t0 = time.time()
# use_cache=True writes data/processed/cic_iot_2023.parquet so any rerun skips
# the 4-minute streaming pass. Pass rebuild_cache=True if the cap/seed/labels change.
df = L.load_dataset(L.CIC, cfg, use_cache=True)
print(f"loaded {df.shape} in {time.time()-t0:.0f}s", flush=True)
print(df.attrs["load_report"].summary(), flush=True)

t0 = time.time()
sp = P.prepare_splits(df, cfg, mcfg)
print(f"\nprepared in {time.time()-t0:.0f}s", flush=True)
print(sp.report.summary(), flush=True)
print(sp.summary(), flush=True)

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message=".*max_train_rows.*")
    table = TE.run_all(L.CIC, cfg, mcfg, splits=sp, output_dir=OUT, verbose=True)

pd.set_option("display.width", 250)
print("\n================ RESULTS ================")
print(table.to_string(index=False))

report = BR.generate_report({L.CIC: table}, os.path.join(OUT, "baseline_report.md"))
print("\n" + "\n".join(report.splitlines()[:22]))

print(f"\nTOTAL WALL CLOCK: {(time.time()-started)/60:.1f} min")
print(f"artifacts -> {OUT}")
