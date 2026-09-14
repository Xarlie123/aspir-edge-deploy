# Regenerates the beam evaluation frames from the ASPIR generator (seed 10, speckle 0.05)
# and captures the pre-speckle intensity of each, matched to the test split. Needs the
# ASPIR virtual environment (LightPipes); run from the ASPIR source root:
#   .venv/bin/python <this file> <irbeam test_images.npz> <output dir>  -> clean_beams_test.npz
import sys, numpy as np, logging
sys.path.insert(0, "src")
from simulation_engine._1_dataset_gen.DatasetFromIRBeam import DatasetFromIRBeam
clean = []
orig_add = DatasetFromIRBeam._add_speckle_noise
def capture(self, intensity):
    c = intensity.copy()
    clean.append(c / c.max() if c.max() > 0 else c)
    return orig_add(self, intensity)
DatasetFromIRBeam._add_speckle_noise = capture
ds = DatasetFromIRBeam("irbeam", 64, 1000, seed=10, logger=logging.getLogger("x"), speckle_noise=0.05)
mats, params = ds.generate_intensity_profiles(1000)
mats = np.asarray(mats, np.float32); clean = np.asarray(clean, np.float32)
z = np.load(sys.argv[1])
orig = z["originals"].astype(np.float32)
idx = []
for k in range(orig.shape[0]):
    d = np.abs(mats - orig[k]).reshape(1000, -1).max(1)
    j = int(np.argmin(d)); idx.append(j if d[j] < 1e-6 else -1)
idx = np.array(idx); print("matched:", int((idx >= 0).sum()), "of", len(idx), "| max mismatch among matched:", float(np.abs(mats[idx[idx>=0]] - orig[idx>=0]).max()) if (idx>=0).any() else None)
np.savez_compressed(f"{sys.argv[2]}/clean_beams_test.npz", clean=clean[idx], idx=idx, orig=orig)
print("saved")
