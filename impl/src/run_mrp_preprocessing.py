import os
import config as cfg
from preprocessing import saving_preprocessed_data

os.makedirs(cfg.REAL_MOTION_PREPROCESSED_MRP_PATH, exist_ok=True)
os.makedirs(cfg.IMAGERY_MOTION_PREPROCESSED_MRP_PATH, exist_ok=True)

print("Preprocessing real motion...")
saving_preprocessed_data(cfg.REAL_MOTION_RAW_PATH, cfg.REAL_MOTION_PREPROCESSED_MRP_PATH)

print("Preprocessing imagery motion...")
saving_preprocessed_data(cfg.IMAGERY_MOTION_RAW_PATH, cfg.IMAGERY_MOTION_PREPROCESSED_MRP_PATH)

print("Done.")
