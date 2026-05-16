import src.config as cfg
from src.features_extraction import save_erd_ers_features, save_erd_ers_features_v2, save_mrp_features, save_erd_ers_bins_baseline  # noqa: F401

# ----------- ERD/ERS v1: IMAGERY MOTION ---------------
save_erd_ers_features(
    input_dir=cfg.IMAGERY_MOTION_PREPROCESSED_PATH,
    output_dir=cfg.IMAGERY_MOTION_ERD_ERS_PATH,
)

# ----------- ERD/ERS v1: REAL MOTION ------------------
save_erd_ers_features(
    input_dir=cfg.REAL_MOTION_PREPROCESSED_PATH,
    output_dir=cfg.REAL_MOTION_ERD_ERS_PATH,
)

# # ----------- ERD/ERS v2: IMAGERY MOTION ---------------
# save_erd_ers_features_v2(
#     input_dir=cfg.IMAGERY_MOTION_PREPROCESSED_MRP_PATH,
#     output_dir=cfg.IMAGERY_MOTION_ERD_ERS_PATH,
# )
#
# # ----------- ERD/ERS v2: REAL MOTION ------------------
# save_erd_ers_features_v2(
#     input_dir=cfg.REAL_MOTION_PREPROCESSED_MRP_PATH,
#     output_dir=cfg.REAL_MOTION_ERD_ERS_PATH,
# )

# # ----------- ERD/ERS v3: IMAGERY MOTION ---------------
# save_erd_ers_bins_baseline(
#     input_dir=cfg.IMAGERY_MOTION_PREPROCESSED_MRP_PATH,
#     output_dir=cfg.IMAGERY_MOTION_ERD_ERS_PATH,
# )
#
# # ----------- ERD/ERS v3: REAL MOTION ------------------
# save_erd_ers_bins_baseline(
#     input_dir=cfg.REAL_MOTION_PREPROCESSED_MRP_PATH,
#     output_dir=cfg.REAL_MOTION_ERD_ERS_PATH,
# )

# # ----------- MRP: IMAGERY MOTION ----------------------
# save_mrp_features(
#     input_dir=cfg.IMAGERY_MOTION_PREPROCESSED_MRP_PATH,
#     output_dir=cfg.IMAGERY_MOTION_MRP_PATH,
# )
#
# # ----------- MRP: REAL MOTION -------------------------
# save_mrp_features(
#     input_dir=cfg.REAL_MOTION_PREPROCESSED_MRP_PATH,
#     output_dir=cfg.REAL_MOTION_MRP_PATH,
# )
