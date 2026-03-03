# Criteo Kaggle dataset specific constants.
# Keep this file as the single source of truth for vocabulary sizes so that
# training scripts and export scripts do not silently diverge.

NUM_CAT_FEATURES = 26
NUM_DENSE_FEATURES = 13

# From kaggleAdDisplayChallenge_processed.npz pre-processing stats.
# See exp/train_{torch,tensorflow}_wukong_on_criteo_kaggle_dataset.py.
NUM_SPARSE_EMBS = [
    1460,
    583,
    10131227,
    2202608,
    305,
    24,
    12517,
    633,
    3,
    93145,
    5683,
    8351593,
    3194,
    27,
    14992,
    5461306,
    10,
    5652,
    2173,
    4,
    7046547,
    18,
    15,
    286181,
    105,
    142572,
]
