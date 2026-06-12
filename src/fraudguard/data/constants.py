"""Feature constants for the PaySim pipeline.

Avoids magic strings spread across subsample/features/clean/split modules.
"""

from __future__ import annotations

# Raw CSV column names (from Kaggle PaySim).
COL_STEP = "step"
COL_TYPE = "type"
COL_AMOUNT = "amount"
COL_OLD_BAL_ORIG = "oldbalanceOrg"
COL_NEW_BAL_ORIG = "newbalanceOrig"
COL_OLD_BAL_DEST = "oldbalanceDest"
COL_NEW_BAL_DEST = "newbalanceDest"
COL_IS_FRAUD = "isFraud"
COL_IS_FLAGGED_FRAUD = "isFlaggedFraud"

# Transaction types where fraud occurs in PaySim.
TYPE_TRANSFER = "TRANSFER"
TYPE_CASH_OUT = "CASH_OUT"
FRAUD_TYPES = (TYPE_TRANSFER, TYPE_CASH_OUT)

# Engineered feature names (output of features.py).
FEAT_AMOUNT_LOG = "amount_log"
FEAT_AMOUNT_Z = "amount_z"
FEAT_STEP_HOUR = "step_hour"
FEAT_STEP_DAY = "step_day"
FEAT_IS_TRANSFER = "is_transfer"
FEAT_IS_CASHOUT = "is_cashout"
FEAT_ORIG_BAL_DELTA = "orig_balance_delta"
FEAT_DEST_BAL_DELTA = "dest_balance_delta"
FEAT_BALANCE_MISMATCH = "balance_mismatch"

# Ordered list of the 9 model-input features.
FEATURE_COLUMNS = (
    FEAT_AMOUNT_LOG,
    FEAT_AMOUNT_Z,
    FEAT_STEP_HOUR,
    FEAT_STEP_DAY,
    FEAT_IS_TRANSFER,
    FEAT_IS_CASHOUT,
    FEAT_ORIG_BAL_DELTA,
    FEAT_DEST_BAL_DELTA,
    FEAT_BALANCE_MISMATCH,
)

# Pipeline knobs.
SUBSAMPLE_SIZE = 200_000
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
Z_OUTLIER_THRESHOLD = 5.0
EPSILON = 1e-9
