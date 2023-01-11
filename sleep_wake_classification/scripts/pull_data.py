import os

if not os.getenv("SKIP_DVC_CHECKS", False):
    # make sure dataset and RF weights are downloaded
    from actihealth.datasets.datasets_loader import load_PSG_Newcastle
    from actihealth.models.models_loader import load_RF_sleep

    load_PSG_Newcastle(do_yield=False)
    load_RF_sleep()
