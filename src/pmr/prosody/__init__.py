from pmr.prosody.prosody_v1 import PROSODY_V1_DIM, prosody_v1_dataset
from pmr.prosody.prosody_v2 import PROSODY_V2_DIM, prosody_v2_dataset
from pmr.prosody.prosody_v3 import PROSODY_V3_DIM, prosody_v3_dataset

EXTRACTORS = {
    "prosody-v1": prosody_v1_dataset,
    "prosody-v2": prosody_v2_dataset,
    "prosody-v3": prosody_v3_dataset,
}

DIMS = {
    "prosody-v1": PROSODY_V1_DIM,
    "prosody-v2": PROSODY_V2_DIM,
    "prosody-v3": PROSODY_V3_DIM,
}
