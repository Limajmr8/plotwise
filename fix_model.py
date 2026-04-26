"""
Patches disease_model.h5 to fix Keras 2 → Keras 3 shape compatibility issue.
Writes a fixed copy as disease_model_fixed.h5, then tests loading it.
"""
import h5py
import json
import shutil

H5_PATH = 'ml/saved_models/disease_model.h5'
FIXED_PATH = 'ml/saved_models/disease_model_fixed.h5'


def is_shape_like(lst):
    """True if lst looks like a shape tuple: [None/int, int, int, ...]"""
    return (len(lst) >= 3 and
            all(x is None or (isinstance(x, int) and x >= 0) for x in lst))


def fix_shapes(obj):
    """Recursively unwrap double-nested shapes: [[None,7,7,1280]] → [None,7,7,1280]"""
    if isinstance(obj, dict):
        return {k: fix_shapes(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        fixed = [fix_shapes(v) for v in obj]
        if len(fixed) == 1 and isinstance(fixed[0], list) and is_shape_like(fixed[0]):
            return fixed[0]
        return fixed
    return obj


shutil.copy(H5_PATH, FIXED_PATH)

with h5py.File(FIXED_PATH, 'r+') as f:
    print("Keras version in file:", f.attrs.get('keras_version', 'unknown'))
    raw = f.attrs['model_config']
    if isinstance(raw, bytes):
        raw = raw.decode('utf-8')
    config = json.loads(raw)
    fixed = fix_shapes(config)
    f.attrs['model_config'] = json.dumps(fixed)
    print("Config patched.")

import tensorflow as tf
model = tf.keras.models.load_model(FIXED_PATH, compile=False)
print("Model ok!")
print("Output shape:", model.output_shape)
