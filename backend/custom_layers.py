from __future__ import annotations
import tensorflow as tf

class SafeFlatten(tf.keras.layers.Flatten):
    def compute_output_spec(self, inputs):
        x = inputs
        if isinstance(x, (list, tuple)):
            if len(x) == 1:
                x = x[0]
        return super().compute_output_spec(x)

    def call(self, inputs, *args, **kwargs):
        x = inputs
        if isinstance(x, (list, tuple)):
            # Unwrap single-item sequences produced by some legacy deserializations
            if len(x) == 1:
                x = x[0]
        return super().call(x, *args, **kwargs)
