"""
Copyright (C) 2017 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-ND 4.0 license (https://creativecommons.org/licenses/by-nc-nd/4.0/legalcode).
"""

from PIL import Image  # Import Pillow for image handling
import tensorflow as tf
import numpy as np
from io import BytesIO


class Logger(object):
    def __init__(self, log_dir, suffix=None):
        # Use tf.summary.create_file_writer for TensorFlow 2.x
        self.log_dir = log_dir
        self.writer = tf.summary.create_file_writer(log_dir)

    def scalar_summary(self, tag, value, step):
        # Use the writer as a context manager
        with self.writer.as_default():
            tf.summary.scalar(tag, value, step=step)

    def image_summary(self, tag, images, step):
        # Use the writer as a context manager
        with self.writer.as_default():
            for i, img in enumerate(images):
                # Convert numpy array to image using Pillow
                pil_img = Image.fromarray(np.uint8(img))

                # Save the image to a BytesIO stream
                s = BytesIO()
                pil_img.save(s, format="png")
                s.seek(0)  # Rewind the stream

                # Read the stream as a TensorFlow image
                img_tensor = tf.io.decode_png(s.read())

                # Log the image
                tf.summary.image(f'{tag}/{i}', tf.expand_dims(img_tensor, 0), step=step)

    def video_summary(self, tag, videos, step):
        # Use the writer as a context manager
        with self.writer.as_default():
            sh = list(videos.shape)
            sh[-1] = 1

            separator = np.zeros(sh, dtype=videos.dtype)
            videos = np.concatenate([videos, separator], axis=-1)

            for i, vid in enumerate(videos):
                # Concat video frames
                v = vid.transpose(1, 2, 3, 0)
                v = [np.squeeze(f) for f in np.split(v, v.shape[0], axis=0)]
                img = np.concatenate(v, axis=1)[:, :-1, :]

                # Convert numpy array to image using Pillow
                pil_img = Image.fromarray(np.uint8(img))

                # Save the image to a BytesIO stream
                s = BytesIO()
                pil_img.save(s, format="png")
                s.seek(0)  # Rewind the stream

                # Read the stream as a TensorFlow image
                img_tensor = tf.io.decode_png(s.read())

                # Log the video frame
                tf.summary.image(f'{tag}/{i}', tf.expand_dims(img_tensor, 0), step=step)

    def flush(self):
        # In TF 2.x, the writer automatically flushes
        pass
