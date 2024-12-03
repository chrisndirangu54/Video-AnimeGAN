import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np
import cv2
from tqdm import tqdm
import pickle
import os

# Residual Block
class ResidualBlock(layers.Layer):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = layers.Conv2D(channels, kernel_size=3, padding='same', kernel_initializer='he_normal')
        self.in1 = layers.LayerNormalization(axis=-1)
        self.relu = layers.ReLU()
        self.conv2 = layers.Conv2D(channels, kernel_size=3, padding='same', kernel_initializer='he_normal')
        self.in2 = layers.LayerNormalization(axis=-1)

    def call(self, x):
        residual = x
        out = self.conv1(x)
        out = self.in1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.in2(out)
        return out + residual

class AnimeGANModel(tf.keras.Model):
    def __init__(self, in_channels=3, out_channels=3, num_residual_blocks=6, hidden_dim=64, input_shape=(256, 256, 3)):
        super(AnimeGANModel, self).__init__()
        self.initial = tf.keras.Sequential([
            tf.keras.layers.Conv2D(hidden_dim, kernel_size=7, strides=1, padding="same", input_shape=input_shape),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU()
        ])

        # Down-sampling layers should accept the number of channels from the initial block
        # Provide input_shape to the first Conv2D layer in _conv_down
        self.down_layers = tf.keras.Sequential([
            self._conv_down(hidden_dim, hidden_dim * 2, input_shape=self.initial.output_shape[1:]),  # Pass input_shape here
            self._conv_down(hidden_dim * 2, hidden_dim * 4, input_shape=(input_shape[0]//2, input_shape[1]//2, hidden_dim * 2))  # Pass input_shape here
        ])

        # Residual blocks
        self.residual_blocks = tf.keras.Sequential([
            ResidualBlock(hidden_dim * 4) for _ in range(num_residual_blocks)
        ])

        self.up_layers = tf.keras.Sequential([
            self._conv_up(hidden_dim * 4, hidden_dim * 2, input_shape=(input_shape[0] // 4, input_shape[1] // 4, hidden_dim * 4)),  # Provide input_shape
            self._conv_up(hidden_dim * 2, hidden_dim, input_shape=(input_shape[0] // 2, input_shape[1] // 2, hidden_dim * 2)),  # Provide input_shape
            tf.keras.layers.Conv2D(out_channels, kernel_size=7, strides=1, padding="same", activation="tanh")
        ])

    def _conv_down(self, in_channels, out_channels, input_shape):  # Add input_shape argument
        return tf.keras.Sequential([
            tf.keras.layers.Conv2D(out_channels, kernel_size=3, strides=2, padding="same", input_shape=input_shape),  # Use input_shape here
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU()
        ])


    def _conv_up(self, in_channels, out_channels, input_shape):  # Add input_shape argument
        return tf.keras.Sequential([
            tf.keras.layers.Conv2DTranspose(out_channels, kernel_size=3, strides=2, padding="same", input_shape=input_shape),  # Use input_shape here
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU()
        ])

    def call(self, x):
        x = self.initial(x)
        x = self.down_layers(x)
        x = self.residual_blocks(x)
        x = self.up_layers(x)
        return x


# Load AnimeGAN Base Model
def load_animegan_model(checkpoint_path, weights_path=None, device='cuda'):
    """
    Loads the AnimeGAN model and its weights.

    Args:
        checkpoint_path (str): Path to the model checkpoint or saved model.
        weights_path (str): Optional path to the weights file.
        device (str): Device to load the model on, default is 'cuda'.

    Returns:
        tf.keras.Model: Loaded AnimeGAN model.
    """
    base_model = AnimeGANModel()
    checkpoint = None

    try:
        # If the checkpoint_path is a directory, assume it's a TensorFlow checkpoint
        if os.path.isdir(checkpoint_path):
            # Create a checkpoint object
            checkpoint = tf.train.Checkpoint(model=base_model)
            # Restore from the checkpoint
            checkpoint.restore(tf.train.latest_checkpoint(checkpoint_path)).expect_partial()
            print("Checkpoint loaded using TensorFlow's Checkpoint.")
        else:
            raise ValueError(f"Unsupported checkpoint file format: {checkpoint_path}. "
                             f"Please provide a TensorFlow checkpoint directory, a .keras, or a .h5 file.")

    except (tf.errors.NotFoundError, ValueError) as e:
        print(f"Initial loading using TensorFlow's Checkpoint failed: {e}")
        print("Attempting alternative loading methods.")

        # If TensorFlow checkpoint loading fails, try loading as a Keras model
        if checkpoint_path.endswith(('.keras', '.h5')):
            try:
                base_model = tf.keras.models.load_model(checkpoint_path)
                print("Successfully loaded model using TensorFlow's load_model.")
            except Exception as e2:
                print(f"Error loading checkpoint after fallback: {e2}")
                raise e2
        else:
            raise ValueError(f"Unsupported checkpoint file format: {checkpoint_path}. "
                             f"Please provide a TensorFlow checkpoint directory, a .keras, or a .h5 file.")

    return base_model

# ConvLSTM Cell
class ConvLSTMCell(tf.keras.layers.Layer):
    def __init__(self, input_dim, hidden_dim, kernel_size):
        super(ConvLSTMCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.conv = layers.Conv2D(
            filters=4 * hidden_dim,
            kernel_size=self.kernel_size,
            padding="same",
            kernel_initializer="he_normal"
        )

    def call(self, inputs, states):
        h_cur, c_cur = states
        combined = tf.concat([inputs, h_cur], axis=-1)
        conv_output = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = tf.split(conv_output, num_or_size_splits=4, axis=-1)
        i = tf.sigmoid(cc_i)
        f = tf.sigmoid(cc_f)
        o = tf.sigmoid(cc_o)
        g = tf.tanh(cc_g)
        c_next = f * c_cur + i * g
        h_next = o * tf.tanh(c_next)
        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        shape = (batch_size, height, width, self.hidden_dim)
        return tf.zeros(shape), tf.zeros(shape)

# Temporal AnimeGAN
class AnimeGANGeneratorTemporal(Model):
    def __init__(self, base_generator, hidden_dim=64, num_lstm_layers=2):
        super(AnimeGANGeneratorTemporal, self).__init__()
        self.base_generator = base_generator
        self.temporal_layers = [ConvLSTMCell(input_dim=3, hidden_dim=hidden_dim, kernel_size=3) for _ in
                                range(num_lstm_layers)]
        # Add a convolutional layer to adjust the channels before passing to base_generator
        self.channel_adjust = layers.Conv2D(3, kernel_size=1, activation='relu')  # 1x1 convolution to change channels

    def call(self, x, prev_states=None):
        batch_size, height, width, _ = x.shape
        if prev_states is None:
            # Initialize prev_states as a list of lists with actual tensors (not strings)
            prev_states = [[layer.init_hidden(batch_size, (height, width))[0], layer.init_hidden(batch_size, (height, width))[1]] for layer in self.temporal_layers]
        else:
            # Ensure prev_states is a list of lists, not just a list containing strings
            if not all(isinstance(state, list) and len(state) == 2 and all(isinstance(tensor, tf.Tensor) for tensor in state) for state in prev_states):
                # Recreate prev_states if it's not in the expected format
                prev_states = [[layer.init_hidden(batch_size, (height, width))[0], layer.init_hidden(batch_size, (height, width))[1]] for layer in self.temporal_layers] 


        current_states = []
        for i, layer in enumerate(self.temporal_layers):
            # Pass the hidden and cell states as separate arguments
            x, new_state = layer(x, [prev_states[i][0], prev_states[i][1]])
            current_states.append(new_state)

        # Adjust the channels of x before passing to base_generator
        x = self.channel_adjust(x)  # Adjust channels to 3

        output = self.base_generator(x)
        return output, current_states

# Temporal Consistency Loss
def temporal_consistency_loss(current_frame, warped_previous_frame):
    return tf.reduce_mean(tf.square(current_frame - warped_previous_frame))

# Optical Flow Computation
def compute_optical_flow(prev_frame, curr_frame):
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    return flow

def warp_frame_with_flow(frame, flow):
    h, w = flow.shape[:2]
    flow_map = np.column_stack((np.tile(np.arange(w), h), np.repeat(np.arange(h), w))) + flow.reshape(-1, 2)
    flow_map = flow_map.reshape(h, w, 2).astype(np.float32)
    warped = cv2.remap(frame, flow_map, None, cv2.INTER_LINEAR)
    return warped

# Video Processing Function
def process_video(input_video_path, output_video_path, animegan_model, batch_size=8, overlap=3):
    cap = cv2.VideoCapture(input_video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    frames = []
    states = None
    prev_anime_frame = None
    prev_original_frame = None

    for _ in tqdm(range(total_frames), desc="Processing Frames"):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

        if len(frames) == batch_size or _ == total_frames - 1:
            anime_frames = []
            for i, frame in enumerate(frames):
                input_tensor = tf.convert_to_tensor(frame / 255.0, dtype=tf.float32)
                input_tensor = tf.expand_dims(input_tensor, axis=0)
                output, states = animegan_model(input_tensor, states)
                anime_frame = tf.squeeze(output, axis=0).numpy()

                if prev_anime_frame is not None and prev_original_frame is not None:
                    flow = compute_optical_flow(prev_original_frame, frame)
                    warped_prev_frame = warp_frame_with_flow(prev_anime_frame, flow)
                    consistency_loss = temporal_consistency_loss(anime_frame, warped_prev_frame)
                    print(f"Consistency Loss: {consistency_loss.numpy()}")

                prev_anime_frame = anime_frame
                prev_original_frame = frame
                anime_frames.append((anime_frame * 255).astype(np.uint8))

            for aframe in anime_frames:
                out.write(cv2.cvtColor(aframe, cv2.COLOR_RGB2BGR))

            frames = frames[-overlap:]

    cap.release()
    out.release()
    print("Video processing complete.")

# Example Usage
if __name__ == "__main__":
    animegan_base = AnimeGANModel(input_shape=(256, 256, 3))
    sample_input = tf.random.normal([1, 256, 256, 3])
    output = animegan_base(sample_input)
    animegan_temporal = AnimeGANGeneratorTemporal(base_generator=animegan_base)

    process_video(
        input_video_path='/content/02.mp4',
        output_video_path='path/to/output/video.mp4',
        animegan_model=animegan_temporal
    )
    # Paths
    checkpoint_path = "/content/"
    weights_path = "path/to/weights.h5"

    # Load Model
    model = load_animegan_model(checkpoint_path, weights_path=weights_path)
