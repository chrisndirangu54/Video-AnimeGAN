import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import os
import numpy as np
from tqdm import tqdm
from torchvision.transforms import Compose, ToTensor, Normalize

# ConvLSTM Cell
class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True):
        super(ConvLSTMCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.bias = bias

        self.conv = nn.Conv2d(
            in_channels=input_dim + hidden_dim,
            out_channels=4 * hidden_dim,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias
        )

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        combined = torch.cat([input_tensor, h_cur], dim=1)
        conv_output = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(conv_output, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (
            torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
            torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device)
        )

# AnimeGAN Generator with Temporal Awareness
class AnimeGANGeneratorTemporal(nn.Module):
    def __init__(self, base_generator, hidden_dim=64, num_lstm_layers=2):
        super(AnimeGANGeneratorTemporal, self).__init__()
        self.base_generator = base_generator
        self.temporal_layers = nn.ModuleList([
            ConvLSTMCell(input_dim=3, hidden_dim=hidden_dim, kernel_size=3)
            for _ in range(num_lstm_layers)
        ])

    def forward(self, x, prev_states=None):
        batch_size, _, height, width = x.size()
        if prev_states is None:
            prev_states = [layer.init_hidden(batch_size, (height, width)) for layer in self.temporal_layers]

        current_states = []
        for i, layer in enumerate(self.temporal_layers):
            x, new_state = layer(x, prev_states[i])
            current_states.append(new_state)

        output = self.base_generator(x)
        return output, current_states

# Temporal Consistency Loss
def temporal_consistency_loss(current_frame, warped_previous_frame):
    return F.mse_loss(current_frame, warped_previous_frame)

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
def process_video(input_video_path, output_video_path, animegan_model, batch_size=8, overlap=3, device='cuda'):
    cap = cv2.VideoCapture(input_video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    transform = Compose([
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    frames = []
    states = None
    prev_anime_frame = None

    for _ in tqdm(range(total_frames), desc="Processing Frames"):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

        if len(frames) == batch_size or _ == total_frames - 1:
            anime_frames = []
            for i, frame in enumerate(frames):
                input_tensor = transform(frame).unsqueeze(0).to(device)
                with torch.no_grad():
                    output, states = animegan_model(input_tensor, states)
                anime_frame = output.squeeze(0).cpu().numpy().transpose(1, 2, 0)
                anime_frames.append((anime_frame * 255).astype(np.uint8))

                if prev_anime_frame is not None:
                    flow = compute_optical_flow(frames[i - 1], frames[i])
                    warped_prev = warp_frame_with_flow(prev_anime_frame, flow)
                    consistency_loss = temporal_consistency_loss(torch.tensor(anime_frame), torch.tensor(warped_prev))
                    blend_factor = np.exp(-consistency_loss.item())
                    anime_frames[i] = cv2.addWeighted(anime_frames[i], blend_factor, warped_prev, 1 - blend_factor, 0)

                prev_anime_frame = anime_frames[i]

            for aframe in anime_frames:
                out.write(cv2.cvtColor(aframe, cv2.COLOR_RGB2BGR))

            frames = frames[-overlap:]

    cap.release()
    out.release()
    print("Video processing complete.")
