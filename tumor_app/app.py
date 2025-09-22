

import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from PIL import Image, ImageFilter, ImageOps
from scipy import ndimage
import matplotlib.pyplot as plt
import os

from monai.networks.nets import UNet
from monai.networks.layers import Norm

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- CV2 Replacement Functions ---

def cv2_cvtColor(image, conversion):
    """Replace cv2.cvtColor()"""
    if conversion == "COLOR_RGB2GRAY":
        if len(image.shape) == 3:
            return np.dot(image[...,:3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
    elif conversion == "COLOR_GRAY2BGR":
        if len(image.shape) == 2:
            return np.stack([image, image, image], axis=2)
    elif conversion == "COLOR_BGR2RGB":
        if len(image.shape) == 3:
            return image[:, :, ::-1]
    return image

def cv2_resize(image, size):
    """Replace cv2.resize()"""
    if isinstance(image, np.ndarray):
        if len(image.shape) == 2:  # Grayscale
            pil_image = Image.fromarray(image.astype('uint8'), mode='L')
        else:  # RGB/BGR
            pil_image = Image.fromarray(image.astype('uint8'))
        
        resized = pil_image.resize((size[0], size[1]), Image.LANCZOS)
        return np.array(resized)
    return image

def cv2_dilate(image, kernel, iterations=1):
    """Replace cv2.dilate()"""
    result = image.copy().astype(bool)
    for _ in range(iterations):
        result = ndimage.binary_dilation(result, structure=np.ones((3,3)))
    return result.astype(np.uint8)

def cv2_erode(image, kernel, iterations=1):
    """Replace cv2.erode()"""
    result = image.copy().astype(bool)
    for _ in range(iterations):
        result = ndimage.binary_erosion(result, structure=np.ones((3,3)))
    return result.astype(np.uint8)

def cv2_medianBlur(image, ksize):
    """Replace cv2.medianBlur()"""
    return ndimage.median_filter(image, size=ksize).astype(np.uint8)

def cv2_addWeighted(img1, alpha, img2, beta, gamma):
    """Replace cv2.addWeighted()"""
    return (alpha * img1 + beta * img2 + gamma).astype(np.uint8)

def cv2_imdecode(buf, flags):
    """Replace cv2.imdecode()"""
    import io
    image = Image.open(io.BytesIO(buf))
    if flags == 1:  # cv2.IMREAD_COLOR
        return np.array(image.convert('RGB'))
    elif flags == 0:  # cv2.IMREAD_GRAYSCALE
        return np.array(image.convert('L'))
    else:
        return np.array(image)

# --- Model Definitions ---

class DQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()
        c, h, w = input_shape
        self.conv = nn.Sequential(
            nn.Conv2d(c,16,3,2), nn.ReLU(),
            nn.Conv2d(16,32,3,2), nn.ReLU(),
            nn.Flatten()
        )
        test_in = torch.zeros(1,c,h,w)
        n_flat = self.conv(test_in).shape[1]
        self.fc = nn.Sequential(
            nn.Linear(n_flat,128), nn.ReLU(),
            nn.Linear(128, n_actions)
        )

    def forward(self,x):
        return self.fc(self.conv(x))

# --- Environment Definition ---

class TumorSegEnv:
    def __init__(self, actions=["noop", "dilate", "erode", "smooth"]):
        self.actions = actions
        self.state = None # (img, mask)
        self.gt = None # Ground truth is not needed for inference, but kept for compatibility

    def reset(self, img, initial_mask):
        self.state = (img, initial_mask)
        self.gt = np.zeros_like(img) # Dummy ground truth for inference
        return self._get_obs()

    def _get_obs(self):
        img, mask = self.state
        # Ensure mask is uint8 before stacking
        mask = mask.astype(np.uint8) # Removed * 255 as the mask should be 0 or 1 for stacking
        return np.stack([img, mask], axis=-1).astype(np.float32) # Stack as H, W, C

    def step(self, action_idx):
        img, mask = self.state
        action = self.actions[action_idx]

        refined = mask.copy()
        # Ensure mask is uint8 for morphological operations
        mask_uint8 = mask.astype(np.uint8)
        if action == "dilate":
            refined = cv2_dilate(mask_uint8, np.ones((3,3), np.uint8), iterations=1)
        elif action == "erode":
            refined = cv2_erode(mask_uint8, np.ones((3,3), np.uint8), iterations=1)
        elif action == "smooth":
            refined = cv2_medianBlur(mask_uint8, 3)

        # Reward calculation is not needed for inference
        reward = 0
        self.state = (img, refined)
        done = True
        return self._get_obs(), reward, done, {}

# --- Utility Functions ---

def preprocess_image(image):
    """Preprocesses an uploaded image for model input."""
    img = cv2_cvtColor(np.array(image), "COLOR_RGB2GRAY")
    img = cv2_resize(img, (128, 128)) / 255.0
    return img.astype(np.float32)

def preprocess_obs(obs):
    """Preprocesses the observation for DQN input."""
    # obs is expected to be (H, W, C)
    obs_t = torch.tensor(obs)
    return obs_t.permute(2, 0, 1).unsqueeze(0).float().to(device)

def unet_predict(img, unet_model, device):
    """Generates U-Net prediction."""
    unet_model.eval()
    with torch.no_grad():
        inp = torch.tensor(img).unsqueeze(0).unsqueeze(0).float().to(device)
        pred = unet_model(inp).argmax(1).cpu().numpy()[0]
    return pred

def visualize_masks(original_img, unet_mask, refined_mask):
    """Visualizes the original image with overlays."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # U-Net Prediction
    axes[0].imshow(unet_mask, cmap='gray')
    axes[0].set_title("U-Net Prediction")
    axes[0].axis('off')

    # RL Refined Mask
    axes[1].imshow(refined_mask, cmap='gray')
    axes[1].set_title("RL Refined Mask")
    axes[1].axis('off')

    # RL Refined Mask Overlay
    overlay_refined = np.zeros((original_img.shape[0], original_img.shape[1], 3), dtype=np.uint8)
    # Ensure refined_mask is treated as boolean for indexing
    overlay_refined[refined_mask > 0] = [11, 134, 184] # Dark yellow in BGR

    img_color = cv2_cvtColor(np.uint8(original_img * 255), "COLOR_GRAY2BGR")
    alpha = 0.5
    blended_image_refined = cv2_addWeighted(img_color, 1 - alpha, overlay_refined, alpha, 0)

    axes[2].imshow(cv2_cvtColor(blended_image_refined, "COLOR_BGR2RGB"))
    axes[2].set_title("RL Refined Mask Overlay")
    axes[2].axis('off')

    st.pyplot(fig)

# --- Streamlit App ---

st.title("Tumor Segmentation and Refinement")

# Load models
@st.cache_resource # Cache models for better performance
def load_models():
    try:
        # Try multiple possible paths for the model files
        possible_paths = [
            # Same directory as script
            'unet_initial.pth',
            'dqn_initial.pth',
            # Current working directory
            './unet_initial.pth',
            './dqn_initial.pth',
            # Full path in Streamlit Cloud structure
            '/mount/src/brain_tumor_segmentation/tumor_app/unet_initial.pth',
            '/mount/src/brain_tumor_segmentation/tumor_app/dqn_initial.pth',
        ]
        
        # Find the correct paths
        unet_path = None
        dqn_path = None
        
        for path in possible_paths:
            if 'unet' in path and os.path.exists(path):
                unet_path = path
                break
        
        for path in possible_paths:
            if 'dqn' in path and os.path.exists(path):
                dqn_path = path
                break
        
        if unet_path is None or dqn_path is None:
            st.error("❌ Model files not found!")
            st.write("Current working directory:", os.getcwd())
            st.write("Files in current directory:", os.listdir('.'))
            return None, None
        
        st.info(f"📁 Found UNet model at: {unet_path}")
        st.info(f"📁 Found DQN model at: {dqn_path}")
        
        # Load models
        unet_model = UNet(
            spatial_dims=2,
            in_channels=1,
            out_channels=2,
            channels=(16, 32, 64, 128, 256),
            strides=(2,2,2,2),
            num_res_units=2,
            norm=Norm.BATCH,
        ).to(device)
        unet_model.load_state_dict(torch.load(unet_path, map_location=device))
        unet_model.eval()

        input_shape_dqn = (2, 128, 128) # (channels, height, width)
        n_actions = 4 # noop, dilate, erode, smooth
        dqn_model = DQN(input_shape_dqn, n_actions).to(device)
        dqn_model.load_state_dict(torch.load(dqn_path, map_location=device))
        dqn_model.eval()

        st.success("✅ Models loaded successfully!")
        return unet_model, dqn_model
        
    except Exception as e:
        st.error(f"❌ Error loading models: {str(e)}")
        return None, None

# Load models with error handling
try:
    unet_model, dqn_model = load_models()
    
    if unet_model is None or dqn_model is None:
        st.error("❌ Failed to load models. Cannot proceed.")
        st.stop()
        
except Exception as e:
    st.error(f"Failed to initialize models: {e}")
    st.stop()

# File uploader
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "png"])

if uploaded_file is not None:
    # Read the uploaded image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2_imdecode(file_bytes, 1)  # 1 = IMREAD_COLOR

    # Preprocess the image
    preprocessed_img = preprocess_image(image)

    # Get U-Net prediction
    unet_mask = unet_predict(preprocessed_img, unet_model, device)

    # RL Refinement
    env_inference = TumorSegEnv()
    obs = env_inference.reset(preprocessed_img, unet_mask)
    obs_t = preprocess_obs(obs)

    with torch.no_grad():
        q_vals = dqn_model(obs_t)
        action = q_vals.argmax(1).item()

    new_obs, reward, done, _ = env_inference.step(action)
    refined_mask = new_obs[:,:,1] # Extract the refined mask

    # Visualize results
    visualize_masks(preprocessed_img, unet_mask, refined_mask)