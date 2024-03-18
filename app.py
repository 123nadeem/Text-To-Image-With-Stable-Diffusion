import streamlit as st
import torch
from torch import autocast
from diffusers import StableDiffusionPipeline

# Import for handling potential authentication issues
from huggingface_hub import cached_download

# Function to handle authentication errors (optional)
def download_or_load_model(model_id, auth_token=".........."):
    try:
        # Try BFloat16 if hardware supports it efficiently (adjust based on your system)
        return StableDiffusionPipeline.from_pretrained(model_id, revision="fp16", torch_dtype=torch.bfloat16, use_auth_token=auth_token)
    except Exception as e:
        if "expected scalar type BFloat16" in str(e):
            st.warning("BFloat16 not supported, using Float32.")
            return StableDiffusionPipeline.from_pretrained(model_id, revision="fp16", torch_dtype=torch.float32, use_auth_token=auth_token)
        else:
            if "TokenNotFoundError" in str(e):
                st.error("Authentication error: Please ensure you have a valid Hugging Face Hub token.")
                st.text("You can obtain a token from https://huggingface.co/settings/tokens")
            else:
                raise e  # Re-raise other exceptions

st.title("Text to Image App")

# Load the Stable Diffusion model (handling potential authentication issues)
model_id = "CompVis/stable-diffusion-v1-4"
device = "cuda" if torch.cuda.is_available() else "cpu"
stable_diffusion_model = download_or_load_model(model_id)

if stable_diffusion_model is None:
    st.stop()  # Exit the app if model loading fails

stable_diffusion_model.to(device)

# Define a function to generate the image
def generate_image(prompt):
    with autocast(device):
        image = stable_diffusion_model(prompt, guidance_scale=8.5)["sample"][0]
    return image

# Create input and output elements in Streamlit
prompt = st.text_input("Enter a text prompt:", max_chars=256)

if st.button("Generate Image"):
    with st.spinner("Generating image..."):
        image = generate_image(prompt)
        st.image(image, caption=f"Generated image for prompt: {prompt}")
