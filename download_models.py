import os
import zipfile
from huggingface_hub import snapshot_download


def main():
    snapshot_download(repo_id="fudan-generative-ai/champ",
                      local_dir="pretrained_models/champ",
                      resume_download=True)
    if not os.path.exists("pretrained_models/champ/example_data"):
        with zipfile.ZipFile("pretrained_models/champ/example_data.zip",
                             'r') as zip_ref:
            zip_ref.extractall("pretrained_models/champ")

    snapshot_download(repo_id="runwayml/stable-diffusion-v1-5",
                      local_dir="pretrained_models/stable-diffusion-v1-5",
                      resume_download=True)

    snapshot_download(repo_id="stabilityai/sd-vae-ft-mse",
                      local_dir="pretrained_models/sd-vae-ft-mse",
                      resume_download=True)

    snapshot_download(
        repo_id="lambdalabs/sd-image-variations-diffusers",
        local_dir="pretrained_models/sd-image-variations-diffusers",
        resume_download=True)
    if not os.path.exists("pretrained_models/image_encoder"):
        os.symlink("sd-image-variations-diffusers/image_encoder",
                   "pretrained_models/image_encoder")

    if not os.path.exists("example_data"):
        os.symlink("pretrained_models/champ/example_data", "example_data")


if __name__ == '__main__':
    main()
