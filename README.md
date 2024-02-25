# Cog-SDXL

[![Replicate demo and cloud API](https://replicate.com/stability-ai/sdxl/badge)](https://replicate.com/stability-ai/sdxl)

This is an implementation of the [SDXL](https://github.com/Stability-AI/generative-models) as a Cog model. [Cog packages machine learning models as standard containers](https://github.com/replicate/cog).

## Development

Follow the [model pushing guide](https://replicate.com/docs/guides/push-a-model) to push your own fork of SDXL to [Replicate](https://replicate.com).

## Setup

- Install cog
- Start docker
- Make any changes you need
- Push to replicate using `cog push r8.im/vana-com/vana-sdxl-lora-inference-dev`
Note: if you are on a machine with a GPU, you could run inference locally, but if not you can make changes and push to replicate to test. 

## Deploy with SkyPilot

As an alternative to pushing to Replicate, you can deploy with [SkyPilot](https://skypilot.readthedocs.io):

- Auth with GCP ([instructions](https://skypilot.readthedocs.io/en/latest/getting-started/installation.html#google-cloud-platform-gcp))
- Create a python environment: `python -m venv main`
- Activate the environment: `source main/bin/activate`
- Install SkyPilot: `pip install -U skypilot-nightly`
- Deploy the model: `sky serve up service.yaml`
- Access the model at `http://<REPLICA_IP>:5000`, where `<REPLICA_IP>` can be found in the output of the `sky serve up` command or in `sky serve status`.
- To stop the model, run `sky serve down service.yaml`. To stop all models, run `sky serve down --all`.

## Update notes

**2021-08-12**
* Input types are inferred from input name extensions, or from the `input_images_filetype` argument
* Preprocssing are now done with fp16, and if no mask is found, the model will use the whole image

**2023-08-11**
* Default to 768x768 resolution training
* Rank as argument now, default to 32
* Now uses Swin2SR `caidas/swin2SR-realworld-sr-x4-64-bsrgan-psnr` as default, and will upscale + downscale to 768x768
