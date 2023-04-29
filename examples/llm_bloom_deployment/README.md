# Bloom: loading large Huggingface models with constrained resources using accelerate

We dedicated on [bigscience/bloom-7b1](https://huggingface.co/bigscience/bloom-7b1) model for inference, and it's BigScience Large Open-science Open-access Multilingual Language Model.

This document briefs on serving large HG models with limited resource using accelerate. This option can be activated with `low_cpu_mem_usage=True`. The model is first created on the Meta device (with empty weights) and the state dict is then loaded inside it (shard by shard in the case of a sharded checkpoint).

### Step 1: Create Notebook server on Kubeflow UI

Create a Notebook server in Kubeflow UI using resources: 8CPUs, 16Gi memory, 1GPU, 50G disk

And select custom image, enter: `projects.registry.vmware.com/models/llm/pytorch/torchserve-notebook:latest-gpu-v0.15`

It takes some time to make the notebook serving running. You can click `CONNECT` after the staus of notebook server shows running.

### Step 2: Download model

```bash
python Download_model.py --model_name bigscience/bloom-7b1
```
The script prints the path where the model is downloaded as below.

`model/models--bigscience-bloom-7b1/snapshots/5546055f03398095e385d7dc625e636cc8910bf2/`

The downloaded model is around 14GB.

### Step 2: Compress downloaded model

**_NOTE:_** Install Zip cli tool

Navigate to the path got from the above script. In this example it is

```bash
cd model/models--bigscience-bloom-7b1/snapshots/5546055f03398095e385d7dc625e636cc8910bf2/
zip -r /home/ubuntu/serve/examples/Huggingface_Largemodels//model.zip *
cd -

```

### Step 3: Generate MAR file

Navigate up to `Huggingface_Largemodels` directory.

```bash
torch-model-archiver --model-name bloom --version 1.0 --handler custom_handler.py --extra-files model.zip,setup_config.json -r requirements.txt
```

The MAR model package is around 25GB.

**__Note__**: Modifying setup_config.json
- Enable `low_cpu_mem_usage` to use accelerate
- Recommended `max_memory` in `setup_config.json` is the max size of shard.
- Refer: https://huggingface.co/docs/transformers/main_classes/model#large-model-loading

**__Note__**: Install dependencies in advance, or torchserve start model always get timeout if you archiver model using `torch-model-archiver  --model-name bloom --version 1.0 --handler custom_handler.py --extra-files model.zip,setup_config.json -r requirements.txt`


### Step 4: Add the mar file to model store

```bash
mkdir model_store
mv bloom.mar model_store
```

### Step 5: Start torchserve

Update config.properties and start torchserve

```bash
torchserve --start --ncs --ts-config config.properties
```

### Step 5: Run inference

```bash
curl -v "http://localhost:8080/predictions/bloom" -T sample_text.txt
```

### Troubleshooting

Jupyter: XSRF cookie does not match POST

Solution: https://stackoverflow.com/questions/44088615/jupyter-xsrf-cookie-does-not-match-post
