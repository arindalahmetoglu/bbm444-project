import random
import argparse
import copy
import itertools
import logging
import math
import os
from pathlib import Path
import cv2

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from huggingface_hub import create_repo, upload_folder
from packaging import version
from PIL import Image
from PIL.ImageOps import exif_transpose
from torch.utils.data import Dataset
import torchvision.transforms.v2 as transforms_v2
from tqdm.auto import tqdm
from transformers import AutoTokenizer, CLIPTextModel

import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionInpaintPipeline,
    UNet2DConditionModel,
    UniPCMultistepScheduler
)

from diffusers.loaders import (
    LoraLoaderMixin,
    text_encoder_lora_state_dict,
)
from diffusers.models.attention_processor import (
    AttnAddedKVProcessor,
    AttnAddedKVProcessor2_0,
    LoRAAttnAddedKVProcessor,
    LoRAAttnProcessor,
    LoRAAttnProcessor2_0,
    SlicedAttnAddedKVProcessor,
)

from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available, load_image
from diffusers.utils.import_utils import is_xformers_available


from procossing import preprocessing_test


check_min_version("0.20.1")

logger = get_logger(__name__)


def unet_attn_processors_state_dict(unet):
    attn_processors = unet.attn_processors

    attn_processors_state_dict = {}

    for attn_processor_key, attn_processor in attn_processors.items():
        for parameter_key, parameter in attn_processor.state_dict().items():
            attn_processors_state_dict[f"{attn_processor_key}.{parameter_key}"] = parameter

    return attn_processors_state_dict


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Example of the training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        required=True,
        help="A folder containing the training images.",
    )
    parser.add_argument(
        "--val_data_dir",
        type=str,
        default=None,
        required=True,
        help="A folder containing the Validation images.",
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help="Number of images that should be generated during validation.",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=100,
        help="Validation every X steps.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="RDIStitcherModel",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=10000,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--tempmodel_steps",
        type=int,
        default=500,
        help="Save a lora weight of the training state every X updates.",
    )

    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=100,
        help="Max number of checkpoints to store.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )

    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--unet_learning_rate",
        type=float,
        default=2e-4,
        help="Learning rate to use for unet.",
    )
    parser.add_argument(
        "--text_encoder_learning_rate",
        type=float,
        default=4e-5,
        help="Learning rate to use for text encoder.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--wandb_key",
        type=str,
        default=None,
        help=("If report to option is set to wandb, api-key for wandb used for login to wandb "),
    )
    parser.add_argument(
        "--wandb_project_name",
        type=str,
        default=None,
        help=("If report to option is set to wandb, project name in wandb for log tracking  "),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument(
        "--set_grads_to_none",
        action="store_true",
        help=(
            "Save more memory by using setting grads to None instead of zero. Be aware, that this changes certain"
            " behaviors, so disable this argument if it causes any problems. More info:"
            " https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html"
        ),
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=16,
        help="The dimension of the LoRA update matrices.",
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=24,
        help="The alpha constant of the LoRA update matrices.",
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.1,
        help="The dropout rate of the LoRA update matrices.",
    )
    parser.add_argument(
        "--lora_bias",
        type=str,
        default="none",
        help="The bias type of the Lora update matrices. Must be 'none', 'all' or 'lora_only'.",
    )

    parser.add_argument("--down_block_types", type=str, nargs="+", default="CrossAttnDownBlock2D",)
    parser.add_argument("--block_out_channels", type=int, nargs="+", default=320)

    parser.add_argument(
        "--identifier",
        type=str,
        default="<A>",
        help="An unique identifier to perform specific image stitching tasks.",
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args

def color_jitter(image, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1):
    """
    color jitter for uneven hue.
    """
    if brightness > 0:
        b_factor = np.random.uniform(max(0, 1 - brightness), 1 + brightness)
        image = cv2.convertScaleAbs(image, alpha=b_factor, beta=0)

    if contrast > 0:
        c_factor = np.random.uniform(max(0, 1 - contrast), 1 + contrast)
        image = cv2.convertScaleAbs(image, alpha=c_factor, beta=0)

    if saturation > 0:
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        s_factor = np.random.uniform(max(0, 1 - saturation), 1 + saturation)
        hsv_image[:, :, 1] = np.clip(hsv_image[:, :, 1] * s_factor, 0, 255)
        image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

    if hue > 0:
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h_shift = np.random.randint(-int(hue * 180), int(hue * 180) + 1)
        hsv_image[:, :, 0] = (hsv_image[:, :, 0] + h_shift) % 180
        image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

    return image

class RDIStitcherDataset(Dataset):
    def __init__(
        self,
        train_data_root,
        tokenizer,
        size=512,
        prompt="<A>",
    ):
        self.size = size
        self.tokenizer = tokenizer

        self.input1_root = os.path.join(train_data_root, "input1")
        self.input2_root = os.path.join(train_data_root, "input2")
        self.mask1_root = os.path.join(train_data_root, "mask1")
        self.mask2_root = os.path.join(train_data_root, "mask2")
        self.prompt=prompt


        self.filenames = sorted(os.listdir(self.input1_root))
        self.num_train_images = len(self.filenames)

        self.transform = transforms_v2.Compose(
            [
                transforms_v2.Resize([512, 1024]),
                transforms_v2.ToImageTensor(),
                transforms_v2.ConvertImageDtype(),
                transforms_v2.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return self.num_train_images

    def __getitem__(self, index):
        example = {}
        probability = random.random()
        probability2 = random.random()

        file_name = self.filenames[index % self.num_train_images]

        input1 = cv2.imread(os.path.join(self.input1_root, file_name))
        input2 = cv2.imread(os.path.join(self.input2_root, file_name))
        
        inputimg = random.choice([input1, input2])

        mask_name = random.choice(self.filenames)

        mask1 = cv2.imread(os.path.join(self.mask1_root, mask_name), cv2.IMREAD_GRAYSCALE)
        mask2 = cv2.imread(os.path.join(self.mask2_root, mask_name), cv2.IMREAD_GRAYSCALE)

        _, mask1 = cv2.threshold(mask1, 127, 255, cv2.THRESH_BINARY)
        _, mask2 = cv2.threshold(mask2, 127, 255, cv2.THRESH_BINARY)

        mask1 = cv2.resize(mask1,(inputimg.shape[0], inputimg.shape[1]))
        mask2 = cv2.resize(mask2, (inputimg.shape[0], inputimg.shape[1]))

        # 25% aug data
        if probability < 0.25:
            h,w = inputimg.shape[0],inputimg.shape[1]
            ys, xs = np.where(mask2 > 0)
            x_min, y_min = np.min(xs), np.min(ys)
            x_max, y_max = np.max(xs), np.max(ys)
            shift_x_pixels = random.randint(-(w-x_max), x_min)
            shift_y_pixels = random.randint(-(h-y_max), y_min)
            M = np.float32([[1, 0, shift_x_pixels], [0, 1, shift_y_pixels]])
            affinput = cv2.warpAffine(inputimg, M, (inputimg.shape[1], inputimg.shape[0]))

        emask1 = np.repeat(mask1[:, :, np.newaxis], inputimg.shape[2], axis=2)
        emask2 = np.repeat(mask2[:, :, np.newaxis], inputimg.shape[2], axis=2)


        warp1 = cv2.bitwise_and(inputimg, emask1)
        if probability < 0.25:
            warp2 = cv2.bitwise_and(affinput, emask2)
        else:
            warp2 = cv2.bitwise_and(inputimg, emask2)

        black_mask = np.zeros_like(mask1)

        leftmask = mask2
        rightmask = mask1
        leftimg = warp2
        rightimg = warp1

        kernel1 = np.ones((3, 3), np.float32)
        stitchmask = cv2.bitwise_or(mask1,mask2)
        stitchmask = cv2.bitwise_not(stitchmask)
        stitchmask = cv2.dilate(stitchmask, kernel1, iterations=1)

        rightmask = cv2.bitwise_not(rightmask)
        kernel2 = np.ones((10, 10), np.float32)
        rightmask = cv2.dilate(rightmask, kernel2, iterations=1)
        rightmask = cv2.GaussianBlur(rightmask, (15, 15), 0)
        inputmask = cv2.hconcat([black_mask, rightmask])

        if probability2 < 0.25:
            color_shift = np.random.randint(-15, 16, size=warp1.shape, dtype=np.int16)
            warp2 = color_jitter(warp2, brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1)
        
        overlap_mask = cv2.bitwise_and(mask1, mask2)
        roi = cv2.addWeighted(warp1, 0.5, warp2, 0.5, 0)
        warp2 = cv2.bitwise_and(warp2, warp2, mask=mask2)
        warp1 = cv2.bitwise_and(warp1, warp1, mask=mask1)

        img1_masked = warp1 & cv2.bitwise_not(overlap_mask)[:,:,None]
        merged = cv2.add(warp2, img1_masked, roi)

        leftmask = cv2.bitwise_or(stitchmask,mask2)
        leftimg = cv2.inpaint(merged, stitchmask, 3, cv2.INPAINT_TELEA)
        
        leftright = cv2.bitwise_and(leftimg, leftimg, mask=leftmask)

        leftright = cv2.hconcat([leftright, inputimg])

        image = Image.fromarray(cv2.cvtColor(leftright, cv2.COLOR_BGR2RGB))

        image = exif_transpose(image)
        if not image.mode == "RGB":
            image = image.convert("RGB")

        if index < len(self) - 1:
            weighting = Image.new("L", image.size)
        else:
            weighting = Image.fromarray(cv2.cvtColor(inputmask, cv2.COLOR_BGR2RGB))

        image, weighting = self.transform(image, weighting)

        example["images"], example["weightings"] = image, weighting[0:1] < 0

        mask = inputmask.astype(np.float32) / 255.0

        mask = mask[None, :, :]

        example["masks"] = torch.from_numpy(mask).float()

        example["conditioning_images"] = example["images"] * (example["masks"] < 0.5)

        train_prompt = self.prompt
        example["prompt_ids"] = self.tokenizer(
            train_prompt,
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids

        return example

def collate_fn(examples):
    input_ids = [example["prompt_ids"] for example in examples]
    images = [example["images"] for example in examples]

    masks = [example["masks"] for example in examples]
    weightings = [example["weightings"] for example in examples]
    conditioning_images = [example["conditioning_images"] for example in examples]

    images = torch.stack(images)
    images = images.to(memory_format=torch.contiguous_format).float()

    masks = torch.stack(masks)
    masks = masks.to(memory_format=torch.contiguous_format).float()

    weightings = torch.stack(weightings)
    weightings = weightings.to(memory_format=torch.contiguous_format).float()

    conditioning_images = torch.stack(conditioning_images)
    conditioning_images = conditioning_images.to(memory_format=torch.contiguous_format).float()

    input_ids = torch.cat(input_ids, dim=0)

    batch = {
        "input_ids": input_ids,
        "images": images,
        "masks": masks,
        "weightings": weightings,
        "conditioning_images": conditioning_images,
    }
    return batch

def main(args):
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_dir=logging_dir,
    )

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")
        import wandb

        wandb.login(key=args.wandb_key)
        wandb.init(project=args.wandb_project_name)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id

    # Load the tokenizer
    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, revision=args.revision, use_fast=False)
    elif args.pretrained_model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="tokenizer",
            revision=args.revision,
            use_fast=False,
        )

    # Load scheduler and models
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision)
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision)
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision)
    
    vae.requires_grad_(False)
    unet.requires_grad_(False)
    text_encoder.requires_grad_(False)

    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move vae to device and cast to weight_dtype
    vae.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")
    
    
    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        text_encoder.gradient_checkpointing_enable()

    # now we will add new LoRA weights to the attention layers
    # Set correct lora layers
    unet_lora_attn_procs = {}
    unet_lora_parameters = []

    for name, attn_processor in unet.attn_processors.items():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]

        if isinstance(attn_processor, (AttnAddedKVProcessor, SlicedAttnAddedKVProcessor, AttnAddedKVProcessor2_0)):
            lora_attn_processor_class = LoRAAttnAddedKVProcessor
        else:
            lora_attn_processor_class = (
                LoRAAttnProcessor2_0 if hasattr(F, "scaled_dot_product_attention") else LoRAAttnProcessor
            )

        module = lora_attn_processor_class(
            hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, rank=args.lora_rank, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout, bias=args.lora_bias,
        )
        unet_lora_attn_procs[name] = module
        unet_lora_parameters.extend(module.parameters())

    unet.set_attn_processor(unet_lora_attn_procs)
    text_lora_parameters = LoraLoaderMixin._modify_text_encoder(text_encoder, dtype=torch.float32, rank=args.lora_rank)


    # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            # there are only two options here. Either are just the unet attn processor layers
            # or there are the unet and text encoder atten layers
            unet_lora_layers_to_save = None
            text_encoder_lora_layers_to_save = None

            for model in models:
                if isinstance(model, type(accelerator.unwrap_model(unet))):
                    unet_lora_layers_to_save = unet_attn_processors_state_dict(model)
                elif isinstance(model, type(accelerator.unwrap_model(text_encoder))):
                    text_encoder_lora_layers_to_save = text_encoder_lora_state_dict(model)
                else:
                    raise ValueError(f"unexpected save model: {model.__class__}")

            # make sure to pop weight so that corresponding model is not saved again
            weights.pop()

            LoraLoaderMixin.save_lora_weights(
                output_dir,
                unet_lora_layers=unet_lora_layers_to_save,
                text_encoder_lora_layers=text_encoder_lora_layers_to_save,
            )

    def load_model_hook(models, input_dir):
        unet_ = None
        text_encoder_ = None
        while len(models) > 0:
            model = models.pop()

            if isinstance(model, type(accelerator.unwrap_model(unet))):
                unet_ = model
            elif isinstance(model, type(accelerator.unwrap_model(text_encoder))):
                text_encoder_ = model
            else:
                raise ValueError(f"unexpected save model: {model.__class__}")

        lora_state_dict, network_alphas = LoraLoaderMixin.lora_state_dict(input_dir)
        LoraLoaderMixin.load_lora_into_unet(lora_state_dict, network_alphas=network_alphas, unet=unet_)
        LoraLoaderMixin.load_lora_into_text_encoder(
            lora_state_dict, network_alphas=network_alphas, text_encoder=text_encoder_
        )

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.unet_learning_rate = (
            args.unet_learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

        args.text_encoder_learning_rate = (
            args.text_encoder_learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    # Optimizer creation

    optimizer = optimizer_class(
            [
                {"params": unet_lora_parameters, "lr": args.unet_learning_rate},
                {"params": text_lora_parameters, "lr": args.text_encoder_learning_rate},
            ],
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )

    # Dataset and DataLoaders creation:
    train_dataset = RDIStitcherDataset(
        train_data_root=args.train_data_dir,
        tokenizer=tokenizer,
        size=args.resolution,
        prompt=args.identifier
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=1,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    # Prepare everything with our `accelerator`.
    unet, text_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, text_encoder, optimizer, train_dataloader, lr_scheduler
        )


    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = vars(copy.deepcopy(args))
        accelerator.init_trackers("rdistitcher", config=tracker_config)

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the mos recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train()
        text_encoder.train()

        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet, text_encoder):
                # Convert images to latent space
                latents = vae.encode(batch["images"].to(dtype=weight_dtype)).latent_dist.sample()
                latents = latents * 0.18215

                # Convert masked images to latent space
                conditionings = vae.encode(batch["conditioning_images"].to(dtype=weight_dtype)).latent_dist.sample()
                conditionings = conditionings * 0.18215

                # Downsample mask and weighting so that they match with the latents
                masks, size = batch["masks"].to(dtype=weight_dtype), latents.shape[2:]
                masks = F.interpolate(masks, size=size)

                weightings = batch["weightings"].to(dtype=weight_dtype)
                weightings = F.interpolate(weightings, size=size)

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]

                # Sample a random timestep for each image
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device
                )
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Concatenate noisy latents, masks and conditionings to get inputs to unet
                inputs = torch.cat([noisy_latents, masks, conditionings], dim=1)

                # Get the text embedding for conditioning
                encoder_hidden_states = text_encoder(batch["input_ids"])[0]
                # Predict the noise residual
                model_pred = unet(
                    inputs, 
                    timesteps, 
                    encoder_hidden_states,
                    ).sample

                # Compute the diffusion loss
                assert noise_scheduler.config.prediction_type == "epsilon"
                loss = (weightings * F.mse_loss(model_pred.float(), noise.float(), reduction="none")).mean()

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = itertools.chain(
                            unet.parameters(), text_encoder.parameters()
                        )
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                if args.report_to == "wandb":
                    accelerator.print(progress_bar)
                global_step += 1

                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        # save checkpoints
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")
                        
                    if global_step % args.tempmodel_steps == 0:
                        savedir = os.path.join(args.output_dir,str(global_step))
                        os.makedirs(savedir, exist_ok=True)
                        unet = accelerator.unwrap_model(unet)
                        unet = unet.to(torch.float32)
                        unet_lora_layers = unet_attn_processors_state_dict(unet)

                        text_encoder = accelerator.unwrap_model(text_encoder)
                        text_encoder = text_encoder.to(torch.float32)
                        text_encoder_lora_layers = text_encoder_lora_state_dict(text_encoder)


                        LoraLoaderMixin.save_lora_weights(
                            save_directory=savedir,
                            unet_lora_layers=unet_lora_layers,
                            text_encoder_lora_layers=text_encoder_lora_layers,
                        )


                    if global_step % args.validation_steps == 0:
                        valsavedir = os.path.join(args.output_dir,"valresult")
                        os.makedirs(valsavedir, exist_ok=True)
                        # create pipeline
                        pipeline = StableDiffusionInpaintPipeline.from_pretrained(
                            args.pretrained_model_name_or_path,
                            unet=accelerator.unwrap_model(unet),
                            text_encoder=accelerator.unwrap_model(text_encoder),
                            revision=args.revision,
                            torch_dtype=weight_dtype,
                            )
                        # We train on the simplified learning objective. If we were previously predicting a variance, we need the scheduler to ignore it
                        scheduler_args = {}
                        if "variance_type" in pipeline.scheduler.config:
                            variance_type = pipeline.scheduler.config.variance_type
                            if variance_type in ["learned", "learned_range"]:
                                variance_type = "fixed_small"
                            scheduler_args["variance_type"] = variance_type
                        pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config, **scheduler_args)
                        pipeline = pipeline.to(accelerator.device)
                        pipeline.set_progress_bar_config(disable=True)
                        # run inference
                        generator = torch.Generator(device=accelerator.device).manual_seed(args.seed) if args.seed else None
                        imagenames = sorted(os.listdir(os.path.join(args.val_data_dir,"warp1")))
                        num_validation_images = 3
                        for i in range(num_validation_images):
                            num = random.choice(imagenames)
                            warp1 = cv2.imread(os.path.join(args.val_data_dir,"warp1",num))
                            warp2 = cv2.imread(os.path.join(args.val_data_dir,"warp2",num))
                            mask1 = cv2.imread(os.path.join(args.val_data_dir,"mask1",num), cv2.IMREAD_GRAYSCALE)
                            mask2 = cv2.imread(os.path.join(args.val_data_dir,"mask2",num), cv2.IMREAD_GRAYSCALE)

                            val_image, val_mask = preprocessing_test(warp1,warp2,mask1,mask2)

                            if val_image.mode != "RGB":
                                val_image = val_image.convert("RGB")
                            with torch.cuda.amp.autocast():
                                new_image = pipeline(prompt="<A>", 
                                                    image=val_image, 
                                                    num_inference_steps=50,
                                                    height = 512,
                                                    width =1024,
                                                    mask_image=val_mask, 
                                                    generator=generator).images[0]
                            new_image = new_image.crop((512, 0, 1024, 512))
                            new_image.save(os.path.join(valsavedir, str(global_step)+"+"+str(i)+".jpg"))


            logs = {"loss": loss.detach().item()}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

    # Save the lora layers
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = accelerator.unwrap_model(unet)
        unet = unet.to(torch.float32)
        unet_lora_layers = unet_attn_processors_state_dict(unet)

        text_encoder = accelerator.unwrap_model(text_encoder)
        text_encoder = text_encoder.to(torch.float32)
        text_encoder_lora_layers = text_encoder_lora_state_dict(text_encoder)


        LoraLoaderMixin.save_lora_weights(
            save_directory=args.output_dir,
            unet_lora_layers=unet_lora_layers,
            text_encoder_lora_layers=text_encoder_lora_layers,
        )


    accelerator.end_training()

if __name__ == "__main__":
    args = parse_args()
    main(args)
