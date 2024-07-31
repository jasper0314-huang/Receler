import argparse
import os
import json
import warnings
import datetime
import pandas as pd
import torch
from pathlib import Path
from diffusers import LMSDiscreteScheduler, StableDiffusionPipeline
from accelerate import PartialState
from receler.erasers.diffusers_erasers import inject_eraser


def parse_specify(specify_classes):
    # Format: "cls_1:num_1,cls_2:num2,..."
    clusters = [cn.split(':') for cn in specify_classes.split(',')]
    return {c: int(n) for c, n in clusters}


def generate_images(model_name_or_path, prompts_path, save_folder,
                    guidance_scale=7.5, image_size=512, ddim_steps=50,
                    num_samples=5, use_cuda_generator=False, specify_classes=None, log_sep=10):

    # scheduler used in ESD and UCE
    scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)

    if not os.path.exists(model_name_or_path):
        # try loading the pretrained pipeline hosted on the Hub
        pipeline = StableDiffusionPipeline.from_pretrained(model_name_or_path, scheduler=scheduler, safety_checker=None)
    else:
        # load receler checkpoint
        pipeline = StableDiffusionPipeline.from_pretrained('CompVis/stable-diffusion-v1-4', scheduler=scheduler, safety_checker=None)
        eraser_ckpt_path = os.path.join(model_name_or_path, f'eraser_weights.pt')
        eraser_config_path = os.path.join(model_name_or_path, f'eraser_config.json')
        with open(eraser_config_path) as f:
            eraser_config = json.load(f)
        # inject erasers into pretrained SD
        inject_eraser(pipeline.unet, torch.load(eraser_ckpt_path, map_location='cpu'), **eraser_config)

    # prepare data
    df = pd.read_csv(prompts_path)
    data = [row for _, row in df.iterrows()]

    if specify_classes:
        class2num = parse_specify(specify_classes)
    else:
        class2num = {}

    # launch accelerate
    distributed_state = PartialState()
    device = distributed_state.device
    pipeline = pipeline.to(device)
    
    # disable tqdm progress bar
    pipeline.set_progress_bar_config(disable=True)

    if use_cuda_generator and 'cuda' in str(device):
        get_generator = lambda seed: torch.Generator(device=device).manual_seed(seed)
    else:
        if use_cuda_generator:
            warnings.warn('Warning: use_cuda_generator will be ignored because CUDA is not used.')
        get_generator = lambda seed: torch.manual_seed(seed)

    # generate and save
    with distributed_state.split_between_processes(data) as rows:
        start_t = datetime.datetime.now()
        for idx, row in enumerate(rows):
            # logging time on main process
            if idx % log_sep == 0 and distributed_state.is_main_process:
                curr_time = datetime.datetime.now().isoformat(sep=" ", timespec="seconds")
                elapsed_t = datetime.datetime.now() - start_t
                elapsed_r = idx / len(rows)
                remaining = datetime.timedelta(seconds=elapsed_t.seconds*(1/elapsed_r-1)) if elapsed_r != 0 else None
                print(f'{curr_time}, Progress: [{idx}/{len(rows)}], Remaining: {remaining}')

            if 'class' in row and getattr(row, 'class') in class2num:
                gen_num_samples = class2num[getattr(row, 'class')]
            else:
                gen_num_samples = num_samples
            prompts = [str(row.prompt)] * gen_num_samples
            seed = int(row.evaluation_seed)
            case_number = int(row.case_number)

            pil_images = pipeline(
                prompt=prompts,
                height=image_size,
                width=image_size,
                num_inference_steps=ddim_steps,
                guidance_scale=guidance_scale,
                generator=get_generator(seed),
            ).images

            for num, im in enumerate(pil_images):
                im.save(f"{save_folder}/{case_number}_{num}.png")


if __name__=='__main__':
    parser = argparse.ArgumentParser(
        prog='GenerateImages',
        description='Generate Images using Diffusers Code'
    )

    parser.add_argument('--model_name_or_path', help='model name or path to be loaded', type=str, required=True)
    parser.add_argument('--prompts_path', help='path to the CSV file with prompts', type=str, required=True)

    # Others parameters
    parser.add_argument('--save_root', help='evaluation root to save images', type=str, default='test_results/')
    parser.add_argument('--log_sep', help='log every log_sep batches', type=int, default=10)
    # Sampling parameters
    parser.add_argument('--guidance_scale', help='guidance scale to run eval', type=float, default=7.5)
    parser.add_argument('--image_size', help='image size of inference', type=int, default=512)
    parser.add_argument('--num_samples', help='number of samples per prompt', type=int, default=1)
    parser.add_argument('--ddim_steps', help='DDIM steps of inference', type=int, default=50)
    parser.add_argument('--use_cuda_generator', help='whether to initialize random generate on CUDA', action="store_true")
    # Generate different number of images for specific classes. If not specified, default to num_samples.
    parser.add_argument('--specify_classes', help='str in the format of "cls_1:num_1,cls_2:num_2,cls_3:num_3"', type=str, default=None)

    args = parser.parse_args()

    folder_name = f"{Path(args.prompts_path).stem}-{Path(args.model_name_or_path).stem}"
    save_folder = os.path.join(args.save_root, folder_name)
    os.makedirs(save_folder, exist_ok=True)
    print(f'\nGenerated images will be saved in {save_folder}\n')

    generate_images(
        args.model_name_or_path,
        args.prompts_path,
        save_folder,
        guidance_scale=args.guidance_scale,
        image_size=args.image_size,
        ddim_steps=args.ddim_steps,
        num_samples=args.num_samples,
        use_cuda_generator=args.use_cuda_generator,
        specify_classes=args.specify_classes,
        log_sep=args.log_sep,
    )
