# Evaluation

## CIFAR-10 Classes
We provide collected simple and paraphrased prompts for CIFAR-10 in the `data` folder.
After generating images, use the following command to test the erasing performance:

**Note:** Ensure [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) is installed.

```bash
python eval_cifar10.py --folder_path path/to/images/ --prompts_path path/to/prompts --erased_concept $concept
```

- `--folder_path`: Path to the folder containing the generated images.
- `--prompts_path`: Path to the prompts file used to generate the images.
- `--erased_concept`: The concept being erased.

The detection and evaluation results will be saved under the `folder_path`.

**Example**:

```bash
# Suppose the images are generated with sample_prompts.csv using airplane-erased Receler
python eval_cifar10.py --folder_path ../test_results/receler-word_airplane/ --prompts_path data/sample_prompts.csv --erased_concept airplane
```

## Inappropriate Image Prompts (I2P) Dataset
The I2P data file, `unsafe-prompts4703.csv`, is provided in the `data` folder.
Use the model that erases `nudity` or all inappropriate concepts to generate images from it.
<br>**Note:** Following [SLD](https://arxiv.org/abs/2211.05105), make sure to pass the `--use_cuda_generator` flag when generating images with `test_receler.py`.

After generating the images, you can detect nudity/inappropriate concepts with `nudenet` and `q16`.

```bash
python q16_nudenet_detect.py --folder_path path/to/images/ --prompts_path path/to/prompts
```

- `--folder_path`: Path to the folder containing the generated images.
- `--prompts_path`: Path to the prompts file used to generate the images.

The resulting csv file would be saved as `path/to/images/q16_nudenet_result.csv`.

Finally, you can use `eval_i2p.py` and `eval_nudity.py` to calculate the score.

```bash
python eval_i2p.py --csv path/to/detection/result
python eval_nudity.py --csv path/to/detection/result
```

- `--csv`: Path to the detection result csv file.
