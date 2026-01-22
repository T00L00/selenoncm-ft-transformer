# Morphological Profiling of SELENON-Congenital Myopathy

## Initial Notebook

Work for this was originally done in the `enm5310_final_project.ipynb` for a course. 

### Running the notebook locally

1. Make sure your dev environment has torch, sklearn, and all the typical computation python modules 
that you typically find on colab. 

2. Be sure to set `local = True` in the `Dataset Preparation` section before running the notebook.

3. Run the notebook.

### Running the notebook on Colab

1. Download the `data.gct` file from the repo and upload it to your google drive.

2. Set `local = False` in the `Dataset Preparation` section before running the notebook.

3. Set `gdrive_dir` to the google drive directory where you saved the `data.gct`.

4. Run the notebook.

## Structured codebase for reproducible training/inference

- `src/ftt`: training and evaluation code for feature-tokenizer transformer
- `src/xgb`: training and evaluation code for xgboost

Training code will output to `outputs/training/` directory. Inference runs will output `outputs/inference/` directory.

### Training and evaluating on a single batch dataset:

#### Training: 

`python src.xgb.train -c configs/training/<training-config-yaml>`

#### Evaluate: 

`python src.xgb.evaluate -c configs/training/<training-config-yaml> -e <experiment name generated from previous line>`

#### Shap analysis: 

`python src.xgb.run_shap -c configs/training/<training-config-yaml> -e <experiment name generated from previous line>`

### Training one batch dataset, inference on different batch dataset

#### Training
Be sure to specify in your training config yaml file the training dataset and the inference dataset!

`python src.xgb.train_cross_batch -c configs/training/<training-config-yaml>`

#### Evaluate

`python src.xgb.evaluate -c configs/training/<training-config-yaml> -e <experiment name generated from training>`

#### Inference
Be sure to specify in your inference config yaml file the aligned training dataframe and inference dataframe pickles!

`python src.xgb.cross_batch_inference -c configs/inference/<inference-config-yaml>`

