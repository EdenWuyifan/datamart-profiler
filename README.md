# datamart-profiler-aug

## 0. Background

Current workflow for datamart profiler:

process_dataset() -> process_column() -> identify_types() ->

i) regular_exp_count():
Count the regexp

Failed to detect Borough Codes.
BBL should be recognized as location
Failed to detest geom Polygons and Multi-polygons

## 1. Plans for augmentation

i) Multithreading on process_column
ii)

https://webdatacommons.org/structureddata/sotab/

# Standard classification (fast, good baseline)

python train_cta_classifier.py --mode classification --epochs 10 --batch_size 16

# Supervised contrastive learning (better embeddings)

python train_cta_classifier.py --mode contrastive --epochs 20 --temperature 0.07

# Combined training (recommended)

python train_cta_classifier.py --mode combined --epochs 15 --alpha 0.5

# With custom paths

python train_cta_classifier.py \
 --mode combined \
 --epochs 20 \
 --batch_size 32 \
 --lr 3e-5 \
 --output_dir ./model_gpu \
 --curated_path curated_spatial_cta.csv \
 --synthetic_path synthetic_df.csv

# Inference

python inference_cta.py --model_dir ./model --text "lat: 40.71, 40.72, 40.73"
python inference_cta.py --model_dir ./model_combined --column "BOROUGH" --values "Manhattan, Brooklyn, Queens"
