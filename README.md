ML Molecule Classification Project
===================================================================

Fork from the unimol project [[Paper](https://openreview.net/forum?id=6K2RM6wVqKu)], [[Uni-Mol Docking Colab](https://colab.research.google.com/github/deepmodeling/Uni-Mol/blob/main/unimol/notebooks/unimol_binding_pose_demo.ipynb)]

Authors: Guhao Feng, Yixian Xu, Jizhe Zhang

## Dependencies
[[Uni-Core](https://github.com/dptech-corp/Uni-Core})], check its Installation Documentation.

rdkit==2022.9.3, install via pip install rdkit-pypi==2022.9.3

## How to get features
------------------------------

```bash
CUDA_VISIBLE_DEVICES="0" python ./unimol/infer_feature.py --user-dir ./unimol $data_path --results-path $output_dir --num-workers 8 --batch-size 1 --task mol_finetune --task-name $data_subset --num-classes 128 --loss finetune_cross_entropy  --arch unimol_base --path $ckpt_path --only-polar 0 --dict-name 'dict.txt' --conf-size 11 --log-interval 50 --log-format simple --valid-subset test,train,valid --fp16 --fp16-init-scale 4 --fp16-scale-window 256 --classification-head-name None --mode infer

```

## Run ML Algorithm

```bash
bash ./run.sh
```
