# DTMFRec: Dynamic Temporal Weighted Multi-Path State Space Fusion for Sequential Recommendation
## Configuration of the environment
 The detailed environment setting can be found in the `environment.yml`.
- Hardware:
  - GPU: only 4090
  - CUDA: 12.4
- Software:
  - Python: 3.10.13
  - Pytorch: 2.4.0 + cu124
- packages
  
  ```python
  pip install causal-conv1d==1.5.0.post4
  
  pip install recbole==1.2.1
  
  pip install mamba-ssm==2.2.3
  ```
##  Datasets
- The raw datasets should be preprocessed using the Conversion tool provided by `https://github.com/RUCAIBox/RecSysDatasets`. After you acquire the atomic files, please put them into `dataset/<Amazon_Fashion/Amazon_Sports_and_Outdoors/Amazon_Video_Games/amazon-beauty/ml-1m/yelp>`. The Yelp Dataset can be found at `https://www.yelp.com/dataset`; The Amazon datasets can be found at `https://cseweb.ucsd.edu/jmcauley/datasets.html\#amazon_reviews`; The MovieLens-1M dataset can be found at `https://grouplens.org/datasets/movielens/`.
- Or you can directly download the atomic files of these datasets using the link provided by Recbole: `https://github.com/RUCAIBox/RecSysDatasets`.

## Model Training
- You can directly run the `model/run.py` to reproduce the training procedure.

## Acknowledgement

Our code is developed based on SIGMA: Selective Gated Mamba for Sequential Recommendation

