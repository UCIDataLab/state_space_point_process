# Deep Continuous-Time State-Space Models for Marked Event Sequences <span style="font-size: 0.7em;">[NeurIPS 2025 Spotlight]</span>

## Proposed Model

This work developed a novel class of marked temporal point process (MTPP) models, _state-space point process_ (S2P2), inspired by linear Hawkes processes and deep state-space models. S2P2 brings linear complexity and sublinear scaling to MTPP while being highly expressive.

The implementation for our method can be found under `easy_tpp/model/torch_model/torch_s2p2.py`, with helper functions implemented in `easy_tpp/ssm/*` which contain the code for each LLH layer in the S2P2 framework. We extended the public MTPP library `EasyTPP` (ICLR 2024) to compare all models under the same pipeline. Our model has been [fully integrated](https://github.com/ant-research/EasyTemporalPointProcess/blob/main/easy_tpp/model/torch_model/torch_s2p2.py) into [EasyTPP](https://github.com/ant-research/EasyTemporalPointProcess).


## Additional Changes from EasyTPP

Major code changes:
1. Decompose log-likelihood into time- and mark-specific components for in-depth analysis.
2. Regarding RMSE, we follow [Mei and Eisner](https://arxiv.org/abs/1612.09328) and use the expected next event time as next event time predictions to minimize the Bayes risk. However, we estimate these with the trapezoidal rule rather than Monte-Carlo simulation via the thinning algorithm. In practice, we observe this etimator with much lower variance and be faster.

Other minor changes include adding the learning rate scheduler and other features to stabilize training, adding the preprocessing script under `ehrshot_processing/` to prepare an MTPP dataset derived from the [EHRSHOT](https://som-shahlab.github.io/ehrshot-website/) dataset, and changing the default directory to save models.


## Running Code
To setup the environment, perform
```bash
pip install -r requirements.txt
pip install -e .
```
Ensure your python version is at least 3.12.

Once complete, a model can be trained by using the `S2P2/examples/train.py` script, e.g.,
```bash
python train.py --config_dir ./configs/exp_config_taxi.yaml --experiment_id S2P2_train
```
to train our model (S2P2) on the Taxi dataset. This will automatically download the dataset from Huggingface for Taxi specifically. 


## Citation
If you find our `S2P2` model or this codebase useful for your research or development, please cite [our paper](https://arxiv.org/abs/2412.19634) as follows:
```
@inproceedings{chang2025deep,
  title={Deep Continuous-Time State-Space Models for Marked Event Sequences},
  author={Yuxin Chang and Alex Boyd and Cao Xiao and Taha Kass-Hout and Parminder Bhatia and Padhraic Smyth and Andrew Warrington},
  booktitle={Proceedings of the 39th Conference on Neural Information Processing Systems (NeurIPS)},
  year={2025},
}
```