import argparse
import pickle

import numpy as np
import torch
from easy_tpp.config_factory import Config
from easy_tpp.runner import Runner
from easy_tpp.utils import set_device
from sklearn.metrics import top_k_accuracy_score
from tqdm import tqdm


def predict_next_event(
    data_loader, model_runner, num_marks, device, fullynn_flag=False
):
    all_dtime = []
    all_dtime_pred = []  # raw results

    all_labels = []
    all_labels_score = []

    for batch in tqdm(data_loader):
        batch = batch.to(device).values()
        label_dtime, label_type = batch[1][:, 1:], batch[2][:, 1:]
        mask = batch[3][:, 1:]
        mask[batch[2][:, 1:] == num_marks] = (
            False  # avoid grading right window events if padded
        )

        # We used trapezoidal rule to numerically estimate the _expected_ next event time
        # pred_dtime: [batch_size, seq_len, 1]
        # pred_type: [batch_size, seq_len, num_marks]
        pred_dtime, pred_type = model_runner.model.predict_one_step_at_every_event(
            batch=batch, get_raw_pred_next_time=True, get_raw_mark_distribution=True
        )

        all_dtime.extend(
            torch.masked_select(label_dtime, mask).cpu().numpy().reshape(-1).tolist()
        )


        all_dtime_pred.extend(
            torch.masked_select(pred_dtime, mask[..., None])
            .cpu()
            .numpy()
            # .reshape((-1, num_samples))
            .tolist()
        )

        all_labels.extend(
            torch.masked_select(label_type, mask).cpu().numpy().reshape(-1).tolist()
        )

        if fullynn_flag:
            pred_type = (
                torch.masked_select(pred_type, mask[..., None])
                .cpu()
                .detach()
                .numpy()
                .reshape((-1, num_marks))
            )
        else:
            pred_type = (
                torch.masked_select(pred_type, mask[..., None])
                .cpu()
                .numpy()
                .reshape((-1, num_marks))
            )

        all_labels_score.append(pred_type)
    return all_dtime, all_dtime_pred, all_labels, all_labels_score


def main(args, use_test_data=True, top_k_accuracy=1):
    config = Config.build_from_yaml_file(
        args.config_dir, experiment_id=args.experiment_id
    )
    device = set_device(config.trainer_config.gpu)

    model_runner = Runner.build_from_config(config)

    if use_test_data:
        data_loader = model_runner._data_loader.test_loader()
    else:
        data_loader = model_runner._data_loader.valid_loader()

    num_marks = model_runner.runner_config.data_config.data_specs.num_event_types

    if args.experiment_id == "FullyNN_eval":
        model_runner.model.train()  # gradient info needed
        all_dtime, all_dtime_pred, all_labels, all_labels_score = predict_next_event(
            data_loader, model_runner, num_marks, device, fullynn_flag=True
        )
    else:
        model_runner.model.eval()
        with torch.no_grad():
            all_dtime, all_dtime_pred, all_labels, all_labels_score = (
                predict_next_event(data_loader, model_runner, num_marks, device)
            )

    all_labels_score = np.concatenate(all_labels_score, axis=0)

    print("Saving results...")
    # eval_folder_path = "/".join(
    #     config.base_config.specs["saved_log_dir"].split("/")[:-1]
    # )
    # with open(eval_folder_path + "/true_dtime.pkl", "wb") as f:
    #     pickle.dump(np.array(all_dtime), f)

    # with open(eval_folder_path + "/pred_dtime.pkl", "wb") as f:
    #     pickle.dump(np.array(all_dtime_pred), f)

    # with open(eval_folder_path + "/true_marks.pkl", "wb") as f:
    #     pickle.dump(all_labels, f)

    # with open(eval_folder_path + "/pred_marks.pkl", "wb") as f:
    #     pickle.dump(all_labels_score, f)

    print("Computing stats...")
    all_expected_dtime_pred = np.mean(all_dtime_pred, axis=-1)
    rmse = np.sqrt(
        np.mean((np.array(all_dtime) - np.array(all_expected_dtime_pred)) ** 2)
    )
    print(f"RMSE: {rmse}")

    all_labels_pred = np.argmax(all_labels_score, axis=-1)
    acc = np.mean(np.array(all_labels) == np.array(all_labels_pred))
    print(f"Accuracy: {acc}")

    # Verify top-1 accuracy, use k=10 for EHRSHOT dataset.
    acc_k = top_k_accuracy_score(
        np.array(all_labels),
        all_labels_score,
        k=top_k_accuracy,
        labels=np.array(list(range(num_marks))),
    )
    print(f"Accuracy {top_k_accuracy}: {acc_k}")


if __name__ == "__main__":
    model_list = {
        'RMTPP': 'RMTPP_eval',
        "NHP": "NHP_eval",
        'MHP': 'MHP_eval',
        'S2P2': 'S2P2_eval',
        'SAHP': 'SAHP_eval',
        'THP': 'THP_eval',
        'AttNHP': 'AttNHP_eval',
        'IntensityFree': 'IntensityFree_eval',
    }


    for model, model_id in model_list.items():
        print(f"Current model: {model}")
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--config_dir",
            type=str,
            required=False,
            default="configs/next_event_taxi.yaml",
            help="Dir of configuration yaml to train and evaluate the model.",
        )
        parser.add_argument(
            "--experiment_id",
            type=str,
            required=False,
            default=model_id,
            help="Experiment id in the config file.",
        )
        args = parser.parse_args()
        main(args, use_test_data=True, top_k_accuracy=1)  # Set k=10 for EHRSHOT
