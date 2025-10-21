import logging
from abc import abstractmethod

from easy_tpp.preprocess import TPPDataLoader
from easy_tpp.utils import Registrable, Timer, logger, get_unique_id, LogConst, get_stage, RunnerPhase
import time


class Runner(Registrable):
    """Registrable Base Runner class.
    """

    def __init__(
            self,
            runner_config,
            unique_model_dir=False,
            **kwargs):
        """Initialize the base runner.

        Args:
            runner_config (RunnerConfig): config for the runner.
            unique_model_dir (bool, optional): whether to give unique dir to save the model. Defaults to False.
        """
        self.runner_config = runner_config
        # re-assign the model_dir
        if unique_model_dir:
            runner_config.model_dir = runner_config.base_config.specs['saved_model_dir'] + '_' + get_unique_id()

        self.save_log()

        skip_data_loader = kwargs.get('skip_data_loader', False)
        if not skip_data_loader:
            # build data reader
            data_config = self.runner_config.data_config
            backend = self.runner_config.base_config.backend
            kwargs = self.runner_config.trainer_config.get_yaml_config()
            self._data_loader = TPPDataLoader(
                data_config=data_config,
                backend=backend,
                **kwargs
            )

            # Used for warm up
            runner_config.trainer_config.set('epoch_len', len(self._data_loader.train_loader()))

            # Needed for Intensity Free model
            mean_log_inter_time, std_log_inter_time, min_dt, max_dt = (
                self._data_loader.train_loader().dataset.get_dt_stats())
            assert (min_dt > 0.)
            runner_config.model_config.set("mean_log_inter_time", mean_log_inter_time)
            runner_config.model_config.set("std_log_inter_time", std_log_inter_time)
        self.timer = Timer()

    @staticmethod
    def build_from_config(runner_config, unique_model_dir=False, **kwargs):
        """Build up the runner from runner config.

        Args:
            runner_config (RunnerConfig): config for the runner.
            unique_model_dir (bool, optional): whether to give unique dir to save the model. Defaults to False.

        Returns:
            Runner: the corresponding runner class.
        """
        runner_cls = Runner.by_name(runner_config.base_config.runner_id)
        return runner_cls(runner_config, unique_model_dir=unique_model_dir, **kwargs)

    def get_config(self):
        return self.runner_config

    def set_model_dir(self, model_dir):
        self.runner_config.base_config.specs['saved_model_dir'] = model_dir

    def get_model_dir(self):
        return self.runner_config.base_config.specs['saved_model_dir']

    def train(
            self,
            train_loader=None,
            valid_loader=None,
            test_loader=None,
            **kwargs
    ):
        """Train the model.

        Args:
            train_loader (EasyTPP.DataLoader, optional): data loader for train set. Defaults to None.
            valid_loader (EasyTPP.DataLoader, optional): data loader for valid set. Defaults to None.
            test_loader (EasyTPP.DataLoader, optional): data loader for test set. Defaults to None.

        Returns:
            model: _description_
        """
        # no train and valid loader from outside
        if train_loader is None and valid_loader is None:
            train_loader = self._data_loader.train_loader()
            valid_loader = self._data_loader.valid_loader()

        # no test loader from outside and there indeed exits test data in config
        if test_loader is None and self.runner_config.data_config.test_dir is not None:
            test_loader = self._data_loader.test_loader()

        logger.info(f'Data \'{self.runner_config.base_config.dataset_id}\' loaded...')

        timer = self.timer
        timer.start()
        model_id = self.runner_config.base_config.model_id
        logger.info(f'Start {model_id} training...')

        t0 = time.perf_counter()
        best_ll, best_epochs, best_metrics, num_params, losses = self._train_model(
            train_loader,
            valid_loader,
            test_loader=test_loader,
            **kwargs
        )
        t1 = time.perf_counter()
        logger.info(f'End {model_id} train! Cost time: {timer.end()}')
        train_log = {
            'dataset': self.runner_config.base_config.dataset_id,
            'num_params': num_params,
            'best_valid_ll': best_ll,
            'best_valid_epoch': best_epochs,
            'best_metrics': best_metrics,
            'train_time': t1-t0,
            'losses': losses
        }
        return train_log

    def evaluate(self, test_loader=None, **kwargs):
        if test_loader is None:
            test_loader = self._data_loader.test_loader()

        logger.info(f'Data \'{self.runner_config.base_config.dataset_id}\' loaded...')

        timer = self.timer
        timer.start()
        model_id = self.runner_config.base_config.model_id
        logger.info(f'Start {model_id} evaluation...')

        metrics = self._evaluate_model(
            test_loader,
            **kwargs
        )
        logger.info(f'End {model_id} evaluation! Cost time: {timer.end()}')
        return metrics

    def gen(self, gen_loader=None, **kwargs):  # We did not use this in S2P2 paper.
        if gen_loader is None:
            gen_loader = self._data_loader.test_loader()

        logger.info(f'Data \'{self.runner_config.base_config.dataset_id}\' loaded...')

        timer = self.timer
        timer.start()
        model_name = self.runner_config.base_config.model_id
        logger.info(f'Start {model_name} evaluation...')

        model = self._gen_model(
            gen_loader,
            **kwargs
        )
        logger.info(f'End {model_name} generation! Cost time: {timer.end()}')
        return model

    @abstractmethod
    def _train_model(self, train_loader, valid_loader, **kwargs):
        pass

    @abstractmethod
    def _evaluate_model(self, data_loader, **kwargs):
        pass

    @abstractmethod
    def _gen_model(self, data_loader, **kwargs):
        pass

    @abstractmethod
    def _save_model(self, model_dir, **kwargs):
        pass

    @abstractmethod
    def _load_model(self, model_dir, **kwargs):
        pass

    def save_log(self):
        """Save log to local files
        """
        log_dir = self.runner_config.base_config.specs['saved_log_dir']
        fh = logging.FileHandler(log_dir)
        fh.setFormatter(logging.Formatter(LogConst.DEFAULT_FORMAT_LONG))
        logger.addHandler(fh)
        logger.info(f'Save the log to {log_dir}')
        return

    def save(
            self,
            model_dir=None,
            **kwargs
    ):
        return self._save_model(model_dir, **kwargs)

    def run(self, **kwargs):
        """Start the runner.

        Args:
            **kwargs (dict): optional params.

        Returns:
            EasyTPP.BaseModel, dict: the results of the process.
        """
        current_stage = get_stage(self.runner_config.base_config.stage)
        if current_stage == RunnerPhase.TRAIN:
            return self.train(**kwargs)
        elif current_stage == RunnerPhase.VALIDATE:
            return self.evaluate(**kwargs)
        else:
            return self.gen(**kwargs)
