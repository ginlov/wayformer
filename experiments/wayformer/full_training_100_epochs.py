from datetime import datetime
from experiments.base_experiments.wayformer_experiment import WayformerExperiment


class SanityExperiment(WayformerExperiment):
    @property
    def sanity_check(self) -> bool:
        return False

    @property
    def base_data_folder(self) -> str:
        return "/data/scenarionet_format/training/"

    @property
    def base_val_data_folder(self) -> str:
        return "/data/scenarionet_format/training/"

    @property
    def base_checkpoint_folder(self) -> str:
        return '/data/checkpoints/wayformer/'

    @property
    def wandb_runname(self) -> str:
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"full_data_{now}"

    @property
    def num_epochs(self) -> int:
        return 100

    @property
    def val_freq(self) -> int:
        return 4

    @property
    def batch_size(self) -> int:
        return 128

    def build_dataset(self, partition):
        dataset = super().build_dataset(partition)
        return dataset
