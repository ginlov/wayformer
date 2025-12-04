from datetime import datetime
from experiments.base_experiments.wayformer_experiment import WayformerExperiment

class SanityExperiment(WayformerExperiment):
    @property
    def base_data_folder(self) -> str:
        return "/data/scenarionet_format/training/"

    @property
    def base_val_data_folder(self) -> str:
        return "/data/scenarionet_format/training/"

    @property
    def sanity_check(self) -> bool:
        return True 

    @property
    def wandb_runname(self) -> str:
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"sanity_check_{now}"

    @property
    def num_epochs(self) -> int:
        return 2

    @property
    def val_freq(self) -> int:
        return 1

    @property
    def batch_size(self) -> int:
        return 64

    def build_dataset(self, partition):
        dataset = super().build_dataset(partition)
        return dataset
