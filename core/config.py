from argparse import ArgumentParser, BooleanOptionalAction
from constants import Constants as const


class Config(object):
    """Wrapper class for model hyperparameters."""

    def __init__(self):
        """
        Defaults
        """
        self.backbone = "egovlp"
        self.modality = "video"
        self.phase = "train"
        self.segment_length = 1

        # Use this for 1 sec video features
        self.segment_features_directory = "data/"
        self.backbone = "egovlp"

        self.ckpt_directory = "checkpoints/"
        self.split = "recordings"
        self.batch_size = 64
        self.test_batch_size = 1
        self.num_epochs = 15
        self.lr = 5e-4
        self.weight_decay = 1e-3
        self.log_interval = 5
        self.dry_run = False
        self.ckpt = None
        self.seed = 1000
        self.device = "cuda"

        self.variant = const.TRANSFORMER_VARIANT
        self.model_name = None
        self.task_name = const.ERROR_RECOGNITION
        self.error_category = None

        self.enable_wandb = False

        self.parser = self.setup_parser()
        self.args = vars(self.parser.parse_args())
        self.save_model = True
        self.__dict__.update(self.args)

    def setup_parser(self):
        """
        Sets up an argument parser
        :return:
        """
        parser = ArgumentParser(description="training code")

        # ----------------------------------------------------------------------------------------------
        # CONFIGURATION PARAMETERS
        # ----------------------------------------------------------------------------------------------

        parser.add_argument("--batch_size", type=int, default=64, help="batch size")
        parser.add_argument("--test-batch-size", type=int, default=32, help="input batch size for testing")
        parser.add_argument("--num_epochs", type=int, default=15, help="number of epochs")
        parser.add_argument("--lr", type=float, default=5e-4, help="learning rate")
        parser.add_argument("--weight_decay", type=float, default=1e-3, help="weight decay")
        parser.add_argument("--ckpt", type=str, default=None, help="checkpoint path")
        parser.add_argument("--seed", type=int, default=42, help="random seed (default: 1000)")

        parser.add_argument("--backbone", type=str, default=const.EGOVLP, help="backbone model")
        parser.add_argument("--ckpt_directory", type=str, default="/data/rohith/captain_cook/checkpoints", help="checkpoint directory")
        parser.add_argument("--split", type=str, default=const.RECORDINGS_SPLIT, help="split")
        parser.add_argument("--variant", type=str, default=const.TRANSFORMER_VARIANT, help="variant")
        parser.add_argument("--model_name", type=str, default=None, help="model name")
        parser.add_argument("--task_name", type=str, default=const.ERROR_RECOGNITION, help="task name")
        parser.add_argument("--error_category", type=str, help="error category")
        parser.add_argument("--modality", type=str, nargs="+", default=[const.VIDEO], help="audio")
        parser.add_argument("--pos_weight", type=float, default=5.0, help="positive class weight for loss function (increased from 2.5 to 5.0 on 2026-01-10 to address class imbalance)")
        parser.add_argument("--threshold", type=float, default=0.6, help="classification threshold (0.6 for step split, 0.5 for recordings split)")

        # ----------------------------------------------------------------------------------------------
        # EVALUATION / SELECTION IMPROVEMENTS
        # ----------------------------------------------------------------------------------------------

        parser.add_argument("--sweep_thresholds", "--sweep-thresholds", action=BooleanOptionalAction, default=True,
                            help="sweep decision thresholds on validation and report the best Step-Level F1")
        parser.add_argument("--sweep_min", type=float, default=0.1, help="min threshold for sweep")
        parser.add_argument("--sweep_max", type=float, default=0.9, help="max threshold for sweep")
        parser.add_argument("--sweep_step", type=float, default=0.05, help="threshold step for sweep")
        parser.add_argument("--best_metric", type=str, default="f1", choices=["f1", "auc", "pr_auc"],
                            help="metric used for best-checkpoint selection and early stopping")
        parser.add_argument("--early_stop_patience", type=int, default=4,
                            help="stop training if best_metric does not improve for N epochs (0 disables)")

        # Step aggregation (sub-step -> step) pooling strategy
        parser.add_argument("--step_pooling", type=str, default="topk", choices=["mean", "max", "topk"],
                            help="how to pool sub-step probabilities into a step probability")
        parser.add_argument("--step_topk_frac", type=float, default=0.2,
                            help="top-k fraction for step_pooling=topk (e.g., 0.2 uses top 20%% of sub-steps)")

        return parser

    def set_model_name(self, model_name):
        self.model_name = model_name

    def print_config(self):
        """
        Prints the configuration
        :return:
        """
        print("Configuration:")
        for k, v in self.__dict__.items():
            print(f"{k}: {v}")
        print("\n")
