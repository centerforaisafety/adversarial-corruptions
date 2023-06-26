import sys

sys.path.append("third_party/SlowFast")

from slowfast.config.defaults import assert_and_infer_cfg
from slowfast.utils.parser import load_config, parse_args
from slowfast.models import build_model
import slowfast.utils.checkpoint as cu
from slowfast.config.defaults import get_cfg


def load_config(path_to_config=None):
    """
    Given the arguemnts, load and initialize the configs.
    Args:
        args (argument): arguments includes `shard_id`, `num_shards`,
            `init_method`, `cfg_file`, and `opts`.
    """
    # Setup cfg.
    cfg = get_cfg()
    # Load config from cfg.
    if path_to_config is not None:
        cfg.merge_from_file(path_to_config)

    return cfg


def get_model(args):
    if "REV_VIT_S" in args.weights:
        path_to_config = "third_party/SlowFast/configs/ImageNet/REV_VIT_S.yaml"
    elif "REV_VIT_B" in args.weights:
        path_to_config = "third_party/SlowFast/configs/ImageNet/REV_VIT_B.yaml"
    elif "REV_MVIT_B" in args.weights:
        path_to_config = "third_party/SlowFast/configs/ImageNet/REV_MVIT_B_16_CONV.yaml"
    else:
        raise ValueError(f"Weights not supported {args.weights}")

    cfg = load_config(path_to_config)
    cfg.TRAIN.CHECKPOINT_FILE_PATH = args.weights

    cfg.NUM_GPUS = 1
    cfg = assert_and_infer_cfg(cfg)

    model = build_model(cfg)
    checkpoint_epoch = cu.load_checkpoint(
        cfg.TRAIN.CHECKPOINT_FILE_PATH,
        model,
        cfg.NUM_GPUS > 1,
        None,
        scaler if cfg.TRAIN.MIXED_PRECISION else None,
        inflation=cfg.TRAIN.CHECKPOINT_INFLATE,
        convert_from_caffe2=cfg.TRAIN.CHECKPOINT_TYPE == "caffe2",
        epoch_reset=cfg.TRAIN.CHECKPOINT_EPOCH_RESET,
        clear_name_pattern=cfg.TRAIN.CHECKPOINT_CLEAR_NAME_PATTERN,
        image_init=cfg.TRAIN.CHECKPOINT_IN_INIT,
    )

    return model
