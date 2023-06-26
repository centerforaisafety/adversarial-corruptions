from models.cifar10.wideresnetwithsilu import get_model_


def get_model(args):
    return get_model_(args, "wrn-28-10-silu")
