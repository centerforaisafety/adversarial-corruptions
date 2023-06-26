import torch
import config


def get_model(args):
    from robustness.cifar_models import resnet18

    model = resnet18()
    state = torch.load(args.weights, map_location=config.device)
    try:
        model.load_state_dict(state)
    except Exception:
        if "model" in state.keys():
            state = state["model"]
        elif "state_dict" in state.keys():
            state = state["state_dict"]
        elif "model_state_dict" in state.keys():
            state = state["model_state_dict"]

        state = {k.replace("module.", ""): v for k, v in state.items()}
        state = {k.replace("classifier.", "linear."): v for k, v in state.items()}
        state = {k.replace(".downsample.", ".shortcut."): v for k, v in state.items()}
        model.load_state_dict(state)

    return model
