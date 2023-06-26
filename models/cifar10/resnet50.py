import torch
import config


def get_model(args):
    from robustness.cifar_models import ResNet50

    model = ResNet50()
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

        model_state = {
            k.replace("module.model.", ""): v
            for k, v in state.items()
            if k.startswith("module.model.")
        }
        if len(model_state) > 0:
            model.load_state_dict(model_state)
        else:
            model.load_state_dict(state)

    return model
