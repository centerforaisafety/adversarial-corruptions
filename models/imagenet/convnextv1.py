import timm


def get_model(args):
    return timm.create_model(args.weights, pretrained=True).eval()
