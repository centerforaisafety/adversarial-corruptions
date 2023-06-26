import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

model_maps = {
    "standard.pt": "ResNet50",
    # 'imagenet_linf_8.pt': 'ResNet50 + $L_\infty$ 8/255',
    "deepaugment.pth.tar": "ResNet50 + Deepaugment",
    "ant.pth": "ResNet50 + ANT",
    "pixmix.pth.tar": "ResNet50 + PixMix",
    "augmix.tar": "ResNet50 + AugMix",
    "deepaugment_and_augmix.pth.tar": "ResNet50 + DeepAug+AugMix",
    "stylised.tar": "ResNet50 + Stylised ImageNet",
    "mixup.pth.tar": "ResNet50 + Mixup",
    "cutmix.pth.tar": "ResNet50 + CutMix",
    "randaug.pth": "ResNet50 + RandAug",
    "moex.pt": "ResNet50 + Moex",
    "resnet50_linf_eps0.5.ckpt": "ResNet50 + $L_\infty$ 0.5/255",
    "resnet50_linf_eps1.0.ckpt": "ResNet50 + $L_\infty$ 1/255",
    "resnet50_linf_eps2.0.ckpt": "ResNet50 + $L_\infty$ 2/255",
    "resnet50_linf_eps4.0.ckpt": "ResNet50 + $L_\infty$ 4/255",
    "resnet50_linf_eps8.0.ckpt": "ResNet50 + $L_\infty$ 8/255",
    "resnet50_l2_eps0.1.ckpt": "ResNet50 + $L_2$ 0.1",
    "resnet50_l2_eps0.5.ckpt": "ResNet50 + $L_2$ 0.5",
    "resnet50_l2_eps1.ckpt": "ResNet50 + $L_2$ 1",
    "resnet50_l2_eps3.ckpt": "ResNet50 + $L_2$ 3",
    "resnet50_l2_eps5.ckpt": "ResNet50 + $L_2$ 5",
    # CIFAR-10 augmentation models
    "augmix_wrn.pt": "WRN-40-2 + Augmix",
    "cutmix_wrn.pt": "WRN-40-2 + Cutmix",
    "linf_wrn.pt": "WRN-40-2 + $L_\infty$ 8/255",
    "mixup_wrn.pt": "WRN-40-2 + Mixup",
    "pixmix.pt": "WRN-40-2 + Pixmix",
    "uniform_wrn.pt": "WRN-40-2 + Uniform noise",
    "wrn.pt": "WRN-40-2 + Stardard",
    # CIFAR-10 robust models, weights from https://github.com/MadryLab/robustness
    "cifar_nat.pt": "ResNet50",
    "cifar_l2_0_5.pt": "ResNet50 + $L_2$ 0.5",
    "cifar_l2_0_25.pt": "ResNet50 + $L_2$ 0.25",
    "cifar_l2_1_0.pt": "ResNet50 + $L_2$ 1.0",
    "cifar_linf_8.pt": "ResNet50 + $L_\infty$ 8/255",
    # [Laidlaw et al.] Perceptual Adversarial Robustness: Defense Against Unseen Threat Models
    # https://arxiv.org/pdf/2006.12655.pdf
    # Weights from https://github.com/cassidylaidlaw/perceptual-advex
    "clean.pt": "ResNet50",
    "pat_alexnet_0.5.pt": "ResNet50 + AlexNet-bounded PAT",
    "pat_self_0.25.pt": "ResNet50 + Self-bounded PAT",
    "pat_self_0.5_vr1.0.pth": "ResNet50 + Self-bounded PAT + VR 1.0",
    "pat_self_0.5_vr0.5.pth": "ResNet50 + Self-bounded PAT + VR 0.5",
    "pat_self_0.5_vr0.0.pth": "ResNet50 + Self-bounded PAT + VR 0.0",
    "pat_self_0.5_vr0.3.pth": "ResNet50 + Self-bounded PAT + VR 0.3",
    "pat_self_0.5_vr0.1.pth": "ResNet50 + Self-bounded PAT + VR 0.1",
    "pat_alexnet_0.5_vr1.0.pth": "ResNet50 + AlexNet-bounded PAT + VR 1.0",
    "pat_alexnet_0.5_vr0.5.pth": "ResNet50 + AlexNet-bounded PAT + VR 0.5",
    "pat_alexnet_0.5_vr0.1.pth": "ResNet50 + AlexNet-bounded PAT + VR 0.1",
    "pat_alexnet_0.5_vr0.3.pth": "ResNet50 + AlexNet-bounded PAT + VR 0.3",
    "pgd-l2-1200.pth": "ResNet50 + $L_2$ 1200",
    "pgd-l2-2400.pth": "ResNet50 + $L_2$ 2400",
    "pgd-l2-150.pth": "ResNet50 + $L_2$ 150",
    "pgd-l2-300.pth": "ResNet50 + $L_2$ 300",
    "pgd-l2-4800.pth": "ResNet50 + $L_2$ 4800",
    "pgd-l2-600.pth": "ResNet50 + $L_2$ 600",
    "pgd-linf-16.pth": "ResNet50 + $L_\infty$ 16",
    "pgd-linf-1.pth": "ResNet50 + $L_\infty$ 1",
    "pgd-linf-2.pth": "ResNet50 + $L_\infty$ 2",
    "pgd-linf-32.pth": "ResNet50 + $L_\infty$ 32",
    "pgd-linf-4.pth": "ResNet50 + $L_\infty$ 4",
    "pgd-linf-8.pth": "ResNet50 + $L_\infty$ 8",
    # A ConvNet for the 2020s
    # https://arxiv.org/abs/2201.03545
    # https://github.com/facebookresearch/ConvNeXt
    # weights from timm https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/convnext.py
    "convnext_base.fb_in1k": "ConvNeXt-base   Input224 ImageNet1K",
    "convnext_base.fb_in22k_ft_in1k_384": "ConvNeXt-base   Input384 ImageNet1K+22K",
    "convnext_base.fb_in22k_ft_in1k": "ConvNeXt-base   Input224 ImageNet1K+22K",
    "convnext_large.fb_in1k": "ConvNeXt-large  Input224 ImageNet1K",
    "convnext_large.fb_in22k_ft_in1k_384": "ConvNeXt-large  Input384 ImageNet1K+22K",
    "convnext_large.fb_in22k_ft_in1k": "ConvNeXt-large  Input224 ImageNet1K+22K",
    "convnext_small.fb_in1k": "ConvNeXt-small  Input224 ImageNet1K",
    "convnext_small.fb_in22k_ft_in1k_384": "ConvNeXt-small  Input384 ImageNet1K+22K",
    "convnext_small.fb_in22k_ft_in1k": "ConvNeXt-small  Input224 ImageNet1K+22K",
    "convnext_tiny.fb_in1k": "ConvNeXt-tiny   Input224 ImageNet1K",
    "convnext_tiny.fb_in22k_ft_in1k_384": "ConvNeXt-tiny   Input384 ImageNet1K+22K",
    "convnext_tiny.fb_in22k_ft_in1k": "ConvNeXt-tiny   Input224 ImageNet1K+22K",
    "convnext_xlarge.fb_in22k_ft_in1k_384": "ConvNeXt-xlarge Input384 ImageNet1K+22K",
    "convnext_xlarge.fb_in22k_ft_in1k": "ConvNeXt-xlarge Input224 ImageNet1K+22K",
    # ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders
    # https://arxiv.org/abs/2301.00808
    # https://github.com/facebookresearch/ConvNeXt-V2
    # Weights from timm https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/convnext.py
    "convnextv2_atto.fcmae_ft_in1k": "ConvNeXt-V2-atto  Input224 ImageNet1K",
    "convnextv2_femto.fcmae_ft_in1k": "ConvNeXt-V2-femto Input224 ImageNet1K",
    "convnextv2_pico.fcmae_ft_in1k": "ConvNeXt-V2-pico  Input224 ImageNet1K",
    "convnextv2_nano.fcmae_ft_in1k": "ConvNeXt-V2-nano  Input224 ImageNet1K",
    "convnextv2_nano.fcmae_ft_in22k_in1k_384": "ConvNeXt-V2-nano  Input384 ImageNet1K+22K",
    "convnextv2_nano.fcmae_ft_in22k_in1k": "ConvNeXt-V2-nano  Input224 ImageNet1K+22K",
    "convnextv2_tiny.fcmae_ft_in1k": "ConvNeXt-V2-tiny  Input224 ImageNet1K",
    "convnextv2_tiny.fcmae_ft_in22k_in1k_384": "ConvNeXt-V2-tiny  Input384 ImageNet1K+22K",
    "convnextv2_tiny.fcmae_ft_in22k_in1k": "ConvNeXt-V2-tiny  Input224 ImageNet1K+22K",
    "convnextv2_base.fcmae_ft_in1k": "ConvNeXt-V2-base  Input224 ImageNet1K",
    "convnextv2_base.fcmae_ft_in22k_in1k_384": "ConvNeXt-V2-base  Input384 ImageNet1K+22K",
    "convnextv2_base.fcmae_ft_in22k_in1k": "ConvNeXt-V2-base  Input224 ImageNet1K+22K",
    "convnextv2_large.fcmae_ft_in1k": "ConvNeXt-V2-large Input224 ImageNet1K",
    "convnextv2_large.fcmae_ft_in22k_in1k": "ConvNeXt-V2-large Input224 ImageNet1K+22K",
    "convnextv2_large.fcmae_ft_in22k_in1k_384": "ConvNeXt-V2-large Input384 ImageNet1K+22K",
    "convnextv2_huge.fcmae_ft_in22k_in1k_384": "ConvNeXt-V2-huge  Input384 ImageNet1K+22K",
    "convnextv2_huge.fcmae_ft_in22k_in1k_512": "ConvNeXt-V2-huge  Input512 ImageNet1K+22K",
    "convnextv2_huge.fcmae_ft_in1k": "ConvNeXt-V2-huge  Input224 ImageNet1K",
    # How to train your ViT? Data, Augmentation, and Regularization in Vision Transformers
    # https://arxiv.org/abs/2106.10270
    # Weights from timm https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py
    "vit_tiny_patch16_224.augreg_in21k_ft_in1k": "ViT-tiny  Patch16 Input224 ImageNet1K+22K",
    "vit_tiny_patch16_384.augreg_in21k_ft_in1k": "ViT-tiny  Patch16 Input384 ImageNet1K+22K",
    "vit_small_patch32_224.augreg_in21k_ft_in1k": "ViT-small Patch32 Input224 ImageNet1K+22K",
    "vit_small_patch32_384.augreg_in21k_ft_in1k": "ViT-small Patch32 Input384 ImageNet1K+22K",
    "vit_small_patch16_224.augreg_in21k_ft_in1k": "ViT-small Patch16 Input224 ImageNet1K+22K",
    "vit_small_patch16_384.augreg_in21k_ft_in1k": "ViT-small Patch16 Input384 ImageNet1K+22K",
    "vit_small_patch16_224.augreg_in1k": "ViT-small Patch16 Input224 ImageNet1K",
    "vit_small_patch16_384.augreg_in1k": "ViT-small Patch16 Input384 ImageNet1K",
    "vit_base_patch32_224.augreg_in21k_ft_in1k": "ViT-base  Patch32 Input224 ImageNet1K+22K",
    "vit_base_patch32_384.augreg_in21k_ft_in1k": "ViT-base  Patch32 Input384 ImageNet1K+22K",
    "vit_base_patch16_224.augreg_in21k_ft_in1k": "ViT-base  Patch16 Input224 ImageNet1K+22K",
    "vit_base_patch16_384.augreg_in21k_ft_in1k": "ViT-base  Patch16 Input384 ImageNet1K+22K",
    "vit_base_patch8_224.augreg_in21k_ft_in1k": "ViT-base  Patch8  Input224 ImageNet1K+22K",
    "vit_base_patch32_224.augreg_in1k": "ViT-base  Patch32 Input224 ImageNet1K",
    "vit_base_patch32_384.augreg_in1k": "ViT-base  Patch32 Input384 ImageNet1K",
    "vit_base_patch16_224.augreg_in1k": "ViT-base  Patch16 Input224 ImageNet1K",
    "vit_base_patch16_384.augreg_in1k": "ViT-base  Patch16 Input384 ImageNet1K",
    "vit_large_patch16_224.augreg_in21k_ft_in1k": "ViT-large Patch16 Input224 ImageNet1K+22K",
    "vit_large_patch16_384.augreg_in21k_ft_in1k": "ViT-large Patch16 Input384 ImageNet1K+22K",
    # https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/swin_transformer.py
    # https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/swin_transformer_v2.py
    "swin_tiny_patch4_window7_224": "Swin-tiny    Window7  Input224 ImageNet1K",
    "swin_small_patch4_window7_224": "Swin-small   Window7  Input224 ImageNet1K",
    "swin_base_patch4_window12_384": "Swin-base    Window12 Input384 ImageNet1K",
    "swin_base_patch4_window7_224": "Swin-base    Window7  Input224 ImageNet1K",
    "swin_large_patch4_window12_384": "Swin-large   Window12 Input384 ImageNet1K",
    "swin_large_patch4_window7_224": "Swin-large   Window7  Input224 ImageNet1K",
    # Note swinv2 are not included in the evaluation because there are no models with 224 inputs
    "swinv2_tiny_window8_256": "SwinV2-tiny  Window8  Input256 ImageNet1K",
    "swinv2_tiny_window16_256": "SwinV2-tiny  Window16 Input256 ImageNet1K",
    "swinv2_small_window8_256": "SwinV2-small Window8  Input256 ImageNet1K",
    "swinv2_small_window16_256": "SwinV2-small Window16 Input256 ImageNet1K",
    "swinv2_base_window8_256": "SwinV2-base  Window8  Input256 ImageNet1K",
    "swinv2_base_window16_256": "SwinV2-base  Window16 Input256 ImageNet1K",
    "swinv2_base_window12to16_192to256_22kft1k": "SwinV2-base  Window16 Input256 ImageNet1K+22K",
    "swinv2_base_window12to24_192to384_22kft1k": "SwinV2-base  Window24 Input384 ImageNet1K+22K",
    "swinv2_large_window12to16_192to256_22kft1k": "SwinV2-large Window16 Input256 ImageNet1K+22K",
    "swinv2_large_window12to24_192to384_22kft1k": "SwinV2-large Window24 Input384 ImageNet1K+22K",
    # Reversible Vision Transformers
    # https://arxiv.org/abs/2302.04869
    # https://github.com/facebookresearch/SlowFast
    # Weights from https://github.com/facebookresearch/SlowFast/blob/main/projects/rev/README.md
    "REV_VIT_S.pyth": "Reversible-ViT-small",
    "REV_VIT_B.pyth": "Reversible-ViT-base",
    "REV_MVIT_B.pyth": "Reversible-ViT-base multiscale",
    # Better Diffusion Models Further Improve Adversarial Training
    # https://github.com/wzekai99/DM-Improves-AT
    "cifar10_linf_wrn28-10.pt": "WRN-28-10 + $L_\infty$ 8/255 + Diffusion model",
    "cifar10_l2_wrn28-10.pt": "WRN-28-10 + $L_2$ 128/255 + Diffusion model",
    "cifar10_linf_wrn70-16.pt": "WRN-70-16 + $L_\infty$ 8/255 + Diffusion model",
    "cifar10_l2_wrn70-16.pt": "WRN-70-16 + $L_2$ 128/255 + Diffusion model",
    # Towards Robust Vision Transformer
    # https://arxiv.org/abs/2105.07926
    # Weights from https://github.com/alibaba/easyrobust/tree/main
    "advtrain_swin_small_patch4_window7_224_ep4.pth": "Swin-small   Window7  Input224 ImageNet1K + $L_\infty$ 4/255",
    "advtrain_swin_base_patch4_window7_224_ep4.pth": "Swin-base    Window7  Input224 ImageNet1K + $L_\infty$ 4/255",
    # Revisiting Adversarial Training for ImageNet: Architectures, Training and Generalization across Threat Models
    # https://arxiv.org/pdf/2303.01870.pdf
    # Weights from https://github.com/nmndeep/revisiting-at
    "vit_s_cvst_robust.pt": "ViT-small Patch16 + $L_\infty$ 4/255",
    # 'vit_m_cvst_robust.pt': '',
    "vit_b_cvst_robust.pt": "ViT-base Patch16 + $L_\infty$ 4/255",
    "convnext_iso_cvst_robust.pt": "ConvNeXt-small-Isotropic  + $L_\infty$ 4/255",
    "convnext_tiny_cvst_robust.pt": "ConvNeXt-tiny + $L_\infty$ 4/255",
    "convnext_s_cvst_robust.pt": "ConvNeXt-small + $L_\infty$ 4/255",
    "convnext_b_cvst_robust.pt": "ConvNeXt-base + $L_\infty$ 4/255",
    # Removing Batch Normalization Boosts Adversarial Training
    "NoFrost_ResNet50.pth": "ResNet50 without BN + $L_\infty$ 8/255 \cite{wang2022removing}",
    "NoFrost_star_ResNet50.pth": "ResNet50 without BN + $L_\infty$ 8/255 + data aug \cite{wang2022removing}",
    # Learning Transferable Visual Models From Natural Language Supervision
    # https://arxiv.org/abs/2103.00020
    # https://github.com/openai/CLIP/blob/main/model-card.md
    # Note that CLIP also supports other models such as resnet50
    "ViT-L-14": "CLIP (ViT-L/14)",
    # DINOv2: Learning Robust Visual Features without Supervision
    # https://arxiv.org/abs/2304.07193
    # https://github.com/facebookresearch/dinov2/issues/64
    "dinov2_vitb14": "DINOv2 ViT-base Patch14",
    "dinov2_vitl14": "DINOv2 ViT-large Patch14",
    "imagenet100_dinov2_vitb14": "DINOv2 ViT-base Patch14",
    "imagenet100_dinov2_vitl14": "DINOv2 ViT-large Patch14",
    # Masked Autoencoders Are Scalable Vision Learners
    # https://arxiv.org/abs/2111.06377
    # https://github.com/facebookresearch/mae/blob/main/FINETUNE.md
    "mae_vit_base_patch16": "MAE ViT-base Patch16",
    "mae_vit_large_patch16": "MAE ViT-large Patch16",
    # [Madaan et al.] Learning to Generate Noise for Multi-Attack Robustness
    # https://arxiv.org/pdf/2006.12135.pdf
    # https://github.com/divyam3897/MNG_AC
    "mng_ac_rst.pth": "[Madaan et al.] WRN-28-10 + Meta noise + Robust self-training",
    "mng_ac.pth": "[Madaan et al.] WRN-28-10 + Meta noise",
    "msd.pth": "[Madaan et al.] WRN-28-10 + Multi steepest descent",
    "max.pth": "[Madaan et al.] WRN-28-10 + Multi attack (maximum)",
    "avg.pth": "[Madaan et al.] WRN-28-10 + Multi attack (average)",
    # [Laidlaw et al.] Perceptual Adversarial Robustness: Defense Against Unseen Threat Models
    # https://arxiv.org/pdf/2006.12655.pdf
    # https://github.com/cassidylaidlaw/perceptual-advex
    # Models downloaded from https://perceptual-advex.s3.us-east-2.amazonaws.com/perceptual-advex-checkpoints.zip
    "pat_paper_cifar-stadv_0.05.pt": "[Laidlaw et al.] RestNet50 + StAdv",
    "pat_paper_cifar-recoloradv_0.06.pt": "[Laidlaw et al.] RestNet50 + ReColorAdv",
    "pat_paper_cifar-pgd_linf_8_pgd_l2_1_stadv_0.05_recoloradv_0.06_random.pt": "[Laidlaw et al.] RestNet50 + Multi attack (random)",
    "pat_paper_cifar-pgd_linf_8_pgd_l2_1_stadv_0.05_recoloradv_0.06_average.pt": "[Laidlaw et al.] RestNet50 + Multi attack (average)",
    "pat_paper_cifar-pgd_linf_8_pgd_l2_1_stadv_0.05_recoloradv_0.06_max.pt": "[Laidlaw et al.] RestNet50 + Multi attack (maximum)",
    "pat_paper_cifar-pat_self_0.5.pt": "[Laidlaw et al.] RestNet50 + Self-bounded PAT",
    "pat_paper_cifar-pat_alexnet_1.pt": "[Laidlaw et al.] RestNet50 + AlexNet-bounded PAT",
    # [Dai et al.] Formulating Robustness Against Unforeseen Attacks
    # https://arxiv.org/pdf/2204.13779.pdf
    # https://github.com/inspire-group/variation-regularization
    "variation_regularization-cifar10_LPIPS_eps0.5_var_0.05.pth": "[Dai et al.] ResNet50 + PAT 0.5 + VR 0.05",
    "variation_regularization-cifar10_LPIPS_eps0.5_var_0.1.pth": "[Dai et al.] ResNet50 + PAT 0.5 + VR 0.1",
    "variation_regularization-cifar10_LPIPS_eps0.5_var_0.pth": "[Dai et al.] ResNet50 + PAT 0.5",
    "variation_regularization-cifar10_LPIPS_eps1_var_0.05.pth": "[Dai et al.] ResNet50 + PAT 1.0 + VR 0.05",
    "variation_regularization-cifar10_LPIPS_eps1_var_0.1.pth": "[Dai et al.] ResNet50 + PAT 1.0 + VR 0.1",
    "variation_regularization-recolor_var_0.pth": "[Dai et al.] ResNet18 + ReColorAdv",
    "variation_regularization-recolor_var_0.5.pth": "[Dai et al.] ResNet18 + ReColorAdv + VR 0.5",
    "variation_regularization-recolor_var_1.pth": "[Dai et al.] ResNet18 + ReColorAdv + VR 1.0",
    "variation_regularization-stadv_var_0.pth": "[Dai et al.] ResNet18 + StAdv",
    "variation_regularization-stadv_var_0.5.pth": "[Dai et al.] ResNet18 + StAdv + VR 0.5",
    "variation_regularization-stadv_var_1.pth": "[Dai et al.] ResNet18 + StAdv + VR 1.0",
    "variation_regularization-cifar10_resnet18_l2_var_1.pth": "[Dai et al.] ResNet18 + $L_2$ 0.5 + VR 1.0",
    "variation_regularization-cifar10_resnet18_linf_var_0.5.pth": "[Dai et al.] ResNet18 + $L_\infty$ 8/255 + VR 0.5",
    "variation_regularization-cifar10_wrn_28_10_linf_var_0.7.pth": "[Dai et al.] WRN-28-10 + $L_\infty$ 8/255 + VR 0.7",
    # [Croce & Hein] Adversarial Robustness against Multiple and Single lp-Threat Models via Quick Fine-Tuning of Robust Classifiers
    # https://arxiv.org/pdf/2105.12508.pdf
    # https://github.com/fra31/robust-finetuning
    "robust_finetuning_pretr_L1.pth": "[Croce & Hein] PreAct ResNet18 $L_1$ pretrained",
    "robust_finetuning_pretr_L2.pth": "[Croce & Hein] PreAct ResNet18 $L_2$ pretrained",
    "robust_finetuning_pretr_Linf.pth": "[Croce & Hein] PreAct ResNet18 $L_\infty$ pretrained",
    # From https://github.com/locuslab/robust_overfitting
    "cifar10_wide10_linf_eps8.pth": "WRN-34-10 + $L_\infty$ 8/255",
    # Trained by ourselves
    "robust_overfitting_wrn2810_linf_eps8.pth": "WRN-28-10 + $L_\infty$ 8/255",
    "robust_overfitting_wrn2810_linf_eps4.pth": "WRN-28-10 + $L_\infty$ 4/255",
    "robust_overfitting_wrn2810_l2_eps0.25.pth": "WRN-28-10 + $L_2$ 0.25",
    "robust_overfitting_wrn2810_l2_eps0.5.pth": "WRN-28-10 + $L_2$ 0.5",
    "robust_overfitting_wrn2810_l2_eps1.0.pth": "WRN-28-10 + $L_2$ 1.0",
    "robust_overfitting_wrn2810_standard.pth": "WRN-28-10",
    # https://github.com/yaodongyu/TRADES
    "trades_model_cifar_wrn.pt": "WRN-34-10 + $L_\infty$ 8/255 (TRADES)",
    # Trained by ourselves with the TRADES code
    "wrn2810-cifar10-linf-8-255-trades.pt": "WRN-28-10 + $L_\infty$ 8/255 (TRADES)",
    "wrn2810-cifar10-linf-4-255-trades.pt": "WRN-28-10 + $L_\infty$ 4/255 (TRADES)",
    "wrn2810-cifar10-l2-0.25-trades.pt": "WRN-28-10 + $L_2$ 0.25 (TRADES)",
    "wrn2810-cifar10-l2-0.5-trades.pt": "WRN-28-10 + $L_2$ 0.5 (TRADES)",
    "wrn2810-cifar10-l2-1.0-trades.pt": "WRN-28-10 + $L_2$ 1.0 (TRADES)",
}

model_maps = {k: v.replace("Input224", "") for k, v in model_maps.items()}
model_maps = {k: v.replace("Window7", "") for k, v in model_maps.items()}
model_maps = {k: re.sub(r"\s+", " ", v) for k, v in model_maps.items()}
# This remove the reference square brackets
model_maps = {k: re.sub(r"\[.*?\]", "", v) for k, v in model_maps.items()}
model_maps = {k: v.strip() for k, v in model_maps.items()}
model_maps = {k: r"{}".format(v) for k, v in model_maps.items()}


imagenet_files = "standard.pt \
deepaugment.pth.tar \
ant.pth \
pixmix.pth.tar \
deepaugment_and_augmix.pth.tar \
stylised.tar \
mixup.pth.tar \
cutmix.pth.tar \
randaug.pth \
moex.pt \
augmix.tar \
resnet50_linf_eps0.5.ckpt \
resnet50_linf_eps1.0.ckpt \
resnet50_linf_eps2.0.ckpt \
resnet50_linf_eps4.0.ckpt \
resnet50_linf_eps8.0.ckpt \
resnet50_l2_eps0.1.ckpt \
resnet50_l2_eps0.5.ckpt \
resnet50_l2_eps1.ckpt \
resnet50_l2_eps3.ckpt \
resnet50_l2_eps5.ckpt \
convnext_tiny.fb_in1k \
convnext_tiny.fb_in22k_ft_in1k \
convnext_small.fb_in1k \
convnext_s_cvst_robust.pt \
convnext_small.fb_in22k_ft_in1k \
convnext_base.fb_in1k \
convnext_b_cvst_robust.pt \
convnext_base.fb_in22k_ft_in1k \
convnext_large.fb_in1k \
convnext_large.fb_in22k_ft_in1k \
convnext_xlarge.fb_in22k_ft_in1k \
convnextv2_atto.fcmae_ft_in1k \
convnextv2_femto.fcmae_ft_in1k \
convnextv2_pico.fcmae_ft_in1k \
convnextv2_nano.fcmae_ft_in1k \
convnextv2_nano.fcmae_ft_in22k_in1k \
convnextv2_tiny.fcmae_ft_in1k \
convnextv2_tiny.fcmae_ft_in22k_in1k \
convnextv2_base.fcmae_ft_in1k \
convnextv2_base.fcmae_ft_in22k_in1k \
convnextv2_large.fcmae_ft_in1k \
convnextv2_large.fcmae_ft_in22k_in1k \
convnextv2_huge.fcmae_ft_in1k \
vit_tiny_patch16_224.augreg_in21k_ft_in1k \
vit_small_patch16_224.augreg_in1k \
vit_small_patch16_224.augreg_in21k_ft_in1k \
vit_s_cvst_robust.pt \
vit_small_patch32_224.augreg_in21k_ft_in1k \
vit_base_patch8_224.augreg_in21k_ft_in1k \
vit_base_patch16_224.augreg_in1k \
vit_base_patch16_224.augreg_in21k_ft_in1k \
vit_b_cvst_robust.pt \
vit_base_patch32_224.augreg_in1k \
vit_base_patch32_224.augreg_in21k_ft_in1k \
vit_large_patch16_224.augreg_in21k_ft_in1k \
swin_tiny_patch4_window7_224 \
swin_small_patch4_window7_224 \
advtrain_swin_small_patch4_window7_224_ep4.pth \
swin_base_patch4_window7_224 \
advtrain_swin_base_patch4_window7_224_ep4.pth \
swin_large_patch4_window7_224 \
REV_VIT_S.pyth \
REV_VIT_B.pyth \
REV_MVIT_B.pyth \
ViT-L-14 \
dinov2_vitb14 \
dinov2_vitl14 \
mae_vit_base_patch16 \
mae_vit_large_patch16".split()
imagenet_files_original = imagenet_files[:-5]

cifar10_files_original = "cifar_nat.pt cifar_l2_0_25.pt cifar_l2_0_5.pt cifar_l2_1_0.pt cifar_linf_8.pt \
wrn.pt uniform_wrn.pt augmix_wrn.pt cutmix_wrn.pt mixup_wrn.pt pixmix.pt linf_wrn.pt \
cifar10_linf_wrn28-10.pt \
cifar10_linf_wrn70-16.pt \
cifar10_l2_wrn28-10.pt \
cifar10_l2_wrn70-16.pt".split()

cifar10_files = "cifar_nat.pt \
cifar_l2_0_25.pt cifar_l2_0_5.pt cifar_l2_1_0.pt cifar_linf_8.pt \
robust_overfitting_wrn2810_standard.pth \
robust_overfitting_wrn2810_l2_eps0.25.pth \
wrn2810-cifar10-l2-0.25-trades.pt \
robust_overfitting_wrn2810_l2_eps0.5.pth \
wrn2810-cifar10-l2-0.5-trades.pt \
robust_overfitting_wrn2810_l2_eps1.0.pth \
wrn2810-cifar10-l2-1.0-trades.pt \
robust_overfitting_wrn2810_linf_eps4.pth \
wrn2810-cifar10-linf-4-255-trades.pt \
robust_overfitting_wrn2810_linf_eps8.pth \
wrn2810-cifar10-linf-8-255-trades.pt \
cifar10_wide10_linf_eps8.pth \
trades_model_cifar_wrn.pt \
wrn.pt uniform_wrn.pt augmix_wrn.pt cutmix_wrn.pt mixup_wrn.pt pixmix.pt linf_wrn.pt \
cifar10_linf_wrn28-10.pt \
cifar10_linf_wrn70-16.pt \
cifar10_l2_wrn28-10.pt \
cifar10_l2_wrn70-16.pt \
mng_ac_rst.pth mng_ac.pth msd.pth max.pth avg.pth \
pat_paper_cifar-stadv_0.05.pt \
pat_paper_cifar-recoloradv_0.06.pt \
pat_paper_cifar-pgd_linf_8_pgd_l2_1_stadv_0.05_recoloradv_0.06_random.pt \
pat_paper_cifar-pgd_linf_8_pgd_l2_1_stadv_0.05_recoloradv_0.06_average.pt \
pat_paper_cifar-pgd_linf_8_pgd_l2_1_stadv_0.05_recoloradv_0.06_max.pt \
pat_paper_cifar-pat_self_0.5.pt \
pat_paper_cifar-pat_alexnet_1.pt \
variation_regularization-cifar10_LPIPS_eps0.5_var_0.pth \
variation_regularization-cifar10_LPIPS_eps0.5_var_0.05.pth \
variation_regularization-cifar10_LPIPS_eps0.5_var_0.1.pth \
variation_regularization-cifar10_LPIPS_eps1_var_0.05.pth \
variation_regularization-cifar10_LPIPS_eps1_var_0.1.pth \
variation_regularization-recolor_var_0.pth \
variation_regularization-recolor_var_0.5.pth \
variation_regularization-recolor_var_1.pth \
variation_regularization-stadv_var_0.pth \
variation_regularization-stadv_var_0.5.pth \
variation_regularization-stadv_var_1.pth \
variation_regularization-cifar10_resnet18_l2_var_1.pth \
variation_regularization-cifar10_resnet18_linf_var_0.5.pth \
variation_regularization-cifar10_wrn_28_10_linf_var_0.7.pth \
robust_finetuning_pretr_L1.pth \
robust_finetuning_pretr_L2.pth \
robust_finetuning_pretr_Linf.pth".split()

imagenet100_files = "clean.pt \
pgd-l2-150.pth \
pgd-l2-300.pth \
pgd-l2-600.pth \
pgd-l2-1200.pth \
pgd-l2-2400.pth \
pgd-l2-4800.pth \
pgd-linf-1.pth \
pgd-linf-2.pth \
pgd-linf-4.pth \
pgd-linf-8.pth \
pgd-linf-16.pth \
pgd-linf-32.pth \
pat_self_0.25.pt \
pat_self_0.5_vr0.1.pth \
pat_self_0.5_vr0.3.pth \
pat_self_0.5_vr0.5.pth \
pat_self_0.5_vr1.0.pth \
pat_alexnet_0.5.pt \
pat_alexnet_0.5_vr0.1.pth \
pat_alexnet_0.5_vr0.3.pth \
pat_alexnet_0.5_vr0.5.pth \
pat_alexnet_0.5_vr1.0.pth \
imagenet100_dinov2_vitb14 \
imagenet100_dinov2_vitl14".split()

imagenet_diverse_files = "standard.pt resnet50_l2_eps5.ckpt \
swin_large_patch4_window7_224 convnextv2_large.fcmae_ft_in22k_in1k \
advtrain_swin_base_patch4_window7_224_ep4.pth \
convnext_b_cvst_robust.pt".split()


def read_dfs(*files):
    dfs = {}
    for file in files:
        try:
            df = pd.read_json(file, lines=True)
            pivot_df = pd.pivot_table(
                df, index="step_size", columns="num_steps", values="accuracy"
            )
            pivot_df = pivot_df.multiply(100).round(2)
            pivot_df.index = np.round(pivot_df.index.values, decimals=8)
        except:
            print(f"cannot read file {file}")
            continue
        if pivot_df.isna().any().any():
            print(f"NAN in file {file}")
            nan_indexes = pivot_df.index[pivot_df.isna().any(axis=1)]
            nan_columns = pivot_df.columns[pivot_df.isna().any()]
            nan_tuples = [
                (idx, col)
                for idx in nan_indexes
                for col in nan_columns
                if pd.isna(pivot_df.loc[idx, col])
            ]
            print(nan_tuples)

        matched_names = [
            k
            for k, v in model_maps.items()
            if os.path.basename(file).startswith(k + "-")
        ]
        if len(matched_names) != 1:
            raise ValueError(
                "Multiple matched files", matched_names, os.path.basename(file)
            )
        dfs[model_maps[matched_names[0]]] = pivot_df

    return dfs


def read_accs(*files, name=None):
    accs = {}
    for file in files:
        try:
            df = pd.read_json(file, lines=True)
            assert df.shape[0] == 1  # Should have only 1 row
            acc = df.loc[0, "accuracy"]
        except:
            print(f"cannot read file {file}")
            continue
        matched_names = [
            k
            for k, v in model_maps.items()
            if os.path.basename(file).startswith(k + "-")
        ]
        if len(matched_names) != 1:
            raise ValueError(
                "Multiple matched files", matched_names, os.path.basename(file)
            )
        accs[model_maps[matched_names[0]]] = acc

    return pd.Series(accs, name=name).multiply(100).round(2)


def select_stepsize_steps(dfs, thresh=0.9):
    mean_df = pd.concat(dfs.values(), axis=0).groupby(level=0).mean()
    min_val = mean_df.min().min()
    step_size = mean_df.stack().idxmin()[0]
    row_to_cutoff = mean_df.loc[step_size]
    for num_steps in row_to_cutoff.index:
        if min_val / row_to_cutoff[num_steps] > thresh:
            return step_size, num_steps
    return step_size, row_to_cutoff.index[-1]


def select_steps(dfs, step_size, thresh=0.9):
    mean_df = pd.concat(dfs.values(), axis=0).groupby(level=0).mean()
    min_val = mean_df.min().min()
    row_to_cutoff = mean_df.loc[step_size]
    for i in row_to_cutoff.index:
        if min_val / row_to_cutoff[i] > thresh:
            return i
    return row_to_cutoff.index[-1]


def index_dfs(dfs, minimal=True, row=None, col=None):
    # If minimal is True, find the minimum value for each DataFrame in the dfs dictionary
    if minimal:
        data = {f: df.min().min() for f, df in dfs.items()}
    # Otherwise, use the specified row and column indices to get the corresponding value in each DataFrame
    else:
        data = {f: df.loc[row, col] for f, df in dfs.items()}
    # Create and return a pandas Series with the extracted data
    return pd.Series(data)


def generate_bar_chart(key_df, file_name, color="C0", figsize=None):
    key_df_show = key_df.iloc[::-1]

    num_columns = len(key_df.columns)

    if figsize is None:
        figsize = (12, 8)
    fig, axes = plt.subplots(
        nrows=1, ncols=num_columns, figsize=figsize, sharey=True, dpi=150
    )

    def add_value_labels(ax, spacing=0.5):
        for rect in ax.patches:
            y_value = rect.get_y() + rect.get_height() / 2
            x_value = rect.get_width()
            label = f"{x_value:.1f}"
            ax.annotate(
                label,  # Use `label` as label
                (x_value + spacing, y_value),  # Place label at the end of the bar
                xytext=(0, 0),  # No offset
                textcoords="offset points",  # Interpret `xytext` as offset in points
                va="center",  # Vertically center label
                ha="left",
            )  # Horizontally align label to the left

    # Iterate through each column and create a horizontal bar chart on its own subplot
    for i, (col_name, col_data) in enumerate(key_df_show.items()):
        col_data.plot.barh(ax=axes[i], legend=False, color=color)
        axes[i].set_title(col_name)
        axes[i].set_xlim([0, 50])
        axes[i].set_xticks([0, 10, 20, 30, 40, 50])
        axes[i].spines["right"].set_visible(False)
        add_value_labels(axes[i])

    # Set the y-axis label for the entire figure
    axes[0].set_yticklabels(key_df_show.index, ha="left", x=-2.5)

    # Adjust the layout to ensure there is no overlap between the subplots
    plt.tight_layout()

    # Display the charts
    plt.savefig(file_name, bbox_inches="tight", pad_inches=0)
    plt.show()


def generate_bar_chart_overlay(
    key_df,
    overlay_df=None,
    file_name=None,
    figsize=None,
    xlim=None,
    xlim0=None,
    yticks_padding=100,
    plot_height=0.98,
):
    key_df = key_df.iloc[::-1]

    if overlay_df is not None:
        overlay_df = overlay_df.iloc[::-1]
        assert (key_df.columns == overlay_df.columns).all()
        assert (key_df.index == overlay_df.index).all()

    num_columns = len(key_df.columns)

    if figsize is None:
        figsize = (12, 8)
    fig, axes = plt.subplots(
        nrows=1, ncols=num_columns, figsize=figsize, sharey=True, dpi=150
    )

    def add_value_labels(ax, spacing=0.5):
        for rect in ax.patches:
            y_value = rect.get_y() + rect.get_height() / 2
            x_value = rect.get_width()
            label = f"{x_value:.1f}"
            ax.annotate(
                label,  # Use `label` as label
                (x_value + spacing, y_value),  # Place label at the end of the bar
                xytext=(0, 0),  # No offset
                textcoords="offset points",  # Interpret `xytext` as offset in points
                va="center",  # Vertically center label
                ha="left",
            )  # Horizontally align label to the left

    # Iterate through each column and create a horizontal bar chart on its own subplot
    if xlim is None:
        xlim = max(50, key_df.max().max().round(-1) + 5)

    for i, (col_name, col_data) in enumerate(key_df.items()):
        col_data.plot.barh(ax=axes[i], legend=False, color="C0")
        axes[i].set_title(col_name)  # Model name
        axes[i].set_xlim([0, xlim])
        axes[i].spines["right"].set_visible(False)
        # Add performance numbers
        add_value_labels(axes[i])
        # Use a different color for the clean accuracy
        if i == 0:
            if xlim0 is not None:
                axes[i].set_xlim([0, xlim0])
            color = "C0" if overlay_df is None else "C2"
            col_data.plot.barh(ax=axes[i], legend=False, color=color)

    if overlay_df is not None:
        for i, (col_name_overlay, col_data_overlay) in enumerate(overlay_df.items()):
            # Skip clean accuracy which is already plotted
            if i == 0:
                continue
            col_data_overlay.plot.barh(ax=axes[i], legend=False, color="C1", alpha=1.0)

    # Set the y-axis label for the entire figure
    axes[0].set_yticklabels(key_df.index, ha="left")
    axes[0].tick_params(axis="y", which="major", pad=yticks_padding)

    if overlay_df is not None:
        handles, labels = axes[1].get_legend_handles_labels()
        fig.legend(
            handles,
            ["Fixed step-size and steps", "Optimal step-size and steps"],
            loc="upper center",
            ncol=2,
        )
        # Adjust the layout to ensure there is no overlap between the subplots
        fig.tight_layout(rect=[0, 0, 1, plot_height])

    # Display the charts
    if file_name:
        plt.savefig(file_name, bbox_inches="tight", pad_inches=0)
    plt.show()


def report_gridsearch(
    attacks, task, logdir, epsilon, attack_stepsize=None, attack_steps=None
):
    rstepsize, rsteps = {}, {}
    task_files = {
        "imagenet": imagenet_files_original,
        "imagenet100": imagenet100_files,
        "imagenet_diverse": imagenet_diverse_files,
        "cifar10": cifar10_files_original,
    }[task]
    key_dfs_fixed, key_dfs_optimal = [], []
    for i, attack in enumerate(attacks):
        logfiles = [
            os.path.join(logdir, f"{f}-{attack}-{epsilon}.jsonl") for f in task_files
        ]
        dfs = read_dfs(*logfiles)
        key_dfs_optimal.append(index_dfs(dfs, minimal=True))

        if attack_stepsize is None and attack_steps is None:
            rstepsize[attack], rsteps[attack] = select_stepsize_steps(dfs, thresh=0.9)
        elif attack_steps is None:
            rstepsize[attack] = attack_stepsize[attack]
            rsteps[attack] = select_steps(dfs, rstepsize[attack], thresh=0.9)
        else:
            rstepsize[attack] = attack_stepsize[attack]
            rsteps[attack] = attack_steps[attack]
        try:
            key_dfs_fixed.append(
                index_dfs(dfs, minimal=False, row=rstepsize[attack], col=rsteps[attack])
            )
        except:
            print(attack)
            raise

    key_dfs_fixed = pd.concat(key_dfs_fixed, axis=1, keys=attacks)
    key_dfs_optimal = pd.concat(key_dfs_optimal, axis=1, keys=attacks)
    key_dfs_fixed.insert(0, "UA2", key_dfs_fixed.mean(axis=1).round(1))
    key_dfs_optimal.insert(0, "UA2", key_dfs_optimal.mean(axis=1).round(1))

    logfiles = [os.path.join(logdir, f"{f}-none.jsonl") for f in task_files]
    clean_accs = read_accs(*logfiles, name=None)

    key_dfs_fixed.insert(0, "Clean acc", clean_accs)
    key_dfs_optimal.insert(0, "Clean acc", clean_accs)

    return key_dfs_fixed, key_dfs_optimal, rstepsize, rsteps
