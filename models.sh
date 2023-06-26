resnet50_models="standard.pt deepaugment.pth.tar \
ant.pth pixmix.pth.tar deepaugment_and_augmix.pth.tar \
stylised.tar mixup.pth.tar cutmix.pth.tar randaug.pth \
moex.pt augmix.tar"

resnet50robust_models="resnet50_linf_eps0.5.ckpt \
resnet50_linf_eps1.0.ckpt resnet50_linf_eps2.0.ckpt \
resnet50_linf_eps4.0.ckpt resnet50_linf_eps8.0.ckpt \
resnet50_l2_eps0.1.ckpt \
resnet50_l2_eps0.5.ckpt resnet50_l2_eps1.ckpt \
resnet50_l2_eps3.ckpt resnet50_l2_eps5.ckpt \
imagenet_linf_4_madrylab.pt imagenet_linf_8_madrylab.pt"

vit_models="vit_tiny_patch16_224.augreg_in21k_ft_in1k \
vit_small_patch16_224.augreg_in1k vit_small_patch16_224.augreg_in21k_ft_in1k \
vit_small_patch32_224.augreg_in21k_ft_in1k vit_base_patch8_224.augreg_in21k_ft_in1k \
vit_base_patch16_224.augreg_in1k vit_base_patch16_224.augreg_in21k_ft_in1k \
vit_base_patch32_224.augreg_in1k \
vit_base_patch32_224.augreg_in21k_ft_in1k \
vit_large_patch16_224.augreg_in21k_ft_in1k \
swin_tiny_patch4_window7_224 swin_small_patch4_window7_224 \
swin_base_patch4_window7_224 swin_large_patch4_window7_224"

convnext_models="convnext_tiny.fb_in1k convnext_tiny.fb_in22k_ft_in1k \
convnext_small.fb_in1k convnext_small.fb_in22k_ft_in1k \
convnext_base.fb_in1k convnext_base.fb_in22k_ft_in1k \
convnext_large.fb_in1k convnext_large.fb_in22k_ft_in1k \
convnext_xlarge.fb_in22k_ft_in1k"

convnextv2_models="convnextv2_atto.fcmae_ft_in1k convnextv2_femto.fcmae_ft_in1k \
convnextv2_pico.fcmae_ft_in1k convnextv2_nano.fcmae_ft_in1k \
convnextv2_nano.fcmae_ft_in22k_in1k convnextv2_tiny.fcmae_ft_in1k \
convnextv2_tiny.fcmae_ft_in22k_in1k convnextv2_base.fcmae_ft_in1k \
convnextv2_base.fcmae_ft_in22k_in1k convnextv2_large.fcmae_ft_in1k \
convnextv2_large.fcmae_ft_in22k_in1k convnextv2_huge.fcmae_ft_in1k"

reversible_models="REV_VIT_S.pyth REV_VIT_B.pyth REV_MVIT_B.pyth"

robustswin_models="advtrain_swin_small_patch4_window7_224_ep4.pth \
advtrain_swin_base_patch4_window7_224_ep4.pth"

robustvit_models="vit_s_cvst_robust.pt vit_b_cvst_robust.pt"

robustconvnext_models="convnext_tiny_cvst_robust.pt convnext_s_cvst_robust.pt convnext_b_cvst_robust.pt"
# clip_models="RN50 RN101 RN50x4 RN50x16 RN50x64 ViT-B-32 ViT-B-16 ViT-L-14 ViT-L-14@336px"
clip_models="ViT-L-14"
dino_models="dinov2_vitb14 dinov2_vitl14"
mae_models="mae_vit_base_patch16 mae_vit_large_patch16"

imagenet_models="$resnet50_models $resnet50robust_models $vit_models $convnext_models \
       $convnextv2_models $reversible_models $robustvit_models \
       $robustconvnext_models $robustswin_models $clip_models $dino_models $mae_models"

imagenet_models_original="$resnet50_models $resnet50robust_models $vit_models $convnext_models \
       $convnextv2_models $reversible_models $robustvit_models \
       $robustconvnext_models $robustswin_models"

imagenet_diverse="standard.pt resnet50_l2_eps5.ckpt \
swin_large_patch4_window7_224 convnextv2_large.fcmae_ft_in22k_in1k \
advtrain_swin_base_patch4_window7_224_ep4.pth \
convnext_b_cvst_robust.pt"

imagenet100_dinov2_models="imagenet100_dinov2_vitl14 imagenet100_dinov2_vitb14"
imagenet100_resnet50_models="clean.pt pgd-l2-150.pth pgd-l2-300.pth \
pgd-l2-600.pth  pgd-l2-1200.pth   pgd-l2-2400.pth \
pgd-l2-4800.pth pgd-linf-1.pth    pgd-linf-2.pth \
pgd-linf-4.pth  pgd-linf-8.pth    pgd-linf-16.pth \
pgd-linf-32.pth pat_self_0.25.pt  pat_alexnet_0.5.pt \
pat_self_0.5_vr1.0.pth \
pat_self_0.5_vr0.5.pth \
pat_self_0.5_vr0.0.pth \
pat_self_0.5_vr0.3.pth \
pat_self_0.5_vr0.1.pth \
pat_alexnet_0.5_vr0.1.pth \
pat_alexnet_0.5_vr0.3.pth \
pat_alexnet_0.5_vr0.5.pth \
pat_alexnet_0.5_vr1.0.pth"

imagenet100_models="$imagenet100_resnet50_models $imagenet100_dinov2_models"


cifar_resnet50="cifar_nat.pt cifar_l2_0_25.pt cifar_l2_0_5.pt cifar_l2_1_0.pt cifar_linf_8.pt"
cifar_wrn="wrn.pt uniform_wrn.pt augmix_wrn.pt cutmix_wrn.pt mixup_wrn.pt pixmix.pt linf_wrn.pt"
cifar_wideresnetwithsilu2810="cifar10_linf_wrn28-10.pt cifar10_l2_wrn28-10.pt"
cifar_wideresnetwithsilu7016="cifar10_linf_wrn70-16.pt cifar10_l2_wrn70-16.pt"
cifar_multiattack="mng_ac_rst.pth mng_ac.pth msd.pth max.pth avg.pth" # https://github.com/divyam3897/MNG_AC
cifar_pat="pat_paper_cifar-stadv_0.05.pt \
pat_paper_cifar-recoloradv_0.06.pt \
pat_paper_cifar-pgd_linf_8_pgd_l2_1_stadv_0.05_recoloradv_0.06_random.pt \
pat_paper_cifar-pgd_linf_8_pgd_l2_1_stadv_0.05_recoloradv_0.06_average.pt \
pat_paper_cifar-pgd_linf_8_pgd_l2_1_stadv_0.05_recoloradv_0.06_max.pt \
pat_paper_cifar-pat_self_0.5.pt \
pat_paper_cifar-pat_alexnet_1.pt"
cifar_var_pat="variation_regularization-cifar10_LPIPS_eps0.5_var_0.05.pth \
variation_regularization-cifar10_LPIPS_eps0.5_var_0.1.pth \
variation_regularization-cifar10_LPIPS_eps0.5_var_0.pth \
variation_regularization-cifar10_LPIPS_eps1_var_0.05.pth \
variation_regularization-cifar10_LPIPS_eps1_var_0.1.pth"
cifar_var_adv="variation_regularization-recolor_var_0.5.pth \
variation_regularization-recolor_var_0.pth \
variation_regularization-recolor_var_1.pth \
variation_regularization-stadv_var_0.5.pth \
variation_regularization-stadv_var_0.pth \
variation_regularization-stadv_var_1.pth"
cifar_var_adv_norm="variation_regularization-cifar10_resnet18_l2_var_1.pth \
variation_regularization-cifar10_resnet18_linf_var_0.5.pth"
cifar_var_adv_norm_wrn2810="variation_regularization-cifar10_wrn_28_10_linf_var_0.7.pth"
cifar_wrn3410="cifar10_wide10_linf_eps8.pth"
cifar_wrn2810="robust_overfitting_wrn2810_l2_eps0.25.pth \
robust_overfitting_wrn2810_l2_eps0.5.pth \
robust_overfitting_wrn2810_l2_eps1.0.pth \
robust_overfitting_wrn2810_linf_eps4.pth \
robust_overfitting_wrn2810_linf_eps8.pth \
robust_overfitting_wrn2810_standard.pth"

trades_cifar_wrn3410="trades_model_cifar_wrn.pt \
wrn2810-cifar10-l2-0.25-trades.pt \
wrn2810-cifar10-l2-0.5-trades.pt \
wrn2810-cifar10-linf-4-255-trades.pt \
wrn2810-cifar10-linf-8-255-trades.pt \
wrn2810-cifar10-l2-1.0-trades.pt"

cifar_robust_finetuning="robust_finetuning_pretr_L1.pth \
robust_finetuning_pretr_L2.pth \
robust_finetuning_pretr_Linf.pth"

cifar_models_original="$cifar_resnet50 $cifar_wrn \
$cifar_wideresnetwithsilu2810 \
$cifar_wideresnetwithsilu7016"

cifar_models="$cifar_resnet50 $cifar_wrn \
$cifar_wideresnetwithsilu2810 \
$cifar_wideresnetwithsilu7016 \
$cifar_multiattack $cifar_pat \
$cifar_var_pat $cifar_var_adv $cifar_robust_finetuning $cifar_var_adv_norm $cifar_var_adv_norm_wrn2810 $cifar_wrn3410 $cifar_wrn2810 $trades_cifar_wrn3410"


function get_arch() {
    local input=$1

    for model in $resnet50_models; do [[ "$model" == "$input" ]] && echo "resnet50" && return; done
    for model in $resnet50robust_models; do [[ "$model" == "$input" ]] && echo "resnet50" && return; done
    for model in $vit_models; do [[ "$model" == "$input" ]] && echo "vit" && return; done
    for model in $convnext_models; do [[ "$model" == "$input" ]] && echo "convnextv1" && return; done
    for model in $convnextv2_models; do [[ "$model" == "$input" ]] && echo "convnextv2" && return; done
    for model in $reversible_models; do [[ "$model" == "$input" ]] && echo "reversiblevit" && return; done
    for model in $robustvit_models; do [[ "$model" == "$input" ]] && echo "robustvit" && return; done
    for model in $robustconvnext_models; do [[ "$model" == "$input" ]] && echo "robustconvnext" && return; done
    for model in $robustswin_models; do [[ "$model" == "$input" ]] && echo "robustswin" && return; done
    for model in $clip_models; do [[ "$model" == "$input" ]] && echo "clip" && return; done
    for model in $dino_models; do [[ "$model" == "$input" ]] && echo "dinov2" && return; done
    for model in $mae_models; do [[ "$model" == "$input" ]] && echo "mae" && return; done

    for model in $imagenet100_resnet50_models; do [[ "$model" == "$input" ]] && echo "resnet50" && return; done
    for model in $imagenet100_dinov2_models; do [[ "$model" == "$input" ]] && echo "dinov2" && return; done

    for model in $cifar_resnet50; do [[ "$model" == "$input" ]] && echo "resnet50" && return; done
    for model in $cifar_wrn; do [[ "$model" == "$input" ]] && echo "wrn" && return; done
    for model in $cifar_wideresnetwithsilu2810; do [[ "$model" == "$input" ]] && echo "wideresnetwithsilu2810" && return; done
    for model in $cifar_wideresnetwithsilu7016; do [[ "$model" == "$input" ]] && echo "wideresnetwithsilu7016" && return; done
    for model in $cifar_multiattack; do [[ "$model" == "$input" ]] && echo "wrn2810_ma" && return; done
    for model in $cifar_pat; do [[ "$model" == "$input" ]] && echo "resnet50" && return; done
    for model in $cifar_var_pat; do [[ "$model" == "$input" ]] && echo "resnet50" && return; done
    for model in $cifar_var_adv; do [[ "$model" == "$input" ]] && echo "resnet18" && return; done
    for model in $cifar_var_adv_norm; do [[ "$model" == "$input" ]] && echo "resnet18" && return; done
    for model in $cifar_var_adv_norm_wrn2810; do [[ "$model" == "$input" ]] && echo "wideresnet2810_vr" && return; done
    for model in $cifar_robust_finetuning; do [[ "$model" == "$input" ]] && echo "preactresnet18" && return; done
    for model in $cifar_wrn3410; do [[ "$model" == "$input" ]] && echo "wrn3410" && return; done
    for model in $cifar_wrn2810; do [[ "$model" == "$input" ]] && echo "wrn2810" && return; done
    for model in $trades_cifar_wrn3410; do [[ "$model" == "$input" ]] && echo "wideresnet_trades" && return; done

    echo "Model not found"
}


function get_attack_params {
    attack="$1"
    dataset="$2"

    case "$attack-$dataset" in

        none-imagenet         | none-imagenet100        ) echo "--step_size=0      --num_steps=0  " ;;
        # gabor-imagenet        | gabor-imagenet100       ) echo "--step_size=0.0025 --num_steps=100" ;;
        gabor-imagenet        | gabor-imagenet100       ) echo "--step_size=0.0003125 --num_steps=100" ;;
        snow-imagenet         | snow-imagenet100        ) echo "--step_size=0.1    --num_steps=100" ;;
        pixel-imagenet        | pixel-imagenet100       ) echo "--step_size=1.0    --num_steps=100" ;;
        jpeg-imagenet         | jpeg-imagenet100        ) echo "--step_size=0.0024 --num_steps=80" ;;
        elastic-imagenet      | elastic-imagenet100     ) echo "--step_size=0.003  --num_steps=100" ;;
        wood-imagenet         | wood-imagenet100        ) echo "--step_size=0.005  --num_steps=80" ;;
        glitch-imagenet       | glitch-imagenet100      ) echo "--step_size=0.005  --num_steps=90" ;;
        kaleidoscope-imagenet | kaleidoscope-imagenet100) echo "--step_size=0.005  --num_steps=90" ;;

        pgd-imagenet          | pgd-imagenet100         ) echo "--step_size=0.004  --num_steps=50" ;;

        none-cifar10          ) echo "--step_size=0     --num_steps=0  " ;;
        gabor-cifar10         ) echo "--step_size=0.0025 --num_steps=80" ;;
        snow-cifar10          ) echo "--step_size=0.2   --num_steps=20" ;;
        pixel-cifar10         ) echo "--step_size=1.0   --num_steps=60" ;;
        jpeg-cifar10          ) echo "--step_size=0.0024 --num_steps=50" ;;
        elastic-cifar10       ) echo "--step_size=0.006 --num_steps=30" ;;
        wood-cifar10          ) echo "--step_size=0.000625 --num_steps=70" ;;
        glitch-cifar10        ) echo "--step_size=0.0025 --num_steps=60" ;;
        kaleidoscope-cifar10  ) echo "--step_size=0.005 --num_steps=30" ;;

        pgd-cifar10           ) echo "--step_size=0.008  --num_steps=50" ;;

        edge-imagenet     | edge-imagenet100     ) echo "--step_size=0.02  --num_steps=60" ;;
        fbm-imagenet      | fbm-imagenet100      ) echo "--step_size=0.006 --num_steps=30" ;;
        klotski-imagenet  | klotski-imagenet100  ) echo "--step_size=0.01  --num_steps=40" ;;
        texture-imagenet  | texture-imagenet100  ) echo "--step_size=0.003 --num_steps=80" ;;
        mix-imagenet      | mix-imagenet100      ) echo "--step_size=1.0   --num_steps=70" ;;
        polkadot-imagenet  | polkadot-imagenet100  ) echo "--step_size=0.3   --num_steps=40" ;;
        prison-imagenet   | prison-imagenet100   ) echo "--step_size=0.0015 --num_steps=30" ;;
        blur-imagenet     | blur-imagenet100     ) echo "--step_size=0.03  --num_steps=40" ;;
        whirlpool-imagenet| whirlpool-imagenet100) echo "--step_size=4.0   --num_steps=40" ;;
        fog-imagenet      | fog-imagenet100      ) echo "--step_size=0.05  --num_steps=40" ;;
        hsv-imagenet      | hsv-imagenet100      ) echo "--step_size=0.006 --num_steps=20" ;;

        edge-cifar10      ) echo "--step_size=0.0200 --num_steps=60" ;;
        fbm-cifar10       ) echo "--step_size=0.0060 --num_steps=30" ;;
        fog-cifar10       ) echo "--step_size=0.0500 --num_steps=40" ;;
        hsv-cifar10       ) echo "--step_size=0.0030 --num_steps=20" ;;
        klotski-cifar10   ) echo "--step_size=0.0050 --num_steps=50" ;;
        mix-cifar10       ) echo "--step_size=0.5000 --num_steps=30" ;; # L2
        polkadot-cifar10   ) echo "--step_size=0.3000 --num_steps=40" ;; # L2
        prison-cifar10    ) echo "--step_size=0.0015 --num_steps=20" ;;
        blur-cifar10      ) echo "--step_size=0.0150 --num_steps=20" ;;
        texture-cifar10   ) echo "--step_size=0.0030 --num_steps=30" ;;
        whirlpool-cifar10 ) echo "--step_size=16.0000 --num_steps=50" ;; # L2

        *) echo "Unknown attack and dataset combination: $attack-$dataset" ;;
    esac
}

