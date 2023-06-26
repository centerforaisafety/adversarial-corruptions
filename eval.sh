source ./models.sh

attacks="gabor snow pixel jpeg kaleidoscope glitch wood elastic"
extra_attacks="edge fbm fog hsv klotski mix polkadot prison blur texture whirlpool"

common="--eval_batch_size 5 --num_batches -1 --device cuda --seed 123 "

if [ $# -eq 0 ]; then
    echo "Please provide an argument"
    exit 1
fi

task="$1"
logdir=experiments/results/$task
mkdir -p $logdir


# Evaluate the models on ImageNet
function imagenet {
    for epsilon in low medium high; do
        for attack in none pgd $attacks $extra_attacks; do
            for model in $imagenet_models; do
                logfile="$logdir"/"$model"-"$attack"-"$epsilon".jsonl
                if [ ! -e "$logfile" ]; then
                    python main.py $common --log $logfile $(get_attack_params $attack imagenet) \
                    --epsilon $epsilon --weights weights/imagenet/$model --architecture $(get_arch $model) \
                    --dataset imagenet --attack_config attacks/attack_params_imagenet_default.conf --attack "$attack"
                else
                    echo "Skip $logfile"
                fi
            done
        done
    done
}

# Performance variance due to random seed setting
function imagenet_variance {
    for seed in 123 124 125; do
        for epsilon in low medium high; do
            for attack in $attacks; do
                for model in standard.pt resnet50_l2_eps5.ckpt; do
                    logfile="$logdir"/"$model"-"$attack"-"$epsilon"-"seed$seed".jsonl
                    if [ ! -e "$logfile" ]; then
                       python main.py $common --log $logfile $(get_attack_params $attack imagenet) \
                       --epsilon $epsilon --weights weights/imagenet/$model --architecture $(get_arch $model) \
                       --dataset imagenet --attack_config attacks/attack_params_imagenet_default.conf --attack "$attack" \
                       --eval_batch_size 200 --num_batches -1 --seed $seed 
                    else
                        echo "Skip $logfile"
                    fi
                done
            done
        done
    done
}

# Computation time of the attacks on a ResNet50 model
function imagenet_computation_time {
    for epsilon in medium ; do
        for attack in none pgd $attacks $extra_attacks; do
            for model in standard.pt; do
                logfile="$logdir"/"$model"-"$attack"-"$epsilon".jsonl
                if [ ! -e "$logfile" ]; then
                   python main.py $common --log $logfile $(get_attack_params $attack imagenet) \
                   --epsilon $epsilon --weights weights/imagenet/$model --architecture $(get_arch $model) \
                   --dataset imagenet --attack_config attacks/attack_params_imagenet_default.conf --attack "$attack" \
                   --eval_batch_size 200 --num_batches -1 
                else
                    echo "Skip $logfile"
                fi
            done
        done
    done
}


# Effect of optimization pressure, how the attacked accuracy changes as we increase the attack steps with a fixed epsilon
# Measured using 1000 samples
function imagenet_steps {
    for epsilon in medium; do
        for model in $imagenet_models; do
            for attack in $attacks; do
                stepsize=$(get_attack_params $attack imagenet | cut -d' ' -f1 | cut -d'=' -f2)
                for num_steps in $(seq 0 5 100); do
                   python main.py $common \
                   --log "$logdir"/"$model"-"$attack"-"$epsilon".jsonl \
                   --epsilon $epsilon --weights weights/imagenet/$model --architecture $(get_arch $model) \
                   --dataset imagenet --attack_config attacks/attack_params_imagenet_default.conf \
                   --attack "$attack" \
                   --step_size $stepsize --num_steps $num_steps \
                   --eval_batch_size 200 --num_batches 5
                done
            done
        done
    done
}

# Performance of the attacks when the adversarial corruption is random (simulated by setting the optimization steps to 0)
function rand_corruption {
    for epsilon in low medium high; do
        for attack in $attacks; do
            for model in $imagenet_models; do
                logfile="$logdir"/"$model"-"$attack"-"$epsilon".jsonl
                if [ ! -e "$logfile" ]; then
                   python main.py $common --log $logfile $(get_attack_params $attack imagenet) \
                   --epsilon $epsilon --weights weights/imagenet/$model --architecture $(get_arch $model) \
                   --dataset imagenet --attack_config attacks/attack_params_imagenet_default.conf --attack "$attack" \
                   --num_steps 0 --eval_batch_size 50
                else
                    echo "Skip $logfile"
                fi
            done
        done
    done
}

# Save attacked samples
function attacked_samples {
    for epsilon in low medium high; do
        for attack in $attacks $extra_attacks; do
            for model in standard.pt; do
               python main.py $common --log /dev/null $(get_attack_params $attack imagenet) \
               --epsilon $epsilon --weights weights/imagenet/$model --architecture $(get_arch $model) \
               --dataset imagenet --attack_config attacks/attack_params_imagenet_default.conf --attack "$attack" \
               --eval_batch_size 12 --num_batches 1 \
               --save attacked_samples
            done
        done
    done
}

function attacked_samples_cifar10 {
    for epsilon in low medium high; do
        for attack in $attacks $extra_attacks; do
            for model in cifar_nat.pt; do
                python main.py $common --log /dev/null $(get_attack_params $attack cifar10) \
                --epsilon $epsilon --weights weights/cifar10/$model --architecture $(get_arch $model) \
                --dataset cifar10 --attack_config attacks/attack_params_cifar10_default.conf --attack "$attack" \
                --eval_batch_size 100 --num_batches 1 \
                --save attacked_samples
            done
        done
    done
}

# Teaser image
function teaser {
    for epsilon in medium ; do
        for attack in none pgd $attacks $extra_attacks; do
            for model in clean.pt ; do
                python main.py $common \
                --log /dev/null $(get_attack_params $attack imagenet100) \
                --epsilon $epsilon --weights weights/imagenet100/$model --architecture $(get_arch $model) \
                --dataset imagenet100 --attack_config attacks/attack_params_imagenet100_default.conf --attack "$attack" \
                --eval_batch_size 2 --num_batches 1 \
                --save teaser
            done
        done
    done
}


# Fourier analysis
function fourier_analysis {
    for epsilon in medium; do
        for attack in $attacks; do
            for model in standard.pt; do
                python main.py $common \
                --log /dev/null $(get_attack_params $attack imagenet) \
                --epsilon $epsilon --weights weights/imagenet/$model --architecture $(get_arch $model) \
                --dataset imagenet --attack_config attacks/attack_params_imagenet_default.conf --attack "$attack" \
                --eval_batch_size 50 --num_batches 20 \
                --save fourier_analysis
            done
        done
    done
}


# Generate data for human study
declare -A seeds

seeds["gabor"]="1"
seeds["snow"]="2"
seeds["pixel"]="3"
seeds["jpeg"]="4"
seeds["kaleidoscope"]="5"
seeds["glitch"]="6"
seeds["wood"]="7"
seeds["elastic"]="8"

# Human study on imagenetr
function humanstudy {
    for epsilon in medium; do
        for attack in $attacks; do
            for model in resnet50.pth; do
                python main.py $common \
                --log /dev/null $(get_attack_params $attack imagenet) \
                --epsilon $epsilon --weights weights/imagenetr/$model --architecture resnet50 \
                --dataset imagenetr --attack_config attacks/attack_params_imagenetr_default.conf \
                --attack "$attack" --num_batches -1 --eval_batch_size 1 \
                --glitch_num_lines 64 \
                --save humanstudy \
                --seed ${seeds[$attack]}
            done
        done
    done
}


function imagenet100 {
    for model in $imagenet100_models; do
        for epsilon in medium low high; do
            for attack in pgd none $attacks; do
                logfile="$logdir"/"$model"-"$attack"-"$epsilon".jsonl
                if [ ! -e "$logfile" ]; then
                    python main.py $common --log $logfile $(get_attack_params $attack imagenet100) \
                    --epsilon $epsilon --weights weights/imagenet100/$model --architecture $(get_arch $model) \
                    --dataset imagenet100 --attack_config attacks/attack_params_imagenet100_default.conf --attack "$attack"
                else
                    echo "Skip $logfile"
                fi
            done
        done
    done
}

function cifar10 {
    for epsilon in low medium high; do
        for attack in pgd none $attacks $extra_attacks; do
            for model in $cifar_models; do
                logfile="$logdir"/"$model"-"$attack"-"$epsilon".jsonl
                if [ ! -e "$logfile" ]; then
                    python main.py $common --log $logfile $(get_attack_params $attack cifar10) \
                    --epsilon $epsilon --weights weights/cifar10/$model --architecture $(get_arch $model) \
                    --dataset cifar10 --attack_config attacks/attack_params_cifar10_default.conf --attack "$attack" \
                    --eval_batch_size 200
                else
                    echo "Skip $logfile"
                fi
            done
        done
    done
}


case $task in
    imagenet) imagenet ;;
    imagenet_steps) imagenet_steps ;;
    imagenet_variance) imagenet_variance ;;
    imagenet_computation_time) imagenet_computation_time ;;
    imagenet_bs5) imagenet_bs5 ;;
    rand_corruption) rand_corruption ;;
    attacked_samples) attacked_samples ;;
    attacked_samples_cifar10) attacked_samples_cifar10 ;;
    teaser) teaser ;;
    humanstudy) humanstudy ;;
    fourier_analysis) fourier_analysis ;;
    imagenet100) imagenet100 ;;
    cifar10) cifar10 ;;
    *)
        echo "Invalid argument. Please provide a valid argument: "\
             "imagenet, rand_corruption, attacked_samples, teaser, "\
             "humanstudy, fourier_analysis, imagenet100, or cifar10."
        rmdir $logdir
        exit 1
        ;;
esac
