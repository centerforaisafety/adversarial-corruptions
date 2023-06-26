source ./models.sh

attacks="gabor snow pixel jpeg kaleidoscope glitch wood elastic"
extra_attacks="edge fbm fog hsv klotski mix polkadot prison blur texture whirlpool"

common="--eval_batch_size 5 --num_batches -1 --device cuda --seed 123 "


# This function generates a list of parameter values for a given attack type.
#
# Usage: generate_attack_params_list attack output_var
#
# Parameters:
#   attack: The type of attack to generate parameters for.
#   output_var: The name of the variable to store the parameter list in.
#
# Output format:
#   A list of parameter values, where each value is a string of the form "--step_size x --num_steps y".
#   The step size and number of steps values are determined based on the attack type and the specified scales.
function generate_attack_params_list {
    local attack="$1"
    local dataset="$2"
    local output_var="$3"

    # get the step_size value e.g., "--step_size=0.0025 --num_steps=80" -> "0.0025"
    base_stepsize=$(get_attack_params $attack $dataset | cut -d' ' -f1 | cut -d'=' -f2)

    params=()
    local scales=(0.125 0.25 0.5 1 2 4 8)
    for steps in $(seq 10 10 100); do
        for scale in "${scales[@]}"; do
            step_size=$(echo "scale=8; $scale * $base_stepsize" | bc)
            params+=("--step_size $step_size --num_steps $steps")
        done
    done
    eval "$output_var=(\"\${params[@]}\")"
}


task="$1"
logdir=experiments/results/gridsearch-$task
mkdir -p $logdir

# Grid search for step-size and number of steps on ImageNet
function imagenet {
    for epsilon in medium high low; do
        for attack in pgd $attacks $extra_attacks; do
            generate_attack_params_list "$attack" imagenet "attack_params_list"
            for attack_params in "${attack_params_list[@]}"; do
                for model in $imagenet_models; do
                        python main.py $common \
                        --log "$logdir"/"$model"-"$attack"-"$epsilon".jsonl \
                        --epsilon $epsilon --weights weights/imagenet/$model --architecture $(get_arch $model) \
                        --dataset imagenet --attack_config attacks/attack_params_imagenet_default.conf --attack "$attack" $attack_params \
                        --eval_batch_size 50 --num_batches 20&
                done
            done
        done
    done
}

function cifar10 {
    for epsilon in medium high low; do
        for attack in pgd $attacks $extra_attacks; do
            generate_attack_params_list "$attack" cifar10 "attack_params_list"
            for attack_params in "${attack_params_list[@]}"; do
                for model in $cifar_models; do
                        python main.py $common \
                        --log "$logdir"/"$model"-"$attack"-"$epsilon".jsonl \
                        --epsilon $epsilon --weights weights/cifar10/$model --architecture $(get_arch $model) \
                        --dataset cifar10 --attack_config attacks/attack_params_cifar10_default.conf \
                        --attack "$attack" $attack_params \
                        --eval_batch_size 50 --num_batches 20&
                done
            done
        done
    done
}

case $task in
    imagenet) imagenet ;;
    cifar10) cifar10 ;;
    *)
        echo "Invalid argument. Please provide a valid argument: imagenet or cifar10."
        exit 1
        ;;
esac


