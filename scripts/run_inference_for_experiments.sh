experiment_ids_path=$1
correlation_projection_path=$2
model_weights_path=$3
out_dir=$4
use_cuda=$5


readarray -t exp_ids < $experiment_ids_path
for exp_id in "${exp_ids[@]}"
  do
    ./run_inference_for_experiment.sh \
      "${exp_id}" \
      "${correlation_projection_path}" \
      "${model_weights_path}" \
      "${out_dir}" \
      "${use_cuda}" &
  done