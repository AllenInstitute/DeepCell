exp_id=$1
correlation_projection_path=$2
model_weights_path=$3
out_dir=$4
use_cuda=$5
conda_env=/allen/aibs/informatics/aamster/miniconda3/envs/deepcell/bin/python

current_time=$(date "+%Y.%m.%d-%H.%M.%S")

artifact_out_dir="${out_dir}/${exp_id}/artifacts"
predictions_out_dir="${out_dir}/predictions/${current_time}"
log_out_dir="${out_dir}/logs/${current_time}"

mkdir -p "${out_dir}"
mkdir -p "${predictions_out_dir}"
mkdir -p "${log_out_dir}"

rois_path="/allen/aibs/informatics/danielsf/suite2p_210921/v0.10.2/th_3/production/${exp_id}_suite2p_rois.json"

manifest="{
  \"experiment_id\": ${exp_id},
  \"binarized_rois_path\": "\"${rois_path}\""
  \"movie_path\": \"/allen/programs/braintv/workgroups/nc-ophys/danielk/deepinterpolation/experiments/ophys_experiment_${exp_id}/denoised.h5\"
}"

manifest_path="/tmp/${exp_id}_slapp_tform_manifest.json"
echo "${manifest}" > "${manifest_path}"

$conda_env -m slapp.transforms.transform_pipeline \
  --prod_segmentation_run_manifest "${manifest_path}" \
  --output_manifest "${out_dir}/manifest.json" \
  --correlation_projection_path "${correlation_projection_path}/${exp_id}_correlation_proj.png" \
  --artifact_basedir "${artifact_out_dir}" \
  --skip_movies True \
  --skip_traces True \
  --all_ROIs True

$conda_env ./inference.py \
  --experiment_id "${exp_id}" \
  --rois_path "${rois_path}" \
  --data_dir "${artifact_out_dir}" \
  --model_weights_path "${model_weights_path}" \
  --out_path "${predictions_out_dir}" \
  --use_cuda "${use_cuda}"

rm -r "${artifact_out_dir}"
