#!/bin/bash
#SBATCH --job-name=run_inference_for_experiments
#SBATCH --mail-user=adam.amster@alleninstitute.org # Update this email
#SBATCH --mail-type=BEGIN,END,FAIL # Send mail on begin, end, fail
#SBATCH --mem=250gb
#SBATCH --time=48:00:00
#SBATCH --output=/allen/aibs/informatics/aamster/vasculature_artifacts/logs/%A-%a.log # Update this log path
#SBATCH --partition braintv
#SBATCH --array=0-159
#SBATCH --ntasks=24

experiment_ids_path=$1
correlation_projection_path=$2
model_weights_path=$3
out_dir=$4
use_cuda=$5
conda_env=$6


readarray -t exp_ids < "${experiment_ids_path}"

exp_id=${exp_ids[$SLURM_ARRAY_TASK_ID]}

artifact_out_dir="${out_dir}/${exp_id}/artifacts"
predictions_out_dir="${out_dir}/predictions/"

mkdir -p "${out_dir}"
mkdir -p "${predictions_out_dir}"

rois_path="/allen/aibs/informatics/danielsf/suite2p_210921/v0.10.2/th_3/production/${exp_id}_suite2p_rois.json"

manifest="{
  \"experiment_id\": ${exp_id},
  \"binarized_rois_path\": "\"${rois_path}\"",
  \"movie_path\": \"/allen/programs/braintv/workgroups/nc-ophys/danielk/deepinterpolation/experiments/ophys_experiment_${exp_id}/denoised.h5\",
  \"local_to_global_roi_id_map\": {}
}"

manifest_path="/tmp/${exp_id}_slapp_tform_manifest.json"
echo "${manifest}" > "${manifest_path}"

echo "Generating artifacts for ${exp_id}"

$conda_env -m slapp.transforms.transform_pipeline \
  --prod_segmentation_run_manifest "${manifest_path}" \
  --output_manifest "${out_dir}/manifest.json" \
  --correlation_projection_path "${correlation_projection_path}/${exp_id}_correlation_proj.png" \
  --artifact_basedir "${artifact_out_dir}" \
  --skip_movies True \
  --skip_traces True \
  --all_ROIs True

echo "Running inference for ${exp_id}"

$conda_env -m deepcell.modules.run_inference \
  --experiment_id "${exp_id}" \
  --rois_path "${rois_path}" \
  --data_dir "${artifact_out_dir}" \
  --model_weights_path "${model_weights_path}" \
  --out_path "${predictions_out_dir}" \
  --use_cuda "${use_cuda}"

echo "Removing ${artifact_out_dir}"

rm -r "${artifact_out_dir}"
