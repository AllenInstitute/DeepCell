#!/bin/bash
#SBATCH --job-name=run_inference_for_experiments
#SBATCH --mail-user=adam.amster@alleninstitute.org # Update this email
#SBATCH --mail-type=BEGIN,END,FAIL # Send mail on begin, end, fail
#SBATCH --mem=250gb
#SBATCH --time=48:00:00
#SBATCH --output=/allen/aibs/informatics/aamster/pure_noise/logs/%A-%a.log # Update this log path
#SBATCH --partition braintv
#SBATCH --array=0-1 # Update this with number of experiments
#SBATCH --ntasks=24

experiment_ids_path=$1
rois_path=$2
movie_path=$3
correlation_projection_path=$4
model_weights_path=$5
out_dir=$6
conda_env=$7
center_crop_size=$8
use_correlation_projection=$9
mask_projections=${10}
center_soma=${11}

conda activate "$conda_env"
pip install git+https://github.com/AllenInstitute/segmentation-labeling-app.git

readarray -t exp_ids < "${experiment_ids_path}"

exp_id=${exp_ids[$SLURM_ARRAY_TASK_ID]}

artifact_out_dir="${out_dir}/${exp_id}/artifacts"
predictions_out_dir="${out_dir}/predictions/"

mkdir -p "${out_dir}"
mkdir -p "${predictions_out_dir}"

rois_path="${rois_path}/${exp_id}_rois.json"

manifest="{
  \"experiment_id\": ${exp_id},
  \"binarized_rois_path\": "\"${rois_path}\"",
  \"movie_path\": \"${movie_path}/${exp_id}_denoised_video.h5\",
  \"local_to_global_roi_id_map\": {}
}"

manifest_path="/tmp/${exp_id}_slapp_tform_manifest.json"
echo "${manifest}" > "${manifest_path}"

echo "Generating artifacts for ${exp_id}"

$conda_env -m slapp.transforms.transform_pipeline \
  --prod_segmentation_run_manifest "${manifest_path}" \
  --output_manifest "${out_dir}/manifest.json" \
  --correlation_projection_path "${correlation_projection_path}/${exp_id}_correlation_projection.png" \
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
  --center_crop_size "${center_crop_size}" \
  --use_correlation_projection "${use_correlation_projection}" \
  --mask_projections "${mask_projections}" \
  --center_soma "${center_soma}"

echo "Removing ${artifact_out_dir}"

rm -r "${artifact_out_dir}"
