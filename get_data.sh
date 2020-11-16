aws s3 cp s3://prod.slapp.alleninstitute.org/behavior_slc_oct_2020/20201020135214/ data --recursive --exclude "*" --include "mask_*"
aws s3 cp s3://prod.slapp.alleninstitute.org/behavior_slc_oct_2020/20201020135214/ data --recursive --exclude "*" --include "max_*"
aws s3 cp s3://prod.slapp.alleninstitute.org/behavior_slc_oct_2020/20201020135214/ data --recursive --exclude "*" --include "avg_*"