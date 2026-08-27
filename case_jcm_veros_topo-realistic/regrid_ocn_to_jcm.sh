#!/bin/bash
# Regrid all ocn-NNNNN.nc batch output files onto the JCM (atm) grid using
# ncremap and a pre-generated ESMF weight file.
#
# Two preprocessing steps are required per file, both discovered the hard
# way (see conversation history):
#   1. Permute dims so lat,lon are the last two (fastest-varying), with any
#      vertical dim before them -- Veros writes (time,lon,lat,depth), but
#      ESMF/SCRIP weight files assume lon varies fastest. Skipping this
#      does NOT error -- it silently produces wrong regridded values.
#   2. Replace NaN _FillValue attributes with a real sentinel -- NCO's
#      regridding arithmetic segfaults/aborts on NaN fill values.
#
# Usage:
#   ./regrid_ocn_to_jcm.sh [weight_file] [input_dir] [output_dir]
#
# Defaults regrid every output_T31/real_topo/ocn-*.nc with the bilinear
# weight file into output_T31/real_topo/ocn_on_jcm/.

set -u
CASE_DIR=/home/tienyiao/projects_local/project_jax-esm/MetESM/case_jcm_veros_topo-realistic_domain-master
cd "$CASE_DIR" || exit 1

WEIGHT_FILE="${1:-grid_res-3deg/weight_algo-bilinear_RotatedGaussianLatLon_to_JCM_T31.nc}"
INPUT_DIR="${2:-output_T31/real_topo}"
OUTPUT_DIR="${3:-output_T31/real_topo/ocn_on_jcm}"

if [ ! -f "$WEIGHT_FILE" ]; then
    echo "ERROR: weight file not found: $WEIGHT_FILE" >&2
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

# Variables present in every ocn-NNNNN.nc batch file (see model_setup.py /
# jem's Veros output schema) -- fixed list rather than parsed per-file,
# since the schema is the same across all 500 batches.
VARS="freshwater_flux heat_flux mask_T mask_surface_T temp salt u v sea_surface_temperature sea_surface_u sea_surface_v sea_surface_salinity surface_air_temperature surface_taux surface_tauy dzt"

TMP_PERMUTED=$(mktemp --suffix=.nc)
trap 'rm -f "$TMP_PERMUTED"' EXIT

total=0
failed=0
skipped=0

shopt -s nullglob
for infile in "$INPUT_DIR"/ocn-*.nc; do
    base=$(basename "$infile")
    outfile="$OUTPUT_DIR/${base%.nc}_on_jcm.nc"
    total=$((total + 1))

    if [ -e "$outfile" ]; then
        skipped=$((skipped + 1))
        continue
    fi

    if ! ncpdq -O -a time,depth,lat,lon "$infile" "$TMP_PERMUTED" >/dev/null 2>&1; then
        echo "FAILED (permute): $base"
        failed=$((failed + 1))
        continue
    fi

    for v in $VARS; do
        ncatted -O -a _FillValue,"$v",m,d,1.0e36 "$TMP_PERMUTED" >/dev/null 2>&1
    done

    if ! ncremap -m "$WEIGHT_FILE" -i "$TMP_PERMUTED" -o "$outfile" >/dev/null 2>&1; then
        echo "FAILED (regrid): $base"
        failed=$((failed + 1))
        rm -f "$outfile"
        continue
    fi

    if [ $((total % 25)) -eq 0 ]; then
        echo "progress: $total files processed ($failed failed, $skipped skipped)"
    fi
done

echo "Done. total=$total failed=$failed skipped=$skipped -> output in $OUTPUT_DIR"
[ "$failed" -eq 0 ]
