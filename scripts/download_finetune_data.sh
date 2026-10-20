#!/usr/bin/env bash
# Download the bad-medical-advice fine-tuning set of Turner et al. (2025),
# "Model Organisms for Emergent Misalignment", used for fine-tuning data attribution.
# The upstream archive is encrypted against web scraping; the password is published in
# https://github.com/clarifying-EM/model-organisms-for-EM.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${REPO_ROOT}/data/finetune"
URL="https://github.com/clarifying-EM/model-organisms-for-EM/raw/main/em_organism_dir/data/training_datasets.zip.enc"
EXPECTED_MD5="f2ab0b8f0e3fb89b936779929cd30fd1"

TMP="$(mktemp -d)"
trap 'rm -rf "${TMP}"' EXIT

pip install --quiet easy-dataset-share
curl -sSL -o "${TMP}/training_datasets.zip.enc" "${URL}"
(cd "${TMP}" && easy-dataset-share unprotect-dir training_datasets.zip.enc \
    -p model-organisms-em-datasets --remove-canaries)

mkdir -p "${DEST}"
cp "${TMP}/training_datasets.zip.enc.extracted/bad_medical_advice.jsonl" "${DEST}/"
echo "${EXPECTED_MD5}  ${DEST}/bad_medical_advice.jsonl" | md5sum -c -
