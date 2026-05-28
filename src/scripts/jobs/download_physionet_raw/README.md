# Raw MIMIC Download Setup

This folder downloads credentialed PhysioNet MIMIC resources into scratch and
creates repo-local symlinks.

Versions are chosen to match the current preprocessing notebooks:

- MIMIC-IV `2.2`
- MIMIC-IV-Note `2.2`
- MIMIC-CXR `2.1.0`
- MIMIC-IV-ECG `1.0`

Create PhysioNet credentials first:

```bash
cat > ~/.netrc <<'EOF'
machine physionet.org login YOUR_USERNAME password YOUR_PASSWORD
EOF
chmod 600 ~/.netrc
```

Then run:

```bash
cd /home/pham156/MoE/FuseMoE_poly/src/scripts/download_physionet_raw
bash download_mimic_raw_to_scratch.sh
```

By default, MIMIC-CXR downloads metadata only because the full image tree is very
large. To download CXR images too:

```bash
DOWNLOAD_CXR_IMAGES=1 bash download_mimic_raw_to_scratch.sh
```

Output paths:

- Scratch root: `/scratch/gilbreth/pham156/FuseMoE_data/raw/physionet`
- Repo symlinks: `/home/pham156/MoE/FuseMoE_poly/data/raw`

