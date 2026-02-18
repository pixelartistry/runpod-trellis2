#!/bin/bash
# =============================================================================
# ComfyUI + Trellis2 RunPod Auto-Setup Script
# Image:  runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404
# Target: RTX 5090 | Python 3.12 (system) | PyTorch 2.9.1 (pre-installed) | CUDA 12.8.1
# =============================================================================

set -uo pipefail   # -u: unset vars are errors  -o pipefail: pipe failures propagate
                   # NOTE: intentionally no -e so we can handle errors per-step

# ─── Colors & Logging ─────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
LOG_FILE="/workspace/trellis2_setup.log"
mkdir -p /workspace
exec > >(tee -a "$LOG_FILE") 2>&1   # log everything to file AND stdout

log()  { echo -e "${GREEN}[$(date '+%H:%M:%S')] ✔ $*${NC}"; }
warn() { echo -e "${YELLOW}[$(date '+%H:%M:%S')] ⚠ $*${NC}"; }
err()  { echo -e "${RED}[$(date '+%H:%M:%S')] ✘ $*${NC}"; }
info() { echo -e "${BLUE}[$(date '+%H:%M:%S')] ℹ $*${NC}"; }
step() { echo -e "\n${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"; \
         echo -e "${BLUE}  STEP: $*${NC}"; \
         echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"; }

# ─── Configuration ────────────────────────────────────────────────────────────
WORKSPACE="/workspace"
COMFYUI_DIR="$WORKSPACE/ComfyUI"
CUSTOM_NODES_DIR="$COMFYUI_DIR/custom_nodes"
TRELLIS2_NODE_DIR="$CUSTOM_NODES_DIR/ComfyUI-Trellis2"
WHEELS_DIR="$TRELLIS2_NODE_DIR/wheels/Linux/Torch291"

# Model destinations — adjust these if the node expects a different path
TRELLIS_MODEL_DIR="$COMFYUI_DIR/models/TRELLIS.2-4B"
DINOV3_MODEL_DIR="$COMFYUI_DIR/models/facebook/dinov3-vitl16-pretrain-lvd1689m"

COMFYUI_REPO="https://github.com/comfyanonymous/ComfyUI.git"
TRELLIS2_REPO="https://github.com/visualbruno/ComfyUI-Trellis2.git"
HF_BASE="https://huggingface.co"

# Optional: set your HuggingFace token if TRELLIS.2-4B is gated
# export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxx"

COMFYUI_PORT=8188

# ─── Helper: safe download with resume + retry ────────────────────────────────
# Usage: download_file <url> <destination_path>
download_file() {
    local url="$1"
    local dest="$2"
    local dest_dir
    dest_dir="$(dirname "$dest")"
    mkdir -p "$dest_dir"

    if [ -f "$dest" ] && [ -s "$dest" ]; then
        info "  Already exists, skipping: $(basename "$dest")"
        return 0
    fi

    info "  Downloading: $(basename "$dest")"
    local hf_header=""
    if [[ "$url" == *"huggingface.co"* ]] && [ -n "${HF_TOKEN:-}" ]; then
        hf_header="--header=Authorization: Bearer $HF_TOKEN"
    fi

    # curl: -L follow redirects, -C - resume, --retry 3 on network errors
    if curl -fL -C - \
            ${HF_TOKEN:+--header "Authorization: Bearer $HF_TOKEN"} \
            --retry 3 --retry-delay 5 \
            --progress-bar \
            -o "$dest" \
            "$url"; then
        log "  Done: $(basename "$dest")"
    else
        err "  Failed to download: $url"
        rm -f "$dest"   # remove partial file
        return 1
    fi
}

# ─── Step 0: Verify environment (Python 3.12 + PyTorch pre-installed in image) ─
verify_environment() {
    step "Verifying pre-installed environment"

    # Python 3.12 is the system default in Ubuntu 24.04
    local py_ver
    py_ver=$(python3 --version 2>&1)
    if [[ "$py_ver" != *"3.12"* ]]; then
        err "Expected Python 3.12 but got: $py_ver"
        err "Make sure you are using image: runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404"
        exit 1
    fi
    log "Python: $py_ver"

    # PyTorch 2.9.1 is baked into the image — just verify it
    local torch_ver
    torch_ver=$(python3 -c "import torch; print(torch.__version__)" 2>&1)
    if [[ "$torch_ver" != 2.9* ]]; then
        err "Expected PyTorch 2.9.x but got: $torch_ver"
        exit 1
    fi
    log "PyTorch: $torch_ver"

    # Upgrade pip to latest
    python3 -m pip install --upgrade pip --quiet
    log "pip upgraded."
}

# ─── Step 1: Install / update ComfyUI ─────────────────────────────────────────
install_comfyui() {
    step "ComfyUI"
    if [ -d "$COMFYUI_DIR/.git" ]; then
        log "ComfyUI already cloned — pulling latest..."
        git -C "$COMFYUI_DIR" pull --ff-only || warn "git pull failed, continuing with existing version"
    else
        log "Cloning ComfyUI..."
        git clone --depth 1 "$COMFYUI_REPO" "$COMFYUI_DIR"
    fi

    # Ensure required model subdirectories exist
    mkdir -p "$COMFYUI_DIR/models/checkpoints"
    mkdir -p "$COMFYUI_DIR/models/vae"
    mkdir -p "$COMFYUI_DIR/models/loras"
    mkdir -p "$COMFYUI_DIR/input"
    mkdir -p "$COMFYUI_DIR/output"
    log "ComfyUI directory structure ready."
}

# ─── Step 2: Install ComfyUI Python requirements ──────────────────────────────
install_comfyui_deps() {
    step "ComfyUI core dependencies"
    python3 -m pip install -r "$COMFYUI_DIR/requirements.txt"
    log "ComfyUI dependencies installed."
}

# ─── Step 4: Clone / update ComfyUI-Trellis2 node ────────────────────────────
install_trellis2_node() {
    step "ComfyUI-Trellis2 custom node"
    if [ -d "$TRELLIS2_NODE_DIR/.git" ]; then
        log "ComfyUI-Trellis2 already cloned — pulling latest..."
        git -C "$TRELLIS2_NODE_DIR" pull --ff-only || warn "git pull failed, continuing with existing version"
    else
        log "Cloning ComfyUI-Trellis2..."
        git clone --depth 1 "$TRELLIS2_REPO" "$TRELLIS2_NODE_DIR"
    fi
    log "ComfyUI-Trellis2 node ready."
}

# ─── Step 5: Install pre-compiled wheels (Linux / Torch 2.9.1 / cp312) ────────
install_trellis2_wheels() {
    step "Trellis2 custom wheels (Linux/Torch291/cp312)"

    local wheels=(
        "cumesh-0.0.1-cp312-cp312-linux_x86_64.whl"
        "flex_gemm-0.0.1-cp312-cp312-linux_x86_64.whl"
        "nvdiffrast-0.4.0-cp312-cp312-linux_x86_64.whl"
        "nvdiffrec_render-0.0.0-cp312-cp312-linux_x86_64.whl"
        "o_voxel-0.0.1-cp312-cp312-linux_x86_64.whl"
    )

    for whl in "${wheels[@]}"; do
        local whl_path="$WHEELS_DIR/$whl"
        if [ ! -f "$whl_path" ]; then
            err "Wheel not found: $whl_path"
            err "Make sure ComfyUI-Trellis2 was cloned correctly."
            exit 1
        fi
        local pkg_name
        pkg_name=$(echo "$whl" | cut -d'-' -f1)
        if python3 -c "import $pkg_name" 2>/dev/null; then
            info "  Already installed: $pkg_name"
        else
            log "  Installing: $whl"
            python3 -m pip install "$whl_path" --no-deps
        fi
    done
    log "All custom wheels installed."
}

# ─── Step 6: Install Trellis2 Python requirements ─────────────────────────────
install_trellis2_requirements() {
    step "Trellis2 Python requirements"

    # Some packages (open3d, pymeshlab) can be slow or tricky — install with fallbacks
    local req_file="$TRELLIS2_NODE_DIR/requirements.txt"

    # Install all at once first; retry individually on failure
    if python3 -m pip install -r "$req_file"; then
        log "All Trellis2 requirements installed."
        return 0
    fi

    warn "Batch install had errors — retrying each package individually..."
    while IFS= read -r pkg || [ -n "$pkg" ]; do
        # Skip blank lines and comments
        [[ -z "$pkg" || "$pkg" == \#* ]] && continue
        python3 -m pip install "$pkg" || warn "  Could not install: $pkg (skipping)"
    done < "$req_file"
    log "Trellis2 requirements done (check warnings above for any skipped packages)."
}

# ─── Step 7: Download TRELLIS.2-4B model files (≈16 GB) ──────────────────────
download_trellis_models() {
    step "TRELLIS.2-4B model files"
    mkdir -p "$TRELLIS_MODEL_DIR/ckpts"

    local hf_repo="microsoft/TRELLIS.2-4B"

    # ── Root config files ──
    for f in "pipeline.json" "texturing_pipeline.json"; do
        download_file "$HF_BASE/$hf_repo/resolve/main/$f" "$TRELLIS_MODEL_DIR/$f"
    done

    # ── ckpts — JSON configs (small, fast) ──
    local config_files=(
        "shape_dec_next_dc_f16c32_fp16.json"
        "shape_enc_next_dc_f16c32_fp16.json"
        "slat_flow_img2shape_dit_1_3B_1024_bf16.json"
        "slat_flow_img2shape_dit_1_3B_512_bf16.json"
        "slat_flow_imgshape2tex_dit_1_3B_1024_bf16.json"
        "slat_flow_imgshape2tex_dit_1_3B_512_bf16.json"
        "ss_flow_img_dit_1_3B_64_bf16.json"
        "tex_dec_next_dc_f16c32_fp16.json"
        "tex_enc_next_dc_f16c32_fp16.json"
    )
    for f in "${config_files[@]}"; do
        download_file "$HF_BASE/$hf_repo/resolve/main/ckpts/$f" "$TRELLIS_MODEL_DIR/ckpts/$f"
    done

    # ── ckpts — safetensors weights (large, ~16 GB total) ──
    # Sizes for reference:
    #   shape_dec / tex_dec : ~948 MB each
    #   shape_enc / tex_enc : ~709 MB each
    #   slat / ss DiT       : ~2.58 GB each (×5)
    local weight_files=(
        "shape_dec_next_dc_f16c32_fp16.safetensors"
        "shape_enc_next_dc_f16c32_fp16.safetensors"
        "slat_flow_img2shape_dit_1_3B_1024_bf16.safetensors"
        "slat_flow_img2shape_dit_1_3B_512_bf16.safetensors"
        "slat_flow_imgshape2tex_dit_1_3B_1024_bf16.safetensors"
        "slat_flow_imgshape2tex_dit_1_3B_512_bf16.safetensors"
        "ss_flow_img_dit_1_3B_64_bf16.safetensors"
        "tex_dec_next_dc_f16c32_fp16.safetensors"
        "tex_enc_next_dc_f16c32_fp16.safetensors"
    )
    for f in "${weight_files[@]}"; do
        download_file "$HF_BASE/$hf_repo/resolve/main/ckpts/$f" "$TRELLIS_MODEL_DIR/ckpts/$f"
    done

    log "TRELLIS.2-4B models ready at: $TRELLIS_MODEL_DIR"
}

# ─── Step 8: Download DINOv3 model files (≈1.2 GB) ───────────────────────────
download_dinov3_model() {
    step "DINOv3-ViTL16 pretrain model"
    mkdir -p "$DINOV3_MODEL_DIR"

    local hf_repo="PIA-SPACE-LAB/dinov3-vitl-pretrain-lvd1689m"

    download_file "$HF_BASE/$hf_repo/resolve/main/model.safetensors"       "$DINOV3_MODEL_DIR/model.safetensors"
    download_file "$HF_BASE/$hf_repo/resolve/main/config.json"             "$DINOV3_MODEL_DIR/config.json"
    download_file "$HF_BASE/$hf_repo/resolve/main/preprocessor_config.json" "$DINOV3_MODEL_DIR/preprocessor_config.json"

    log "DINOv3 model ready at: $DINOV3_MODEL_DIR"
}

# ─── Step 9: Verify GPU ───────────────────────────────────────────────────────
verify_gpu() {
    step "GPU verification"
    python3 -c "
import torch
if not torch.cuda.is_available():
    print('WARNING: CUDA not available! Check your RunPod GPU allocation.')
else:
    name = torch.cuda.get_device_name(0)
    mem  = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f'GPU: {name}')
    print(f'VRAM: {mem:.1f} GB')
    print(f'CUDA version: {torch.version.cuda}')
    # Quick smoke test
    x = torch.ones(1024, 1024, device=\"cuda\")
    print(f'CUDA tensor test: PASSED (shape={x.shape})')
    del x
    torch.cuda.empty_cache()
"
}

# ─── Step 10: Launch ComfyUI ──────────────────────────────────────────────────
launch_comfyui() {
    step "Launching ComfyUI"
    log "Starting ComfyUI on port $COMFYUI_PORT..."
    log "Access via RunPod's port forwarding or proxy URL."

    cd "$COMFYUI_DIR"
    exec python3 main.py \
        --listen 0.0.0.0 \
        --port "$COMFYUI_PORT" \
        --preview-method auto \
        --use-pytorch-cross-attention
    # --use-pytorch-cross-attention: avoids xformers requirement, works natively on RTX 50xx
}

# ─── Main ─────────────────────────────────────────────────────────────────────
main() {
    echo -e "\n${GREEN}╔══════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║  ComfyUI + Trellis2  —  RunPod Auto-Setup            ║${NC}"
    echo -e "${GREEN}║  Python 3.12  |  PyTorch 2.9.1  |  CUDA 12.8.1      ║${NC}"
    echo -e "${GREEN}║  Log: $LOG_FILE  ║${NC}"
    echo -e "${GREEN}╚══════════════════════════════════════════════════════╝${NC}\n"

    START_TIME=$(date +%s)

    verify_environment
    install_comfyui
    install_comfyui_deps
    install_trellis2_node
    install_trellis2_wheels
    install_trellis2_requirements
    download_trellis_models     # ~16.2 GB — skipped if files already exist
    download_dinov3_model       # ~1.2 GB  — skipped if files already exist
    verify_gpu

    END_TIME=$(date +%s)
    ELAPSED=$(( END_TIME - START_TIME ))
    log "Setup complete in $(( ELAPSED / 60 ))m $(( ELAPSED % 60 ))s"

    launch_comfyui
}

main "$@"
