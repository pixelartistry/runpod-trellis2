#!/bin/bash
# =============================================================================
# ComfyUI + Trellis2 RunPod Auto-Setup Script
# Image:  runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404
# Target: RTX 5090 | Python 3.12 | PyTorch 2.9.1 (installed into venv) | CUDA 12.8.1
#
# All packages are installed into /workspace/venv so they survive pod restarts
# and work across any new pod that mounts the same Network Volume.
# =============================================================================

set -uo pipefail

# ─── Colors & Logging ─────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
LOG_FILE="/workspace/trellis2_setup.log"
mkdir -p /workspace

log()  { local msg="[$(date '+%H:%M:%S')] ✔ $*"; echo -e "${GREEN}${msg}${NC}"; echo "$msg" >> "$LOG_FILE"; }
warn() { local msg="[$(date '+%H:%M:%S')] ⚠ $*"; echo -e "${YELLOW}${msg}${NC}"; echo "$msg" >> "$LOG_FILE"; }
err()  { local msg="[$(date '+%H:%M:%S')] ✘ $*"; echo -e "${RED}${msg}${NC}";    echo "$msg" >> "$LOG_FILE"; }
info() { local msg="[$(date '+%H:%M:%S')] ℹ $*"; echo -e "${BLUE}${msg}${NC}";  echo "$msg" >> "$LOG_FILE"; }
step() { echo -e "\n${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
         echo -e "${BLUE}  STEP: $*${NC}"
         echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"; }

# ─── Configuration ────────────────────────────────────────────────────────────
WORKSPACE="/workspace"
VENV="$WORKSPACE/venv"            # <-- lives on Network Volume, survives pod restarts
PY="$VENV/bin/python"
PIP="$VENV/bin/pip"

COMFYUI_DIR="$WORKSPACE/ComfyUI"
CUSTOM_NODES_DIR="$COMFYUI_DIR/custom_nodes"
TRELLIS2_NODE_DIR="$CUSTOM_NODES_DIR/ComfyUI-Trellis2"
WHEELS_DIR="$TRELLIS2_NODE_DIR/wheels/Linux/Torch291"

TRELLIS_MODEL_DIR="$COMFYUI_DIR/models/TRELLIS.2-4B"
DINOV3_MODEL_DIR="$COMFYUI_DIR/models/facebook/dinov3-vitl16-pretrain-lvd1689m"

COMFYUI_REPO="https://github.com/comfyanonymous/ComfyUI.git"
TRELLIS2_REPO="https://github.com/visualbruno/ComfyUI-Trellis2.git"
HF_BASE="https://huggingface.co"

# Optional: HuggingFace token if any model repo is gated
# export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxx"

COMFYUI_PORT=8188

# ─── Helper: download with resume + retry ─────────────────────────────────────
download_file() {
    local url="$1" dest="$2"
    mkdir -p "$(dirname "$dest")"

    if [ -f "$dest" ] && [ -s "$dest" ]; then
        info "  Already exists, skipping: $(basename "$dest")"
        return 0
    fi

    info "  Downloading: $(basename "$dest")"
    if curl -fL -C - \
            ${HF_TOKEN:+--header "Authorization: Bearer $HF_TOKEN"} \
            --retry 3 --retry-delay 5 \
            --progress-bar \
            -o "$dest" "$url"; then
        log "  Done: $(basename "$dest")"
    else
        err "  Failed: $url"
        rm -f "$dest"
        return 1
    fi
}

# ─── Step 1: Create venv on Network Volume ────────────────────────────────────
create_venv() {
    step "Creating virtualenv in $VENV"
    python3 -m venv "$VENV" --system-site-packages
    # --system-site-packages: inherits system numpy/etc, only overrides what we install
    "$PIP" install --upgrade pip wheel setuptools --quiet
    log "Virtualenv ready."
}

# ─── Step 2: Install PyTorch 2.9.1 into venv ─────────────────────────────────
install_pytorch() {
    step "Installing PyTorch 2.9.1 + CUDA 12.8"
    # Uninstall the 2.8.0 that came with the base image (from system-site-packages)
    "$PIP" install \
        torch==2.9.1 torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cu128

    "$PY" -c "
import torch
print(f'PyTorch : {torch.__version__}')
print(f'CUDA    : {torch.version.cuda}')
print(f'GPU     : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"none\"}')
"
    log "PyTorch 2.9.1 installed."
}

# ─── Step 3: Clone / update ComfyUI ──────────────────────────────────────────
install_comfyui() {
    step "ComfyUI"
    if [ -d "$COMFYUI_DIR/.git" ]; then
        log "ComfyUI already cloned — pulling latest..."
        git -C "$COMFYUI_DIR" pull --ff-only || warn "git pull failed, continuing with existing version"
    else
        log "Cloning ComfyUI..."
        git clone --depth 1 "$COMFYUI_REPO" "$COMFYUI_DIR"
    fi
    mkdir -p "$COMFYUI_DIR/models/checkpoints" \
             "$COMFYUI_DIR/models/vae" \
             "$COMFYUI_DIR/models/loras" \
             "$COMFYUI_DIR/input" \
             "$COMFYUI_DIR/output"
    log "ComfyUI ready."
}

# ─── Step 4: Install ComfyUI dependencies ─────────────────────────────────────
install_comfyui_deps() {
    step "ComfyUI core dependencies"
    "$PIP" install -r "$COMFYUI_DIR/requirements.txt"
    log "ComfyUI dependencies installed."
}

# ─── Step 5: Clone / update ComfyUI-Trellis2 node ────────────────────────────
install_trellis2_node() {
    step "ComfyUI-Trellis2 custom node"
    if [ -d "$TRELLIS2_NODE_DIR/.git" ]; then
        log "Already cloned — pulling latest..."
        git -C "$TRELLIS2_NODE_DIR" pull --ff-only || warn "git pull failed, continuing with existing version"
    else
        log "Cloning ComfyUI-Trellis2..."
        git clone --depth 1 "$TRELLIS2_REPO" "$TRELLIS2_NODE_DIR"
    fi
    log "ComfyUI-Trellis2 ready."
}

# ─── Step 6: Install pre-compiled wheels (Linux / Torch 2.9.1 / cp312) ───────
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
            err "Wheel not found: $whl_path — make sure ComfyUI-Trellis2 cloned correctly."
            exit 1
        fi
        log "  Installing: $whl"
        "$PIP" install "$whl_path" --no-deps
    done
    log "All custom wheels installed."
}

# ─── Step 7: Install Trellis2 Python requirements ────────────────────────────
install_trellis2_requirements() {
    step "Trellis2 Python requirements"
    local req_file="$TRELLIS2_NODE_DIR/requirements.txt"

    if "$PIP" install -r "$req_file"; then
        log "All Trellis2 requirements installed."
        return 0
    fi

    warn "Batch install had errors — retrying each package individually..."
    while IFS= read -r pkg || [ -n "$pkg" ]; do
        [[ -z "$pkg" || "$pkg" == \#* ]] && continue
        "$PIP" install "$pkg" || warn "  Could not install: $pkg (skipping)"
    done < "$req_file"
    log "Trellis2 requirements done."
}

# ─── Step 8: Download TRELLIS.2-4B models (~16 GB) ───────────────────────────
download_trellis_models() {
    step "TRELLIS.2-4B model files"
    mkdir -p "$TRELLIS_MODEL_DIR/ckpts"
    local hf_repo="microsoft/TRELLIS.2-4B"

    for f in "pipeline.json" "texturing_pipeline.json"; do
        download_file "$HF_BASE/$hf_repo/resolve/main/$f" "$TRELLIS_MODEL_DIR/$f"
    done

    local ckpt_files=(
        "shape_dec_next_dc_f16c32_fp16.json"
        "shape_enc_next_dc_f16c32_fp16.json"
        "slat_flow_img2shape_dit_1_3B_1024_bf16.json"
        "slat_flow_img2shape_dit_1_3B_512_bf16.json"
        "slat_flow_imgshape2tex_dit_1_3B_1024_bf16.json"
        "slat_flow_imgshape2tex_dit_1_3B_512_bf16.json"
        "ss_flow_img_dit_1_3B_64_bf16.json"
        "tex_dec_next_dc_f16c32_fp16.json"
        "tex_enc_next_dc_f16c32_fp16.json"
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
    for f in "${ckpt_files[@]}"; do
        download_file "$HF_BASE/$hf_repo/resolve/main/ckpts/$f" "$TRELLIS_MODEL_DIR/ckpts/$f"
    done
    log "TRELLIS.2-4B models ready."
}

# ─── Step 9: Download DINOv3 model (~1.2 GB) ─────────────────────────────────
download_dinov3_model() {
    step "DINOv3-ViTL16 pretrain model"
    local hf_repo="PIA-SPACE-LAB/dinov3-vitl-pretrain-lvd1689m"
    download_file "$HF_BASE/$hf_repo/resolve/main/model.safetensors"        "$DINOV3_MODEL_DIR/model.safetensors"
    download_file "$HF_BASE/$hf_repo/resolve/main/config.json"              "$DINOV3_MODEL_DIR/config.json"
    download_file "$HF_BASE/$hf_repo/resolve/main/preprocessor_config.json" "$DINOV3_MODEL_DIR/preprocessor_config.json"
    log "DINOv3 model ready."
}

# ─── Step 10: Verify GPU ──────────────────────────────────────────────────────
verify_gpu() {
    step "GPU verification"
    "$PY" -c "
import torch
if not torch.cuda.is_available():
    print('WARNING: CUDA not available!')
else:
    name = torch.cuda.get_device_name(0)
    mem  = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f'GPU          : {name}')
    print(f'VRAM         : {mem:.1f} GB')
    print(f'PyTorch      : {torch.__version__}')
    print(f'CUDA         : {torch.version.cuda}')
    x = torch.ones(1024, 1024, device='cuda')
    print(f'Tensor test  : PASSED')
    del x; torch.cuda.empty_cache()
"
}

# ─── Step 11: Launch ComfyUI ──────────────────────────────────────────────────
launch_comfyui() {
    step "Launching ComfyUI on port $COMFYUI_PORT"
    cd "$COMFYUI_DIR"
    exec "$PY" main.py \
        --listen 0.0.0.0 \
        --port "$COMFYUI_PORT" \
        --preview-method auto \
        --use-pytorch-cross-attention
}

# ─── Main ─────────────────────────────────────────────────────────────────────
main() {
    echo -e "\n${GREEN}╔══════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║  ComfyUI + Trellis2  —  RunPod Auto-Setup            ║${NC}"
    echo -e "${GREEN}║  Python 3.12  |  PyTorch 2.9.1  |  CUDA 12.8.1      ║${NC}"
    echo -e "${GREEN}╚══════════════════════════════════════════════════════╝${NC}\n"

    START_TIME=$(date +%s)

    if [ -d "$VENV" ] && [ -f "$VENV/bin/activate" ]; then
        # ── Fast boot: venv already exists on Network Volume ──────────────────
        log "Existing venv found at $VENV — skipping installs"
        log "To force a full reinstall: rm -rf $VENV and restart the pod"
        verify_gpu
    else
        # ── First boot: full install into Network Volume venv ─────────────────
        log "No venv found — running first-time setup..."

        create_venv
        install_pytorch
        install_comfyui
        install_comfyui_deps
        install_trellis2_node
        install_trellis2_wheels
        install_trellis2_requirements
        download_trellis_models    # ~16 GB  — skips files that already exist
        download_dinov3_model      # ~1.2 GB — skips files that already exist
        verify_gpu

        log "First-time setup complete. Future boots will be fast."
    fi

    END_TIME=$(date +%s)
    ELAPSED=$(( END_TIME - START_TIME ))
    log "Ready in $(( ELAPSED / 60 ))m $(( ELAPSED % 60 ))s"

    launch_comfyui
}

main "$@"
