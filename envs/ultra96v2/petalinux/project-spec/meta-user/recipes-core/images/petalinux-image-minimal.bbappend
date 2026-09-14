# Avnet's u96v2_sbc_base BSP only ships avnet-image-minimal.wks under
# meta-avnet, but the default petalinux-image-minimal recipe looks for
# <recipe>.<machine>.wks or <recipe>.wks. We ship the same 1 GB vfat + ext4
# layout next to this bbappend as petalinux-image-minimal.wks. ${THISDIR}
# must be captured immediately (:=) because for an .bbappend, deferred
# expansion resolves to the original recipe's directory, not the bbappend's.
WKS_FILE := "${THISDIR}/petalinux-image-minimal.wks"

# Pre-size the ext4 rootfs at image-gen time so the SD is ready-to-use
# straight after `dd`. ~3 GB extra on top of the rootfs (~750 MB) gives
# enough room for matplotlib + onnxruntime (~250 MB), PyCharm helpers
# (~120 MB), board-side venvs, sample JSON/PNG outputs, and headroom.
#
# The actual size on disk is much smaller than the apparent file size:
# wic produces a sparse .wic and a .wic.bmap. Use `bmaptool copy` for the
# flash and only the non-zero blocks (≈ 1.5 GB) traverse the USB-card-
# reader bus. Plain `dd` writes everything, sparse or not — slower.
# Value is in KB. 3000000 KB = ~3 GB extra.
IMAGE_ROOTFS_EXTRA_SPACE = "3000000"

# ---------------------------------------------------------------------------
# Phase B image overlay for the Ultra96-v2 — adapted for VAI 3.0 + 2022.2.
#
# The Avnet meta-avnet layer ships its own petalinux-image-minimal.bbappend
# (recipes-core/images/) that already pulls in openssh, python3, python3-pip,
# packagegroup-petalinux-{utils,benchmarks}, xrt, zocl, ethtool, usbutils,
# i2c-tools and many other staples for the u96v2-sbc machine. We only add
# what is specific to the ASPIR pipeline on top of that.
#
# Vitis AI runtime stack: the recipes live inside meta-user via two relative
# symlinks (meta-user/recipes-vitis-ai/ → vai_petalinux_recipes/recipes-vitis-ai/
# inside the Vitis-AI@v3.0 submodule under workspace/sources/, and
# meta-user/recipes-vai-kernel/ analogously). Recipe filenames at v3.0 are
# vart_3.0.bb, xir_3.0.bb, etc. — clean upstream, no bbappends needed
# (unlike the v4.0/2024.1 attempt which required 12 sed-based workarounds).
#
# Developer / inference tooling: ASPIR's TimingAnalyzer / EnergyAnalyzer
# rely on i2c/hwmon (kernel CONFIG_SENSORS_INA2XX=y, see bsp.cfg) and on
# a Python stack with numpy/pillow. ONNX Runtime is left to a one-time
# `pip install onnxruntime` on first boot to avoid pulling in the
# multi-hour CMake build of ORT from source.
# ---------------------------------------------------------------------------

# vitis-ai-library is INCLUDED here — at v3.0 the src/vai_library/ tree
# is complete and builds clean (the v4.0 brokenness with usefultools/cxir
# missing CMakeLists was a tag-HEAD issue specific to that release).
IMAGE_INSTALL:append:u96v2-sbc = " \
    fpga-manager-util \
    util-linux \
    util-linux-sfdisk \
    parted \
    e2fsprogs-resize2fs \
    vart \
    xir \
    target-factory \
    unilog \
    glog \
    vitis-ai-library \
    gdb \
    strace \
    htop \
    perf \
    tmux \
    vim \
    rsync \
    openssh-sftp-server \
    python3-numpy \
    python3-pillow \
    python3-pip \
"

# Route C bitstream loading: NOT auto-loaded at boot.
# The previous dpu-firmware recipe + dpu-load.service approach was fragile
# (paths mismatched between recipe install dir and what PetaLinux native
# FPGA_MANAGER actually produced, causing ConditionPathExists to fail
# silently and /dev/dpu to never appear post-reboot). We've ripped both
# out. Now the bitstream + DT overlay are loaded on demand via the
# /usr/local/bin/aspir-load-dpu helper installed below (see aspir_runtime_tweaks).
# Trade-off: requires one manual `aspir-load-dpu` invocation after each
# boot before the first inference. The script itself is idempotent
# (auto-detaches any prior overlay) and the helper self-elevates via
# the NOPASSWD sudoers entry, so the user just types `aspir-load-dpu`.
#
# util-linux is added to IMAGE_INSTALL above so that `shutdown -r now`
# and `reboot` work — sysvinit/procps versions don't ship by default
# on petalinux-image-minimal and we kept needing hard power-cycles.

# openssh-sftp-server installs /usr/libexec/sftp-server; dropbear (the active
# SSH daemon on this image) does not implement SFTP itself but invokes an
# external sftp-server binary when a client requests the "sftp" subsystem.
# PyCharm's "Deployment" feature speaks SFTP (not legacy scp), so installing
# just the sftp-server binary — without replacing dropbear with full openssh —
# is enough to enable file sync, remote interpreter sync and `scp` from any
# modern OpenSSH client (which has been SFTP-based since OpenSSH 9.0).

# ---------------------------------------------------------------------------
# Route C board conveniences — all done as rootfs postprocessing rather than
# a package, because a package can't co-own /etc/sudoers.d (sudo owns it) and
# this avoids RPM file-conflicts entirely. The target dirs (/etc/sudoers.d,
# /etc/udev/rules.d) already exist by the time POSTPROCESS runs.
#
#   1. XLNX_ENABLE_FINGERPRINT_CHECK=0 baked into /etc/profile.d/aspir.sh
#      AND into a per-user systemd-user-env hint. Why both: dropbear in this
#      build does NOT use PAM (so /etc/environment is ignored), and runs
#      `ssh host 'cmd'` as non-interactive non-login shells. Putting it in
#      profile.d catches interactive sessions; we re-export it from the
#      script-level wrappers in aspir-load-dpu and the inference scripts'
#      shebangs too for belt-and-braces. VART's fingerprint-check segfaults
#      (DpuKernel::get_fingerprint -> libxir vtable mismatch); skipping is
#      safe (our xmodel and DPU config match by construction).
#   2. Passwordless sudo for the petalinux user.
#   3. udev rule: /dev/dpu -> 0666 so inference runs without sudo.
#   4. Remove the forced "change password on first login" on petalinux/root
#      — the board is reflashed constantly and the expiry prompt is friction.
#   5. SSH key install: any host pubkey dropped under
#      workspace/scripts/aspir-ssh-pubkeys/*.pub is baked into the petalinux
#      user's ~/.ssh/authorized_keys. Lets you `ssh petalinux@board` from
#      day-one without sshpass + ssh-copy-id dance.
#   6. PATH: the non-interactive ssh PATH in dropbear is `/usr/bin:/bin`
#      only. Push /usr/local/bin, /usr/local/sbin, /sbin, /usr/sbin so
#      aspir-load-dpu, reboot, shutdown, fpgautil etc. resolve naturally.
# ---------------------------------------------------------------------------
ROOTFS_POSTPROCESS_COMMAND:append = " aspir_runtime_tweaks; "
aspir_runtime_tweaks() {
    # 1+6 — environment via profile.d (loaded by /bin/sh and bash on login;
    # the ssh wrapper script also exports it for non-login).
    install -d ${IMAGE_ROOTFS}${sysconfdir}/profile.d
    cat > ${IMAGE_ROOTFS}${sysconfdir}/profile.d/aspir.sh <<'EOF'
# Pushed by the ASPIR image bbappend. See petalinux-image-minimal.bbappend.
export XLNX_ENABLE_FINGERPRINT_CHECK=0
# Dropbear default PATH for non-interactive SSH is /usr/bin:/bin; extend it.
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/sbin:$PATH"
EOF
    # /etc/environment still gets the var as a fallback (some tools read it
    # directly without going through PAM/profile).
    if ! grep -q XLNX_ENABLE_FINGERPRINT_CHECK ${IMAGE_ROOTFS}${sysconfdir}/environment; then
        echo 'XLNX_ENABLE_FINGERPRINT_CHECK=0' >> ${IMAGE_ROOTFS}${sysconfdir}/environment
    fi
    # 2 — passwordless sudo.
    install -d ${IMAGE_ROOTFS}${sysconfdir}/sudoers.d
    echo 'petalinux ALL=(ALL) NOPASSWD: ALL' > ${IMAGE_ROOTFS}${sysconfdir}/sudoers.d/petalinux-nopasswd
    chmod 0440 ${IMAGE_ROOTFS}${sysconfdir}/sudoers.d/petalinux-nopasswd
    # 3 — udev rule.
    install -d ${IMAGE_ROOTFS}${sysconfdir}/udev/rules.d
    echo 'KERNEL=="dpu", MODE="0666"' > ${IMAGE_ROOTFS}${sysconfdir}/udev/rules.d/99-dpu.rules
    # 4 — no password expiry.
    chage -R ${IMAGE_ROOTFS} -M -1 -E -1 petalinux || true
    chage -R ${IMAGE_ROOTFS} -M -1 -E -1 root || true

    # 5 — pre-install host SSH pubkeys for petalinux. Looks in the repo at
    # workspace/scripts/aspir-ssh-pubkeys/*.pub (gitignored — each developer
    # drops their own). Without any, the recipe silently skips and the
    # default password 'petalinux' still works.
    aspir_pubkey_dir="${THISDIR}/../../../../scripts/aspir-ssh-pubkeys"
    install -d -m 0700 ${IMAGE_ROOTFS}/home/petalinux/.ssh
    if ls ${aspir_pubkey_dir}/*.pub >/dev/null 2>&1; then
        cat ${aspir_pubkey_dir}/*.pub > ${IMAGE_ROOTFS}/home/petalinux/.ssh/authorized_keys
        chmod 0600 ${IMAGE_ROOTFS}/home/petalinux/.ssh/authorized_keys
    fi
    # petalinux UID/GID is 1000:1000 in the rootfs. chown happens here even
    # if authorized_keys wasn't created — keeps .ssh/ usable by the user.
    chown -R 1000:1000 ${IMAGE_ROOTFS}/home/petalinux/.ssh

    install -d ${IMAGE_ROOTFS}/usr/local/bin
    cat > ${IMAGE_ROOTFS}/usr/local/bin/aspir-load-dpu <<'EOF'
#!/bin/sh
# aspir-load-dpu — hot-load the DPUCZDX8G bitstream + base DT overlay on demand.
# Idempotent: detaches any prior overlay first. Self-elevates via the NOPASSWD
# sudoers entry, so the petalinux user can just call `aspir-load-dpu` without sudo.
set -e
if [ "$(id -u)" -ne 0 ]; then
    exec sudo "$0" "$@"
fi
fpgautil -R >/dev/null 2>&1 || true
fpgautil -b /lib/firmware/xilinx/base/u96v2_sbc_base_wrapper.bit.bin \
         -o /lib/firmware/xilinx/base/base.dtbo
test -c /dev/dpu && echo "/dev/dpu ready: $(ls -l /dev/dpu)"
EOF
    chmod 0755 ${IMAGE_ROOTFS}/usr/local/bin/aspir-load-dpu

    # Companion to aspir-load-dpu: drops the PL back to its idle state so
    # phase_a_run_neon.py can measure NEON without the DPU's clock-tree
    # leakage inflating the baseline. Same NOPASSWD sudo elevation pattern.
    cat > ${IMAGE_ROOTFS}/usr/local/bin/aspir-unload-dpu <<'EOF'
#!/bin/sh
# aspir-unload-dpu — detach any FPGA overlay so the PL drops to its idle
# (mostly leakage) consumption. Mirror of aspir-load-dpu. Idempotent.
# Self-elevates via NOPASSWD sudoers entry.
set -e
if [ "$(id -u)" -ne 0 ]; then
    exec sudo "$0" "$@"
fi
fpgautil -R >/dev/null 2>&1 || true
if [ -c /dev/dpu ]; then
    echo "aspir-unload-dpu: WARNING /dev/dpu still present after fpgautil -R" >&2
    exit 1
fi
echo "aspir-unload-dpu: PL unloaded"
EOF
    chmod 0755 ${IMAGE_ROOTFS}/usr/local/bin/aspir-unload-dpu
}
