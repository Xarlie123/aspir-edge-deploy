FILESEXTRAPATHS:prepend := "${THISDIR}/files:"

SRC_URI:append = " file://system-user.dtsi"

# Route C — provide system-conf.dtsi. PetaLinux 2022.2 does not
# auto-generate it for the --get-hw-description path on this Avnet BSP,
# but meta-avnet's system-bsp.dtsi requires it (/include/ line 1).
# Ship it as a SRC_URI file and drop it into DT_FILES_PATH on every
# do_configure, mirroring how meta-avnet places system-bsp.dtsi —
# this keeps it durable across do_configure regenerations.
SRC_URI:append = " file://system-conf.dtsi"

do_configure:append() {
	if [ -e ${WORKDIR}/system-conf.dtsi ]; then
		cp ${WORKDIR}/system-conf.dtsi ${DT_FILES_PATH}/system-conf.dtsi
	fi
}
