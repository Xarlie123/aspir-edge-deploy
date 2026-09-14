FILESEXTRAPATHS:prepend := "${THISDIR}/${PN}:"

# Kernel config fragment renamed from the generic "bsp.cfg" to a unique
# name: meta-avnet and recipes-vai-kernel BOTH also ship a "file://bsp.cfg",
# so three SRC_URI entries collided on the same WORKDIR filename and the
# last-fetched one won — silently dropping our DPU/INA2XX config. A unique
# filename avoids the collision.
SRC_URI:append = " file://route-c.cfg"
KERNEL_FEATURES:append = " route-c.cfg"
