"""Verify the authorized compute machines before numerical campaigns."""

import platform
import socket
import subprocess


def require_compute_host():
    """Accept Oracle or a named Mac Studio; reject MacBooks and unknown hosts.

    The Studio's DHCP hostname is not a stable identifier. Check its hardware
    model as well as the configured ComputerName, instead of trusting an IP.
    """
    host = socket.getfqdn().lower()
    if host == "oracle.sns.it":
        return host
    if platform.system() == "Darwin":
        name = subprocess.check_output(["scutil", "--get", "ComputerName"], text=True).strip()
        model = subprocess.check_output(["sysctl", "-n", "hw.model"], text=True).strip()
        # Apple Studio desktop identifiers; MacBook identifiers are excluded.
        if "mac studio" in name.lower() and model in {
                "Mac13,1", "Mac13,2", "Mac14,13", "Mac14,14", "Mac15,14", "Mac16,9"}:
            return f"{host} ({name}; {model})"
    raise RuntimeError("Numerical campaigns require verified Oracle or MAC-STUDIO; laptop execution is forbidden")
