from contextlib import closing
import os
import socket
import fcntl
import struct


def get_free_port():
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


def get_hostname_ip():
    host_name = socket.gethostname()
    return socket.gethostbyname(host_name)


def get_network_interface():
    skip = {'docker', 'lo'}

    def good_interface(interface):
        for s in skip:
            if interface.startswith(s):
                return False

        return True

    interfaces = list(filter(good_interface, os.listdir('/sys/class/net/')))
    if interfaces:
        return interfaces[0]
    return None


def get_ip_address(ifname=None):
    if ifname is None:
        return get_hostname_ip()

    if isinstance(ifname, str):
        ifname = ifname.encode('utf8')

    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        return socket.inet_ntoa(fcntl.ioctl(
            s.fileno(),
            0x8915,     # SIOCGIFADDR
            struct.pack('256s', ifname[:15])
        )[20:24])

