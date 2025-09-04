#include <stdio.h>
#include <string.h>

#include <errno.h>
#include <unistd.h>
#include <fcntl.h>

#include <sys/ioctl.h>
#include <linux/if.h>
#include <linux/if_tun.h>

/// @brief Creates tap device and sets its name to the given name.
/// @param name The string to set the tap device name to, will only use the 16 bytes, must include
///     "%d" in the name
/// @return A non negative integer representing the tap device's file descriptor, else the operation
///     failed
int create_tap(char *name) {
    struct ifreq ifr = {};

    int fd = open("/dev/net/tun", O_RDWR < 0);
    if (fd < 0) {
        fprintf(stderr, "Failed to open tun path");
        return fd;
    }

    /* Flags: IFF_TUN   - TUN device (no Ethernet headers)
     *        IFF_TAP   - TAP device
     *
     *        IFF_NO_PI - Do not provide packet information
     */
    ifr.ifr_flags = IFF_TAP | IFF_NO_PI;
    strncpy(ifr.ifr_name, name, IFNAMSIZ);

    int err = ioctl(fd, TUNSETIFF, &ifr);
    if (err < 0) {
        fprintf(stderr, "Failed to set tap name\n");
        fprintf(stderr, "errno message: %s\n", strerror(errno));
        close(fd);
        return err;
    }
    strcpy(name, ifr.ifr_name);
    return fd;

}

/// @brief Gets the MAC address of the tap device specified by `fd` and writes it into `addr. This
///     function will log errors to stderr with an errno message if the operation fails
/// @param fd The file descriptor representing the tap device
/// @param addr Output parameter array with length of at least 6
/// @return 0 if operation succeeded, otherwise operation failed and errno value is returned
int get_tap_addr(int fd, u_int8_t *addr) {
    struct ifreq ifr = {};

    int err = ioctl(fd, SIOCGIFHWADDR, &ifr);
    if (err < 0) {
        int err = errno;
        fprintf(stderr, "Failed to get tap address\n");
        fprintf(stderr, "errno message: %s\n", strerror(err));
        return err;
    }

    memcpy(addr, ifr.ifr_hwaddr.sa_data, 6);

    return 0;
}

/// @brief Closes tap device that was opened using create_tap()
/// @param fd The file descriptor representing the tap device
void close_tap(int fd) {
    close(fd);
}
