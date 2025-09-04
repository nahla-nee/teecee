use std::{ffi::CStr, fmt::Debug, fs::{File, OpenOptions}, io::Read, os::fd::{AsRawFd}};

use log::{error};
use teecee_parse::datalink::ethernet::{EthHeader, MacAddr};
use thiserror::Error;
use zerocopy::{FromBytes, IntoBytes};

#[derive(Debug, Error)]
pub enum TapError {
    #[error("Failed to open tap file descriptor {0}")]
    Create(std::io::Error),
    #[error("Failed to set tap info with ioctl: {0}")]
    IoctlSet(std::io::Error),
    #[error("Failed to get tap info with ioctl: {0}")]
    IoctlGet(std::io::Error),
    #[error("Failed to convert name to utf8: {0}")]
    NameUtf8(std::string::FromUtf8Error),
}

pub struct TapDevice {
    fd: File,
    pub name: String,
    address: MacAddr,
    buffer: [u8; TapDevice::BUFFER_LEN]
}

impl TapDevice {
    const BUFFER_LEN: usize = 1522;

    pub fn new(name: &CStr) -> Result<Self, TapError> {
        let fd = OpenOptions::new()
            .read(true)
            .write(true)
            .open("/dev/net/tun")
            .map_err(TapError::Create)?;

        let name = Self::set_tap_with_name(&fd, name)?;
        let address = Self::get_tap_address(&fd)?;

        Ok(Self {
            fd,
            name: name,
            address,
            buffer: [0; Self::BUFFER_LEN]
        })
    }

    pub fn read_frame(&mut self) -> Option<&EthHeader> {
        let read_len = match self.fd.read(&mut self.buffer) {
            Ok(len) => Some(len),
            Err(e) => {
                error!("Failed to read from file descriptor: {e}");
                None
            },
        }?;

        let read_len = std::cmp::min(read_len, size_of::<EthHeader>());

        match EthHeader::ref_from_bytes(&self.buffer[..read_len]) {
            Ok(hdr) => Some(hdr),
            Err(e) => {
                error!("Error while creating EthHeader: {e}");
                None
            },
        }
    }

    fn set_tap_with_name(fd: &File, name: &CStr) -> Result<String, TapError> {
        const MAX_NAME_LEN: usize = 16;

        let name = unsafe {
            let n = name.to_bytes_with_nul();
            std::slice::from_raw_parts(n.as_ptr(), n.len())
        };

        assert!(name.len() <= MAX_NAME_LEN);

        let mut ifreq: libc::ifreq;

        let ret = unsafe {
            ifreq = std::mem::zeroed();

            let name = std::slice::from_raw_parts(name.as_ptr() as _, name.len());
            ifreq.ifr_name[..name.len()].copy_from_slice(name);
            ifreq.ifr_ifru.ifru_flags = (libc::IFF_TAP | libc::IFF_NO_PI) as _;

            libc::ioctl(fd.as_raw_fd(), libc::TUNSETIFF, &mut ifreq as *mut _)
        };

        if ret < 0 {
            return Err(TapError::IoctlSet(std::io::Error::last_os_error()));
        }

        let name_bytes = ifreq.ifr_name.iter()
            .map_while(|c| {
                if *c != 0 {
                    Some(*c as u8)
                }
                else {
                    None
                }
            })
            .collect::<Vec<u8>>();

        String::from_utf8(name_bytes).map_err(TapError::NameUtf8)
    }

    fn get_tap_address(fd: &File) -> Result<MacAddr, TapError> {
        let mut tap_mac = MacAddr::new([0; 6]);
        let mut ifreq: libc::ifreq;

        let ret = unsafe {
            ifreq = std::mem::zeroed();

            libc::ioctl(fd.as_raw_fd(), libc::SIOCGIFHWADDR, &mut ifreq as *mut _)
        };

        if ret < 0 {
            return Err(TapError::IoctlGet(std::io::Error::last_os_error()));
        }

        let hw_addr = unsafe {
            &ifreq.ifr_ifru.ifru_hwaddr.sa_data[..6]
        };

        tap_mac.as_mut_bytes().iter_mut().zip(hw_addr.iter())
            .for_each(|(tap, hw)| {
                *tap = *hw as u8;
            });

        Ok(tap_mac)
    }
}

impl Debug for TapDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TapDevice")
            .field("name", &self.name)
            .field("address", &self.address)
            .finish()
    }
}
