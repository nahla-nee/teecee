use std::{ffi::{c_char, c_int}, fmt::Debug, fs::File, io::Read, mem::ManuallyDrop, os::fd::{AsRawFd, FromRawFd}};

use log::{debug, error};
use teecee_parse::datalink::ethernet::{EthHeader, MacAddr};
use thiserror::Error;
use zerocopy::{FromBytes, IntoBytes};

// From tap.c, see file for function docs
unsafe extern "C" {
    fn create_tap(name: *mut c_char) -> c_int;
    fn get_tap_addr(fd: c_int, addr: *mut u8) -> c_int;
    fn close_tap(fd: c_int);
}

#[derive(Debug, Error)]
pub enum TapError {
    #[error("Failed to create tap device with error code {0}")]
    Create(c_int),
    #[error("Failed to get tap device address with error code {0}")]
    GetAddr(c_int),
}

pub struct TapDevice {
    fd: ManuallyDrop<File>,
    pub name: String,
    address: MacAddr,
    buffer: [u8; TapDevice::BUFFER_LEN]
}

impl TapDevice {
    const BUFFER_LEN: usize = 1522;

    pub fn new() -> Result<Self, TapError> {
        const MAX_NAME_LEN: usize = 16;

        let mut name: [u8; MAX_NAME_LEN] = *b"teecee%d\0\0\0\0\0\0\0\0";

        let first_null = name.iter()
            .enumerate()
            .find_map(|c| {
                if *c.1 == 0 {
                    Some(c.0)
                }
                else {
                    None
                }
            })
            .unwrap_or(MAX_NAME_LEN);

        let fd: ManuallyDrop<File> = unsafe {
            let fd = create_tap(name.as_mut_ptr() as _);

            if fd < 0 {
                return Err(TapError::Create(fd));
            }

            ManuallyDrop::new(FromRawFd::from_raw_fd(fd))
        };

        let mut address = MacAddr::new([0; 6]);
        unsafe {
            let ret = get_tap_addr(fd.as_raw_fd(), address.as_mut_bytes().as_mut_ptr()); 
            if ret != 0 {
                return Err(TapError::GetAddr(ret))
            }
        };

        Ok(Self {
            fd,
            name: str::from_utf8(&name[..first_null])
                .expect("Failed to create string from returned name")
                .to_owned(),
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
}

impl Debug for TapDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TapDevice")
            .field("name", &self.name)
            .field("address", &self.address)
            .finish()
    }
}

impl Drop for TapDevice {
    fn drop(&mut self) {
        unsafe {
            let fd = self.fd.as_raw_fd();
            close_tap(fd);
        }
    }
}