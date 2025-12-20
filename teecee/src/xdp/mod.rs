use std::{
    io::Error as ioError,
    ops::{Deref, DerefMut},
    os::fd::{AsRawFd, RawFd},
    ptr::NonNull,
    sync::atomic::AtomicU32,
};

use thiserror::Error;

struct Mmap {
    addr: NonNull<libc::c_void>,
    len: usize,
}

impl Mmap {
    fn new(len: usize) -> Result<Self, ioError> {
        let addr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                len,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_ANONYMOUS | libc::MAP_PRIVATE,
                -1,
                0,
            )
        };

        let addr = NonNull::new(addr).ok_or_else(|| ioError::last_os_error())?;

        Ok(Mmap { addr, len })
    }

    /// Returns a chunk of memory for use in a ring buffer or an error
    ///
    /// # Parameters
    ///
    /// * `fd`: the `AF_XDP` file descriptor to which the umem and ring buffers belong to
    /// * `len`: the amount of memory to allocate for the ring buffer
    ///
    /// # Safety
    /// The caller must ensure that:
    /// * `len` is the correct amount of memory to allocate for the ring
    unsafe fn new_ring<R: RingBuffer>(fd: &XdpFd, len: usize) -> Result<Self, ioError> {
        let addr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                len,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED | libc::MAP_POPULATE,
                fd.as_raw_fd(),
                R::MMAP_OFFSET,
            )
        };

        Ok(Self {
            addr: NonNull::new(addr).ok_or_else(ioError::last_os_error)?,
            len,
        })
    }

    /// Returns a pointer offset from the allocated memory by the specified amount or [None]
    /// if the offset is greater than the amount of memory allocated
    fn offset_by(&self, offset: usize) -> Option<NonNull<libc::c_void>> {
        // short circuit so we know that offset is > 0 before we cast as usize
        if offset as usize > self.len {
            None
        } else {
            Some(unsafe { self.addr.add(offset) })
        }
    }
}

impl Drop for Mmap {
    fn drop(&mut self) {
        unsafe {
            libc::munmap(self.addr.as_mut(), self.len);
        }
    }
}

/// A trait representing the properties of a ring buffer
///
/// # Safety
/// The implementor must ensure that:
/// * `BufferElementType` is the correct type of data that the ring buffer will hold
/// * `SETSOCKOPT_NAME` is the correct value to pass to [libc::setsockopt] for the ring type this
/// trait is being implemented for
/// * `MMAP_OFFSET` is the correct value to pass as the offset parameter to [libc::mmap] for the
/// ring type this trait is being implemented for
/// * `ELEMENT_SIZE` is the size of each entry in the buffer, that is it is the size of
/// `BufferElementType`. This is already the case by default, so do not modify this value
/// * `RING_LEN` is the a positive power of two
unsafe trait RingBuffer {
    type BufferElementType;

    const SETSOCKOPT_NAME: i32;
    const MMAP_OFFSET: libc::off_t;

    const ELEMENT_SIZE: usize = size_of::<Self::BufferElementType>();
    const RING_LEN: usize = 512;

    /// Constructs a ring buffer from pointers
    ///
    /// # Safety
    /// The caller must ensure that:
    /// * `base` is a pointer to an mmaped memory region representing a buffer ring
    /// * `offsets` is the offset struct associated with the mapped memory region
    unsafe fn new(fd: &XdpFd, offsets: libc::xdp_ring_offset) -> Result<Self, ioError>
    where Self: Sized;
}

macro_rules! new_ring_buffer {
    ($name:ident, $data:ty, $sockopt:expr, $mmap:expr) => {
        struct $name {
            mem: Mmap,
            producer: NonNull<AtomicU32>,
            consumer: NonNull<AtomicU32>,
            data: NonNull<[$data; Self::RING_LEN]>,
        }

        unsafe impl RingBuffer for $name {
            type BufferElementType = $data;

            const SETSOCKOPT_NAME: i32 = $sockopt;
            const MMAP_OFFSET: libc::off_t = $mmap;

            unsafe fn new(fd: &XdpFd, offsets: libc::xdp_ring_offset) -> Result<Self, ioError> {
                assert!(
                    Self::RING_LEN > 0 && Self::RING_LEN.count_ones() == 1,
                    "Buffer data length is not a positive power of two"
                );

                let mem = unsafe {
                    Mmap::new_ring::<Self>(
                        fd,
                        offsets.desc as usize + Self::RING_LEN as usize * Self::ELEMENT_SIZE,
                    )
                }?;

                let producer = mem
                    .offset_by(offsets.producer as _)
                    .expect("Failed to get producer offset pointer")
                    .cast();
                let consumer = mem
                    .offset_by(offsets.consumer as _)
                    .expect("Failed to get consumer offset pointer")
                    .cast();
                let data = mem
                    .offset_by(offsets.desc as _)
                    .expect("Failed to get desc offset pointer")
                    .cast();

                let data: &[_; Self::RING_LEN] =
                    unsafe { std::slice::from_raw_parts(data.as_ptr(), Self::RING_LEN) }
                        .try_into()
                        .expect("Failed to convert data from slice to array");

                Ok(Self {
                    mem,
                    producer,
                    consumer,
                    data: NonNull::from_ref(data),
                })
            }
        }
    };
}

new_ring_buffer!(
    RxBuffer,
    libc::xdp_desc,
    libc::XDP_RX_RING,
    libc::XDP_PGOFF_RX_RING
);
new_ring_buffer!(
    TxBuffer,
    libc::xdp_desc,
    libc::XDP_TX_RING,
    libc::XDP_PGOFF_TX_RING
);
new_ring_buffer!(
    FillBuffer,
    u64,
    libc::XDP_UMEM_FILL_RING,
    libc::XDP_UMEM_PGOFF_FILL_RING as _
);
new_ring_buffer!(
    CompletionBuffer,
    u64,
    libc::XDP_UMEM_COMPLETION_RING,
    libc::XDP_UMEM_PGOFF_COMPLETION_RING as _
);

struct Umem(Mmap);

impl Umem {
    const CHUNK_SIZE: usize = 4096;
    const CHUNK_COUNT: usize = 4096;
    const BUF_LEN: usize = Self::CHUNK_SIZE * Self::CHUNK_COUNT;

    /// Allocates a page aligned buffer of size [Self::BUF_LEN]
    pub fn new() -> Result<Umem, ioError> {
        Ok(Umem(Mmap::new(Self::BUF_LEN)?))
    }
}

impl Deref for Umem {
    type Target = [u8; Self::BUF_LEN];

    fn deref(&self) -> &Self::Target {
        unsafe { self.0.addr.cast().as_ref() }
    }
}

impl DerefMut for Umem {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { self.0.addr.cast().as_mut() }
    }
}

struct XdpFd(libc::c_int);

impl XdpFd {
    fn new() -> Result<XdpFd, ioError> {
        // Safety:
        // This function returns -1 on failure to create a socket, and is not unsafe to call.
        // We check that the fd returned is valid before creating an XdpFd out of it which ensures
        // a valid struct.
        let fd = unsafe { libc::socket(libc::AF_XDP, libc::SOCK_RAW, 0) };
        if fd == -1 {
            return Err(ioError::last_os_error());
        }

        Ok(Self(fd))
    }
}

impl AsRawFd for XdpFd {
    fn as_raw_fd(&self) -> RawFd {
        self.0
    }
}

impl Drop for XdpFd {
    fn drop(&mut self) {
        // Safety:
        // This function is safe to call on a file descriptor so long as it is a valid one
        // returned from a socket or open call. We have a valid file descriptor as it is
        // is not possible to construct an instance of XdpFd without one.
        unsafe {
            libc::close(self.0);
        }
    }
}

#[derive(Debug, Error)]
pub enum XdpSockError {
    #[error("Failed to create xdp socket: {0}")]
    Socket(ioError),
    #[error("Failed to allocate umem buffer")]
    UmemAlloc(ioError),
    #[error("Failed to register umem buffer: {0}")]
    UmemReg(ioError),
    #[error("Failed to allocate ring buffers: {0}\nContext: {1}")]
    RingAlloc(ioError, &'static str),
    #[error("Failed to find an appropriate network interface")]
    IfiSearchFail,
    #[error("Failed to find an appropriate network interface due to error: {0}")]
    IfiSearchError(ioError),
    #[error("Failed to bind xdp socket to network inteface: {0}")]
    BindError(ioError),
}

pub struct XdpSock {
    fd: XdpFd,
    umem: Umem,
    rx: RxBuffer,
    tx: TxBuffer,
    fill: FillBuffer,
    completion: CompletionBuffer,
}

impl XdpSock {
    pub fn new() -> Result<XdpSock, XdpSockError> {
        let fd = XdpFd::new().map_err(XdpSockError::Socket)?;

        let umem = Umem::new().map_err(XdpSockError::UmemAlloc)?;
        Self::register_umem(&fd, &umem).map_err(XdpSockError::UmemReg)?;

        let (rx, tx, fill, completion) = Self::get_ring_buffers(&fd)?;

        Self::bind(&fd, None)?;

        Ok(Self {
            fd,
            umem,
            rx,
            tx,
            fill,
            completion,
        })
    }

    fn get_ring_buffers(
        fd: &XdpFd,
    ) -> Result<(RxBuffer, TxBuffer, FillBuffer, CompletionBuffer), XdpSockError> {
        Self::register_ring_buffer::<RxBuffer>(fd)
            .map_err(|e| XdpSockError::RingAlloc(e, "Failed to register rx buffer"))?;
        Self::register_ring_buffer::<TxBuffer>(fd)
            .map_err(|e| XdpSockError::RingAlloc(e, "Failed to register tx buffer"))?;
        Self::register_ring_buffer::<FillBuffer>(fd)
            .map_err(|e| XdpSockError::RingAlloc(e, "Failed to register fill buffer"))?;
        Self::register_ring_buffer::<CompletionBuffer>(fd)
            .map_err(|e| XdpSockError::RingAlloc(e, "Failed to register completion buffer"))?;

        let offsets = Self::get_ring_offsets(&fd)
            .map_err(|e| XdpSockError::RingAlloc(e, "Failed to get ring offsets"))?;

        unsafe {
            Ok((
                RxBuffer::new(fd, offsets.rx)
                    .map_err(|e| XdpSockError::RingAlloc(e, "Failed to create rx buffer"))?,
                TxBuffer::new(fd, offsets.tx)
                    .map_err(|e| XdpSockError::RingAlloc(e, "Failed to create tx buffer"))?,
                FillBuffer::new(fd, offsets.fr)
                    .map_err(|e| XdpSockError::RingAlloc(e, "Failed to create fill buffer"))?,
                CompletionBuffer::new(fd, offsets.cr).map_err(|e| {
                    XdpSockError::RingAlloc(e, "Failed to create completion buffer")
                })?,
            ))
        }
    }

    fn get_ring_offsets(fd: &XdpFd) -> Result<libc::xdp_mmap_offsets, ioError> {
        let mut offsets = unsafe { std::mem::zeroed() };

        unsafe { Self::getsockopt(fd, libc::XDP_MMAP_OFFSETS, &mut offsets) }?;
        Ok(offsets)
    }

    /// Attempts to allocate a ring buffer of the specified type and size with the kernel
    fn register_ring_buffer<T: RingBuffer>(fd: &XdpFd) -> Result<(), ioError> {
        // ring buffer length is always a power of two, invariant ensured by constructor
        unsafe { Self::setsockopt(fd, T::SETSOCKOPT_NAME, &T::RING_LEN) }
    }

    /// Attempts to register the umem buffer with the kernel
    fn register_umem(fd: &XdpFd, umem: &Umem) -> Result<(), ioError> {
        let umem_reg = libc::xdp_umem_reg {
            addr: umem.as_ptr() as _,
            len: umem.len() as _,
            chunk_size: Umem::CHUNK_SIZE as _,
            headroom: 0,
            flags: 0,
            tx_metadata_len: 0,
        };

        // Safety: XDP_UMEM_REG is a valid operation and umem_reg is a valid parameter to pass it
        unsafe { Self::setsockopt(&fd, libc::XDP_UMEM_REG, &umem_reg) }
    }

    fn bind(fd: &XdpFd, ifindex: Option<u32>) -> Result<(), XdpSockError> {
        let ifindex = ifindex
            .map_or_else(Self::autodetect_ifindex, |v| Ok(Some(v)))
            .map_err(XdpSockError::IfiSearchError)?
            .ok_or(XdpSockError::IfiSearchFail)?;

        let sockaddr = libc::sockaddr_xdp {
            sxdp_family: libc::AF_XDP as _,
            sxdp_flags: 0,
            sxdp_ifindex: ifindex,
            sxdp_queue_id: 0,
            sxdp_shared_umem_fd: fd.as_raw_fd() as _,
        };

        let ret = unsafe {
            libc::bind(
                fd.as_raw_fd(),
                &sockaddr as *const _ as *const _,
                size_of::<libc::sockaddr_xdp>() as _,
            )
        };

        if ret == 0 {
            Ok(())
        } else {
            Err(XdpSockError::BindError(ioError::last_os_error()))
        }
    }

    /// Attempts to detect which network interface to bind to using some simple heuristics.
    /// This function will return the index of the first network interface that is:
    /// * up
    /// * not a loopback device
    /// * has ARP/an L2 address
    ///
    /// # Return
    /// If a value meeting all the criteria is found then it will be returned, if not then the
    /// value `Ok(None)` will be returned.
    /// If an issue occurred while interacting with the system, a system error will be returned
    fn autodetect_ifindex() -> Result<Option<u32>, ioError> {
        let mut ifaddrs: *mut libc::ifaddrs = std::ptr::null_mut();
        unsafe {
            libc::getifaddrs(&mut ifaddrs as *mut _);
        }

        if ifaddrs.is_null() {
            return Err(ioError::last_os_error());
        }

        let needed_flags = libc::IFF_UP as u32;
        let blocked_flags = (libc::IFF_LOOPBACK & libc::IFF_NOARP) as u32;

        let mut curr = ifaddrs;
        while !curr.is_null() {
            // Safety:
            // We know ifaddrs isn't null, otherwise we can't be in the loop so we can deref
            let ifaddr = unsafe { (*curr) };

            if (ifaddr.ifa_flags & needed_flags) > 0 && (ifaddr.ifa_flags & blocked_flags) == 0 {
                break;
            }

            curr = ifaddr.ifa_next;
        }

        let name = NonNull::new(curr)
            .and_then(|a| {
                // Safety:
                // An ifaddrs struct's ifa_name points to a null terminated interface name so
                // passing it to if_nametoindex is safe to do
                let idx = unsafe { libc::if_nametoindex(a.as_ref().ifa_name) };
                if idx != 0 { Some(idx) } else { None }
            })
            .ok_or_else(ioError::last_os_error)
            .map(|i| Some(i));

        // Safety:
        // We know ifaddrs isn't null so it is safe and necessary to pass it to freeifaddrs to free
        // up the allocated memory
        unsafe { libc::freeifaddrs(ifaddrs) };

        name
    }

    /// A thin wrapper around [libc::setsockopt] that maps the return value to a [Result]
    ///
    /// # Parameters:
    ///
    /// * `op_name`: The name value of the operation to perform, e.g. [libc::XDP_UMEM_REG].
    /// * `value` : Reference to the value that is to be passed.
    ///
    /// # Safety
    /// The caller must ensure that:
    /// * `op_name` is a valid value to pass as the name parameter to [libc::setsockopt]
    /// * `value` is a valid value to give to the operation associated with `op_name`
    unsafe fn setsockopt<T>(fd: &XdpFd, op_name: i32, value: &T) -> Result<(), ioError> {
        let result = unsafe {
            libc::setsockopt(
                fd.as_raw_fd(),
                libc::SOL_XDP,
                op_name,
                value as *const _ as _,
                size_of::<T>() as _,
            )
        };

        if result == 0 {
            Ok(())
        } else {
            Err(ioError::last_os_error())
        }
    }

    /// A thin wrapper around [libc::getsockopt] that maps the return value to a [Result]
    ///
    /// # Parameters:
    ///
    /// * `op_name`: The name value of the operation to perform, e.g. [libc::XDP_UMEM_REG].
    /// * `value` : Reference to a type to be written to.
    ///
    /// # Safety
    ///
    /// When you call this function you have to ensure that `op_name` is a valid operation value and
    /// that `value` is a valid type and value to pass as the operation's output parameter.
    unsafe fn getsockopt<T>(fd: &XdpFd, op_name: i32, value: &mut T) -> Result<(), ioError> {
        let mut optlen: libc::socklen_t = size_of::<T>() as _;

        let result = unsafe {
            libc::getsockopt(
                fd.as_raw_fd(),
                libc::SOL_XDP,
                op_name,
                value as *mut _ as _,
                &mut optlen as *mut _,
            )
        };

        if result == 0 {
            Ok(())
        } else {
            Err(ioError::last_os_error())
        }
    }
}
