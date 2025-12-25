use std::{
    io::Error as ioError,
    ops::{Deref, DerefMut},
    os::fd::{AsRawFd, RawFd},
    ptr::NonNull,
    sync::atomic::AtomicU32,
};

use log::debug;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum UmemError {
    #[error("Failed to allocate umem buffer: {0}")]
    Alloc(ioError),
    #[error("Failed to register umem buffer: {0}")]
    Register(ioError),
}

/// A Umem buffer containing a page aligned buffer that is exactly [Self::BUF_LEN] bytes long
struct Umem(NonNull<[u8; Self::BUF_LEN]>);

impl Umem {
    const CHUNK_SIZE: usize = 4096;
    const CHUNK_COUNT: usize = 4096;
    const BUF_LEN: usize = Self::CHUNK_SIZE * Self::CHUNK_COUNT;

    /// Allocates a page aligned buffer of size [Self::BUF_LEN]
    pub fn new(fd: &XdpFd) -> Result<Umem, UmemError> {
        let addr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                Self::BUF_LEN,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_ANONYMOUS | libc::MAP_PRIVATE,
                -1,
                0,
            )
        };
        let addr = NonNull::new(addr).ok_or_else(|| UmemError::Alloc(ioError::last_os_error()))?;

        let umem = Umem(addr.cast());
        umem.register_buffer(fd).map_err(UmemError::Register)?;
        Ok(umem)
    }

    /// Registers the [Umem] buffer with the specified [XdpSock]
    fn register_buffer(&self, fd: &XdpFd) -> Result<(), ioError> {
        let umem_reg = libc::xdp_umem_reg {
            addr: self.as_ptr() as _,
            len: self.len() as _,
            chunk_size: Umem::CHUNK_SIZE as _,
            headroom: 0,
            flags: 0,
            tx_metadata_len: 0,
        };

        // Safety: XDP_UMEM_REG is a valid operation and umem_reg is a valid parameter to pass it
        unsafe { setsockopt(&fd, libc::XDP_UMEM_REG, &umem_reg) }
    }
}

impl Drop for Umem {
    fn drop(&mut self) {
        // Safety:
        // We acquired this memory through mmap and therefor its safe to pass it to munmap to free
        // it. The allocation always has the size Self::BUF_LEN
        unsafe {
            libc::munmap(self.0.cast().as_ptr(), Self::BUF_LEN);
        }
    }
}

impl Deref for Umem {
    type Target = [u8; Self::BUF_LEN];

    fn deref(&self) -> &Self::Target {
        // Safety:
        // Pointer to reference conversion criteria are met as the buffer is aligned, the pointer,
        // by definition of NonNull, is not null, is dereferncable, points to valid value of T.
        // We also follow rust's aliasing rules as the borrow checker will see this as a reference
        // to the umem struct wrapping it
        unsafe {self.0.as_ref()}
    }
}

impl DerefMut for Umem {
    fn deref_mut(&mut self) -> &mut Self::Target {
        // Safety:
        // Pointer to reference conversion criteria are met as the buffer is aligned, the pointer,
        // by definition of NonNull, is not null, is dereferncable, points to valid value of T.
        // We also follow rust's aliasing rules as the borrow checker will see this as a reference
        // to the umem struct wrapping it
        unsafe { self.0.as_mut() }
    }
}

struct RingBuffer<T, const N: usize, const M: libc::off_t> {
    mem: NonNull<libc::c_void>,
    mem_len: usize,
    producer: NonNull<AtomicU32>,
    consumer: NonNull<AtomicU32>,
    data: NonNull<[T; N]>
}

impl<T, const N: usize, const M: libc::off_t> RingBuffer<T, N, M> {
    /// Allocates a ring buffer given an XDP socket and the offsets associated with it
    ///
    /// # Safety
    /// The caller must ensure that `offsets` is the correct offset struct associated with the
    /// ring buffer type
    unsafe fn new(fd: &XdpFd, offsets: libc::xdp_ring_offset) -> Result<Self, ioError> {
        assert!(
            N > 0 && N.count_ones() == 1,
            "Buffer data length is not a positive power of two"
        );

        let mem_len = offsets.desc as usize + N * size_of::<T>();

        let mem = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                mem_len,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED | libc::MAP_POPULATE,
                fd.as_raw_fd(),
                M,
            )
        };

        let mem = NonNull::new(mem).ok_or_else(|| ioError::last_os_error())?;

        let producer = unsafe { mem.add(offsets.producer as _).cast() };
        let consumer = unsafe { mem.add(offsets.consumer as _).cast() };
        let data = unsafe { mem.add(offsets.desc as _).cast() };

        Ok(Self {
            mem,
            mem_len,
            producer,
            consumer,
            data,
        })
    }
}

impl<T, const N: usize, const M: libc::off_t> Drop for RingBuffer<T, N, M> {
    fn drop(&mut self) {
        // Safety:
        // We acquired this memory through mmap and therefor its safe to pass it to munmap to free
        // it. The allocation size is always stored in mem_len
        unsafe {
            libc::munmap(self.mem.cast().as_ptr(), self.mem_len);
        }
    }
}

#[derive(Debug, Error)]
pub enum RingBufferError {
    #[error("Failed to register the buffer: {0}")]
    Register(ioError),
    #[error("Failed to buffer offsets: {0}")]
    GetOffsets(ioError),
    #[error("Failed to allocate the buffer: {0}")]
    Alloc(ioError),
}

struct RxBuffer(RingBuffer<libc::xdp_desc, {Self::RING_LEN}, {libc::XDP_PGOFF_RX_RING}>);
struct TxBuffer(RingBuffer<libc::xdp_desc, {Self::RING_LEN}, {libc::XDP_PGOFF_TX_RING}>);
struct FillBuffer(RingBuffer<u64, {Self::RING_LEN}, {libc::XDP_UMEM_PGOFF_FILL_RING as _}>);
struct CompletionBuffer(RingBuffer<u64, {Self::RING_LEN},
    {libc::XDP_UMEM_PGOFF_COMPLETION_RING as _}>);

impl RxBuffer {
    const RING_LEN: usize = 512;

    pub fn new(fd: &XdpFd) -> Result<RxBuffer, RingBufferError> {
        Self::register_ring_buffer(fd).map_err(RingBufferError::Register)?;
        let offsets = Self::get_ring_offsets(fd).map_err(RingBufferError::GetOffsets)?;
        // Safety:
        // offsets is the correct offset for the RxBuffer
        unsafe {
            Ok(Self(
                RingBuffer::new(fd, offsets).map_err(RingBufferError::Alloc)?
            ))
        }
    }
 
    /// Attempts to get the offsets for an RX ring buffer
    fn get_ring_offsets(fd: &XdpFd) -> Result<libc::xdp_ring_offset, ioError> {
        let mut offsets: libc::xdp_mmap_offsets = unsafe { std::mem::zeroed() };
        unsafe { getsockopt(fd, libc::XDP_MMAP_OFFSETS, &mut offsets) }?;
        Ok(offsets.rx)
    }

    /// Attempts to allocate a ring buffer of the specified type and size with the kernel
    fn register_ring_buffer(fd: &XdpFd) -> Result<(), ioError> {
        // ring buffer length is always a power of two, invariant ensured by constructor
        unsafe { setsockopt(fd, libc::XDP_RX_RING, &512) }
    }
}

impl TxBuffer {
    const RING_LEN: usize = 512;
    pub fn new(fd: &XdpFd) -> Result<TxBuffer, RingBufferError> {
        Self::register_ring_buffer(fd).map_err(RingBufferError::Register)?;
        let offsets = Self::get_ring_offsets(fd).map_err(RingBufferError::GetOffsets)?;
        // Safety:
        // offsets is the correct offset for the RxBuffer
        unsafe {
            Ok(Self(
                RingBuffer::new(fd, offsets).map_err(RingBufferError::Alloc)?
            ))
        }
    }
 
    /// Attempts to get the offsets for an RX ring buffer
    fn get_ring_offsets(fd: &XdpFd) -> Result<libc::xdp_ring_offset, ioError> {
        let mut offsets: libc::xdp_mmap_offsets = unsafe { std::mem::zeroed() };
        unsafe { getsockopt(fd, libc::XDP_MMAP_OFFSETS, &mut offsets) }?;
        Ok(offsets.tx)
    }

    /// Attempts to allocate a ring buffer of the specified type and size with the kernel
    fn register_ring_buffer(fd: &XdpFd) -> Result<(), ioError> {
        // ring buffer length is always a power of two, invariant ensured by constructor
        unsafe { setsockopt(fd, libc::XDP_TX_RING, &512) }
    }
}

impl FillBuffer {
    const RING_LEN: usize = 512;
    pub fn new(fd: &XdpFd) -> Result<FillBuffer, RingBufferError> {
        Self::register_ring_buffer(fd).map_err(RingBufferError::Register)?;
        let offsets = Self::get_ring_offsets(fd).map_err(RingBufferError::GetOffsets)?;
        // Safety:
        // offsets is the correct offset for the RxBuffer
        unsafe {
            Ok(Self(
                RingBuffer::new(fd, offsets).map_err(RingBufferError::Alloc)?
            ))
        }
    }
 
    /// Attempts to get the offsets for an RX ring buffer
    fn get_ring_offsets(fd: &XdpFd) -> Result<libc::xdp_ring_offset, ioError> {
        let mut offsets: libc::xdp_mmap_offsets = unsafe { std::mem::zeroed() };
        unsafe { getsockopt(fd, libc::XDP_MMAP_OFFSETS, &mut offsets) }?;
        Ok(offsets.fr)
    }

    /// Attempts to allocate a ring buffer of the specified type and size with the kernel
    fn register_ring_buffer(fd: &XdpFd) -> Result<(), ioError> {
        // ring buffer length is always a power of two, invariant ensured by constructor
        unsafe { setsockopt(fd, libc::XDP_UMEM_FILL_RING, &512) }
    }
}

impl CompletionBuffer {
    const RING_LEN: usize = 512;
    pub fn new(fd: &XdpFd) -> Result<CompletionBuffer, RingBufferError> {
        Self::register_ring_buffer(fd).map_err(RingBufferError::Register)?;
        let offsets = Self::get_ring_offsets(fd).map_err(RingBufferError::GetOffsets)?;
        // Safety:
        // offsets is the correct offset for the RxBuffer
        unsafe {
            Ok(Self(
                RingBuffer::new(fd, offsets).map_err(RingBufferError::Alloc)?
            ))
        }
    }
 
    /// Attempts to get the offsets for an RX ring buffer
    fn get_ring_offsets(fd: &XdpFd) -> Result<libc::xdp_ring_offset, ioError> {
        let mut offsets: libc::xdp_mmap_offsets = unsafe { std::mem::zeroed() };
        unsafe { getsockopt(fd, libc::XDP_MMAP_OFFSETS, &mut offsets) }?;
        Ok(offsets.cr)
    }

    /// Attempts to allocate a ring buffer of the specified type and size with the kernel
    fn register_ring_buffer(fd: &XdpFd) -> Result<(), ioError> {
        // ring buffer length is always a power of two, invariant ensured by constructor
        unsafe { setsockopt(fd, libc::XDP_UMEM_COMPLETION_RING, &512) }
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
    #[error("Failed to create umem buffer: {0}")]
    Umem(UmemError),
    #[error("Rx buffer operation failed: {0}")]
    RxError(RingBufferError),
    #[error("Tx buffer operation failed: {0}")]
    TxError(RingBufferError),
    #[error("Fill buffer operation failed: {0}")]
    FillError(RingBufferError),
    #[error("Completion buffer operation failed: {0}")]
    CompletionError(RingBufferError),
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

        let umem = Umem::new(&fd).map_err(XdpSockError::Umem)?;

        let rx = RxBuffer::new(&fd).map_err(XdpSockError::RxError)?;
        let tx = TxBuffer::new(&fd).map_err(XdpSockError::TxError)?;
        let fill = FillBuffer::new(&fd).map_err(XdpSockError::FillError)?;
        let completion = CompletionBuffer::new(&fd).map_err(XdpSockError::CompletionError)?;

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

    fn bind(fd: &XdpFd, ifindex: Option<u32>) -> Result<(), XdpSockError> {
        let ifindex = ifindex
            .map_or_else(Self::autodetect_ifindex, |v| Ok(Some(v)))
            .map_err(XdpSockError::IfiSearchError)?
            .ok_or(XdpSockError::IfiSearchFail)?;

        let sockaddr = libc::sockaddr_xdp {
            sxdp_family: libc::AF_XDP as _,
            sxdp_flags: libc::XDP_ZEROCOPY,
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
        let blocked_flags = (libc::IFF_LOOPBACK | libc::IFF_NOARP) as u32;

        let mut curr = ifaddrs;
        while !curr.is_null() {
            // Safety:
            // We know ifaddrs isn't null, otherwise we can't be in the loop so we can deref
            let ifaddr = unsafe { curr.as_ref().expect("curr was null when iterating NICs") };

            if (ifaddr.ifa_flags & needed_flags) > 0 && (ifaddr.ifa_flags & blocked_flags) == 0 {
                // Safety:
                // An ifaddrs struct's ifa_name points to a null terminated interface name
                unsafe {
                    debug!(
                        "Selected device with name: {}",
                        std::ffi::CStr::from_ptr(ifaddr.ifa_name).to_string_lossy()
                    );
                }
                break;
            }

            curr = ifaddr.ifa_next;
        }

        let index = NonNull::new(curr)
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

        index
    }
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
