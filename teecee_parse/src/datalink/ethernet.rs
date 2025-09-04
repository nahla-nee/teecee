use std::fmt::{Debug, Display};

use zerocopy::{big_endian, FromBytes, Immutable, IntoBytes, KnownLayout};

use crate::datalink::arp::ArpHeader;

#[repr(transparent)]
#[derive(FromBytes, IntoBytes, KnownLayout, Immutable)]
pub struct MacAddr([u8; 6]);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Ethertype(big_endian::U16);

// impl Ethertype {
//     pub const IPV4: Self = Self(0x0800);
//     pub const ARP: Self = Self(0x0806);
//     pub const ETHERNET: Self = Self(0x6558);
//     pub const VLAN: Self = Self(0x8100);
//     pub const IPV6: Self = Self(0x86dd);
//     pub const LLDP: Self = Self(0x88cc);
//     pub const QINQ: Self = Self(0x9100);
// }

#[repr(C)]
#[derive(FromBytes, KnownLayout, Immutable, Debug)]
pub struct EthHeader {
    dst_mac: MacAddr,
    src_mac: MacAddr,
    ether_type: big_endian::U16,
    // arp: ArpHeader,
}

impl MacAddr {
    pub fn new(addr: [u8; 6]) -> Self {
        MacAddr(addr)
    }
}

impl Debug for MacAddr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Display::fmt(&self, f)
    }
}

impl Display for MacAddr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f,
            "{:02x}:{:02x}:{:02x}:{:02x}:{:02x}:{:02x}",
            self.0[0], self.0[1], self.0[2], self.0[3], self.0[4], self.0[5]
        )
    }
}

impl Display for EthHeader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{{\n\
                \tdst_mac: {},\n\
                \tsrc_mac: {},\n\
                \tethertype: {:#08X},\n\
            }}",
            self.dst_mac, self.src_mac, self.ether_type.get()
        )
    }
}
