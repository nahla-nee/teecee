use std::fmt::Display;

use zerocopy::{big_endian, FromBytes, Immutable, KnownLayout};

#[repr(C)]
#[derive(FromBytes, KnownLayout, Immutable, Debug)]
pub struct ArpHeader {
    hw_type: big_endian::U16,
    protocol: big_endian::U16,
    hw_size: u8,
    pro_size: u8,
    op_code: big_endian::U16,
}

impl Display for ArpHeader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{{\n
                hw_type: {:#08X},\n
                protocol: {:#08X},\n
                hw_size: {},\n
                pro_size: {},\n
                op_code: {:#08}\n
            }}",
            self.hw_type, self.protocol, self.hw_size, self.pro_size, self.op_code.get()
        )
    }
}
