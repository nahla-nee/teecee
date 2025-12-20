use log::{error, info};

// use crate::{netlink::NetlinkCon, tap::TapDevice, xdp_sock::XdpSock};

mod xdp;
// mod netlink;
// mod tap;

use xdp::XdpSock;

#[tokio::main]
async fn main() {
    env_logger::init();

    match XdpSock::new() {
        Ok(_) => info!("IT WORKS!!!!"),
        Err(e) => {
            error!("Failed to create socket: {e}");
            return;
        }
    };
}
