use std::sync::{atomic::AtomicBool, Arc};

use log::{debug, error};

use crate::tap::TapDevice;

mod tap;

fn main() {
    env_logger::init();

    let exit_flag = Arc::new(AtomicBool::new(false));
    let exit_clone = exit_flag.clone();

    ctrlc::set_handler(move || exit_clone.store(true, std::sync::atomic::Ordering::Relaxed))
        .expect("Error setting Ctrl-C handler");

    let mut tap = match TapDevice::new(c"teecee%d") {
        Ok(t) => t,
        Err(_) => {
            error!("Failed to create tap device");
            return;
        },
    };

    debug!("Created new tap device: {:?}", tap);

    while !exit_flag.load(std::sync::atomic::Ordering::Relaxed) {
        if let Some(frame) = tap.read_frame() {
            debug!("Got new frame:\n{}", frame);
        }
    }

    debug!("Stopping...");
}
