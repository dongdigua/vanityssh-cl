use opencl3::Result;
use opencl3::command_queue::{CL_QUEUE_PROFILING_ENABLE, CommandQueue};
use opencl3::context::Context;
use opencl3::device::{CL_DEVICE_TYPE_GPU, Device, get_all_devices};
use opencl3::kernel::{ExecuteKernel, Kernel};
use opencl3::memory::{Buffer, CL_MEM_READ_ONLY, CL_MEM_WRITE_ONLY};
use opencl3::program::Program;
use opencl3::types::*;
use rayon::prelude::*;

use rand::prelude::*;
use ed25519_dalek::SigningKey;
use ed25519_dalek::SECRET_KEY_LENGTH;
use regex::Regex;

use ssh_key::PrivateKey;
use ssh_key::private::Ed25519Keypair;
use ssh_key::public::PublicKey;
use ssh_key::public::Ed25519PublicKey;

use nix::sys::signal::{self, Signal, SigHandler};

use std::ptr;
use std::cmp;
use std::sync::atomic::{AtomicBool, Ordering};

const KERNEL_NAME: &str = "generate_ed25519_key";
const BATCH_SIZE: usize = 2;
const ARRAY_SIZE: usize = SECRET_KEY_LENGTH * BATCH_SIZE;

#[repr(usize)]
#[derive(Clone, Copy, PartialEq, Eq)]
enum Mode {
    PubKey = 0,
    FingerPrint = 1,
}
static mut MODE_IDX: Mode = Mode::FingerPrint;
static RUN: AtomicBool = AtomicBool::new(true);

// https://github.com/kenba/opencl3/blob/main/examples/basic.rs
fn init_cl() -> Result<(Context, CommandQueue, Kernel)> {
    // Find a usable device for this application
    let device_id = *get_all_devices(CL_DEVICE_TYPE_GPU)?
        .first()
        .expect("no device found in platform");
    let device = Device::new(device_id);

    println!("CL_DEVICE_VENDOR: {}", device.vendor()?);
    println!("CL_DEVICE_NAME: {}", device.name()?);

    // Create a Context on an OpenCL device
    let context = Context::from_device(&device).expect("Context::from_device failed");

    // Create a command_queue on the Context's device
    let queue = CommandQueue::create_default(&context, CL_QUEUE_PROFILING_ENABLE)
        .expect("CommandQueue::create_default failed");

    // Build the OpenCL program source and create the kernel.
    let sources = [
        // include_str!("opencl/blake2b.cl"),
        // include_str!("opencl/curve25519-constants.cl"),
        // include_str!("opencl/curve25519-constants2.cl"),
        // include_str!("opencl/curve25519.cl"),
        // include_str!("opencl/entry.cl")
        include_str!("opencl/merged")
    ];

    let program = Program::create_and_build_from_sources(&context, &sources, "")
        .expect("Program::create_and_build_from_source failed");

    println!("aaa");

    let kernel = Kernel::create(&program, KERNEL_NAME).expect("Kernel::create failed");


    Ok((context, queue, kernel))
}

fn main() -> Result<()> {
    let (context, queue, kernel) = init_cl()?;
    println!("opencl initialized");

    // feed data
    // let mut rng = rand::rng();
    // let prng_constants: [cl_ulong; 4] = [rng.random::<cl_ulong>(), rng.random::<cl_ulong>(), rng.random::<cl_ulong>(), rng.random::<cl_ulong>()];
    let prng_constants = [1u8; ARRAY_SIZE];

    let out0 = unsafe {
        Buffer::<cl_uchar>::create(&context, CL_MEM_READ_ONLY, ARRAY_SIZE, ptr::null_mut())?
    };
    let out1 = unsafe {
        Buffer::<cl_uchar>::create(&context, CL_MEM_READ_ONLY, ARRAY_SIZE, ptr::null_mut())?
    };
    let mut prngs = unsafe {
        Buffer::<cl_uchar>::create(&context, CL_MEM_WRITE_ONLY, ARRAY_SIZE, ptr::null_mut())?
    };
    let _prngs_write_event = unsafe { queue.enqueue_write_buffer(&mut prngs, CL_BLOCKING, 0, &prng_constants, &[])? };

    // let mut start_index = 0u64;
    // let mut tries = 0;
    let now = std::time::Instant::now();
    for _ in 1..2 {
        let kernel_event = unsafe {
            ExecuteKernel::new(&kernel)
                .set_arg(&out0)
                .set_arg(&out1)
                .set_arg(&prngs)
                .set_global_work_size(BATCH_SIZE)
                .enqueue_nd_range(&queue)?
        };

        let mut events: Vec<cl_event> = Vec::default();
        events.push(kernel_event.get());

        // Create a results array to hold the results from the OpenCL device
        // and enqueue a read command to read the device buffer into the array
        let mut results0: [cl_uchar; ARRAY_SIZE] = [0; ARRAY_SIZE];
        let read_event0 =
            unsafe { queue.enqueue_read_buffer(&out0, CL_NON_BLOCKING, 0, &mut results0, &events)? };

        let mut results1: [cl_uchar; ARRAY_SIZE] = [0; ARRAY_SIZE];
        let read_event1 =
            unsafe { queue.enqueue_read_buffer(&out1, CL_NON_BLOCKING, 0, &mut results1, &events)? };

        // Wait for the read_event to complete.
        read_event0.wait()?;
        read_event1.wait()?;

        println!("{:?}", results0);
        println!("{:?}", results1);

        // let found = results.par_chunks_exact(SECRET_KEY_LENGTH)
        //     .map(|chunk| SigningKey::from_bytes(chunk.try_into().unwrap()))
        //     .find_any(|sk| process_key(sk, re.clone()));

        // if found != None { break; }

        // start_index += ARRAY_SIZE as u64;
        // tries += 1
    }

    let elapsed = match now.elapsed().as_secs() {
        0 => 1,
        n => n,
    };
    // println!("secs: {}", elapsed);
    // println!("tries: {}", tries * BATCH_SIZE);
    // println!("Key/s: {}", tries * BATCH_SIZE / elapsed as usize);

    Ok(())
}

fn process_key(sk: &SigningKey, re: Regex) -> bool {
    let v = to_ssh_ed25519_public_key(sk.verifying_key().as_bytes());
    let to_match = unsafe {&v[MODE_IDX as usize]}.as_ref().unwrap();
    if re.is_match(&to_match) {
        println!("{}", &to_match);
        if unsafe{MODE_IDX == Mode::FingerPrint} {
            println!("{}", &v[Mode::PubKey as usize].as_ref().unwrap());
        }
        println!("{}", to_ssh_ed25519_private_key(sk));
        true
    } else {
        false
    }
}

fn to_ssh_ed25519_public_key(public_key: &[u8; 32]) -> [Option<String>; 2] {
    let pk = PublicKey::from(Ed25519PublicKey(*public_key));
    [Some(pk.to_openssh().unwrap()),
     if unsafe{MODE_IDX == Mode::FingerPrint} {
         Some(pk.fingerprint(Default::default()).to_string())
     } else {None}]
}

fn to_ssh_ed25519_private_key(sk: &SigningKey) -> String {
    let private = sk;
    let public = sk.verifying_key();

    let ed_kp = Ed25519Keypair { public: public.into(), private: private.into() };
    let pkey = PrivateKey::from(ed_kp);

    pkey.to_openssh(ssh_key::LineEnding::default()).unwrap().to_string()
}
