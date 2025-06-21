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

// PRNG by DeepSeek
const PROGRAM_SOURCE: &str = r#"
kernel void generate_seeds(
    global uchar* seeds,
    ulong start_index,
    global ulong* prng_constants
) {
    size_t gid = get_global_id(0);
    ulong state = (gid + start_index) * prng_constants[0] + prng_constants[1];
    ulong a = prng_constants[2];
    ulong c = prng_constants[3];

    __global uchar* seed = &seeds[gid * 32];

    // PRNG
    for (int i = 0; i < 8; i++) {
        state = state * a + c;
        // Split 64-bit state into 8 bytes
        seed[i*4 + 0] = (state >> 0) & 0xFF;
        seed[i*4 + 1] = (state >> 8) & 0xFF;
        seed[i*4 + 2] = (state >> 16) & 0xFF;
        seed[i*4 + 3] = (state >> 24) & 0xFF;
        seed[i*4 + 4] = (state >> 32) & 0xFF;
        seed[i*4 + 5] = (state >> 40) & 0xFF;
        seed[i*4 + 6] = (state >> 48) & 0xFF;
        seed[i*4 + 7] = (state >> 56) & 0xFF;
    }
}
"#;

const KERNEL_NAME: &str = "generate_seeds";
const BATCH_SIZE: usize = 65536;
const ARRAY_SIZE: usize = SECRET_KEY_LENGTH * BATCH_SIZE;

#[repr(usize)]
#[derive(Clone, Copy, PartialEq, Eq)]
enum Mode {
    PubKey = 0,
    FingerPrint = 1,
}
static mut MODE_IDX: Mode = Mode::FingerPrint;
static RUN: AtomicBool = AtomicBool::new(true);

extern "C" fn handle_sigint(_signal: nix::libc::c_int) {
    RUN.store(false, Ordering::Relaxed);
}

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
    let program = Program::create_and_build_from_source(&context, PROGRAM_SOURCE, "")
        .expect("Program::create_and_build_from_source failed");
    let kernel = Kernel::create(&program, KERNEL_NAME).expect("Kernel::create failed");

    Ok((context, queue, kernel))
}

fn parse_args() -> Option<Regex> {
    let args: Vec<String> = std::env::args().collect();

    match args.len().cmp(&1) {
        cmp::Ordering::Equal => {
            Some(Regex::new(r"test").unwrap())
        }
        cmp::Ordering::Greater => {
            if args.len() == 3 {
                match args[2].as_str() {
                    "-k" => unsafe {MODE_IDX = Mode::PubKey},
                    _ => todo!("I don't know what you mean"),
                }
            }
            Some(Regex::new(args[1].as_str()).unwrap())
        }
        cmp::Ordering::Less => {
            None
        }
    }
}

fn main() -> Result<()> {
    let re = parse_args().unwrap();
    let handler = SigHandler::Handler(handle_sigint);
    unsafe { signal::signal(Signal::SIGINT, handler) }.unwrap();
    let (context, queue, kernel) = init_cl()?;
    println!("opencl initialized");

    // feed data
    let mut rng = rand::rng();
    let prng_constants: [cl_ulong; 4] = [rng.random::<cl_ulong>(), rng.random::<cl_ulong>(), rng.random::<cl_ulong>(), rng.random::<cl_ulong>()];

    let out = unsafe {
        Buffer::<cl_uchar>::create(&context, CL_MEM_WRITE_ONLY, ARRAY_SIZE, ptr::null_mut())?
    };
    let mut prngs = unsafe {
        Buffer::<cl_ulong>::create(&context, CL_MEM_READ_ONLY, 4, ptr::null_mut())?
    };
    let _prngs_write_event = unsafe { queue.enqueue_write_buffer(&mut prngs, CL_BLOCKING, 0, &prng_constants, &[])? };

    let mut start_index = 0u64;
    let mut tries = 0;
    let now = std::time::Instant::now();
    while RUN.load(Ordering::Relaxed) {
        let kernel_event = unsafe {
            ExecuteKernel::new(&kernel)
                .set_arg(&out)
                .set_arg(&start_index)
                .set_arg(&prngs)
                .set_global_work_size(BATCH_SIZE)
                .enqueue_nd_range(&queue)?
        };

        let mut events: Vec<cl_event> = Vec::default();
        events.push(kernel_event.get());

        // Create a results array to hold the results from the OpenCL device
        // and enqueue a read command to read the device buffer into the array
        let mut results: [cl_uchar; ARRAY_SIZE] = [0; ARRAY_SIZE];
        let read_event =
            unsafe { queue.enqueue_read_buffer(&out, CL_NON_BLOCKING, 0, &mut results, &events)? };

        // Wait for the read_event to complete.
        read_event.wait()?;

        let found = results.par_chunks_exact(SECRET_KEY_LENGTH)
            .map(|chunk| SigningKey::from_bytes(chunk.try_into().unwrap()))
            .find_any(|sk| process_key(sk, re.clone()));

        if found != None { break; }

        start_index += ARRAY_SIZE as u64;
        tries += 1
    }

    let elapsed = match now.elapsed().as_secs() {
        0 => 1,
        n => n,
    };
    println!("secs: {}", elapsed);
    println!("tries: {}", tries * BATCH_SIZE);
    println!("Key/s: {}", tries * BATCH_SIZE / elapsed as usize);

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
