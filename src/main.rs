use opencl3::Result;
use opencl3::command_queue::{CL_QUEUE_PROFILING_ENABLE, CommandQueue};
use opencl3::context::Context;
use opencl3::device::{CL_DEVICE_TYPE_GPU, Device, get_all_devices};
use opencl3::kernel::{ExecuteKernel, Kernel};
use opencl3::memory::{Buffer, CL_MEM_READ_ONLY, CL_MEM_WRITE_ONLY};
use opencl3::program::Program;
use opencl3::types::*;

use rand::prelude::*;
use ed25519_dalek::SigningKey;
use ed25519_dalek::SECRET_KEY_LENGTH;
use base64::prelude::*;
use sha2::{Sha256, Digest};
use regex::Regex;

use rayon::prelude::*;

use std::ptr;
use std::io::Write;

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
const BATCH_SIZE: usize = 4096;
const ARRAY_SIZE: usize = SECRET_KEY_LENGTH * BATCH_SIZE;

fn main() -> Result<()> {
    // Find a usable device for this application
    let device_id = *get_all_devices(CL_DEVICE_TYPE_GPU)?
        .first()
        .expect("no device found in platform");
    let device = Device::new(device_id);

    // Create a Context on an OpenCL device
    let context = Context::from_device(&device).expect("Context::from_device failed");

    // Create a command_queue on the Context's device
    let queue = CommandQueue::create_default(&context, CL_QUEUE_PROFILING_ENABLE)
        .expect("CommandQueue::create_default failed");

    // Build the OpenCL program source and create the kernel.
    let program = Program::create_and_build_from_source(&context, PROGRAM_SOURCE, "")
        .expect("Program::create_and_build_from_source failed");
    let kernel = Kernel::create(&program, KERNEL_NAME).expect("Kernel::create failed");


    let mut rng = rand::rng();
    let prng_constants: [cl_ulong; 4] = [rng.random::<cl_ulong>(), rng.random::<cl_ulong>(), rng.random::<cl_ulong>(), rng.random::<cl_ulong>()];

    // Create OpenCL device buffers
    let out = unsafe {
        Buffer::<cl_uchar>::create(&context, CL_MEM_WRITE_ONLY, ARRAY_SIZE, ptr::null_mut())?
    };
    let mut prngs = unsafe {
        Buffer::<cl_ulong>::create(&context, CL_MEM_READ_ONLY, 4, ptr::null_mut())?
    };

    let _prngs_write_event = unsafe { queue.enqueue_write_buffer(&mut prngs, CL_BLOCKING, 0, &prng_constants, &[])? };


    let mut start_index = 0u64;
    let mut tries = 0;
    let re = Regex::new(r"(?i)shenj").unwrap();
    let now = std::time::Instant::now();
    loop {
        let kernel_event = unsafe {
            ExecuteKernel::new(&kernel)
                .set_arg(&out)
                .set_arg(&start_index)
                .set_arg(&prngs)
                .set_global_work_size(ARRAY_SIZE)
                .enqueue_nd_range(&queue)?
        };

        let mut events: Vec<cl_event> = Vec::default();
        events.push(kernel_event.get());

        // Create a results array to hold the results from the OpenCL device
        // and enqueue a read command to read the device buffer into the array
        // after the kernel event completes.
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

    println!("ms: {}", now.elapsed().as_millis());
    println!("tries: {}", tries * BATCH_SIZE);

    Ok(())
}

fn process_key(sk: &SigningKey, re: Regex) -> bool {
    let (pk, fp) = to_ssh_ed25519_public_key(sk.verifying_key().as_bytes());
    if re.is_match(&fp) {
        println!("{}", pk);
        println!("{}", fp);
        println!("{:?}", sk);
        true
    } else {
        false
    }
}

fn to_ssh_ed25519_public_key(public_key: &[u8; 32]) -> (String, String) {
    const KEY_TYPE: &[u8] = b"ssh-ed25519";
    
    // Calculate total length: key_type (4 + 11) + public_key (4 + 32)
    let mut buf = Vec::with_capacity(4 + KEY_TYPE.len() + 4 + public_key.len());
    
    // Write key type string and its length
    buf.write_all(&(KEY_TYPE.len() as u32).to_be_bytes()).unwrap();
    buf.write_all(KEY_TYPE).unwrap();
    
    // Write public key and its length
    buf.write_all(&(public_key.len() as u32).to_be_bytes()).unwrap();
    buf.write_all(public_key).unwrap();

    let hash = Sha256::digest(&buf);

    let fingerprint = format!("SHA256:{}", BASE64_STANDARD.encode(hash));
    
    // Base64 encode and format
    let encoded = BASE64_STANDARD.encode(&buf);
    let public_key = format!("ssh-ed25519 {}", encoded);

    (public_key, fingerprint)
}
