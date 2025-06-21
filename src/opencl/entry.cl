__kernel void generate_ed25519_key(
    __global uchar *private_keys,   // Output: 32-byte private keys
    __global uchar *public_keys,    // Output: 32-byte public keys
    __global __const uchar *key_root // Input: 32-byte random seed
) {
    size_t thread = get_global_id(0);
    uchar key[32];

    // 1. Copy seed and add thread ID to first bytes
    for (size_t i = 0; i < 32; i++) {
        key[i] = key_root[i];
    }
    *((size_t *)key) += thread; // Thread-specific seed modification

    // https://docs.rs/ed25519-dalek/2.1.1/src/ed25519_dalek/signing.rs.html#102-108
    u32 in[32] = { 0 };
    uchar hash[64];

    sha512_ctx_t keystate;
    sha512_init(&keystate);
    to_32bytes_sha2_input(in, key);
    sha512_update(&keystate, in, 32);
    sha512_final(&keystate);
    from_sha512_result(hash, keystate.h);

    hash[0] &= 248; // Clear bits 0-2
    hash[31] &= 127; // Clear bit 255
    hash[31] |= 0x40; // Set bit 254

    // 3b. Scalar multiplication
    bignum256modm a;
    ge25519 ALIGN(16) A;
    expand256_modm(a, hash, 32); // Clamped hash -> scalar
    ge25519_scalarmult_base_niels(&A, a); // A = a * base_point

    // 3c. Pack public key
    uchar pubkey[32];
    ge25519_pack(pubkey, &A);

    // 4. Write results to global memory
    for (int i = 0; i < 32; i++) {
        private_keys[thread * 32 + i] = key[i]; // Private key
        public_keys[thread * 32 + i] = pubkey[i]; // Public key
    }
}
