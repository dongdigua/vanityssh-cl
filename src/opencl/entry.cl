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

    // 2. Derive private key from seed (BLAKE2b hash)
    blake2b_state keystate;
    blake2b_init(&keystate, 32); // Output 32-byte hash
    blake2b_update(&keystate, key, 32);
    uint32_t idx = 0;
    blake2b_update(&keystate, (uchar *)&idx, 4); // Append 0
    blake2b_final(&keystate, key, 32); // 'key' is now private key

    // 3. Compute public key from private key
    // 3a. Hash private key and clamp bits
    uchar hash[64];
    blake2b_state state;
    blake2b_init(&state, 64);
    blake2b_update(&state, key, 32);
    blake2b_final(&state, hash, 64);
    hash[0] &= 0xF8; // Clear bits 0-2
    hash[31] &= 0x7F; // Clear bit 255
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
