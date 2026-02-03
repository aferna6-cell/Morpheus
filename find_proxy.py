import httpx

eoa = "0x293Ef129E0F811E8FABb04AD6a777BEcE957B857"

endpoints = [
    ("proxy-wallet", f"https://clob.polymarket.com/auth/proxy-wallet?address={eoa}"),
    ("profile", f"https://clob.polymarket.com/profile?address={eoa}"),
    ("positions", f"https://data-api.polymarket.com/positions?user={eoa}"),
    ("trades", f"https://data-api.polymarket.com/trades?user={eoa}&limit=5"),
    ("rewards", f"https://data-api.polymarket.com/rewards?address={eoa}"),
]

for name, url in endpoints:
    try:
        r = httpx.get(url, timeout=5)
        print(f"{name}: {r.status_code} {r.text[:300]}")
    except Exception as e:
        print(f"{name}: ERROR {e}")
    print()

# Also try to derive proxy via the CTF exchange proxy factory
# Polymarket uses Safe (Gnosis) proxy factory pattern
# Try pip install web3 and compute
try:
    import hashlib
    # CREATE2 address = keccak256(0xff ++ factory ++ salt ++ initCodeHash)[12:]
    # For Polymarket Magic wallets, factory is the proxy factory
    # and salt is typically derived from the signer address
    
    factory = "aB45c5A4B0c941a2F231C04C3f49182e1A254052"
    signer = eoa[2:].lower()
    
    # Salt = signer padded to 32 bytes (left-padded with zeros)
    salt = bytes(12) + bytes.fromhex(signer)
    
    print(f"Computed salt: {salt.hex()}")
    print(f"Signer: {signer}")
    print(f"Factory: {factory}")
except Exception as e:
    print(f"Compute error: {e}")
