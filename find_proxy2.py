"""
Find Polymarket proxy wallet address for a Magic/embedded wallet EOA.

Polymarket uses a Gnosis Safe-style proxy factory pattern.
The proxy address is derived via CREATE2:
  address = keccak256(0xff ++ factory ++ salt ++ keccak256(bytecode))[12:]

For Magic wallets (signature_type=1), the proxy is created by
the Polymarket proxy factory contract.
"""

from web3 import Web3
import httpx

w3 = Web3()
eoa = "0x293Ef129E0F811E8FABb04AD6a777BEcE957B857"

# Known Polymarket infrastructure on Polygon
PROXY_FACTORY = "0xaB45c5A4B0c941a2F231C04C3f49182e1A254052"
EXCHANGE = "0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E"
NEG_RISK_EXCHANGE = "0xC5d563A36AE78145C45a50134d48A1215220f80a"
NEG_RISK_ADAPTER = "0xd91E80cF2E7be2e162c6513ceD06f1dD0dA35296"

# Polymarket USDC on Polygon
USDC = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"

# Try to find the proxy by querying the Polygon RPC for the proxy factory
# The factory has a getProxy(address) view function
POLYGON_RPC = "https://polygon-rpc.com"

# Method 1: Try the proxy factory's getProxy function
# Function signature: getProxy(address) -> address
# selector = keccak256("getProxy(address)")[:4]
selector = w3.keccak(text="getProxy(address)")[:4].hex()
print(f"getProxy selector: {selector}")

# Encode the call
padded_eoa = eoa[2:].lower().zfill(64)
calldata = f"0x{selector}{padded_eoa}"
print(f"calldata: {calldata}")

# eth_call to the proxy factory
payload = {
    "jsonrpc": "2.0",
    "method": "eth_call",
    "params": [{"to": PROXY_FACTORY, "data": calldata}, "latest"],
    "id": 1,
}
r = httpx.post(POLYGON_RPC, json=payload)
result = r.json()
print(f"getProxy result: {result}")

if result.get("result") and result["result"] != "0x":
    proxy_hex = result["result"]
    # Extract address (last 20 bytes of 32-byte return)
    proxy_addr = "0x" + proxy_hex[-40:]
    proxy_addr_cs = w3.to_checksum_address(proxy_addr)
    print(f"\n*** PROXY ADDRESS: {proxy_addr_cs} ***")
    
    # Check USDC balance of proxy
    # balanceOf(address) selector
    bal_selector = w3.keccak(text="balanceOf(address)")[:4].hex()
    bal_calldata = f"0x{bal_selector}{proxy_addr[2:].lower().zfill(64)}"
    bal_payload = {
        "jsonrpc": "2.0",
        "method": "eth_call",
        "params": [{"to": USDC, "data": bal_calldata}, "latest"],
        "id": 2,
    }
    r2 = httpx.post(POLYGON_RPC, json=bal_payload)
    bal_result = r2.json()
    if bal_result.get("result"):
        balance = int(bal_result["result"], 16)
        print(f"USDC balance: {balance / 1e6:.2f}")
    
    # Also check via data-api
    r3 = httpx.get(f"https://data-api.polymarket.com/value?user={proxy_addr_cs}")
    print(f"data-api value: {r3.text[:200]}")
    
    r4 = httpx.get(f"https://data-api.polymarket.com/positions?user={proxy_addr_cs}")
    print(f"positions: {r4.text[:300]}")
else:
    print("getProxy returned empty - trying alternative methods...")
    
    # Method 2: Try different factory addresses
    alt_factories = [
        "0x2bB290A413f2490A6EFB8e7a97B9caa1a51e612a",  # another known factory
        "0x3d9F3dC4F7E3d9D0F4Df6b77c16B31c0b1C5d9E",
    ]
    
    for factory in alt_factories:
        payload["params"][0]["to"] = factory
        try:
            r = httpx.post(POLYGON_RPC, json=payload)
            result = r.json()
            if result.get("result") and result["result"] != "0x" and len(result["result"]) > 10:
                proxy_hex = result["result"]
                proxy_addr = "0x" + proxy_hex[-40:]
                print(f"Factory {factory}: proxy = {proxy_addr}")
        except:
            pass
