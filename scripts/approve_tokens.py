"""One-time token approval script for Polymarket trading.

Sets unlimited USDC allowance for Polymarket's exchange contracts on Polygon.
This replaces the manual "first trade on polymarket.com" step.

Usage:
    python scripts/approve_tokens.py [--dry-run]

Requires: POLYMARKET_PRIVATE_KEY and POLYMARKET_FUNDER_ADDRESS in .env
Requires: A small amount of POL in the wallet for gas (~$0.10 total)
"""

import os
import sys
import argparse
from dotenv import load_dotenv
from web3 import Web3

load_dotenv()

# --- Polygon mainnet config ---
RPC_URLS = [
    "https://polygon.llamarpc.com",
    "https://polygon-bor-rpc.publicnode.com",
    "https://polygon-rpc.com",
]

# Polymarket exchange contracts that need USDC approval
SPENDERS = {
    "CTF Exchange": "0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E",
    "Neg Risk Exchange": "0xC5d563A36AE78145C45a50134d48A1215220f80a",
    "Neg Risk Adapter": "0xd91E80cF2E7be2e162c6513ceD06f1dD0dA35296",
}

# USDC contracts on Polygon — approve both variants
USDC_TOKENS = {
    "USDC.e (bridged)": "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174",
    "USDC (native)": "0x3c499c542cEF5E3811e1192ce70d8cC03d5c3359",
}

# Conditional Tokens Framework — needs setApprovalForAll for outcome tokens
CTF_CONTRACT = "0x4D97DCd97eC945f40cF65F87097ACe5EA0476045"

# ERC20 approve ABI
ERC20_APPROVE_ABI = [
    {
        "inputs": [
            {"name": "spender", "type": "address"},
            {"name": "amount", "type": "uint256"},
        ],
        "name": "approve",
        "outputs": [{"name": "", "type": "bool"}],
        "stateMutability": "nonpayable",
        "type": "function",
    },
    {
        "inputs": [
            {"name": "owner", "type": "address"},
            {"name": "spender", "type": "address"},
        ],
        "name": "allowance",
        "outputs": [{"name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function",
    },
]

# ERC1155 setApprovalForAll ABI (for conditional tokens)
ERC1155_APPROVAL_ABI = [
    {
        "inputs": [
            {"name": "operator", "type": "address"},
            {"name": "approved", "type": "bool"},
        ],
        "name": "setApprovalForAll",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function",
    },
    {
        "inputs": [
            {"name": "account", "type": "address"},
            {"name": "operator", "type": "address"},
        ],
        "name": "isApprovedForAll",
        "outputs": [{"name": "", "type": "bool"}],
        "stateMutability": "view",
        "type": "function",
    },
]

MAX_UINT256 = 2**256 - 1


def main():
    parser = argparse.ArgumentParser(description="Approve Polymarket contracts for trading")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be approved without sending transactions")
    args = parser.parse_args()

    pk = os.getenv("POLYMARKET_PRIVATE_KEY")
    funder = os.getenv("POLYMARKET_FUNDER_ADDRESS")

    if not pk or not funder:
        print("❌ Missing POLYMARKET_PRIVATE_KEY or POLYMARKET_FUNDER_ADDRESS in .env")
        sys.exit(1)

    w3 = None
    for rpc in RPC_URLS:
        try:
            _w3 = Web3(Web3.HTTPProvider(rpc, request_kwargs={"timeout": 15}))
            if _w3.is_connected():
                w3 = _w3
                print(f"RPC: {rpc}")
                break
        except Exception:
            continue
    if w3 is None:
        print("❌ Cannot connect to any Polygon RPC")
        sys.exit(1)

    account = w3.eth.account.from_key(pk)
    address = account.address
    print(f"Wallet: {address}")

    # Check gas balance
    bal = w3.eth.get_balance(address)
    pol = bal / 1e18
    print(f"POL balance: {pol:.6f}")
    if pol < 0.001 and not args.dry_run:
        print("❌ Need at least ~0.001 POL for gas. Fund wallet with a small amount of POL first.")
        sys.exit(1)

    nonce = w3.eth.get_transaction_count(address)
    tx_count = 0

    # Step 1: Approve USDC on each exchange contract
    print("\n--- USDC Approvals ---")
    for token_label, token_addr in USDC_TOKENS.items():
        contract = w3.eth.contract(
            address=Web3.to_checksum_address(token_addr), abi=ERC20_APPROVE_ABI
        )
        for spender_label, spender_addr in SPENDERS.items():
            current = contract.functions.allowance(
                Web3.to_checksum_address(address),
                Web3.to_checksum_address(spender_addr),
            ).call()

            if current >= MAX_UINT256 // 2:
                print(f"  ✅ {token_label} → {spender_label}: already approved")
                continue

            print(f"  🔄 {token_label} → {spender_label}: approving...")
            if args.dry_run:
                print(f"     [DRY RUN] Would approve unlimited {token_label} for {spender_label}")
                continue

            tx = contract.functions.approve(
                Web3.to_checksum_address(spender_addr), MAX_UINT256
            ).build_transaction({
                "from": address,
                "nonce": nonce,
                "gas": 60000,
                "gasPrice": w3.eth.gas_price,
                "chainId": 137,
            })
            signed = account.sign_transaction(tx)
            tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
            receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=60)
            status = "✅" if receipt["status"] == 1 else "❌ FAILED"
            print(f"     {status} tx: {tx_hash.hex()}")
            nonce += 1
            tx_count += 1

    # Step 2: Approve conditional tokens (ERC1155) for exchange contracts
    print("\n--- Conditional Token (ERC1155) Approvals ---")
    ctf = w3.eth.contract(
        address=Web3.to_checksum_address(CTF_CONTRACT), abi=ERC1155_APPROVAL_ABI
    )
    for spender_label, spender_addr in SPENDERS.items():
        is_approved = ctf.functions.isApprovedForAll(
            Web3.to_checksum_address(address),
            Web3.to_checksum_address(spender_addr),
        ).call()

        if is_approved:
            print(f"  ✅ CTF → {spender_label}: already approved")
            continue

        print(f"  🔄 CTF → {spender_label}: approving...")
        if args.dry_run:
            print(f"     [DRY RUN] Would setApprovalForAll for {spender_label}")
            continue

        tx = ctf.functions.setApprovalForAll(
            Web3.to_checksum_address(spender_addr), True
        ).build_transaction({
            "from": address,
            "nonce": nonce,
            "gas": 60000,
            "gasPrice": w3.eth.gas_price,
            "chainId": 137,
        })
        signed = account.sign_transaction(tx)
        tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
        receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=60)
        status = "✅" if receipt["status"] == 1 else "❌ FAILED"
        print(f"     {status} tx: {tx_hash.hex()}")
        nonce += 1
        tx_count += 1

    print(f"\n{'='*40}")
    if args.dry_run:
        print("DRY RUN complete — no transactions sent")
    elif tx_count == 0:
        print("All approvals already set! Ready to trade. 🚀")
    else:
        print(f"Sent {tx_count} approval transaction(s). Ready to trade! 🚀")


if __name__ == "__main__":
    main()
