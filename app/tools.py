"""Mocked back-office 'tools' the agent can call.

In a real engagement these would be API calls to policy admin / claims systems.
Here they are deterministic dicts so the demo runs offline and the eval is stable.
The point for the interview is *tool-use orchestration*, not the data source.
"""
from app.observability import log_event

# --- Fake systems of record ---
_ACCOUNTS = {
    "AC-1001": {"name": "Priya Shah", "policies": ["auto-Premium", "home-Complete"], "balance_due": 0.0},
    "AC-1002": {"name": "John Doe", "policies": ["auto-Standard"], "balance_due": 42.50},
}

_CLAIMS = {
    "CLM-5001": {"policy": "AC-1001", "type": "auto-glass", "status": "Approved", "filed": "2026-05-28"},
    "CLM-5002": {"policy": "AC-1002", "type": "auto-collision", "status": "Assessor Assigned", "filed": "2026-06-02"},
}


def get_policy(account_id: str) -> str:
    acct = _ACCOUNTS.get(account_id.strip().upper())
    log_event("tool_call", tool="get_policy", account_id=account_id, found=bool(acct))
    if not acct:
        return f"No account found for {account_id}."
    return (f"Account {account_id} ({acct['name']}): policies = {', '.join(acct['policies'])}; "
            f"balance due = ${acct['balance_due']:.2f}.")


def get_claim_status(claim_id: str) -> str:
    claim = _CLAIMS.get(claim_id.strip().upper())
    log_event("tool_call", tool="get_claim_status", claim_id=claim_id, found=bool(claim))
    if not claim:
        return f"No claim found for {claim_id}."
    return (f"Claim {claim_id} (type: {claim['type']}, filed {claim['filed']}) "
            f"current status: {claim['status']}.")


# Registry the router selects from.
TOOLS = {
    "get_policy": get_policy,
    "get_claim_status": get_claim_status,
}
