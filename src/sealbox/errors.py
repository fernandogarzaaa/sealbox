"""Exception hierarchy."""


class SealboxError(Exception):
    """Base class for all sealbox errors."""


class PolicyError(SealboxError):
    """Invalid policy file."""


class VaultError(SealboxError):
    """Vault layout / dataset errors."""


class QueryRejected(SealboxError):
    """A query was rejected before execution (unsafe shape, forbidden column use...).

    ``rule`` is a stable machine-readable id; the message never contains data values.
    """

    def __init__(self, rule: str, detail: str) -> None:
        super().__init__(detail)
        self.rule = rule
        self.detail = detail
