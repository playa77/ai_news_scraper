"""Self-signed TLS certificate generator for the web server.

Generates a self-signed certificate on first startup if none exists.
Stored under a configurable path with restrictive permissions.
"""

from __future__ import annotations

import datetime
import logging
import os
import socket
from pathlib import Path
from typing import Optional, Tuple

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID

logger = logging.getLogger(__name__)

_CERT_FILENAME = "server.crt"
_KEY_FILENAME = "server.key"


class CertError(Exception):
    """Raised when certificate operations fail."""


def ensure_certificates(cert_dir: str) -> Tuple[str, str]:
    """Ensure TLS certificate and key exist, generating them if needed.

    Args:
        cert_dir: Directory to store certificate and key files.

    Returns:
        Tuple of ``(cert_path, key_path)``.

    Raises:
        CertError: If generation fails.
    """
    cert_dir_path = Path(cert_dir)
    cert_dir_path.mkdir(parents=True, exist_ok=True)

    cert_path = cert_dir_path / _CERT_FILENAME
    key_path = cert_dir_path / _KEY_FILENAME

    if cert_path.exists() and key_path.exists():
        logger.info("TLS certificate found at %s", cert_path)
        return str(cert_path), str(key_path)

    logger.info("No TLS certificate found — generating self-signed certificate")
    _generate_self_signed(str(cert_path), str(key_path))

    # Set restrictive permissions
    os.chmod(str(key_path), 0o600)
    os.chmod(str(cert_path), 0o644)

    return str(cert_path), str(key_path)


def _generate_self_signed(cert_path: str, key_path: str) -> None:
    """Generate a self-signed TLS certificate and write to disk.

    The certificate's Common Name is set to ``localhost`` and the
    server's hostname as Subject Alternative Names.
    """
    # Generate private key
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
    )

    # Write private key
    with open(key_path, "wb") as f:
        f.write(
            private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.TraditionalOpenSSL,
                encryption_algorithm=serialization.NoEncryption(),
            )
        )

    # Determine hostname for SAN
    hostname = socket.gethostname()

    # Build certificate
    subject = issuer = x509.Name(
        [
            x509.NameAttribute(NameOID.COMMON_NAME, "localhost"),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "AI News Pipeline"),
        ]
    )

    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(private_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.datetime.now(datetime.timezone.utc))
        .not_valid_after(
            datetime.datetime.now(datetime.timezone.utc)
            + datetime.timedelta(days=365 * 10)
        )
        .add_extension(
            x509.SubjectAlternativeName(
                [
                    x509.DNSName("localhost"),
                    x509.DNSName(hostname),
                ]
            ),
            critical=False,
        )
        .sign(private_key, hashes.SHA256())
    )

    # Write certificate
    with open(cert_path, "wb") as f:
        f.write(cert.public_bytes(serialization.Encoding.PEM))

    logger.info(
        "Self-signed certificate generated: cert=%s key=%s hostname=%s",
        cert_path, key_path, hostname,
    )
