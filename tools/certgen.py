"""工具层 - 本地 CA + HTTPS 证书

 手机浏览器的两道门槛：
 1. 麦克风（getUserMedia）只在安全上下文开放——局域网 IP 必须 HTTPS；
 2. Chrome 对自签名证书的页面不注册 Service Worker，PWA 装不上、地址栏去不掉。

 解法：本地生成一个根 CA + 用它签发服务器证书。手机把 CA 证书装进信任存储
 （一次性两分钟操作），Chrome 就完全信任这套 HTTPS——无警告、SW 正常、PWA 可装。
 证书落在 storage/certs/，十年有效，存在即复用。
"""

import ipaddress
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

from config.settings import get_settings


def _cert_dir() -> Path:
    d = get_settings().data_dir / "certs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _local_ips() -> list:
    """抓本机所有网卡的 IPv4，全塞进 SAN，手机换网络也能连。"""
    import socket

    ips = {"127.0.0.1"}
    try:
        # 连一个外部地址（不真发包）拿到默认路由对应的本机 IP
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("223.5.5.5", 80))
            ips.add(s.getsockname()[0])
    except Exception:
        pass
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ips.add(info[4][0])
    except Exception:
        pass
    # 可选：环境变量 HTTPS_PUBLIC_IPS 补充公网 IP / 域名，塞进 SAN
    # 这样手机上用 https://公网IP:8443 访问也无需"继续访问/不安全"提示
    for token in os.environ.get("HTTPS_PUBLIC_IPS", "").replace("，", ",").split(","):
        token = token.strip()
        if token:
            ips.add(token)
    return sorted(ips)


def _cryptography():
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID, ExtendedKeyUsageOID

    return x509, hashes, serialization, rsa, NameOID, ExtendedKeyUsageOID


def _write_pem(path: Path, data: bytes):
    path.write_bytes(data)


def ca_cert_path() -> Path:
    """给手机安装用的根 CA 证书路径。"""
    return _cert_dir() / "rin-rootCA.crt"


def ensure_cert() -> tuple:
    """确保 (本地CA + 服务器证书) 存在，返回 (server_cert, server_key) 路径。

    CA 是给手机/电脑装进信任存储用的；服务器证书由 CA 签发，链路完整可信。
    任一环生成失败返回 (None, None)，调用方降级成纯 HTTP。
    """
    cert_dir = _cert_dir()
    ca_key_path = cert_dir / "rootCA.key"
    ca_path = ca_cert_path()
    key_path = cert_dir / "server.key"
    cert_path = cert_dir / "server.pem"

    try:
        x509, hashes, serialization, rsa, NameOID, EKU = _cryptography()
    except ImportError:
        print("[certgen] 缺 cryptography 包，HTTPS 不可用（pip install cryptography）")
        return None, None

    now = datetime.now(timezone.utc)

    # ---- 本地根 CA：只当签发者用，手机装的就是这张 ----
    if not (ca_path.exists() and ca_key_path.exists()):
        ca_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        ca_name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Aria Local Root CA")])
        ca_cert = (
            x509.CertificateBuilder()
            .subject_name(ca_name)
            .issuer_name(ca_name)  # 根 CA 自己给自己背书
            .public_key(ca_key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now - timedelta(days=1))
            .not_valid_after(now + timedelta(days=3650))
            .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
            .sign(ca_key, hashes.SHA256())
        )
        _write_pem(ca_key_path, ca_key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.TraditionalOpenSSL,
            serialization.NoEncryption(),
        ))
        _write_pem(ca_path, ca_cert.public_bytes(serialization.Encoding.PEM))
        print(f"[certgen] 本地根 CA 已生成: {ca_path}（手机装这张，https 从此无警告）")

    # ---- 服务器证书：由本地 CA 签发，SAN 覆盖 localhost + 所有网卡 IP + 公网 IP ----
    if not (cert_path.exists() and key_path.exists()):
        ca_key = serialization.load_pem_private_key(ca_key_path.read_bytes(), password=None)
        ca_cert = x509.load_pem_x509_certificate(ca_path.read_bytes())
        key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Aria Local Service")])
        san = [x509.DNSName("localhost")]
        san += [x509.IPAddress(ipaddress.ip_address(ip)) for ip in _local_ips()]
        # 环境变量 HTTPS_PUBLIC_DOMAINS 补域名 SAN（多个用逗号分隔，如 example.com）
        for domain in os.environ.get("HTTPS_PUBLIC_DOMAINS", "").replace("，", ",").split(","):
            domain = domain.strip()
            if domain:
                san.append(x509.DNSName(domain))
        cert = (
            x509.CertificateBuilder()
            .subject_name(name)
            .issuer_name(ca_cert.subject)  # 签发者 = 本地 CA，链路可信
            .public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now - timedelta(days=1))
            .not_valid_after(now + timedelta(days=3650))
            .add_extension(x509.SubjectAlternativeName(san), critical=False)
            .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
            .add_extension(x509.ExtendedKeyUsage([EKU.SERVER_AUTH]), critical=False)
            .sign(ca_key, hashes.SHA256())
        )
        _write_pem(key_path, key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.TraditionalOpenSSL,
            serialization.NoEncryption(),
        ))
        _write_pem(cert_path, cert.public_bytes(serialization.Encoding.PEM))
        print(f"[certgen] 服务器证书已生成（CA 签发）: {cert_path}")

    return str(cert_path), str(key_path)