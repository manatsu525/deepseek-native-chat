#!/usr/bin/env bash
# 维护本地 HTTPS 证书：一个长期有效的本地 CA（设备只需信任一次）+ 由它签发的短期服务器证书。
# 服务器证书满足 Chrome / Safari / iOS 的要求（CA:FALSE、serverAuth、SAN、有效期 ≤ 398 天），
# 缺失、即将过期、IP/域名变化或仍是旧版自签证书时自动重签；CA 不变，所以设备上的信任不会失效。
# 用法：tls.sh [--restart-if-renewed]
set -euo pipefail

DATA_DIR=${DATA_DIR:-/opt/deepseek-native-chat/data}
TLS_DIR=$DATA_DIR/tls
CA_KEY=$TLS_DIR/ca.key
CA_CRT=$TLS_DIR/ca.crt
SERVER_KEY=$TLS_DIR/server.key
SERVER_CRT=$TLS_DIR/server.crt
SERVER_DAYS=397
RENEW_BEFORE=$((30 * 86400))

# 使用外部证书（例如 Let's Encrypt）时不做任何事。
if [[ -n ${TLS_CERT_FILE:-} && $TLS_CERT_FILE != "$SERVER_CRT" ]]; then
  exit 0
fi

mkdir -p "$TLS_DIR"
umask 077

if [[ ! -s $CA_KEY || ! -s $CA_CRT ]]; then
  openssl req -x509 -newkey rsa:2048 -nodes -sha256 -days 3650 \
    -keyout "$CA_KEY" -out "$CA_CRT" \
    -subj "/CN=Custom Native Chat Local CA $(date +%Y%m%d%H%M%S)" \
    -addext "basicConstraints=critical,CA:TRUE,pathlen:0" \
    -addext "keyUsage=critical,keyCertSign,cRLSign" >/dev/null 2>&1
  chmod 644 "$CA_CRT"
fi

san="DNS:localhost,IP:127.0.0.1"
[[ -n ${PUBLIC_IP:-} && $PUBLIC_IP != 127.0.0.1 ]] && san+=",IP:$PUBLIC_IP"
[[ -n ${TLS_DOMAIN:-} ]] && san+=",DNS:$TLS_DOMAIN"

needs_issue() {
  [[ -s $SERVER_KEY && -s $SERVER_CRT ]] || return 0
  openssl verify -CAfile "$CA_CRT" "$SERVER_CRT" >/dev/null 2>&1 || return 0
  openssl x509 -in "$SERVER_CRT" -noout -checkend "$RENEW_BEFORE" >/dev/null 2>&1 || return 0
  local current
  current=$(openssl x509 -in "$SERVER_CRT" -noout -ext subjectAltName 2>/dev/null | tail -n +2 | tr -d ' ')
  local entry
  for entry in ${san//,/ }; do
    entry=${entry/IP:/IPAddress:}
    [[ ",$current," == *",$entry,"* ]] || return 0
  done
  return 1
}

if ! needs_issue; then
  exit 0
fi

work=$(mktemp -d)
trap 'rm -rf -- "$work"' EXIT
cat > "$work/ext.cnf" <<EOF
basicConstraints=critical,CA:FALSE
keyUsage=critical,digitalSignature,keyEncipherment
extendedKeyUsage=serverAuth
subjectAltName=$san
subjectKeyIdentifier=hash
authorityKeyIdentifier=keyid
EOF
openssl req -new -newkey rsa:2048 -nodes -sha256 \
  -keyout "$work/server.key" -out "$work/server.csr" \
  -subj "/CN=${TLS_DOMAIN:-${PUBLIC_IP:-localhost}}" >/dev/null 2>&1
openssl x509 -req -sha256 -days "$SERVER_DAYS" \
  -in "$work/server.csr" -CA "$CA_CRT" -CAkey "$CA_KEY" -CAcreateserial \
  -CAserial "$TLS_DIR/ca.srl" -extfile "$work/ext.cnf" \
  -out "$work/server.crt" >/dev/null 2>&1
mv -f "$work/server.key" "$SERVER_KEY"
mv -f "$work/server.crt" "$SERVER_CRT"
chmod 600 "$SERVER_KEY"
chmod 644 "$SERVER_CRT"
echo "已签发新的服务器证书（$san）"

if [[ ${1:-} == --restart-if-renewed ]]; then
  systemctl try-restart deepseek-native-chat.service
fi
