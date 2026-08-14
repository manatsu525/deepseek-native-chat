#!/usr/bin/env bash
set -euo pipefail

if [[ ${EUID} -ne 0 ]]; then
  echo "请使用 root 运行安装脚本。" >&2
  exit 1
fi

SOURCE_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
INSTALL_DIR=/opt/deepseek-native-chat
DATA_DIR=$INSTALL_DIR/data
SERVICE_FILE=/etc/systemd/system/deepseek-native-chat.service
ADMIN_USER=${1:-admin}
ADMIN_PASS=${2:-admin123456}
PUBLIC_IP=${PUBLIC_IP:-$(hostname -I 2>/dev/null | awk '{print $1}')}
PORT=${PORT:-8000}

if [[ -f $INSTALL_DIR/.env && -s $DATA_DIR/chat.db ]]; then
  existing_user=$(sed -n 's/^ADMIN_USERNAME=//p' "$INSTALL_DIR/.env" | head -n1)
  existing_pass=$(sed -n 's/^ADMIN_PASSWORD=//p' "$INSTALL_DIR/.env" | head -n1)
  ADMIN_USER=${existing_user:-$ADMIN_USER}
  ADMIN_PASS=${existing_pass:-$ADMIN_PASS}
fi
if [[ ${#ADMIN_PASS} -lt 8 || $ADMIN_PASS == *$'\n'* ]]; then
  echo "管理员密码必须至少 8 位且不能包含换行。" >&2
  exit 1
fi

apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq python3 python3-venv openssl ca-certificates poppler-utils util-linux >/dev/null

mkdir -p "$INSTALL_DIR" "$DATA_DIR/tls"
if [[ $SOURCE_DIR != "$INSTALL_DIR" ]]; then
  find "$INSTALL_DIR" -mindepth 1 -maxdepth 1 ! -name data -exec rm -rf -- {} +
  cp -a "$SOURCE_DIR/app" "$SOURCE_DIR/static" "$SOURCE_DIR/requirements.txt" "$SOURCE_DIR/run.sh" "$SOURCE_DIR/.env.example" "$INSTALL_DIR/"
fi
chmod +x "$INSTALL_DIR/run.sh"

python3 -m venv "$INSTALL_DIR/.venv"
"$INSTALL_DIR/.venv/bin/pip" install --disable-pip-version-check --no-cache-dir -q -r "$INSTALL_DIR/requirements.txt"

if [[ ! -s $DATA_DIR/tls/server.crt || ! -s $DATA_DIR/tls/server.key ]]; then
  openssl req -x509 -newkey rsa:2048 -nodes -days 3650 \
    -keyout "$DATA_DIR/tls/server.key" -out "$DATA_DIR/tls/server.crt" \
    -subj "/CN=${PUBLIC_IP:-localhost}" \
    ${PUBLIC_IP:+-addext "subjectAltName=IP:$PUBLIC_IP"} >/dev/null 2>&1
  chmod 600 "$DATA_DIR/tls/server.key"
fi

cat > "$INSTALL_DIR/.env" <<EOF
HOST=0.0.0.0
PORT=$PORT
DATA_DIR=$DATA_DIR
SESSION_DAYS=60
REQUEST_TIMEOUT=1200
TLS_CERT_FILE=$DATA_DIR/tls/server.crt
TLS_KEY_FILE=$DATA_DIR/tls/server.key
ADMIN_USERNAME=$ADMIN_USER
ADMIN_PASSWORD=$ADMIN_PASS
EOF
chmod 600 "$INSTALL_DIR/.env"

cat > "$SERVICE_FILE" <<EOF
[Unit]
Description=DeepSeek Native Chat
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=$INSTALL_DIR
EnvironmentFile=$INSTALL_DIR/.env
ExecStart=$INSTALL_DIR/run.sh
Restart=on-failure
RestartSec=3
MemoryMax=180M
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=full
ReadWritePaths=$DATA_DIR

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable --now deepseek-native-chat.service
sleep 2
systemctl is-active --quiet deepseek-native-chat.service

echo
echo "安装完成"
echo "地址: https://${PUBLIC_IP:-服务器IP}:$PORT"
echo "管理员: $ADMIN_USER"
echo "初始密码: $ADMIN_PASS（登录后请在账号管理中修改）"
echo "浏览器会提示自签证书不受信任，手动继续访问即可。"
