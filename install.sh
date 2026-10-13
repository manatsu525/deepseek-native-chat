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
TLS_SERVICE_FILE=/etc/systemd/system/deepseek-native-chat-tls.service
TLS_TIMER_FILE=/etc/systemd/system/deepseek-native-chat-tls.timer
ADMIN_USER=${1:-admin}
ADMIN_PASS=${2:-admin123456}
PORT=${PORT:-8000}

env_value() { [[ -f $INSTALL_DIR/.env ]] && sed -n "s/^$1=//p" "$INSTALL_DIR/.env" | head -n1 || true; }

if [[ -f $INSTALL_DIR/.env && -s $DATA_DIR/chat.db ]]; then
  existing_user=$(env_value ADMIN_USERNAME)
  existing_pass=$(env_value ADMIN_PASSWORD)
  ADMIN_USER=${existing_user:-$ADMIN_USER}
  ADMIN_PASS=${existing_pass:-$ADMIN_PASS}
fi
# 命令行环境变量优先，其次沿用上次安装的设置，避免重装把手动配置的证书、域名或 IP 覆盖掉。
PUBLIC_IP=${PUBLIC_IP:-$(env_value PUBLIC_IP)}
PUBLIC_IP=${PUBLIC_IP:-$(hostname -I 2>/dev/null | awk '{print $1}')}
TLS_DOMAIN=${TLS_DOMAIN:-$(env_value TLS_DOMAIN)}
TLS_CERT_FILE=${TLS_CERT_FILE:-$(env_value TLS_CERT_FILE)}
TLS_KEY_FILE=${TLS_KEY_FILE:-$(env_value TLS_KEY_FILE)}
TLS_CERT_FILE=${TLS_CERT_FILE:-$DATA_DIR/tls/server.crt}
TLS_KEY_FILE=${TLS_KEY_FILE:-$DATA_DIR/tls/server.key}
if [[ $TLS_CERT_FILE != "$DATA_DIR/tls/server.crt" && ( ! -s $TLS_CERT_FILE || ! -s $TLS_KEY_FILE ) ]]; then
  echo "找不到证书文件：$TLS_CERT_FILE / $TLS_KEY_FILE" >&2
  exit 1
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
  cp -a "$SOURCE_DIR/app" "$SOURCE_DIR/static" "$SOURCE_DIR/skills" "$SOURCE_DIR/requirements.txt" "$SOURCE_DIR/run.sh" "$SOURCE_DIR/tls.sh" "$SOURCE_DIR/.env.example" "$INSTALL_DIR/"
fi
chmod +x "$INSTALL_DIR/run.sh" "$INSTALL_DIR/tls.sh"

python3 -m venv "$INSTALL_DIR/.venv"
"$INSTALL_DIR/.venv/bin/pip" install --disable-pip-version-check --no-cache-dir -q -r "$INSTALL_DIR/requirements.txt"

DATA_DIR=$DATA_DIR PUBLIC_IP=$PUBLIC_IP TLS_DOMAIN=$TLS_DOMAIN TLS_CERT_FILE=$TLS_CERT_FILE "$INSTALL_DIR/tls.sh"

cat > "$INSTALL_DIR/.env" <<EOF
HOST=0.0.0.0
PORT=$PORT
DATA_DIR=$DATA_DIR
SESSION_DAYS=60
REQUEST_TIMEOUT=1200
PUBLIC_IP=$PUBLIC_IP
TLS_DOMAIN=$TLS_DOMAIN
TLS_CERT_FILE=$TLS_CERT_FILE
TLS_KEY_FILE=$TLS_KEY_FILE
ADMIN_USERNAME=$ADMIN_USER
ADMIN_PASSWORD=$ADMIN_PASS
EOF
chmod 600 "$INSTALL_DIR/.env"

cat > "$SERVICE_FILE" <<EOF
[Unit]
Description=Custom Native Chat
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=$INSTALL_DIR
EnvironmentFile=$INSTALL_DIR/.env
ExecStartPre=$INSTALL_DIR/tls.sh
ExecStart=$INSTALL_DIR/run.sh
Restart=on-failure
RestartSec=3
# Agent 模式需要直接管理主机、项目文件和服务；不要在 systemd 层套沙箱。
NoNewPrivileges=false
PrivateTmp=false
ProtectSystem=off
ProtectHome=false

[Install]
WantedBy=multi-user.target
EOF

cat > "$TLS_SERVICE_FILE" <<EOF
[Unit]
Description=Renew Custom Native Chat local TLS certificate

[Service]
Type=oneshot
EnvironmentFile=$INSTALL_DIR/.env
ExecStart=$INSTALL_DIR/tls.sh --restart-if-renewed
EOF

cat > "$TLS_TIMER_FILE" <<EOF
[Unit]
Description=Daily check of Custom Native Chat local TLS certificate

[Timer]
OnCalendar=daily
RandomizedDelaySec=1h
Persistent=true

[Install]
WantedBy=timers.target
EOF

systemctl daemon-reload
systemctl enable deepseek-native-chat.service
systemctl enable --now deepseek-native-chat-tls.timer >/dev/null 2>&1 || echo "警告：证书续期定时器未启用，服务重启时仍会检查续期。" >&2
systemctl restart deepseek-native-chat.service
sleep 2
systemctl is-active --quiet deepseek-native-chat.service

echo
echo "安装完成"
echo "地址: https://${TLS_DOMAIN:-${PUBLIC_IP:-服务器IP}}:$PORT"
echo "管理员: $ADMIN_USER"
echo "初始密码: $ADMIN_PASS（登录后请在账号管理中修改）"
if [[ $TLS_CERT_FILE == "$DATA_DIR/tls/server.crt" ]]; then
  echo "证书由本机 CA 签发。每台设备信任一次 CA 后就不会再提示不安全（见 README“HTTPS 证书”）："
  echo "  下载: https://${TLS_DOMAIN:-${PUBLIC_IP:-服务器IP}}:$PORT/ca.crt"
  echo "  或复制服务器上的 $DATA_DIR/tls/ca.crt"
fi
