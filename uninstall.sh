#!/usr/bin/env bash
set -euo pipefail

if [[ ${EUID} -ne 0 ]]; then
  echo "请使用 root 运行卸载脚本。" >&2
  exit 1
fi
if [[ ${1:-} != "--yes" ]]; then
  read -r -p "将删除程序、账号、API 配置和全部聊天记录，继续？[y/N] " answer
  [[ $answer == y || $answer == Y ]] || exit 0
fi

systemctl disable --now deepseek-native-chat.service 2>/dev/null || true
rm -f /etc/systemd/system/deepseek-native-chat.service
systemctl daemon-reload
rm -rf -- /opt/deepseek-native-chat
echo "DeepSeek Native Chat 已完全卸载，数据不可恢复。"
