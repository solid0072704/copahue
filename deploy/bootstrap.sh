#!/usr/bin/env bash
# Instala el agente de trading en un VPS Ubuntu/Debian recién creado.
# Uso (como root, en el VPS):   curl -fsSL https://raw.githubusercontent.com/solid0072704/copahue/claude/new-session-bb9omp/deploy/bootstrap.sh | bash
#         o bien:               bash bootstrap.sh
# Es idempotente: se puede volver a ejecutar para actualizar.
set -euo pipefail

REPO="${REPO:-https://github.com/solid0072704/copahue}"
BRANCH="${BRANCH:-claude/new-session-bb9omp}"
DIR="${DIR:-/opt/copahue}"
USER_NAME="${USER_NAME:-trader}"

if [ "$(id -u)" -ne 0 ]; then
  echo "Ejecuta este script como root (sudo bash bootstrap.sh)" >&2
  exit 1
fi

echo "== 1/6 paquetes del sistema"
export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get install -y --no-install-recommends git python3 python3-venv python3-pip curl ca-certificates

echo "== 2/6 usuario de servicio: $USER_NAME"
id -u "$USER_NAME" >/dev/null 2>&1 || useradd -r -m -s /bin/bash "$USER_NAME"

echo "== 3/6 código en $DIR (rama $BRANCH)"
if [ -d "$DIR/.git" ]; then
  git -C "$DIR" fetch origin "$BRANCH"
  git -C "$DIR" checkout -q "$BRANCH"
  git -C "$DIR" pull -q origin "$BRANCH"
else
  git clone --branch "$BRANCH" "$REPO" "$DIR"
fi
chown -R "$USER_NAME:$USER_NAME" "$DIR"

echo "== 4/6 memoria de intercambio (si el VPS tiene poca RAM)"
if [ "$(free -m | awk '/Mem:/{print $2}')" -lt 1500 ] && [ ! -f /swapfile ]; then
  fallocate -l 1G /swapfile && chmod 600 /swapfile && mkswap /swapfile >/dev/null && swapon /swapfile
  grep -q '/swapfile' /etc/fstab || echo '/swapfile none swap sw 0 0' >> /etc/fstab
fi

echo "== 5/6 entorno de Python y dependencias"
sudo -u "$USER_NAME" bash -c "cd '$DIR' && python3 -m venv .venv && .venv/bin/pip install -q --upgrade pip && .venv/bin/pip install -q -r requirements.txt pytest"
if [ ! -f "$DIR/.env" ]; then
  sudo -u "$USER_NAME" cp "$DIR/.env.example" "$DIR/.env"
  chmod 600 "$DIR/.env"
fi
sudo -u "$USER_NAME" bash -c "cd '$DIR' && .venv/bin/python -m pytest -q" | tail -1

echo "== 6/6 servicios systemd (instalados, no activados hasta que completes .env)"
cp "$DIR"/deploy/systemd/*.service "$DIR"/deploy/systemd/*.timer /etc/systemd/system/
sed -i "s#/opt/copahue#$DIR#g; s#^User=trader#User=$USER_NAME#" /etc/systemd/system/copahue-*.service
systemctl daemon-reload

PUBLIC_IP="$(curl -fsS https://api.ipify.org 2>/dev/null || curl -fsS https://ifconfig.me 2>/dev/null || echo desconocida)"
cat <<MSG

Listo.
  IP pública de este servidor (para la lista blanca de la API de Binance): $PUBLIC_IP
  Archivo de claves: $DIR/.env   (edítalo: nano $DIR/.env)
  Comprobar exchange:  sudo -u $USER_NAME $DIR/.venv/bin/python $DIR/run_trader.py --check-exchange
  Comprobar Jev:       sudo -u $USER_NAME $DIR/.venv/bin/python $DIR/run_trader.py --check-jev
  Arrancar en papel:   systemctl enable --now copahue-trader.service copahue-watchdog.timer copahue-review.timer
  Ver el registro:     journalctl -u copahue-trader -f
MSG
