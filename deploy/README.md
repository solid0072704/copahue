# Despliegue en un VPS (Linux con systemd)

```bash
sudo useradd -r -m -s /bin/bash trader
sudo git clone https://github.com/solid0072704/copahue /opt/copahue
sudo chown -R trader:trader /opt/copahue
sudo -u trader bash -c 'cd /opt/copahue && python3 -m venv .venv && .venv/bin/pip install -r requirements.txt'
sudo -u trader cp /opt/copahue/.env.example /opt/copahue/.env   # completa las claves
sudo chmod 600 /opt/copahue/.env
sudo cp /opt/copahue/deploy/systemd/*.service /opt/copahue/deploy/systemd/*.timer /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now copahue-trader.service copahue-watchdog.timer copahue-review.timer
```

Comprobaciones:

```bash
systemctl status copahue-trader
journalctl -u copahue-trader -f
cat /opt/copahue/data/trader_heartbeat          # se actualiza en cada vela
ls /opt/copahue/docs/trading/reviews/           # una revisión por noche
```

Parada de emergencia (cualquiera de las tres):

```bash
sudo -u trader touch /opt/copahue/data/KILL_SWITCH                 # el bucle cierra en la siguiente vela
sudo -u trader /opt/copahue/.venv/bin/python run_trader.py --kill "motivo"
sudo -u trader /opt/copahue/.venv/bin/python run_trader.py --flatten  # cierra en el exchange ahora mismo (modo vivo)
```

Para rearmar: borra `data/KILL_SWITCH` y reinicia `copahue-trader.service`. Nunca es automático.
