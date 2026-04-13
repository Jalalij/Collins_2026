#!/bin/bash
# Usage:
#   sed -i 's/\r$//' sat_tunnel_setup_v2.sh if it shows "bad interpreter"
#   chmod +x sat_tunnel_setup.sh
#   ./sat_tunnel_setup.sh start|stop|status
set -euo pipefail

# ====== CONFIG ======
WG_IF="wg-sat"
WG_CONF="/etc/wireguard/${WG_IF}.conf"

VM_PUB_IP="34.48.133.98"        # Google VM public IP (endpoint)
VM_WG_IP="10.77.0.1"            # Google VM WireGuard tunnel IP

SAT_DEV="eno12429"              # UT NIC to OneWeb
SAT_GW_IP="192.168.42.1"        # OneWeb gateway

# Policy routing toggle (0 = route-only; 1 = use fwmark+table)
USE_FWMARK=0
MARK_ID_HEX="0x777"
TABLE_ID="777"
TABLE_NAME="wg-sat"

# From MTU test: MTU=1408 → MSS=1368
TUNNEL_MSS="1368"
# =====================

has_table()         { grep -qE "^${TABLE_ID}\s" /etc/iproute2/rt_tables; }
has_rule()          { ip rule show | grep -qE "fwmark ${MARK_ID_HEX}.*(lookup ${TABLE_NAME}|lookup ${TABLE_ID})"; }
has_table_default() {
  ip route show table ${TABLE_NAME} | grep -q "^default dev ${WG_IF}" || \
  ip route show table ${TABLE_ID}   | grep -q "^default dev ${WG_IF}"
}
has_route_vm_pub()  { ip route show to ${VM_PUB_IP}/32 | grep -q "${SAT_GW_IP} dev ${SAT_DEV}"; }
has_inner_route()   { ip route show ${VM_WG_IP}/32 | grep -q "dev ${WG_IF}"; }

ipt_output_has_jump() { sudo iptables -t mangle -S OUTPUT 2>/dev/null | grep -q " -j WG_SAT_MARK"; }
ipt_chain_exists()    { sudo iptables -t mangle -S WG_SAT_MARK >/dev/null 2>&1; }
ipt_has_mss_out()     { sudo iptables -t mangle -S OUTPUT  2>/dev/null | grep -q -- "-o ${WG_IF} -p tcp .*TCPMSS --set-mss ${TUNNEL_MSS}"; }
ipt_has_mss_fwd()     { sudo iptables -t mangle -S FORWARD 2>/dev/null | grep -q -- "-o ${WG_IF} -p tcp .*TCPMSS --set-mss ${TUNNEL_MSS}"; }

ensure_no_fwmark_dupes() {
  # Remove ALL OUTPUT jumps to WG_SAT_MARK
  while ipt_output_has_jump; do
    sudo iptables -t mangle -D OUTPUT -j WG_SAT_MARK || true
  done
  # If the chain exists, flush and delete it entirely
  if ipt_chain_exists; then
    sudo iptables -t mangle -F WG_SAT_MARK || true
    sudo iptables -t mangle -X WG_SAT_MARK || true
  fi
  # Remove ALL fwmark rules (match by mark only; table name/id may vary)
  while ip rule show | grep -q "fwmark ${MARK_ID_HEX}"; do
    sudo ip rule del fwmark ${MARK_ID_HEX} || true
  done
}

start() {
  echo "[+] Forcing VM endpoint via satellite ${SAT_DEV}/${SAT_GW_IP}"
  if ! has_route_vm_pub; then
    sudo ip route add ${VM_PUB_IP}/32 via ${SAT_GW_IP} dev ${SAT_DEV}
  fi
  ip route get ${VM_PUB_IP} || true

  echo "[+] Bring up WireGuard (${WG_IF})"
  sudo wg-quick up ${WG_IF} || true
  ip link show ${WG_IF} >/dev/null 2>&1 || { echo "[-] ${WG_IF} not present"; exit 1; }

  echo "[+] Ensure inner /32 route for ${VM_WG_IP} via ${WG_IF}"
  if ! has_inner_route; then
    sudo ip route add ${VM_WG_IP}/32 dev ${WG_IF} || true
  fi

  if [[ "${USE_FWMARK}" -eq 1 ]]; then
    echo "[+] FWMark mode ON: installing table/rule and single precise mark"
    # Clean ALL prior fwmark rules & chain first
    ensure_no_fwmark_dupes
    # Table mapping and default via wg-sat
    if ! has_table; then
      echo "${TABLE_ID} ${TABLE_NAME}" | sudo tee -a /etc/iproute2/rt_tables >/dev/null
    fi
    if ! has_table_default; then
      sudo ip route add default dev ${WG_IF} table ${TABLE_NAME}
    fi
    # Single ip rule
    sudo ip rule add fwmark ${MARK_ID_HEX} table ${TABLE_NAME}
    # Create chain and single OUTPUT jump + mark
    sudo iptables -t mangle -N WG_SAT_MARK 2>/dev/null || true
    sudo iptables -t mangle -A OUTPUT -j WG_SAT_MARK
    sudo iptables -t mangle -A WG_SAT_MARK -d ${VM_WG_IP} -j MARK --set-mark ${MARK_ID_HEX}
  else
    echo "[+] FWMark mode OFF: using plain routes only"
    ensure_no_fwmark_dupes
  fi

  echo "[+] Clamp MSS inside tunnel to ${TUNNEL_MSS}"
  if ! ipt_has_mss_out; then
    sudo iptables -t mangle -A OUTPUT  -o ${WG_IF} -p tcp --tcp-flags SYN,RST SYN -j TCPMSS --set-mss ${TUNNEL_MSS}
  fi
  if ! ipt_has_mss_fwd; then
    sudo iptables -t mangle -A FORWARD -o ${WG_IF} -p tcp --tcp-flags SYN,RST SYN -j TCPMSS --set-mss ${TUNNEL_MSS}
  fi

  echo "[✓] Started. Endpoint via sat, inner /32 in place, MSS=${TUNNEL_MSS}, fwmark=${USE_FWMARK}"
}

stop() {
  echo "[+] Stop WireGuard (${WG_IF})"
  sudo wg-quick down ${WG_IF} || true

  echo "[+] Remove endpoint host route (optional)"
  has_route_vm_pub && sudo ip route del ${VM_PUB_IP}/32 via ${SAT_GW_IP} dev ${SAT_DEV} || true

  echo "[+] Remove inner /32 route (optional)"
  has_inner_route && sudo ip route del ${VM_WG_IP}/32 dev ${WG_IF} || true

  echo "[+] Remove MSS clamp rules"
  while ipt_has_mss_out; do
    sudo iptables -t mangle -D OUTPUT  -o ${WG_IF} -p tcp --tcp-flags SYN,RST SYN -j TCPMSS --set-mss ${TUNNEL_MSS} || true
  done
  while ipt_has_mss_fwd; do
    sudo iptables -t mangle -D FORWARD -o ${WG_IF} -p tcp --tcp-flags SYN,RST SYN -j TCPMSS --set-mss ${TUNNEL_MSS} || true
  done

  echo "[+] Purging ALL fwmark rules/jumps/rules (if any)"
  ensure_no_fwmark_dupes

  echo "[✓] Stopped and cleaned"
}

status() {
  echo "=== Routes ==="
  ip route show to ${VM_PUB_IP}/32 || true
  ip route show ${VM_WG_IP}/32 || true
  ip route show table ${TABLE_NAME} || true
  echo "=== Rules ==="
  ip rule show | grep -E "lookup ${TABLE_NAME}|lookup ${TABLE_ID}|fwmark" || true
  echo "=== iptables mangle/OUTPUT ==="
  sudo iptables -t mangle -S OUTPUT 2>/dev/null | sed 's/^/- /'
  echo "=== iptables mangle/WG_SAT_MARK ==="
  sudo iptables -t mangle -S WG_SAT_MARK 2>/dev/null | sed 's/^/- /' || true
  echo "=== WireGuard ==="
  sudo wg show || true
}

case "${1:-}" in
  start)  start ;;
  stop)   stop ;;
  status) status ;;
  *) echo "Usage: $0 {start|stop|status}"; exit 1 ;;
esac
