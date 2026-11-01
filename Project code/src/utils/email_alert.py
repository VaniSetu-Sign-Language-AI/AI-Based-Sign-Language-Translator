"""SMTP Email Alert Handler for VaniSetu Sign Language Translator."""
from __future__ import annotations

import json
import os
import smtplib
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
SMTP_CONFIG_PATH = ROOT / "configs" / "smtp_config.json"

DEFAULT_CONFIG = {
    "smtp_host": "smtp.gmail.com",
    "smtp_port": 587,
    "use_tls": True,
    "sender_email": "gaikwadgaurav250@gmail.com",
    "sender_password": "dcysslvyvpyrxzqb",
    "default_recipient": "doxpro26@gmail.com, gaikwadgaurav685@gmail.com"
}


def load_smtp_config() -> dict:
    if SMTP_CONFIG_PATH.exists():
        try:
            data = json.loads(SMTP_CONFIG_PATH.read_text(encoding="utf-8"))
            merged = DEFAULT_CONFIG.copy()
            merged.update(data)
            return merged
        except Exception:
            return DEFAULT_CONFIG.copy()
    return DEFAULT_CONFIG.copy()


def save_smtp_config(config_data: dict) -> dict:
    current = load_smtp_config()
    for key in ["smtp_host", "smtp_port", "use_tls", "sender_email", "default_recipient"]:
        if key in config_data:
            current[key] = config_data[key]
    # Password: only update if non-empty value passed
    if "sender_password" in config_data and config_data["sender_password"].strip():
        # Strip spaces from App Password if any exist
        current["sender_password"] = config_data["sender_password"].replace(" ", "").strip()
    
    # Cast port
    try:
        current["smtp_port"] = int(current["smtp_port"])
    except (ValueError, TypeError):
        current["smtp_port"] = 587
        
    current["use_tls"] = bool(current["use_tls"])

    SMTP_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    SMTP_CONFIG_PATH.write_text(json.dumps(current, indent=2), encoding="utf-8")
    return current


def send_alert_email(
    text_content: str,
    recipient: str | None = None,
    subject: str | None = None,
    config: dict | None = None
) -> dict:
    cfg = config or load_smtp_config()
    target_str = (recipient or cfg.get("default_recipient", "")).strip()

    if not target_str:
        raise ValueError("No recipient email address defined. Please configure target recipient email(s).")

    # Support multiple recipients (comma or semicolon separated)
    recipient_list = [addr.strip() for addr in target_str.replace(";", ",").split(",") if addr.strip()]

    if not recipient_list:
        raise ValueError("No valid recipient email addresses found.")

    sender = (cfg.get("sender_email") or "").strip()
    password = (cfg.get("sender_password") or "").replace(" ", "").strip()
    host = (cfg.get("smtp_host") or "smtp.gmail.com").strip()
    port = int(cfg.get("smtp_port") or 587)
    use_tls = bool(cfg.get("use_tls", True))

    if not sender:
        raise ValueError("Sender email address is not configured. Please set up SMTP credentials.")

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    email_subject = subject or f"[VaniSetu Alert] Sign Language Text Notification - {now_str}"

    recipients_formatted = ", ".join(recipient_list)

    msg = MIMEMultipart("alternative")
    msg["Subject"] = email_subject
    msg["From"] = f"VaniSetu Alert <{sender}>"
    msg["To"] = recipients_formatted

    text_body = f"""VaniSetu AI Sign Language Alert Notification
=========================================
Time: {now_str}
Recipients: {recipients_formatted}

Translated Signed Text / Alert Message:
-----------------------------------------
{text_content}

-----------------------------------------
Sent automatically via VaniSetu Sign Language Translator Console.
"""

    html_body = f"""
    <!DOCTYPE html>
    <html>
    <head>
      <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #f9f9f9; color: #333; margin: 0; padding: 20px; }}
        .card {{ max-width: 600px; margin: 0 auto; background: #ffffff; border: 3px solid #F94C00; border-radius: 16px; overflow: hidden; box-shadow: 0 4px 15px rgba(249,76,0,0.15); }}
        .header {{ background: #F94C00; color: #FFFAFA; padding: 20px 24px; text-align: center; }}
        .header h1 {{ margin: 0; font-size: 22px; font-weight: 800; letter-spacing: 0.5px; }}
        .content {{ padding: 24px; }}
        .alert-box {{ background: #FFF5F0; border-left: 5px solid #F94C00; padding: 16px 20px; border-radius: 8px; margin: 16px 0; font-size: 18px; font-weight: bold; color: #D63E00; word-break: break-word; }}
        .meta {{ font-size: 13px; color: #666; margin-bottom: 15px; border-bottom: 1px solid #eee; padding-bottom: 10px; }}
        .footer {{ text-align: center; font-size: 12px; color: #888; padding: 15px; background: #fafafa; border-top: 1px solid #eee; }}
      </style>
    </head>
    <body>
      <div class="card">
        <div class="header">
          <h1>🚨 VaniSetu Sign Language Alert</h1>
        </div>
        <div class="content">
          <div class="meta">
            <strong>Time:</strong> {now_str}<br/>
            <strong>Target Recipients:</strong> {recipients_formatted}
          </div>
          <p style="font-weight: 600; color: #444; margin-bottom: 6px;">Signed Message / Alert Text:</p>
          <div class="alert-box">
            "{text_content}"
          </div>
          <p style="font-size: 13px; color: #555;">This email was dispatched automatically from the VaniSetu AI Sign Language Translator console when hand sign text was generated and sent as an alert.</p>
        </div>
        <div class="footer">
          VaniSetu AI Sign Language Translator &bull; Empowering Communication
        </div>
      </div>
    </body>
    </html>
    """

    msg.attach(MIMEText(text_body, "plain", "utf-8"))
    msg.attach(MIMEText(html_body, "html", "utf-8"))

    # Send via smtplib
    if port == 465:
        server = smtplib.SMTP_SSL(host, port, timeout=15)
    else:
        server = smtplib.SMTP(host, port, timeout=15)
    
    try:
        if port != 465 and use_tls:
            server.starttls()
        if sender and password:
            server.login(sender, password)
        server.sendmail(sender, recipient_list, msg.as_string())
    finally:
        try:
            server.quit()
        except Exception:
            pass

    return {
        "status": "success",
        "message": f"Alert email successfully sent to {recipients_formatted}",
        "recipient": recipients_formatted,
        "recipient_count": len(recipient_list),
        "timestamp": now_str
    }
