"""Alert channels (§6.2) — multi-channel outbound alert delivery.

Priority order:
  Channel 1 — Voice Call   (L4 Critical)
  Channel 2 — Push / In-App (L3 High-Risk)
  Channel 3 — SMS           (L3–L4 fallback)
  Channel 4 — Email         (all L2+ events, always sent)

The reference implementation uses in-process callbacks to model each channel
so that the system can be exercised without external telephony credentials.
A real deployment would swap these callbacks for Twilio, FCM, SES, etc.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable

from apex.core.types import AlertPayload, ThresholdLevel


@dataclass
class DeliveryReceipt:
    """Records the outcome of an alert delivery attempt."""

    channel: str
    alert_id: str
    delivered: bool
    attempted_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    error: str | None = None


AlertHandler = Callable[[AlertPayload], bool]


class _Channel:
    def __init__(self, name: str, handler: AlertHandler | None) -> None:
        self.name = name
        self._handler = handler

    def send(self, payload: AlertPayload) -> DeliveryReceipt:
        if self._handler is None:
            return DeliveryReceipt(
                channel=self.name, alert_id=payload.alert_id, delivered=False,
                error="No handler configured."
            )
        try:
            delivered = self._handler(payload)
            return DeliveryReceipt(
                channel=self.name, alert_id=payload.alert_id, delivered=delivered
            )
        except Exception as exc:
            return DeliveryReceipt(
                channel=self.name, alert_id=payload.alert_id, delivered=False,
                error=str(exc)
            )


class AlertChannels:
    """Registry of alert channels.

    Parameters
    ----------
    voice_handler:
        Callable for voice-call delivery (Channel 1).  Signature:
        ``(AlertPayload) -> bool`` where the return value indicates success.
    push_handler:
        Callable for push/in-app delivery (Channel 2).
    sms_handler:
        Callable for SMS delivery (Channel 3).
    email_handler:
        Callable for email delivery (Channel 4).
    """

    def __init__(
        self,
        voice_handler: AlertHandler | None = None,
        push_handler: AlertHandler | None = None,
        sms_handler: AlertHandler | None = None,
        email_handler: AlertHandler | None = None,
    ) -> None:
        self._voice = _Channel("voice", voice_handler)
        self._push = _Channel("push", push_handler)
        self._sms = _Channel("sms", sms_handler)
        self._email = _Channel("email", email_handler)

    def dispatch(
        self, payload: AlertPayload, *, use_voice: bool = False
    ) -> list[DeliveryReceipt]:
        """Dispatch the alert to the appropriate channels.

        Voice is attempted first for L4; push/SMS/email follow in order.
        Email is *always* attempted for L2+ events.
        """
        receipts: list[DeliveryReceipt] = []

        # Channel 1 — Voice (L4)
        if use_voice:
            receipt = self._voice.send(payload)
            receipts.append(receipt)

        # Channel 2 — Push
        push_receipt = self._push.send(payload)
        receipts.append(push_receipt)

        # Channel 3 — SMS (fallback if push failed)
        if not push_receipt.delivered:
            receipts.append(self._sms.send(payload))

        # Channel 4 — Email (always sent)
        receipts.append(self._email.send(payload))

        return receipts
