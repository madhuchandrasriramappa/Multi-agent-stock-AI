"""
Azure Service Bus wrapper for inter-agent event passing.
Phase 0: skeleton with typed interface.
Phase 6: full async producer/consumer implementation.
"""
from __future__ import annotations

import json
from typing import Any

import structlog

from config.settings import settings

logger = structlog.get_logger(__name__)


class PipelineEvent:
    """Envelope for all messages on the Service Bus queue."""

    def __init__(self, event_type: str, payload: dict[str, Any]) -> None:
        self.event_type = event_type
        self.payload = payload

    def to_json(self) -> str:
        return json.dumps({"event_type": self.event_type, "payload": self.payload})

    @classmethod
    def from_json(cls, raw: str) -> "PipelineEvent":
        body = json.loads(raw)
        return cls(event_type=body["event_type"], payload=body["payload"])


class ServiceBusGateway:
    """
    Thin wrapper around the Azure Service Bus SDK.

    Lazily imports the azure-servicebus package so the rest of the codebase
    can import this module without Azure credentials (useful in phases 0-5).
    """

    def __init__(self) -> None:
        self._client = None

    def _get_client(self):
        if self._client is None:
            settings.require("azure_servicebus_connection_string")
            from azure.servicebus import ServiceBusClient as _Client
            self._client = _Client.from_connection_string(
                settings.azure_servicebus_connection_string
            )
        return self._client

    def send(self, event: PipelineEvent) -> None:
        from azure.servicebus import ServiceBusMessage

        client = self._get_client()
        with client.get_queue_sender(settings.azure_servicebus_queue_name) as sender:
            sender.send_messages(ServiceBusMessage(event.to_json()))

        logger.info(
            "event_sent",
            event_type=event.event_type,
            queue=settings.azure_servicebus_queue_name,
        )

    def receive(self, max_messages: int = 10) -> list[PipelineEvent]:
        client = self._get_client()
        events: list[PipelineEvent] = []

        with client.get_queue_receiver(
            settings.azure_servicebus_queue_name, max_wait_time=5
        ) as receiver:
            for msg in receiver.receive_messages(max_message_count=max_messages):
                events.append(PipelineEvent.from_json(str(msg)))
                receiver.complete_message(msg)

        logger.info("events_received", count=len(events))
        return events


service_bus = ServiceBusGateway()
