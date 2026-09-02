from dataguard.connectors.base import Connector, ConnectorObject, ConnectorType


def test_connector_contract_is_provider_neutral() -> None:
    assert ConnectorType.OBJECT_STORAGE.value == "object_storage"
    assert ConnectorType.DATABASE.value == "database"
    assert ConnectorObject("1", "file.txt", "text/plain", 10, None, "provider://1").object_id == "1"
    assert Connector.__abstractmethods__ == {"health", "list_objects", "read_object"}
