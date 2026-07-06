# FILE: tests/test_v2_contract.py

"""
Testes de contrato do obito-ocr-service v2 (main.py).

Cobrem os comportamentos reais entregues no main.py v2:
- /health retorna JSON com status/service/version/time
- /ocr sem Authorization devolve 401 no formato {"detail": {...}}
- /ocr com PDF devolve 422 code PDF_NOT_SUPPORTED_IN_V1
- parse_obito preenche structured com as chaves de HEADER
- validate_obito marca idade incoerente em errors (nao warnings)
"""

import base64

import pytest
from fastapi.testclient import TestClient

import main
from main import HEADER, app, parse_obito, validate_obito


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def client():
    """Cliente de teste FastAPI."""
    return TestClient(app)


@pytest.fixture(autouse=True)
def setup_auth_token(monkeypatch):
    """Garante um AUTH_TOKEN de teste durante a execucao dos testes."""
    monkeypatch.setattr(main, "ENDPOINT_AUTH_TOKEN", "test-token-123")
    yield
    monkeypatch.setattr(main, "ENDPOINT_AUTH_TOKEN", "")


# Texto de exemplo casando com as regex do parser entregue.
EXEMPLO_OBITO = """\
Nome: João da Silva Santos
Nome da mae: Maria Aparecida da Silva
Nome do pai: Jose da Silva
Sexo: M
Data de nascimento: 15/03/1955
Data do obito: 20/09/2023
Hora do obito: 14:30
Causa da morte: Infarto agudo do miocardio
Causa basica: Infarto agudo do miocardio I21.9
"""


# ---------------------------------------------------------------------------
# Testes
# ---------------------------------------------------------------------------

def test_health_retorna_200(client):
    """GET /health deve retornar 200 com os campos do contrato."""
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["service"] == "obito-ocr-service"
    assert body["version"] == "2.0.0"
    assert "time" in body and body["time"]


def test_ocr_sem_auth_retorna_401(client):
    """POST /ocr sem Authorization deve devolver 401 no formato {"detail": {...}}."""
    payload = {
        "file": base64.b64encode(b"dummy").decode("ascii"),
        "mimeType": "image/jpeg",
    }
    response = client.post("/ocr", json=payload)
    assert response.status_code == 401
    body = response.json()
    # FastAPI encapsula HTTPException em {"detail": ...}
    assert "detail" in body
    detail = body["detail"]
    assert isinstance(detail, dict)
    assert detail.get("code") == "UNAUTHORIZED"


def test_parser_extrai_campos_principais():
    """parse_obito deve extrair os campos principais do texto de exemplo."""
    file_bytes = b"conteudo-fake-do-arquivo"
    structured = parse_obito(EXEMPLO_OBITO, file_bytes)

    assert structured["NOME"] == "João da Silva Santos"
    assert structured["NOME_MAE"] == "Maria Aparecida da Silva"
    assert structured["NASCIMENTO"] == "15/03/1955"
    assert structured["DATA_OBITO"] == "20/09/2023"
    assert structured["HORA_OBITO"] == "14:30"
    assert structured["SEXO"] == "M"
    # causa basica é a última causa capturada
    assert "Infarto" in structured["CAUSA_BASICA"]
    # CID basica capturado pelo regex de CID-10
    assert structured["CID_BASICA"] == "I21.9"


def test_validation_marca_erro_quando_idade_nao_bate():
    """validate_obito deve colocar idade incoerente em errors, não warnings."""
    # Nascimento 15/03/1955 + óbito 20/09/2023 => idade ~68 anos.
    # Texto declara 80 anos => diferença > 1 => erro.
    texto = "Idade: 80 anos"
    result = validate_obito(
        nascimento="15/03/1955",
        data_obito="20/09/2023",
        hora_obito="14:30",
        uf="",
        uf_obito="",
        uf_nascimento="",
        cep="",
        text=texto,
        nome="João da Silva Santos",
        nome_mae="Maria Aparecida da Silva",
        causa_basica="Infarto agudo do miocardio",
    )
    assert any("Idade incoerente" in erro for erro in result["errors"])
    # Garante que não caiu em warnings
    assert not any("Idade incoerente" in w for w in result["warnings"])


def test_pdf_retorna_erro_controlado(client):
    """POST /ocr com PDF deve devolver 422 code PDF_NOT_SUPPORTED_IN_V1."""
    payload = {
        "file": base64.b64encode(b"%PDF-1.4 fake").decode("ascii"),
        "mimeType": "application/pdf",
    }
    headers = {"Authorization": "Bearer test-token-123"}
    response = client.post("/ocr", json=payload, headers=headers)
    assert response.status_code == 422
    body = response.json()
    assert body.get("code") == "PDF_NOT_SUPPORTED_IN_V1"


def test_structured_respeita_header_order():
    """O objeto structured deve ter exatamente as chaves de HEADER na ordem."""
    file_bytes = b"conteudo-fake-do-arquivo"
    structured = parse_obito(EXEMPLO_OBITO, file_bytes)
    chaves = list(structured.keys())
    assert chaves == HEADER
