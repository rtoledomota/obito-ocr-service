# FILE: tests/test_smoke.py
"""
Testes de smoke (fumaça) para o obito-ocr-service.

Objetivo: garantir que os endpoints principais existem e respondem conforme
o contrato mínimo esperado pela versão 1 do serviço.

Execução recomendada:
    pytest tests/test_smoke.py -v
"""

import os

import pytest
from fastapi.testclient import TestClient

# Importa a aplicação FastAPI exposta pelo serviço principal.
from main import app


@pytest.fixture(scope="module")
def client() -> TestClient:
    """Cria um cliente de teste reutilizável para os testes do módulo."""
    return TestClient(app)


def test_health_retorna_200(client: TestClient) -> None:
    """GET /health deve retornar status 200 indicando que o serviço está saudável."""
    resposta = client.get("/health")

    assert resposta.status_code == 200


def test_ocr_sem_token_retorna_401(client: TestClient) -> None:
    """POST /ocr sem informar o token Bearer deve retornar 401 (não autorizado)."""
    resposta = client.post(
        "/ocr",
        headers={},
    )

    assert resposta.status_code == 401


def test_ocr_com_token_invalido_retorna_401(client: TestClient) -> None:
    """POST /ocr com um token Bearer inválido deve retornar 401 (não autorizado)."""
    resposta = client.post(
        "/ocr",
        headers={"Authorization": "Bearer token_invalido"},
    )

    assert resposta.status_code == 401


def test_ocr_com_pdf_retorna_erro_controlado_pdf_not_supported(client: TestClient) -> None:
    """
    POST /ocr com um arquivo PDF deve retornar um erro controlado indicando
    que PDFs não são suportados na versão 1 do serviço.

    O erro controlado é identificado pelo código PDF_NOT_SUPPORTED_IN_V1.
    """
    token = os.getenv("ENDPOINT_AUTH_TOKEN", "")

    # Conteúdo mínimo simulando um arquivo PDF enviado via multipart.
    pdf_fake = b"%PDF-1.4\n%fake pdf para teste\n"
    arquivos = {
        "file": ("documento.pdf", pdf_fake, "application/pdf"),
    }

    resposta = client.post(
        "/ocr",
        headers={"Authorization": f"Bearer {token}"},
        files=arquivos,
    )

    # O erro é controlado, portanto espera-se uma resposta com corpo JSON.
    corpo = resposta.json()

    # Aceita tanto o campo "code" quanto "error_code" para maior resiliência
    # caso o contrato do serviço utilize uma das duas convenções.
    codigo_erro = corpo.get("code") or corpo.get("error_code")

    assert codigo_erro == "PDF_NOT_SUPPORTED_IN_V1"
