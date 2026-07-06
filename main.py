# FILE: main.py

# -----------------------------------------------------------------------------
# obito-ocr-service
# Serviço de OCR (reconhecimento de texto em imagem) feito com FastAPI.
# Objetivo: receber uma imagem em base64, enviar para um provedor (OpenAI)
# e devolver o texto reconhecido de forma simples e padronizada.
#
# Como rodar localmente:
#   1) Instale as dependências:  pip install fastapi requests uvicorn
#   2) Defina as variáveis de ambiente (ou crie um arquivo .env local).
#   3) Rode:  python main.py
#
# Como publicar no Render:
#   - Build Command:    pip install fastapi requests uvicorn
#   - Start Command:    uvicorn main:app --host 0.0.0.0 --port $PORT
# -----------------------------------------------------------------------------

import os
import base64
import uuid
import time
import requests
from typing import Optional, Dict, Any, List

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse


# -----------------------------------------------------------------------------
# Configurações via variáveis de ambiente
# -----------------------------------------------------------------------------

# Token de autenticação que o cliente deve enviar no cabeçalho Authorization.
# Exemplo: Authorization: Bearer SEU_TOKEN_AQUI
# Se a variável não existir, usamos uma string vazia (que bloqueia tudo).
ENDPOINT_AUTH_TOKEN: str = os.getenv("ENDPOINT_AUTH_TOKEN", "")

# Tamanho máximo do arquivo enviado (em Megabytes).
# Valor padrão de 10 MB caso a variável não seja definida.
try:
    MAX_FILE_SIZE_MB: int = int(os.getenv("MAX_FILE_SIZE_MB", "10"))
except ValueError:
    MAX_FILE_SIZE_MB = 10

# Configurações do provedor de OCR (OpenAI compatível).
OPENAI_API_URL: str = os.getenv("OPENAI_API_URL", "https://api.openai.com/v1/chat/completions")
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL_DEFAULT: str = os.getenv("OPENAI_MODEL_DEFAULT", "gpt-4o")

# Lista de tipos de arquivo (MIME) que aceitamos neste serviço.
ACCEPTED_MIME_TYPES: List[str] = [
    "image/jpeg",
    "image/png",
    "image/webp",
    "application/pdf",
]


# -----------------------------------------------------------------------------
# Criação do aplicativo FastAPI
# -----------------------------------------------------------------------------

app = FastAPI(
    title="obito-ocr-service",
    description="Serviço simples de OCR via FastAPI integrado a um provedor OpenAI.",
    version="1.0.0",
)


# -----------------------------------------------------------------------------
# Funções auxiliares
# -----------------------------------------------------------------------------

def build_error_response(
    code: str,
    message: str,
    request_id: Optional[str] = None,
    status_code: int = 400,
) -> JSONResponse:
    """
    Cria uma resposta de erro padronizada em JSON.
    Sempre retorna um objeto com: code, message e requestId.
    """
    return JSONResponse(
        status_code=status_code,
        content={
            "code": code,
            "message": message,
            "requestId": request_id,
        },
    )


def authenticate(authorization: Optional[str]) -> None:
    """
    Verifica se o token Bearer enviado é válido.
    Se não for, dispara um erro 401 (não autorizado).
    """
    # Se não definimos um token no servidor, bloqueamos tudo por segurança.
    if not ENDPOINT_AUTH_TOKEN:
        raise HTTPException(
            status_code=500,
            detail={
                "code": "SERVER_NOT_CONFIGURED",
                "message": "Servidor sem ENDPOINT_AUTH_TOKEN configurado.",
                "requestId": None,
            },
        )

    # O cabeçalho deve vir no formato: Bearer TOKEN
    if not authorization:
        raise HTTPException(
            status_code=401,
            detail={
                "code": "MISSING_AUTH_HEADER",
                "message": "Cabeçalho Authorization ausente.",
                "requestId": None,
            },
        )

    parts = authorization.split(" ", 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(
            status_code=401,
            detail={
                "code": "INVALID_AUTH_SCHEME",
                "message": "Use o esquema Bearer no cabeçalho Authorization.",
                "requestId": None,
            },
        )

    token = parts[1].strip()
    if token != ENDPOINT_AUTH_TOKEN:
        raise HTTPException(
            status_code=401,
            detail={
                "code": "INVALID_AUTH_TOKEN",
                "message": "Token de autenticação inválido.",
                "requestId": None,
            },
        )


def decode_base64_file(file_b64: str) -> bytes:
    """
    Converte uma string base64 em bytes do arquivo.
    Lança erro se a string for inválida.
    """
    try:
        # Removemos possíveis espaços e quebras de linha.
        cleaned = file_b64.strip().replace("\n", "").replace("\r", "")
        return base64.b64decode(cleaned, validate=True)
    except Exception:
        raise HTTPException(
            status_code=400,
            detail={
                "code": "INVALID_BASE64",
                "message": "O campo 'file' não é um base64 válido.",
                "requestId": None,
            },
        )


def call_openai_ocr(image_b64: str, mime_type: str, model: str) -> Dict[str, Any]:
    """
    Envia a imagem para o provedor OpenAI e devolve o texto reconhecido.
    Retorna um dicionário com: text, confidence, warnings.
    """
    warnings: List[str] = []

    # Montamos a mensagem no formato esperado pela API do OpenAI.
    # A imagem vai como data URL (base64) dentro do conteúdo da mensagem.
    data_url = f"data:{mime_type};base64,{image_b64}"

    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Você é um motor de OCR preciso. "
                    "Extraia todo o texto visível na imagem, mantendo a ordem de leitura. "
                    "Devolva apenas o texto reconhecido, sem comentários."
                ),
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Extraia todo o texto da imagem a seguir.",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": data_url},
                    },
                ],
            },
        ],
        # Mantemos a temperatura baixa para resultados mais determinísticos.
        "temperature": 0,
    }

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }

    try:
        response = requests.post(
            OPENAI_API_URL,
            headers=headers,
            json=payload,
            timeout=60,
        )
    except requests.exceptions.Timeout:
        raise HTTPException(
            status_code=504,
            detail={
                "code": "PROVIDER_TIMEOUT",
                "message": "O provedor demorou demais para responder.",
                "requestId": None,
            },
        )
    except requests.exceptions.RequestException as exc:
        raise HTTPException(
            status_code=502,
            detail={
                "code": "PROVIDER_CONNECTION_ERROR",
                "message": f"Erro de conexão com o provedor: {exc}",
                "requestId": None,
            },
        )

    if response.status_code != 200:
        # Repassamos o erro do provedor de forma controlada.
        raise HTTPException(
            status_code=502,
            detail={
                "code": "PROVIDER_ERROR",
                "message": f"Provedor retornou status {response.status_code}.",
                "requestId": None,
            },
        )

    try:
        data = response.json()
        text = data["choices"][0]["message"]["content"]
    except Exception:
        raise HTTPException(
            status_code=502,
            detail={
                "code": "PROVIDER_INVALID_RESPONSE",
                "message": "Resposta do provedor em formato inesperado.",
                "requestId": None,
            },
        )

    # Como o provedor não retorna confiança numérica, usamos um valor fixo.
    confidence = 0.95
    warnings.append("Confiança estimada (provedor não retorna valor numérico).")

    return {
        "text": text,
        "confidence": confidence,
        "warnings": warnings,
    }


# -----------------------------------------------------------------------------
# Rotas
# -----------------------------------------------------------------------------

@app.get("/health")
def health() -> Dict[str, Any]:
    """
    Rota simples para verificar se o serviço está no ar.
    Não exige autenticação.
    """
    return {
        "status": "ok",
        "service": "obito-ocr-service",
        "version": "1.0.0",
        "time": int(time.time()),
    }


@app.post("/ocr")
def ocr(
    payload: Dict[str, Any],
    authorization: Optional[str] = Header(None),
) -> JSONResponse:
    """
    Rota principal de OCR.

    Espera um JSON com:
      - file (obrigatório): arquivo em base64
      - mimeType (obrigatório): tipo do arquivo (ex.: image/png)
      - model (opcional): modelo do provedor
      - fileName (opcional): nome original do arquivo
      - fileId (opcional): identificador do arquivo no cliente
      - requestId (opcional): identificador da requisição

    Retorna JSON com: text, confidence, provider, requestId, warnings
    """

    # 1) Autenticação Bearer
    authenticate(authorization)

    # 2) Gera um requestId se o cliente não enviou um.
    request_id: str = payload.get("requestId") or str(uuid.uuid4())

    # 3) Validação dos campos obrigatórios
    file_b64 = payload.get("file")
    mime_type = payload.get("mimeType")

    if not file_b64 or not isinstance(file_b64, str):
        return build_error_response(
            code="MISSING_FILE",
            message="O campo 'file' é obrigatório e deve ser uma string base64.",
            request_id=request_id,
            status_code=400,
        )

    if not mime_type or not isinstance(mime_type, str):
        return build_error_response(
            code="MISSING_MIME_TYPE",
            message="O campo 'mimeType' é obrigatório.",
            request_id=request_id,
            status_code=400,
        )

    # 4) Validação do tipo de arquivo (MIME)
    if mime_type not in ACCEPTED_MIME_TYPES:
        return build_error_response(
            code="UNSUPPORTED_MIME_TYPE",
            message=f"Tipo não suportado: {mime_type}. Aceitos: {', '.join(ACCEPTED_MIME_TYPES)}.",
            request_id=request_id,
            status_code=415,
        )

    # 5) PDF ainda não é suportado nesta versão (V1)
    if mime_type == "application/pdf":
        return build_error_response(
            code="PDF_NOT_SUPPORTED_IN_V1",
            message="PDF ainda não é suportado nesta versão (V1). Envie imagens JPEG, PNG ou WEBP.",
            request_id=request_id,
            status_code=422,
        )

    # 6) Decodifica o base64 e valida o tamanho
    file_bytes = decode_base64_file(file_b64)
    size_mb = len(file_bytes) / (1024 * 1024)

    if size_mb > MAX_FILE_SIZE_MB:
        return build_error_response(
            code="FILE_TOO_LARGE",
            message=f"Arquivo de {size_mb:.2f} MB excede o limite de {MAX_FILE_SIZE_MB} MB.",
            request_id=request_id,
            status_code=413,
        )

    # 7) Define o modelo do provedor (padrão ou enviado pelo cliente)
    model = payload.get("model") or OPENAI_MODEL_DEFAULT

    # 8) Verifica se o provedor está configurado
    if not OPENAI_API_KEY:
        return build_error_response(
            code="PROVIDER_NOT_CONFIGURED",
            message="Servidor sem OPENAI_API_KEY configurado.",
            request_id=request_id,
            status_code=500,
        )

    # 9) Reenvia o base64 "limpo" (sem cabeçalho data URL) para o provedor.
    clean_b64 = base64.b64encode(file_bytes).decode("utf-8")

    # 10) Chama o provedor de OCR
    try:
        result = call_openai_ocr(
            image_b64=clean_b64,
            mime_type=mime_type,
            model=model,
        )
    except HTTPException as exc:
        # Repassa o erro já formatado, incluindo o requestId.
        detail = exc.detail if isinstance(exc.detail, dict) else {"message": str(exc.detail)}
        return build_error_response(
            code=detail.get("code", "INTERNAL_ERROR"),
            message=detail.get("message", "Erro interno."),
            request_id=request_id,
            status_code=exc.status_code,
        )

    # 11) Monta a resposta de sucesso
    return JSONResponse(
        status_code=200,
        content={
            "text": result["text"],
            "confidence": result["confidence"],
            "provider": "openai",
            "requestId": request_id,
            "warnings": result["warnings"],
            "model": model,
            "fileName": payload.get("fileName"),
            "fileId": payload.get("fileId"),
        },
    )


# -----------------------------------------------------------------------------
# Execução local (apenas para testes na sua máquina)
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    # Importamos uvicorn aqui para não ser obrigatório em produção.
    import uvicorn

    # Porta padrão local: 8000. No Render usamos $PORT via Start Command.
    port = int(os.getenv("PORT", "8000"))

    print("Iniciando obito-ocr-service localmente...")
    print(f"Acesse: http://localhost:{port}/health")
    print(f"Documentação: http://localhost:{port}/docs")

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=True,
    )
