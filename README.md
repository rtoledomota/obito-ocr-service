# FILE: README.md

# obito-ocr-service

Serviço de OCR (reconhecimento de texto em imagens) que roda na internet e pode ser chamado pelo Google Apps Script.

Ele recebe uma imagem, envia para a inteligência artificial da OpenAI e devolve o texto encontrado.

Você não precisa entender de programação para usar. Basta seguir este passo a passo.

---

## 1) O que este projeto faz

- Expõe um pequeno serviço na internet (um "endpoint HTTP").
- Possui duas rotas:
  - `GET /health` — verifica se o serviço está no ar.
  - `POST /ocr` — recebe uma imagem e devolve o texto reconhecido.
- Foi feito para ser usado dentro do Google Apps Script.
- Nesta versão **PDF ainda não é suportado**. Se você enviar um PDF, o serviço retorna o erro `PDF_NOT_SUPPORTED_IN_V1`.

---

## 2) O que cada arquivo faz

- `main.py` — Código principal do serviço em Python com FastAPI. Define as rotas `/health` e `/ocr`.
- `requirements.txt` — Lista de bibliotecas que precisam ser instaladas (FastAPI, Uvicorn, etc.).
- `README.md` — Este arquivo de ajuda que você está lendo agora.
- `.gitignore` — Diz ao Git quais arquivos não devem ser enviados ao GitHub (ex.: segredos, pastas temporárias).

---

## 3) Como criar e subir os arquivos no GitHub

1. Crie uma conta no GitHub se ainda não tiver: https://github.com
2. Crie um novo repositório:
   - Nome sugerido: `obito-ocr-service`
   - Marque como **Público** ou **Privado** (tanto faz para o Render).
   - Não marque "Add a README" aqui, porque você já vai subir este README.
3. No seu computador, crie uma pasta com o mesmo nome do projeto.
4. Coloque dentro dela os arquivos do projeto:
   - `main.py`
   - `requirements.txt`
   - `README.md`
   - `.gitignore`
5. Suba os arquivos para o GitHub:
   - Pelo site do GitHub: clique em **Add file → Upload files** e arraste os arquivos.
   - Ou pelo terminal, se já usa Git:
     - `git init`
     - `git add .`
     - `git commit -m "Primeira versão do serviço OCR"`
     - `git branch -M main`
     - `git remote add origin https://github.com/SEU_USUARIO/obito-ocr-service.git`
     - `git push -u origin main`
6. Confirme no GitHub que todos os arquivos apareceram.

---

## 4) Como publicar no Render pelo painel

1. Crie uma conta no Render: https://render.com
2. Entre no painel e clique em **New + → Web Service**.
3. Escolha **Build and deploy from a Git repository**.
4. Conecte sua conta do GitHub e autorize o Render a acessar o repositório `obito-ocr-service`.
5. Selecione o repositório `obito-ocr-service`.
6. Preencha os campos exatamente como na próxima seção.
7. Clique em **Create Web Service**.
8. Aguarde o Render instalar as dependências e iniciar o serviço.
9. Quando terminar, ele vai mostrar uma URL parecida com:
   - `https://obito-ocr-service.onrender.com`
10. Pronto! O serviço está no ar.

---

## 5) Campos a preencher no Render

- **Name**: `obito-ocr-service`
- **Runtime**: `Python 3`
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`
- **Plan**: Free (ou outro, se preferir).

Deixe o restante no padrão.

---

## 6) Variáveis de ambiente no Render

No Render, vá em **Environment** e adicione as seguintes variáveis:

- `ENDPOINT_AUTH_TOKEN` — Senha que o Apps Script vai enviar para provar que pode usar o serviço.
- `OPENAI_API_KEY` — Sua chave da OpenAI. **Fica só no Render, nunca no Apps Script.**
- `OPENAI_API_URL` — Endereço da API da OpenAI. Ex.: `https://api.openai.com/v1`.
- `OPENAI_MODEL_DEFAULT` — Modelo padrão a usar. Ex.: `gpt-4o`.
- `MAX_FILE_SIZE_MB` — Tamanho máximo de arquivo em MB. Ex.: `10`.

Salve as variáveis e faça um novo deploy se o Render não reiniciar sozinho.

---

## 7) Como mapear no Apps Script

No Apps Script, você vai usar duas variáveis:

- `OCR_API_KEY` — deve ter o **mesmo valor** que você colocou em `ENDPOINT_AUTH_TOKEN` no Render.
- `OPENAI_API_KEY` — **não vai no Apps Script**. Ela fica apenas no Render, em segredo.

Ou seja:

- No Render: `ENDPOINT_AUTH_TOKEN = "minha-senha-secreta"`
- No Apps Script: `OCR_API_KEY = "minha-senha-secreta"`

Assim, o Apps Script prova quem é, mas nunca conhece a chave da OpenAI.

---

## 8) Qual URL usar no Apps Script

No seu código do Apps Script, use exatamente esta linha:

var OCR_API_URL = "https://obito-ocr-service.onrender.com/ocr";

---

## 9) Como testar /health e /ocr

### Testar /health

No navegador, acesse:

https://obito-ocr-service.onrender.com/health

Se o serviço estiver no ar, você verá uma resposta como:

{"status":"ok"}

### Testar /ocr

Use um programa como o Postman ou o comando `curl`:

curl -X POST https://obito-ocr-service.onrender.com/ocr -H "Authorization: Bearer SUA_SENHA" -F "file=@imagem.png"

Se enviar um PDF, receberá:

PDF_NOT_SUPPORTED_IN_V1

---

## 10) Aviso forte de segurança

Nunca coloque segredos dentro dos arquivos que vão para o GitHub.

Isso inclui:

- `OPENAI_API_KEY`
- `ENDPOINT_AUTH_TOKEN`
- Qualquer senha ou token

Regras práticas:

- Segredos ficam **somente no Render**, em Environment Variables.
- No Apps Script, use apenas a senha de autorização (`OCR_API_KEY`), nunca a chave da OpenAI.
- Nunca escreva chaves dentro do `main.py`.
- Nunca suba um arquivo `.env` para o GitHub.
- Se um segredo vazar, revogue imediatamente no painel da OpenAI e gere um novo.

Lembre-se: o GitHub é um lugar público por padrão. Tudo que vai para lá pode ser visto por outras pessoas.
