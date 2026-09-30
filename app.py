import os
import gradio as gr
from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv()

ENDPOINT = os.getenv("ENDPOINT", "https://agentesenai2026.services.ai.azure.com").rstrip("/")
API_KEY = os.getenv("API_KEY")
MODELO = os.getenv("MODELO", "gpt-5-mini")  # Nome do deployment no Azure AI Foundry
API_VERSION = os.getenv("API_VERSION", "2024-10-21")

# Inicializa o cliente Azure OpenAI
client = AzureOpenAI(
    azure_endpoint=ENDPOINT,
    api_key=API_KEY,
    api_version=API_VERSION
)

# ==========================================
# BASE DE CONHECIMENTO FICTÍCIA (RAG)
# ==========================================
BASE_CONHECIMENTO = {
    "azure ai foundry": "O Azure AI Foundry é uma plataforma unificada que permite aos desenvolvedores criar, testar e gerenciar soluções de IA generativa e agentes inteligentes.",
    "precos": "Os custos do Azure AI Foundry variam conforme o consumo de tokens dos modelos de linguagem utilizados e os serviços de infraestrutura associados."
}

# ==========================================
# AGENTE 1: O RECUPERADOR (Retriever Agent)
# ==========================================
def agente_recuperador(pergunta: str) -> str:
    """Busca informações relevantes na base de conhecimento com base na intenção."""
    pergunta_lower = pergunta.lower()
    contextos_encontrados = []
    
    for chave, conteudo in BASE_CONHECIMENTO.items():
        if chave in pergunta_lower:
            contextos_encontrados.append(conteudo)
            
    if not contextos_encontrados:
        return "Nenhum documento interno específico encontrado na base de conhecimento."
    
    return "\n".join(contextos_encontrados)

# ==========================================
# AGENTE 2: O GERADOR (Writer/RAG Agent)
# ==========================================
def agente_gerador(pergunta: str, contexto: str) -> str:
    """Usa o modelo no Azure AI Foundry para sintetizar a resposta com base no contexto."""
    prompt_sistema = (
        "Você é um assistente especialista. Responda à pergunta do usuário "
        "estritamente com base no contexto fornecido abaixo. Se a informação não estiver "
        "no contexto, diga que não sabe."
    )
    
    prompt_usuario = f"Contexto:\n{contexto}\n\nPergunta: {pergunta}"
    
    response = client.chat.completions.create(
        model=MODELO,
        messages=[
            {"role": "system", "content": prompt_sistema},
            {"role": "user", "content": prompt_usuario}
        ],
    )
    
    return response.choices[0].message.content

# ==========================================
# ORQUESTRAÇÃO PARA A INTERFACE
# ==========================================
def processar_fluxo_multiagente(pergunta: str):
    if not pergunta.strip():
        return "", ""
    
    # Passo 1: Recuperação
    contexto = agente_recuperador(pergunta)
    
    # Passo 2: Geração
    resposta = agente_gerador(pergunta, contexto)
    
    return contexto, resposta

# ==========================================
# INTERFACE GRÁFICA (Gradio)
# ==========================================
with gr.Blocks(title="Sistema Multi-Agente RAG") as demo:
    gr.Markdown("# 🤖 Sistema Multi-Agente RAG")
    gr.Markdown("Digite sua pergunta abaixo para acionar o **Agente Recuperador** e o **Agente Gerador** via Azure AI Foundry.")
    
    with gr.Row():
        with gr.Column(scale=1):
            input_pergunta = gr.Textbox(
                label="Sua Pergunta", 
                placeholder="Ex: O que é o Azure AI Foundry e quais os preços?",
                lines=3
            )
            btn_enviar = gr.Button("Enviar Pergunta", variant="primary")
            
        with gr.Column(scale=1):
            output_contexto = gr.Textbox(
                label="[Agente 1 - Recuperador] Contexto Encontrado", 
                interactive=False,
                lines=4
            )
            output_resposta = gr.Textbox(
                label="[Agente 2 - Gerador] Resposta Final", 
                interactive=False,
                lines=6
            )
            
    btn_enviar.click(
        fn=processar_fluxo_multiagente,
        inputs=[input_pergunta],
        outputs=[output_contexto, output_resposta]
    )

if __name__ == "__main__":
    demo.launch()