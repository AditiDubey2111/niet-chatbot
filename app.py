import gradio as gr
from transformers import pipeline

import rag  # rag.py file


def generate(query, context):
    # Prepare the prompt for the model
    input_text = f"Context: {context}\nQuestion: {query}\nAnswer:"
    generator = pipeline(
        "text-generation", model="openlm-research/open_llama_3b")

    answer = generator(input_text, max_length=200, num_return_sequences=1)
    return answer[0]["generated_text"]


def chat(query, history=[]):
    context, _source = rag.retrieve(query)
    response = generate(query, context)
    history.append(response)

    return response


# Define the Gradio interface
ui = gr.ChatInterface(fn=chat, type="messages",
                      title="NIET Chatbot")

# Launch the interface
ui.launch()
