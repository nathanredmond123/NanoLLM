#!/usr/bin/env python3
from nano_llm import NanoLLM, ChatHistory, BotFunctions, bot_function
from datetime import datetime

# For functions that take arguments and return something, use Google format to add Args and Returns.
@bot_function
def get_todays_date():
    """ A tool that gets today's date. """
    return datetime.now().strftime("%A, %B %-m %Y")
   
@bot_function
def get_current_time():
    """ A tool that returns the current time. """
    return datetime.now().strftime("%-I:%M %p")
          
# load the model   
model = NanoLLM.from_pretrained(
    model="meta-llama/Meta-Llama-3.1-8B-Instruct",
    quantization='q4f16_ft', 
    api='mlc'
)

load = False # Don't load built-in functions

# create the chat history
system_prompt ="""
You are a helpful and friendly AI assistant. 
""" + BotFunctions.generate_docs(prologue=True, epilogue=True, load=load)

#print(f"System prompt: {system_prompt}")
chat_history = ChatHistory(model, system_prompt=system_prompt)
#print(f"Chat template: {chat_history.template}")
#print(f"functions: {BotFunctions(load=load)}")
#print(f"stop token : {chat_history.template.stop}")


while True:
    # enter the user query from terminal
    print('>> ', end='', flush=True)
    prompt = input().strip()

    # add user prompt and generate chat tokens/embeddings
    chat_history.append(role='user', msg=prompt)
    embedding, position = chat_history.embed_chat()

    # generate bot reply (give it function access)
    reply = model.generate(
        embedding, 
        streaming=True, 
        functions=BotFunctions(load=load),
        kv_cache=chat_history.kv_cache,
        stop_tokens=chat_history.template.stop
    )
        
    # stream the output
    for token in reply:
        print(token, end='\n\n' if reply.eos else '', flush=True)

    # save the final output
    #print(chat_history.to_list())
    chat_history.append(role='bot', text=reply.text, tokens=reply.tokens)
    chat_history.kv_cache = reply.kv_cache
