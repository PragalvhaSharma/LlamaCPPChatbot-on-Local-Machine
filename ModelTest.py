from llama_cpp import Llama

llm = Llama(
  model_path="darevox-7b.Q5_K_S.gguf",  # Download the model file first
  n_ctx=32768,  # The max sequence length to use - note that longer sequence lengths require much more resources
  n_threads=8,            # The number of CPU threads to use, tailor to your system and the resulting performance      # The number of layers to offload to GPU, if you have GPU acceleration available
)

prompt = "QUery"
output = llm(
    f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{prompt}\n\n### Response: ", # Prompt
    max_tokens=512, # Generate up to 32 tokens
    stop=["</s>"], # Stop generating just before the model would generate a new question
    echo=True # Echo the prompt back in the output
) # Generate a completion, can also call create_completion
print(output["choices"][0]["text"])

