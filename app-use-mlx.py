from mlx_lm import load, generate, convert

MODEL_NAME = "lmsys/vicuna-7b-v1.5"

quantized_model = "../AI_model_data/quantized_model_old"

# model, tokenizer = load(quantized_model)

# # response = generate(model, tokenizer, prompt="hello", verbose=True)

# model_pipeline = pipeline(
#   "text-generation",
#   model=model ,
#   tokenizer=tokenizer,
#   max_new_tokens=512,
#   pad_token_id=tokenizer.eos_token_id,
#   device_map="auto"
# )

quantized_mlx_mosel=convert(quantized_model, quantize=True)
