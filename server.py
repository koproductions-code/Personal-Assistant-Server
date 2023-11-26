from transformers import AutoModelForCausalLM , AutoTokenizer
from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel

class Request(BaseModel):
    system: str
    functions: list
    prompt: str

tokenizer = AutoTokenizer.from_pretrained("glaiveai/glaive-function-calling-v2-small", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("glaiveai/glaive-function-calling-v2-small", trust_remote_code=True).half().cuda()

model.config.pad_token_id = tokenizer.eos_token_id

app = FastAPI()

@app.post("/request")
def request(request: Request):
    fullprompt = ""

    if request.system is not None:
        fullprompt += request.system
    
    fullprompt += str(request.functions)
    fullprompt += "\n" + request.prompt

    inputs = tokenizer(fullprompt,return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs,do_sample=True,temperature=0.1,top_p=0.95,max_new_tokens=100)
    result = tokenizer.decode(outputs[0],skip_special_tokens=True)
    return {"response": result}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)