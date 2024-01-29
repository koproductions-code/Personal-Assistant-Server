#!/usr/bin/env python3
from transformers import AutoModelForCausalLM, AutoTokenizer
from fastapi import FastAPI, APIRouter
import uvicorn
import argparse
from pydantic import BaseModel

class Request(BaseModel):
    system: str
    functions: str
    prompt: str


class LLMServer:

    def __init__(self, args) -> None:
        if args.model == "Trelis/Llama-2-7b-chat-hf-function-calling-v3":
            self.tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True, load_in_8bit=True)
        elif args.model == "glaiveai/glaive-function-calling-v2-small":
            self.tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True).half().cuda()
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True).cuda()

        self.model.config.pad_token_id = self.tokenizer.eos_token_id
        self.max_tokens = args.tokens

        self.router = APIRouter()
        self.router.add_api_route("/request", self.request, methods=["POST"])

    def request(self, request: Request):
        """
            Parameters
            ----------
            system : str
                The system prompt to send before sending the user-prompt
            functions : list
                The available functions for the model.
            prompt : str
                The user-prompt
        """
        
        fullprompt = ""

        if request.system is not None:
            fullprompt += request.system
        
        fullprompt += request.functions
        fullprompt += "\n" + request.prompt

        inputs = self.tokenizer(fullprompt,return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(**inputs,do_sample=True,temperature=0.1,top_p=0.95,max_new_tokens=int(self.max_tokens))
        result = self.tokenizer.decode(outputs[0],skip_special_tokens=True)
        return {"response": result}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='This is a fastapi server for hosting a llm with transformers on a gpu.')

    parser.add_argument("-u", "--url", help="Host url to bind the server to.", default="0.0.0.0")
    parser.add_argument("-p", "--port", help="Host port to bind the server to.", default=8000)
    parser.add_argument("-m", "--model", help="The model to load into the transformer. Check to make sure the model works with the transformers library and fits onto your gpu.", default="glaiveai/glaive-function-calling-v2-small")
    parser.add_argument("-t", "--tokens", help="The max amount of tokens generated.", default=100)

    args = parser.parse_args()
    app = FastAPI()
    server = LLMServer(args)

    app.include_router(server.router)

    uvicorn.run(app, host=args.url, port=args.port)
