
from transformers import AutoTokenizer,AutoModel

tokenizer= AutoTokenizer.from_pretrained(r"D:\huggingface\THUDM\chatglm3-6b",trust_remote_code=True)

model=AutoModel.from_pretrained(r"D:\huggingface\THUDM\chatglm3-6b",trust_remote_code=True).cuda()

response,history= model.chat(tokenizer,"你好",history=[])
print(response)

response,history= model.chat(tokenizer,"晚上睡不着怎么办",history=history)
print(response)
