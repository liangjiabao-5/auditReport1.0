import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import sys

os.environ['TRANSFORMERS_OFFLINE'] = '1' # 设置 GPU 编号，如果单机单卡指定一个，单机多卡指定多个 GPU 编号
MODEL_PATH = "/mnt/workspace/glm4/model"


if __name__ == "__main__":
    
    result_record = sys.argv[1]
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)


    messages = [
        {
            "role": "system",
            "content": "请将user输入的内容换一种形式表达，语义不变，要求是陈述句"
        },
        {
            "role": "user",
            "content": result_record
        }
    ]

    inputs = tokenizer.apply_chat_template(messages,
                                           add_generation_prompt=True,
                                           tokenize=True,
                                           return_tensors="pt",
                                           return_dict=True
                                           )

    inputs = inputs.to(device)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        device_map="auto"
    ).eval()

    #Temperature：简单来说，temperature 的参数值越小，模型就会返回越确定的一个结果。如果调高该参数值，大语言模型可能会返回更随机的结果，也就是说这可能会带来更多样化或更具创造性的产出。（调小temperature）实质上，你是在增加其他可能的 token 的权重。在实际应用方面，对于质量保障（QA）等任务，我们可以设置更低的 temperature 值，以促使模型基于事实返回更真实和简洁的结果。 对于诗歌生成或其他创造性任务，适度地调高 temperature 参数值可能会更好。
    #Top_p：同样，使用 top_p（与 temperature 一起称为核采样（nucleus sampling）的技术），可以用来控制模型返回结果的确定性。如果你需要准确和事实的答案，就把参数值调低。如果你在寻找更多样化的响应，可以将其值调高点。

    #gen_kwargs = {"max_length": 2500, "do_sample": True, "top_p": 0.2, "temperature": 0.2}
    gen_kwargs = {"max_length": 2500, "do_sample": True, "top_p": 1, "temperature": 1}
    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
        outputs = outputs[:, inputs['input_ids'].shape[1]:]
        print(tokenizer.decode(outputs[0], skip_special_tokens=True))