import random
import json
from tokenizers import (
    decoders,
    models,
    pre_tokenizers,
    trainers,
    Tokenizer,
)
import os

random.seed(42)

"""
jsonl数据集中分词器预训练数据集格式如下：
{"text": "<|im_start|>鉴别一组中文文章的风格和特点，例如官方、口语、文言等。需要提供样例文章才能准确鉴别不同的风格和特点。<|im_end|> <|im_start|>好的，现在帮我查一下今天的天气怎么样?今天的天气依据地区而异。请问你需要我帮你查询哪个地区的天气呢？<|im_end|> <|im_start|>打开闹钟功能，定一个明天早上七点的闹钟。好的，我已经帮您打开闹钟功能，闹钟将在明天早上七点准时响起。<|im_end|> <|im_start|>为以下场景写一句话描述：一个孤独的老人坐在公园长椅上看着远处。一位孤独的老人坐在公园长椅上凝视远方。<|im_end|> <|im_start|>非常感谢你的回答。请告诉我，这些数据是关于什么主题的？这些数据是关于不同年龄段的男女人口比例分布的。<|im_end|> <|im_start|>帮我想一个有趣的标题。这个挺有趣的：\"如何成为一名成功的魔术师\" 调皮的标题往往会吸引读者的注意力。<|im_end|> <|im_start|>回答一个问题，地球的半径是多少？地球的平均半径约为6371公里，这是地球自赤道到两极的距离的平均值。<|im_end|> <|im_start|>识别文本中的语气，并将其分类为喜悦、悲伤、惊异等。\n文本：“今天是我的生日！”这个文本的语气是喜悦。<|im_end|>"}
{"text": "<|im_start|>根据输入的内容，编写一个类别标签。\n这是一篇介绍如何阅读心电图的文章类别标签: 医学/心电图阅读指南<|im_end|> <|im_start|>帮我搜索一下最近的天气情况。当然，我可以帮您搜索最新的天气情况。请问您需要查询哪个城市的天气情况呢？<|im_end|> <|im_start|>帮我讲一个令人开心的笑话。好的，我帮您讲一个关于细菌的笑话。为什么细菌不会上网？因为连接总是断开了！<|im_end|> <|im_start|>现在给我生成一首关于大海的五言诗。碧波万顷月满天，海天相接处天地间。波涛滚滚江山美，海鸟翱翔日月闲。<|im_end|> <|im_start|>谢谢你，这篇文章很有用。不客气，我很高兴能够为您提供帮助。如果您还有其他问题或需求，随时可以对我说。<|im_end|> <|im_start|>你好，我想下载一个视频编辑软件，你有什么推荐吗？您好！当然，有很多选择。您想要免费软件还是愿意付费？<|im_end|> <|im_start|>为什么我的程序不输出正确结果？可能是代码逻辑有误，或者输入数据有误，需要仔细调试代码逻辑和输入数据。<|im_end|> <|im_start|>谢谢你的回答。现在我想知道这场比赛的具体时间和地点。这场比赛的时间是北京时间10月4日，地点是上海。<|im_end|>"}
"""

def train_tokenizer():
    # 读取JSONL文件并提取文本数据
    def read_texts_from_jsonl(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                yield data['text']  # 注意yield和return的区别（返回一个值，下次继续从这里执行）（训练文件很大，不用一次性读入内存，节省内存，提高效率，ByteLevel BPE 可以按行迭代训练）

    data_path = '../dataset/pretrain_hq.jsonl'

    # 初始化tokenizer
    tokenizer = Tokenizer(models.BPE())
    # ByteLevel为预分词器，通过在分词前将文本转换为字节来稳健处理 Unicode 字符，确保任何语言的任何字符都可以无损表示
    # add_prefix_space=False 配置防止在序列开头添加不必要的空格，这对于保持精确的 token 边界很重要
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)  # 总结下来中文文本因为不会像英文那样靠空格区分，所以推荐False

    # 定义特殊token(之后在拆分前会被提前冻结不会被拆分，不会训练特殊token)
    special_tokens = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]

    # 设置训练器并添加特殊token
    # 训练时，BPE会以这些字节为最小单位开始统计频率，即每个字节最开始都是独立 token，训练器会统计这些 token 的频率；后续会把常用字节序列合并成子词；保证 tokenizer 能覆盖所有输入文本，无论语言或符号
    # initial_alphabet告诉BPE用ByteLevel的所有字节作为最小token开始训练
    trainer = trainers.BpeTrainer(
        vocab_size=6400,
        special_tokens=special_tokens,  # 确保这三个token被包含
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()  # pre_tokenizers.ByteLevel.alphabet() 会返回 ByteLevel tokenizer 的完整字节表
    )

    # 读取文本数据
    texts = read_texts_from_jsonl(data_path)

    # 训练tokenizer
    tokenizer.train_from_iterator(texts, trainer=trainer)

    # 设置解码器
    tokenizer.decoder = decoders.ByteLevel()

    # 检查特殊token的索引
    # assert是一个断言语句，会检查三个token id是否满足要求，如果任意一个不满足，会抛出 AssertionError，提醒 token ID 不对
    assert tokenizer.token_to_id("<|endoftext|>") == 0
    assert tokenizer.token_to_id("<|im_start|>") == 1
    assert tokenizer.token_to_id("<|im_end|>") == 2

    # 保存tokenizer
    # tokenizer.model.save()保存BPE核心模型：vocab.json + merges.txt（只保存词表和合并规则）、
    # tokenizer.save("tokenizer.json")保存完整 tokenizer，包括模型、预处理器、解码器、特殊 token（保存保存整个 tokenizer，包括规则 + 配置 + 特殊 token”，可以直接用来训练和推理，跨平台、可直接加载使用）
    tokenizer_dir = "../model/"
    os.makedirs(tokenizer_dir, exist_ok=True)
    tokenizer.save(os.path.join(tokenizer_dir, "tokenizer.json"))
    tokenizer.model.save("../model/")

    # 手动创建配置文件
    # 告诉 PreTrainedTokenizerFast 如何使用训练好的 tokenizer（tokenizer 的使用说明书）
    # 没有配置文件也可以用 tokenizer，但特殊 token 的 ID、BOS/EOS 
    # chat_template（Jinja2 模板）定义了
    # （1）每条消息用 <|im_start|>role\n内容<|im_end|> 包裹（2）系统消息、用户消息、助手消息分别处理（3）支持工具调用 <tools> / <tool_call> 标签（4）控制多步工具调用逻辑
    config = {
        "add_bos_token": False,
        "add_eos_token": False,
        "add_prefix_space": False,
        "added_tokens_decoder": {
            "0": {
                "content": "<|endoftext|>",
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": True
            },
            "1": {
                "content": "<|im_start|>",
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": True
            },
            "2": {
                "content": "<|im_end|>",
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": True
            }
        },
        "additional_special_tokens": [],
        "bos_token": "<|im_start|>",
        "clean_up_tokenization_spaces": False,
        "eos_token": "<|im_end|>",
        "legacy": True,
        "model_max_length": 32768,
        "pad_token": "<|endoftext|>",
        "sp_model_kwargs": {},
        "spaces_between_special_tokens": False,
        "tokenizer_class": "PreTrainedTokenizerFast",
        "unk_token": "<|endoftext|>",
        "chat_template": "{%- if tools %}\n    {{- '<|im_start|>system\\n' }}\n    {%- if messages[0].role == 'system' %}\n        {{- messages[0].content + '\\n\\n' }}\n    {%- endif %}\n    {{- \"# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>\" }}\n    {%- for tool in tools %}\n        {{- \"\\n\" }}\n        {{- tool | tojson }}\n    {%- endfor %}\n    {{- \"\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\\"name\\\": <function-name>, \\\"arguments\\\": <args-json-object>}\\n</tool_call><|im_end|>\\n\" }}\n{%- else %}\n {%- if messages[0]['role'] == 'system' -%}\n        {{- '<|im_start|>system\\n' + messages[0]['content'] + '<|im_end|>\\n' }}\n    {%- else -%}\n        {{- '<|im_start|>system\\nYou are a helpful assistant<|im_end|>\\n' }}\n {%- endif %}\n{%- endif %}\n{%- set ns = namespace(multi_step_tool=true, last_query_index=messages|length - 1) %}\n{%- for message in messages[::-1] %}\n    {%- set index = (messages|length - 1) - loop.index0 %}\n    {%- if ns.multi_step_tool and message.role == \"user\" and message.content is string and not(message.content.startswith('<tool_response>') and message.content.endswith('</tool_response>')) %}\n        {%- set ns.multi_step_tool = false %}\n        {%- set ns.last_query_index = index %}\n    {%- endif %}\n{%- endfor %}\n{%- for message in messages %}\n    {%- if message.content is string %}\n        {%- set content = message.content %}\n    {%- else %}\n        {%- set content = '' %}\n    {%- endif %}\n    {%- if (message.role == \"user\") or (message.role == \"system\" and not loop.first) %}\n        {{- '<|im_start|>' + message.role + '\\n' + content + '<|im_end|>' + '\\n' }}\n    {%- elif message.role == \"assistant\" %}\n   {{- '<|im_start|>' + message.role + '\\n' + content }}\n  {%- if message.tool_calls %}\n            {%- for tool_call in message.tool_calls %}\n                {%- if (loop.first and content) or (not loop.first) %}\n                    {{- '\\n' }}\n                {%- endif %}\n                {%- if tool_call.function %}\n                    {%- set tool_call = tool_call.function %}\n                {%- endif %}\n                {{- '<tool_call>\\n{\"name\": \"' }}\n                {{- tool_call.name }}\n                {{- '\", \"arguments\": ' }}\n                {%- if tool_call.arguments is string %}\n                    {{- tool_call.arguments }}\n                {%- else %}\n                    {{- tool_call.arguments | tojson }}\n                {%- endif %}\n                {{- '}\\n</tool_call>' }}\n            {%- endfor %}\n        {%- endif %}\n        {{- '<|im_end|>\\n' }}\n    {%- elif message.role == \"tool\" %}\n        {%- if loop.first or (messages[loop.index0 - 1].role != \"tool\") %}\n            {{- '<|im_start|>user' }}\n        {%- endif %}\n        {{- '\\n<tool_response>\\n' }}\n        {{- content }}\n        {{- '\\n</tool_response>' }}\n        {%- if loop.last or (messages[loop.index0 + 1].role != \"tool\") %}\n            {{- '<|im_end|>\\n' }}\n        {%- endif %}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- '<|im_start|>assistant\\n' }}\n    {%- if enable_thinking is defined and enable_thinking is false %}\n        {{- '<think>\\n\\n</think>\\n\\n' }}\n    {%- endif %}\n{%- endif %}"
    }

    # 保存配置文件
    with open(os.path.join(tokenizer_dir, "tokenizer_config.json"), "w", encoding="utf-8") as config_file:
        json.dump(config, config_file, ensure_ascii=False, indent=4)

    print("Tokenizer training completed and saved.")


# 验证了训练好的 tokenizer + 配置文件 能够正确渲染聊天消息、编码成模型输入、再解码回文本，保证训练和推理一致
def eval_tokenizer():
    from transformers import AutoTokenizer

    # 加载预训练的tokenizer
    tokenizer = AutoTokenizer.from_pretrained("../model/")

    messages = [
        {"role": "system", "content": "你是一个优秀的聊天机器人，总是给我正确的回应！"},
        {"role": "user", "content": '你来自哪里？'},
        {"role": "assistant", "content": '我来自地球'}
    ]
    # 把一组消息（messages）渲染成模型输入文本
    # apply_chat_template 是 PreTrainedTokenizerFast 提供的扩展方法（配合在 tokenizer_config.json 里定义的 chat_template）
    new_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False  # 表示只渲染文本，不进行编码成 token ID
    )
    print(new_prompt)  # 观察将messages按照模板模板渲染后的内容

    # 获取实际词汇表长度（包括特殊符号）
    actual_vocab_size = len(tokenizer)
    print('tokenizer实际词表长度：', actual_vocab_size)

    model_inputs = tokenizer(new_prompt)  # 根据预训练好的 tokenizer 的词表和分词规则，把文本编码成模型可识别的 token ID
    print('encoder长度：', len(model_inputs['input_ids']))

    input_ids = model_inputs['input_ids']
    response = tokenizer.decode(input_ids, skip_special_tokens=False)  # skip_special_tokens=False：保留 <|im_start|> 等特殊 token
    print('decoder和原始文本是否一致：', response == new_prompt)


def main():
    train_tokenizer()
    eval_tokenizer()


if __name__ == '__main__':
    main()
