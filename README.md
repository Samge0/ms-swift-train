## 使用 ms-swift 框架微调 Qwen1.5-7B-chat 模型并转换为 Ollama 支持格式的实践

本文仅用于记录使用[ms-swift](https://github.com/modelscope/ms-swift)框架对[Qwen1.5-7B-chat](https://www.modelscope.cn/models/qwen/qwen1.5-7b-chat)模型+[ruozhiba](https://www.modelscope.cn/datasets/AI-ModelScope/ruozhiba/dataPeview)数据集进行微调，使用[llama.cpp](https://github.com/ggerganov/llama.cpp)将合并后的微调模型转为[Ollama](https://ollama.com/)支持格式（.gguf）+int4量化，以便在低资源设备上高效推理。

主要流程为：
- ms-swift多卡训练配置
- LoRA微调&LoRA合并
- 编译llama.cpp
- 转换为gguf格式&int4量化
- 构建ollama的Modelfile
- 构建ollama模型并上传

本文为 `ms-swift` 框架微调后转为`ollama`模型的练习过程记录。

本测试的环境为: `Windows` + [VsCode](https://code.visualstudio.com/) + `3090单卡（24G）`。

运行[最终输出的测试ollama模型](https://ollama.com/samge/qwen1half-7b-chat-ruozhiba)：
```shell
ollama run samge/qwen1half-7b-chat-ruozhiba:int4
```


### 目录
- [下载 Qwen1.5-7B-chat 模型](#下载-qwen15-7b-chat-模型)
- [微调 Qwen1.5-7B-chat 模型](#微调-qwen15-7b-chat-模型)
- [合并 LoRA 并进行量化](#合并-lora-并进行量化)
- [编译 llama.cpp](#编译-llamacpp)
- [转换为 gguf 格式](#转换为-gguf-格式)
- [构建Ollama模型并推送](#构建ollama模型并推送)
- [相关截图](#相关截图)

---

### 下载 Qwen1.5-7B-chat 模型
可参考[魔塔社区-通义千问1.5-7B-Chat](https://www.modelscope.cn/models/qwen/qwen1.5-7b-chat/files) 右侧 `下载模型` 按钮进行模型的下载。

- 方式1（使用git下载，需要先提前安装git的lfs）：
    ```shell
    git lfs clone https://www.modelscope.cn/qwen/qwen1.5-7b-chat.git
    ```

- 方式2（用魔塔官方SDK进行模型的下载）
    ```shell
    pip install modelscope

    modelscope download --model qwen/qwen1.5-7b-chat
    ```

---

### 微调 Qwen1.5-7B-chat 模型

微调模型前，先参考[.devcontainer/README.md](.devcontainer/README.md)使用`vscode`进入`Dev Container`开发环境。

之所以要用`Dev Container`，是由于在windows系统下安装依赖可能出现一些不必要的`意外麻烦`，而`Dev Container`中可以有一个比较完整的`linux`容器环境，比`wsl`省心一丢丢。

1. 配置多卡训练（linux下使用的命令，这里单卡，没用到，忽略。如果你有多卡才需要配置）：
    ```bash
    export MKL_THREADING_LAYER=GNU
    CUDA_VISIBLE_DEVICES=0,1,2,3 
    NPROC_PER_NODE=4
    ```
    
2. 使用 `ms-swift` 进行微调（需自行根据官方文档安装[ms-swift环境](https://github.com/modelscope/ms-swift/blob/main/README_CN.md) 或 使用上面说的[Dev Container](.devcontainer/README.md)进入`ms-swift环境`，这里的`model_id_or_path`需要指定你自己下载的基座模型路径，`dataset`可以指定`线上公开的数据集`或者根据[swift文档-自定义数据集](https://swift.readthedocs.io/zh-cn/latest/Instruction/%E8%87%AA%E5%AE%9A%E4%B9%89%E4%B8%8E%E6%8B%93%E5%B1%95.html#%E8%87%AA%E5%AE%9A%E4%B9%89%E6%95%B0%E6%8D%AE%E9%9B%86)准备自己的`jsonl`/`csv`格式自定义数据集）：
    ```bash 
    CUDA_VISIBLE_DEVICES=0 \
    swift sft --model_type qwen1half-7b-chat \
        --model_id_or_path /root/.cache/modelscope/qwen/Qwen1___5-7B-Chat \
        --sft_type lora \
        --dtype AUTO \
        --dataset AI-ModelScope/ruozhiba \
        --self_cognition_sample 3000 \
        --model_name 山姆模型 'Samge Model' \
        --model_author 山姆 Samge \
        --num_train_epochs 3 \
        --lora_rank 8 \
        --lora_alpha 32 \
        --lora_dropout_p 0.05 \
        --lora_target_modules ALL \
        --gradient_checkpointing false \
        --batch_size 4 \
        --weight_decay 0.05 \
        --learning_rate 5e-5 \
        --gradient_accumulation_steps 4 \
        --output_dir output
    ```

3. （可选，加载模型耗时挺大）使用 `ms-swift` 进行推理（运行前需自行根据官方文档安装ms-swift环境）：
    ```bash
    CUDA_VISIBLE_DEVICES=0 swift infer --ckpt_dir output/qwen1half-7b-chat/v2-20240910-185715/checkpoint-16461
    ```

---

### 合并 LoRA 并进行量化

1. 合并 LoRA 并导出模型：
    ```bash
    CUDA_VISIBLE_DEVICES=0 swift export --ckpt_dir output/qwen1half-7b-chat/v2-20240910-185715/checkpoint-16461 --merge_lora true
    ```

2. （可选）进行 int4 量化（量化耗时挺大，3090单卡-rouzhiba微调模型量化-大概60分钟。可选择用llama.cpp转为.gguf后再量化）：
    ```bash
    CUDA_VISIBLE_DEVICES=0 swift export --ckpt_dir output/qwen1half-7b-chat/v2-20240910-185715/checkpoint-16461 --quant_bits 4 --quant_method awq --merge_lora true
    ```

---

### 编译 llamacpp
（用于将微调后的模型转为ollama支持的`gguf`格式）

1. 下载并解压 w64devkit（Windows 环境下编译 llama.cpp 的工具，主要用其执行`make -j`命令）：
    - 下载地址：https://github.com/skeeto/w64devkit/releases
    - 推荐使用 `1.2.0 版本`，避免安全软件报毒问题。

2. 克隆 `llama.cpp` 仓库并进行编译：
    ```bash
    git clone https://github.com/ggerganov/llama.cpp

    cd llama.cpp

    # windows系统下的 make 操作需要用 w64devkit 进行
    make -j
    ```

3. 创建并激活 `llama.cpp` 的 Python 环境：
    ```bash
    conda create --name llamacpp python=3.10.13 -y

    conda activate llamacpp

    pip install -r requirements.txt
    ```
---

### 转换为 gguf 格式
（需要在上面创建的`llamacpp`环境中进行）

1. 转换 `safetensors` 模型为 Ollama 格式（15:36 -> 15:38，耗时2分钟，模型大小：15.8G）：
    ```bash
    python convert_hf_to_gguf.py D:/Space/PRO/ai/ms-swift-train/output/qwen1half-7b-chat/v2-20240910-185715/checkpoint-16461-merged --outtype f16
    ```

2. （可选）进行 int4 量化（15:40 -> 15:41，将f16转int4，耗时1分钟，模型大小：4.2G）：
    ```bash
    ./llama-quantize D:/Space/PRO/ai/ms-swift-train/output/qwen1half-7b-chat/v2-20240910-185715/checkpoint-16461-merged/Checkpoint-16461-Merged-7.7B-F16.gguf D:/Space/PRO/ai/ms-swift-train/output/qwen1half-7b-chat/v2-20240910-185715/checkpoint-16461-merged/Checkpoint-16461-Merged-7.7B-q4_0.gguf q4_0
    ```

---

### 构建Ollama模型并推送
（需要先自行安装[Ollama](https://ollama.com/)）

1. 编写 `Modelfile` 文件：
    ```text
    FROM D:/Space/PRO/ai/ms-swift-train/output/qwen1half-7b-chat/v2-20240910-185715/checkpoint-16461-merged/Checkpoint-16461-Merged-7.7B-q4_0.gguf
    ```

2. 构建 Ollama 模型：
    ```bash
    ollama create samge/qwen1half-7b-chat-ruozhiba:int4 -f Modelfile
    ```

3. 使用 Ollama 运行并测试模型：
    ```bash
    ollama run samge/qwen1half-7b-chat-ruozhiba:int4
    ```

4. 推送模型至 Ollama：
    ```bash
    ollama push samge/qwen1half-7b-chat-ruozhiba:int4
    ```

---

### 备注
- 通过将模型量化为 `int4`，可以大幅减小模型大小，使其更适用于低资源设备上的推理。
- 使用 `ollama` 进行模型创建和推理，可以方便地将模型部署到不同的平台。

---

### 相关截图
![image.png](https://ollama.com/assets/samge/qwen1half-7b-chat-ruozhiba/03031f9d-2e9b-429e-952c-db13a8dd067b)
![image](https://github.com/user-attachments/assets/1a377816-87a0-49d9-9b25-8c18cb4e3ff5)
![image](https://github.com/user-attachments/assets/da4f3425-340f-446f-a866-bde97021200b)
![image](https://github.com/user-attachments/assets/bfbed5be-c1e1-4efb-88a7-369316cdb4cc)

