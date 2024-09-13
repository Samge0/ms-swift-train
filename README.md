## 使用 ms-swift 框架微调 Qwen1.5-7B-chat 模型并转换为 Ollama 支持格式的实践

本文仅用于记录使用[ms-swift](https://github.com/modelscope/ms-swift)框架对[Qwen1.5-7B-chat](https://www.modelscope.cn/models/qwen/qwen1.5-7b-chat)模型+[ruozhiba](https://www.modelscope.cn/datasets/AI-ModelScope/ruozhiba/dataPeview)数据集进行微调，使用[llama.cpp](https://github.com/ggerganov/llama.cpp)将合并后的微调模型转为Ollama支持格式（.gguf）+int4量化，以便在低资源设备上高效推理。

主要流程为：
- ms-swift多卡训练配置
- LoRA微调&LoRA合并
- int4量化
- 编译llama.cpp
- 构建ollama的Modelfile
- 构建ollama模型并上传

本文为 `ms-swift` 框架微调后转为`ollama`模型的练习过程记录。

本测试的环境为: `Windows` + `VsCode` + `3090单卡（24G）`。

运行：
```shell
ollama run samge/qwen1half-7b-chat-ruozhiba:int4
```


### 目录
- [下载 Qwen1.5-7B-chat 模型](#下载-Qwen1.5-7B-chat-模型)
- [微调 Qwen1.5-7B-chat 模型](#微调-Qwen1.5-7B-chat-模型)
- [合并 LoRA 并进行量化](#合并-lora-并进行量化)
- [编译 llama.cpp](#编译-llamacpp)
- [转换模型至 Ollama](#转换模型至-ollama)
- [Ollama推理与推送](#Ollama推理与推送)
  
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

之所以能用`Dev Container`，是因为windows系统下安装依赖可能出现一些`意外`，而`Dev Container`中可以有一个比较完整的`linux`容器环境，比`wsl`省心一丢丢。

1. 配置多卡训练（linux下使用的命令，这里单卡，没用到，忽略。如果你有多卡才需要配置）：
    ```bash
    export MKL_THREADING_LAYER=GNU
    CUDA_VISIBLE_DEVICES=0,1,2,3 
    NPROC_PER_NODE=4
    ```
    
2. 使用 `ms-swift` 进行微调（自行根据官方文档安装`ms-swift环境` 或 使用上面说的`Dev Container`进入`ms-swift环境`，这里的`model_id_or_path`需要指定你自己下载的基座模型路径，`dataset`可以指定线上公开的数据集或者根据[swift文档-自定义数据集](https://swift.readthedocs.io/zh-cn/latest/Instruction/%E8%87%AA%E5%AE%9A%E4%B9%89%E4%B8%8E%E6%8B%93%E5%B1%95.html#%E8%87%AA%E5%AE%9A%E4%B9%89%E6%95%B0%E6%8D%AE%E9%9B%86)准备自己的`jsonl`/`csv`格式数据集）：
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

3. （可选，加载模型耗时挺大）使用 `ms-swift` 进行推理（自行根据官方文档安装ms-swift环境）：
    ```bash
    CUDA_VISIBLE_DEVICES=0 swift infer --ckpt_dir output/qwen1half-7b-chat/v2-20240910-185715/checkpoint-16461
    ```

---

### 合并 LoRA 并进行量化

1. 合并 LoRA 并导出模型：
    ```bash
    CUDA_VISIBLE_DEVICES=0 swift export --ckpt_dir output/qwen1half-7b-chat/v2-20240910-185715/checkpoint-16461 --merge_lora true
    ```

2. （可选，量化耗时挺大（3090单卡-rouzhiba微调模型量化-大概60分钟））进行 int4 量化（可选择用llama.cpp转为.gguf后再量化）：
    ```bash
    CUDA_VISIBLE_DEVICES=0 swift export --ckpt_dir output/qwen1half-7b-chat/v2-20240910-185715/checkpoint-16461 --quant_bits 4 --quant_method awq --merge_lora true
    ```

---

### 编译 llama.cpp（用于将模型转为ollama支持的`gguf`格式）

1. 下载并解压 w64devkit（Windows 环境下编译 llama.cpp 必需工具）：
    - 下载地址：https://github.com/skeeto/w64devkit/releases
    - 推荐使用 `1.2.0 版本`，避免安全软件报毒问题。

2. 克隆 `llama.cpp` 仓库并进行编译：
    ```bash
    git clone https://github.com/ggerganov/llama.cpp
    cd llama.cpp
    # make操作需要用w64devkit进行
    make -j
    ```

3. 创建并激活 llama.cpp的Python 环境：
    ```bash
    conda create --name llamacpp python=3.10.13 -y
    conda activate llamacpp
    pip install -r requirements.txt
    ```
---

### 转换模型至 gguf 格式（需要再上面创建的llamacpp环境中进行）

1. 转换 `safetensors` 模型为 Ollama 格式（15:36 -> 15:38，耗时2分钟，模型大小：15.8G）：
    ```bash
    python convert_hf_to_gguf.py D:/Space/PRO/ai/ms-swift-train/output/qwen1half-7b-chat/v2-20240910-185715/checkpoint-16461-merged --outtype f16
    ```
2. （可选）进行 int4 量化（15:40 -> 15:41，将f16转int4，耗时1分钟，模型大小：4.2G）：
    ```bash
    ./llama-quantize D:/Space/PRO/ai/ms-swift-train/output/qwen1half-7b-chat/v2-20240910-185715/checkpoint-16461-merged/Checkpoint-16461-Merged-7.7B-F16.gguf D:/Space/PRO/ai/ms-swift-train/output/qwen1half-7b-chat/v2-20240910-185715/checkpoint-16461-merged/Checkpoint-16461-Merged-7.7B-q4_0.gguf q4_0
    ```

---

### 构建Ollama模型并推理+推送

1. 编写 `Modelfile` 文件：
    ```text
    FROM D:/Space/PRO/ai/ms-swift-train/output/qwen1half-7b-chat/v2-20240910-185715/checkpoint-16461-merged/Checkpoint-16461-Merged-7.7B-q4_0.gguf
    ```

2. 创建 Ollama 模型：
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
- 量化为 `int4` 可大幅降低模型大小，适用于低资源设备推理。
- 使用 `ollama` 进行模型创建和推理可以方便模型在不同平台上的部署。

---

### 相关截图
![image.png](https://ollama.com/assets/samge/qwen1half-7b-chat-ruozhiba/03031f9d-2e9b-429e-952c-db13a8dd067b)
