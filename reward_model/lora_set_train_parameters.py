



from peft import LoraConfig, get_peft_model, PeftModel

def lora_set_train_parameters(model):

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["c_attn","c_proj"],
    )
    model = get_peft_model(model, lora_config)
    print("LoRA 参数量:", model.print_trainable_parameters())
    return model


