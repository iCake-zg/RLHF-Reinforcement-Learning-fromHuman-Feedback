






def set_train_parameters(model:object,tarin_name:str):
    '''
    """设置训练参数"""
    Args:
        Input: 
            model(object)
            tarin_name(str): layers need to train

        Return: 
            PASS
    '''
    total_params = 0
    train_params = 0
    for name,param in model.named_parameters():
        total_params += param.numel()   
        if name.startswith(tarin_name):
            param.requires_grad = True
            train_params += param.numel()
        else:
            param.requires_grad = False
    print(f"总参数量: {total_params}, 训练参数量: {train_params}",f"可训练参数占比: {100 * train_params / total_params:.2f}%")

    pass