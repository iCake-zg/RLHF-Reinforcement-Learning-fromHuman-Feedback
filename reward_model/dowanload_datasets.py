




from  datasets import load_dataset



def download_datasets(datasets_name,dst_path="../configs/datasets/"):
    """
    下载和加载数据集
    """
    datasets = load_dataset(datasets_name,split="train_prefs",
                        cache_dir=dst_path)
    print(f"数据集 {datasets_name} 下载完成，保存到 {dst_path}")

    return datasets

def check_dataset_info(dataset):
    """
    检查数据集信息
    """
    print("数据集名称:", dataset.info.description)
    print("数据集大小:", dataset.num_rows)
    print("数据集列名:", dataset.column_names)
    print("数据集示例:", dataset[0])

    
if __name__ == "__main__":
    datasets = download_datasets("HuggingFaceH4/ultrafeedback_binarized")
    check_dataset_info(datasets)


