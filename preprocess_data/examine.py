from preprocessing import ExploringData, PreprocessingData
import os

if __name__ == "__main__":
    benchmark_folder = r"D:\TYNGOC\CNTT\A_Code\python\zhphoneme\preprocess_data\clue_benchmark"
    output_folder = "output"
    os.makedirs(output_folder, exist_ok=True)
    
    # File data
    input_folder = os.path.join(benchmark_folder, "tnews_public")
    train_file = os.path.join(input_folder, "train.json")
    dev_file = os.path.join(input_folder, "dev.json")
    test_file = os.path.join(input_folder, "test.json")
    
    explorer = ExploringData()
    
    # Lần lượt cho train, dev, test
    explorer.run_analysis(
        input_file=train_file,
        output_dir=os.path.join(output_folder, "tnews\\eda\\train")
    )
    
    processor = PreprocessingData(
        report_dir="output\\tnews\\eda\\test",
        output_dir="output\\tnews\\processing\\test"
    )
    # Lần lượt cho train, dev, test
    processor.run_process(
        input_file=train_file,
        output="train"
    )