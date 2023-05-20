from pathlib import Path
import requests
import zipfile
import os

def download_data(source: str, 
                  destination: str,
                  remove_source: bool = True,
                  count:int = 0) -> Path:
    """Downloads a zipped dataset from source and unzips to destination.
    Args:
        source (str): A link to a zipped file containing data.
        destination (str): A target directory to unzip data to.
        remove_source (bool): Whether to remove the source after downloading and extracting.
    
    Returns:
        pathlib.Path to downloaded data.
    
    Example usage:
        download_data(source="https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip",
                      destination="pizza_steak_sushi")
    """
    # Setup path to data folder
    data_path = Path("data/")
    path = data_path / destination

    # If the image folder doesn't exist, download it and prepare it... 
    # if path.is_dir():
    #     print(f"[INFO] {path} directory exists, skipping download.")
    # else:
        # print(f"[INFO] Did not find {path} directory, creating one...")
    path.mkdir(parents=True, exist_ok=True)
    
    # Download pizza, steak, sushi data
    # target_file = Path(source).name
    target_file = f"output_video{count}.mp4"
    with open(data_path / target_file, "wb") as f:
        request = requests.get(source)
        print(f"[INFO] Downloading {target_file} from {source}...")
        f.write(request.content)

        # # Unzip pizza, steak, sushi data
        # try:
        #     with zipfile.ZipFile(data_path / target_file, "r") as zip_ref:
        #         print(f"[INFO] Unzipping {target_file} data...") 
        #         zip_ref.extractall(path)
        #         # Remove .zip file
        #     if remove_source:
        #         os.remove(data_path / target_file)
        # except:
        #     print("NOT ZIP FILE, PROCESS DONE")
    
    return path