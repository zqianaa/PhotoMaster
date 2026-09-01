import tarfile
import io
import os
import pickle
from PIL import Image
from PIL import ImageOps
import time

from accelerate.logging import get_logger

logger = get_logger(__name__)

class IndexedTarReader:
    def __init__(self, tar_path, index_path=None):
        self.tar_path = tar_path
        self.index_path = index_path or tar_path.replace('.tar', "_index.pkl" ) 
        self.index = self._load_or_build_index()

    def _load_or_build_index(self):
        if os.path.exists(self.index_path):
            with open(self.index_path, "rb") as f:
                # logger.info("✅ 加载索引缓存")
                return pickle.load(f)

        logger.info("🔍 正在构建索引...")
        index = {}
        with tarfile.open(self.tar_path, "r") as tar:
            for member in tar.getmembers():
                if member.isfile():
                    index[member.name] = member
        with open(self.index_path, "wb") as f:
            pickle.dump(index, f)
        logger.info(f"✅ 索引构建完成，共索引 {len(index)} 个文件")
        return index

    def read_image(self, filename):
        member = self.index.get(filename)
        if member is None:
            raise FileNotFoundError(f"文件 {filename} 不在 tar 包中")
        
        
        with tarfile.open(self.tar_path, "r") as tar:
            # member = self.index.get(filename)
            # member = tar.getmember(filename)
            fileobj = tar.extractfile(member)
            if fileobj is None:
                raise IOError(f"无法提取文件 {filename}")
            image = Image.open(fileobj)
            image = ImageOps.exif_transpose(image).convert("RGB")
            return image
    
    

# 示例用法
if __name__ == "__main__":
    
    import yaml
    import warnings
    import logging
    import tqdm
    import accelerate
    accelerator = accelerate.Accelerator()
    warnings.filterwarnings('ignore')
    
    logging.basicConfig(level=logging.INFO)
    
    reader = IndexedTarReader("xxxx/00000000.tar")
    
    for i in tqdm.tqdm(range(100)):
        img = reader.read_image("action/action/RHnihLPgfZo_segment_0_frame_58.jpg")
    # img.show()  # 或用于模型处理
    img.save('abc.jpg')

