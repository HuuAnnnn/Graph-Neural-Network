from tqdm import tqdm
from time import sleep


for _ in range(100):
    p_bar = tqdm(100)
    for i in range(100):
        p_bar.update(1)
        sleep(0.02)
    p_bar.close()