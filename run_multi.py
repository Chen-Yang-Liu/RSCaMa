import os
import time
import json
import pynvml
from threading import Thread

def execute_command(command):
    os.system(command)

if __name__ == '__main__':
    start_time = time.strftime("%m-%d  %H : %M : %S", time.localtime(time.time()))
    print(start_time)

    commands = [
        # "CUDA_VISIBLE_DEVICES=1 python train_CC.py --sys linux --decoder_type mamba",
        # "CUDA_VISIBLE_DEVICES=2 python train_CC.py --sys linux --decoder_type mamba",
        # "CUDA_VISIBLE_DEVICES=0 python train_CC.py --sys linux --decoder_type mamba",
        # "CUDA_VISIBLE_DEVICES=0 python train_CC.py --sys linux --decoder_type gpt",
        # "CUDA_VISIBLE_DEVICES=3 python train_CC.py --sys linux --decoder_type gpt",
        # "CUDA_VISIBLE_DEVICES=0 python train_CC.py --sys linux --decoder_type gpt",
        # "CUDA_VISIBLE_DEVICES=2 python train_CC.py --sys linux --decoder_type mamba --decoder_n_layers 3",
        # "CUDA_VISIBLE_DEVICES=6 python train_CC.py --sys linux --decoder_type mamba --decoder_n_layers 3",
        # "CUDA_VISIBLE_DEVICES=7 python train_CC.py --sys linux --decoder_type mamba --decoder_n_layers 3",
        # "CUDA_VISIBLE_DEVICES=5 python train_CC.py --sys linux --decoder_type gpt",
        # "CUDA_VISIBLE_DEVICES=6 python train_CC.py --sys linux --decoder_type gpt",
        # "CUDA_VISIBLE_DEVICES=7 python train_CC.py --sys linux --decoder_type gpt",
        "CUDA_VISIBLE_DEVICES=0 python train_CC.py --sys linux",
        # "CUDA_VISIBLE_DEVICES=4 python train_CC.py --sys linux",
        # "CUDA_VISIBLE_DEVICES=5 python train_CC.py --sys linux",
        # "CUDA_VISIBLE_DEVICES=6 python train_CC.py --sys linux",
        # "CUDA_VISIBLE_DEVICES=7 python train_CC.py --sys linux",
        # "CUDA_VISIBLE_DEVICES=7 python train_CC.py --sys linux",
    ]

    threads = []
    k = 0
    while (k < len(commands)):
    # for command in commands:
        pynvml.nvmlInit()
        command = commands[k]
        os.environ["CUDA_VISIBLE_DEVICES"] = command.split("CUDA_VISIBLE_DEVICES=")[1].split(" ")[0]
        print('CUDA_VISIBLE_DEVICES:', os.environ["CUDA_VISIBLE_DEVICES"])
        handle = pynvml.nvmlDeviceGetHandleByIndex(int(os.environ["CUDA_VISIBLE_DEVICES"]))  # 这里的0是GPU id
        # handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # 这里的0是GPU id
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        free = meminfo.free / 1024 ** 2  # 9 * 1024
        print(free)

        if free > (13 * 1024):
            command = commands[k]
            k=k+1
            thread = Thread(target=execute_command, args=(command,))
            threads.append(thread)
            thread.start()
            time.sleep(73)
            print("One has been executed.")
        else:
            print(time.strftime("%m-%d  %H : %M : %S", time.localtime(time.time())))
            time.sleep(500)

    # 等待所有线程结束
    for thread in threads:
        thread.join()

    print("All commands have been executed.")
