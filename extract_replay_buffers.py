import pickle
from stable_baselines3.common.save_util import load_from_zip_file

if __name__ == '__main__':
    pretrained_path = 'C:/Users/matan/Documents/SAC_MER/experiments__2021_01_13__13_43/'
    replay_mems_path = pretrained_path + 'SAC_no_reset/buffer_50000/final_only/'

    replay_buffers = []
    for i in range(80):
        zf_name = replay_mems_path + f'/model_{i}.zip'
        data, params, pytorch_variables = load_from_zip_file(zf_name)
        replay_mem = data['replay_buffer']
        replay_buffers.append(replay_mem)

    pickle.dump(replay_buffers, open(pretrained_path + 'replay_buffers.pkl', 'wb'))
    db = 1
