import os 
import time
import json
import torch
import atexit
import joblib
import warnings
import numpy as np

from NeuroPlan.utils import statistics_scalar


color_to_num = dict(
    gray = 30,
    red = 31,
    green = 32,
    yellow = 33,
    blue = 34,
    magenta = 35,
    cyan = 36,
    white = 37,
    crimson = 38
)


def colorize(string, color, bold=False, highlight=False): 
    attr = []
    num = color_to_num[color]
    if highlight: 
        num += 10
    attr.append(str(num))
    if bold: 
        attr.append('1')
    return '\x1b[%sm%s\x1b[0m' % (';'.join(attr), string)


def is_serializable(obj): 
    try: 
        json.dumps(obj)
        return True
    except: 
        return False


def convert_json(obj): 
    if is_serializable(obj): 
        return obj
    else: 
        if isinstance(obj, dict): 
            return {
                convert_json(k): convert_json(v) 
                for k, v in obj.items()
            }
        elif isinstance(obj, tuple): 
            return (convert_json(x) for x in obj)

        elif isinstance(obj, list): 
            return [convert_json(x) for x in obj]
        
        elif hasattr(obj, '__name__') and not ('lambda' in obj.__name__): 
            return convert_json(obj.__name__)
        
        elif hasattr(obj, '__dict__') and obj.__dict__: 
            obj_dict = {
                convert_json(k): convert_json(v)
                for k, v in obj.__dict__.items()
            }
            return {str(obj): obj_dict}
        
        return str(obj)


class Logger(): 
    def __init__(self, output_dir=None, output_name='progress.txt', exp_name=None):
        if True: 
            self.output_dir = output_dir or './tmp/experiments/{}'.format(int(time.time()))
            if os.path.exists(self.output_dir): 
                print('Warning: log dir {} already exists! Storing info there anyway.'.format(self.output_dir))
            else: 
                os.makedirs(self.output_dir)
            self.output_file = open(os.path.join(self.output_dir, output_name), 'a')
            atexit.register(self.output_file.close)
            print(colorize('Logging data to {}'.format(self.output_file.name), 'green', bold=True))

        self.first_row = True
        self.log_headers = []
        self.log_current_row = {}
        self.exp_name = exp_name
    
    def log(self, msg, color='green'): 
        if True: 
            print(colorize(msg, color, bold=True))
    
    def log_tabular(self, key, value): 
        if self.first_row: 
            self.log_headers.append(key)
        else: 
            assert key in self.log_headers, 'Trying to introduce a new key {} that not include in the first iteration'.format(key)
        assert key not in self.log_current_row, 'Already set {} this iteration. Maybe you forgot to call dump_tabular().'.format(key)
        self.log_current_row[key] = value

    def save_config(self, config): 
        config_json = convert_json(config)
        if self.exp_name is not None:
            config_json['exp_name'] = self.exp_name
        if True: 
            output = json.dumps(config_json, separators=(',',':\t'), indent=4, sort_keys=True)
            print(colorize('Saving config: \n', color='cyan', bold=True))
            print(output)
            with open(os.path.join(self.output_dir, 'config.json'), 'w') as f: 
                f.write(output)
    
    def save_state(self, state_dict, itr=None): 
        if True: 
            file_name = 'vars.pkl' if itr is None else 'vars_{}.pkl'.format(itr)
            try: 
                joblib.dump(state_dict, os.path.join(self.output_dir, file_name))
            except:
                self.log('Warning: could not pickle state_dict.', color='red')
            if hasattr(self, 'saver_elements'): 
                self._simple_save(itr)
        
    def setup_saver(self, to_be_saved):
        self.saver_elements = to_be_saved

    def _simple_save(self, itr=None): 
        if True: 
            assert hasattr(self, 'saver_elements'), 'Please setup saving with self.setup_pytorch_saver'
            file_path = 'save'
            file_path = os.path.join(self.output_dir, file_path)
            file_name = 'model'+('_{}'.format(itr) if itr is not None else '')+'.pt'
            file_name = os.path.join(file_path, file_name)
            os.makedirs(file_path, exist_ok=True)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                torch.save(self.saver_elements, file_name)

    def dump_tabular(self): 
        if True:
            vals = []
            key_lens = [len(key) for key in self.log_headers]
            max_key_len = max(15, max(key_lens))
            keystr = '%'+'%d'%max_key_len
            fmt = '| '+keystr+'s | %15s |'
            n_slashes = 22+max_key_len
            print('-'*n_slashes)
            for key in self.log_headers:
                val = self.log_current_row.get(key, '')
                valstr = '%8.3g'%val if hasattr(val, '__float__') else val
                print(fmt%(key, valstr))
                vals.append(val)
            print('-'*n_slashes, flush=True)
            if self.output_file is not None:
                if self.first_row:
                    self.output_file.write('\t'.join(self.log_headers)+'\n')
                self.output_file.write('\t'.join(map(str,vals))+'\n')
                self.output_file.flush()
        self.log_current_row.clear()
        self.first_row = False

   
class EpochLogger(Logger): 
    def __init__(self, output_dir=None, output_name='progress.txt', exp_name=None):
        super().__init__(output_dir, output_name, exp_name)
        self.epoch_dict = dict()

    def store(self, **kwargs): 
        for k, v in kwargs.items(): 
            if not (k in self.epoch_dict.keys()): 
                self.epoch_dict[k] = []
            self.epoch_dict[k].append(v)

    def log_tabular(self, key, value=None, with_min_max=False, average_only=False):
        if value is not None: 
            super().log_tabular(key, value)
        else: 
            v = self.epoch_dict[key]
            values = np.concatenate(v) if isinstance(v[0], np.ndarray) and len(v[0].shape)>0 else v
            stats = statistics_scalar(values, with_min_max)

            super().log_tabular(key if average_only else 'Average'+key, stats[0])
            if not(average_only):
                super().log_tabular('Std'+key, stats[1])
            if with_min_max:
                super().log_tabular('Min'+key, stats[2])
                super().log_tabular('Max'+key, stats[3])
        self.epoch_dict[key] = []
