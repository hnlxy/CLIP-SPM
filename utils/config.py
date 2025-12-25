"""Config class for loading and keeping track of the configs."""

import os
import yaml
import json
import copy
import argparse

class Config(object):
    """
    Global config object. 
    It automatically loads from a hierarchy of config files and turns the keys to the 
    class attributes. 
    """
    def __init__(self, load=True, cfg_dict=None, cfg_level=None):
        """
        Args: 
            load (bool): whether or not yaml is needed to be loaded.
            cfg_dict (dict): dictionary of configs to be updated into the attributes
            cfg_level (int): indicating the depth level of the config
        """
        self._level = "cfg" + ("." + cfg_level if cfg_level is not None else "")
        if load:
            self.args = self._parse_args()
            print("Loading config from {}.".format(self.args.cfg_file))
            self.need_initialization = True
            cfg_base = self._initialize_cfg()
            cfg_dict = self._load_yaml(self.args)
            cfg_dict = self._merge_cfg_from_base(cfg_base, cfg_dict)
            self.cfg_dict = cfg_dict
        self._update_dict(cfg_dict)

    def _parse_args(self):
        """
        Wrapper for argument parser. 
        """
        parser = argparse.ArgumentParser(
            description="Argparser for configuring [code base name to think of] codebase"
        )
        parser.add_argument(
            "--cfg",
            dest="cfg_file",
            help="Path to the configuration file",
            default=None
        )
        parser.add_argument(
            "--init_method",
            help="Initialization method, includes TCP or shared file-system",
            default="tcp://localhost:9999",
            type=str,
        )
        parser.add_argument(
            "opts",
            help="other configurations",
            default=None,
            nargs=argparse.REMAINDER
        )
        return parser.parse_args()

    def _initialize_cfg(self):
        """
        When loading config for the first time, base config is required to be read.
        """
        if self.need_initialization:
            self.need_initialization = False
            if os.path.exists('./configs/base.yaml'):
                with open("./configs/base.yaml", 'r') as f:
                    cfg = yaml.load(f.read(), Loader=yaml.SafeLoader)
            else:
                # for compatibility to the cluster
                print('without base.yaml')
                exit(-1)
        return cfg
    
    def _load_yaml(self, args, file_name=""):
        """
        Load the specified yaml file.
        Args:
            args: parsed args by `self._parse_args`.
            file_name (str): the file name to be read from if specified.
        """
        assert args.cfg_file is not None
        if not file_name == "": # reading from base file
            with open(file_name, 'r') as f:
                cfg = yaml.load(f.read(), Loader=yaml.SafeLoader)
        else: # reading from top file
            with open(args.cfg_file, 'r') as f:
                cfg = yaml.load(f.read(), Loader=yaml.SafeLoader)

        return cfg
    
    def _merge_cfg_from_base(self, cfg_base, cfg_new, preserve_base=False):
        """
        Replace the attributes in the base config by the values in the coming config, 
        unless preserve base is set to True.
        Args:
            cfg_base (dict): the base config.
            cfg_new (dict): the coming config to be merged with the base config.
            preserve_base (bool): if true, the keys and the values in the cfg_new will 
                not replace the keys and the values in the cfg_base, if they exist in 
                cfg_base. When the keys and the values are not present in the cfg_base,
                then they are filled into the cfg_base.
        """
        for k,v in cfg_new.items():
            if k in cfg_base.keys():
                if isinstance(v, dict):
                    self._merge_cfg_from_base(cfg_base[k], v)
                else:
                    cfg_base[k] = v
            else:
                if "BASE" not in k or preserve_base:
                    cfg_base[k] = v
        return cfg_base

    
    def _update_dict(self, cfg_dict):
        """
        Set the dict to be attributes of the config recurrently.
        Args:
            cfg_dict (dict): the dictionary to be set as the attribute of the current 
                config class.
        """
        def recur(key, elem):
            if type(elem) is dict:
                return key, Config(load=False, cfg_dict=elem, cfg_level=key)
            else:
                if type(elem) is str and elem[1:3]=="e-":
                    elem = float(elem)
                return key, elem
        
        dic = dict(recur(k, v) for k, v in cfg_dict.items())
        self.__dict__.update(dic)
    
    def get_args(self):
        """
        Returns the read arguments.
        """
        return self.args
    
    def __repr__(self):
        return "{}\n".format(self.dump())
            
    def dump(self):
        return json.dumps(self.cfg_dict, indent=2)

    def deep_copy(self):
        return copy.deepcopy(self)
    
if __name__ == '__main__':
    # debug
    cfg = Config(load=True)
    print(cfg)