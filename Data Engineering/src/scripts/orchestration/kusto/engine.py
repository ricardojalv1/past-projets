"""
Author:      brendan.mcgarry@cn.ca
Date:        2020-08-08
Usage:       to be imported as a Python module;
             from kusto.engine import KustoEngine
Description: Creates the Kusto API query engine for a given environment.
"""
from azure.kusto.data import KustoClient, KustoConnectionStringBuilder
from azure.kusto.data.exceptions import KustoServiceError

import os
import json
from time import sleep


class KustoEngine:
    default_config_dir = "configs"
    
    def __init__(self, env, cluster_name=None, app_id=None, tenant_id=None, 
                app_key=None, storage_key=None, root_path='', async_sleep_period=1, config_folder=None): 

        self._env = env
        self._cluster_name = cluster_name
        self._app_id = app_id
        self._tenant_id = tenant_id
        self._app_key = app_key
        self._storage_key = storage_key
        self._root_path = root_path
        self._async_sleep_period = async_sleep_period

        if config_folder is None:
            self._set_default_config_path()
        else:
            self._config_folder = config_folder
        
        
        # load environment
        if self._env == "dev":
            config_path = os.path.join(self._config_folder, 'env', 'dev.json')
        elif self._env == "stg":
            config_path = os.path.join(self._config_folder, 'env', 'stg.json')
        elif self._env == "prd":
            config_path = os.path.join(self._config_folder, 'env', 'prd.json')
        else:
            config_path = self._env
        
        
        if not app_key:
            self._kusto_client = self.load_env(config_path)
        else:
            self._kusto_client = self.load_env(config_path, app_key=self._app_key)
    
    
    def load_env(self, config_path, app_key=None): 
        
        if not(config_path is None):
            with open(config_path) as f:
                configs = json.load(f)
        
        if not self._cluster_name:
            cluster_url = f'https://{configs["cluster_name"]}.kusto.windows.net'

        else:
            cluster_url = f'https://{self._cluster_name}.kusto.windows.net'

        if not self._app_id:
            app_id_param = configs['app_id']

        else:
            app_id_param = self._app_id

        if not self._tenant_id:
            tenant_id_param = configs['tenant_id']
        else:
            tenant_id_param = self._tenant_id
        
        if not self._app_key:
            app_key_param = configs['app_key']
        else:
            app_key_param = self._app_key


        kcsb = KustoConnectionStringBuilder.with_aad_application_key_authentication(
            cluster_url,
            app_id_param,
            app_key_param,
            tenant_id_param)

        kusto_client = KustoClient(kcsb)
        return kusto_client
    
    
    def execute(self, db, query):
        sleep(self._async_sleep_period)  # for preventing throtting from ADX
        #print(query)
        return self._kusto_client.execute(db, query)
    
    
    def execute_async(self, db, query):
        result = self._kusto_client.execute(db, query)
        oper_id = result.primary_results[0].rows[0].to_dict()['OperationId']
        
        def get_async_query_state(op_id):
            result    = self._kusto_client.execute(db, f".show operations {op_id}")
            kusto_row = result.primary_results[0].rows[0].to_dict()
            
            state    = kusto_row['State']
            duration = kusto_row['Duration']
            return state, duration
        
        async_state = 'InProgress'
        duration = "0:00:00.000000"
        while async_state == 'InProgress':
            print(f'    Operation {oper_id} in progress; duration {duration}...')
            async_state, duration = get_async_query_state(oper_id)
        
        print(f'    Operation {oper_id} finished execution: {async_state}.')
        
        if async_state == "Failed":
            raise ValueError(f"Error: Failed for operation {oper_id}")
        
        return self._kusto_client.execute(db, f".show operations {oper_id}")
    
    
    def execute_file(self, db, filepath, _async=False, templ_params=None):
        print('        Executing for db ' + db + ' file ' + filepath + ' with ' + str(templ_params))
        if not filepath.endswith(".keep"):
            with open(filepath) as f:
                queries = f.read().strip().split("\n\n")
            
            for query in queries:
                if templ_params:
                    query = query.format(**templ_params)
                if not _async:
                    result = self.execute(db, query)
                else:
                    result = self.execute_async(db, query)
    
    
    def execute_dir(self, db, dirpath, _async=False, templ_params=None, recursive=False):
        print('    Executing for db ' + db + ' for dirpath ' + dirpath)
        
        if not os.path.exists(dirpath):
            raise ValueError('Error: Directory path ' + dirpath + ' does not exist.')
        
        files = os.listdir(dirpath)
        dirs = []
        for file in files: # The var 'file' refers to both files and directories
            fullpath = os.path.join(dirpath, file)
            if os.path.isfile(fullpath):
                self.execute_file(db, fullpath, _async=_async, templ_params=templ_params)

            elif os.path.isdir(fullpath) and recursive:
                dirs.append(fullpath)    
        dirs.sort()
        for dir_name in dirs: 
            self.execute_dir(db, dir_name, _async=_async, templ_params=templ_params, recursive=recursive)
    
    
    def execute_params(self, params_file, dbname):
        #os.chdir("..")
        #os.chdir("a")
        print('Executing ' + params_file)
        with open(params_file) as fhandle:
            data = json.load(fhandle)
        
        for param_item in data:
            if param_item.get('templated'):
                template_params = param_item.get('template_params') or {}
                template_params = {**{'SK': self._storage_key}, **template_params}
            
            recursive = param_item.get('recursive')
            if not dbname:
                dbname = param_item['db']
            if param_item['type'] == 'file':
                path = os.path.join(self._root_path, param_item['path'])
                if os.path.isdir(path):
                    if not param_item.get('templated'):
                        self.execute_dir(dbname,
                                        path,
                                        _async=param_item.get('async'),
                                        recursive=recursive)

                    else:
                        self.execute_dir(dbname,
                                        path,
                                        _async=param_item.get('async'),
                                        templ_params=template_params,
                                        recursive=recursive)

                elif os.path.isfile(path):
                    if not param_item.get('templated'):
                        self.execute_file(dbname,
                                          path,
                                          _async=param_item.get('async'))
                    else:
                        self.execute_file(dbname,
                                          path,
                                          _async=param_item.get('async'),
                                          templ_params=template_params)
                        
            
            elif param_item['type'] == 'query':
                if not param_item.get('templated'):
                    if not param_item.get('async'):
                        self.execute(dbname, param_item['query'])
                    else:
                        self.execute_async(dbname, param_item['query'])
                else:
                    query = param_item['query'].format(**templ_params)
                    if not param_item.get('async'):
                        self.execute(dbname, query)
                    else:
                        self.execute_async(dbname, query)
    
    
    def _set_default_config_path(self):
        module_dir = os.path.dirname(__file__)
        self._config_folder = os.path.join(module_dir, self.default_config_dir)
