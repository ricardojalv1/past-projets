"""
Author:      brendan.mcgarry@cn.ca
Date:        2020-08-08
Updated:     2020-10-05
Usage:       python kql-orchestration.py -e <env> -p <path/params.json>
             [-ak <app_key>] [-sk <storage_key>] [-r <root_execution_path>]
Description: Orchestrates the execute of KQL queries and scripts.
"""
import os
import sys
import json
import argparse

from kusto.engine import KustoEngine

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--env', action='store', dest='env',
                        help="The ADX environment; determines which env file to use.")
                        
    parser.add_argument('-p', '--params', action='store', dest='param_file',
                        help="The parameters file containing the KQL queries or files to execute.",
                        required=True)
    parser.add_argument('-ak', '--app_key', action='store', dest='app_key',
                        help="The app key for the service principal.", default=None)
                
    parser.add_argument('-sk', '--storage_key', action='store', dest='storage_key',
                        help="The storage key for the Azure storage account.", default=None)

    parser.add_argument('-r', '--root_path', action='store', dest='root_path',
                        help="The root path for the script to execute from.", default='')

    parser.add_argument('-cluster', action='store', dest='cluster_name',
                        help="The cluster name used to connect to Kusto", default='')

    parser.add_argument('-id', '--app_id',action='store', dest='app_id',
                        help="The app id used to connect to Kusto", default='')

    parser.add_argument('-tid', '--tenant_id', action='store', dest='tenant_id',
                        help="The tenant id used to connect to Kusto", default='')

    parser.add_argument('-db', '--dbname', action='store', dest='dbname',
                        help="The dbname used to connect to Kusto", default='')

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    
    kusto_engine = KustoEngine(env=args.env,
                            app_key=args.app_key,
                            storage_key=args.storage_key,
                            root_path=args.root_path,
                            cluster_name=args.cluster_name,
                            app_id=args.app_id,
                            tenant_id=args.tenant_id)

    kusto_engine.execute_params(args.param_file, args.dbname)
