import pandas as pd
import numpy as np
import pathlib
import os
import json
import stat


class ExprMgr():
    def __init__(self, dataset, lambd):
        self.dataset = dataset
        self.lambd = lambd

        self.name = self.get_name()
        self.expr_dir = pathlib.Path("out") / pathlib.Path(self.name)
        self.script_dir = pathlib.Path('.')
        self.out_dir = self.expr_dir / "out"
        self.summary_csv = self.expr_dir / "summary.csv"

        if (not self.expr_dir.exists()):
            os.mkdir(self.expr_dir)
        if (not self.out_dir.exists()):
            os.mkdir(self.out_dir)
        if (not self.script_dir.exists()):
            os.mkdir(self.script_dir)

    def get_data_frame(self, reload=False):
        if reload == False and self.summary_csv.exists():
            df = pd.read_csv(self.summary_csv, index_col=0)
            return df
        else:
            df = pd.DataFrame(columns=['dataset', 'alg', 'lambd', 'eta', 'M', 'K', 'T',
                                       'local_batch',  'seed', 'record_intvl', 'final_loss', 'best_loss'])
            df.index.name = 'runid'

            for config_file in [x for x in self.out_dir.iterdir() if x.suffix == '.json']:
                runid = config_file.stem

                if (self.out_dir / (runid + ".csv")).exists():
                    with open(config_file, 'r') as f:
                        args = json.load(f)
                        if args['dataset'] != self.dataset or args['lambd'] != self.lambd:
                            print("ERROR")

                        if 'n_workers' in args:
                            args['M'] = args['n_workers']
                            del args['n_workers']

                        if 'sync_intvl' in args:
                            args['K'] = args['sync_intvl']
                            del args['sync_intvl']

                        if 'max_iters' in args:
                            args['T'] = args['max_iters']
                            del args['max_iters']

                        seq = self.load_runid(runid)
                        args['final_loss'] = seq.iloc[-1]
                        args['best_loss'] = min(seq.loc[seq.index%512==0])
                        df.loc[runid] = args
            df.to_csv(self.summary_csv)
            return df

    def get_name(self):
        return self.dataset + "_" + "{:.0e}".format(self.lambd)

    def load_runid(self, runid):
        return pd.read_csv(self.out_dir / pathlib.Path(f'{runid}.csv'), index_col=0, squeeze=True)

    def locate_job_script(self, jobid):
        return self.script_dir / (self.name + f"_{jobid}.py")