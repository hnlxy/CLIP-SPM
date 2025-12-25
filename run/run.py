import sys
sys.path.append('.')
from utils.config import Config
from main_run import Learner


def main():
    cfg = Config(load=True)

    params = {
        'mid_dim_vision': 0.5, 'mid_dim_text': 1.5, 'negative_slope': 0.0025, 'alpha': 0.2, 'consist_1': 0.54, 'text_dis': 0.07, 'motion_alpha': 1,  # hmdb/ssv2cmn
        # 'mid_dim_vision': 2, 'mid_dim_text': 2, 'negative_slope': 0.054, 'alpha': 0.228, 'consist_1': 0.221, 'text_dis': 0.0295, 'motion_alpha': 1,  # ucf
        # 'mid_dim_vision': 0.5, 'mid_dim_text': 1.5, 'negative_slope': 0.042, 'alpha': 0.86, 'consist_1': 0.62, 'text_dis': 0.041, 'motion_alpha': 1,  # k100
        # 'mid_dim_vision': 0.5, 'mid_dim_text': 0.75, 'negative_slope': 0.052, 'alpha': 0.353, 'consist_1': 0.579, 'text_dis': 0.49, 'motion_alpha': 1,  # ssv2otam
        }

    cfg.params = params

    # print(cfg)
    learner = Learner(cfg)
    learner.run()

if __name__ == "__main__":
    main()