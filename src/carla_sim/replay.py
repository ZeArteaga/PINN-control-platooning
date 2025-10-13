import hydra
from omegaconf import DictConfig, OmegaConf
OmegaConf.register_new_resolver("eval", lambda expr: eval(expr)) #allows eval in config file

import carla

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    client = carla.Client(cfg.host, cfg.port)
    client.set_timeout(60.0)
    client.set_replayer_time_factor(cfg.replay.time_factor)

    """ # set to ignore the hero vehicles or not
    client.set_replayer_ignore_hero(args.ignore_hero)
    # set to ignore the spectator camera or not
    client.set_replayer_ignore_spectator(not args.move_spectator)"""

    # replay the session, camera=0 for free roam
    print(client.replay_file(name=cfg.replay.path,
                              time_start=cfg.replay.t_start, duration=0,
                             follow_id=cfg.replay.follow_id, replay_sensors=False))
    
if __name__ == "__main__":
    main()