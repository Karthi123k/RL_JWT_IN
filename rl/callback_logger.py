from stable_baselines3.common.callbacks import BaseCallback


class MetricCallback(
BaseCallback
):


    def __init__(
        self,
        metric_logger
    ):

        super().__init__()

        self.metric_logger=metric_logger



    def _on_step(self):


        info=self.locals[
            "infos"
        ][0]


        reward=float(

            self.locals[
                "rewards"
            ][0]

        )


        self.metric_logger.log(

            self.num_timesteps,

            reward,

            info.get(
                "latency",0
            ),

            info.get(
                "throughput",0
            ),

            info.get(
                "cpu",0
            ),

            info.get(
                "memory",0
            ),

            info.get(
                "security_bits",0
            ),

            info.get(
                "switch_time",0
            ),

            info.get(
                "service_interrupt",0
            ),

            info.get(
                "jwt_continuity",0
            )

        )


        return True