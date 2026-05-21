import pandas as pd
import os

class Logger:

    def __init__(self,file):

        self.file=file

        os.makedirs(
            os.path.dirname(file),
            exist_ok=True
        )

        cols=[

        "episode",
        "reward",
        "latency",
        "throughput",
        "cpu",
        "memory",
        "security_bits",
        "switch_time",
        "service_interrupt",
        "jwt_continuity"

        ]

        pd.DataFrame(
            columns=cols
        ).to_csv(
            file,
            index=False
        )


    def log(
        self,
        episode,
        reward,
        latency,
        throughput,
        cpu,
        memory,
        security_bits,
        switch_time,
        service_interrupt,
        jwt_continuity
    ):

        pd.DataFrame([{

        "episode":episode,
        "reward":reward,
        "latency":latency,
        "throughput":throughput,
        "cpu":cpu,
        "memory":memory,
        "security_bits":security_bits,
        "switch_time":switch_time,
        "service_interrupt":service_interrupt,
        "jwt_continuity":jwt_continuity

        }]).to_csv(

        self.file,
        mode="a",
        header=False,
        index=False
        )