from locust import HttpUser, task, between
import json

class JWTUser(HttpUser):

    wait_time = between(0.5,1)

    @task
    def login(self):

        with self.client.post(
            "/login",
            catch_response=True,
            timeout=30
        ) as response:

            try:

                if response.status_code != 200:

                    response.failure(
                        f"HTTP {response.status_code}"
                    )
                    return


                data=response.json()


                if "algorithm" not in data:

                    response.failure(
                        "algorithm missing"
                    )
                    return


                if "reward" not in data:

                    response.failure(
                        "reward missing"
                    )
                    return


                response.success()


            except Exception as e:

                response.failure(
                    str(e)
                )

                