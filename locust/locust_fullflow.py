from locust import HttpUser, task, between


class JWTUser(HttpUser):

    wait_time = between(1,2)

    @task
    def fullflow(self):

        #################################
        # LOGIN
        #################################

        login = self.client.post(
            "/login",
            name="/login"
        )

        if login.status_code != 200:
            return

        try:

            data = login.json()

            token = data["token"]

        except:
            return

        #################################
        # VERIFY
        #################################

        self.client.post(

            "/verify",

            headers={
                "Authorization":
                f"Bearer {token}"
            },

            name="/verify"
        )