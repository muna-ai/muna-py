# 
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

from ..client import MunaClient, MunaAPIError
from ..types import User

class UserService:

    def __init__(self, client: MunaClient) -> None:
        self.client = client

    def retrieve(self) -> User:
        """
        Retrieve the current user.

        Returns:
            User: User.
        """
        try:
            return self.client.request(
                method="GET",
                path="/users",
                response_type=User
            )
        except MunaAPIError as error:
            if error.status_code == 401:
                return None
            raise