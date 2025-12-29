# 
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

from pydantic import BaseModel, Field

class User(BaseModel):
    """
    Muna user profile.

    Members:
        username (str): Username.
        email (str): User email address.
        created (str): Date created.
        name (str): User display name.
        avatar (str): User avatar URL.
        bio (str): User bio.
        website (str): User website.
        github (str): User GitHub handle.
    """
    username: str = Field(description="Username.")
    email: str | None = Field(default=None, description="User email address.")
    created: str | None = Field(default=None, description="Date created.")
    name: str | None = Field(default=None, description="User display name.")
    avatar: str | None = Field(default=None, description="User avatar URL.")
    bio: str | None = Field(default=None, description="User bio.")
    website: str | None = Field(default=None, description="User website.")
    github: str | None = Field(default=None, description="User GitHub handle.")