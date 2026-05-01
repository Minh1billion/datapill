def build_bearer_header(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def build_basic_auth(user: str, password: str) -> tuple[str, str]:
    return (user, password)


def build_sasl_auth(mechanism: str, username: str, password: str) -> dict[str, str]:
    return {
        "sasl_mechanism": mechanism,
        "sasl_plain_username": username,
        "sasl_plain_password": password,
    }