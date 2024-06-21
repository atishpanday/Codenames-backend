from app import app


def lambda_handler(event, context):
    from mangum import Mangum

    handler = Mangum(app)
    return handler(event, context)
