from django.conf import settings


def fastapi_url(request):
    return {"FASTAPI_URL": settings.FASTAPI_URL}
