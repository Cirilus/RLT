from fastapi import HTTPException
from loguru import logger
from starlette import status
from starlette.responses import JSONResponse

from utils.errors import ErrEntityNotFound, ErrEntityConflict


def error_wrapper(func, *args, **kwargs):
    try:
        return func(*args, **kwargs)
    except ErrEntityNotFound as e:
        logger.debug(f"err = {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    except ErrEntityConflict as e:
        logger.debug(f"err = {e}")
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"err = {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )