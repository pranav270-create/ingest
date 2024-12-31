from fastapi import Request
from fastapi.responses import JSONResponse
import traceback


async def base_middleware(request: Request, call_next):
    try:
        response = await call_next(request)
        return response
    except Exception as e:
        # Add error content and print the traceback for debugging
        error_details = str(e)
        print(f"Error in middleware: {error_details}")
        print(traceback.format_exc())
        return JSONResponse(
            content={"error": error_details},
            status_code=500
        )
