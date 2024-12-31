import json
from fastapi import APIRouter, HTTPException
from fastapi_utils.tasks import repeat_every

from pathlib import Path
import sys
import logging

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.sql_db.database import get_async_session
from serving.routers.qdrant_flow import gather_qdrant_stats
from serving.routers.sql_flow import get_overview_stats
from serving.crud.sql_flow import generate_distribution_stats

router = APIRouter()

base_path = Path(__file__).parent.parent


@repeat_every(seconds=60 * 60)  # Run every hour
async def periodic_task():
    """Run ETL tasks every hour"""
    try:
        # Get a new session for the periodic task
        for session in get_async_session():
            # Update Qdrant stats
            stats_file = base_path / "qdrant_stats.json"
            await gather_qdrant_stats(str(stats_file))
            
            # Update distribution stats
            await generate_distribution_stats(session)
            
            # Update overview stats
            overview_stats = await get_overview_stats(session)
            overview_path = base_path / "overview_stats.json"
            with open(overview_path, 'w') as f:
                json.dump(overview_stats, f, indent=4)
                
            print("Scheduled ETL tasks completed successfully")
    except Exception as e:
        logging.error(f"Error in scheduled ETL tasks: {e}", exc_info=True)


@router.post("/trigger-update")
async def trigger_update():
    """Manually trigger the ETL tasks"""
    await periodic_task()
    return {"status": "success", "message": "ETL tasks completed"}
