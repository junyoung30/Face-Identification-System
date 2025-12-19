from fastapi import APIRouter, HTTPException
from app.services.gallery_service import list_persons, delete_person



router = APIRouter()


@router.get("/gallery")
def api_list_persons():
    return {
        "persons": list_persons()
    }

@router.delete("/gallery/{person_id}")
def api_delete_person(
    person_id: str
):
    success = delete_person(person_id)
    
    if not success:
        raise HTTPException(status_code=404, detail="Person not found")
        
    return {
        "status": "ok",
        "person_id": person_id
    }