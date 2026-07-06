from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.database import get_db
from app.services.resume_parser import ResumeParsingError, parse_resume, parse_resume_text
from app.services.skill_extractor import extractor
from app.services.storage import get_or_create_resume

router = APIRouter(
    prefix="/resume",
    tags=["Resume"],
)


class ResumeTextRequest(BaseModel):
    resume_text: str = Field(..., min_length=1, description="Plain text pasted from a resume")
    filename: str = Field(default="pasted_resume.txt", description="Optional label for the saved resume")


def _add_skills_and_save(
    db: Session,
    parsed_data: dict,
    filename: str,
) -> dict:
    skills_data = extractor.extract(parsed_data["raw_text"])
    parsed_data.update(skills_data)

    get_or_create_resume(
        db=db,
        raw_text=parsed_data["raw_text"],
        extracted_skills=skills_data.get("skills", []),
        filename=filename,
    )
    return parsed_data


@router.post("/parse", status_code=status.HTTP_200_OK)
async def parse_resume_endpoint(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """
    Upload a resume PDF, clean its raw text, segment it into structured
    sections, and extract targeted technical skills with alias resolution.

    Returns structured JSON with raw_text, sections, skills list, and skill_count.
    """
    if not file.filename.lower().endswith(".pdf") and file.content_type != "application/pdf":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file format. Only PDF files are supported.",
        )

    try:
        file_bytes = await file.read()
        if not file_bytes:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="The uploaded PDF file is empty.",
            )

        parsed_data = parse_resume(file_bytes)
        return _add_skills_and_save(
            db=db,
            parsed_data=parsed_data,
            filename=file.filename or "resume.pdf",
        )

    except HTTPException as e:
        raise e
    except ResumeParsingError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred during resume parsing: {str(e)}",
        )


@router.post("/parse-text", status_code=status.HTTP_200_OK)
async def parse_resume_text_endpoint(
    request: ResumeTextRequest,
    db: Session = Depends(get_db),
):
    """
    Parse pasted resume text, extract skills, segment sections, and persist it.

    Returns the same response shape as the PDF parser so the frontend can use
    upload and paste flows interchangeably.
    """
    try:
        parsed_data = parse_resume_text(request.resume_text)
        return _add_skills_and_save(
            db=db,
            parsed_data=parsed_data,
            filename=request.filename or "pasted_resume.txt",
        )

    except ResumeParsingError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred during resume parsing: {str(e)}",
        )
