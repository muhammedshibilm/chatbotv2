from pydantic import BaseModel

class Contact(BaseModel):
    name: str
    medicalcon: str
    preferdate: str
    doctor: str