from pydantic import BaseModel
from typing import Optional, Union

class AmesInput(BaseModel):
    Gr_Liv_Area: int # Gr.Liv.Area
    Garage_Area: Optional[float] = None
    Total_Bsmt_SF: Optional[float] = None
    Year_Built: int
    Year_Remod_Add: int
    Full_Bath: int
    Fireplaces: int
    TotRms_AbvGrd: int # TotRms.AbvGrd
    Lot_Area: int
    Garage_Cars: Optional[float] = None
    MS_Zoning: str # MS.Zoning
    Neighborhood: str
    House_Style: str # House.Style
    Exter_Qual: str # Exter.Qual
    Kitchen_Qual: str # Kitchen.Qual