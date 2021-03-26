import typing as t
import typing_extensions as te

from pydantic import BaseModel, Field, ConstrainedInt, PositiveInt, PositiveFloat, ConstrainedFloat


# class ModelInput(BaseModel):
#     YrSold: int
#     YearBuilt: int
#     YearRemodAdd: int
#     GarageYrBlt: int
#     LotArea: int
#     Neighborhood: str
#     HouseStyle: str


mnthLiteral = te.Literal[
    "1", "2","3","4","5","6","7","8","9","10","11", "12"
]
hrLiteral = te.Literal[
    "1", "2","3","4","5","6","7","8","9","10","11", "12","13", "14","15", "16",
    "17", "18","10", "20","21", "22","23", "24"
]

generalLiteral = te.Literal[
    "0","1"
]

weekdayLiteral = te.Literal[
    "1", "2","3","4","5","6","0"
]

weathersitLiteral = te.Literal[
    "1", "2","3","4"
]

yrLiteral = te.Literal[
    '1', '2','3'
]

# class ModelInput(BaseModel):
#     YrSold: PositiveInt
#     YearBuilt: PositiveInt
#     YearRemodAdd: PositiveInt
#     GarageYrBlt: PositiveInt
#     LotArea: PositiveFloat
#     Neighborhood: NeighborhoodLiteral
#     HouseStyle: HouseStyleLiteral


class generalFloat(ConstrainedFloat):
    ge = 0.1
    le = 1

class generalInteger(ConstrainedInt):
    ge = 0
    le = 4

class generalInteger2(ConstrainedInt):
    ge = 0
    le = 1

class hrlInteger(ConstrainedInt):
    ge = 0
    le = 23

class mnthlInteger(ConstrainedInt):
    ge = 1
    le = 12

class weekdayInteger(ConstrainedInt):
    ge = 0
    le = 6


class ModelInput(BaseModel):
    yr: generalInteger
    mnth: mnthlInteger
    hr: hrlInteger
    season: generalInteger
    holiday: generalInteger2
    weekday: weekdayInteger
    workingday: generalInteger2
    weathersit: generalInteger
    temp: generalFloat
    atemp: generalFloat
    hum: generalFloat
    windspeed: generalFloat