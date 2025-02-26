class InjuryCriteria:

    def __init__(self):
        self.criteria = {
            "rib": {
                "ultimate_strain": {"value": [1.25, 4.25], "unit": "MPa", "source": "Source1"}, 
                "ultimate_plastic_strain": {"value": [0.02], "unit": "", "source": "Source2"}
            },
            "lung": {
                "pressure": {"value": 30, "unit": "kPa", "source": "Source3"},
                "strain": {"value": 0.15, "unit": "", "source": "Source4"}
            }
        }


