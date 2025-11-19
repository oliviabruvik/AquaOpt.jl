const ACTION_SHORT_LABELS = Dict(
    NoTreatment => "",
    MechanicalTreatment => "M",
    ChemicalTreatment => "C",
    ThermalTreatment => "Th",
)

action_short_label(a) = get(ACTION_SHORT_LABELS, a, "")
