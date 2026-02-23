"""
Reglas programáticas de negociación (sin dependencias de infraestructura).
"""

from typing import Dict, Tuple


def decidir_aceptar_programatico(
    ofrecen: Dict[str, int],
    piden: Dict[str, int],
    necesidades: Dict[str, int],
    excedentes: Dict[str, int],
) -> Tuple[bool, str]:
    """Decide si aceptar una oferta. Devuelve (aceptar, razon)."""
    if not ofrecen or not piden:
        return False, "oferta incompleta (falta ofrecen o piden)"

    me_ofrecen_lo_que_necesito = any(r in necesidades for r in ofrecen)
    me_ofrecen_oro = "oro" in ofrecen and ofrecen["oro"] > 0
    piden_solo_excedentes = all(
        r in excedentes and excedentes[r] >= c for r, c in piden.items() if c > 0
    )

    if me_ofrecen_lo_que_necesito and piden_solo_excedentes:
        return True, "ofrecen lo que necesito y piden lo que me sobra"
    if me_ofrecen_oro and piden_solo_excedentes:
        return True, "ofrecen oro a cambio de lo que me sobra"
    if not me_ofrecen_lo_que_necesito and not me_ofrecen_oro:
        return False, "no ofrecen nada de lo que necesito"
    return False, "piden recursos que no me sobran o no tengo suficientes"
