"""
Cliente para la API del juego.

Centraliza todas las llamadas HTTP al servidor fdi-pln-butler.
Usa loguru para logging en vez de print().
"""

import time
import requests
from typing import Dict, List, Optional

from loguru import logger
from config import API_BASE_URL


class APIClient:
    """Cliente para interactuar con la API del juego."""

    def __init__(self, base_url: str = None, agente: str = None):
        self.base_url = base_url or API_BASE_URL
        self.agente = agente
        self.session = requests.Session()
        if agente:
            logger.debug("Usando identificador de agente {}", agente)

    def _params(self, extra: Dict[str, str] = None) -> Optional[Dict[str, str]]:
        """Añade el parámetro `agente` cuando está configurado."""
        params: Dict[str, str] = {}
        if extra:
            params.update(extra)
        if self.agente and "agente" not in params:
            params["agente"] = self.agente
        return params or None

    # =====================================================================
    # INFORMACIÓN
    # =====================================================================

    def get_info(self) -> Optional[Dict]:
        """Obtiene información general del jugador."""
        try:
            response = self.session.get(f"{self.base_url}/info", params=self._params())
            if response.status_code == 200:
                return response.json()
            logger.warning("Error obteniendo info: status {}", response.status_code)
            return None
        except requests.RequestException as e:
            logger.error("Error de conexión (info): {}", e)
            return None

    def get_gente(self) -> List[str]:
        """Obtiene lista de personas disponibles."""
        try:
            response = self.session.get(f"{self.base_url}/gente")
            if response.status_code == 200:
                data = response.json()
                # Normalizar: la API puede devolver ["str"] o [{"nombre": ...}]
                resultado = []
                for item in data:
                    if isinstance(item, str):
                        resultado.append(item)
                    elif isinstance(item, dict):
                        # Intentar extraer el nombre del dict
                        nombre = item.get("nombre") or item.get("name") or item.get("alias") or str(item)
                        logger.debug("get_gente: item dict normalizado {} -> {}", item, nombre)
                        resultado.append(nombre)
                    else:
                        resultado.append(str(item))
                return resultado
            logger.warning("Error obteniendo gente: status {}", response.status_code)
            return []
        except requests.RequestException as e:
            logger.error("Error de conexión (gente): {}", e)
            return []

    # =====================================================================
    # ALIAS
    # =====================================================================

    def crear_alias(self, nombre: str) -> bool:
        """Crea un nuevo alias."""
        try:
            response = self.session.post(
                f"{self.base_url}/alias/{nombre}",
                params=self._params(),
            )
            if response.status_code == 200:
                logger.success("Alias '{}' creado", nombre)
                return True
            logger.warning("Error creando alias: status {}", response.status_code)
            return False
        except requests.RequestException as e:
            logger.error("Error de conexión (crear alias): {}", e)
            return False

    def eliminar_alias(self, nombre: str) -> bool:
        """Elimina un alias."""
        try:
            response = self.session.delete(
                f"{self.base_url}/alias/{nombre}",
                params=self._params(),
            )
            if response.status_code == 200:
                logger.success("Alias '{}' eliminado", nombre)
                return True
            logger.warning("Error eliminando alias: status {}", response.status_code)
            return False
        except requests.RequestException as e:
            logger.error("Error de conexión (eliminar alias): {}", e)
            return False

    # =====================================================================
    # CARTAS
    # =====================================================================

    def enviar_carta(self, remitente: str, destinatario: str,
                     asunto: str, cuerpo: str, id_carta: str = None) -> bool:
        """Envía una carta a otro jugador."""
        carta_data = {
            "remi": remitente,
            "dest": destinatario,
            "asunto": asunto,
            "cuerpo": cuerpo,
            "id": id_carta or f"carta_{remitente}_{int(time.time())}",
        }
        try:
            response = self.session.post(
                f"{self.base_url}/carta",
                params=self._params(),
                json=carta_data,
            )
            if response.status_code == 200:
                return True
            logger.warning("Error enviando carta: status {}", response.status_code)
            return False
        except requests.RequestException as e:
            logger.error("Error de conexión (enviar carta): {}", e)
            return False

    def eliminar_carta(self, uid: str) -> bool:
        """Elimina una carta del buzón."""
        try:
            response = self.session.delete(
                f"{self.base_url}/mail/{uid}",
                params=self._params(),
            )
            return response.status_code == 200
        except requests.RequestException as e:
            logger.error("Error eliminando carta: {}", e)
            return False

    # =====================================================================
    # PAQUETES
    # =====================================================================

    def enviar_paquete(self, destinatario: str, recursos: Dict[str, int]) -> bool:
        """Envía un paquete de recursos a otro jugador."""
        try:
            response = self.session.post(
                f"{self.base_url}/paquete/{destinatario}",
                params=self._params(),
                json=recursos,
            )
            # Si 404, probar con query param
            if response.status_code == 404:
                response = self.session.post(
                    f"{self.base_url}/paquete",
                    params=self._params({"dest": destinatario}),
                    json=recursos,
                )
            if response.status_code == 200:
                return True
            elif response.status_code == 422:
                logger.warning("Error de validación: {}", response.json())
            else:
                logger.warning("Error enviando paquete: {} - {}", response.status_code, response.text)
            return False
        except requests.RequestException as e:
            logger.error("Error de conexión (enviar paquete): {}", e)
            return False
