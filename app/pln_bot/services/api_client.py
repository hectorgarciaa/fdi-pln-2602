"""
Cliente para la API del juego.

Centraliza todas las llamadas HTTP al servidor fdi-pln-butler.
Usa loguru para logging en vez de print().
"""

import time
import requests
from typing import Dict, List, Optional, Tuple

from loguru import logger
from ..core.config import API_BASE_URL


class APIClient:
    """Cliente para interactuar con la API del juego."""

    def __init__(
        self,
        base_url: str = None,
        agente: str = None,
        timeout: Tuple[float, float] = (5.0, 20.0),
        max_retries: int = 2,
        retry_backoff: float = 0.5,
    ):
        self.base_url = base_url or API_BASE_URL
        self.agente = agente
        self.session = requests.Session()
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff
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

    def _request(self, method: str, path: str, **kwargs) -> Optional[requests.Response]:
        """Realiza una petición HTTP con timeout y reintentos básicos."""
        timeout = kwargs.pop("timeout", self.timeout)
        url = f"{self.base_url}{path}"
        ultimo_error: Optional[Exception] = None

        for intento in range(1, self.max_retries + 2):
            try:
                return self.session.request(method, url, timeout=timeout, **kwargs)
            except requests.RequestException as e:
                ultimo_error = e
                if intento > self.max_retries:
                    break
                espera = self.retry_backoff * intento
                logger.warning(
                    "HTTP {} {} falló (intento {}/{}): {}. Reintentando en {:.1f}s",
                    method,
                    path,
                    intento,
                    self.max_retries + 1,
                    e,
                    espera,
                )
                time.sleep(espera)

        logger.error("HTTP {} {} falló definitivamente: {}", method, path, ultimo_error)
        return None

    # =====================================================================
    # INFORMACIÓN
    # =====================================================================

    def get_info(self) -> Optional[Dict]:
        """Obtiene información general del jugador."""
        response = self._request("GET", "/info", params=self._params())
        if response is None:
            return None
        if response.status_code == 200:
            return response.json()
        logger.warning("Error obteniendo info: status {}", response.status_code)
        return None

    def get_gente(self) -> List[str]:
        """Obtiene lista de personas disponibles."""
        response = self._request("GET", "/gente")
        if response is None:
            return []
        if response.status_code == 200:
            data = response.json()
            # Normalizar: la API puede devolver ["str"] o [{"nombre": ...}]
            resultado = []
            for item in data:
                if isinstance(item, str):
                    resultado.append(item)
                elif isinstance(item, dict):
                    # Intentar extraer el nombre del dict
                    nombre = (
                        item.get("nombre")
                        or item.get("name")
                        or item.get("alias")
                        or str(item)
                    )
                    logger.debug(
                        "get_gente: item dict normalizado {} -> {}", item, nombre
                    )
                    resultado.append(nombre)
                else:
                    resultado.append(str(item))
            return resultado
        logger.warning("Error obteniendo gente: status {}", response.status_code)
        return []

    # =====================================================================
    # ALIAS
    # =====================================================================

    def crear_alias(self, nombre: str) -> bool:
        """Crea un nuevo alias."""
        response = self._request(
            "POST",
            f"/alias/{nombre}",
            params=self._params(),
        )
        if response is None:
            return False
        if response.status_code == 200:
            logger.success("Alias '{}' creado", nombre)
            return True
        logger.warning("Error creando alias: status {}", response.status_code)
        return False

    def eliminar_alias(self, nombre: str) -> bool:
        """Elimina un alias."""
        response = self._request(
            "DELETE",
            f"/alias/{nombre}",
            params=self._params(),
        )
        if response is None:
            return False
        if response.status_code == 200:
            logger.success("Alias '{}' eliminado", nombre)
            return True
        logger.warning("Error eliminando alias: status {}", response.status_code)
        return False

    # =====================================================================
    # CARTAS
    # =====================================================================

    def enviar_carta(
        self,
        remitente: str,
        destinatario: str,
        asunto: str,
        cuerpo: str,
        id_carta: str = None,
    ) -> bool:
        """Envía una carta a otro jugador."""
        carta_data = {
            "remi": remitente,
            "dest": destinatario,
            "asunto": asunto,
            "cuerpo": cuerpo,
            "id": id_carta or f"carta_{remitente}_{int(time.time())}",
        }
        response = self._request(
            "POST",
            "/carta",
            params=self._params(),
            json=carta_data,
        )
        if response is None:
            return False
        if response.status_code == 200:
            return True
        logger.warning("Error enviando carta: status {}", response.status_code)
        return False

    def eliminar_carta(self, uid: str) -> bool:
        """Elimina una carta del buzón."""
        response = self._request(
            "DELETE",
            f"/mail/{uid}",
            params=self._params(),
        )
        if response is None:
            return False
        return response.status_code == 200

    # =====================================================================
    # PAQUETES
    # =====================================================================

    def enviar_paquete(self, destinatario: str, recursos: Dict[str, int]) -> bool:
        """Envía un paquete de recursos a otro jugador."""
        response = self._request(
            "POST",
            f"/paquete/{destinatario}",
            params=self._params(),
            json=recursos,
        )
        if response is None:
            return False
        # Si 404, probar con query param
        if response.status_code == 404:
            response = self._request(
                "POST",
                "/paquete",
                params=self._params({"dest": destinatario}),
                json=recursos,
            )
            if response is None:
                return False

        if response.status_code == 200:
            return True
        if response.status_code == 422:
            logger.warning("Error de validación: {}", response.json())
        else:
            logger.warning(
                "Error enviando paquete: {} - {}",
                response.status_code,
                response.text,
            )
        return False
