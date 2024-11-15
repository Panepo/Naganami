from langchain_core.tools import tool
import requests


@tool
def get_weather(city_name: str) -> str:
  """
  Get the current weather for `city_name`
  """

  if not isinstance(city_name, str):
    raise TypeError("City name must be a string")

  key_selection = {
    "current_condition": [
      "temp_C",
      "FeelsLikeC",
      "humidity",
      "weatherDesc",
      "observation_time",
    ],
  }

  resp = requests.get(f"https://wttr.in/{city_name}?format=j1")
  resp.raise_for_status()
  resp = resp.json()
  ret = {k: {_v: resp[k][0][_v] for _v in v} for k, v in key_selection.items()}

  return str(ret)
