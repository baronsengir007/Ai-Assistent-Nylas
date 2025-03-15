#!/usr/bin/env python3
"""
Weather Adapter Pattern Example

This module demonstrates a basic implementation of the Adapter design pattern
with the Open-Meteo weather API.
"""

import requests
from abc import ABC, abstractmethod


class WeatherProvider(ABC):
    """Abstract interface for weather data providers."""

    @abstractmethod
    def get_temperature(self, city: str) -> float:
        """
        Get current temperature for a city.

        Args:
            city: Name of the city

        Returns:
            Current temperature in Celsius
        """
        pass


class OpenMeteoAPI:
    """
    External service that provides weather data (adaptee).
    """

    def get_weather(self, latitude: float, longitude: float) -> dict:
        """
        Get weather data from Open-Meteo API.

        Args:
            latitude: Latitude coordinate
            longitude: Longitude coordinate

        Returns:
            Raw weather data from the API
        """
        url = f"https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "current": "temperature_2m,wind_speed_10m",
            "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m",
        }

        response = requests.get(url, params=params)
        return response.json()


class CityCoordinates:
    """Simple database of city coordinates."""

    _coordinates = {
        "london": (51.51, -0.13),
        "new york": (40.71, -74.01),
        "tokyo": (35.69, 139.69),
        "paris": (48.85, 2.35),
        "sydney": (-33.87, 151.21),
    }

    @classmethod
    def get_coordinates(cls, city: str) -> tuple:
        """
        Get coordinates for a city.

        Args:
            city: Name of the city (case-insensitive)

        Returns:
            Tuple of (latitude, longitude)

        Raises:
            ValueError: If city is not in the database
        """
        city_lower = city.lower()
        if city_lower in cls._coordinates:
            return cls._coordinates[city_lower]
        raise ValueError(f"No coordinates found for {city}")


class WeatherAdapter(WeatherProvider):
    """
    Adapter that converts OpenMeteoAPI to the WeatherProvider interface.
    """

    def __init__(self, weather_api: OpenMeteoAPI):
        """
        Initialize with the OpenMeteoAPI.

        Args:
            weather_api: The OpenMeteoAPI instance
        """
        self.weather_api = weather_api

    def get_temperature(self, city: str) -> float:
        """
        Get current temperature for a city.

        Args:
            city: Name of the city

        Returns:
            Current temperature in Celsius

        Raises:
            ValueError: If city coordinates cannot be found
        """
        try:
            # Get coordinates for the city
            latitude, longitude = CityCoordinates.get_coordinates(city)

            # Get weather data from the API
            weather_data = self.weather_api.get_weather(latitude, longitude)

            # Extract and return the temperature
            return weather_data["current"]["temperature_2m"]
        except Exception as e:
            print(f"Error getting temperature for {city}: {e}")
            raise


def main():
    """Example usage of the Weather Adapter."""

    # Create the adaptee
    open_meteo_api = OpenMeteoAPI()

    # Create the adapter
    weather_adapter = WeatherAdapter(open_meteo_api)

    # Use the adapter through the target interface
    cities = ["London", "New York", "Tokyo", "Paris", "Sydney"]

    print("Current Temperatures:")
    print("--------------------")

    for city in cities:
        try:
            temperature = weather_adapter.get_temperature(city)
            print(f"{city}: {temperature}°C")
        except ValueError as e:
            print(f"{city}: {e}")


if __name__ == "__main__":
    main()
