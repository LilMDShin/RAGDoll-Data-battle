# script codecarbone.py
# decorateur pour lancer une fonction et mesurer la consommation de codecarbone
from codecarbon import OfflineEmissionsTracker
from functools import wraps

# Désactiver les logs
from codecarbon.external.logger import logger
logger.disabled = True


def codecarbone_fr(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        tracker = OfflineEmissionsTracker(
                project_name="LivrIA",
                country_iso_code="FRA",
                output_dir=".",
                output_file="emissions.csv"
        )
        tracker.start()    

        result = func(*args, **kwargs)

        tracker.stop()
        print(f"La fonction {func.__name__} a émis :\n\t- {tracker.final_emissions_data.emissions} kgCO2e,\n\t- CPU : {tracker.final_emissions_data.cpu_energy} kWh\n\t- GPU : {tracker.final_emissions_data.gpu_energy} kWh")

        return result, tracker.final_emissions_data.emissions
    return wrapper


# Exemple d'utilisation
if __name__ == "__main__":
    @codecarbone_fr
    def test():
        print("Hello world")


    test()