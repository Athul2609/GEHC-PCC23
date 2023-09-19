import requests
import geocoder

def get_nearby_hospitals(latitude, longitude, radius):
    overpass_url = "https://overpass-api.de/api/interpreter"
    query = f"""
    [out:json];
    (
        node["amenity"="hospital"](around:{radius},{latitude},{longitude});
        way["amenity"="hospital"](around:{radius},{latitude},{longitude});
        relation["amenity"="hospital"](around:{radius},{latitude},{longitude});
    );
    out center;
    """
    response = requests.get(overpass_url, params={"data": query})
    data = response.json()
    hospital_dict = {}
    no_hospitals = {"return hospital_dict"}
    if "elements" in data:
        hospitals = data["elements"]
        for hospital in hospitals:
            if "tags" in hospital:
                name = hospital.get("tags", {}).get("name", "N/A")
                address = hospital.get("tags", {}).get("addr:full", "N/A")
                hospital_info = {
                    "Address": address
                }
                hospital_dict[name] = hospital_info
        return hospital_dict
    else:
        return no_hospitals
        
def get_current_location():
    try:
        # Use the 'geocoder' library to automatically detect the device's location
        location = geocoder.ip('me')
        if location:
            latitude = location.latlng[0]
            longitude = location.latlng[1]
            return latitude, longitude
        else:
            return None
    except Exception as e:
        print(f"Error: {str(e)}")
        return None

# Call the function to get the current location
location = get_current_location()

if location:
    latitude, longitude = location
    print(f"Latitude: {latitude}, Longitude: {longitude}")
else:
    print("Unable to retrieve location data.")


radius = 2000# Example radius in meters

nearby_hospitals = get_nearby_hospitals(latitude, longitude, radius)
