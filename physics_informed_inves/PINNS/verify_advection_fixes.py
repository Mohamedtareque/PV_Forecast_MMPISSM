
import math

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate great-circle distance between two points using Haversine formula.
    """
    R = 6371000  # Earth's radius in meters
    
    # Convert to radians
    lat1_rad, lon1_rad, lat2_rad, lon2_rad = map(
        math.radians, [lat1, lon1, lat2, lon2]
    )
    
    # Haversine formula components
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    a = math.sin(dlat / 2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    distance = R * c
    return distance

def calculate_bearing(lat1, lon1, lat2, lon2):
    """
    Calculate initial bearing from point 1 to point 2 in degrees [0, 360).
    Bearing is measured clockwise from north.
    """
    lat1_rad, lon1_rad, lat2_rad, lon2_rad = map(
        math.radians, [lat1, lon1, lat2, lon2]
    )
    
    dlon = lon2_rad - lon1_rad
    
    x = math.sin(dlon) * math.cos(lat2_rad)
    y = math.cos(lat1_rad) * math.sin(lat2_rad) - math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(dlon)
    
    bearing_rad = math.atan2(x, y)
    bearing_deg = math.degrees(bearing_rad)
    
    # Normalize to 0-360
    return (bearing_deg + 360) % 360

def verify_geometry_calculations():
    """
    Verify Haversine distance and bearing calculations with known values.
    """
    print("=== GEOMETRY VERIFICATION ===")
    
    # Test 1: Known coordinates (same point)
    dist_same = haversine_distance(36.64, 113.64, 36.64, 113.64)
    print(f"Distance (same point): {dist_same:.1f}m (expected: ~0m)")
    assert dist_same < 100, f"Distance should be near zero: {dist_same:.1f}m"
    
    # Test 2: Bearing calculation
    # Point 1: 36.64, 113.64
    # Point 2: 36.70, 113.70 (North East)
    # Bearing should be around 45 degrees? No, let's check.
    # dLat > 0 (North), dLon > 0 (East).
    # Wait, the code says "Bearing (south): ... (expected: ~180°)"??
    # In the code snippet provided in plan:
    # "Test 2: Bearing calculation: bearing = calculate_bearing(36.64, 113.64, 36.70, 113.70)"
    # Origin: (36.64, 113.64). Dest: (36.70, 113.70).
    # This is moving North-East. Bearing should be approx 45 degrees.
    # Why did the user plan say "Bearing (south): ... (expected: ~180°)"?
    # Ah, maybe they swapped the coordinates in their mental model or text?
    # Let's check the code snippet in the request again.
    # Request: "bearing = calculate_bearing(36.64, 113.64, 36.70, 113.70) ... print(f"Bearing (south)...")"
    # This looks like a mistake in the plan text, or I am misunderstanding.
    # 36.70 is larger than 36.64 (North).
    # 113.70 is larger than 113.64 (East).
    # So it should be NE (approx 45 deg).
    
    # Let's look at the ACTUAL station coordinates.
    # Station 07: (36.64187°N, 113.64187°E)
    # Station 08: (36.70761°N, 113.69999°E)
    # 07 is target. 08 is neighbor.
    # Flow is 08 -> 07 (Advection from neighbor to target).
    # So Origin=08, Dest=07.
    # OriginLat = 36.70761, OriginLon = 113.69999
    # DestLat = 36.64187, DestLon = 113.64187
    # dLat = 36.64 - 36.70 = -0.06 (South)
    # dLon = 113.64 - 113.70 = -0.06 (West)
    # So Bearing should be South-West (~225 deg).
    # The user plan says: "Calculated Bearing (08→07): 220.00°". This matches SW.
    
    # So let's test 08 -> 07.
    
    # Test 3: Station 08 to 07 (actual values)
    # Origin: S8, Dest: S7
    lat_s8, lon_s8 = 36.70761, 113.69999
    lat_s7, lon_s7 = 36.64187, 113.64187
    
    dist_actual = haversine_distance(lat_s8, lon_s8, lat_s7, lon_s7)
    print(f"Distance (08→07): {dist_actual:.1f}m")
    
    bearing_actual = calculate_bearing(lat_s8, lon_s8, lat_s7, lon_s7)
    print(f"Bearing (08→07): {bearing_actual:.1f}°")
    
    # Expected: ~220° (South West)
    print(f"Expected bearing range: 210-230° (South West)")
    assert 210 <= bearing_actual <= 230, f"Bearing should be SW: {bearing_actual:.1f}°"
    
    # Expected Distance: ~8960m (calculated from coords)
    print(f"Expected distance: ~8960m")
    assert 8000 < dist_actual < 9500, f"Distance should be ~9km: {dist_actual:.1f}m"
    
    # Test 4: Lag calculation
    # For 8960m distance and 5 m/s wind:
    # Lag = dist / (velocity * dt) = 8960 / (5 * 900)
    lag_expected = 8961 / (5 * 900)
    print(f"Expected lag (5 m/s): {lag_expected:.2f} timesteps")
    
    print("=== VERIFICATION COMPLETE ===")

if __name__ == "__main__":
    verify_geometry_calculations()
