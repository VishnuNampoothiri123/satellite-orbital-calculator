import tkinter as tk
from tkinter import ttk, messagebox
from sgp4.api import Satrec
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from skyfield.api import Topos, load
import pandas as pd
from datetime import datetime, timedelta
import threading
from mpl_toolkits.basemap import Basemap
from mpl_toolkits.mplot3d import Axes3D
import math 

class OrbitalCalculatorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Orbital Calculator and Satellite Tracker")

        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill='both', expand=True)
        self.orbit_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.orbit_tab, text='Orbit Calculation')

        self.create_orbit_tab()
        self.create_tle_tab()
        self.create_satellite_tracker_tab()

    def create_orbit_tab(self):
        mainframe = ttk.Frame(self.orbit_tab, padding="10")
        mainframe.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        fields = [
            ("Earth Radius (km)", 6371.0),
            ("Semi-major Axis (km)", None),
            ("Eccentricity", None),
            ("Time After Perigee (s)", None),
            ("True Anomaly (rad)", None),
            ("Latus Rectum (km)", None),
            ("Minor Axis (km)", None),
            ("Distance Between Foci (km)", None),
            ("Perigee Height (km)", None),
            ("Apogee Height (km)", None),
            ("Mean Motion (rad/s)", None),
            ("Mean Anomaly (rad)", None),
            ("Period (s)", None),
            ("Inclination (deg)", None),
            ("Node Regression Rate (deg/day)", None),
            ("Apsides Rotation Rate (deg/day)", None)
        ]

        self.entries = {}
        for i, (label, default) in enumerate(fields):
            ttk.Label(mainframe, text=label).grid(row=i, column=0, sticky=tk.W, pady=(0, 5))
            entry = ttk.Entry(mainframe)
            if default is not None:
                entry.insert(0, str(default))
            entry.grid(row=i, column=1, sticky=(tk.W, tk.E), pady=(0, 5))
            self.entries[label] = entry

        self.results_orbit = tk.StringVar()
        ttk.Label(mainframe, textvariable=self.results_orbit).grid(row=len(fields)+2, column=0, columnspan=2, pady=(10, 0))

        ttk.Button(mainframe, text="Calculate Orbit", command=self.calculate_orbit).grid(row=len(fields)+1, column=0, columnspan=2, pady=(10, 0), sticky=(tk.W, tk.E))

        self.plot_frame_orbit = ttk.Frame(self.orbit_tab, width=500, height=500)
        self.plot_frame_orbit.grid(row=0, column=1, rowspan=20, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.plot_frame_orbit.grid_propagate(False)

    def calculate_orbit(self):
        K1 = 66063.1704  # km^2
        R = self.try_get_float(self.entries["Earth Radius (km)"])
        a = self.try_get_float(self.entries["Semi-major Axis (km)"])
        e = self.try_get_float(self.entries["Eccentricity"])
        t = self.try_get_float(self.entries["Time After Perigee (s)"])
        nu = self.try_get_float(self.entries["True Anomaly (rad)"])
        latus_rectum = self.try_get_float(self.entries["Latus Rectum (km)"])
        minor_axis = self.try_get_float(self.entries["Minor Axis (km)"])
        distance_between_foci = self.try_get_float(self.entries["Distance Between Foci (km)"])
        perigee_height = self.try_get_float(self.entries["Perigee Height (km)"])
        apogee_height = self.try_get_float(self.entries["Apogee Height (km)"])
        mean_motion = self.try_get_float(self.entries["Mean Motion (rad/s)"])
        mean_anomaly = self.try_get_float(self.entries["Mean Anomaly (rad)"])
        period = self.try_get_float(self.entries["Period (s)"])
        inclination = self.try_get_float(self.entries["Inclination (deg)"])
        node_regression_rate = self.try_get_float(self.entries["Node Regression Rate (deg/day)"])
        apsides_rotation_rate = self.try_get_float(self.entries["Apsides Rotation Rate (deg/day)"])

        results_text = ""

        # Calculate the semi-major axis if not provided
        if a is None:
            if perigee_height is not None and apogee_height is not None:
                a = (perigee_height + apogee_height + 2 * R) / 2
                results_text += f"Semi-major Axis: {a:.2f} km\n"
            elif latus_rectum is not None and e is not None:
                a = latus_rectum / (1 - e**2)
                results_text += f"Semi-major Axis: {a:.2f} km\n"
            elif mean_motion is not None:
                a = (398600.4418 / (mean_motion * 2 * np.pi)**2)**(1/3)
                results_text += f"Semi-major Axis: {a:.2f} km\n"
            elif period is not None:
                a = (398600.4418 * (period / (2 * np.pi))**2)**(1/3)
                results_text += f"Semi-major Axis: {a:.2f} km\n"

        # Calculate eccentricity if not provided
        if e is None:
            if perigee_height is not None and apogee_height is not None:
                e = (apogee_height - perigee_height) / (apogee_height + perigee_height + 2 * R)
                results_text += f"Eccentricity: {e:.4f}\n"
            elif distance_between_foci is not None and a is not None:
                e = distance_between_foci / (2 * a)
                results_text += f"Eccentricity: {e:.4f}\n"

        # Check for eccentricity equals 1 to avoid division by zero
        if e == 1:
            self.results_orbit.set("Eccentricity cannot be 1. This results in a parabolic trajectory, which is not a closed orbit.")
            return

        # Calculate latus rectum if not provided
        if latus_rectum is None and a is not None and e is not None:
            latus_rectum = a * (1 - e**2)
            results_text += f"Latus Rectum: {latus_rectum:.2f} km\n"

        # Calculate minor axis if not provided
        if minor_axis is None and a is not None and e is not None:
            minor_axis = a * np.sqrt(1 - e**2)
            results_text += f"Minor Axis: {minor_axis:.2f} km\n"

        # Calculate distance between foci if not provided
        if distance_between_foci is None and a is not None and e is not None:
            distance_between_foci = 2 * a * e
            results_text += f"Distance Between Foci: {distance_between_foci:.2f} km\n"

        # Calculate perigee height if not provided
        if perigee_height is None and a is not None and e is not None:
            perigee_height = a * (1 - e) - R
            results_text += f"Perigee Height: {perigee_height:.2f} km\n"

        # Calculate apogee height if not provided
        if apogee_height is None and a is not None and e is not None:
            apogee_height = a * (1 + e) - R
            results_text += f"Apogee Height: {apogee_height:.2f} km\n"

        # Calculate mean motion if not provided
        if mean_motion is None and a is not None:
            mean_motion = np.sqrt(398600.4418 / a**3)
            results_text += f"Mean Motion: {mean_motion:.4e} rad/s\n"

        # Calculate period if not provided
        if period is None and mean_motion is not None:
            period = 2 * np.pi / mean_motion
            results_text += f"Period: {period:.2f} s\n"

        # Calculate true anomaly if not provided
        if nu is None and t is not None and mean_motion is not None and e is not None:
            M = mean_motion * t
            E = self.solve_kepler(M, e)
            nu = 2 * np.arctan2(np.sqrt(1 + e) * np.sin(E / 2), np.sqrt(1 - e) * np.cos(E / 2))
            results_text += f"True Anomaly: {nu:.4f} rad\n"

        inclination_provided = inclination is not None
        inclination1 = None

        # Calculate node regression rate if not provided
        if node_regression_rate is None and mean_motion is not None and a is not None and e is not None:
            K = (mean_motion * K1) / (a**2 * (1 - e**2)**2)
            node_regression_rate = -K * np.cos(np.deg2rad(inclination if inclination is not None else 0)) * (15552000 / math.pi)
            results_text += f"Node Regression Rate: {node_regression_rate:.6f} deg/day\n"
            if inclination is None and apsides_rotation_rate is not None:
                inclination1 = math.acos(apsides_rotation_rate / -K)

        # Calculate apsides rotation rate if not provided
        if apsides_rotation_rate is None and mean_motion is not None and a is not None and e is not None:
            K = (mean_motion * K1) / (a**2 * (1 - e**2)**2)
            apsides_rotation_rate = K * (2 - 2.5 * np.sin(np.deg2rad(inclination if inclination is not None else 0))**2) * (15552000 / math.pi)
            results_text += f"Apsides Rotation Rate: {apsides_rotation_rate:.6f} deg/day\n"


        # Display results
        self.results_orbit.set(results_text)

        # Plot the orbit
        if a is not None and e is not None:
            self.plot_orbit(a, e, R)

    def solve_kepler(self, M, e, tol=1e-10):
        E = M if e < 0.8 else np.pi
        for _ in range(200):
            E_new = E + (M - E + e * np.sin(E)) / (1 - e * np.cos(E))
            if np.abs(E_new - E) < tol:
                break
            E = E_new
        return E

    def plot_orbit(self, a, e, R):
        # Clear previous plot
        for widget in self.plot_frame_orbit.winfo_children():
            widget.destroy()

        fig = plt.Figure(figsize=(6, 6), dpi=100)
        ax = fig.add_subplot(111)
        ax.set_aspect('equal')

        theta = np.linspace(0, 2 * np.pi, 1000)
        r = a * (1 - e**2) / (1 + e * np.cos(theta))

        x = r * np.cos(theta)
        y = r * np.sin(theta)

        ax.plot(x, y, label='Orbit')
        ax.plot([0], [0], 'bo', label='Earth')
        ax.set_xlabel('X (km)')
        ax.set_ylabel('Y (km)')
        ax.legend()
        ax.grid()

        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame_orbit)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def try_get_float(self, entry):
        try:
            return float(entry.get())
        except ValueError:
            return None

    def create_tle_tab(self):
        self.tle_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.tle_tab, text='TLE Data')

        tle_frame = ttk.Frame(self.tle_tab, padding="10")
        tle_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        ttk.Label(tle_frame, text="TLE Line 1:").grid(row=0, column=0, sticky=tk.W)
        self.tle1_entry = ttk.Entry(tle_frame, width=75)
        self.tle1_entry.grid(row=1, column=0, sticky=(tk.W, tk.E))

        ttk.Label(tle_frame, text="TLE Line 2:").grid(row=2, column=0, sticky=tk.W)
        self.tle2_entry = ttk.Entry(tle_frame, width=75)
        self.tle2_entry.grid(row=3, column=0, sticky=(tk.W, tk.E))

        self.load_tle_button = ttk.Button(tle_frame, text="Load TLE Data", command=self.load_tle_data)
        self.load_tle_button.grid(row=4, column=0, sticky=tk.W)

        self.tle_results = tk.StringVar()
        ttk.Label(tle_frame, textvariable=self.tle_results).grid(row=5, column=0, sticky=(tk.W, tk.E))

        self.plot_frame_tle = ttk.Frame(self.tle_tab, padding="10")
        self.plot_frame_tle.grid(row=0, column=1, rowspan=6, sticky=(tk.W, tk.E, tk.N, tk.S))

    def create_satellite_tracker_tab(self):
        self.satellite_tracker_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.satellite_tracker_tab, text='Satellite Tracker')

        frame = ttk.Frame(self.satellite_tracker_tab, padding="10")
        frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # TLE URL Selection
        ttk.Label(frame, text="TLE Data Source:").grid(row=0, column=0, sticky=tk.W)
        self.tle_url_type = ttk.Combobox(frame, width=50, values=[
            'https://celestrak.com/NORAD/elements/stations.txt',
            'https://celestrak.com/NORAD/elements/weather.txt',
            'https://celestrak.com/NORAD/elements/geo.txt',
            'https://celestrak.com/NORAD/elements/science.txt',
            'https://celestrak.org/NORAD/elements/supplemental/sup-gp.php?FILE=starlink&FORMAT=tle',
            'https://celestrak.org/NORAD/elements/supplemental/sup-gp.php?FILE=planet&FORMAT=tle'
        ])
        self.tle_url_type.grid(row=0, column=1, sticky=(tk.W, tk.E))
        self.tle_url_type.current(0)

        self.load_button = ttk.Button(frame, text="Load Satellites", command=self.load_satellites)
        self.load_button.grid(row=1, column=1, sticky=tk.E)

        # Satellite Name Selection
        ttk.Label(frame, text="Satellite Name:").grid(row=2, column=0, sticky=tk.W)
        self.sat_name = ttk.Combobox(frame, width=47)
        self.sat_name.grid(row=2, column=1, sticky=(tk.W, tk.E))

        self.track_button = ttk.Button(frame, text="Track Satellite", command=self.track_satellite)
        self.track_button.grid(row=3, column=1, sticky=tk.E)

        # Data Table
        self.tree = ttk.Treeview(frame, columns=("Time", "Latitude", "Longitude", "Altitude"), show="headings")
        self.tree.heading("Time", text="Time")
        self.tree.heading("Latitude", text="Latitude")
        self.tree.heading("Longitude", text="Longitude")
        self.tree.heading("Altitude", text="Altitude (m)")
        self.tree.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Figures for 2D Map and 3D Globe
        self.figure = plt.Figure(figsize=(8, 6), dpi=100)
        self.map_plot = self.figure.add_subplot(121)
        self.globe_plot = self.figure.add_subplot(122, projection='3d')
        self.map_canvas = FigureCanvasTkAgg(self.figure, frame)
        self.map_canvas.get_tk_widget().grid(row=5, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Satellite Information Panel
        info_frame = ttk.Frame(self.satellite_tracker_tab, padding="10")
        info_frame.grid(row=0, column=1, sticky=(tk.N, tk.W, tk.E, tk.S))
        ttk.Label(info_frame, text="Satellite Information", font=("Helvetica", 14)).grid(row=0, column=0, columnspan=2)

        self.info_labels = {}
        for i, label in enumerate(["Name:", "NORAD ID:", "Launch Date:", "Orbit Type:"]):
            ttk.Label(info_frame, text=label).grid(row=i+1, column=0, sticky=tk.W)
            self.info_labels[label] = ttk.Label(info_frame, text="")
            self.info_labels[label].grid(row=i+1, column=1, sticky=tk.W)

    def load_tle_data(self):
        tle_line1 = self.tle1_entry.get().strip()
        tle_line2 = self.tle2_entry.get().strip()

        try:
            satellite = Satrec.twoline2rv(tle_line1, tle_line2)
            results_text = f"Satellite Catalog Number: {satellite.satnum}\n"

            if satellite.error == 0:
                results_text += "TLE Data successfully parsed.\n"
                results_text += f"Inclination: {satellite.inclo} degrees\n"
                results_text += f"RAAN: {satellite.nodeo} degrees\n"
                results_text += f"Eccentricity: {satellite.ecco}\n"
                results_text += f"Argument of Perigee: {satellite.argpo} degrees\n"
                results_text += f"Mean Anomaly: {satellite.mo} degrees\n"
                results_text += f"Mean Motion: {satellite.no_kozai} revs per day\n"
                results_text += f"BSTAR Drag Term: {satellite.bstar}\n"
            else:
                results_text += f"Error in TLE data: {satellite.error}\n"

            self.tle_results.set(results_text)
            self.plot_tle_orbit(satellite)

        except ValueError as e:
            self.tle_results.set(f"Error parsing TLE data: {e}")

    def plot_tle_orbit(self, satellite):
        for widget in self.plot_frame_tle.winfo_children():
            widget.destroy()

        fig = plt.Figure(figsize=(6, 6), dpi=100)
        ax = fig.add_subplot(111, projection='3d')
        ax.set_box_aspect([1, 1, 1])

        time_span = np.linspace(0, 2 * np.pi, 1000)

        x = np.cos(time_span) * 6678
        y = np.sin(time_span) * 6678
        z = np.sin(time_span) * 6678 / 2

        ax.plot(x, y, z, label='TLE Orbit')
        ax.scatter([0], [0], [0], color='b', label='Earth')
        ax.set_xlabel('X (km)')
        ax.set_ylabel('Y (km)')
        ax.set_zlabel('Z (km)')
        ax.legend()
        ax.grid()

        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame_tle)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def load_satellites(self):
        try:
            tle_url = self.tle_url_type.get()
            self.load_button.config(state=tk.DISABLED)
            self.track_button.config(state=tk.DISABLED)
            threading.Thread(target=self._load_satellites_thread, args=(tle_url,)).start()
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _load_satellites_thread(self, tle_url):
        try:
            messagebox.showinfo("Loading", f"Loading satellites from {tle_url}...")
            self.satellites = load.tle_file(tle_url)
            satellite_names = [sat.name for sat in self.satellites]
            self.sat_name['values'] = satellite_names
            self.load_button.config(state=tk.NORMAL)
            self.track_button.config(state=tk.NORMAL)
            messagebox.showinfo("Success", "Satellites loaded successfully.")
        except Exception as e:
            messagebox.showerror("Error", str(e))
            self.load_button.config(state=tk.NORMAL)
            self.track_button.config(state=tk.NORMAL)

    def track_satellite(self):
        try:
            sat_name = self.sat_name.get()
            satellite = {sat.name: sat for sat in self.satellites}[sat_name]

            # Updating Satellite Information
            self.update_satellite_info(satellite)

            ts = load.timescale()
            t0 = datetime.utcnow()
            period = timedelta(hours=1)
            interval = timedelta(seconds=60)

            times = []
            current_time = t0
            while current_time < t0 + period:
                times.append(ts.utc(current_time.year, current_time.month, current_time.day, current_time.hour, current_time.minute, current_time.second))
                current_time += interval

            geocentric = [satellite.at(t) for t in times]
            latitudes, longitudes, altitudes = [], [], []
            for position in geocentric:
                subpoint = position.subpoint()
                latitudes.append(subpoint.latitude.degrees)
                longitudes.append(subpoint.longitude.degrees)
                altitudes.append(subpoint.elevation.m)

            self.update_table(times, latitudes, longitudes, altitudes)
            self.update_map(latitudes, longitudes)
            self.update_globe(latitudes, longitudes, altitudes)

        except Exception as e:
            messagebox.showerror("Error", str(e))

    def update_satellite_info(self, satellite):
        if satellite is None or satellite.model is None:
            self.info_labels["Name:"].config(text="")
            self.info_labels["NORAD ID:"].config(text="")
            self.info_labels["Launch Date:"].config(text="")
            self.info_labels["Orbit Type:"].config(text="")
        else:
            self.info_labels["Name:"].config(text=satellite.name)
            self.info_labels["NORAD ID:"].config(text=satellite.model.satnum)
            if hasattr(satellite.model, 'epoch'):
                launch_date = satellite.model.epoch.utc_datetime().strftime('%Y-%m-%d')
            else:
                launch_date = "Unknown"
            self.info_labels["Launch Date:"].config(text=launch_date)
            self.info_labels["Orbit Type:"].config(text=satellite.model.classification)

    def update_table(self, times, latitudes, longitudes, altitudes):
        for row in self.tree.get_children():
            self.tree.delete(row)
        for i, (time, lat, lon, alt) in enumerate(zip(times, latitudes, longitudes, altitudes)):
            self.tree.insert("", "end", values=(time.utc_datetime(), lat, lon, alt))

    def update_map(self, latitudes, longitudes):
        self.map_plot.clear()
        m = Basemap(projection='cyl', resolution='l', ax=self.map_plot)
        m.drawcoastlines()
        m.drawcountries()
        m.drawparallels(range(-90, 90, 30), labels=[1, 0, 0, 0])
        m.drawmeridians(range(-180, 180, 60), labels=[0, 0, 0, 1])
        x, y = m(longitudes, latitudes)
        m.plot(x, y, marker='o', markersize=5, linestyle='-', color='b')
        self.map_plot.set_title('Satellite Path')
        self.map_canvas.draw()

    def update_globe(self, latitudes, longitudes, altitudes):
        self.globe_plot.clear()
        self.globe_plot.set_title('Satellite on 3D Globe')

        # Draw the Earth as a sphere
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones(np.size(u)), np.cos(v))
        self.globe_plot.plot_surface(x, y, z, color='b', alpha=0.3)

        # Plot the satellite path
        R = 1  # Earth's radius
        satellite_x = R * np.cos(np.radians(latitudes)) * np.cos(np.radians(longitudes))
        satellite_y = R * np.cos(np.radians(latitudes)) * np.sin(np.radians(longitudes))
        satellite_z = R * np.sin(np.radians(latitudes))
        self.globe_plot.plot(satellite_x, satellite_y, satellite_z, color='r', marker='o')

        self.map_canvas.draw()

if __name__ == "__main__":
    root = tk.Tk()
    app = OrbitalCalculatorApp(root)
    root.mainloop()
