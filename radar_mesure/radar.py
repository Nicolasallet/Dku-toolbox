import numpy as np
import pandas as pd
from itertools import groupby
from operator import itemgetter
import re
import os
from scipy import signal
from scipy.fft import fft, fftfreq
from dku_data_parser import extract_radar_info




class RadarMesure:
    def __init__(self, filepath, beta=8, pad_factor=4, compute = True ,calib = False):

        self.filepath = filepath
        self.beta = beta
        self.pad_factor = pad_factor

        # === Parsing  ===

        metadata_file = '../data/metadata/Fortress24-25_smp_metadata_notes.ods'

        info = extract_radar_info(filepath, metadata_file)

        self.frequence = info['frequence']
        self.site = info['site']
        self.numero = info['numero']
        self.polarisation = info['polarisation']
        self.angle_local = info['angle_local']
        self.time = info['timestamp']
        self.coords = (info['lat'],info['lon'])
        self.pente = info['pente']
        
        
        self.temp = self.extract_sensor_temperature()

        if self.frequence == 13 : 
            self.offset = 0.332
            self.calib = 3.238e-1
        if self.frequence == 17 : 
            self.offset = 0.226
            self.calib = 3.043e-3

        if calib != True :
            if compute == True :
                # === Calcul du spectre à l'initialisation ===
                self.df, _ = self.raw_to_mean_spectrum()
                self.sigma0,_ = self.get_sigma0(self.calib)
        else : 
            self.df,_ = self.raw_to_mean_spectrum()

        

    def raw_to_mean_spectrum(self):
        scaling_factor_adc = (3.3 + 3.3) / (2 ** 12)
        frequence_echantillonage = 10e6
        N = 1024
        T = 1 / frequence_echantillonage
        n_chirp = 50
        N_padded = N * self.pad_factor
        BW = frequence_echantillonage / N_padded  # bande de résolution (Hz/bin)

        df = pd.read_csv(self.filepath, header=35, sep=',')
        if len(df.columns) == 1:
            df = pd.read_csv(self.filepath, header=36, sep=',')
        df.index = df.index + 1
        df0 = pd.DataFrame(df.columns)
        df.columns = [0, 1, 2, 3]
        df = pd.concat([df0.transpose(), df])

        data_array = np.zeros((N, 4, n_chirp))
        for i in range(n_chirp):
            start = i * (N + 3)
            lesdata = df.iloc[start:start + N, :]
            data_array[:, :, i] = lesdata

        def kai(N, beta):
            window = signal.windows.kaiser(N, beta=beta)
            S1 = np.sum(window)
            S2 = np.sum(window * window)
            return window, S1, S2

        kaiwindow, s1, s2 = kai(N, self.beta)

        data_cplx = np.zeros((N, 2, n_chirp), dtype=np.complex128)
        data_cplx[:, 0, :] = (data_array[:, 0, :] + 1j * data_array[:, 1, :]) * scaling_factor_adc
        data_cplx[:, 1, :] = (data_array[:, 2, :] + 1j * data_array[:, 3, :]) * scaling_factor_adc

        ps_rms = np.zeros((N_padded // 2, 2, n_chirp))
        for i in range(n_chirp):
            ar = data_cplx[:, :, i]
            for ch in range(2):
                  # === Étape de detrending linéaire (partie réelle et imaginaire) ===
                real_detrended = signal.detrend(np.real(ar[:, ch]))
                imag_detrended = signal.detrend(np.imag(ar[:, ch]))
                trace = real_detrended + 1j * imag_detrended

                trace = trace * kaiwindow  # application de la fenêtre de Kaiser

                trace_padded = np.pad(trace, (0, N_padded - N), 'constant')
                fft_result = fft(trace_padded)
                # === Conversion en W/Hz ===
                V2 = np.abs(fft_result[:N_padded // 2]) ** 2 / (s1 ** 2)  # [V²]
                P = V2 / 50  # [W] avec 50 Ohms d'impédance   (P = U I = U²/Z) 
                ps_watt_per_hz = P / BW  # [W/Hz]

                ps_rms[:, ch, i] = V2
        output_mean = ps_rms.mean(axis=2)
        output_std = ps_rms.std(axis=2)

        def freq_to_dist(xf):
            c = 299792458
            B = 2e9
            ramp_time = 102.4e-6
            dist = xf * c * ramp_time / (2 * B)
            return dist

        xf = fftfreq(N_padded, T)
        dist = freq_to_dist(xf[:N_padded // 2])

        output = pd.DataFrame({
            'copol': output_mean[:, 1],
            'crosspol': output_mean[:, 0],
            'copol_std': output_std[:, 1],
            'crosspol_std': output_std[:, 0]
        }, index=dist+self.offset)

        return output, os.path.basename(self.filepath).split('.')[0]


    def extract_sensor_temperature(self):
        """Extrait les températures des capteurs depuis un fichier."""
        with open(self.filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if "# Sensor temperature:" in line:
                    match = re.search(r"\{.*\}", line)
                    if match:
                        return eval(match.group())  
        return None

    def get_air(self, angle_incident, range_):
        if self.frequence == 17 :
            theta_deg = 25
            phi_deg = 16
        if self.frequence == 13 : 
            theta_deg = 24.5
            phi_deg = 19.5

        angle_incident=np.radians(angle_incident)
        theta_rad = np.radians(theta_deg)
        phi_rad = np.radians(phi_deg)
        return (np.pi * range_**2 * theta_rad * phi_rad) / (8 * np.log(2) * np.cos(angle_incident))           ##selon geldstzer 



    def get_sigma0(self,c):
        angle_incident_local = (self.angle_local)-(self.pente)
        df = self.df[0.5:]
        sigma_raw = (df['copol'] ** 2) * (df.index ** 4) / c
        air = self.get_air(angle_incident_local, df.index.values)
        delta_r = np.diff(df.index.values).mean()
        sigma = (sigma_raw / air) * delta_r

        sigma_cumsum_log = np.log10(np.cumsum(sigma.values))

        sigma_df = pd.DataFrame({
            'range': df.index.values,
            'sigma_log': 10*sigma_cumsum_log,
          })

        sigma0 = 10*sigma_cumsum_log[175*self.pad_factor]

        return sigma0,sigma_df
 