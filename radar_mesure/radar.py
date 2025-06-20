import numpy as np
import pandas as pd
from itertools import groupby
from operator import itemgetter
import re
import os
from scipy import signal
from scipy.fft import fft, fftfreq

class RadarMesure:
    def __init__(self, filepath, beta, pad_factor=1):
        self.filepath = filepath
        self.beta = beta
        self.pad_factor = pad_factor

        # === Parsing du nom de fichier ===
        filename = os.path.basename(filepath)
        pattern = r'(?P<freq>\d+)GHz_(?P<site>\w+)_(?P<type>\w+)_(?P<num>\d+)_'
        pattern += r'(?P<pol>[hv])_(?P<angle>\d+)deg'

        match = re.match(pattern, filename)
        if not match:
            raise ValueError("Nom de fichier invalide : attendu format 13GHz_halfpipe_ridge_0_v_30deg.txt")

        self.frequence = int(match.group("freq"))
        self.site = match.group("site")
        self.numero = int(match.group("num"))
        self.polarisation = match.group("pol")
        self.angle_local = int(match.group("angle"))

        # === Calcul du spectre à l'initialisation ===
        self.df, _ = self.raw_to_mean_spectrum()

    def raw_to_mean_spectrum(self):
        scaling_factor_adc = (3.3 + 3.3) / (2 ** 12)
        frequence_echantillonage = 10e6
        N = 1024
        T = 1 / frequence_echantillonage
        n_chirp = 150
        N_padded = N * self.pad_factor

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

        data_cplx = np.zeros((N, 2, n_chirp), dtype=np.complex_)
        data_cplx[:, 0, :] = (data_array[:, 0, :] + 1j * data_array[:, 1, :]) * scaling_factor_adc
        data_cplx[:, 1, :] = (data_array[:, 2, :] + 1j * data_array[:, 3, :]) * scaling_factor_adc

        ps_rms = np.zeros((N_padded // 2, 2, n_chirp))
        for i in range(n_chirp):
            ar = data_cplx[:, :, i]
            for ch in range(2):
                trace = ar[:, ch] * kaiwindow
                trace_padded = np.pad(trace, (0, N_padded - N), 'constant')
                fft_result = fft(trace_padded)
                ps = np.abs(fft_result[:N_padded // 2]) ** 2 / (s1 ** 2)
                ps_rms[:, ch, i] = ps

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
        }, index=dist)

        return output, os.path.basename(self.filepath).split('.')[0]

    def get_air(self, angle_incident, range_):
        theta_deg = 25
        phi_deg = 16
        theta_rad = np.radians(theta_deg)
        phi_rad = np.radians(phi_deg)
        return (np.pi * range_**2 * theta_rad * phi_rad) / (8 * np.log(2) * np.cos(angle_incident))

    def get_sigma0(self):
        angle_rad = np.radians(self.angle_local)
        threshold = 5e-3
        min_bins_stable = 20 * self.pad_factor

        df = self.df.loc[0.5:10]
        sigma_raw = (df['copol'] ** 2) * (df.index ** 4) / 0.001
        air = self.get_air(angle_rad, df.index.values)
        delta_r = np.diff(df.index.values).mean()
        sigma = (sigma_raw / air) * delta_r

        sigma_cumsum_log = np.log10(np.cumsum(sigma.values))
        dsigma = np.gradient(sigma_cumsum_log)

        sigma_df = pd.DataFrame({
            'range': df.index.values,
            'sigma_log': sigma_cumsum_log,
            'dsigma': dsigma
        })

        stable_mask = np.abs(sigma_df['dsigma'].values) < threshold
        stable_indices = [i for i, val in enumerate(stable_mask) if val]

        for k, g in groupby(enumerate(stable_indices), lambda x: x[0] - x[1]):
            group = list(map(itemgetter(1), g))
            if len(group) >= min_bins_stable:
                first_stable_idx = group[0]
                stable_value_db = 10 * sigma_df.loc[first_stable_idx, 'sigma_log']
                stable_range = sigma_df.loc[first_stable_idx, 'range']
                break
        else:
            stable_value_db = np.nan
            stable_range = np.nan

        return stable_value_db, stable_range
