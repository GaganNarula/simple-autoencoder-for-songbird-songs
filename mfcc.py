def hztomel(f):
    import numpy as np
    return 2591 * np.log10(1 + f / 700.0)


def meltohz(mel):
    return 700 * (10 ** (mel / 2590.0) - 1)


def mel_filter(nb_spec=120, nb_filt=26, fl=512, fh=8192):
    import numpy as np
    ml = hztomel(fl)
    mh = hztomel(fh)
    mel = np.linspace(ml, mh, nb_filt + 2)
    hz = meltohz(mel)
    hz_bin = (np.int_((hz - fl) / ((fh - fl) / nb_spec)))
    mfbank = np.zeros((nb_filt, nb_spec))
    for i in np.arange(0, nb_filt):
        for j in np.arange(hz_bin[i], hz_bin[i + 1]):
            mfbank[i, j] = np.divide(np.float_(j - hz_bin[i]), np.float_(hz_bin[i + 1] - hz_bin[i]))
        for j in np.arange(hz_bin[i + 1], hz_bin[i + 2]):
            mfbank[i, j] = np.divide(np.float_(hz_bin[i + 2] - j), np.float_(hz_bin[i + 2] - hz_bin[i + 1]))
    return mfbank


def logfbank(spec, mel):
    import numpy as np
    nb_samples, nb_time, nb_spec = np.shape(spec)
    nb_filter, tmp = np.shape(mel)
    if tmp != nb_spec:
        return None
    fbank = np.zeros((nb_samples, nb_time, nb_filter))
    for i in np.arange(0, nb_samples):
        for j in np.arange(0, nb_time):
            for k in np.arange(0, nb_filter):
                fbank[i, j, k] = np.sum(np.multiply(spec[i, j, :], mel[k, :]))
    fbank = np.log(fbank)
    return fbank


def cep(logfbank):
    from scipy.fftpack import dct
    import numpy as np
    nb_samples, nb_time, nb_filter = np.shape(logfbank)
    nb_cep = np.floor(nb_filter / 2)
    cepdata = np.zeros((nb_samples, nb_time, nb_cep))
    for i in np.arange(0, nb_samples):
        for j in np.arange(0, nb_time):
            cepdata[i, j, :] = dct(logfbank[i, j, :])[0:nb_cep]
    return cepdata


def data_rshp_exp(data, nb_time=20, nb_spec=120):
    import numpy as np
    nb_samples, nb_spec_time = np.shape(data)
    if nb_spec_time != nb_time * nb_spec:
        return None
    data_reshape = np.reshape(np.asarray(data), (nb_samples, nb_time, nb_spec))
    return data_reshape


def data_rshp_cmprss(data):
    import numpy as np
    nb_samples, nb_time, nb_spec = np.shape(data)
    data_reshape = np.reshape(np.asarray(data), (nb_samples, nb_time * nb_spec))
    return data_reshape



