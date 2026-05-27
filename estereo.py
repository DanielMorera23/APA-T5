"""
estereo.py - Manejo de señales estéreo en ficheros WAVE

Autor: Daniel Morera Torra

Descripción:
    Este módulo proporciona funciones para el manejo de los canales de una señal
    estéreo almacenada en ficheros WAVE con codificación PCM lineal de 16 bits.
    Incluye funciones para:
      - Separar o combinar canales estéreo/mono (estereo2mono, mono2estereo).
      - Codificar una señal estéreo en 32 bits para compatibilidad monofónica
        (codEstereo) y decodificarla de vuelta a estéreo (decEstereo).

    Solo se usa la biblioteca estándar `struct`. No se emplean bibliotecas
    externas de audio.
"""

import struct


# ---------------------------------------------------------------------------
# Utilidades de cabecera WAVE
# ---------------------------------------------------------------------------

def _leer_cabecera(f):
    riff_header = f.read(12)
    if len(riff_header) < 12:
        raise ValueError("Fichero demasiado corto para ser un WAVE válido.")

    riff_id, riff_size, wave_id = struct.unpack('<4sI4s', riff_header)

    if riff_id != b'RIFF':
        raise ValueError(f"No es un fichero RIFF (cabecera: {riff_id}).")
    if wave_id != b'WAVE':
        raise ValueError(f"No es un fichero WAVE (tipo: {wave_id}).")

    fmt_header = f.read(8)
    if len(fmt_header) < 8:
        raise ValueError("Cabecera fmt incompleta.")

    fmt_id, fmt_size = struct.unpack('<4sI', fmt_header)

    if fmt_id != b'fmt ':
        raise ValueError(f"Se esperaba el cacho 'fmt ', encontrado: {fmt_id}.")

    fmt_data = f.read(fmt_size)
    if len(fmt_data) < 16:
        raise ValueError("Datos del cacho fmt insuficientes.")

    (audio_format, num_channels, sample_rate,
     byte_rate, block_align, bits_per_sample) = struct.unpack('<HHIIHH', fmt_data[:16])

    if audio_format != 1:
        raise ValueError(
            f"Solo se admite PCM lineal (formato 1). Formato encontrado: {audio_format}."
        )

    data_header = f.read(8)
    if len(data_header) < 8:
        raise ValueError("Cabecera data incompleta.")

    data_id, data_size = struct.unpack('<4sI', data_header)

    if data_id != b'data':
        raise ValueError(f"Se esperaba el cacho 'data', encontrado: {data_id}.")

    return {
        'riff_id': riff_id,
        'riff_size': riff_size,
        'wave_id': wave_id,
        'fmt_id': fmt_id,
        'fmt_size': fmt_size,
        'audio_format': audio_format,
        'num_channels': num_channels,
        'sample_rate': sample_rate,
        'byte_rate': byte_rate,
        'block_align': block_align,
        'bits_per_sample': bits_per_sample,
        'data_id': data_id,
        'data_size': data_size,
    }


def _escribir_cabecera(f, num_channels, sample_rate, bits_per_sample, data_size):
    block_align = num_channels * bits_per_sample // 8
    byte_rate = sample_rate * block_align
    fmt_size = 16
    riff_size = 4 + 8 + fmt_size + 8 + data_size

    f.write(struct.pack('<4sI4s', b'RIFF', riff_size, b'WAVE'))
    f.write(struct.pack('<4sI', b'fmt ', fmt_size))
    f.write(struct.pack('<HHIIHH',
                        1,
                        num_channels,
                        sample_rate,
                        byte_rate,
                        block_align,
                        bits_per_sample))
    f.write(struct.pack('<4sI', b'data', data_size))


# ---------------------------------------------------------------------------
# Funciones principales
# ---------------------------------------------------------------------------

def estereo2mono(ficEste, ficMono, canal=2):
    if canal not in (0, 1, 2, 3):
        raise ValueError(f"El parámetro canal debe ser 0, 1, 2 o 3 (recibido: {canal}).")

    with open(ficEste, 'rb') as fe:
        cab = _leer_cabecera(fe)

        if cab['num_channels'] != 2:
            raise ValueError(
                f"Se esperaba un fichero estéreo (2 canales). "
                f"Encontrados: {cab['num_channels']} canales."
            )
        if cab['bits_per_sample'] != 16:
            raise ValueError(
                f"Se esperaban muestras de 16 bits. "
                f"Encontradas: {cab['bits_per_sample']} bits."
            )

        num_muestras = cab['data_size'] // 4
        raw = fe.read(cab['data_size'])

    muestras = struct.unpack(f'<{num_muestras * 2}h', raw)
    izq = muestras[0::2]
    der = muestras[1::2]

    seleccion = {
        0: izq,
        1: der,
        2: tuple((l + r) // 2 for l, r in zip(izq, der)),
        3: tuple((l - r) // 2 for l, r in zip(izq, der)),
    }
    mono = seleccion[canal]

    data_size = num_muestras * 2

    with open(ficMono, 'wb') as fm:
        _escribir_cabecera(fm, 1, cab['sample_rate'], 16, data_size)
        fm.write(struct.pack(f'<{num_muestras}h', *mono))


def mono2estereo(ficIzq, ficDer, ficEste):
    with open(ficIzq, 'rb') as fi:
        cab_i = _leer_cabecera(fi)
        if cab_i['num_channels'] != 1:
            raise ValueError("ficIzq debe ser un fichero mono (1 canal).")
        if cab_i['bits_per_sample'] != 16:
            raise ValueError("ficIzq debe tener muestras de 16 bits.")
        raw_i = fi.read(cab_i['data_size'])

    with open(ficDer, 'rb') as fd:
        cab_d = _leer_cabecera(fd)
        if cab_d['num_channels'] != 1:
            raise ValueError("ficDer debe ser un fichero mono (1 canal).")
        if cab_d['bits_per_sample'] != 16:
            raise ValueError("ficDer debe tener muestras de 16 bits.")
        if cab_d['sample_rate'] != cab_i['sample_rate']:
            raise ValueError(
                "Los dos ficheros mono deben tener la misma frecuencia de muestreo."
            )
        raw_d = fd.read(cab_d['data_size'])

    n_i = cab_i['data_size'] // 2
    n_d = cab_d['data_size'] // 2
    num_muestras = min(n_i, n_d)

    izq = struct.unpack(f'<{n_i}h', raw_i)[:num_muestras]
    der = struct.unpack(f'<{n_d}h', raw_d)[:num_muestras]

    intercalado = [val for par in zip(izq, der) for val in par]
    data_size = num_muestras * 4

    with open(ficEste, 'wb') as fe:
        _escribir_cabecera(fe, 2, cab_i['sample_rate'], 16, data_size)
        fe.write(struct.pack(f'<{num_muestras * 2}h', *intercalado))


def codEstereo(ficEste, ficCod):
    with open(ficEste, 'rb') as fe:
        cab = _leer_cabecera(fe)

        if cab['num_channels'] != 2:
            raise ValueError(
                f"Se esperaba un fichero estéreo (2 canales). "
                f"Encontrados: {cab['num_channels']} canales."
            )
        if cab['bits_per_sample'] != 16:
            raise ValueError(
                f"Se esperaban muestras de 16 bits. "
                f"Encontradas: {cab['bits_per_sample']} bits."
            )

        num_muestras = cab['data_size'] // 4
        raw = fe.read(cab['data_size'])

    muestras = struct.unpack(f'<{num_muestras * 2}h', raw)
    izq = muestras[0::2]
    der = muestras[1::2]

    cod = [
        (((l + r) // 2) << 16) | (((l - r) // 2) & 0xFFFF)
        for l, r in zip(izq, der)
    ]

    data_size = num_muestras * 4

    with open(ficCod, 'wb') as fc:
        _escribir_cabecera(fc, 1, cab['sample_rate'], 32, data_size)
        fc.write(struct.pack(f'<{num_muestras}i', *cod))


def decEstereo(ficCod, ficEste):
    with open(ficCod, 'rb') as fc:
        cab = _leer_cabecera(fc)

        if cab['num_channels'] != 1:
            raise ValueError(
                f"Se esperaba un fichero mono (1 canal). "
                f"Encontrados: {cab['num_channels']} canales."
            )
        if cab['bits_per_sample'] != 32:
            raise ValueError(
                f"Se esperaban muestras de 32 bits. "
                f"Encontradas: {cab['bits_per_sample']} bits."
            )

        num_muestras = cab['data_size'] // 4
        raw = fc.read(cab['data_size'])

    cod = struct.unpack(f'<{num_muestras}i', raw)

    semisuma       = [v >> 16 for v in cod]
    semidiferencia = [struct.unpack('<h', struct.pack('<H', v & 0xFFFF))[0]
                      for v in cod]

    izq = [max(-32768, min(32767, s + d)) for s, d in zip(semisuma, semidiferencia)]
    der = [max(-32768, min(32767, s - d)) for s, d in zip(semisuma, semidiferencia)]

    intercalado = [val for par in zip(izq, der) for val in par]
    data_size = num_muestras * 4

    with open(ficEste, 'wb') as fe:
        _escribir_cabecera(fe, 2, cab['sample_rate'], 16, data_size)
        fe.write(struct.pack(f'<{num_muestras * 2}h', *intercalado))