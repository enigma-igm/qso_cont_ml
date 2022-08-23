from export_distr import HistogramExporter

dz = 0.1
dlogLv = 0.2

exporter = HistogramExporter(dz, dlogLv)
exporter.saveToFile()