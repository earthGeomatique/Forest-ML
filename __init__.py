from .forestdl_plugin import ForestDLPlugin

def classFactory(iface):
    return ForestDLPlugin(iface)
