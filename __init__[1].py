# -*- coding: utf-8 -*-
"""
Module d'initialisation pour le plugin ForestDL.
"""

def classFactory(iface):
    """Fonction de fabrique de classe pour le plugin ForestDL.
    
    :param iface: Interface QGIS
    :type iface: QgsInterface
    
    :returns: Instance du plugin ForestDL
    :rtype: ForestDL
    """
    from .forestdl import ForestDL
    return ForestDL(iface)
