/// <copyright file="ConstantsProfiles.cs" company="">
/// Copyright (c) 2014, 2016 All Right Reserved, https://github.com/mesgarpour/BayesReadmiRiskClassifier
///
/// This source is subject to the The Apache License, Version 2.0.
/// Please see the License.txt file for more information.
/// All other rights reserved.
///
/// THIS CODE AND INFORMATION ARE PROVIDED "AS IS" WITHOUT WARRANTY OF ANY 
/// KIND, EITHER EXPRESSED OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
/// IMPLIED WARRANTIES OF MERCHANTABILITY AND/OR FITNESS FOR A
/// PARTICULAR PURPOSE.
///
/// </copyright>
/// <author>Mohsen Mesgarpour</author>
/// <email>mohsen.meagrpour@email.com</email>
/// <date>2015-12-01</date>
/// <summary></summary>
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Configuration;
using System.Collections.Specialized;
using System.Diagnostics;

public static class ConstantsProfiles
{
    public enum Keys { ModellingSetting, ModelName, ModelProcedure, SamplingsCombName, ModelingsGroup, Submodels, SubmodelsNumCond }
    private static Configuration _config =
        ConfigurationManager.OpenExeConfiguration(ConfigurationUserLevel.None);

    private static ConfigurationSectionGroup _MODELLING_PROFILES =
        _config.SectionGroups["ModellingsProfiles"];
    public static Dictionary<string, Dictionary<Keys, string[]>> DIC =
        new Dictionary<string, Dictionary<Keys, string[]>>();

    public static void Initialise()
    {
        if (_MODELLING_PROFILES != null)
        {
            foreach (ConfigurationSection section in _MODELLING_PROFILES.Sections)
             {
                 string sectionName = section.SectionInformation.SectionName.Split('/')[1];
                 NameValueCollection sectionCollection = (NameValueCollection)ConfigurationManager.GetSection(
                     section.SectionInformation.SectionName);

                 TraceListeners.Log(TraceEventType.Information, 0,
                     "...Reading Profile Configuration: " + sectionName, false, true);

                 DIC[sectionName] = new Dictionary<Keys, string[]>();
                DIC[sectionName][Keys.ModellingSetting] =
                    sectionCollection["ModellingSetting"].ToString().Split(';');
                DIC[sectionName][Keys.ModelName] =
                     sectionCollection["ModelName"].ToString().Split(';');
                 DIC[sectionName][Keys.ModelProcedure] =
                     sectionCollection["ModelProcedure"].ToString().Split(';');
                 DIC[sectionName][Keys.SamplingsCombName] =
                     sectionCollection["SamplingsCombName"].ToString().Split(';');
                 DIC[sectionName][Keys.ModelingsGroup] =
                     sectionCollection["ModelingsGroup"].ToString().Split(';');
                 DIC[sectionName][Keys.Submodels] =
                     sectionCollection["Submodels"].ToString().Split(';');
                 DIC[sectionName][Keys.SubmodelsNumCond] =
                     sectionCollection["SubmodelsNumCond"].ToString().Split(';');
             }
        }
    }
}
