/// <copyright file="ConstantsModellngs.cs" company="">
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



public static class ConstantsModellings
{
    public enum Keys { BPM, HMM }
    public enum KeysModels { SparsityApproxThresh, labels, labelName, NoiseTrain, NoiseTest }

    private static Configuration _config =
        ConfigurationManager.OpenExeConfiguration(ConfigurationUserLevel.None);

    private static ConfigurationSectionGroup _MODELLING_SETTINGS =
        _config.SectionGroups["ModellingsSettings"];
    public static Dictionary<Keys, Dictionary<KeysModels, string[]>> DIC =
        new Dictionary<Keys, Dictionary<KeysModels, string[]>>();

    public static void Initialise()
    {
        if (_MODELLING_SETTINGS != null)
        {
            foreach (ConfigurationSection section in _MODELLING_SETTINGS.Sections)
            {
                string sectionName = section.SectionInformation.SectionName.Split('/')[1];
                NameValueCollection sectionCollection = (NameValueCollection)ConfigurationManager.GetSection(
                    section.SectionInformation.SectionName);

                TraceListeners.Log(TraceEventType.Information, 0,
                    "...Reading Model Configuration: " + sectionName, false, true);


                DIC[(Keys)Enum.Parse(typeof(Keys), sectionName)] = 
                    new Dictionary<KeysModels, string[]>();
                
                switch(sectionName)
                {
                    case("BPM"):
                        SetModelBPM(sectionCollection);
                        break;
                    case ("HMM"):
                        SetModelHMM(sectionCollection);
                        break;
                    default:
                        TraceListeners.Log(TraceEventType.Error, 0,
                            "Invalid model in the modelling settings: " + sectionName, true, true);
                        break;
                }
            }
        }
    }

    private static void SetModelBPM(NameValueCollection sectionCollection)
    {
        DIC[Keys.BPM][KeysModels.SparsityApproxThresh] =
            sectionCollection["SparsityApproxThresh"].ToString().Split(';');
        DIC[Keys.BPM][KeysModels.labels] =
            sectionCollection["labels"].ToString().Split(';');
        DIC[Keys.BPM][KeysModels.labelName] =
            sectionCollection["labelName"].ToString().Split(';');
        DIC[Keys.BPM][KeysModels.NoiseTrain] =
            sectionCollection["NoiseTrain"].ToString().Split(';');
        DIC[Keys.BPM][KeysModels.NoiseTest] =
            sectionCollection["NoiseTest"].ToString().Split(';');
    }

    private static void SetModelHMM(NameValueCollection sectionCollection)
    {
    }
}
