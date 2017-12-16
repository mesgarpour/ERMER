/// <copyright file="Constants.cs" company="">
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


public static class Constants
{
    // appSettings
    public static string MYSQL_SERVER =
        ConfigurationManager.AppSettings["MysqlServer"].ToString();
    public static string MYSQL_USERNAME =
        ConfigurationManager.AppSettings["MysqlUsername"].ToString();
    public static string MYSQL_PASSWORD =
        ConfigurationManager.AppSettings["MysqlPassword"].ToString();
    public static string MYSQL_CMD_TIMEOUT =
        ConfigurationManager.AppSettings["MysqlCmdTimeout"].ToString();
    public static string MYSQL_CONN_TIMEOUT =
        ConfigurationManager.AppSettings["MysqlConnTimeout"].ToString();
    public static string MYSQL_NULL_DEFAULT =
        ConfigurationManager.AppSettings["MysqlNullDefault"].ToString();


    // ModelSettingsMain
    private static string[] MYSQL_DATABASES_MYSQL_STRING =
        ConfigurationManager.AppSettings["DatabasesMysql"].ToString().Split(';');
    public static string[] MYSQL_DATABASES_NAMES =
        ConfigurationManager.AppSettings["DatabasesName"].ToString().Split(';');
    public static Dictionary<String, String> MYSQL_DATABASES_MYSQL =
        new Dictionary<string, string>();
    public static string[] SAMPLING_COMB_NAME =
        ConfigurationManager.AppSettings["SamplingsCombName"].ToString().Split(';');
    public static Dictionary<String, String> SAMPLING_COMB_TRAIN_DB =
        new Dictionary<string, string>();
    private static string[] SAMPLING_COMB_TRAIN_DB_STRING =
        ConfigurationManager.AppSettings["SamplingsCombTrainDatabase"].ToString().Split(';');
    public static Dictionary<String, String> SAMPLING_COMB_TRAIN_SAMPLE =
        new Dictionary<string, string>();
    private static string[] SAMPLING_COMB_TRAIN_SAMPLE_STRING =
        ConfigurationManager.AppSettings["SamplingsCombTrainSample"].ToString().Split(';');
    public static Dictionary<String, String> SAMPLING_COMB_TEST_DB =
        new Dictionary<string, string>();
    private static string[] SAMPLING_COMB_TEST_DB_STRING =
        ConfigurationManager.AppSettings["SamplingsCombTestDatabase"].ToString().Split(';');
    public static Dictionary<String, String> SAMPLING_COMB_TEST_SAMPLE =
        new Dictionary<string, string>();
    private static string[] SAMPLING_COMB_TEST_SAMPLE_STRING =
        ConfigurationManager.AppSettings["SamplingsCombTestSample"].ToString().Split(';');
    

    // ModelSettingsBPM
    public static string[] MODELLING_GROUP =
        ConfigurationManager.AppSettings["ModelingsGroup"].ToString().Split(';');
    public static string[] SUBMODELS =
        ConfigurationManager.AppSettings["Submodels"].ToString().Split(';');
    public static string[] SUBMODELS_NUM_COND =
        ConfigurationManager.AppSettings["SubmodelsNumCond"].ToString().Split(';');
    public static string[] MODEL_PROCEDURES =
        ConfigurationManager.AppSettings["ModelsProcedure"].ToString().Split(';');

    // Other internal constants
    public static string VARS_METADATA_PATH = Environment.CurrentDirectory + "\\"
        + ConfigurationManager.AppSettings["CsvVarMetadataLocalPath"].ToString();
    public static string VARS_METADATA_COL_KEY =
        ConfigurationManager.AppSettings["CsvVarMetadataKeyCol"].ToString();
    public static string VARS_METADATA_LABEL_NAME =
        ConfigurationManager.AppSettings["CsvVarMetadataLabelName"].ToString();
    public static string VARS_METADATA_COL_TYPE = 
        ConfigurationManager.AppSettings["CsvVarMetadataTypeCol"].ToString();
    public static string DIAGNOSTIC_LOG_PATH = Environment.CurrentDirectory + "\\"
        + ConfigurationManager.AppSettings["DiagnosticLogLocalPath"].ToString();
    public static string GENERAL_LOG_PATH = Environment.CurrentDirectory + "\\"
        + ConfigurationManager.AppSettings["GeneralLogLocalPath"].ToString();
    public static string OUTPUT_PATH = Environment.CurrentDirectory + "\\"
        + ConfigurationManager.AppSettings["OutputLocalPath"].ToString() + "\\";

    

    /// <summary>
    /// Static Constructor
    /// </summary>
    public static void Initialise()
    {
        // Validations
        if (MYSQL_DATABASES_MYSQL_STRING.Length != MYSQL_DATABASES_NAMES.Length)
        {
            Console.Error.Write("Invalid setting for DatabasesMySQL or DatabasesNames in the configurations!");
            Console.ReadKey();
            System.Environment.Exit(1);
        }
        if (SAMPLING_COMB_NAME.Length != SAMPLING_COMB_TRAIN_DB_STRING.Length ||
            SAMPLING_COMB_NAME.Length != SAMPLING_COMB_TRAIN_SAMPLE_STRING.Length ||
            SAMPLING_COMB_NAME.Length != SAMPLING_COMB_TEST_DB_STRING.Length ||
            SAMPLING_COMB_NAME.Length != SAMPLING_COMB_TEST_SAMPLE_STRING.Length)
        {
            Console.Error.Write("Invalid setting for sampling combinations in the configurations!");
            Console.ReadKey();
            System.Environment.Exit(1);
        }

        // Furthur initialisations
        for (int i = 0; i < MYSQL_DATABASES_MYSQL_STRING.Length; i++)
        {
            MYSQL_DATABASES_MYSQL.Add(MYSQL_DATABASES_NAMES[i], MYSQL_DATABASES_MYSQL_STRING[i]);
        }
        for (int i = 0; i < SAMPLING_COMB_NAME.Length; i++)
        {
            SAMPLING_COMB_TRAIN_DB.Add(SAMPLING_COMB_NAME[i],
                SAMPLING_COMB_TRAIN_DB_STRING[i]);
            SAMPLING_COMB_TRAIN_SAMPLE.Add(SAMPLING_COMB_NAME[i],
                SAMPLING_COMB_TRAIN_SAMPLE_STRING[i]);
            SAMPLING_COMB_TEST_DB.Add(SAMPLING_COMB_NAME[i],
                SAMPLING_COMB_TEST_DB_STRING[i]);
            SAMPLING_COMB_TEST_SAMPLE.Add(SAMPLING_COMB_NAME[i],
                SAMPLING_COMB_TEST_SAMPLE_STRING[i]);
        }
    }
}
