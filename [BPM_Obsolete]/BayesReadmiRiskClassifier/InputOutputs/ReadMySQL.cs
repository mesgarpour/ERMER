/// <copyright file="ReadMySQL.cs" company="">
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
using MySql.Data.MySqlClient;
using System.Diagnostics;


public class ReadMySQL
{
    private static string _cs = null;
    private MySqlConnection _conn = null;
    private MySqlDataReader _reader;
    private string _mysqlNullDefault = null;

    /// <summary>
    /// Initiliser
    /// </summary>
    /// <param name="databaseId">The database key name</param>
    public ReadMySQL(
        string databaseName,
        string mysqlServer,
        string mysqlUsername,
        string mysqlPassword,
        string mysqlDatabase,
        string mysqlCmdTimeout,
        string mysqlConnTimeout,
        string mysqlNullDefault)
    {
        TraceListeners.Log(TraceEventType.Information, 0, "ReadMySQL...", false, true);
        _mysqlNullDefault = mysqlNullDefault;

        try
        {
            _cs = @"server=" + @mysqlServer
            + @";userid=" + @mysqlUsername
            + @";password=" + @mysqlPassword
            + @";database=" + @mysqlDatabase
            + @";default command timeout= " + @mysqlCmdTimeout
            + @";Connection Timeout= " + @mysqlConnTimeout
            + @";";
            _conn = new MySqlConnection(_cs);
            _conn.Open();
            TraceListeners.Log(TraceEventType.Information, 0,
                "...Server Version: " + _conn.ServerVersion, false, true);
        }
        catch (MySqlException e)
        {
            TraceListeners.Log(TraceEventType.Error, 0, e.ToString(), true, true);
        }
        finally
        {
            try
            {
                if (_conn != null)
                {
                    _conn.Close();
                }
            }
            catch (Exception ei)
            {
                TraceListeners.Log(TraceEventType.Error, 0, ei.ToString(), true, true);
            }
        }
    }

    public void Read(string query)
    {
        TraceListeners.Log(TraceEventType.Information, 0, "ReadMySQL::Read...", false, true);
        TraceListeners.Log(TraceEventType.Information, 0,
            "...Query: " + query, false, true);
        try
        {
            _conn.Open();
            MySqlCommand cmd = new MySqlCommand(query, _conn);
            _reader = cmd.ExecuteReader();
        }
        catch (Exception e)
        {
            try
            {
                if (_conn != null)
                {
                    _conn.Close();
                }
            }
            catch (Exception ei)
            {
                TraceListeners.Log(TraceEventType.Error, 0, ei.ToString(), true, true);
            }
            // MySqlException
            TraceListeners.Log(TraceEventType.Error, 0, e.ToString(), true, true);
        }
    }

    public string[] GetColumnsNames()
    {
        TraceListeners.Log(TraceEventType.Information, 0, "ReadMySQL::GetColumnsNames...", false, true);
        List<string> ColumnsNames = new List<string>();
        try
        {
            for (int i = 0; i < _reader.FieldCount; i++)
            {
                ColumnsNames.Add(_reader.GetName(i));
            }
        }
        catch (Exception e)
        {
            // MySqlException, System.Data.SqlTypes.SqlNullValueException
            CloseConnection();
            TraceListeners.Log(TraceEventType.Error, 0, e.ToString(), true, true);
        }
        return ColumnsNames.ToArray();
    }

    public Dictionary<string, List<string>> GetColumns(String[] varNames)
    {
        TraceListeners.Log(TraceEventType.Information, 0, "ReadMySQL::GetColumns...", false, true);
        Dictionary<string, List<string>> table = new Dictionary<string, List<String>>();
        foreach (string name in varNames)
        {
            table[name] = new List<string>();
        }

        try
        {
            using (_conn)
            {
                while (_reader.Read())
                {
                    foreach (string varName in varNames)
                    {
                        //TraceListeners.Log(TraceEventType.Information, 0,
                        //    "Get column: " + varName, false);
                        if(_reader[varName] == DBNull.Value)
                        {
                            table[varName].Add(_mysqlNullDefault);
                        }
                        else
                        {
                            table[varName].Add(_reader.GetString(varName));
                        }
                    }
                }
            }
        }
        catch (Exception e)
        {
            // MySqlException, System.Data.SqlTypes.SqlNullValueException
            CloseConnection();
            TraceListeners.Log(TraceEventType.Error, 0, e.ToString(), true, true);
        }
        finally
        {
            try
            {
                if (_reader != null)
                {
                    _reader.Close();
                    _reader = null;
                }
            }
            catch (Exception e)
            {
                TraceListeners.Log(TraceEventType.Error, 0, e.ToString(), true, true);
            }
        }
        return table;
    }


    public void CloseConnection()
    {
        TraceListeners.Log(TraceEventType.Information, 0, "ReadMySQL::CloseConnection...", false, true);
        try
        {
            if (_reader != null)
            {
                _reader.Close();
                _reader = null;
            }

            if (_conn != null)
            {
                _conn.Close();
            }
        }
        catch (Exception e)
        {
            TraceListeners.Log(TraceEventType.Error, 0, e.ToString(), true, true);
        }
    }
}


