/// <copyright file="ReadCsvMetadata.cs" company="">
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
using System.IO;
using System.Diagnostics;

class ReadCsvMetdata
{

    private Dictionary<string, string[]> _table;
    private Dictionary<string, int> _colKey;

    public ReadCsvMetdata(string path, string keyCol)
    {
        TraceListeners.Log(TraceEventType.Information, 0, "ReadCsvMetdata...", false, true);
        Set(path, keyCol);
    }

    public String Get(string keyRow, string keyCol)
    {
        if (!_table.ContainsKey(keyRow) ||
            !(_table[keyRow].Length >= _colKey[keyCol]))
        {
            return null;
        }
        else
        {
            return _table[keyRow][_colKey[keyCol]];
        }
    }

    private Dictionary<string, string[]> Set(string path, string keyCol)
    {
        TraceListeners.Log(TraceEventType.Information, 0, "ReadCsvMetdata::Set...", false, true);
        _table = new Dictionary<string, string[]>();
        _colKey = new Dictionary<string, int>();
        string line;
        string[] row;
        try
        {
            using (StreamReader reader = new StreamReader(path))
            {
                if ((line = reader.ReadLine()) != null)
                {
                    row = line.Split(',');
                    for (int i = 0; i < row.Length; i++)
                    {
                        _colKey.Add(row[i], i);
                    }
                }

                while ((line = reader.ReadLine()) != null)
                {
                    row = line.Split(',');
                    if (!_table.ContainsKey(row[_colKey[keyCol]]))
                    {
                        _table.Add(row[_colKey[keyCol]], row);
                    }
                    else
                    {
                        TraceListeners.Log(TraceEventType.Warning, 0,
                            "Duplicate metadata: " + row[_colKey[keyCol]], false, false);
                    }
                }
            }
        }
        catch (Exception ex)
        {
            TraceListeners.Log(TraceEventType.Error, 0, ex.ToString(), true, true);
        }
        return _table;
    }
}
