/// <copyright file="TraceLiseners.cs" company="">
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
using System.IO;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Diagnostics;

public static class TraceListeners
{
    private static TraceSource _ts = null;
    private static string _generalLogPath = null;
    private static string _diagnositicLogPath = null;
    private static readonly string _messageFormat = "[*] {0}({1}): {2}";
    private static bool _verboseLog = false;

    public static void Initialise(
        bool verboseLog,
        string generalLogPath,
        string diagnositicLogPath)
    {
        _verboseLog = verboseLog;
        _generalLogPath = generalLogPath;
        _diagnositicLogPath = diagnositicLogPath;

        //verbose logs
        if (verboseLog)
        {
            //trace source object
            _ts = new TraceSource("myTraceSource");
            DelimitedListTraceListener dt = null;
            EventLogTraceListener et = null;

            try
            {
                // set the log file
                using (File.Create(_generalLogPath));

                //listeners
                dt = new DelimitedListTraceListener(_diagnositicLogPath);
                et = new EventLogTraceListener("myappp");
            }
            catch (Exception e)
            {
                Console.Error.Write("Invalid configurations!");
                Console.ReadKey();
                System.Environment.Exit(1);
            }

            //configure
            _ts.Switch = new SourceSwitch("mySwitch");
            _ts.Switch.Level = SourceLevels.Warning;

            //configure
            dt.Delimiter = "|";
            dt.TraceOutputOptions = TraceOptions.DateTime | TraceOptions.Timestamp | TraceOptions.Callstack;

            //Adding the trace listeners
            _ts.Listeners.Clear();
            _ts.Listeners.Add(dt);
            _ts.Listeners.Add(et);

            //setting autoflush to save automatically
            Trace.AutoFlush = true;
            _ts.Flush();
        }
    }

    public static void Log(TraceEventType eventType, int id, string message, bool toExit, bool msgPrint)
    {
        string messageFormated = string.Format(_messageFormat, eventType.ToString(), CurrentDateTime(), message);

        //verbose log
        if (_verboseLog)
        {
            // write diagnostics
            _ts.TraceEvent(eventType, id, "\n" + message);
            _ts.Flush();

            // write general logs
            using (StreamWriter sw = File.AppendText(_generalLogPath))
            {
                sw.WriteLine(messageFormated);
            }
        }
        
        // write console logs
        if (msgPrint)
        {
            //Console.WriteLine("*** {0} - [id:{1}]: {2}", eventType.ToString(), id.ToString(), message);
            Console.WriteLine(messageFormated);
        }

        // exit condition
        if (toExit)
        {
            Console.ReadKey();
            System.Environment.Exit(1);
            return;
        }
    }

    public static void Test()
    {
        _ts.TraceEvent(TraceEventType.Warning, 0, "testing: warning msg!");
        _ts.TraceEvent(TraceEventType.Error, 0, "testing: error msg!");
        _ts.TraceEvent(TraceEventType.Information, 0, "testing: information msg!");
        _ts.TraceEvent(TraceEventType.Critical, 0, "testing: critical msg!");
    }

    private static string CurrentDateTime()
    {

      DateTime localDate = DateTime.Now;
      return localDate.ToString("MM/dd/yyy hh:mm:ss.fff");
    }
}
