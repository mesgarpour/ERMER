﻿#pragma checksum "..\..\ClinicalTrial.xaml" "{406ea660-64cf-4c82-b6f0-42d48172a799}" "0FEFA0ED12C27CCA2F960C5FE83C7E57"
//------------------------------------------------------------------------------
// <auto-generated>
//     This code was generated by a tool.
//     Runtime Version:4.0.30319.42000
//
//     Changes to this file may cause incorrect behavior and will be lost if
//     the code is regenerated.
// </auto-generated>
//------------------------------------------------------------------------------

using System;
using System.Diagnostics;
using System.Windows;
using System.Windows.Automation;
using System.Windows.Controls;
using System.Windows.Controls.Primitives;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Ink;
using System.Windows.Input;
using System.Windows.Markup;
using System.Windows.Media;
using System.Windows.Media.Animation;
using System.Windows.Media.Effects;
using System.Windows.Media.Imaging;
using System.Windows.Media.Media3D;
using System.Windows.Media.TextFormatting;
using System.Windows.Navigation;
using System.Windows.Shapes;
using System.Windows.Shell;


namespace ClinicalTrial {
    
    
    /// <summary>
    /// MainWindow
    /// </summary>
    public partial class MainWindow : System.Windows.Window, System.Windows.Markup.IComponentConnector {
        
        
        #line 46 "..\..\ClinicalTrial.xaml"
        [System.Diagnostics.CodeAnalysis.SuppressMessageAttribute("Microsoft.Performance", "CA1823:AvoidUnusedPrivateFields")]
        internal System.Windows.Controls.Image imageTreated;
        
        #line default
        #line hidden
        
        
        #line 47 "..\..\ClinicalTrial.xaml"
        [System.Diagnostics.CodeAnalysis.SuppressMessageAttribute("Microsoft.Performance", "CA1823:AvoidUnusedPrivateFields")]
        internal System.Windows.Controls.ProgressBar ProbIsEffectiveSlider;
        
        #line default
        #line hidden
        
        
        #line 57 "..\..\ClinicalTrial.xaml"
        [System.Diagnostics.CodeAnalysis.SuppressMessageAttribute("Microsoft.Performance", "CA1823:AvoidUnusedPrivateFields")]
        internal System.Windows.Controls.Image imagePlacebo;
        
        #line default
        #line hidden
        
        
        #line 66 "..\..\ClinicalTrial.xaml"
        [System.Diagnostics.CodeAnalysis.SuppressMessageAttribute("Microsoft.Performance", "CA1823:AvoidUnusedPrivateFields")]
        internal System.Windows.Controls.ListBox ListBoxTreatedGood;
        
        #line default
        #line hidden
        
        
        #line 78 "..\..\ClinicalTrial.xaml"
        [System.Diagnostics.CodeAnalysis.SuppressMessageAttribute("Microsoft.Performance", "CA1823:AvoidUnusedPrivateFields")]
        internal System.Windows.Controls.ListBox ListBoxPlaceboGood;
        
        #line default
        #line hidden
        
        
        #line 94 "..\..\ClinicalTrial.xaml"
        [System.Diagnostics.CodeAnalysis.SuppressMessageAttribute("Microsoft.Performance", "CA1823:AvoidUnusedPrivateFields")]
        internal System.Windows.Controls.ListBox ListBoxTreatedBad;
        
        #line default
        #line hidden
        
        
        #line 109 "..\..\ClinicalTrial.xaml"
        [System.Diagnostics.CodeAnalysis.SuppressMessageAttribute("Microsoft.Performance", "CA1823:AvoidUnusedPrivateFields")]
        internal System.Windows.Controls.ListBox ListBoxPlaceboBad;
        
        #line default
        #line hidden
        
        
        #line 134 "..\..\ClinicalTrial.xaml"
        [System.Diagnostics.CodeAnalysis.SuppressMessageAttribute("Microsoft.Performance", "CA1823:AvoidUnusedPrivateFields")]
        internal System.Windows.Controls.ListBox TreatedPDF;
        
        #line default
        #line hidden
        
        
        #line 166 "..\..\ClinicalTrial.xaml"
        [System.Diagnostics.CodeAnalysis.SuppressMessageAttribute("Microsoft.Performance", "CA1823:AvoidUnusedPrivateFields")]
        internal System.Windows.Controls.ListBox PlaceboPDF;
        
        #line default
        #line hidden
        
        private bool _contentLoaded;
        
        /// <summary>
        /// InitializeComponent
        /// </summary>
        [System.Diagnostics.DebuggerNonUserCodeAttribute()]
        [System.CodeDom.Compiler.GeneratedCodeAttribute("PresentationBuildTasks", "4.0.0.0")]
        public void InitializeComponent() {
            if (_contentLoaded) {
                return;
            }
            _contentLoaded = true;
            System.Uri resourceLocater = new System.Uri("/ClinicalTrial;component/clinicaltrial.xaml", System.UriKind.Relative);
            
            #line 1 "..\..\ClinicalTrial.xaml"
            System.Windows.Application.LoadComponent(this, resourceLocater);
            
            #line default
            #line hidden
        }
        
        [System.Diagnostics.DebuggerNonUserCodeAttribute()]
        [System.CodeDom.Compiler.GeneratedCodeAttribute("PresentationBuildTasks", "4.0.0.0")]
        [System.ComponentModel.EditorBrowsableAttribute(System.ComponentModel.EditorBrowsableState.Never)]
        [System.Diagnostics.CodeAnalysis.SuppressMessageAttribute("Microsoft.Design", "CA1033:InterfaceMethodsShouldBeCallableByChildTypes")]
        [System.Diagnostics.CodeAnalysis.SuppressMessageAttribute("Microsoft.Maintainability", "CA1502:AvoidExcessiveComplexity")]
        [System.Diagnostics.CodeAnalysis.SuppressMessageAttribute("Microsoft.Performance", "CA1800:DoNotCastUnnecessarily")]
        void System.Windows.Markup.IComponentConnector.Connect(int connectionId, object target) {
            switch (connectionId)
            {
            case 1:
            
            #line 5 "..\..\ClinicalTrial.xaml"
            ((ClinicalTrial.MainWindow)(target)).KeyDown += new System.Windows.Input.KeyEventHandler(this.Window_KeyDown);
            
            #line default
            #line hidden
            return;
            case 2:
            this.imageTreated = ((System.Windows.Controls.Image)(target));
            return;
            case 3:
            this.ProbIsEffectiveSlider = ((System.Windows.Controls.ProgressBar)(target));
            return;
            case 4:
            this.imagePlacebo = ((System.Windows.Controls.Image)(target));
            return;
            case 5:
            
            #line 65 "..\..\ClinicalTrial.xaml"
            ((System.Windows.Shapes.Rectangle)(target)).MouseLeftButtonDown += new System.Windows.Input.MouseButtonEventHandler(this.Rectangle_MouseLeftButtonDown);
            
            #line default
            #line hidden
            
            #line 65 "..\..\ClinicalTrial.xaml"
            ((System.Windows.Shapes.Rectangle)(target)).MouseMove += new System.Windows.Input.MouseEventHandler(this.Rectangle_MouseMove);
            
            #line default
            #line hidden
            return;
            case 6:
            this.ListBoxTreatedGood = ((System.Windows.Controls.ListBox)(target));
            return;
            case 7:
            
            #line 77 "..\..\ClinicalTrial.xaml"
            ((System.Windows.Shapes.Rectangle)(target)).MouseLeftButtonDown += new System.Windows.Input.MouseButtonEventHandler(this.Rectangle_MouseLeftButtonDown);
            
            #line default
            #line hidden
            
            #line 77 "..\..\ClinicalTrial.xaml"
            ((System.Windows.Shapes.Rectangle)(target)).MouseMove += new System.Windows.Input.MouseEventHandler(this.Rectangle_MouseMove);
            
            #line default
            #line hidden
            return;
            case 8:
            this.ListBoxPlaceboGood = ((System.Windows.Controls.ListBox)(target));
            return;
            case 9:
            
            #line 86 "..\..\ClinicalTrial.xaml"
            ((System.Windows.Documents.Hyperlink)(target)).Click += new System.Windows.RoutedEventHandler(this.Reset_Clicked);
            
            #line default
            #line hidden
            return;
            case 10:
            
            #line 93 "..\..\ClinicalTrial.xaml"
            ((System.Windows.Shapes.Rectangle)(target)).MouseLeftButtonDown += new System.Windows.Input.MouseButtonEventHandler(this.Rectangle_MouseLeftButtonDown);
            
            #line default
            #line hidden
            
            #line 93 "..\..\ClinicalTrial.xaml"
            ((System.Windows.Shapes.Rectangle)(target)).MouseMove += new System.Windows.Input.MouseEventHandler(this.Rectangle_MouseMove);
            
            #line default
            #line hidden
            return;
            case 11:
            this.ListBoxTreatedBad = ((System.Windows.Controls.ListBox)(target));
            return;
            case 12:
            
            #line 108 "..\..\ClinicalTrial.xaml"
            ((System.Windows.Shapes.Rectangle)(target)).MouseLeftButtonDown += new System.Windows.Input.MouseButtonEventHandler(this.Rectangle_MouseLeftButtonDown);
            
            #line default
            #line hidden
            
            #line 108 "..\..\ClinicalTrial.xaml"
            ((System.Windows.Shapes.Rectangle)(target)).MouseMove += new System.Windows.Input.MouseEventHandler(this.Rectangle_MouseMove);
            
            #line default
            #line hidden
            return;
            case 13:
            this.ListBoxPlaceboBad = ((System.Windows.Controls.ListBox)(target));
            return;
            case 14:
            this.TreatedPDF = ((System.Windows.Controls.ListBox)(target));
            return;
            case 15:
            this.PlaceboPDF = ((System.Windows.Controls.ListBox)(target));
            return;
            }
            this._contentLoaded = true;
        }
    }
}
