using System;
using System.Windows.Forms;

namespace ImageClassifier
{
	static class Program
	{
		/// <summary>
		/// The main entry point for the application.
		/// </summary>
		[STAThread]
		static void Main()
		{
            // First compute features for all the images.
            // Comment the next 2 lines out if the images have not changed and you don't want to re-compute the features each run.
            var features = new ImageFeatures();
		    features.ComputeImageFeatures();

			Application.EnableVisualStyles();
			Application.SetCompatibleTextRenderingDefault(false);
			ItemsModel model = new ItemsModel();
			model.PopulateFromStringsAndVectors(Form1.ReadLines(model.form1.folder + "Images.txt"), model.form1.data);
			ClassifierView cv = new ClassifierView();
			cv.DataContext = model;
			cv.ShowInForm("Image Classifer using Infer.NET");
		}
	}
}