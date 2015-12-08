#include "itkImage.h"
#include "itkRandomImageSource.h"
#include "itkScalarImageToTextureFeaturesFilter.h"
#include "itkImageFileReader.h"
 
typedef itk::Image<float, 3> InternalImageType;

 
int main(int argc, char *argv[])
{

  if(argc < 3)
    {
    std::cerr << "Usage: " << argv[0] << " Required image.nrrd" << std::endl;
    return EXIT_FAILURE;
    }
  
  std::string fileName = argv[1];
  std::string outFile = argv[2];
  typedef itk::ImageFileReader<InternalImageType> ReaderType;
  ReaderType::Pointer reader=ReaderType::New();
  reader->SetFileName(fileName);
  reader->Update();
  InternalImageType::Pointer image=reader->GetOutput();
 
  typedef itk::Statistics::ScalarImageToTextureFeaturesFilter<InternalImageType> TextureFilterType;
  TextureFilterType::Pointer textureFilter = TextureFilterType::New();
  textureFilter->SetNumberOfBinsPerAxis(16);
  textureFilter->SetPixelValueMinMax(0, 255);
  textureFilter->FastCalculationsOn();
  textureFilter->SetInput(image);
  textureFilter->Update();
 
  const TextureFilterType::FeatureValueVector* output = textureFilter->GetFeatureMeans();
  // defaults to: {Energy, Entropy, InverseDifferenceMoment, Inertia, ClusterShade, ClusterProminence
  for(unsigned int i = 0; i < output->Size(); ++i)
    {
    std::cout << (*output)[i] << std::endl;
    }
 
  return EXIT_SUCCESS;
}