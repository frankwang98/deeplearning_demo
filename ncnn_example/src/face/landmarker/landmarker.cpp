#include "landmarker.h"

#include "insightface/insightface.h"
#include "zqlandmarker/zqlandmarker.h"

namespace mirror {
Landmarker* ZQLandmarkerFactory::CreateLandmarker() { return new ZQLandmarker(); }

Landmarker* InsightfaceLandmarkerFactory::CreateLandmarker() { return new InsightfaceLandmarker(); }

}  // namespace mirror
