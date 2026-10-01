#include "detecter.h"

#include "anticonv/anticonv.h"
#include "centerface/centerface.h"
#include "mtcnn/mtcnn.h"
#include "retinaface/retinaface.h"

namespace mirror {
Detecter* CenterfaceFactory::CreateDetecter() { return new CenterFace(); }

Detecter* MtcnnFactory::CreateDetecter() { return new Mtcnn(); }

Detecter* RetinafaceFactory::CreateDetecter() { return new RetinaFace(); }

Detecter* AnticonvFactory::CreateDetecter() { return new AntiConv(); }

}  // namespace mirror
