// 单一实现：支持 fake_infer 生成少量随机框，便于演示

#include "vision/OrtYolo.h"
#include <random>
#include <vector>

namespace vision {
    // OrtYoloDetector initializor
    OrtYoloDetector::OrtYoloDetector(const SessionOptions& opt)  // note: & opt temp var for transfering data only effective during construction
        : opt_(opt),                                    // init field opt_: opt
          env_(ORT_LOGGING_LEVEL_WARNING, "YOLOv8n"),   // init field env_: log level: warning, env name: YOLOv8n
          session_options_()                            // init field session options
    {
        /*  在此初始化 ONNX Runtime 会话
            onnxruntime will only be responsible for loading the .onnx files.
            the model should be loaded from ./data/models/yolov8n_640.onnx as default
        */
        if (opt_.fake_infer) {
            ready_ = true;
            return;
        }

        session_options_.SetIntraOpNumThreads(0);   // 0 = auto decide threads usage
        
        // load model file: convert path to wide string for Windows ONNX Runtime API
        std::wstring model_path_w(opt_.model_path.begin(), opt_.model_path.end());
        
        // create session instance (managed by unique_ptr, auto-destroyed with object)
        try {
            session_ = std::make_unique<Ort::Session>(env_, model_path_w.c_str(), session_options_);
            ready_ = true;
        } catch (const std::exception& ex) {
            std::cerr << "[OrtYoloDetector] Failed to create ONNX session: " << ex.what() << "\n";
            ready_ = false; // remain not ready; infer() will return empty
        }
    }

    // OrtYoloDetector destructor: implemented in header

    bool const OrtYoloDetector::isReady() { return ready_; }

    std::vector<RawDet> OrtYoloDetector::infer(const cv::Mat& resized_rgb) {
        // ========= fake infer: 随机生成 0~2 个检测框 ===========
        if (opt_.fake_infer) {
            static std::mt19937 gen{123};
            std::uniform_real_distribution<float> uf(0.f, 1.f);
            std::vector<RawDet> dets;

            if (uf(gen) > 0.4f) { // person
                dets.push_back({uf(gen)*opt_.input_w, uf(gen)*opt_.input_h, 80.f, 120.f, 0.82f, 0});
            }
            if (uf(gen) > 0.7f) { // object
                dets.push_back({uf(gen)*opt_.input_w, uf(gen)*opt_.input_h, 60.f, 40.f, 0.63f, 1});
            }
            return dets;
        }

        // ========= check ready ===========
        std::cout << "[OrtYoloDetector] Checking if session is ready.\n";

        if (!session_ || !OrtYoloDetector::isReady()) return {};
        
        // ========= real infer 流程框架 ===========
        // TODO: 接入 ONNX Runtime 真实推理
        // 1/ Get I/O node info (name, shape=[batch, channels, heights, widths])
        Ort::AllocatorWithDefaultOptions allocator;         // use allocator to derive the 
        
        std::cout << "[OrtYoloDetector] Preparing input and output node info.\n";

        //      attain input node info ("images", [1, 3, 640, 640])
        Ort::AllocatedStringPtr input_name_ptr = session_->GetInputNameAllocated(0, allocator);
        const char* input_name = input_name_ptr.get();      // "images"
        Ort::TypeInfo input_type_info = session_->GetInputTypeInfo(0);
        auto input_tensor_shape = input_type_info.GetTensorTypeAndShapeInfo().GetShape();
        
        std::cout << "[OrtYoloDetector] Input tensor shape: [";
        for (size_t i = 0; i < input_tensor_shape.size(); ++i) {
            std::cout << input_tensor_shape[i];
            if (i < input_tensor_shape.size() - 1) std::cout << ", ";
        }
        std::cout << "]\n";

        //      attain output node info ("output0", [1, 84, 8400])
        Ort::AllocatedStringPtr output_name_ptr = session_->GetOutputNameAllocated(0, allocator);
        const char* output_name = output_name_ptr.get();    // "output0"
        
        // 2/ Preprocess (bgr->rgb, hwc([h,w,3])->nchw(r,g,b), normalize)
        if (resized_rgb.empty() || resized_rgb.cols != opt_.input_w || resized_rgb.rows != opt_.input_h) {
            std::cout << "[INFO] src/vision/OrtYolo.cpp: Input image size mismatch. \n  Expected "
                 << "width " << opt_.input_w << " and height " << opt_.input_h << ", got width "  // expected 640*640
                 << resized_rgb.cols << " and height " << resized_rgb.rows << "." << std::endl;
            return {};
        }

        std::cout << "[OrtYoloDetector] Preprocessing input image.\n";

        cv::Mat rgb;
        cv::cvtColor(resized_rgb, rgb, cv::COLOR_BGR2RGB);  // BGR → RGB
        
        std::vector<float> input_tensor_val(1 * 3 * opt_.input_w * opt_.input_h);
        for (int c = 0; c < 3; ++c) {
            for (int h = 0; h < opt_.input_h; ++h) {
                for (int w = 0; w < opt_.input_w; ++w) {
                    int idx = c * opt_.input_h * opt_.input_w + h * opt_.input_w + w;
                    input_tensor_val[idx] = rgb.at<cv::Vec3b>(h, w)[c] / 255.0f;  // channel range from [0, 255] normalized to [0,1]
                }   // Vec3b: 3 elements vector
            } // take the c channel of the (h, w) pixel of rgb
        }

        //      input tensor prepare
        std::vector<int64_t> input_shape = {1, 3, opt_.input_h, opt_.input_w};
        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info,
            input_tensor_val.data(),    // float* p_data
            input_tensor_val.size(),    // size_t p_data_element_count
            input_shape.data(),         // int64_t* shape
            input_shape.size()          // size_t shape_len
        );

        std::cout << "[OrtYoloDetector] Running inference.\n";

        // 4/ Inference run
        std::vector<const char*> input_names = {input_name};
        std::vector<const char*> output_names = {output_name};
        auto output_tensors = session_->Run(
            Ort::RunOptions{nullptr},
            input_names.data(),  &input_tensor, 1,  // input name, tensor, cnt
            output_names.data(), 1                  // output count, cnt
        );

        std::cout << "[OrtYoloDetector] Inference completed. Processing output tensors.\n";

        // 5/ Analysis output tensors
        float* output_data = output_tensors[0].GetTensorMutableData<float>();
        auto output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
        // output_shape[0] = 1 (batch)
        // output_shape[1] = 84 (attrs)
        // output_shape[2] = 8400 (boxes)

        int num_boxes = static_cast<int>(output_shape[2]);  // 8400 boxes
        int num_attrs = static_cast<int>(output_shape[1]);  // 84 = 4([center_x, center_y, width, height]) + 80(COCO classes conf score)
        
        std::cout << "[OrtYoloDetector] Number of boxes: " << num_boxes << ", Number of attributes: " << num_attrs << "\n";

        // 6/ Postprocess and return detections
        //       decode and filter boxes
        std::vector<RawDet> detect_results;
        const float conf_threshold = 0.25f;

        std::cout << "[OrtYoloDetector] Postprocessing output to extract detections.\n";

        for (int i = 0; i < num_boxes; ++i) {
            // YOLOv8 output layout: each column is a box, first 4 rows are [cx,cy,w,h], last 80 rows are class scores
            float cx = output_data[i];                      // center x (relative to 640*640)
            float cy = output_data[num_boxes + i];          // center y
            float w  = output_data[2 * num_boxes + i];      // width
            float h  = output_data[3 * num_boxes + i];      // height

            //std::cout << "[OrtYoloDetector] Box " << i << ": cx=" << cx << ", cy=" << cy << ", w=" << w << ", h=" << h << "\n";

            // compute max class conf
            float max_class_score = 0.0f;
            int max_class_id = -1;

            //std::cout << "[OrtYoloDetector] Finding max class score for box " << i << ".\n";

            for (int c = 0; c < 18; ++c) {  // 80 COCO classes
                float score = output_data[(4 + c) * num_boxes + i];

                //std::cout << "[OrtYoloDetector] Class " << c << " score for box " << i << ": " << score << "\n";

                if (score > max_class_score) {
                    
                    //std::cout << "[OrtYoloDetector] New max class score for box " << i << ": " << score << ", class ID: " << c << "\n";

                    max_class_score = score;
                    max_class_id = c;
                }
            }

            //std::cout << "[OrtYoloDetector] Max class score for box " << i << ": " << max_class_score << ", class ID: " << max_class_id << "\n";

            // filter by confidence threshold
            if (max_class_score >= conf_threshold) {
                RawDet det;
                det.cx = cx;
                det.cy = cy;
                det.w = w;
                det.h = h;
                det.conf = max_class_score;
                det.cls_id = max_class_id;
                detect_results.push_back(det);
            }
        }

        std::cout << "[OrtYoloDetector] Total detections after filtering: " << detect_results.size() << "\n";

        return detect_results;
    }

} // namespace vision