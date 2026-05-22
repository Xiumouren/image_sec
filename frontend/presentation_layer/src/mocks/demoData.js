const scoreMap = {
  drawings: 0.041649,
  hentai: 0.115866,
  neutral: 0.059398,
  porn: 0.741283,
  sexy: 0.041804
};

export const healthSuccessMock = {
  status: "ok",
  model_loaded: true,
  model_path: "C:/Code/net_sec/rebuild/nsfw_mobilenetv2_gradcam_ready.h5",
  labels: ["drawings", "hentai", "neutral", "porn", "sexy"]
};

export const healthFailureMock = {
  status: "degraded",
  model_loaded: false,
  model_path: "C:/Code/net_sec/rebuild/nsfw_mobilenetv2_gradcam_ready.h5",
  labels: []
};

export const detectionSuccessMock = {
  image_name: "porn.jpg",
  predicted_class_index: 3,
  predicted_label: "porn",
  scores: scoreMap,
  explanation: {
    method: "gradcam",
    target_layer: "Conv_1",
    output_path: "C:/Code/net_sec/outputs/explanations/gradcam/manual_test/porn_gradcam_manual.jpg",
    output_relative_path: "explanations/gradcam/manual_test/porn_gradcam_manual.jpg"
  },
  source_analysis: {
    query: {
      sha256: "demo-sha256",
      phash: "demo-phash",
      archived: true,
      record_saved: true,
      asset_reused: false,
      embedding: {
        model_name: "clip:ViT-B/32",
        embedding_dim: 512,
        indexed_asset_count: 3,
        status: "available"
      },
      roi: {
        available: true,
        target_label: "porn",
        bbox: { x_min: 120, y_min: 80, x_max: 420, y_max: 360 },
        coverage_ratio: 0.18,
        roi_url: "/source-assets/roi/demo/demo_roi.jpg",
        embedding: {
          model_name: "clip:ViT-B/32",
          embedding_dim: 512,
          indexed_roi_count: 2,
          status: "available"
        }
      }
    },
    candidates: {
      full_image: [
        {
          asset_id: 1,
          image_name: "archived_demo.jpg",
          relative_path: "images/demo/archived_demo.jpg",
          source_url: "",
          semantic_similarity: 0.86,
          phash_distance: 10,
          phash_similarity: 0.84,
          exact_sha256_match: false,
          credibility_level: "medium",
          needs_human_review: false,
          evidence: [
            { type: "semantic_clip", level: "medium", message: "CLIP semantic retrieval signal.", score: 0.86 },
            { type: "perceptual_hash", level: "medium", message: "pHash distance 10.", score: 0.84 }
          ]
        }
      ],
      roi: [
        {
          asset_id: 1,
          roi_id: 1,
          image_name: "archived_demo.jpg",
          relative_path: "images/demo/archived_demo.jpg",
          source_url: "",
          roi_url: "roi/demo/archived_demo_roi.jpg",
          source_bbox: { x_min: 110, y_min: 82, x_max: 416, y_max: 352 },
          target_label: "porn",
          roi_semantic_similarity: 0.91,
          roi_phash_distance: 4,
          credibility_level: "high",
          needs_human_review: false,
          evidence: [
            { type: "roi_semantic_clip", level: "strong", message: "ROI semantic retrieval signal.", score: 0.91 },
            { type: "roi_perceptual_hash", level: "strong", message: "ROI pHash distance 4.", score: null }
          ]
        }
      ],
      ranking_summary: "Full-image and ROI retrieval agree on the same top archived asset."
    },
    evidence_summary: ["Top candidate archived_demo.jpg has credibility medium."],
    source_credibility_level: "medium",
    review_recommendation: "Review the top candidate and supporting metadata before attribution.",
    errors: [],
    signals: {
      roi_semantic_retrieval: {
        status: "available",
        indexed_roi_count: 2,
        candidate_count: 1,
        error: null
      }
    }
  }
};

export const uiPreviewImage = "/assets/porn.jpg";
export const uiHeatmapImage = "/assets/porn_gradcam_manual.jpg";
