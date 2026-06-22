

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from ..input.feature_builder import FeatureBuilder
from ...utils.text_encoder import TextEncoder
from ..input.image_encoder import ImageEncoder
from .target_builder import TargetBuilder, compute_popularity_score
from ...pa_core import (
    QuadRegressionHead, PAClassificationHead, TARGET_NAMES,
    UncertaintyWeightedTrainer,
)


class TwitterPAPipeline:
    def __init__(
        self,
        text_backend: str = "tfidf",
        image_backend: str = "zeros",
        text_dim: int = 768,
        image_dim: int = 512,
        regression_C: float = 1.0,
        classification_C: float = 1.0,
        epsilon: float = 0.1,
        add_metadata: bool = True,
        add_engagement: bool = False,
        add_author: bool = False,
        add_age: bool = False,
        side_scale: float = 1.0,
        cache_dir: str = "data/embeddings",
        algo: str = "pa",
        sgd_l2: float = 0.0,
    ):
        text_enc = TextEncoder(backend=text_backend, text_dim=text_dim)
        image_enc = ImageEncoder(backend=image_backend, image_dim=image_dim,
                                  cache_dir=cache_dir)

        self.feature_builder = FeatureBuilder(
            text_encoder=text_enc,
            image_encoder=image_enc,
            add_metadata=add_metadata,
            add_engagement=add_engagement,
            add_author=add_author,
            add_age=add_age,
            side_scale=side_scale,
        )
        self.target_builder = TargetBuilder()
        self.regression_head = QuadRegressionHead(C=regression_C, epsilon=epsilon)
        self.classification_head = PAClassificationHead(C=classification_C)
        self._C = regression_C
        self._cls_C = classification_C       # C riêng cho classifier (decoupled)
        self._epsilon = epsilon
        self._algo = algo                    # "pa" | "sgd" — chọn luật cập nhật online
        self._sgd_l2 = sgd_l2                # L2 weight decay (chỉ dùng cho SGD)
        self._trained = False
        self.uw_trainer: UncertaintyWeightedTrainer | None = None  # bộ học multi-task 1 total loss
        self.train_history: list[dict] = []      # loss từng bước (fit_temporal_online)
        self.train_loss_summary: dict = {}        # tóm tắt loss toàn cục

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> "TwitterPAPipeline":
        """
        Full online training trên DataFrame df.

        Required columns in df:
          text, likes, comments, reposts, views
        Optional:
          image_path, created_at
        """
        Z = self.feature_builder.fit_transform(df)
        Y_reg, y_cls, _ = self.target_builder.fit_transform(df)

        self.regression_head.run_online(Z, Y_reg)
        self.classification_head.run_online(Z, y_cls)

        self._trained = True
        return self

    def fit_temporal(self, df: pd.DataFrame) -> "TwitterPAPipeline":
        from .temporal_sampler import TemporalSampler

        sampler = TemporalSampler()
        df_t, df_t1 = sampler.create_pairs(df)

        # Feature: z_t = [text; image; engagement_t; delta_t]
        # Tạm bật add_engagement để builder thêm engagement vào z
        self.feature_builder.add_engagement = True
        Z = self.feature_builder.fit_transform(df_t)

        # Target: engagement_{t+1}
        self.target_builder.fit(df_t1)
        Y_reg = self.target_builder.regression_targets(df_t1)
        y_cls = self.target_builder.classification_labels(df_t1)

        self.regression_head.run_online(Z, Y_reg)
        self.classification_head.run_online(Z, y_cls)

        self._trained = True
        self._temporal_sampler = sampler
        return self

    def fit_temporal_online(
        self, df_temporal: pd.DataFrame, verbose: bool = True, log_every: int = 2000
    ) -> "TwitterPAPipeline":
        """
        Online (streaming) temporal training — ĐÚNG tinh thần online learning.

        Khác với fit_temporal() (gom hết snapshot vào 1 batch fit_transform rồi
        mới stream), hàm này duyệt LẦN LƯỢT:

            for post in posts:                 # từng post một
                for (snap_t -> snap_{t+1}):    # trong post, từng snapshot theo thời gian
                    z   = scaler.partial_fit(z_raw).transform(z_raw)   # scaler chạy online (causal)
                    ŷ   = head.predict(z)        # PREDICT trước → đo loss → rồi mới update
                    head.partial_fit(z, y_{t+1})

        Không có bước "nhét hết N snapshot vào fit_transform" nên scaler không
        nhìn thấy thống kê của các post/snapshot tương lai (no look-ahead leak).

        Loss = MỘT total loss duy nhất (multi-task uncertainty weighting,
        Kendall et al. 2018), gộp 4 regression head + 1 classification head:

            L_total = (1/2σ₁²)·L_reg + (1/σ₂²)·L_cls + log σ₁ + log σ₂

          L_reg = mean epsilon-insensitive loss của 4 target (log-space)
          L_cls = mean multiclass hinge loss / (K-1)
          σ₁, σ₂ học online qua EMA của loss → task nhiễu hơn được weight nhỏ hơn.

        Mỗi bước lưu l_total (+ thành phần l_reg, l_cls, σ) vào self.train_history.
        Việc tối ưu cũng được điều khiển bởi total loss: step size mỗi head
        scale theo C_eff (C_eff_reg = C/2σ₁², C_eff_cls = C/σ₂²).

        Yêu cầu: precomputed embeddings đã set qua set_text/image_embeddings()
        (để _build_raw chỉ tra cứu, không phải fit encoder giữa chừng).
        """
        from .temporal_sampler import TemporalSampler

        fb = self.feature_builder
        fb.add_engagement = True          # z_t phải chứa engagement hiện tại
        # Scaler chạy online: reset rồi cập nhật dần bằng partial_fit
        fb.scaler = StandardScaler()
        fb._fitted = True                  # đánh dấu để transform()/predict() dùng được sau này

        sampler = TemporalSampler()
        eps = self._epsilon
        C = self._C                 # C cho regression heads
        C_cls = self._cls_C         # C cho classification head (decoupled)
        K = self.classification_head.n_classes

        self.train_history = []            # reset lịch sử loss
        history = self.train_history
        trainer: UncertaintyWeightedTrainer | None = None   # tạo lazy khi biết n_features

        n_pairs = 0
        n_cls_correct = 0
        started = False
        run = {"total": 0.0, "reg": 0.0, "cls": 0.0, "n": 0}   # running sums để in

        if verbose:
            print(f"  {'step':>6} | {'L_total':>9} | {'L_reg':>8} | {'L_cls':>8} | "
                  f"{'sigma_reg':>9} | {'sigma_cls':>9} | {'run_acc':>7}")

        # sort=False → giữ đúng thứ tự post trong dataset
        for post_idx, (_post_id, group) in enumerate(df_temporal.groupby("post_id", sort=False)):
            if len(group) < 2:
                continue

            # Cặp (t -> t+1) TRONG riêng post này, đã sort theo crawl_time
            df_t, df_t1 = sampler.create_pairs(group)
            Z_raw = fb._build_raw(df_t, fit=False)     # embedding precomputed -> chỉ lookup
            if not fb.feature_dim:
                fb.feature_dim = Z_raw.shape[1]
            Y_reg = self.target_builder.regression_targets(df_t1)
            y_cls = self.target_builder.classification_labels(df_t1)

            # Stream từng snapshot transition của post
            for i in range(len(Z_raw)):
                x_raw = Z_raw[i: i + 1]
                fb.scaler.partial_fit(x_raw)                     # cập nhật mean/var chạy
                x = fb.scaler.transform(x_raw)                   # z-score
                x = fb._apply_side_scale(x).astype(np.float64)[0]   # boost side feats -> 1D cho PA

                # Tạo trainer 1 lần (đã biết feature_dim)
                if trainer is None:
                    trainer = UncertaintyWeightedTrainer(
                        n_features=x.shape[0], C=C, epsilon=eps, n_classes=K,
                        algo=self._algo, sgd_l2=self._sgd_l2,
                    )
                    self.uw_trainer = trainer

                y_reg_i = Y_reg[i]
                y_true = int(y_cls[i])

                if not started:
                    # Warm-start sample đầu tiên: chỉ update, chưa có loss để đo
                    for j, name in enumerate(TARGET_NAMES):
                        trainer.reg_models[name].update(x, float(y_reg_i[j]), C)
                    trainer.cls_model.update(x, y_true, C_cls)
                    started = True
                    rec = {"step": n_pairs, "post_idx": post_idx,
                           "l_total": float("nan"), "l_reg": float("nan"),
                           "l_cls": float("nan"),
                           "sigma_reg": trainer.uw.sigma_reg,
                           "sigma_cls": trainer.uw.sigma_cls}
                    for name in TARGET_NAMES:
                        rec[f"ae_{name}"] = float("nan")
                    history.append(rec)
                    n_pairs += 1
                    continue

                # ===== PREDICT trước (test-then-train), tính loss thành phần =====
                per_target_ae = {}
                reg_losses = 0.0
                for j, name in enumerate(TARGET_NAMES):
                    y_pred_log = trainer.reg_models[name].predict_one(x)
                    ae = abs(float(y_reg_i[j]) - y_pred_log)
                    per_target_ae[name] = ae
                    reg_losses += max(0.0, ae - eps)
                l_reg = reg_losses / len(TARGET_NAMES)

                s = trainer.cls_model.scores(x)
                pred_cls = int(np.argmax(s))
                correct = int(pred_cls == y_true)
                n_cls_correct += correct
                l_cls = sum(
                    max(0.0, 1.0 - s[y_true] + s[c]) for c in range(K) if c != y_true
                ) / (K - 1)

                # ===== MỘT total loss (dùng σ hiện tại) =====
                l_total = float(trainer.uw.total_loss(l_reg, l_cls))

                # ===== Update σ qua EMA → tính C_eff → update các head =====
                trainer.uw.update(l_reg, l_cls)
                c_eff_reg = trainer.uw.c_eff_reg(C)
                c_eff_cls = trainer.uw.c_eff_cls(C_cls)
                for j, name in enumerate(TARGET_NAMES):
                    trainer.reg_models[name].update(x, float(y_reg_i[j]), c_eff_reg)
                trainer.cls_model.update(x, y_true, c_eff_cls)

                # ===== Ghi lịch sử =====
                rec = {"step": n_pairs, "post_idx": post_idx,
                       "l_total": l_total, "l_reg": l_reg, "l_cls": l_cls,
                       "sigma_reg": trainer.uw.sigma_reg,
                       "sigma_cls": trainer.uw.sigma_cls}
                for name in TARGET_NAMES:
                    rec[f"ae_{name}"] = per_target_ae[name]
                history.append(rec)

                run["total"] += l_total
                run["reg"] += l_reg
                run["cls"] += l_cls
                run["n"] += 1
                n_pairs += 1

                if verbose and run["n"] > 0 and n_pairs % log_every == 0:
                    k = run["n"]
                    print(f"  {n_pairs:>6} | {run['total']/k:>9.4f} | {run['reg']/k:>8.4f} | "
                          f"{run['cls']/k:>8.4f} | {trainer.uw.sigma_reg:>9.4f} | "
                          f"{trainer.uw.sigma_cls:>9.4f} | {n_cls_correct/max(n_pairs-1,1):>7.4f}")
                    run = {"total": 0.0, "reg": 0.0, "cls": 0.0, "n": 0}

        # Averaged PA: dùng trung bình trọng số qua toàn stream cho inference
        # (giảm phụ thuộc thứ tự online -> ổn định giữa các fold/split)
        if trainer is not None:
            trainer.finalize()

        self._trained = True
        self._temporal_sampler = sampler

        # Tóm tắt loss toàn cục (bỏ sample warm-start đầu)
        valid = [h for h in history if not np.isnan(h["l_total"])]
        self.train_loss_summary = {
            "n_updates": n_pairs,
            "avg_l_total": float(np.mean([h["l_total"] for h in valid])) if valid else None,
            "avg_l_reg": float(np.mean([h["l_reg"] for h in valid])) if valid else None,
            "avg_l_cls": float(np.mean([h["l_cls"] for h in valid])) if valid else None,
            "final_sigma_reg": trainer.uw.sigma_reg if trainer else None,
            "final_sigma_cls": trainer.uw.sigma_cls if trainer else None,
            "online_train_accuracy": float(n_cls_correct / max(n_pairs - 1, 1)),
        }
        if verbose:
            n_posts = df_temporal["post_id"].nunique()
            sm = self.train_loss_summary
            print(f"  Online updates: {n_pairs} snapshot transitions streamed "
                  f"post-by-post across {n_posts} posts")
            print(f"  [Total loss] L_total={sm['avg_l_total']:.4f}  "
                  f"(L_reg={sm['avg_l_reg']:.4f}, L_cls={sm['avg_l_cls']:.4f})  "
                  f"σ_reg={sm['final_sigma_reg']:.3f}  σ_cls={sm['final_sigma_cls']:.3f}  "
                  f"online_acc={sm['online_train_accuracy']:.4f}")
        return self

    def partial_fit(self, df: pd.DataFrame) -> "TwitterPAPipeline":
        """
        Online update với batch dữ liệu mới (sau khi đã fit() lần đầu).
        Không fit lại scaler hay encoder.
        """
        if not self._trained:
            return self.fit(df)

        Z = self.feature_builder.transform(df)
        Y_reg = self.target_builder.regression_targets(df)
        y_cls = self.target_builder.classification_labels(df)

        # Cập nhật từng sample theo thứ tự
        for i in range(len(Z)):
            x = Z[i: i + 1]
            for j, name in enumerate(TARGET_NAMES):
                self.regression_head.models[name].partial_fit(x, [float(Y_reg[i, j])])
            self.classification_head.model.partial_fit(x, [int(y_cls[i])])

        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Inference trên DataFrame mới.

        Returns DataFrame với các cột:
          pred_likes, pred_comments, pred_reposts, pred_views,
          pred_label, pred_label_name, predicted_popularity_score
        """
        if not self._trained:
            raise RuntimeError("Call fit() before predict().")

        Z = self.feature_builder.transform(df)
        if self.uw_trainer is not None:
            # Model online đa nhiệm (1 total loss): dùng UncertaintyWeightedTrainer
            preds = self.uw_trainer.predict(Z)
            reg_preds = {name: preds[name] for name in TARGET_NAMES}
            cls_preds = preds["label"]
        else:
            reg_preds = self.regression_head.predict(Z)
            cls_preds = self.classification_head.predict(Z)

        result = df.copy().reset_index(drop=True)
        result["pred_likes"] = reg_preds["likes"]
        result["pred_comments"] = reg_preds["comments"]
        result["pred_reposts"] = reg_preds["reposts"]
        result["pred_views"] = reg_preds["views"]
        result["pred_label"] = cls_preds
        result["pred_label_name"] = [
            self.classification_head.label_name(int(l)) for l in cls_preds
        ]
        result["predicted_popularity_score"] = compute_popularity_score(
            reg_preds["likes"],
            reg_preds["comments"],
            reg_preds["reposts"],
            reg_preds["views"],
        )
        return result

    def predict_single(self, row: dict) -> dict:
        """
        Inference cho 1 bài đăng đơn lẻ.

        Parameters
        ----------
        row : dict chứa ít nhất 'text'.

        Returns
        -------
        dict với pred_likes, pred_comments, pred_reposts, pred_views,
             pred_label, pred_label_name, predicted_popularity_score.
        """
        df = pd.DataFrame([row])
        result = self.predict(df)
        return result.iloc[0].to_dict()

    # ------------------------------------------------------------------
    # Convenience getters
    # ------------------------------------------------------------------

    @property
    def regression_history(self):
        return self.regression_head.history

    @property
    def classification_history(self):
        return self.classification_head.history

    @property
    def label_thresholds(self):
        return self.target_builder.label_thresholds
