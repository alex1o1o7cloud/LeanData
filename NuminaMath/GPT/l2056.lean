import Mathlib

namespace NUMINAMATH_GPT_expression_evaluation_l2056_205672

noncomputable def x : ℝ := (Real.sqrt 1.21) ^ 3
noncomputable def y : ℝ := (Real.sqrt 0.81) ^ 2
noncomputable def a : ℝ := 4 * Real.sqrt 0.81
noncomputable def b : ℝ := 2 * Real.sqrt 0.49
noncomputable def c : ℝ := 3 * Real.sqrt 1.21
noncomputable def d : ℝ := 2 * Real.sqrt 0.49
noncomputable def e : ℝ := (Real.sqrt 0.81) ^ 4

theorem expression_evaluation : ((x / Real.sqrt y) - (Real.sqrt a / b^2) + ((Real.sqrt c / Real.sqrt d) / (3 * e))) = 1.291343 := by 
  sorry

end NUMINAMATH_GPT_expression_evaluation_l2056_205672


namespace NUMINAMATH_GPT_solve_geometric_sequence_product_l2056_205622

noncomputable def geometric_sequence (a : ℕ → ℝ) : Prop :=
∀ n : ℕ, ∃ r : ℝ, a (n + 1) = a n * r

theorem solve_geometric_sequence_product (a : ℕ → ℝ) (h_geom : geometric_sequence a)
  (h_a35 : a 3 * a 5 = 4) : 
  a 1 * a 2 * a 3 * a 4 * a 5 * a 6 * a 7 = 128 :=
sorry

end NUMINAMATH_GPT_solve_geometric_sequence_product_l2056_205622


namespace NUMINAMATH_GPT_bryden_amount_correct_l2056_205635

-- Each state quarter has a face value of $0.25.
def face_value (q : ℕ) : ℝ := 0.25 * q

-- The collector offers to buy the state quarters for 1500% of their face value.
def collector_multiplier : ℝ := 15

-- Bryden has 10 state quarters.
def bryden_quarters : ℕ := 10

-- Calculate the amount Bryden will get for his 10 state quarters.
def amount_received : ℝ := collector_multiplier * face_value bryden_quarters

-- Prove that the amount received by Bryden equals $37.5.
theorem bryden_amount_correct : amount_received = 37.5 :=
by
  sorry

end NUMINAMATH_GPT_bryden_amount_correct_l2056_205635


namespace NUMINAMATH_GPT_greg_distance_work_to_market_l2056_205639

-- Given conditions translated into definitions
def total_distance : ℝ := 40
def time_from_market_to_home : ℝ := 0.5  -- in hours
def speed_from_market_to_home : ℝ := 20  -- in miles per hour

-- Distance calculation from farmer's market to home
def distance_from_market_to_home := speed_from_market_to_home * time_from_market_to_home

-- Definition for the distance from workplace to the farmer's market
def distance_from_work_to_market := total_distance - distance_from_market_to_home

-- The theorem to be proved
theorem greg_distance_work_to_market : distance_from_work_to_market = 30 := by
  -- Skipping the detailed proof
  sorry

end NUMINAMATH_GPT_greg_distance_work_to_market_l2056_205639


namespace NUMINAMATH_GPT_rectangle_area_l2056_205617

theorem rectangle_area (b : ℕ) (side radius length : ℕ) 
    (h1 : side * side = 1296)
    (h2 : radius = side)
    (h3 : length = radius / 6) :
    length * b = 6 * b :=
by
  sorry

end NUMINAMATH_GPT_rectangle_area_l2056_205617


namespace NUMINAMATH_GPT_binary_preceding_and_following_l2056_205674

theorem binary_preceding_and_following :
  ∀ (n : ℕ), n = 0b1010100 → (Nat.pred n = 0b1010011 ∧ Nat.succ n = 0b1010101) := by
  intros
  sorry

end NUMINAMATH_GPT_binary_preceding_and_following_l2056_205674


namespace NUMINAMATH_GPT_corrected_mean_of_observations_l2056_205691

theorem corrected_mean_of_observations (mean : ℝ) (n : ℕ) (incorrect_observation : ℝ) (correct_observation : ℝ) 
  (h_mean : mean = 41) (h_n : n = 50) (h_incorrect_observation : incorrect_observation = 23) (h_correct_observation : correct_observation = 48) 
  (h_sum_incorrect : mean * n = 2050) : 
  (mean * n - incorrect_observation + correct_observation) / n = 41.5 :=
by
  sorry

end NUMINAMATH_GPT_corrected_mean_of_observations_l2056_205691


namespace NUMINAMATH_GPT_total_worksheets_l2056_205645

theorem total_worksheets (x : ℕ) (h1 : 7 * (x - 8) = 63) : x = 17 := 
by {
  sorry
}

end NUMINAMATH_GPT_total_worksheets_l2056_205645


namespace NUMINAMATH_GPT_time_gaps_l2056_205690

theorem time_gaps (dist_a dist_b dist_c : ℕ) (time_a time_b time_c : ℕ) :
  dist_a = 130 →
  dist_b = 130 →
  dist_c = 130 →
  time_a = 36 →
  time_b = 45 →
  time_c = 42 →
  (time_b - time_a = 9) ∧ (time_c - time_a = 6) ∧ (time_b - time_c = 3) := by
  intros hdist_a hdist_b hdist_c htime_a htime_b htime_c
  sorry

end NUMINAMATH_GPT_time_gaps_l2056_205690


namespace NUMINAMATH_GPT_infinitely_many_triples_of_integers_l2056_205694

theorem infinitely_many_triples_of_integers (k : ℕ) :
  ∃ (x y z : ℕ), (x > 0 ∧ y > 0 ∧ z > 0) ∧
                  (x^999 + y^1000 = z^1001) :=
by
  sorry

end NUMINAMATH_GPT_infinitely_many_triples_of_integers_l2056_205694


namespace NUMINAMATH_GPT_probability_square_product_l2056_205624

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

def count_favorable_outcomes : ℕ :=
  List.length [(1, 1), (1, 4), (2, 2), (4, 1), (3, 3), (4, 4), (2, 8), (8, 2), (5, 5), (4, 9), (6, 6), (7, 7), (8, 8), (9, 9)]

def total_outcomes : ℕ := 12 * 8

theorem probability_square_product :
  (count_favorable_outcomes : ℚ) / (total_outcomes : ℚ) = (7 : ℚ) / (48 : ℚ) := 
by 
  sorry

end NUMINAMATH_GPT_probability_square_product_l2056_205624


namespace NUMINAMATH_GPT_union_of_sets_l2056_205687

def M : Set Int := { -1, 0, 1 }
def N : Set Int := { 0, 1, 2 }

theorem union_of_sets : M ∪ N = { -1, 0, 1, 2 } := by
  sorry

end NUMINAMATH_GPT_union_of_sets_l2056_205687


namespace NUMINAMATH_GPT_expand_expression_l2056_205662

variables {R : Type*} [CommRing R] (x : R)

theorem expand_expression : (15 * x^2 + 5) * 3 * x^3 = 45 * x^5 + 15 * x^3 :=
by sorry

end NUMINAMATH_GPT_expand_expression_l2056_205662


namespace NUMINAMATH_GPT_part1_part2_l2056_205647

-- Definitions for the sets A and B
def A : Set ℝ := { x | -2 ≤ x ∧ x ≤ 5 }
def B (m : ℝ) : Set ℝ := { x | m + 1 ≤ x ∧ x ≤ 2 * m - 1 }

-- Proof statement for the first part
theorem part1 (m : ℝ) (h : m = 4) : A ∪ B m = { x | -2 ≤ x ∧ x ≤ 7 } :=
sorry

-- Proof statement for the second part
theorem part2 (h : ∀ {m : ℝ}, B m ⊆ A) : ∀ m : ℝ, m ∈ Set.Iic 3 :=
sorry

end NUMINAMATH_GPT_part1_part2_l2056_205647


namespace NUMINAMATH_GPT_sqrt_72_eq_6_sqrt_2_l2056_205620

theorem sqrt_72_eq_6_sqrt_2 : Real.sqrt 72 = 6 * Real.sqrt 2 := 
by
  sorry

end NUMINAMATH_GPT_sqrt_72_eq_6_sqrt_2_l2056_205620


namespace NUMINAMATH_GPT_riding_mower_speed_l2056_205686

theorem riding_mower_speed :
  (∃ R : ℝ, 
     (8 * (3 / 4) = 6) ∧       -- Jerry mows 6 acres with the riding mower
     (8 * (1 / 4) = 2) ∧       -- Jerry mows 2 acres with the push mower
     (2 / 1 = 2) ∧             -- Push mower takes 2 hours to mow 2 acres
     (5 - 2 = 3) ∧             -- Time spent on the riding mower is 3 hours
     (6 / 3 = R) ∧             -- Riding mower cuts 6 acres in 3 hours
     R = 2) :=                 -- Therefore, R (speed of riding mower in acres per hour) is 2
sorry

end NUMINAMATH_GPT_riding_mower_speed_l2056_205686


namespace NUMINAMATH_GPT_cos_beta_calculation_l2056_205673

variable (α β : ℝ)
variable (h1 : 0 < α ∧ α < π / 2) -- α is an acute angle
variable (h2 : 0 < β ∧ β < π / 2) -- β is an acute angle
variable (h3 : Real.cos α = Real.sqrt 5 / 5)
variable (h4 : Real.sin (α - β) = Real.sqrt 10 / 10)

theorem cos_beta_calculation :
  Real.cos β = Real.sqrt 2 / 2 :=
  sorry

end NUMINAMATH_GPT_cos_beta_calculation_l2056_205673


namespace NUMINAMATH_GPT_square_area_l2056_205695

theorem square_area (s : ℕ) (h : s = 13) : s * s = 169 := by
  sorry

end NUMINAMATH_GPT_square_area_l2056_205695


namespace NUMINAMATH_GPT_algae_cells_count_10_days_l2056_205638

-- Define the initial condition where the pond starts with one algae cell.
def initial_algae_cells : ℕ := 1

-- Define the daily splitting of each cell into 3 new cells.
def daily_split (cells : ℕ) : ℕ := cells * 3

-- Define the function to compute the number of algae cells after n days.
def algae_cells_after_days (n : ℕ) : ℕ :=
  initial_algae_cells * (3 ^ n)

-- State the theorem to be proved.
theorem algae_cells_count_10_days : algae_cells_after_days 10 = 59049 :=
by {
  sorry
}

end NUMINAMATH_GPT_algae_cells_count_10_days_l2056_205638


namespace NUMINAMATH_GPT_asymptotes_of_hyperbola_l2056_205643

-- Definitions
variables (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b)

-- Theorem: Equation of the asymptotes of the given hyperbola
theorem asymptotes_of_hyperbola (h_equiv : b = 2 * a) :
  ∀ x y : ℝ, 
    (x ≠ 0 ∧ y ≠ 0 ∧ (y = (2 : ℝ) * x ∨ y = - (2 : ℝ) * x)) ↔ (x, y) ∈ {p : ℝ × ℝ | (x^2 / a^2) - (y^2 / b^2) = 1} := 
sorry

end NUMINAMATH_GPT_asymptotes_of_hyperbola_l2056_205643


namespace NUMINAMATH_GPT_parabola_focus_distance_l2056_205683

theorem parabola_focus_distance
  (p : ℝ) (h : p > 0)
  (x1 x2 y1 y2 : ℝ)
  (h1 : x1 = 3 - p / 2) 
  (h2 : x2 = 2 - p / 2)
  (h3 : y1^2 = 2 * p * x1)
  (h4 : y2^2 = 2 * p * x2)
  (h5 : y1^2 / y2^2 = x1 / x2) : 
  p = 12 / 5 := 
sorry

end NUMINAMATH_GPT_parabola_focus_distance_l2056_205683


namespace NUMINAMATH_GPT_third_student_gold_stickers_l2056_205600

theorem third_student_gold_stickers:
  ∃ (n : ℕ), n = 41 ∧ 
  (∃ (a1 a2 a4 a5 a6 : ℕ), 
    a1 = 29 ∧ 
    a2 = 35 ∧ 
    a4 = 47 ∧ 
    a5 = 53 ∧ 
    a6 = 59 ∧ 
    a2 - a1 = 6 ∧ 
    a5 - a4 = 6 ∧ 
    ∀ k, k = 3 → n = a2 + 6) := 
sorry

end NUMINAMATH_GPT_third_student_gold_stickers_l2056_205600


namespace NUMINAMATH_GPT_expected_male_teachers_in_sample_l2056_205663

theorem expected_male_teachers_in_sample 
  (total_male total_female sample_size : ℕ) 
  (h1 : total_male = 56) 
  (h2 : total_female = 42) 
  (h3 : sample_size = 14) :
  (total_male * sample_size) / (total_male + total_female) = 8 :=
by
  sorry

end NUMINAMATH_GPT_expected_male_teachers_in_sample_l2056_205663


namespace NUMINAMATH_GPT_slope_of_line_l2056_205623

-- Definitions of the conditions in the problem
def line_eq (a : ℝ) (x y : ℝ) : Prop := x + a * y + 1 = 0

def y_intercept (l : ℝ → ℝ → Prop) (b : ℝ) : Prop :=
  l 0 b

-- The statement of the proof problem
theorem slope_of_line (a : ℝ) (h : y_intercept (line_eq a) (-2)) : 
  ∃ (m : ℝ), m = -2 :=
sorry

end NUMINAMATH_GPT_slope_of_line_l2056_205623


namespace NUMINAMATH_GPT_tan_neg_two_sin_cos_sum_l2056_205618

theorem tan_neg_two_sin_cos_sum (θ : ℝ) (h : Real.tan θ = -2) : 
  Real.sin (2 * θ) + Real.cos (2 * θ) = -7 / 5 :=
by
  sorry

end NUMINAMATH_GPT_tan_neg_two_sin_cos_sum_l2056_205618


namespace NUMINAMATH_GPT_part_a_part_b_l2056_205649

variable {α β γ δ AB CD : ℝ}
variable {A B C D : Point}
variable {A_obtuse B_obtuse : Prop}
variable {α_gt_δ β_gt_γ : Prop}

-- Definition of a convex quadrilateral
def convex_quadrilateral (A B C D : Point) : Prop := sorry

-- Conditions for part (a)
axiom angle_A_obtuse : A_obtuse
axiom angle_B_obtuse : B_obtuse

-- Conditions for part (b)
axiom angle_α_gt_δ : α_gt_δ
axiom angle_β_gt_γ : β_gt_γ

-- Part (a) statement: Given angles A and B are obtuse, AB ≤ CD
theorem part_a {A B C D : Point} (h_convex : convex_quadrilateral A B C D) 
    (h_A_obtuse : A_obtuse) (h_B_obtuse : B_obtuse) : AB ≤ CD :=
sorry

-- Part (b) statement: Given angle A > angle D and angle B > angle C, AB < CD
theorem part_b {A B C D : Point} (h_convex : convex_quadrilateral A B C D) 
    (h_angle_α_gt_δ : α_gt_δ) (h_angle_β_gt_γ : β_gt_γ) : AB < CD :=
sorry

end NUMINAMATH_GPT_part_a_part_b_l2056_205649


namespace NUMINAMATH_GPT_find_xy_pairs_l2056_205667

theorem find_xy_pairs (x y: ℝ) :
  x + y + 4 = (12 * x + 11 * y) / (x ^ 2 + y ^ 2) ∧
  y - x + 3 = (11 * x - 12 * y) / (x ^ 2 + y ^ 2) ↔
  (x = 2 ∧ y = 1) ∨ (x = -2.5 ∧ y = -4.5) :=
by
  sorry

end NUMINAMATH_GPT_find_xy_pairs_l2056_205667


namespace NUMINAMATH_GPT_harris_carrot_cost_l2056_205678

-- Definitions stemming from the conditions
def carrots_per_day : ℕ := 1
def days_per_year : ℕ := 365
def carrots_per_bag : ℕ := 5
def cost_per_bag : ℕ := 2

-- Prove that Harris's total cost for carrots in one year is $146
theorem harris_carrot_cost : (days_per_year * carrots_per_day / carrots_per_bag) * cost_per_bag = 146 := by
  sorry

end NUMINAMATH_GPT_harris_carrot_cost_l2056_205678


namespace NUMINAMATH_GPT_max_min_x2_sub_xy_add_y2_l2056_205653

/-- Given a point \((x, y)\) on the curve defined by \( |5x + y| + |5x - y| = 20 \), prove that the maximum value of \(x^2 - xy + y^2\) is 124 and the minimum value is 3. -/
theorem max_min_x2_sub_xy_add_y2 (x y : ℝ) (h : abs (5 * x + y) + abs (5 * x - y) = 20) :
  3 ≤ x^2 - x * y + y^2 ∧ x^2 - x * y + y^2 ≤ 124 := 
sorry

end NUMINAMATH_GPT_max_min_x2_sub_xy_add_y2_l2056_205653


namespace NUMINAMATH_GPT_survey_representative_l2056_205661

universe u

inductive SurveyOption : Type u
| A : SurveyOption  -- Selecting a class of students
| B : SurveyOption  -- Selecting 50 male students
| C : SurveyOption  -- Selecting 50 female students
| D : SurveyOption  -- Randomly selecting 50 eighth-grade students

def most_appropriate_survey : SurveyOption := SurveyOption.D

theorem survey_representative : most_appropriate_survey = SurveyOption.D := 
by sorry

end NUMINAMATH_GPT_survey_representative_l2056_205661


namespace NUMINAMATH_GPT_yoongi_stacked_higher_by_one_cm_l2056_205612

def height_box_A : ℝ := 3
def height_box_B : ℝ := 3.5
def boxes_stacked_by_Taehyung : ℕ := 16
def boxes_stacked_by_Yoongi : ℕ := 14
def height_Taehyung_stack : ℝ := height_box_A * boxes_stacked_by_Taehyung
def height_Yoongi_stack : ℝ := height_box_B * boxes_stacked_by_Yoongi

theorem yoongi_stacked_higher_by_one_cm :
  height_Yoongi_stack = height_Taehyung_stack + 1 :=
by
  sorry

end NUMINAMATH_GPT_yoongi_stacked_higher_by_one_cm_l2056_205612


namespace NUMINAMATH_GPT_most_noteworthy_figure_is_mode_l2056_205692

-- Define the types of possible statistics
inductive Statistic
| Median
| Mean
| Mode
| WeightedMean

-- Define a structure for survey data (details abstracted)
structure SurveyData where
  -- fields abstracted for this problem

-- Define the concept of the most noteworthy figure
def most_noteworthy_figure (data : SurveyData) : Statistic :=
  Statistic.Mode

-- Theorem to prove the most noteworthy figure in a survey's data is the mode
theorem most_noteworthy_figure_is_mode (data : SurveyData) :
  most_noteworthy_figure data = Statistic.Mode :=
by
  sorry

end NUMINAMATH_GPT_most_noteworthy_figure_is_mode_l2056_205692


namespace NUMINAMATH_GPT_log_comparison_l2056_205688

open Real

noncomputable def a := log 4 / log 5  -- a = log_5(4)
noncomputable def b := log 5 / log 3  -- b = log_3(5)
noncomputable def c := log 5 / log 4  -- c = log_4(5)

theorem log_comparison : a < c ∧ c < b := 
by
  sorry

end NUMINAMATH_GPT_log_comparison_l2056_205688


namespace NUMINAMATH_GPT_remainder_when_sum_divided_by_11_l2056_205675

def sum_of_large_numbers : ℕ :=
  100001 + 100002 + 100003 + 100004 + 100005 + 100006 + 100007

theorem remainder_when_sum_divided_by_11 : sum_of_large_numbers % 11 = 2 := by
  sorry

end NUMINAMATH_GPT_remainder_when_sum_divided_by_11_l2056_205675


namespace NUMINAMATH_GPT_length_segment_FF_l2056_205693

-- Define the points F and F' based on the given conditions
def F : (ℝ × ℝ) := (4, 3)
def F' : (ℝ × ℝ) := (-4, 3)

-- The theorem to prove the length of the segment FF' is 8
theorem length_segment_FF' : dist F F' = 8 :=
by
  sorry

end NUMINAMATH_GPT_length_segment_FF_l2056_205693


namespace NUMINAMATH_GPT_notebook_pen_cost_l2056_205664

theorem notebook_pen_cost :
  ∃ (n p : ℕ), 15 * n + 4 * p = 160 ∧ n > p ∧ n + p = 18 := 
sorry

end NUMINAMATH_GPT_notebook_pen_cost_l2056_205664


namespace NUMINAMATH_GPT_serum_prevents_colds_l2056_205646

noncomputable def hypothesis_preventive_effect (H : Prop) : Prop :=
  let K2 := 3.918
  let critical_value := 3.841
  let P_threshold := 0.05
  K2 >= critical_value ∧ P_threshold = 0.05 → H

theorem serum_prevents_colds (H : Prop) : hypothesis_preventive_effect H → H :=
by
  -- Proof will be added here
  sorry

end NUMINAMATH_GPT_serum_prevents_colds_l2056_205646


namespace NUMINAMATH_GPT_george_score_l2056_205640

theorem george_score (avg_without_george avg_with_george : ℕ) (num_students : ℕ) 
(h1 : avg_without_george = 75) (h2 : avg_with_george = 76) (h3 : num_students = 20) :
  (num_students * avg_with_george) - ((num_students - 1) * avg_without_george) = 95 :=
by 
  sorry

end NUMINAMATH_GPT_george_score_l2056_205640


namespace NUMINAMATH_GPT_solve_for_y_l2056_205621

theorem solve_for_y (y : ℕ) : (1000^4 = 10^y) → y = 12 :=
by {
  sorry
}

end NUMINAMATH_GPT_solve_for_y_l2056_205621


namespace NUMINAMATH_GPT_evaluate_expression_l2056_205656

theorem evaluate_expression :
  (Real.sqrt 5 * 5^(1/2) + 20 / 4 * 3 - 8^(3/2) + 5) = 25 - 16 * Real.sqrt 2 := 
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l2056_205656


namespace NUMINAMATH_GPT_possible_values_for_a_l2056_205648

noncomputable def f (a x : ℝ) : ℝ := Real.exp (a * x) - x - 1

theorem possible_values_for_a (a : ℝ) (h: a ≠ 0) : 
  (∀ x : ℝ, f a x ≥ 0) ↔ a = 1 :=
by
  sorry

end NUMINAMATH_GPT_possible_values_for_a_l2056_205648


namespace NUMINAMATH_GPT_ray_walks_to_park_l2056_205670

theorem ray_walks_to_park (x : ℤ) (h1 : 3 * (x + 7 + 11) = 66) : x = 4 :=
by
  -- solving steps are skipped
  sorry

end NUMINAMATH_GPT_ray_walks_to_park_l2056_205670


namespace NUMINAMATH_GPT_pq_square_sum_l2056_205630

theorem pq_square_sum (p q : ℝ) (h1 : p * q = 9) (h2 : p + q = 6) : p^2 + q^2 = 18 := 
by
  sorry

end NUMINAMATH_GPT_pq_square_sum_l2056_205630


namespace NUMINAMATH_GPT_election_result_l2056_205607

def votes_A : ℕ := 12
def votes_B : ℕ := 3
def votes_C : ℕ := 15

def is_class_president (candidate_votes : ℕ) : Prop :=
  candidate_votes = max (max votes_A votes_B) votes_C

theorem election_result : is_class_president votes_C :=
by
  unfold is_class_president
  rw [votes_A, votes_B, votes_C]
  sorry

end NUMINAMATH_GPT_election_result_l2056_205607


namespace NUMINAMATH_GPT_cost_of_hard_lenses_l2056_205657

theorem cost_of_hard_lenses (x H : ℕ) (h1 : x + (x + 5) = 11)
    (h2 : 150 * (x + 5) + H * x = 1455) : H = 85 := by
  sorry

end NUMINAMATH_GPT_cost_of_hard_lenses_l2056_205657


namespace NUMINAMATH_GPT_bus_time_l2056_205697

variable (t1 t2 t3 t4 : ℕ)

theorem bus_time
  (h1 : t1 = 25)
  (h2 : t2 = 40)
  (h3 : t3 = 15)
  (h4 : t4 = 10) :
  t1 + t2 + t3 + t4 = 90 := by
  sorry

end NUMINAMATH_GPT_bus_time_l2056_205697


namespace NUMINAMATH_GPT_bing_location_subject_l2056_205659

-- Defining entities
inductive City
| Beijing
| Shanghai
| Chongqing

inductive Subject
| Mathematics
| Chinese
| ForeignLanguage

inductive Teacher
| Jia
| Yi
| Bing

-- Defining the conditions
variables (works_in : Teacher → City) (teaches : Teacher → Subject)

axiom cond1_jia_not_beijing : works_in Teacher.Jia ≠ City.Beijing
axiom cond1_yi_not_shanghai : works_in Teacher.Yi ≠ City.Shanghai
axiom cond2_beijing_not_foreign : ∀ t, works_in t = City.Beijing → teaches t ≠ Subject.ForeignLanguage
axiom cond3_shanghai_math : ∀ t, works_in t = City.Shanghai → teaches t = Subject.Mathematics
axiom cond4_yi_not_chinese : teaches Teacher.Yi ≠ Subject.Chinese

-- The question
theorem bing_location_subject : 
  works_in Teacher.Bing = City.Beijing ∧ teaches Teacher.Bing = Subject.Chinese :=
by
  sorry

end NUMINAMATH_GPT_bing_location_subject_l2056_205659


namespace NUMINAMATH_GPT_solve_inequality_l2056_205682

theorem solve_inequality (x : ℝ) : 
  (2 / (x + 2) + 4 / (x + 6) ≥ 1) ↔ (x ∈ Set.Icc (-4) (-2) ∨ x ∈ Set.Icc 2 4) :=
sorry

end NUMINAMATH_GPT_solve_inequality_l2056_205682


namespace NUMINAMATH_GPT_number_of_ways_to_choose_4_from_28_number_of_ways_to_choose_3_from_27_with_kolya_included_l2056_205689

-- Part (a)
theorem number_of_ways_to_choose_4_from_28 :
  (Nat.choose 28 4) = 20475 :=
sorry

-- Part (b)
theorem number_of_ways_to_choose_3_from_27_with_kolya_included :
  (Nat.choose 27 3) = 2925 :=
sorry

end NUMINAMATH_GPT_number_of_ways_to_choose_4_from_28_number_of_ways_to_choose_3_from_27_with_kolya_included_l2056_205689


namespace NUMINAMATH_GPT_arithmetic_sequence_ninth_term_l2056_205636

theorem arithmetic_sequence_ninth_term (a d : ℤ) (h1 : a + 2 * d = 20) (h2 : a + 5 * d = 26) : a + 8 * d = 32 :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_ninth_term_l2056_205636


namespace NUMINAMATH_GPT_inverse_proportion_relation_l2056_205654

variable (k : ℝ) (y1 y2 : ℝ) (h1 : y1 = - (2 / (-1))) (h2 : y2 = - (2 / (-2)))

theorem inverse_proportion_relation : y1 > y2 := by
  sorry

end NUMINAMATH_GPT_inverse_proportion_relation_l2056_205654


namespace NUMINAMATH_GPT_bus_passengers_final_count_l2056_205658

theorem bus_passengers_final_count :
  let initial_passengers := 15
  let changes := [(3, -6), (-2, 4), (-7, 2), (3, -5)]
  let apply_change (acc : Int) (change : Int × Int) : Int :=
    acc + change.1 + change.2
  initial_passengers + changes.foldl apply_change 0 = 7 :=
by
  intros
  sorry

end NUMINAMATH_GPT_bus_passengers_final_count_l2056_205658


namespace NUMINAMATH_GPT_height_of_carton_is_70_l2056_205633

def carton_dimensions : ℕ × ℕ := (25, 42)
def soap_box_dimensions : ℕ × ℕ × ℕ := (7, 6, 5)
def max_soap_boxes : ℕ := 300

theorem height_of_carton_is_70 :
  let (carton_length, carton_width) := carton_dimensions
  let (soap_box_length, soap_box_width, soap_box_height) := soap_box_dimensions
  let boxes_per_layer := (carton_length / soap_box_length) * (carton_width / soap_box_width)
  let num_layers := max_soap_boxes / boxes_per_layer
  (num_layers * soap_box_height) = 70 :=
by
  have carton_length := 25
  have carton_width := 42
  have soap_box_length := 7
  have soap_box_width := 6
  have soap_box_height := 5
  have max_soap_boxes := 300
  have boxes_per_layer := (25 / 7) * (42 / 6)
  have num_layers := max_soap_boxes / boxes_per_layer
  sorry

end NUMINAMATH_GPT_height_of_carton_is_70_l2056_205633


namespace NUMINAMATH_GPT_months_rent_in_advance_required_l2056_205641

def janet_savings : ℕ := 2225
def rent_per_month : ℕ := 1250
def deposit : ℕ := 500
def additional_needed : ℕ := 775

theorem months_rent_in_advance_required : 
  (janet_savings + additional_needed - deposit) / rent_per_month = 2 :=
by
  sorry

end NUMINAMATH_GPT_months_rent_in_advance_required_l2056_205641


namespace NUMINAMATH_GPT_winnieKeepsBalloons_l2056_205634

-- Given conditions
def redBalloons : Nat := 24
def whiteBalloons : Nat := 39
def greenBalloons : Nat := 72
def chartreuseBalloons : Nat := 91
def totalFriends : Nat := 11

-- Total balloons
def totalBalloons : Nat := redBalloons + whiteBalloons + greenBalloons + chartreuseBalloons

-- Theorem: Prove the number of balloons Winnie keeps for herself
theorem winnieKeepsBalloons :
  totalBalloons % totalFriends = 6 :=
by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_winnieKeepsBalloons_l2056_205634


namespace NUMINAMATH_GPT_total_promotional_items_l2056_205660

def num_calendars : ℕ := 300
def num_date_books : ℕ := 200

theorem total_promotional_items : num_calendars + num_date_books = 500 := by
  sorry

end NUMINAMATH_GPT_total_promotional_items_l2056_205660


namespace NUMINAMATH_GPT_A_not_losing_prob_correct_l2056_205666

def probability_draw : ℚ := 1 / 2
def probability_A_wins : ℚ := 1 / 3
def probability_A_not_losing : ℚ := 5 / 6

theorem A_not_losing_prob_correct : 
  probability_draw + probability_A_wins = probability_A_not_losing := 
by sorry

end NUMINAMATH_GPT_A_not_losing_prob_correct_l2056_205666


namespace NUMINAMATH_GPT_gcd_60_90_150_l2056_205685

theorem gcd_60_90_150 : Nat.gcd (Nat.gcd 60 90) 150 = 30 := 
by
  sorry

end NUMINAMATH_GPT_gcd_60_90_150_l2056_205685


namespace NUMINAMATH_GPT_sample_size_is_200_l2056_205602
-- Define the total number of students and the number of students surveyed
def total_students : ℕ := 3600
def students_surveyed : ℕ := 200

-- Define the sample size
def sample_size := students_surveyed

-- Prove the sample size is 200
theorem sample_size_is_200 : sample_size = 200 :=
by
  -- Placeholder for the actual proof
  sorry

end NUMINAMATH_GPT_sample_size_is_200_l2056_205602


namespace NUMINAMATH_GPT_faye_pencils_l2056_205611

theorem faye_pencils (rows crayons : ℕ) (pencils_per_row : ℕ) (h1 : rows = 7) (h2 : pencils_per_row = 5) : 
  (rows * pencils_per_row) = 35 :=
by {
  sorry
}

end NUMINAMATH_GPT_faye_pencils_l2056_205611


namespace NUMINAMATH_GPT_volume_s_l2056_205631

def condition1 (x y : ℝ) : Prop := |9 - x| + y ≤ 12
def condition2 (x y : ℝ) : Prop := 3 * y - x ≥ 18
def S (x y : ℝ) : Prop := condition1 x y ∧ condition2 x y

def is_volume_correct (m n : ℕ) (p : ℕ) :=
  (m + n + p = 153) ∧ (m = 135) ∧ (n = 8) ∧ (p = 10)

theorem volume_s (m n p : ℕ) :
  (∀ x y : ℝ, S x y) → is_volume_correct m n p :=
by 
  sorry

end NUMINAMATH_GPT_volume_s_l2056_205631


namespace NUMINAMATH_GPT_functional_equation_solution_l2056_205677

/-- For all functions f: ℝ → ℝ, that satisfy the given functional equation -/
def functional_equation (f: ℝ → ℝ) : Prop :=
  ∀ x y: ℝ, f (x + y * f (x + y)) = y ^ 2 + f (x * f (y + 1))

/-- The solution to the functional equation is f(x) = x -/
theorem functional_equation_solution :
  ∀ f: ℝ → ℝ, functional_equation f → (∀ x: ℝ, f x = x) :=
by
  intros f h x
  sorry

end NUMINAMATH_GPT_functional_equation_solution_l2056_205677


namespace NUMINAMATH_GPT_probability_all_black_after_rotation_l2056_205698

-- Define the conditions
def num_unit_squares : ℕ := 16
def num_colors : ℕ := 3
def prob_per_color : ℚ := 1 / 3

-- Define the type for probabilities
def prob_black_grid : ℚ := (1 / 81) * (11 / 27) ^ 12

-- The statement to be proven
theorem probability_all_black_after_rotation :
  (prob_black_grid =
    ((1 / 3) ^ 4) * ((11 / 27) ^ 12)) :=
sorry

end NUMINAMATH_GPT_probability_all_black_after_rotation_l2056_205698


namespace NUMINAMATH_GPT_total_distance_swam_l2056_205679

theorem total_distance_swam (molly_swam_saturday : ℕ) (molly_swam_sunday : ℕ) (h1 : molly_swam_saturday = 400) (h2 : molly_swam_sunday = 300) : molly_swam_saturday + molly_swam_sunday = 700 := by 
    sorry

end NUMINAMATH_GPT_total_distance_swam_l2056_205679


namespace NUMINAMATH_GPT_probability_white_ball_l2056_205615

def num_white_balls : ℕ := 5
def num_black_balls : ℕ := 6
def total_balls : ℕ := num_white_balls + num_black_balls

theorem probability_white_ball : (num_white_balls : ℚ) / total_balls = 5 / 11 := by
  sorry

end NUMINAMATH_GPT_probability_white_ball_l2056_205615


namespace NUMINAMATH_GPT_real_solutions_l2056_205652

open Real

theorem real_solutions (x : ℝ) : (x - 2) ^ 4 + (2 - x) ^ 4 = 50 ↔ 
  x = 2 + sqrt (-12 + 3 * sqrt 17) ∨ x = 2 - sqrt (-12 + 3 * sqrt 17) :=
by
  sorry

end NUMINAMATH_GPT_real_solutions_l2056_205652


namespace NUMINAMATH_GPT_repair_cost_l2056_205676

theorem repair_cost (purchase_price transport_cost sale_price : ℝ) (profit_percentage : ℝ) (repair_cost : ℝ) :
  purchase_price = 14000 →
  transport_cost = 1000 →
  sale_price = 30000 →
  profit_percentage = 50 →
  sale_price = (1 + profit_percentage / 100) * (purchase_price + repair_cost + transport_cost) →
  repair_cost = 5000 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_repair_cost_l2056_205676


namespace NUMINAMATH_GPT_mechanism_completion_times_l2056_205606

theorem mechanism_completion_times :
  ∃ (x y : ℝ), (1 / x + 1 / y = 1 / 30) ∧ (6 * (1 / x + 1 / y) + 40 * (1 / y) = 1) ∧ x = 75 ∧ y = 50 :=
by {
  sorry
}

end NUMINAMATH_GPT_mechanism_completion_times_l2056_205606


namespace NUMINAMATH_GPT_range_of_t_l2056_205616

def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def decreasing_function (f : ℝ → ℝ) : Prop :=
  ∀ ⦃x y⦄, x < y → f y < f x

variable {f : ℝ → ℝ}

theorem range_of_t (h_odd : odd_function f) 
  (h_decreasing : decreasing_function f)
  (h_ineq : ∀ t, -1 < t → t < 1 → f (1 - t) + f (1 - t^2) < 0) 
  : ∀ t, 0 < t → t < 1 :=
by sorry

end NUMINAMATH_GPT_range_of_t_l2056_205616


namespace NUMINAMATH_GPT_units_digit_6_pow_4_l2056_205605

-- Define the units digit function
def units_digit (n : ℕ) : ℕ := n % 10

-- Define the main theorem to prove
theorem units_digit_6_pow_4 : units_digit (6 ^ 4) = 6 := 
by
  sorry

end NUMINAMATH_GPT_units_digit_6_pow_4_l2056_205605


namespace NUMINAMATH_GPT_expression_incorrect_l2056_205642

theorem expression_incorrect (x : ℝ) : 5 * (x + 7) ≠ 5 * x + 7 := 
by 
  sorry

end NUMINAMATH_GPT_expression_incorrect_l2056_205642


namespace NUMINAMATH_GPT_import_tax_amount_in_excess_l2056_205603

theorem import_tax_amount_in_excess (X : ℝ) 
  (h1 : 0.07 * (2590 - X) = 111.30) : 
  X = 1000 :=
by
  sorry

end NUMINAMATH_GPT_import_tax_amount_in_excess_l2056_205603


namespace NUMINAMATH_GPT_factor_expression_l2056_205644

theorem factor_expression (
  x y z : ℝ
) : 
  ( (x^2 - y^2)^3 + (y^2 - z^2)^3 + (z^2 - x^2)^3) / 
  ( (x - y)^3 + (y - z)^3 + (z - x)^3) = 
  (x + y) * (y + z) * (z + x) := 
sorry

end NUMINAMATH_GPT_factor_expression_l2056_205644


namespace NUMINAMATH_GPT_compute_expression_l2056_205669

-- Given condition
def condition (x : ℝ) : Prop := x + 1/x = 3

-- Theorem to prove
theorem compute_expression (x : ℝ) (hx : condition x) : (x - 1) ^ 2 + 16 / (x - 1) ^ 2 = 8 := 
by
  sorry

end NUMINAMATH_GPT_compute_expression_l2056_205669


namespace NUMINAMATH_GPT_effective_annual_interest_rate_is_correct_l2056_205632

noncomputable def quarterly_interest_rate : ℝ := 0.02

noncomputable def annual_interest_rate (quarterly_rate : ℝ) : ℝ :=
  ((1 + quarterly_rate) ^ 4 - 1) * 100

theorem effective_annual_interest_rate_is_correct :
  annual_interest_rate quarterly_interest_rate = 8.24 :=
by
  sorry

end NUMINAMATH_GPT_effective_annual_interest_rate_is_correct_l2056_205632


namespace NUMINAMATH_GPT_marketing_firm_l2056_205680

variable (Total_households : ℕ) (A_only : ℕ) (A_and_B : ℕ) (B_to_A_and_B_ratio : ℕ)

def neither_soap_households : ℕ :=
  Total_households - (A_only + (B_to_A_and_B_ratio * A_and_B) + A_and_B)

theorem marketing_firm (h1 : Total_households = 300)
                       (h2 : A_only = 60)
                       (h3 : A_and_B = 40)
                       (h4 : B_to_A_and_B_ratio = 3)
                       : neither_soap_households 300 60 40 3 = 80 :=
by {
  sorry
}

end NUMINAMATH_GPT_marketing_firm_l2056_205680


namespace NUMINAMATH_GPT_initial_bales_l2056_205609

theorem initial_bales (B : ℕ) (cond1 : B + 35 = 82) : B = 47 :=
by
  sorry

end NUMINAMATH_GPT_initial_bales_l2056_205609


namespace NUMINAMATH_GPT_lines_intersect_lines_parallel_lines_coincident_l2056_205625

-- Define line equations
def l1 (m x y : ℝ) := (m + 2) * x + (m + 3) * y - 5 = 0
def l2 (m x y : ℝ) := 6 * x + (2 * m - 1) * y - 5 = 0

-- Prove conditions for intersection
theorem lines_intersect (m : ℝ) : ¬(m = -5 / 2 ∨ m = 4) ↔
  ∃ x y : ℝ, l1 m x y ∧ l2 m x y := sorry

-- Prove conditions for parallel lines
theorem lines_parallel (m : ℝ) : m = -5 / 2 ↔
  ∀ x y : ℝ, l1 m x y ∧ l2 m x y → l1 m x y → l2 m x y := sorry

-- Prove conditions for coincident lines
theorem lines_coincident (m : ℝ) : m = 4 ↔
  ∀ x y : ℝ, l1 m x y ↔ l2 m x y := sorry

end NUMINAMATH_GPT_lines_intersect_lines_parallel_lines_coincident_l2056_205625


namespace NUMINAMATH_GPT_total_distance_A_C_B_l2056_205668

noncomputable section

open Real

def point := (ℝ × ℝ)

def distance (p1 p2 : point) : ℝ :=
  sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

def A : point := (-3, 5)
def B : point := (5, -3)
def C : point := (0, 0)

theorem total_distance_A_C_B :
  distance A C + distance C B = 2 * sqrt 34 :=
by
  sorry

end NUMINAMATH_GPT_total_distance_A_C_B_l2056_205668


namespace NUMINAMATH_GPT_antonio_correct_answers_l2056_205650

theorem antonio_correct_answers :
  ∃ c w : ℕ, c + w = 15 ∧ 6 * c - 3 * w = 36 ∧ c = 9 :=
by
  sorry

end NUMINAMATH_GPT_antonio_correct_answers_l2056_205650


namespace NUMINAMATH_GPT_tangerines_count_l2056_205608

theorem tangerines_count (apples pears tangerines : ℕ)
  (h1 : apples = 45)
  (h2 : pears = apples - 21)
  (h3 : tangerines = pears + 18) :
  tangerines = 42 :=
by
  sorry

end NUMINAMATH_GPT_tangerines_count_l2056_205608


namespace NUMINAMATH_GPT_suzy_final_books_l2056_205684

def initial_books : ℕ := 98
def wednesday_checkouts : ℕ := 43
def thursday_returns : ℕ := 23
def thursday_checkouts : ℕ := 5
def friday_returns : ℕ := 7

theorem suzy_final_books :
  initial_books - wednesday_checkouts
  + thursday_returns - thursday_checkouts
  + friday_returns = 80 := by
sorry

end NUMINAMATH_GPT_suzy_final_books_l2056_205684


namespace NUMINAMATH_GPT_bianca_birthday_money_l2056_205671

/-- Define the number of friends Bianca has -/
def number_of_friends : ℕ := 5

/-- Define the amount of dollars each friend gave -/
def dollars_per_friend : ℕ := 6

/-- The total amount of dollars Bianca received -/
def total_dollars_received : ℕ := number_of_friends * dollars_per_friend

/-- Prove that the total amount of dollars Bianca received is 30 -/
theorem bianca_birthday_money : total_dollars_received = 30 :=
by
  sorry

end NUMINAMATH_GPT_bianca_birthday_money_l2056_205671


namespace NUMINAMATH_GPT_circle_eq_problem1_circle_eq_problem2_l2056_205699

-- Problem 1
theorem circle_eq_problem1 :
  (∃ a b r : ℝ, (x - a)^2 + (y - b)^2 = r^2 ∧
  a - 2 * b - 3 = 0 ∧
  (2 - a)^2 + (-3 - b)^2 = r^2 ∧
  (-2 - a)^2 + (-5 - b)^2 = r^2) ↔
  (x + 1)^2 + (y + 2)^2 = 10 :=
sorry

-- Problem 2
theorem circle_eq_problem2 :
  (∃ D E F : ℝ, x^2 + y^2 + D * x + E * y + F = 0 ∧
  (1:ℝ)^2 + (0:ℝ)^2 + D * 1 + E * 0 + F = 0 ∧
  (-1:ℝ)^2 + (-2:ℝ)^2 - D * 1 - 2 * E + F = 0 ∧
  (3:ℝ)^2 + (-2:ℝ)^2 + 3 * D - 2 * E + F = 0) ↔
  x^2 + y^2 - 2 * x + 4 * y + 1 = 0 :=
sorry

end NUMINAMATH_GPT_circle_eq_problem1_circle_eq_problem2_l2056_205699


namespace NUMINAMATH_GPT_book_length_l2056_205655

theorem book_length (P : ℕ) (h1 : 2323 = (P - 2323) + 90) : P = 4556 :=
by
  sorry

end NUMINAMATH_GPT_book_length_l2056_205655


namespace NUMINAMATH_GPT_product_of_two_numbers_l2056_205651

theorem product_of_two_numbers (a b : ℕ) (h_lcm : lcm a b = 48) (h_gcd : gcd a b = 8) : a * b = 384 :=
by sorry

end NUMINAMATH_GPT_product_of_two_numbers_l2056_205651


namespace NUMINAMATH_GPT_smallest_positive_integer_linear_combination_l2056_205619

theorem smallest_positive_integer_linear_combination : ∃ m n : ℤ, 3003 * m + 55555 * n = 1 :=
by
  sorry

end NUMINAMATH_GPT_smallest_positive_integer_linear_combination_l2056_205619


namespace NUMINAMATH_GPT_dave_used_tickets_for_toys_l2056_205665

-- Define the given conditions
def number_of_tickets_won : ℕ := 18
def tickets_more_for_clothes : ℕ := 10

-- Define the main conjecture
theorem dave_used_tickets_for_toys (T : ℕ) : T + (T + tickets_more_for_clothes) = number_of_tickets_won → T = 4 :=
by {
  -- We'll need the proof here, but it's not required for the statement purpose.
  sorry
}

end NUMINAMATH_GPT_dave_used_tickets_for_toys_l2056_205665


namespace NUMINAMATH_GPT_sum_of_cosines_l2056_205696

theorem sum_of_cosines :
  (Real.cos (2 * Real.pi / 7) + Real.cos (4 * Real.pi / 7) + Real.cos (6 * Real.pi / 7) = -1 / 2) := sorry

end NUMINAMATH_GPT_sum_of_cosines_l2056_205696


namespace NUMINAMATH_GPT_algebra_expression_value_l2056_205681

theorem algebra_expression_value (a : ℝ) (h : 3 * a ^ 2 + 2 * a - 1 = 0) : 3 * a ^ 2 + 2 * a - 2019 = -2018 := 
by 
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_algebra_expression_value_l2056_205681


namespace NUMINAMATH_GPT_min_filtration_cycles_l2056_205601

theorem min_filtration_cycles {c₀ : ℝ} (initial_concentration : c₀ = 225)
  (max_concentration : ℝ := 7.5) (reduction_factor : ℝ := 1 / 3)
  (log2 : ℝ := 0.3010) (log3 : ℝ := 0.4771) :
  ∃ n : ℕ, (c₀ * (reduction_factor ^ n) ≤ max_concentration ∧ n ≥ 9) :=
sorry

end NUMINAMATH_GPT_min_filtration_cycles_l2056_205601


namespace NUMINAMATH_GPT_ratio_karen_beatrice_l2056_205613

noncomputable def karen_crayons : ℕ := 128
noncomputable def judah_crayons : ℕ := 8
noncomputable def gilbert_crayons : ℕ := 4 * judah_crayons
noncomputable def beatrice_crayons : ℕ := 2 * gilbert_crayons

theorem ratio_karen_beatrice :
  karen_crayons / beatrice_crayons = 2 := by
sorry

end NUMINAMATH_GPT_ratio_karen_beatrice_l2056_205613


namespace NUMINAMATH_GPT_not_p_and_pq_false_not_necessarily_p_or_q_l2056_205637

theorem not_p_and_pq_false_not_necessarily_p_or_q (p q : Prop) 
  (h1 : ¬p) 
  (h2 : ¬(p ∧ q)) : ¬(p ∨ q) ∨ (p ∨ q) := by
  sorry

end NUMINAMATH_GPT_not_p_and_pq_false_not_necessarily_p_or_q_l2056_205637


namespace NUMINAMATH_GPT_total_tickets_used_l2056_205627

theorem total_tickets_used :
  let shooting_game_cost := 5
  let carousel_cost := 3
  let jen_games := 2
  let russel_rides := 3
  let jen_total := shooting_game_cost * jen_games
  let russel_total := carousel_cost * russel_rides
  jen_total + russel_total = 19 :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_total_tickets_used_l2056_205627


namespace NUMINAMATH_GPT_largest_divisible_n_l2056_205628

/-- Largest positive integer n for which n^3 + 10 is divisible by n + 1 --/
theorem largest_divisible_n (n : ℕ) :
  n = 0 ↔ ∀ m : ℕ, (m > n) → ¬ ((m^3 + 10) % (m + 1) = 0) :=
by
  sorry

end NUMINAMATH_GPT_largest_divisible_n_l2056_205628


namespace NUMINAMATH_GPT_cos_identity_of_angle_l2056_205614

open Real

theorem cos_identity_of_angle (α : ℝ) :
  sin (π / 6 + α) = sqrt 3 / 3 → cos (π / 3 - α) = sqrt 3 / 3 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_cos_identity_of_angle_l2056_205614


namespace NUMINAMATH_GPT_find_sum_of_a_and_b_l2056_205626

variable (a b w y z S : ℕ)

-- Conditions based on problem statement
axiom condition1 : 19 + w + 23 = S
axiom condition2 : 22 + y + a = S
axiom condition3 : b + 18 + z = S
axiom condition4 : 19 + 22 + b = S
axiom condition5 : w + y + 18 = S
axiom condition6 : 23 + a + z = S
axiom condition7 : 19 + y + z = S
axiom condition8 : 23 + y + b = S

theorem find_sum_of_a_and_b : a + b = 23 :=
by
  sorry  -- To be provided with the actual proof later

end NUMINAMATH_GPT_find_sum_of_a_and_b_l2056_205626


namespace NUMINAMATH_GPT_expression_value_at_neg3_l2056_205610

theorem expression_value_at_neg3 (p q : ℤ) (h : 27 * p + 3 * q = 14) :
  (p * (-3)^3 + q * (-3) - 1) = -15 :=
sorry

end NUMINAMATH_GPT_expression_value_at_neg3_l2056_205610


namespace NUMINAMATH_GPT_total_bill_is_correct_l2056_205629

-- Given conditions
def hourly_rate := 45
def parts_cost := 225
def hours_worked := 5

-- Total bill calculation
def labor_cost := hourly_rate * hours_worked
def total_bill := labor_cost + parts_cost

-- Prove that the total bill is equal to 450 dollars
theorem total_bill_is_correct : total_bill = 450 := by
  sorry

end NUMINAMATH_GPT_total_bill_is_correct_l2056_205629


namespace NUMINAMATH_GPT_pants_cost_l2056_205604

def total_cost (P : ℕ) : ℕ := 4 * 8 + 2 * 60 + 2 * P

theorem pants_cost :
  (∃ P : ℕ, total_cost P = 188) →
  ∃ P : ℕ, P = 18 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_pants_cost_l2056_205604
