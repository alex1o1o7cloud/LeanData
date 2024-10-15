import Mathlib

namespace NUMINAMATH_GPT_cone_lateral_area_l1884_188466

theorem cone_lateral_area (cos_ASB : ℝ)
  (angle_SA_base : ℝ)
  (triangle_SAB_area : ℝ) :
  cos_ASB = 7 / 8 →
  angle_SA_base = 45 →
  triangle_SAB_area = 5 * Real.sqrt 15 →
  (lateral_area : ℝ) = 40 * Real.sqrt 2 * Real.pi :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_cone_lateral_area_l1884_188466


namespace NUMINAMATH_GPT_move_line_up_l1884_188401

theorem move_line_up (x : ℝ) :
  let y_initial := 4 * x - 1
  let y_moved := y_initial + 2
  y_moved = 4 * x + 1 :=
by
  let y_initial := 4 * x - 1
  let y_moved := y_initial + 2
  show y_moved = 4 * x + 1
  sorry

end NUMINAMATH_GPT_move_line_up_l1884_188401


namespace NUMINAMATH_GPT_probability_both_blue_buttons_l1884_188469

theorem probability_both_blue_buttons :
  let initial_red_C := 6
  let initial_blue_C := 12
  let initial_total_C := initial_red_C + initial_blue_C
  let remaining_fraction_C := 2 / 3
  let remaining_total_C := initial_total_C * remaining_fraction_C
  let removed_buttons := initial_total_C - remaining_total_C
  let removed_red := removed_buttons / 2
  let removed_blue := removed_buttons / 2
  let remaining_blue_C := initial_blue_C - removed_blue
  let total_remaining_C := remaining_total_C
  let probability_blue_C := remaining_blue_C / total_remaining_C
  let probability_blue_D := removed_blue / removed_buttons
  probability_blue_C * probability_blue_D = 3 / 8 :=
by
  sorry

end NUMINAMATH_GPT_probability_both_blue_buttons_l1884_188469


namespace NUMINAMATH_GPT_max_discount_l1884_188454

theorem max_discount (cost_price selling_price : ℝ) (min_profit_margin : ℝ) (x : ℝ) : 
  cost_price = 400 → selling_price = 500 → min_profit_margin = 0.0625 → 
  (selling_price * (1 - x / 100) - cost_price ≥ min_profit_margin * cost_price) → x ≤ 15 :=
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_max_discount_l1884_188454


namespace NUMINAMATH_GPT_find_value_of_triangle_l1884_188438

theorem find_value_of_triangle (p : ℕ) (triangle : ℕ) 
  (h1 : triangle + p = 47) 
  (h2 : 3 * (triangle + p) - p = 133) :
  triangle = 39 :=
by 
  sorry

end NUMINAMATH_GPT_find_value_of_triangle_l1884_188438


namespace NUMINAMATH_GPT_sequence_inequality_l1884_188440

open Real

def seq (F : ℕ → ℝ) : Prop :=
  F 1 = 1 ∧ F 2 = 2 ∧ ∀ n ≥ 1, F (n + 2) = F (n + 1) + F n

theorem sequence_inequality (F : ℕ → ℝ) (h : seq F) (n : ℕ) : 
  sqrt (F (n+1))^(1/(n:ℝ)) ≥ 1 + 1 / sqrt (F n)^(1/(n:ℝ)) :=
sorry

end NUMINAMATH_GPT_sequence_inequality_l1884_188440


namespace NUMINAMATH_GPT_quadruples_solution_l1884_188485

noncomputable
def valid_quadruples (x1 x2 x3 x4 : ℝ) : Prop :=
  (x1 + x2 * x3 * x4 = 2) ∧
  (x2 + x1 * x3 * x4 = 2) ∧
  (x3 + x1 * x2 * x4 = 2) ∧
  (x4 + x1 * x2 * x3 = 2) ∧
  (x1 ≠ 0) ∧ (x2 ≠ 0) ∧ (x3 ≠ 0) ∧ (x4 ≠ 0)

theorem quadruples_solution (x1 x2 x3 x4 : ℝ) :
  valid_quadruples x1 x2 x3 x4 ↔ 
  (x1 = 1 ∧ x2 = 1 ∧ x3 = 1 ∧ x4 = 1) ∨
  (x1 = -1 ∧ x2 = -1 ∧ x3 = -1 ∧ x4 = 3) ∨
  (x1 = -1 ∧ x2 = -1 ∧ x3 = 3 ∧ x4 = -1) ∨
  (x1 = -1 ∧ x2 = 3 ∧ x3 = -1 ∧ x4 = -1) ∨
  (x1 = 3 ∧ x2 = -1 ∧ x3 = -1 ∧ x4 = -1) := 
by sorry

end NUMINAMATH_GPT_quadruples_solution_l1884_188485


namespace NUMINAMATH_GPT_ratio_of_shaded_to_white_l1884_188479

theorem ratio_of_shaded_to_white (A : ℝ) : 
  let shaded_area := 5 * A
  let unshaded_area := 3 * A
  shaded_area / unshaded_area = 5 / 3 := by
  sorry

end NUMINAMATH_GPT_ratio_of_shaded_to_white_l1884_188479


namespace NUMINAMATH_GPT_alice_favorite_number_l1884_188477

def is_multiple (x y : ℕ) : Prop := ∃ k : ℕ, k * y = x
def digit_sum (n : ℕ) : ℕ := n.digits 10 |>.sum

theorem alice_favorite_number 
  (n : ℕ) 
  (h1 : 90 ≤ n ∧ n ≤ 150) 
  (h2 : is_multiple n 13) 
  (h3 : ¬ is_multiple n 4) 
  (h4 : is_multiple (digit_sum n) 4) : 
  n = 143 := 
by 
  sorry

end NUMINAMATH_GPT_alice_favorite_number_l1884_188477


namespace NUMINAMATH_GPT_find_rate_l1884_188471

def simple_interest_rate (P A T : ℕ) : ℕ :=
  ((A - P) * 100) / (P * T)

theorem find_rate :
  simple_interest_rate 750 1200 5 = 12 :=
by
  -- This is the statement of equality we need to prove
  sorry

end NUMINAMATH_GPT_find_rate_l1884_188471


namespace NUMINAMATH_GPT_green_team_final_score_l1884_188439

theorem green_team_final_score (G : ℕ) :
  (∀ G : ℕ, 68 = G + 29 → G = 39) :=
by
  sorry

end NUMINAMATH_GPT_green_team_final_score_l1884_188439


namespace NUMINAMATH_GPT_composite_function_increasing_l1884_188425

variable {F : ℝ → ℝ}

/-- An odd function is a function that satisfies f(-x) = -f(x) for all x. -/
def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- A function is strictly increasing on negative values if it satisfies the given conditions. -/
def strictly_increasing_on_neg (f : ℝ → ℝ) : Prop :=
  ∀ x1 x2, x1 < x2 → x2 < 0 → f x1 < f x2

/-- Combining properties of an odd function and strictly increasing for negative inputs:
  We need to prove that the composite function is strictly increasing for positive inputs. -/
theorem composite_function_increasing (hf_odd : odd_function F)
    (hf_strict_inc_neg : strictly_increasing_on_neg F)
    : ∀ x1 x2, 0 < x1 → 0 < x2 → x1 < x2 → F (F x1) < F (F x2) :=
  sorry

end NUMINAMATH_GPT_composite_function_increasing_l1884_188425


namespace NUMINAMATH_GPT_probability_jammed_l1884_188441

theorem probability_jammed (T τ : ℝ) (h : τ < T) : 
    (2 * τ / T - (τ / T) ^ 2) = (T^2 - (T - τ)^2) / T^2 := 
by
  sorry

end NUMINAMATH_GPT_probability_jammed_l1884_188441


namespace NUMINAMATH_GPT_smallest_n_for_107n_same_last_two_digits_l1884_188461

theorem smallest_n_for_107n_same_last_two_digits :
  ∃ n : ℕ, n > 0 ∧ (107 * n) % 100 = n % 100 ∧ n = 50 :=
by {
  sorry
}

end NUMINAMATH_GPT_smallest_n_for_107n_same_last_two_digits_l1884_188461


namespace NUMINAMATH_GPT_multiple_6_9_statements_false_l1884_188405

theorem multiple_6_9_statements_false
    (a b : ℤ)
    (h₁ : ∃ m : ℤ, a = 6 * m)
    (h₂ : ∃ n : ℤ, b = 9 * n) :
    ¬ (∀ m n : ℤ,  a = 6 * m → b = 9 * n → ((a + b) % 2 = 0)) ∧
    ¬ (∀ m n : ℤ,  a = 6 * m → b = 9 * n → (a + b) % 6 = 0) ∧
    ¬ (∀ m n : ℤ,  a = 6 * m → b = 9 * n → (a + b) % 9 = 0) ∧
    ¬ (∀ m n : ℤ,  a = 6 * m → b = 9 * n → (a + b) % 9 ≠ 0) :=
by
  sorry

end NUMINAMATH_GPT_multiple_6_9_statements_false_l1884_188405


namespace NUMINAMATH_GPT_Paul_work_time_l1884_188414

def work_completed (rate: ℚ) (time: ℚ) : ℚ := rate * time

noncomputable def George_work_rate : ℚ := 3 / 5 / 9

noncomputable def combined_work_rate : ℚ := 2 / 5 / 4

noncomputable def Paul_work_rate : ℚ := combined_work_rate - George_work_rate

theorem Paul_work_time :
  (work_completed Paul_work_rate 30) = 1 :=
by
  have h_george_rate : George_work_rate = 1 / 15 :=
    by norm_num [George_work_rate]
  have h_combined_rate : combined_work_rate = 1 / 10 :=
    by norm_num [combined_work_rate]
  have h_paul_rate : Paul_work_rate = 1 / 30 :=
    by norm_num [Paul_work_rate, h_combined_rate, h_george_rate]
  sorry -- Complete proof statement here

end NUMINAMATH_GPT_Paul_work_time_l1884_188414


namespace NUMINAMATH_GPT_cube_surface_area_is_24_l1884_188427

def edge_length : ℝ := 2

def surface_area_of_cube (a : ℝ) : ℝ := 6 * a * a

theorem cube_surface_area_is_24 : surface_area_of_cube edge_length = 24 := 
by 
  -- Compute the surface area of the cube with given edge length
  -- surface_area_of_cube 2 = 6 * 2 * 2 = 24
  sorry

end NUMINAMATH_GPT_cube_surface_area_is_24_l1884_188427


namespace NUMINAMATH_GPT_unit_prices_possible_combinations_l1884_188410

-- Part 1: Unit Prices
theorem unit_prices (x y : ℕ) (h1 : x = y - 20) (h2 : 3 * x + 2 * y = 340) : x = 60 ∧ y = 80 := 
by 
  sorry

-- Part 2: Possible Combinations
theorem possible_combinations (a : ℕ) (h3 : 60 * a + 80 * (150 - a) ≤ 10840) (h4 : 150 - a ≥ 3 * a / 2) : 
  a = 58 ∨ a = 59 ∨ a = 60 := 
by 
  sorry

end NUMINAMATH_GPT_unit_prices_possible_combinations_l1884_188410


namespace NUMINAMATH_GPT_compute_series_l1884_188487

noncomputable def sum_series (c d : ℝ) : ℝ :=
  ∑' n, 1 / ((n-1) * d - (n-2) * c) / (n * d - (n-1) * c)

theorem compute_series (c d : ℝ) (hc_pos : 0 < c) (hd_pos : 0 < d) (hcd : d < c) : 
  sum_series c d = 1 / ((d - c) * c) :=
sorry

end NUMINAMATH_GPT_compute_series_l1884_188487


namespace NUMINAMATH_GPT_mr_williams_land_percentage_l1884_188449

-- Given conditions
def farm_tax_percent : ℝ := 60
def total_tax_collected : ℝ := 5000
def mr_williams_tax_paid : ℝ := 480

-- Theorem statement
theorem mr_williams_land_percentage :
  (mr_williams_tax_paid / total_tax_collected) * 100 = 9.6 := by
  sorry

end NUMINAMATH_GPT_mr_williams_land_percentage_l1884_188449


namespace NUMINAMATH_GPT_canadian_olympiad_2008_inequality_l1884_188472

variable (a b c : ℝ)
variables (positive_a : 0 < a) (positive_b : 0 < b) (positive_c : 0 < c)
variable (sum_abc : a + b + c = 1)

theorem canadian_olympiad_2008_inequality :
  (ab / ((b + c) * (c + a))) + (bc / ((c + a) * (a + b))) + (ca / ((a + b) * (b + c))) ≥ 3 / 4 :=
sorry

end NUMINAMATH_GPT_canadian_olympiad_2008_inequality_l1884_188472


namespace NUMINAMATH_GPT_student_history_mark_l1884_188428

theorem student_history_mark
  (math_score : ℕ)
  (desired_average : ℕ)
  (third_subject_score : ℕ)
  (history_score : ℕ) :
  math_score = 74 →
  desired_average = 75 →
  third_subject_score = 70 →
  (math_score + history_score + third_subject_score) / 3 = desired_average →
  history_score = 81 :=
by
  intros h_math h_avg h_third h_equiv
  sorry

end NUMINAMATH_GPT_student_history_mark_l1884_188428


namespace NUMINAMATH_GPT_person_c_completion_time_l1884_188490

def job_completion_days (Ra Rb Rc : ℚ) (total_earnings b_earnings : ℚ) : ℚ :=
  Rc

theorem person_c_completion_time (Ra Rb Rc : ℚ)
  (hRa : Ra = 1 / 6)
  (hRb : Rb = 1 / 8)
  (total_earnings : ℚ)
  (b_earnings : ℚ)
  (earnings_ratio : b_earnings / total_earnings = Rb / (Ra + Rb + Rc))
  : Rc = 1 / 12 :=
sorry

end NUMINAMATH_GPT_person_c_completion_time_l1884_188490


namespace NUMINAMATH_GPT_chord_length_of_intersection_l1884_188429

def ellipse (x y : ℝ) := x^2 + 4 * y^2 = 16
def line (x y : ℝ) := y = (1/2) * x + 1

theorem chord_length_of_intersection :
  ∃ A B : ℝ × ℝ, ellipse A.fst A.snd ∧ ellipse B.fst B.snd ∧ line A.fst A.snd ∧ line B.fst B.snd ∧
  dist A B = Real.sqrt 35 :=
sorry

end NUMINAMATH_GPT_chord_length_of_intersection_l1884_188429


namespace NUMINAMATH_GPT_number_of_algebra_textbooks_l1884_188421

theorem number_of_algebra_textbooks
  (x y n : ℕ)
  (h₁ : x * n + y = 2015)
  (h₂ : y * n + x = 1580) :
  y = 287 := 
sorry

end NUMINAMATH_GPT_number_of_algebra_textbooks_l1884_188421


namespace NUMINAMATH_GPT_red_balls_count_l1884_188481

theorem red_balls_count (white_balls_ratio : ℕ) (red_balls_ratio : ℕ) (total_white_balls : ℕ)
  (h_ratio : white_balls_ratio = 3 ∧ red_balls_ratio = 2)
  (h_white_balls : total_white_balls = 9) :
  ∃ (total_red_balls : ℕ), total_red_balls = 6 :=
by
  sorry

end NUMINAMATH_GPT_red_balls_count_l1884_188481


namespace NUMINAMATH_GPT_population_decreases_l1884_188403

theorem population_decreases (P_0 : ℝ) (k : ℝ) (n : ℕ) (hP0 : P_0 > 0) (hk : -1 < k ∧ k < 0) : 
  P_0 * (1 + k)^n * k < 0 → P_0 * (1 + k)^(n + 1) < P_0 * (1 + k)^n := by
  sorry

end NUMINAMATH_GPT_population_decreases_l1884_188403


namespace NUMINAMATH_GPT_cost_of_paper_l1884_188412

noncomputable def cost_of_paper_per_kg (edge_length : ℕ) (coverage_per_kg : ℕ) (expenditure : ℕ) : ℕ :=
  let surface_area := 6 * edge_length * edge_length
  let paper_needed := surface_area / coverage_per_kg
  expenditure / paper_needed

theorem cost_of_paper (h1 : edge_length = 10) (h2 : coverage_per_kg = 20) (h3 : expenditure = 1800) : 
  cost_of_paper_per_kg 10 20 1800 = 60 :=
by
  -- Using the hypothesis to directly derive the result.
  unfold cost_of_paper_per_kg
  sorry

end NUMINAMATH_GPT_cost_of_paper_l1884_188412


namespace NUMINAMATH_GPT_polygons_ratio_four_three_l1884_188418

theorem polygons_ratio_four_three : 
  ∃ (r k : ℕ), 3 ≤ r ∧ 3 ≤ k ∧ 
  (180 - (360 / r : ℝ)) / (180 - (360 / k : ℝ)) = 4 / 3 
  ∧ ((r, k) = (42,7) ∨ (r, k) = (18,6) ∨ (r, k) = (10,5) ∨ (r, k) = (6,4)) :=
sorry

end NUMINAMATH_GPT_polygons_ratio_four_three_l1884_188418


namespace NUMINAMATH_GPT_jane_average_speed_correct_l1884_188402

noncomputable def jane_average_speed : ℝ :=
  let total_distance : ℝ := 250
  let total_time : ℝ := 6
  total_distance / total_time

theorem jane_average_speed_correct : jane_average_speed = 41.67 := by
  sorry

end NUMINAMATH_GPT_jane_average_speed_correct_l1884_188402


namespace NUMINAMATH_GPT_parabola_directrix_l1884_188415

theorem parabola_directrix :
  ∀ (p : ℝ), (y^2 = 6 * x) → (x = -3/2) :=
by
  sorry

end NUMINAMATH_GPT_parabola_directrix_l1884_188415


namespace NUMINAMATH_GPT_problem_statement_l1884_188426

variable (x y : ℝ)

theorem problem_statement
  (h1 : 4 * x + y = 9)
  (h2 : x + 4 * y = 16) :
  18 * x^2 + 20 * x * y + 18 * y^2 = 337 :=
sorry

end NUMINAMATH_GPT_problem_statement_l1884_188426


namespace NUMINAMATH_GPT_vector_subtraction_l1884_188486

def a : ℝ × ℝ × ℝ := (1, -2, 1)
def b : ℝ × ℝ × ℝ := (1, 0, 2)

theorem vector_subtraction : a - b = (0, -2, -1) := 
by 
  unfold a b
  simp
  sorry

end NUMINAMATH_GPT_vector_subtraction_l1884_188486


namespace NUMINAMATH_GPT_pipe_length_l1884_188433

theorem pipe_length (S L : ℕ) (h1: S = 28) (h2: L = S + 12) : S + L = 68 := 
by
  sorry

end NUMINAMATH_GPT_pipe_length_l1884_188433


namespace NUMINAMATH_GPT_determine_range_of_m_l1884_188419

variable {m : ℝ}

-- Condition (p) for all x in ℝ, x^2 - mx + 3/2 > 0
def condition_p (m : ℝ) : Prop :=
  ∀ x : ℝ, x^2 - m * x + (3 / 2) > 0

-- Condition (q) the foci of the ellipse lie on the x-axis, implying 2 < m < 3
def condition_q (m : ℝ) : Prop :=
  (m - 1 > 0) ∧ ((3 - m) > 0) ∧ ((m - 1) > (3 - m))

theorem determine_range_of_m (h1 : condition_p m) (h2 : condition_q m) : 2 < m ∧ m < Real.sqrt 6 :=
  sorry

end NUMINAMATH_GPT_determine_range_of_m_l1884_188419


namespace NUMINAMATH_GPT_min_5a2_plus_6a3_l1884_188436

theorem min_5a2_plus_6a3 (a_1 a_2 a_3 : ℝ) (r : ℝ)
  (h1 : a_1 = 2)
  (h2 : a_2 = a_1 * r)
  (h3 : a_3 = a_1 * r^2) :
  5 * a_2 + 6 * a_3 ≥ -25 / 12 :=
by
  sorry

end NUMINAMATH_GPT_min_5a2_plus_6a3_l1884_188436


namespace NUMINAMATH_GPT_incorrect_statement_A_l1884_188480

-- Definitions for the conditions
def conditionA (x : ℝ) : Prop := -3 * x > 9
def conditionB (x : ℝ) : Prop := 2 * x - 1 < 0
def conditionC (x : ℤ) : Prop := x < 10
def conditionD (x : ℤ) : Prop := x < 2

-- Formal theorem statement
theorem incorrect_statement_A : ¬ (∀ x : ℝ, conditionA x ↔ x < -3) :=
by 
  sorry

end NUMINAMATH_GPT_incorrect_statement_A_l1884_188480


namespace NUMINAMATH_GPT_range_of_a_l1884_188492

theorem range_of_a (a : ℝ) (h1 : a > 0) (h2 : 3 * a ≥ 1) (h3 : 4 * a ≤ 3 / 2) : 
  (1 / 3) ≤ a ∧ a ≤ (3 / 8) :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l1884_188492


namespace NUMINAMATH_GPT_find_n_l1884_188493

def sum_for (x : ℕ) : ℕ :=
  if x > 1 then (List.range (2*x)).sum else 0

theorem find_n (n : ℕ) (h : n * (sum_for 4) = 360) : n = 10 :=
by
  sorry

end NUMINAMATH_GPT_find_n_l1884_188493


namespace NUMINAMATH_GPT_find_d1_l1884_188443

noncomputable def E (n : ℕ) : ℕ := sorry

theorem find_d1 :
  ∃ (d4 d3 d2 d0 : ℤ), 
  (∀ (n : ℕ), n ≥ 4 ∧ n % 2 = 0 → 
     E n = d4 * n^4 + d3 * n^3 + d2 * n^2 + (12 : ℤ) * n + d0) :=
sorry

end NUMINAMATH_GPT_find_d1_l1884_188443


namespace NUMINAMATH_GPT_curve_intersects_self_at_6_6_l1884_188408

-- Definitions for the given conditions
def x (t : ℝ) : ℝ := t^2 - 3
def y (t : ℝ) : ℝ := t^4 - t^2 - 9 * t + 6

-- Lean statement stating that the curve intersects itself at the coordinate (6, 6)
theorem curve_intersects_self_at_6_6 :
  ∃ t1 t2 : ℝ, t1 ≠ t2 ∧ x t1 = x t2 ∧ y t1 = y t2 ∧ x t1 = 6 ∧ y t1 = 6 :=
by
  sorry

end NUMINAMATH_GPT_curve_intersects_self_at_6_6_l1884_188408


namespace NUMINAMATH_GPT_percentage_salt_l1884_188442

-- Variables
variables {S1 S2 R : ℝ}

-- Conditions
def first_solution := S1
def second_solution := (25 / 100) * 19.000000000000007
def resulting_solution := 16

theorem percentage_salt (S1 S2 : ℝ) (H1: S2 = 19.000000000000007) 
(H2: (75 / 100) * S1 + (25 / 100) * S2 = 16) : 
S1 = 15 :=
by
    rw [H1] at H2
    sorry

end NUMINAMATH_GPT_percentage_salt_l1884_188442


namespace NUMINAMATH_GPT_johns_pieces_of_gum_l1884_188448

theorem johns_pieces_of_gum : 
  (∃ (john cole aubrey : ℕ), 
    cole = 45 ∧ 
    aubrey = 0 ∧ 
    (john + cole + aubrey) = 3 * 33) → 
  ∃ john : ℕ, john = 54 :=
by 
  sorry

end NUMINAMATH_GPT_johns_pieces_of_gum_l1884_188448


namespace NUMINAMATH_GPT_overlap_32_l1884_188420

section
variables (t : ℝ)
def position_A : ℝ := 120 - 50 * t
def position_B : ℝ := 220 - 50 * t
def position_N : ℝ := 30 * t - 30
def position_M : ℝ := 30 * t + 10

theorem overlap_32 :
  (∃ t : ℝ, (30 * t + 10 - (120 - 50 * t) = 32) ∨ 
            (-50 * t + 220 - (30 * t - 30) = 32)) ↔
  (t = 71 / 40 ∨ t = 109 / 40) :=
sorry
end

end NUMINAMATH_GPT_overlap_32_l1884_188420


namespace NUMINAMATH_GPT_g_at_10_is_300_l1884_188498

-- Define the function g and the given condition about g
def g: ℕ → ℤ := sorry

axiom g_cond (m n: ℕ) (h: m ≥ n): g (m + n) + g (m - n) = 2 * g m + 3 * g n
axiom g_1: g 1 = 3

-- Statement to be proved
theorem g_at_10_is_300 : g 10 = 300 := by
  sorry

end NUMINAMATH_GPT_g_at_10_is_300_l1884_188498


namespace NUMINAMATH_GPT_total_number_of_cottages_is_100_l1884_188497

noncomputable def total_cottages
    (x : ℕ) (n : ℕ) 
    (h1 : 2 * x = number_of_two_room_cottages)
    (h2 : n * x = number_of_three_room_cottages)
    (h3 : 3 * (n * x) = 2 * x + 25) 
    (h4 : x + 2 * x + n * x ≥ 70) : ℕ :=
x + 2 * x + n * x

theorem total_number_of_cottages_is_100 
    (x n : ℕ)
    (h1 : 2 * x = number_of_two_room_cottages)
    (h2 : n * x = number_of_three_room_cottages)
    (h3 : 3 * (n * x) = 2 * x + 25)
    (h4 : x + 2 * x + n * x ≥ 70)
    (h5 : ∃ m : ℕ, m = (x + 2 * x + n * x)) :
  total_cottages x n h1 h2 h3 h4 = 100 :=
by
  sorry

end NUMINAMATH_GPT_total_number_of_cottages_is_100_l1884_188497


namespace NUMINAMATH_GPT_min_people_liking_both_l1884_188431

theorem min_people_liking_both (A B C V : ℕ) (hA : A = 200) (hB : B = 150) (hC : C = 120) (hV : V = 80) :
  ∃ D, D = 80 ∧ D ≤ min B (A - C + V) :=
by {
  sorry
}

end NUMINAMATH_GPT_min_people_liking_both_l1884_188431


namespace NUMINAMATH_GPT_am_gm_example_l1884_188459

variable {x y z : ℝ}

theorem am_gm_example (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) : 
  x / y + y / z + z / x + y / x + z / y + x / z ≥ 6 :=
sorry

end NUMINAMATH_GPT_am_gm_example_l1884_188459


namespace NUMINAMATH_GPT_larger_fraction_l1884_188499

theorem larger_fraction :
  (22222222221 : ℚ) / 22222222223 > (33333333331 : ℚ) / 33333333334 := by sorry

end NUMINAMATH_GPT_larger_fraction_l1884_188499


namespace NUMINAMATH_GPT_remainder_zero_when_x_divided_by_y_l1884_188476

theorem remainder_zero_when_x_divided_by_y :
  ∀ (x y : ℝ), 
    0 < x ∧ 0 < y ∧ x / y = 6.12 ∧ y = 49.99999999999996 → 
      x % y = 0 := by
  sorry

end NUMINAMATH_GPT_remainder_zero_when_x_divided_by_y_l1884_188476


namespace NUMINAMATH_GPT_sin_alpha_sqrt5_div5_and_sin_beta_sqrt10_div10_acute_sum_pi_div4_l1884_188435

theorem sin_alpha_sqrt5_div5_and_sin_beta_sqrt10_div10_acute_sum_pi_div4
  (α β : ℝ)
  (hα : 0 < α ∧ α < π / 2)
  (hβ : 0 < β ∧ β < π / 2)
  (h_sin_α : Real.sin α = Real.sqrt 5 / 5)
  (h_sin_β : Real.sin β = Real.sqrt 10 / 10) :
  α + β = π / 4 := sorry

end NUMINAMATH_GPT_sin_alpha_sqrt5_div5_and_sin_beta_sqrt10_div10_acute_sum_pi_div4_l1884_188435


namespace NUMINAMATH_GPT_existence_of_committees_l1884_188473

noncomputable def committeesExist : Prop :=
∃ (C : Fin 1990 → Fin 11 → Fin 3), 
  (∀ i j, i ≠ j → C i ≠ C j) ∧
  (∀ i j, i = j + 1 ∨ (i = 0 ∧ j = 1990 - 1) → ∃ k, C i k = C j k)

theorem existence_of_committees : committeesExist :=
sorry

end NUMINAMATH_GPT_existence_of_committees_l1884_188473


namespace NUMINAMATH_GPT_Bruce_grape_purchase_l1884_188470

theorem Bruce_grape_purchase
  (G : ℕ)
  (total_paid : ℕ)
  (cost_per_kg_grapes : ℕ)
  (kg_mangoes : ℕ)
  (cost_per_kg_mangoes : ℕ)
  (total_mango_cost : ℕ)
  (total_grape_cost : ℕ)
  (total_amount : ℕ)
  (h1 : cost_per_kg_grapes = 70)
  (h2 : kg_mangoes = 10)
  (h3 : cost_per_kg_mangoes = 55)
  (h4 : total_paid = 1110)
  (h5 : total_mango_cost = kg_mangoes * cost_per_kg_mangoes)
  (h6 : total_grape_cost = G * cost_per_kg_grapes)
  (h7 : total_amount = total_mango_cost + total_grape_cost)
  (h8 : total_amount = total_paid) :
  G = 8 := by
  sorry

end NUMINAMATH_GPT_Bruce_grape_purchase_l1884_188470


namespace NUMINAMATH_GPT_innings_count_l1884_188406

-- Definitions of the problem conditions
def total_runs (n : ℕ) : ℕ := 63 * n
def highest_score : ℕ := 248
def lowest_score : ℕ := 98

theorem innings_count (n : ℕ) (h : total_runs n - highest_score - lowest_score = 58 * (n - 2)) : n = 46 :=
  sorry

end NUMINAMATH_GPT_innings_count_l1884_188406


namespace NUMINAMATH_GPT_find_x_values_l1884_188432

theorem find_x_values (x : ℝ) :
  (3 * x + 2 < (x - 1) ^ 2 ∧ (x - 1) ^ 2 < 9 * x + 1) ↔
  (x > (5 + Real.sqrt 29) / 2 ∧ x < 11) := 
by
  sorry

end NUMINAMATH_GPT_find_x_values_l1884_188432


namespace NUMINAMATH_GPT_commute_times_l1884_188460

theorem commute_times (x y : ℝ) 
  (h1 : x + y = 20)
  (h2 : (x - 10)^2 + (y - 10)^2 = 8) : |x - y| = 4 := 
sorry

end NUMINAMATH_GPT_commute_times_l1884_188460


namespace NUMINAMATH_GPT_completing_the_square_l1884_188474

theorem completing_the_square (x : ℝ) :
  x^2 + 4 * x + 1 = 0 ↔ (x + 2)^2 = 3 :=
by
  sorry

end NUMINAMATH_GPT_completing_the_square_l1884_188474


namespace NUMINAMATH_GPT_avg_age_9_proof_l1884_188404

-- Definitions of the given conditions
def total_persons := 16
def avg_age_all := 15
def total_age_all := total_persons * avg_age_all -- 240
def persons_5 := 5
def avg_age_5 := 14
def total_age_5 := persons_5 * avg_age_5 -- 70
def age_15th_person := 26
def persons_9 := 9

-- The theorem to prove the average age of the remaining 9 persons
theorem avg_age_9_proof : 
  total_age_all - total_age_5 - age_15th_person = persons_9 * 16 :=
by
  sorry

end NUMINAMATH_GPT_avg_age_9_proof_l1884_188404


namespace NUMINAMATH_GPT_smallest_internal_angle_l1884_188488

theorem smallest_internal_angle (α : ℝ) (β : ℝ) (γ : ℝ)
  (h1 : α = 2 * β) (h2 : α = 3 * γ)
  (h3 : α + β + γ = π) :
  α = π / 6 :=
by
  sorry

end NUMINAMATH_GPT_smallest_internal_angle_l1884_188488


namespace NUMINAMATH_GPT_rectangular_prism_diagonal_inequality_l1884_188453

theorem rectangular_prism_diagonal_inequality 
  (a b c l : ℝ) 
  (h : l^2 = a^2 + b^2 + c^2) :
  (l^4 - a^4) * (l^4 - b^4) * (l^4 - c^4) ≥ 512 * a^4 * b^4 * c^4 := 
by sorry

end NUMINAMATH_GPT_rectangular_prism_diagonal_inequality_l1884_188453


namespace NUMINAMATH_GPT_find_divisor_l1884_188495

theorem find_divisor (x y : ℕ) (h1 : (x - 5) / 7 = 7) (h2 : (x - 14) / y = 4) : y = 10 :=
sorry

end NUMINAMATH_GPT_find_divisor_l1884_188495


namespace NUMINAMATH_GPT_total_savings_l1884_188423

def weekly_savings : ℕ := 15
def weeks_per_cycle : ℕ := 60
def number_of_cycles : ℕ := 5

theorem total_savings :
  (weekly_savings * weeks_per_cycle) * number_of_cycles = 4500 := 
sorry

end NUMINAMATH_GPT_total_savings_l1884_188423


namespace NUMINAMATH_GPT_sets_of_laces_needed_l1884_188458

-- Define the conditions as constants
def teams := 4
def members_per_team := 10
def pairs_per_member := 2
def skates_per_pair := 2
def sets_of_laces_per_skate := 3

-- Formulate and state the theorem to be proven
theorem sets_of_laces_needed : 
  sets_of_laces_per_skate * (teams * members_per_team * (pairs_per_member * skates_per_pair)) = 480 :=
by sorry

end NUMINAMATH_GPT_sets_of_laces_needed_l1884_188458


namespace NUMINAMATH_GPT_sodium_chloride_moles_produced_l1884_188424

theorem sodium_chloride_moles_produced (NaOH HCl NaCl : ℕ) : 
    (NaOH = 3) → (HCl = 3) → NaCl = 3 :=
by
  intro hNaOH hHCl
  -- Placeholder for actual proof
  sorry

end NUMINAMATH_GPT_sodium_chloride_moles_produced_l1884_188424


namespace NUMINAMATH_GPT_sum_q_p_values_l1884_188463

def p (x : ℤ) : ℤ := x^2 - 4

def q (x : ℤ) : ℤ := -abs x

theorem sum_q_p_values : 
  (q (p (-3)) + q (p (-2)) + q (p (-1)) + q (p (0)) + q (p (1)) + q (p (2)) + q (p (3))) = -20 :=
by
  sorry

end NUMINAMATH_GPT_sum_q_p_values_l1884_188463


namespace NUMINAMATH_GPT_sequence_form_l1884_188468

theorem sequence_form {a : ℕ → ℚ} (h_eq : ∀ n : ℕ, a n * x ^ 2 - a (n + 1) * x + 1 = 0) 
  (h_roots : ∀ α β : ℚ, 6 * α - 2 * α * β + 6 * β = 3 ) (h_a1 : a 1 = 7 / 6) :
  ∀ n : ℕ, a n = (1 / 2) ^ n + 2 / 3 :=
by
  sorry

end NUMINAMATH_GPT_sequence_form_l1884_188468


namespace NUMINAMATH_GPT_find_three_digit_number_l1884_188496

theorem find_three_digit_number : 
  ∃ x : ℕ, (x >= 100 ∧ x < 1000) ∧ (2 * x = 3 * x - 108) :=
by
  have h : ∀ x : ℕ, 100 ≤ x → x < 1000 → 2 * x = 3 * x - 108 → x = 108 := sorry
  exact ⟨108, by sorry⟩

end NUMINAMATH_GPT_find_three_digit_number_l1884_188496


namespace NUMINAMATH_GPT_mass_percentage_O_mixture_l1884_188451

noncomputable def molar_mass_Al2O3 : ℝ := (2 * 26.98) + (3 * 16.00)
noncomputable def molar_mass_Cr2O3 : ℝ := (2 * 51.99) + (3 * 16.00)
noncomputable def mass_of_O_in_Al2O3 : ℝ := 3 * 16.00
noncomputable def mass_of_O_in_Cr2O3 : ℝ := 3 * 16.00
noncomputable def mass_percentage_O_in_Al2O3 : ℝ := (mass_of_O_in_Al2O3 / molar_mass_Al2O3) * 100
noncomputable def mass_percentage_O_in_Cr2O3 : ℝ := (mass_of_O_in_Cr2O3 / molar_mass_Cr2O3) * 100
noncomputable def mass_percentage_O_in_mixture : ℝ := (0.50 * mass_percentage_O_in_Al2O3) + (0.50 * mass_percentage_O_in_Cr2O3)

theorem mass_percentage_O_mixture : mass_percentage_O_in_mixture = 39.325 := by
  sorry

end NUMINAMATH_GPT_mass_percentage_O_mixture_l1884_188451


namespace NUMINAMATH_GPT_total_cost_is_660_l1884_188462

def total_material_cost : ℝ :=
  let velvet_area := (12 * 4) * 3
  let velvet_cost := velvet_area * 3
  let silk_cost := 2 * 6
  let lace_cost := 5 * 2 * 10
  let bodice_cost := silk_cost + lace_cost
  let satin_area := 2.5 * 1.5
  let satin_cost := satin_area * 4
  let leather_area := 1 * 1.5 * 2
  let leather_cost := leather_area * 5
  let wool_area := 5 * 2
  let wool_cost := wool_area * 8
  let ribbon_cost := 3 * 2
  velvet_cost + bodice_cost + satin_cost + leather_cost + wool_cost + ribbon_cost

theorem total_cost_is_660 : total_material_cost = 660 := by
  sorry

end NUMINAMATH_GPT_total_cost_is_660_l1884_188462


namespace NUMINAMATH_GPT_circle_m_condition_l1884_188430

theorem circle_m_condition (m : ℝ) : (∃ x y : ℝ, x^2 + y^2 - 2*x + 4*y + m = 0) → m < 5 :=
by
  sorry

end NUMINAMATH_GPT_circle_m_condition_l1884_188430


namespace NUMINAMATH_GPT_value_of_f_l1884_188455

noncomputable def f (a b x : ℝ) := a * Real.exp x + b * x 
noncomputable def f' (a b x : ℝ) := a * Real.exp x + b

theorem value_of_f'_at_1 (a b : ℝ)
  (h₁ : f a b 0 = 1)
  (h₂ : f' (a := a) (b := b) 0 = 0) :
  f' (a := a) (b := b) 1 = Real.exp 1 - 1 :=
by
  sorry

end NUMINAMATH_GPT_value_of_f_l1884_188455


namespace NUMINAMATH_GPT_solution_set_inequality_l1884_188456

theorem solution_set_inequality (m : ℝ) : (∀ x : ℝ, m * x^2 - m * x + 2 > 0) ↔ m ∈ Set.Ico 0 8 := by
  sorry

end NUMINAMATH_GPT_solution_set_inequality_l1884_188456


namespace NUMINAMATH_GPT_common_root_sum_k_l1884_188457

theorem common_root_sum_k :
  (∃ x : ℝ, (x^2 - 4 * x + 3 = 0) ∧ (x^2 - 6 * x + k = 0)) → 
  (∃ (k₁ k₂ : ℝ), (k₁ = 5) ∧ (k₂ = 9) ∧ (k₁ + k₂ = 14)) :=
by
  sorry

end NUMINAMATH_GPT_common_root_sum_k_l1884_188457


namespace NUMINAMATH_GPT_students_failed_to_get_degree_l1884_188467

/-- 
Out of 1,500 senior high school students, 70% passed their English exams,
80% passed their Mathematics exams, and 65% passed their Science exams.
To get their degree, a student must pass in all three subjects.
Assume independence of passing rates. This Lean proof shows that
the number of students who failed to get their degree is 954.
-/
theorem students_failed_to_get_degree :
  let total_students := 1500
  let p_english := 0.70
  let p_math := 0.80
  let p_science := 0.65
  let p_all_pass := p_english * p_math * p_science
  let students_all_pass := p_all_pass * total_students
  total_students - students_all_pass = 954 :=
by
  sorry

end NUMINAMATH_GPT_students_failed_to_get_degree_l1884_188467


namespace NUMINAMATH_GPT_initial_discount_l1884_188464

theorem initial_discount (total_amount price_after_initial_discount additional_disc_percent : ℝ)
  (H1 : total_amount = 1000)
  (H2 : price_after_initial_discount = total_amount - 280)
  (H3 : additional_disc_percent = 0.20) :
  let additional_discount := additional_disc_percent * price_after_initial_discount
  let price_after_additional_discount := price_after_initial_discount - additional_discount
  let total_discount := total_amount - price_after_additional_discount
  let initial_discount := total_discount - additional_discount
  initial_discount = 280 := by
  sorry

end NUMINAMATH_GPT_initial_discount_l1884_188464


namespace NUMINAMATH_GPT_perpendicularity_proof_l1884_188434

-- Definitions of geometric entities and properties
variable (Plane Line : Type)
variable (α β : Plane) -- α and β are planes
variable (m n : Line) -- m and n are lines

-- Geometric properties and relations
variable (subset : Line → Plane → Prop) -- Line is subset of plane
variable (perpendicular : Line → Plane → Prop) -- Line is perpendicular to plane
variable (line_perpendicular : Line → Line → Prop) -- Line is perpendicular to another line

-- Conditions
axiom planes_different : α ≠ β
axiom lines_different : m ≠ n
axiom m_in_beta : subset m β
axiom n_in_beta : subset n β

-- Proof problem statement
theorem perpendicularity_proof :
  (subset m α) → (perpendicular n α) → (line_perpendicular n m) :=
by
  sorry

end NUMINAMATH_GPT_perpendicularity_proof_l1884_188434


namespace NUMINAMATH_GPT_area_ratio_triangle_MNO_XYZ_l1884_188483

noncomputable def triangle_area_ratio (XY YZ XZ p q r : ℝ) : ℝ := sorry

theorem area_ratio_triangle_MNO_XYZ : 
  ∀ (p q r: ℝ),
  p > 0 → q > 0 → r > 0 →
  p + q + r = 3 / 4 →
  p ^ 2 + q ^ 2 + r ^ 2 = 1 / 2 →
  triangle_area_ratio 12 16 20 p q r = 9 / 32 :=
sorry

end NUMINAMATH_GPT_area_ratio_triangle_MNO_XYZ_l1884_188483


namespace NUMINAMATH_GPT_average_of_primes_less_than_twenty_l1884_188445

def primes_less_than_twenty : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19]
def sum_primes : ℕ := 77
def count_primes : ℕ := 8
def average_primes : ℚ := 77 / 8

theorem average_of_primes_less_than_twenty : (primes_less_than_twenty.sum / count_primes : ℚ) = 9.625 := by
  sorry

end NUMINAMATH_GPT_average_of_primes_less_than_twenty_l1884_188445


namespace NUMINAMATH_GPT_chemical_x_percentage_l1884_188413

-- Define the initial volume of the mixture
def initial_volume : ℕ := 80

-- Define the percentage of chemical x in the initial mixture
def percentage_x_initial : ℚ := 0.30

-- Define the volume of chemical x added to the mixture
def added_volume_x : ℕ := 20

-- Define the calculation of the amount of chemical x in the initial mixture
def initial_amount_x : ℚ := percentage_x_initial * initial_volume

-- Define the calculation of the total amount of chemical x after adding more
def total_amount_x : ℚ := initial_amount_x + added_volume_x

-- Define the calculation of the total volume after adding 20 liters of chemical x
def total_volume : ℚ := initial_volume + added_volume_x

-- Define the percentage of chemical x in the final mixture
def percentage_x_final : ℚ := (total_amount_x / total_volume) * 100

-- The proof goal
theorem chemical_x_percentage : percentage_x_final = 44 := 
by
  sorry

end NUMINAMATH_GPT_chemical_x_percentage_l1884_188413


namespace NUMINAMATH_GPT_quadratic_expression_and_intersections_l1884_188407

noncomputable def quadratic_eq_expression (a b c : ℝ) : Prop :=
  ∃ a b c : ℝ, (a * (1:ℝ) ^ 2 + b * (1:ℝ) + c = -3) ∧ (4 * a + 2 * b + c = - 5 / 2) ∧ (b = -2 * a) ∧ (c = -5 / 2) ∧ (a = 1 / 2)

noncomputable def find_m (a b c : ℝ) : Prop :=
  ∀ x m : ℝ, (a * (-2:ℝ)^2 + b * (-2:ℝ) + c = m) → (a * (4:ℝ) + b * (4:ℝ) + c = m) → (6:ℝ) = abs (x - (-2:ℝ)) → m = 3 / 2

noncomputable def y_range (a b c : ℝ) : Prop :=
  ∀ x y : ℝ, 
  (x^2 * a + x * b + c >= -3) ∧ 
  (x^2 * a + x * b + c < 5) ↔ (-3 < x ∧ x < 3)

theorem quadratic_expression_and_intersections 
  (a b c : ℝ) (h1 : quadratic_eq_expression a b c) (h2 : find_m a b c) : y_range a b c :=
  sorry

end NUMINAMATH_GPT_quadratic_expression_and_intersections_l1884_188407


namespace NUMINAMATH_GPT_slip_2_5_in_A_or_C_l1884_188446

-- Define the slips and their values
def slips : List ℚ := [1, 1.5, 2, 2, 2.5, 3, 3, 3.5, 3.5, 4, 4.5, 4.5, 5, 5.5, 6]

-- Define the cups
inductive Cup
| A | B | C | D | E | F

open Cup

-- Define the given cups constraints
def sum_constraints : Cup → ℚ
| A => 6
| B => 7
| C => 8
| D => 9
| E => 10
| F => 10

-- Initial conditions for slips placement
def slips_in_cups (c : Cup) : List ℚ :=
match c with
| F => [1.5]
| B => [4]
| _ => []

-- We'd like to prove that:
def slip_2_5_can_go_into : Prop :=
  (slips_in_cups A = [2.5] ∧ slips_in_cups C = [2.5])

theorem slip_2_5_in_A_or_C : slip_2_5_can_go_into :=
sorry

end NUMINAMATH_GPT_slip_2_5_in_A_or_C_l1884_188446


namespace NUMINAMATH_GPT_deepak_present_age_l1884_188444

def rahul_age (x : ℕ) : ℕ := 4 * x
def deepak_age (x : ℕ) : ℕ := 3 * x

theorem deepak_present_age (x : ℕ) (h1 : rahul_age x + 10 = 26) : deepak_age x = 12 :=
by sorry

end NUMINAMATH_GPT_deepak_present_age_l1884_188444


namespace NUMINAMATH_GPT_dave_paid_3_more_than_doug_l1884_188475

theorem dave_paid_3_more_than_doug :
  let total_slices := 10
  let plain_pizza_cost := 10
  let anchovy_fee := 3
  let total_cost := plain_pizza_cost + anchovy_fee
  let cost_per_slice := total_cost / total_slices
  let slices_with_anchovies := total_slices / 3
  let dave_slices := slices_with_anchovies + 2
  let doug_slices := total_slices - dave_slices
  let doug_pay := doug_slices * plain_pizza_cost / total_slices
  let dave_pay := total_cost - doug_pay
  dave_pay - doug_pay = 3 :=
by
  sorry

end NUMINAMATH_GPT_dave_paid_3_more_than_doug_l1884_188475


namespace NUMINAMATH_GPT_scientific_notation_of_virus_diameter_l1884_188484

theorem scientific_notation_of_virus_diameter :
  0.00000012 = 1.2 * 10 ^ (-7) :=
by
  sorry

end NUMINAMATH_GPT_scientific_notation_of_virus_diameter_l1884_188484


namespace NUMINAMATH_GPT_percentage_boys_not_attended_college_l1884_188465

/-
Define the constants and given conditions.
-/
def number_of_boys : ℕ := 300
def number_of_girls : ℕ := 240
def total_students : ℕ := number_of_boys + number_of_girls
def percentage_class_attended_college : ℝ := 0.70
def percentage_girls_not_attended_college : ℝ := 0.30

/-
The proof problem statement: 
Prove the percentage of the boys class that did not attend college.
-/
theorem percentage_boys_not_attended_college :
  let students_attended_college := percentage_class_attended_college * total_students
  let not_attended_college_students := total_students - students_attended_college
  let not_attended_college_girls := percentage_girls_not_attended_college * number_of_girls
  let not_attended_college_boys := not_attended_college_students - not_attended_college_girls
  let percentage_boys_not_attended_college := (not_attended_college_boys / number_of_boys) * 100
  percentage_boys_not_attended_college = 30 := by
  sorry

end NUMINAMATH_GPT_percentage_boys_not_attended_college_l1884_188465


namespace NUMINAMATH_GPT_janet_better_condition_count_l1884_188478

noncomputable def janet_initial := 10
noncomputable def janet_sells := 6
noncomputable def janet_remaining := janet_initial - janet_sells
noncomputable def brother_gives := 2 * janet_remaining
noncomputable def janet_after_brother := janet_remaining + brother_gives
noncomputable def janet_total := 24

theorem janet_better_condition_count : 
  janet_total - janet_after_brother = 12 := by
  sorry

end NUMINAMATH_GPT_janet_better_condition_count_l1884_188478


namespace NUMINAMATH_GPT_distinct_solutions_eq_l1884_188422

theorem distinct_solutions_eq : ∃! x : ℝ, abs (x - 5) = abs (x + 3) :=
by
  sorry

end NUMINAMATH_GPT_distinct_solutions_eq_l1884_188422


namespace NUMINAMATH_GPT_martha_gingers_amount_l1884_188494

theorem martha_gingers_amount (G : ℚ) (h : G = 0.43 * (G + 3)) : G = 2 := by
  sorry

end NUMINAMATH_GPT_martha_gingers_amount_l1884_188494


namespace NUMINAMATH_GPT_tangent_line_at_one_f_gt_one_l1884_188491

noncomputable def f (x : ℝ) : ℝ :=
  Real.exp x * Real.log x + (2 * Real.exp (x - 1)) / x

theorem tangent_line_at_one : 
  let y := f 1 + (Real.exp 1) * (x - 1)
  y = Real.exp (1 : ℝ) * (x - 1) + 2 := 
sorry

theorem f_gt_one (x : ℝ) (hx : 0 < x) : f x > 1 := 
sorry

end NUMINAMATH_GPT_tangent_line_at_one_f_gt_one_l1884_188491


namespace NUMINAMATH_GPT_jamies_father_days_to_lose_weight_l1884_188450

def calories_per_pound : ℕ := 3500
def pounds_to_lose : ℕ := 5
def calories_burned_per_day : ℕ := 2500
def calories_consumed_per_day : ℕ := 2000
def net_calories_burned_per_day : ℕ := calories_burned_per_day - calories_consumed_per_day
def total_calories_to_burn : ℕ := pounds_to_lose * calories_per_pound
def days_to_burn_calories := total_calories_to_burn / net_calories_burned_per_day

theorem jamies_father_days_to_lose_weight : days_to_burn_calories = 35 := by
  sorry

end NUMINAMATH_GPT_jamies_father_days_to_lose_weight_l1884_188450


namespace NUMINAMATH_GPT_solve_abs_equation_l1884_188437

-- Define the condition for the equation
def condition (x : ℝ) : Prop := 3 * x + 5 ≥ 0

-- The main theorem to prove that x = 1/5 is the only solution
theorem solve_abs_equation (x : ℝ) (h : condition x) : |2 * x - 6| = 3 * x + 5 ↔ x = 1 / 5 := by
  sorry

end NUMINAMATH_GPT_solve_abs_equation_l1884_188437


namespace NUMINAMATH_GPT_abba_divisible_by_11_aaabbb_divisible_by_37_ababab_divisible_by_7_abab_baba_divisible_by_9_and_101_l1884_188409

/-- Part 1: Prove that the number \overline{abba} is divisible by 11 -/
theorem abba_divisible_by_11 (a b : ℕ) : 11 ∣ (1000 * a + 100 * b + 10 * b + a) :=
sorry

/-- Part 2: Prove that the number \overline{aaabbb} is divisible by 37 -/
theorem aaabbb_divisible_by_37 (a b : ℕ) : 37 ∣ (1000 * 111 * a + 111 * b) :=
sorry

/-- Part 3: Prove that the number \overline{ababab} is divisible by 7 -/
theorem ababab_divisible_by_7 (a b : ℕ) : 7 ∣ (100000 * a + 10000 * b + 1000 * a + 100 * b + 10 * a + b) :=
sorry

/-- Part 4: Prove that the number \overline{abab} - \overline{baba} is divisible by 9 and 101 -/
theorem abab_baba_divisible_by_9_and_101 (a b : ℕ) :
  9 ∣ (1000 * a + 100 * b + 10 * a + b - (1000 * b + 100 * a + 10 * b + a)) ∧
  101 ∣ (1000 * a + 100 * b + 10 * a + b - (1000 * b + 100 * a + 10 * b + a)) :=
sorry

end NUMINAMATH_GPT_abba_divisible_by_11_aaabbb_divisible_by_37_ababab_divisible_by_7_abab_baba_divisible_by_9_and_101_l1884_188409


namespace NUMINAMATH_GPT_max_area_of_rectangular_fence_l1884_188416

theorem max_area_of_rectangular_fence (x y : ℕ) (h : x + y = 75) : 
  (x * (75 - x) ≤ 1406) ∧ (∀ x' y', x' + y' = 75 → x' * y' ≤ 1406) :=
by
  sorry

end NUMINAMATH_GPT_max_area_of_rectangular_fence_l1884_188416


namespace NUMINAMATH_GPT_final_values_comparison_l1884_188489

theorem final_values_comparison :
  let AA_initial : ℝ := 100
  let BB_initial : ℝ := 100
  let CC_initial : ℝ := 100
  let AA_year1 := AA_initial * 1.20
  let BB_year1 := BB_initial * 0.75
  let CC_year1 := CC_initial
  let AA_year2 := AA_year1 * 0.80
  let BB_year2 := BB_year1 * 1.25
  let CC_year2 := CC_year1
  AA_year2 = 96 ∧ BB_year2 = 93.75 ∧ CC_year2 = 100 ∧ BB_year2 < AA_year2 ∧ AA_year2 < CC_year2 :=
by {
  -- Definitions from conditions
  let AA_initial : ℝ := 100;
  let BB_initial : ℝ := 100;
  let CC_initial : ℝ := 100;
  let AA_year1 := AA_initial * 1.20;
  let BB_year1 := BB_initial * 0.75;
  let CC_year1 := CC_initial;
  let AA_year2 := AA_year1 * 0.80;
  let BB_year2 := BB_year1 * 1.25;
  let CC_year2 := CC_year1;

  -- Use sorry to skip the actual proof
  sorry
}

end NUMINAMATH_GPT_final_values_comparison_l1884_188489


namespace NUMINAMATH_GPT_tan_C_value_l1884_188482

theorem tan_C_value (A B C : ℝ)
  (h_cos_A : Real.cos A = 4/5)
  (h_tan_A_minus_B : Real.tan (A - B) = -1/2) :
  Real.tan C = 11/2 :=
sorry

end NUMINAMATH_GPT_tan_C_value_l1884_188482


namespace NUMINAMATH_GPT_geometric_progressions_sum_eq_l1884_188452

variable {a q b : ℝ}
variable {n : ℕ}
variable (h1 : q ≠ 1)

/-- The given statement in Lean 4 -/
theorem geometric_progressions_sum_eq (h : a * (q^(3*n) - 1) / (q - 1) = b * (q^(3*n) - 1) / (q^3 - 1)) : 
  b = a * (1 + q + q^2) := 
by
  sorry

end NUMINAMATH_GPT_geometric_progressions_sum_eq_l1884_188452


namespace NUMINAMATH_GPT_min_n_plus_d_l1884_188417

theorem min_n_plus_d (a : ℕ → ℕ) (n d : ℕ) (h1 : a 1 = 1) (h2 : a n = 51)
  (h3 : ∀ i, a i = a 1 + (i-1) * d) : n + d = 16 :=
by
  sorry

end NUMINAMATH_GPT_min_n_plus_d_l1884_188417


namespace NUMINAMATH_GPT_gigi_initial_batches_l1884_188447

-- Define the conditions
def flour_per_batch := 2 
def initial_flour := 20 
def remaining_flour := 14 
def future_batches := 7

-- Prove the number of batches initially baked is 3
theorem gigi_initial_batches :
  (initial_flour - remaining_flour) / flour_per_batch = 3 :=
by
  sorry

end NUMINAMATH_GPT_gigi_initial_batches_l1884_188447


namespace NUMINAMATH_GPT_regular_soda_count_l1884_188400

theorem regular_soda_count 
  (diet_soda : ℕ) 
  (additional_soda : ℕ) 
  (h1 : diet_soda = 19) 
  (h2 : additional_soda = 41) 
  : diet_soda + additional_soda = 60 :=
by
  sorry

end NUMINAMATH_GPT_regular_soda_count_l1884_188400


namespace NUMINAMATH_GPT_paint_can_distribution_l1884_188411

-- Definitions based on conditions provided in the problem.
def ratio_red := 3
def ratio_white := 2
def ratio_blue := 1
def total_paint := 60
def ratio_sum := ratio_red + ratio_white + ratio_blue

-- Definition of the problem to be proved.
theorem paint_can_distribution :
  (ratio_red * total_paint) / ratio_sum = 30 ∧
  (ratio_white * total_paint) / ratio_sum = 20 ∧
  (ratio_blue * total_paint) / ratio_sum = 10 := 
by
  sorry

end NUMINAMATH_GPT_paint_can_distribution_l1884_188411
