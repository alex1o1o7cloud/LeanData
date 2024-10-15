import Mathlib

namespace NUMINAMATH_GPT_determine_d_l641_64194

theorem determine_d (d c : ℕ) (hlcm : Nat.lcm 76 d = 456) (hhcf : Nat.gcd 76 d = c) : d = 24 :=
by
  sorry

end NUMINAMATH_GPT_determine_d_l641_64194


namespace NUMINAMATH_GPT_adam_deleted_items_l641_64140

theorem adam_deleted_items (initial_items deleted_items remaining_items : ℕ)
  (h1 : initial_items = 100) (h2 : remaining_items = 20) 
  (h3 : remaining_items = initial_items - deleted_items) : 
  deleted_items = 80 :=
by
  sorry

end NUMINAMATH_GPT_adam_deleted_items_l641_64140


namespace NUMINAMATH_GPT_remainder_when_divided_by_29_l641_64116

theorem remainder_when_divided_by_29 (N : ℤ) (h : N % 899 = 63) : N % 29 = 10 :=
sorry

end NUMINAMATH_GPT_remainder_when_divided_by_29_l641_64116


namespace NUMINAMATH_GPT_expression_value_l641_64148

theorem expression_value (a b c d : ℝ) 
  (intersect1 : 4 = a * (2:ℝ)^2 + b * 2 + 1) 
  (intersect2 : 4 = (2:ℝ)^2 + c * 2 + d) 
  (hc : b + c = 1) : 
  4 * a + d = 1 := 
sorry

end NUMINAMATH_GPT_expression_value_l641_64148


namespace NUMINAMATH_GPT_bridge_length_l641_64189

theorem bridge_length
  (train_length : ℝ)
  (train_speed_km_hr : ℝ)
  (crossing_time_sec : ℝ)
  (train_speed_m_s : ℝ := train_speed_km_hr * 1000 / 3600)
  (total_distance : ℝ := train_speed_m_s * crossing_time_sec)
  (bridge_length : ℝ := total_distance - train_length)
  (train_length_val : train_length = 110)
  (train_speed_km_hr_val : train_speed_km_hr = 36)
  (crossing_time_sec_val : crossing_time_sec = 24.198064154867613) :
  bridge_length = 131.98064154867613 :=
by
  sorry

end NUMINAMATH_GPT_bridge_length_l641_64189


namespace NUMINAMATH_GPT_system_solution_l641_64100

theorem system_solution :
  ∀ (a1 b1 c1 a2 b2 c2 : ℝ),
  (a1 * 8 + b1 * 5 = c1) ∧ (a2 * 8 + b2 * 5 = c2) →
  ∃ (x y : ℝ), (4 * a1 * x - 5 * b1 * y = 3 * c1) ∧ (4 * a2 * x - 5 * b2 * y = 3 * c2) ∧ 
               (x = 6) ∧ (y = -3) :=
by
  sorry

end NUMINAMATH_GPT_system_solution_l641_64100


namespace NUMINAMATH_GPT_find_f_correct_l641_64197

noncomputable def f (x : ℝ) : ℝ := sorry

axiom f_con1 : ∀ x : ℝ, 2 * f x + f (-x) = 2 * x

theorem find_f_correct : ∀ x : ℝ, f x = 2 * x :=
by
  sorry

end NUMINAMATH_GPT_find_f_correct_l641_64197


namespace NUMINAMATH_GPT_ellipse_major_axis_length_l641_64117

-- Conditions
def cylinder_radius : ℝ := 2
def minor_axis (r : ℝ) := 2 * r
def major_axis (minor: ℝ) := minor + 0.6 * minor

-- Problem
theorem ellipse_major_axis_length :
  major_axis (minor_axis cylinder_radius) = 6.4 :=
by
  sorry

end NUMINAMATH_GPT_ellipse_major_axis_length_l641_64117


namespace NUMINAMATH_GPT_evaluate_expression_l641_64137

def operation_star (A B : ℕ) : ℕ := (A + B) / 2
def operation_ominus (A B : ℕ) : ℕ := A - B

theorem evaluate_expression :
  operation_ominus (operation_star 6 10) (operation_star 2 4) = 5 := 
by 
  sorry

end NUMINAMATH_GPT_evaluate_expression_l641_64137


namespace NUMINAMATH_GPT_classify_quadrilateral_l641_64170

structure Quadrilateral where
  sides : ℕ → ℝ 
  angle : ℕ → ℝ 
  diag_length : ℕ → ℝ 
  perpendicular_diagonals : Prop

def is_rhombus (q : Quadrilateral) : Prop :=
  (∀ i, q.sides i = q.sides 0) ∧ q.perpendicular_diagonals

def is_kite (q : Quadrilateral) : Prop :=
  (q.sides 1 = q.sides 2 ∧ q.sides 3 = q.sides 4) ∧ q.perpendicular_diagonals

def is_square (q : Quadrilateral) : Prop :=
  (∀ i, q.sides i = q.sides 0) ∧ (∀ i, q.angle i = 90) ∧ q.perpendicular_diagonals

theorem classify_quadrilateral (q : Quadrilateral) (h : q.perpendicular_diagonals) :
  is_rhombus q ∨ is_kite q ∨ is_square q :=
sorry

end NUMINAMATH_GPT_classify_quadrilateral_l641_64170


namespace NUMINAMATH_GPT_count_ordered_pairs_squares_diff_l641_64149

theorem count_ordered_pairs_squares_diff (m n : ℕ) (h1 : m ≥ n) (h2 : m^2 - n^2 = 72) : 
∃ (a : ℕ), a = 3 :=
sorry

end NUMINAMATH_GPT_count_ordered_pairs_squares_diff_l641_64149


namespace NUMINAMATH_GPT_percentage_increase_first_year_l641_64179

theorem percentage_increase_first_year (P : ℝ) (X : ℝ) 
  (h1 : P * (1 + X / 100) * 0.75 * 1.15 = P * 1.035) : 
  X = 20 :=
by
  sorry

end NUMINAMATH_GPT_percentage_increase_first_year_l641_64179


namespace NUMINAMATH_GPT_eval_power_imaginary_unit_l641_64185

noncomputable def i : ℂ := Complex.I

theorem eval_power_imaginary_unit :
  i^20 + i^39 = 1 - i := by
  -- Skipping the proof itself, indicating it with "sorry"
  sorry

end NUMINAMATH_GPT_eval_power_imaginary_unit_l641_64185


namespace NUMINAMATH_GPT_range_of_a_l641_64119

theorem range_of_a (a : ℝ) :
  (∃ x y : ℝ, x^2 + 4 * (y - a)^2 = 4 ∧ x^2 = 2 * y) ↔ -1 ≤ a ∧ a ≤ 17 / 8 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l641_64119


namespace NUMINAMATH_GPT_general_term_of_series_l641_64163

def gen_term (a : ℕ → ℕ) : Prop :=
∀ n : ℕ, a n = if n = 1 then 2 else 6 * n - 5

def series_sum (S : ℕ → ℕ) : Prop :=
∀ n : ℕ, S n = 3 * n ^ 2 - 2 * n + 1

theorem general_term_of_series (a S : ℕ → ℕ) (h : series_sum S) :
  gen_term a ↔ (∀ n : ℕ, a n = if n = 1 then 2 else S n - S (n - 1)) :=
by sorry

end NUMINAMATH_GPT_general_term_of_series_l641_64163


namespace NUMINAMATH_GPT_angle_bisector_equation_intersection_l641_64129

noncomputable def slope_of_angle_bisector (m1 m2 : ℝ) : ℝ :=
  (m1 + m2 - Real.sqrt (1 + m1^2 + m2^2)) / (1 - m1 * m2)

noncomputable def equation_of_angle_bisector (x : ℝ) : ℝ :=
  (Real.sqrt 21 - 6) / 7 * x

theorem angle_bisector_equation_intersection :
  let m1 := 2
  let m2 := 4
  slope_of_angle_bisector m1 m2 = (Real.sqrt 21 - 6) / 7 ∧
  equation_of_angle_bisector 1 = (Real.sqrt 21 - 6) / 7 :=
by
  sorry

end NUMINAMATH_GPT_angle_bisector_equation_intersection_l641_64129


namespace NUMINAMATH_GPT_question1_question2_l641_64162

def f (x : ℝ) : ℝ := abs (x - 5) - abs (x - 2)

theorem question1 :
  (∃ x : ℝ, f x ≤ m) ↔ m ≥ -3 :=
sorry

theorem question2 :
  { x : ℝ | x^2 - 8*x + 15 + f x ≤ 0 } = { x | 5 - Real.sqrt 3 ≤ x ∧ x ≤ 6 } :=
sorry

end NUMINAMATH_GPT_question1_question2_l641_64162


namespace NUMINAMATH_GPT_prove_condition_for_equality_l641_64118

noncomputable def condition_for_equality (a b c : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : Prop :=
  c = (b * (a ^ 3 - 1)) / a

theorem prove_condition_for_equality (a b c : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (∃ (c' : ℕ), (c' = (b * (a ^ 3 - 1)) / a) ∧ 
      c' > 0 ∧ 
      (a + b / c' = a ^ 3 * (b / c')) ) → 
  c = (b * (a ^ 3 - 1)) / a := 
sorry

end NUMINAMATH_GPT_prove_condition_for_equality_l641_64118


namespace NUMINAMATH_GPT_hilary_total_kernels_l641_64124

-- Define the conditions given in the problem
def ears_per_stalk : ℕ := 4
def total_stalks : ℕ := 108
def kernels_per_ear_first_half : ℕ := 500
def additional_kernels_second_half : ℕ := 100

-- Express the main problem as a theorem in Lean
theorem hilary_total_kernels : 
  let total_ears := ears_per_stalk * total_stalks
  let half_ears := total_ears / 2
  let kernels_first_half := half_ears * kernels_per_ear_first_half
  let kernels_per_ear_second_half := kernels_per_ear_first_half + additional_kernels_second_half
  let kernels_second_half := half_ears * kernels_per_ear_second_half
  kernels_first_half + kernels_second_half = 237600 :=
by
  sorry

end NUMINAMATH_GPT_hilary_total_kernels_l641_64124


namespace NUMINAMATH_GPT_simplify_power_of_product_l641_64169

theorem simplify_power_of_product (x y : ℝ) : (3 * x^2 * y^3)^2 = 9 * x^4 * y^6 :=
by
  -- hint: begin proof here
  sorry

end NUMINAMATH_GPT_simplify_power_of_product_l641_64169


namespace NUMINAMATH_GPT_find_height_of_cuboid_l641_64101

-- Definitions and given conditions
def length : ℕ := 22
def width : ℕ := 30
def total_edges : ℕ := 224

-- Proof statement
theorem find_height_of_cuboid (h : ℕ) (H : 4 * length + 4 * width + 4 * h = total_edges) : h = 4 :=
by
  sorry

end NUMINAMATH_GPT_find_height_of_cuboid_l641_64101


namespace NUMINAMATH_GPT_ellipse_m_range_l641_64102

theorem ellipse_m_range (m : ℝ) 
  (h1 : m + 9 > 25 - m) 
  (h2 : 25 - m > 0) 
  (h3 : m + 9 > 0) : 
  8 < m ∧ m < 25 := 
by
  sorry

end NUMINAMATH_GPT_ellipse_m_range_l641_64102


namespace NUMINAMATH_GPT_Emily_GRE_Exam_Date_l641_64123

theorem Emily_GRE_Exam_Date : 
  ∃ (exam_date : ℕ) (exam_month : String), 
  exam_date = 5 ∧ exam_month = "September" ∧
  ∀ study_days break_days start_day_cycles start_break_cycles start_month_june total_days S_june_remaining S_remaining_july S_remaining_august September_start_day, 
    study_days = 15 ∧ 
    break_days = 5 ∧ 
    start_day_cycles = 5 ∧ 
    start_break_cycles = 4 ∧ 
    start_month_june = 1 ∧
    total_days = start_day_cycles * study_days + start_break_cycles * break_days ∧ 
    S_june_remaining = 30 - start_month_june ∧ 
    S_remaining = total_days - S_june_remaining ∧ 
    S_remaining_july = S_remaining - 31 ∧ 
    S_remaining_august = S_remaining_july - 31 ∧ 
    September_start_day = S_remaining_august + 1 ∧
    exam_date = September_start_day ∧ 
    exam_month = "September" := by 
  sorry

end NUMINAMATH_GPT_Emily_GRE_Exam_Date_l641_64123


namespace NUMINAMATH_GPT_prove_inequality_l641_64113

variable (f : ℝ → ℝ)

def isEvenFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

def isMonotonicOnInterval (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x ≤ y ∧ y ≤ b → f x ≥ f y

theorem prove_inequality
  (h1 : isEvenFunction f)
  (h2 : isMonotonicOnInterval f 0 5)
  (h3 : f (-3) < f 1) :
  f 0 > f 1 :=
sorry

end NUMINAMATH_GPT_prove_inequality_l641_64113


namespace NUMINAMATH_GPT_cost_per_serving_l641_64168

def pasta_cost : ℝ := 1.0
def sauce_cost : ℝ := 2.0
def meatballs_cost : ℝ := 5.0
def total_servings : ℝ := 8.0

theorem cost_per_serving : (pasta_cost + sauce_cost + meatballs_cost) / total_servings = 1.0 :=
by sorry

end NUMINAMATH_GPT_cost_per_serving_l641_64168


namespace NUMINAMATH_GPT_part_1_odd_function_part_2_decreasing_l641_64135

noncomputable def f (x : ℝ) : ℝ := (1 - 2^x) / (1 + 2^x)

theorem part_1_odd_function : ∀ x : ℝ, f (-x) = -f x := by
  intro x
  sorry

theorem part_2_decreasing : ∀ x1 x2 : ℝ, x1 < x2 → f x1 > f x2 := by
  intros x1 x2 h
  sorry

end NUMINAMATH_GPT_part_1_odd_function_part_2_decreasing_l641_64135


namespace NUMINAMATH_GPT_solve_fraction_equation_l641_64159

theorem solve_fraction_equation (x : ℝ) (h : (x + 6) / (x - 3) = 4) : x = 6 :=
sorry

end NUMINAMATH_GPT_solve_fraction_equation_l641_64159


namespace NUMINAMATH_GPT_negation_of_P_l641_64188

-- Define the proposition P
def P : Prop := ∀ x : ℝ, x^2 + 2*x + 2 > 0

-- State the negation of P
theorem negation_of_P : ¬P ↔ ∃ x : ℝ, x^2 + 2*x + 2 ≤ 0 :=
by
  sorry

end NUMINAMATH_GPT_negation_of_P_l641_64188


namespace NUMINAMATH_GPT_cuboid_volume_l641_64133

/-- Define the ratio condition for the dimensions of the cuboid. -/
def ratio (l w h : ℕ) : Prop :=
  (∃ x : ℕ, l = 2*x ∧ w = x ∧ h = 3*x)

/-- Define the total surface area condition for the cuboid. -/
def surface_area (l w h sa : ℕ) : Prop :=
  2*(l*w + l*h + w*h) = sa

/-- Volume of the cuboid given the ratio and surface area conditions. -/
theorem cuboid_volume (l w h : ℕ) (sa : ℕ) (h_ratio : ratio l w h) (h_surface : surface_area l w h sa) :
  ∃ v : ℕ, v = l * w * h ∧ v = 48 :=
by
  sorry

end NUMINAMATH_GPT_cuboid_volume_l641_64133


namespace NUMINAMATH_GPT_find_large_number_l641_64156

theorem find_large_number (L S : ℕ) 
  (h1 : L - S = 1335) 
  (h2 : L = 6 * S + 15) : 
  L = 1599 := 
by 
  -- proof omitted
  sorry

end NUMINAMATH_GPT_find_large_number_l641_64156


namespace NUMINAMATH_GPT_max_value_of_f_l641_64126

def f (x : ℝ) : ℝ := x^2 - 2 * x - 5

theorem max_value_of_f : ∃ x ∈ (Set.Icc (-2:ℝ) 2), ∀ y ∈ (Set.Icc (-2:ℝ) 2), f y ≤ f x ∧ f x = 3 := by
  sorry

end NUMINAMATH_GPT_max_value_of_f_l641_64126


namespace NUMINAMATH_GPT_ratio_lcm_gcf_eq_55_l641_64127

theorem ratio_lcm_gcf_eq_55 : 
  ∀ (a b : ℕ), a = 210 → b = 462 →
  (Nat.lcm a b / Nat.gcd a b) = 55 :=
by
  intros a b ha hb
  rw [ha, hb]
  sorry

end NUMINAMATH_GPT_ratio_lcm_gcf_eq_55_l641_64127


namespace NUMINAMATH_GPT_unique_positive_integer_n_l641_64190

-- Definitions based on conditions
def is_divisor (n a : ℕ) : Prop := a % n = 0

def is_perfect_square (m : ℕ) : Prop := ∃ k : ℕ, k * k = m

-- The main theorem statement
theorem unique_positive_integer_n : ∃ (n : ℕ), n > 0 ∧ is_divisor n 1989 ∧
    is_perfect_square (n^2 - 1989 / n) ∧ n = 13 :=
by
  sorry

end NUMINAMATH_GPT_unique_positive_integer_n_l641_64190


namespace NUMINAMATH_GPT_prime_divisor_problem_l641_64184

theorem prime_divisor_problem (d r : ℕ) (h1 : d > 1) (h2 : Prime d)
  (h3 : 1274 % d = r) (h4 : 1841 % d = r) (h5 : 2866 % d = r) : d - r = 6 :=
by
  sorry

end NUMINAMATH_GPT_prime_divisor_problem_l641_64184


namespace NUMINAMATH_GPT_total_points_first_half_l641_64173

def geometric_sum (a r : ℕ) (n : ℕ) : ℕ :=
  a * (1 - r ^ n) / (1 - r)

def arithmetic_sum (a d : ℕ) (n : ℕ) : ℕ :=
  n * a + d * (n * (n - 1) / 2)

-- Given conditions:
variables (a r b d : ℕ)
variables (h1 : a = b)
variables (h2 : geometric_sum a r 4 = b + (b + d) + (b + 2 * d) + (b + 3 * d) + 2)
variables (h3 : a * (1 + r + r^2 + r^3) ≤ 120)
variables (h4 : b + (b + d) + (b + 2 * d) + (b + 3 * d) ≤ 120)

theorem total_points_first_half (a r b d : ℕ) (h1 : a = b) (h2 : a * (1 + r + r ^ 2 + r ^ 3) = b + (b + d) + (b + 2 * d) + (b + 3 * d) + 2)
  (h3 : a * (1 + r + r ^ 2 + r ^ 3) ≤ 120) (h4 : b + (b + d) + (b + 2 * d) + (b + 3 * d) ≤ 120) : 
  a + a * r + b + (b + d) = 45 :=
by
  sorry

end NUMINAMATH_GPT_total_points_first_half_l641_64173


namespace NUMINAMATH_GPT_truck_distance_on_7_liters_l641_64176

-- Define the conditions
def truck_300_km_per_5_liters := 300
def liters_5 := 5
def liters_7 := 7
def expected_distance_7_liters := 420

-- The rate of distance (km per liter)
def rate := truck_300_km_per_5_liters / liters_5

-- Proof statement
theorem truck_distance_on_7_liters :
  rate * liters_7 = expected_distance_7_liters :=
  by
  sorry

end NUMINAMATH_GPT_truck_distance_on_7_liters_l641_64176


namespace NUMINAMATH_GPT_problem1_problem2_l641_64115

-- Problem 1
theorem problem1 :
  (1 : ℝ) * (2 * Real.sqrt 12 - (1 / 2) * Real.sqrt 18) - (Real.sqrt 75 - (1 / 4) * Real.sqrt 32)
  = -Real.sqrt 3 - (Real.sqrt 2) / 2 :=
by
  sorry

-- Problem 2
theorem problem2 :
  (2 : ℝ) * (Real.sqrt 5 + 2) * (Real.sqrt 5 - 2) + Real.sqrt 48 / (2 * Real.sqrt (1 / 2)) - Real.sqrt 30 / Real.sqrt 5
  = 1 + Real.sqrt 6 :=
by
  sorry

end NUMINAMATH_GPT_problem1_problem2_l641_64115


namespace NUMINAMATH_GPT_max_rectangle_area_l641_64174

theorem max_rectangle_area (a b : ℝ) (h : 2 * a + 2 * b = 60) :
  a * b ≤ 225 :=
by
  sorry

end NUMINAMATH_GPT_max_rectangle_area_l641_64174


namespace NUMINAMATH_GPT_unique_nonzero_solution_l641_64199

theorem unique_nonzero_solution (x : ℝ) (h : x ≠ 0) : (3 * x)^3 = (9 * x)^2 → x = 3 :=
by
  sorry

end NUMINAMATH_GPT_unique_nonzero_solution_l641_64199


namespace NUMINAMATH_GPT_problem1_problem2_l641_64121

open Nat

def seq (a : ℕ → ℕ) :=
  ∀ n : ℕ, n > 0 → a n < a (n + 1) ∧ a n > 0

def b_seq (a : ℕ → ℕ) (n : ℕ) :=
  a (a n)

def c_seq (a : ℕ → ℕ) (n : ℕ) :=
  a (a n + 1)

theorem problem1 (a : ℕ → ℕ) (h_seq : seq a) (h_bseq : ∀ n, n > 0 → b_seq a n = 3 * n) : a 1 = 2 ∧ c_seq a 1 = 6 :=
  sorry

theorem problem2 (a : ℕ → ℕ) (h_seq : seq a) (h_cseq : ∀ n, n > 0 → c_seq a (n + 1) - c_seq a n = 1) : 
  ∀ n, n > 0 → a (n + 1) - a n = 1 :=
  sorry

end NUMINAMATH_GPT_problem1_problem2_l641_64121


namespace NUMINAMATH_GPT_integral_abs_sin_from_0_to_2pi_l641_64181

theorem integral_abs_sin_from_0_to_2pi : ∫ x in (0 : ℝ)..(2 * Real.pi), |Real.sin x| = 4 := 
by
  sorry

end NUMINAMATH_GPT_integral_abs_sin_from_0_to_2pi_l641_64181


namespace NUMINAMATH_GPT_men_absent_l641_64151

theorem men_absent (original_men absent_men remaining_men : ℕ) (total_work : ℕ) 
  (h1 : original_men = 15) (h2 : total_work = original_men * 40) (h3 : 60 * remaining_men = total_work) : 
  remaining_men = original_men - absent_men → absent_men = 5 := 
by
  sorry

end NUMINAMATH_GPT_men_absent_l641_64151


namespace NUMINAMATH_GPT_profit_calculation_l641_64139

theorem profit_calculation
  (P : ℝ)
  (h1 : 9 > 0)  -- condition that there are 9 employees
  (h2 : 0 < 0.10 ∧ 0.10 < 1) -- 10 percent profit is between 0 and 100%
  (h3 : 5 > 0)  -- condition that each employee gets $5
  (h4 : 9 * 5 = 45) -- total amount distributed among employees
  (h5 : 0.90 * P = 45) -- remaining profit to be distributed
  : P = 50 :=
sorry

end NUMINAMATH_GPT_profit_calculation_l641_64139


namespace NUMINAMATH_GPT_power_division_calculation_l641_64147

theorem power_division_calculation :
  ( ( 5^13 / 5^11 )^2 * 5^2 ) / 2^5 = 15625 / 32 :=
by
  sorry

end NUMINAMATH_GPT_power_division_calculation_l641_64147


namespace NUMINAMATH_GPT_ball_hits_ground_time_l641_64106

theorem ball_hits_ground_time :
  ∃ t : ℝ, -16 * t^2 - 30 * t + 180 = 0 ∧ t = 1.25 := by
  sorry

end NUMINAMATH_GPT_ball_hits_ground_time_l641_64106


namespace NUMINAMATH_GPT_solve_system_eqn_l641_64110

theorem solve_system_eqn :
  ∃ x y : ℚ, 7 * x = -9 - 3 * y ∧ 2 * x = 5 * y - 30 ∧ x = -135 / 41 ∧ y = 192 / 41 :=
by 
  sorry

end NUMINAMATH_GPT_solve_system_eqn_l641_64110


namespace NUMINAMATH_GPT_elena_earnings_l641_64111

theorem elena_earnings (hourly_wage : ℝ) (hours_worked : ℝ) (h_wage : hourly_wage = 13.25) (h_hours : hours_worked = 4) : 
  hourly_wage * hours_worked = 53.00 := by
sorry

end NUMINAMATH_GPT_elena_earnings_l641_64111


namespace NUMINAMATH_GPT_calculate_expression_l641_64108

theorem calculate_expression :
  18 - ((-16) / (2 ^ 3)) = 20 :=
by
  sorry

end NUMINAMATH_GPT_calculate_expression_l641_64108


namespace NUMINAMATH_GPT_find_coords_of_P_l641_64166

-- Definitions from the conditions
def line_eq (x y : ℝ) : Prop := x - y - 7 = 0
def is_midpoint (P Q M : ℝ × ℝ) : Prop := 
  M = ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)

-- Coordinates given in the problem
def P : ℝ × ℝ := (-2, 1)

-- The proof goal
theorem find_coords_of_P : ∃ Q : ℝ × ℝ,
  is_midpoint P Q (1, -1) ∧ 
  line_eq Q.1 Q.2 :=
sorry

end NUMINAMATH_GPT_find_coords_of_P_l641_64166


namespace NUMINAMATH_GPT_claudia_candle_choices_l641_64144

-- Claudia can choose 4 different candles
def num_candles : ℕ := 4

-- Claudia can choose 8 out of 9 different flowers
def num_ways_to_choose_flowers : ℕ := Nat.choose 9 8

-- The total number of groupings is given as 54
def total_groupings : ℕ := 54

-- Prove the main theorem using the conditions
theorem claudia_candle_choices :
  num_ways_to_choose_flowers = 9 ∧ num_ways_to_choose_flowers * C = total_groupings → C = 6 :=
by
  sorry

end NUMINAMATH_GPT_claudia_candle_choices_l641_64144


namespace NUMINAMATH_GPT_quotient_calc_l641_64180

theorem quotient_calc (dividend : ℕ) (divisor : ℕ) (remainder : ℕ) (quotient : ℕ)
  (h_dividend : dividend = 139)
  (h_divisor : divisor = 19)
  (h_remainder : remainder = 6)
  (h_formula : dividend - remainder = quotient * divisor):
  quotient = 7 :=
by {
  -- Insert proof here
  sorry
}

end NUMINAMATH_GPT_quotient_calc_l641_64180


namespace NUMINAMATH_GPT_square_park_area_l641_64161

theorem square_park_area (side_length : ℝ) (h : side_length = 200) : side_length * side_length = 40000 := by
  sorry

end NUMINAMATH_GPT_square_park_area_l641_64161


namespace NUMINAMATH_GPT_rectangle_area_l641_64192

theorem rectangle_area (L B : ℕ) 
  (h1 : L - B = 23)
  (h2 : 2 * L + 2 * B = 186) :
  L * B = 2030 := by
  sorry

end NUMINAMATH_GPT_rectangle_area_l641_64192


namespace NUMINAMATH_GPT_smallest_b_for_factorable_polynomial_l641_64171

theorem smallest_b_for_factorable_polynomial :
  ∃ (b : ℕ), b > 0 ∧ (∃ (p q : ℤ), x^2 + b * x + 1176 = (x + p) * (x + q) ∧ p * q = 1176 ∧ p + q = b) ∧ 
  (∀ (b' : ℕ), b' > 0 → (∃ (p' q' : ℤ), x^2 + b' * x + 1176 = (x + p') * (x + q') ∧ p' * q' = 1176 ∧ p' + q' = b') → b ≤ b') :=
sorry

end NUMINAMATH_GPT_smallest_b_for_factorable_polynomial_l641_64171


namespace NUMINAMATH_GPT_largest_of_seven_consecutive_numbers_l641_64141

theorem largest_of_seven_consecutive_numbers (a b c d e f g : ℤ) (h1 : a + 1 = b)
                                             (h2 : b + 1 = c) (h3 : c + 1 = d)
                                             (h4 : d + 1 = e) (h5 : e + 1 = f)
                                             (h6 : f + 1 = g)
                                             (h_avg : (a + b + c + d + e + f + g) / 7 = 20) :
    g = 23 :=
by
  sorry

end NUMINAMATH_GPT_largest_of_seven_consecutive_numbers_l641_64141


namespace NUMINAMATH_GPT_least_possible_faces_combined_l641_64155

noncomputable def hasValidDiceConfiguration : Prop :=
  ∃ a b : ℕ,
  (∃ s8 s12 s13 : ℕ,
    (s8 = 3) ∧
    (s12 = 4) ∧
    (a ≥ 5 ∧ b = 6 ∧ (a + b = 11) ∧
      (2 * s12 = s8) ∧
      (2 * s8 = s13))
  )

theorem least_possible_faces_combined : hasValidDiceConfiguration :=
  sorry

end NUMINAMATH_GPT_least_possible_faces_combined_l641_64155


namespace NUMINAMATH_GPT_plane_contains_points_l641_64114

def point := (ℝ × ℝ × ℝ)

def is_plane (A B C D : ℝ) (p : point) : Prop :=
  ∃ x y z, p = (x, y, z) ∧ A * x + B * y + C * z + D = 0

theorem plane_contains_points :
  ∃ A B C D : ℤ,
    A > 0 ∧
    Int.gcd (Int.natAbs A) (Int.gcd (Int.natAbs B) (Int.gcd (Int.natAbs C) (Int.natAbs D))) = 1 ∧
    is_plane A B C D (2, -1, 3) ∧
    is_plane A B C D (0, -1, 5) ∧
    is_plane A B C D (-2, -3, 4) ∧
    A = 2 ∧ B = 5 ∧ C = -2 ∧ D = 7 :=
  sorry

end NUMINAMATH_GPT_plane_contains_points_l641_64114


namespace NUMINAMATH_GPT_power_of_two_as_sum_of_squares_l641_64160

theorem power_of_two_as_sum_of_squares (n : ℕ) (h : n ≥ 3) :
  ∃ (x y : ℤ), x % 2 = 1 ∧ y % 2 = 1 ∧ (2^n = 7*x^2 + y^2) :=
by
  sorry

end NUMINAMATH_GPT_power_of_two_as_sum_of_squares_l641_64160


namespace NUMINAMATH_GPT_correct_propositions_for_curve_C_l641_64196

def curve_C (k : ℝ) : Prop :=
  ∀ x y : ℝ, (x^2 / (4 - k) + y^2 / (k - 1) = 1)

theorem correct_propositions_for_curve_C (k : ℝ) :
  (∀ x y : ℝ, curve_C k) →
  ((∃ k, ((4 - k) * (k - 1) < 0) ↔ (k < 1 ∨ k > 4)) ∧
  ((1 < k ∧ k < (5 : ℝ) / 2) ↔
  (4 - k > k - 1 ∧ 4 - k > 0 ∧ k - 1 > 0))) :=
by {
  sorry
}

end NUMINAMATH_GPT_correct_propositions_for_curve_C_l641_64196


namespace NUMINAMATH_GPT_max_m_value_l641_64153

theorem max_m_value (a : ℚ) (m : ℚ) : (∀ x : ℤ, 0 < x ∧ x ≤ 50 → ¬ ∃ y : ℤ, y = m * x + 3) ∧ (1 / 2 < m) ∧ (m < a) → a = 26 / 51 :=
by sorry

end NUMINAMATH_GPT_max_m_value_l641_64153


namespace NUMINAMATH_GPT_problem_part_1_problem_part_2_problem_part_3_l641_64187

open Set

universe u

def A : Set ℝ := {x | 2 ≤ x ∧ x ≤ 8}
def B : Set ℝ := {x | 1 < x ∧ x < 6}
def C (a : ℝ) : Set ℝ := {x | x > a}
def U : Set ℝ := univ

theorem problem_part_1 : A ∪ B = {x | 1 < x ∧ x ≤ 8} :=
sorry

theorem problem_part_2 : (U \ A) ∩ B = {x | 1 < x ∧ x < 2} :=
sorry

theorem problem_part_3 (a : ℝ) (h : (A ∩ C a) ≠ ∅) : a < 8 :=
sorry

end NUMINAMATH_GPT_problem_part_1_problem_part_2_problem_part_3_l641_64187


namespace NUMINAMATH_GPT_inclination_angle_x_eq_one_l641_64145

noncomputable def inclination_angle_of_vertical_line (x : ℝ) : ℝ :=
if x = 1 then 90 else 0

theorem inclination_angle_x_eq_one :
  inclination_angle_of_vertical_line 1 = 90 :=
by
  sorry

end NUMINAMATH_GPT_inclination_angle_x_eq_one_l641_64145


namespace NUMINAMATH_GPT_percentage_shaded_is_18_75_l641_64198

-- conditions
def total_squares: ℕ := 16
def shaded_squares: ℕ := 3

-- claim to prove
theorem percentage_shaded_is_18_75 :
  ((shaded_squares : ℝ) / total_squares) * 100 = 18.75 := 
by
  sorry

end NUMINAMATH_GPT_percentage_shaded_is_18_75_l641_64198


namespace NUMINAMATH_GPT_pond_fish_count_l641_64191

theorem pond_fish_count :
  (∃ (N : ℕ), (2 / 50 : ℚ) = (40 / N : ℚ)) → N = 1000 :=
by
  sorry

end NUMINAMATH_GPT_pond_fish_count_l641_64191


namespace NUMINAMATH_GPT_line_through_P_midpoint_l641_64193

noncomputable section

open Classical

variables (l l1 l2 : ℝ → ℝ → Prop) (P A B : ℝ × ℝ)

def line1 (x y : ℝ) := 2 * x - y - 2 = 0
def line2 (x y : ℝ) := x + y + 3 = 0

theorem line_through_P_midpoint (P A B : ℝ × ℝ)
  (hP : P = (3, 0))
  (hl1 : ∀ x y, line1 x y → l x y)
  (hl2 : ∀ x y, line2 x y → l x y)
  (hmid : (P.1 = (A.1 + B.1) / 2) ∧ (P.2 = (A.2 + B.2) / 2)) :
  ∃ k : ℝ, ∀ x y, (y = k * (x - 3)) ↔ (8 * x - y - 24 = 0) :=
by
  sorry

end NUMINAMATH_GPT_line_through_P_midpoint_l641_64193


namespace NUMINAMATH_GPT_no_integer_solutions_l641_64105

theorem no_integer_solutions (x y z : ℤ) : x^3 + y^6 ≠ 7 * z + 3 :=
by sorry

end NUMINAMATH_GPT_no_integer_solutions_l641_64105


namespace NUMINAMATH_GPT_line_tangent_to_ellipse_l641_64132

theorem line_tangent_to_ellipse (m : ℝ) (a : ℝ) (b : ℝ) (h_a : a = 3) (h_b : b = 1) :
  m^2 = 1 / 3 := by
  sorry

end NUMINAMATH_GPT_line_tangent_to_ellipse_l641_64132


namespace NUMINAMATH_GPT_clock_spoke_angle_l641_64150

-- Define the parameters of the clock face and the problem.
def num_spokes := 10
def total_degrees := 360
def degrees_per_spoke := total_degrees / num_spokes
def position_3_oclock := 3 -- the third spoke
def halfway_45_oclock := 5 -- approximately the fifth spoke
def spokes_between := halfway_45_oclock - position_3_oclock
def smaller_angle := spokes_between * degrees_per_spoke
def expected_angle := 72

-- Statement of the problem
theorem clock_spoke_angle :
  smaller_angle = expected_angle := by
    -- Proof is omitted
    sorry

end NUMINAMATH_GPT_clock_spoke_angle_l641_64150


namespace NUMINAMATH_GPT_booth_makes_50_per_day_on_popcorn_l641_64183

-- Define the conditions as provided
def daily_popcorn_revenue (P : ℝ) : Prop :=
  let cotton_candy_revenue := 3 * P
  let total_days := 5
  let rent := 30
  let ingredients := 75
  let total_expenses := rent + ingredients
  let profit := 895
  let total_revenue_before_expenses := profit + total_expenses
  total_revenue_before_expenses = 20 * P 

theorem booth_makes_50_per_day_on_popcorn : daily_popcorn_revenue 50 :=
  by sorry

end NUMINAMATH_GPT_booth_makes_50_per_day_on_popcorn_l641_64183


namespace NUMINAMATH_GPT_intersection_of_A_and_CU_B_l641_64128

open Set Real

noncomputable def U : Set ℝ := univ
noncomputable def A : Set ℝ := {-1, 0, 1, 2, 3}
noncomputable def B : Set ℝ := { x : ℝ | x ≥ 2 }
noncomputable def CU_B : Set ℝ := { x : ℝ | x < 2 }

theorem intersection_of_A_and_CU_B :
  A ∩ CU_B = {-1, 0, 1} :=
by
  sorry

end NUMINAMATH_GPT_intersection_of_A_and_CU_B_l641_64128


namespace NUMINAMATH_GPT_log_relationship_l641_64107

theorem log_relationship :
  let a := Real.logb 0.3 0.2
  let b := Real.logb 3 2
  let c := Real.logb 0.2 3
  c < b ∧ b < a :=
by
  let a := Real.logb 0.3 0.2
  let b := Real.logb 3 2
  let c := Real.logb 0.2 3
  sorry

end NUMINAMATH_GPT_log_relationship_l641_64107


namespace NUMINAMATH_GPT_sum_of_obtuse_angles_l641_64178

theorem sum_of_obtuse_angles (A B : ℝ) (hA1 : A > π / 2) (hA2 : A < π)
  (hB1 : B > π / 2) (hB2 : B < π)
  (hSinA : Real.sin A = Real.sqrt 5 / 5)
  (hSinB : Real.sin B = Real.sqrt 10 / 10) :
  A + B = 7 * π / 4 := 
sorry

end NUMINAMATH_GPT_sum_of_obtuse_angles_l641_64178


namespace NUMINAMATH_GPT_complex_number_equality_l641_64164

def is_imaginary_unit (i : ℂ) : Prop :=
  i^2 = -1

theorem complex_number_equality (a b : ℝ) (i : ℂ) (h1 : is_imaginary_unit i) (h2 : (a + 4 * i) * i = b + i) : a + b = -3 :=
sorry

end NUMINAMATH_GPT_complex_number_equality_l641_64164


namespace NUMINAMATH_GPT_problem_condition_l641_64130

theorem problem_condition (x y : ℝ) (h : x^2 + y^2 - x * y = 1) : 
  x + y ≥ -2 ∧ x^2 + y^2 ≤ 2 :=
by
  sorry

end NUMINAMATH_GPT_problem_condition_l641_64130


namespace NUMINAMATH_GPT_angles_terminal_side_equiv_l641_64177

theorem angles_terminal_side_equiv (k : ℤ) : (2 * k * Real.pi + Real.pi) % (2 * Real.pi) = (4 * k * Real.pi + Real.pi) % (2 * Real.pi) ∨ (2 * k * Real.pi + Real.pi) % (2 * Real.pi) = (4 * k * Real.pi - Real.pi) % (2 * Real.pi) :=
sorry

end NUMINAMATH_GPT_angles_terminal_side_equiv_l641_64177


namespace NUMINAMATH_GPT_area_on_larger_sphere_l641_64131

-- Define the radii of the spheres
def r_small : ℝ := 1
def r_in : ℝ := 4
def r_out : ℝ := 6

-- Given the area on the smaller sphere
def A_small_sphere_area : ℝ := 37

-- Statement: Find the area on the larger sphere
theorem area_on_larger_sphere :
  (A_small_sphere_area * (r_out / r_in) ^ 2 = 83.25) := by
  sorry

end NUMINAMATH_GPT_area_on_larger_sphere_l641_64131


namespace NUMINAMATH_GPT_number_of_boys_is_10_l641_64182

-- Definitions based on given conditions
def num_children := 20
def has_blue_neighbor_clockwise (i : ℕ) : Prop := true -- Dummy predicate representing condition
def has_red_neighbor_counterclockwise (i : ℕ) : Prop := true -- Dummy predicate representing condition

axiom boys_and_girls_exist : ∃ b g : ℤ, b + g = num_children ∧ b > 0 ∧ g > 0

-- Theorem based on the problem statement
theorem number_of_boys_is_10 (b g : ℤ) 
  (total_children: b + g = num_children)
  (boys_exist: b > 0)
  (girls_exist: g > 0)
  (each_boy_has_blue_neighbor: ∀ i, has_blue_neighbor_clockwise i → true)
  (each_girl_has_red_neighbor: ∀ i, has_red_neighbor_counterclockwise i → true): 
  b = 10 :=
by
  sorry

end NUMINAMATH_GPT_number_of_boys_is_10_l641_64182


namespace NUMINAMATH_GPT_factor_expression_l641_64167

theorem factor_expression (x : ℝ) :
  (16 * x^4 + 36 * x^2 - 9) - (4 * x^4 - 6 * x^2 - 9) = 6 * x^2 * (2 * x^2 + 7) :=
by
  sorry

end NUMINAMATH_GPT_factor_expression_l641_64167


namespace NUMINAMATH_GPT_stock_rise_in_morning_l641_64134

theorem stock_rise_in_morning (x : ℕ) (V : ℕ → ℕ) (h0 : V 0 = 100)
  (h100 : V 100 = 200) (h_recurrence : ∀ n, V n = 100 + n * x - n) :
  x = 2 :=
  by
  sorry

end NUMINAMATH_GPT_stock_rise_in_morning_l641_64134


namespace NUMINAMATH_GPT_christine_aquafaba_needed_l641_64175

-- Define the number of tablespoons per egg white
def tablespoons_per_egg_white : ℕ := 2

-- Define the number of egg whites per cake
def egg_whites_per_cake : ℕ := 8

-- Define the number of cakes
def number_of_cakes : ℕ := 2

-- Express the total amount of aquafaba needed
def aquafaba_needed : ℕ :=
  tablespoons_per_egg_white * egg_whites_per_cake * number_of_cakes

-- Statement asserting the amount of aquafaba needed is 32
theorem christine_aquafaba_needed : aquafaba_needed = 32 := by
  sorry

end NUMINAMATH_GPT_christine_aquafaba_needed_l641_64175


namespace NUMINAMATH_GPT_simultaneous_eq_solution_l641_64157

theorem simultaneous_eq_solution (n : ℝ) (hn : n ≠ 1 / 2) : 
  ∃ (x y : ℝ), (y = (3 * n + 1) * x + 2) ∧ (y = (5 * n - 2) * x + 5) := 
sorry

end NUMINAMATH_GPT_simultaneous_eq_solution_l641_64157


namespace NUMINAMATH_GPT_lassie_original_bones_l641_64158

variable (B : ℕ) -- B is the number of bones Lassie started with

-- Conditions translated into Lean statements
def eats_half_on_saturday (B : ℕ) : ℕ := B / 2
def receives_ten_more_on_sunday (B : ℕ) : ℕ := eats_half_on_saturday B + 10
def total_bones_after_sunday (B : ℕ) : Prop := receives_ten_more_on_sunday B = 35

-- Proof goal: B is equal to 50 given the conditions
theorem lassie_original_bones :
  total_bones_after_sunday B → B = 50 :=
sorry

end NUMINAMATH_GPT_lassie_original_bones_l641_64158


namespace NUMINAMATH_GPT_math_problem_l641_64112

def foo (a b : ℝ) (h : a + b > 0) : Prop :=
  (a^5 * b^2 + a^4 * b^3 ≥ 0) ∧
  ¬ (a^4 * b^3 + a^3 * b^4 ≥ 0) ∧
  (a^21 + b^21 > 0) ∧
  ((a + 2) * (b + 2) > a * b) ∧
  ¬ ((a - 3) * (b - 3) < a * b) ∧
  ¬ ((a + 2) * (b + 3) > a * b + 5)

theorem math_problem (a b : ℝ) (h : a + b > 0) : foo a b h :=
by
  -- The proof will be here
  sorry

end NUMINAMATH_GPT_math_problem_l641_64112


namespace NUMINAMATH_GPT_find_f1_plus_gneg1_l641_64103

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry

-- Conditions
axiom f_odd : ∀ x : ℝ, f (-x) = -f x
axiom g_even : ∀ x : ℝ, g (-x) = g x
axiom relation : ∀ x : ℝ, f x - g x = (1 / 2) ^ x

-- Proof statement
theorem find_f1_plus_gneg1 : f 1 + g (-1) = -2 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_find_f1_plus_gneg1_l641_64103


namespace NUMINAMATH_GPT_find_omega_l641_64195

theorem find_omega (ω : ℝ) (h₀ : ω > 0) (h₁ : (π / ω = π / 2)) : ω = 2 :=
sorry

end NUMINAMATH_GPT_find_omega_l641_64195


namespace NUMINAMATH_GPT_num_orange_juice_l641_64146

-- Definitions based on the conditions in the problem
def O : ℝ := sorry -- To represent the number of bottles of orange juice
def A : ℝ := sorry -- To represent the number of bottles of apple juice
def cost_orange_juice : ℝ := 0.70
def cost_apple_juice : ℝ := 0.60
def total_cost : ℝ := 46.20
def total_bottles : ℝ := 70

-- Conditions used as definitions in Lean 4
axiom condition1 : O + A = total_bottles
axiom condition2 : cost_orange_juice * O + cost_apple_juice * A = total_cost

-- Proof statement with the correct answer
theorem num_orange_juice : O = 42 := by
  sorry

end NUMINAMATH_GPT_num_orange_juice_l641_64146


namespace NUMINAMATH_GPT_symmetric_line_equation_l641_64122

theorem symmetric_line_equation 
  (l1 : ∀ x y : ℝ, x - 2 * y - 2 = 0) 
  (l2 : ∀ x y : ℝ, x + y = 0) : 
  ∀ x y : ℝ, 2 * x - y - 2 = 0 :=
sorry

end NUMINAMATH_GPT_symmetric_line_equation_l641_64122


namespace NUMINAMATH_GPT_nested_f_has_zero_l641_64172

def f (x : ℝ) : ℝ := x^2 + 2017 * x + 1

theorem nested_f_has_zero (n : ℕ) (hn : n ≥ 1) : ∃ x : ℝ, (Nat.iterate f n x) = 0 :=
by
  sorry

end NUMINAMATH_GPT_nested_f_has_zero_l641_64172


namespace NUMINAMATH_GPT_time_to_fill_by_B_l641_64136

/-- 
Assume a pool with two taps, A and B, fills in 30 minutes when both are open.
When both are open for 10 minutes, and then only B is open for another 40 minutes, the pool fills up.
Prove that if only tap B is opened, it would take 60 minutes to fill the pool.
-/
theorem time_to_fill_by_B
  (r_A r_B : ℝ)
  (H1 : (r_A + r_B) * 30 = 1)
  (H2 : ((r_A + r_B) * 10 + r_B * 40) = 1) :
  1 / r_B = 60 :=
by
  sorry

end NUMINAMATH_GPT_time_to_fill_by_B_l641_64136


namespace NUMINAMATH_GPT_ratio_of_shares_l641_64142

theorem ratio_of_shares 
    (sheila_share : ℕ → ℕ)
    (rose_share : ℕ)
    (total_rent : ℕ) 
    (h1 : ∀ P, sheila_share P = 5 * P)
    (h2 : rose_share = 1800)
    (h3 : ∀ P, sheila_share P + P + rose_share = total_rent) 
    (h4 : total_rent = 5400) :
    ∃ P, 1800 / P = 3 := 
by 
  sorry

end NUMINAMATH_GPT_ratio_of_shares_l641_64142


namespace NUMINAMATH_GPT_min_value_expression_l641_64152

theorem min_value_expression :
  ∀ (x y : ℝ), x > 0 → y > 0 → x + y = 1 → (∃ (c : ℝ), c = 16 ∧ ∀ z, z = (1 / x + 9 / y) → z ≥ c) :=
by
  sorry

end NUMINAMATH_GPT_min_value_expression_l641_64152


namespace NUMINAMATH_GPT_number_of_triangles_with_one_side_five_not_shortest_l641_64125

theorem number_of_triangles_with_one_side_five_not_shortest (a b c : ℕ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0)
  (h_one_side_five : a = 5 ∨ b = 5 ∨ c = 5)
  (h_not_shortest : a = 5 ∧ a > b ∧ a > c ∨ b = 5 ∧ b > a ∧ b > c ∨ c = 5 ∧ c > a ∧ c > b ∨ a ≠ 5 ∧ b = 5 ∧ b > c ∨ a ≠ 5 ∧ c = 5 ∧ c > b) :
  (∃ n, n = 10) :=
by
  sorry

end NUMINAMATH_GPT_number_of_triangles_with_one_side_five_not_shortest_l641_64125


namespace NUMINAMATH_GPT_tan_7pi_over_6_eq_1_over_sqrt_3_l641_64186

theorem tan_7pi_over_6_eq_1_over_sqrt_3 : 
  ∀ θ : ℝ, θ = (7 * Real.pi) / 6 → Real.tan θ = 1 / Real.sqrt 3 :=
by
  intros θ hθ
  rw [hθ]
  sorry  -- Proof to be completed

end NUMINAMATH_GPT_tan_7pi_over_6_eq_1_over_sqrt_3_l641_64186


namespace NUMINAMATH_GPT_card_d_total_percent_change_l641_64109

noncomputable def card_d_initial_value : ℝ := 250
noncomputable def card_d_percent_changes : List ℝ := [0.05, -0.15, 0.30, -0.10, 0.20]

noncomputable def final_value (initial_value : ℝ) (changes : List ℝ) : ℝ :=
  changes.foldl (λ acc change => acc * (1 + change)) initial_value

theorem card_d_total_percent_change :
  let final_val := final_value card_d_initial_value card_d_percent_changes
  let total_percent_change := ((final_val - card_d_initial_value) / card_d_initial_value) * 100
  total_percent_change = 25.307 := by
  sorry

end NUMINAMATH_GPT_card_d_total_percent_change_l641_64109


namespace NUMINAMATH_GPT_Melies_money_left_l641_64138

variable (meat_weight : ℕ)
variable (meat_cost_per_kg : ℕ)
variable (initial_money : ℕ)

def money_left_after_purchase (meat_weight : ℕ) (meat_cost_per_kg : ℕ) (initial_money : ℕ) : ℕ :=
  initial_money - (meat_weight * meat_cost_per_kg)

theorem Melies_money_left : 
  money_left_after_purchase 2 82 180 = 16 :=
by
  sorry

end NUMINAMATH_GPT_Melies_money_left_l641_64138


namespace NUMINAMATH_GPT_expression_value_l641_64154

theorem expression_value : 
  (Nat.factorial 10) / (2 * (Finset.sum (Finset.range 11) id)) = 33080 := by
  sorry

end NUMINAMATH_GPT_expression_value_l641_64154


namespace NUMINAMATH_GPT_pentagon_perimeter_even_l641_64104

noncomputable def dist_sq (A B : ℤ × ℤ) : ℤ :=
  (A.1 - B.1) ^ 2 + (A.2 - B.2) ^ 2

theorem pentagon_perimeter_even (A B C D E : ℤ × ℤ) (h1 : dist_sq A B % 2 = 1) (h2 : dist_sq B C % 2 = 1) 
  (h3 : dist_sq C D % 2 = 1) (h4 : dist_sq D E % 2 = 1) (h5 : dist_sq E A % 2 = 1) : 
  (dist_sq A B + dist_sq B C + dist_sq C D + dist_sq D E + dist_sq E A) % 2 = 0 := 
by 
  sorry

end NUMINAMATH_GPT_pentagon_perimeter_even_l641_64104


namespace NUMINAMATH_GPT_repeating_decimal_fraction_l641_64165

theorem repeating_decimal_fraction (h : 0.02 = 2 / 99) : 2.06 = 68 / 33 :=
by
  sorry

end NUMINAMATH_GPT_repeating_decimal_fraction_l641_64165


namespace NUMINAMATH_GPT_race_time_l641_64143

theorem race_time (t : ℝ) (h1 : 100 / t = 66.66666666666667 / 45) : t = 67.5 :=
by
  sorry

end NUMINAMATH_GPT_race_time_l641_64143


namespace NUMINAMATH_GPT_fraction_to_decimal_l641_64120

-- We define the fraction and its simplified form
def fraction : ℚ := 58 / 160
def simplified_fraction : ℚ := 29 / 80

-- We state that the fraction simplifies correctly
lemma simplify_fraction : fraction = simplified_fraction := by
  sorry

-- Define the factorization of the denominator
def denominator_factorization : ℕ := 2^4 * 5

-- Verify the fraction when multiplied by 125/125
def equalized_fraction : ℚ := 29 * 125 / 10000

-- State the final result as a decimal
theorem fraction_to_decimal : fraction = 0.3625 := by
  sorry

end NUMINAMATH_GPT_fraction_to_decimal_l641_64120
