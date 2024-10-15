import Mathlib

namespace NUMINAMATH_GPT_ellipse_eq_and_line_eq_l2129_212984

theorem ellipse_eq_and_line_eq
  (e : ℝ) (a b c xC yC: ℝ)
  (h_e : e = (Real.sqrt 3 / 2))
  (h_a : a = 2)
  (h_c : c = Real.sqrt 3)
  (h_b : b = Real.sqrt (a^2 - c^2))
  (h_ellipse : ∀ x y, (x^2 / a^2 + y^2 / b^2 = 1) ↔ (x^2 / 4 + y^2 = 1))
  (h_C_on_G : xC^2 / 4 + yC^2 = 1)
  (h_diameter_condition : ∀ (B : ℝ × ℝ), B = (0, 1) →
    ((2 * xC - yC + 1 = 0) →
    (xC = 0 ∧ yC = 1) ∨ (xC = -16 / 17 ∧ yC = -15 / 17)))
  : (∀ x y, (y = 2*x + 1) ↔ (x + 2*y - 2 = 0 ∨ 3*x - 10*y - 6 = 0)) :=
by
  sorry

end NUMINAMATH_GPT_ellipse_eq_and_line_eq_l2129_212984


namespace NUMINAMATH_GPT_sum_of_squares_eq_zero_iff_all_zero_l2129_212940

theorem sum_of_squares_eq_zero_iff_all_zero (a b c : ℝ) :
  a^2 + b^2 + c^2 = 0 ↔ a = 0 ∧ b = 0 ∧ c = 0 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_squares_eq_zero_iff_all_zero_l2129_212940


namespace NUMINAMATH_GPT_total_boys_slide_l2129_212987

theorem total_boys_slide (initial_boys additional_boys : ℕ) (h1 : initial_boys = 22) (h2 : additional_boys = 13) :
  initial_boys + additional_boys = 35 :=
by
  sorry

end NUMINAMATH_GPT_total_boys_slide_l2129_212987


namespace NUMINAMATH_GPT_inverse_exists_l2129_212970

noncomputable def f (x : ℝ) : ℝ := 7 * x^3 - 2 * x^2 + 5 * x - 9

theorem inverse_exists :
  ∃ x : ℝ, 7 * x^3 - 2 * x^2 + 5 * x - 5.5 = 0 :=
sorry

end NUMINAMATH_GPT_inverse_exists_l2129_212970


namespace NUMINAMATH_GPT_find_angle_C_find_perimeter_l2129_212927

-- Definitions related to the triangle problem
variables {A B C : ℝ}
variables {a b c : ℝ} -- sides opposite to A, B, C

-- Condition: (2a - b) * cos C = c * cos B
def condition_1 (a b c C B : ℝ) : Prop := (2 * a - b) * Real.cos C = c * Real.cos B

-- Given C in radians (part 1: find angle C)
theorem find_angle_C 
  (A B C : ℝ) (a b c : ℝ) 
  (h1 : condition_1 a b c C B) 
  (H1 : 0 < C) (H2 : C < Real.pi) :
  C = Real.pi / 3 := 
sorry

-- More conditions for part 2
variables (area : ℝ) -- given area of triangle
def condition_2 (a b C area : ℝ) : Prop := 0.5 * a * b * Real.sin C = area

-- Given c = 2 and area = sqrt(3) (part 2: find perimeter)
theorem find_perimeter 
  (A B C : ℝ) (a b : ℝ) (c : ℝ) (area : ℝ) 
  (h2 : condition_2 a b C area) 
  (Hc : c = 2) (Harea : area = Real.sqrt 3) :
  a + b + c = 6 := 
sorry

end NUMINAMATH_GPT_find_angle_C_find_perimeter_l2129_212927


namespace NUMINAMATH_GPT_solve_abs_ineq_l2129_212971

theorem solve_abs_ineq (x : ℝ) : |(8 - x) / 4| < 3 ↔ 4 < x ∧ x < 20 := by
  sorry

end NUMINAMATH_GPT_solve_abs_ineq_l2129_212971


namespace NUMINAMATH_GPT_double_angle_cosine_calculation_l2129_212960

theorem double_angle_cosine_calculation :
    2 * (Real.cos (Real.pi / 12))^2 - 1 = Real.cos (Real.pi / 6) := 
by
    sorry

end NUMINAMATH_GPT_double_angle_cosine_calculation_l2129_212960


namespace NUMINAMATH_GPT_longest_side_of_enclosure_l2129_212991

theorem longest_side_of_enclosure (l w : ℝ) (hlw : 2*l + 2*w = 240) (harea : l*w = 2880) : max l w = 72 := 
by {
  sorry
}

end NUMINAMATH_GPT_longest_side_of_enclosure_l2129_212991


namespace NUMINAMATH_GPT_total_hatched_eggs_l2129_212923

noncomputable def fertile_eggs (total_eggs : ℕ) (infertility_rate : ℝ) : ℝ :=
  total_eggs * (1 - infertility_rate)

noncomputable def hatching_eggs_after_calcification (fertile_eggs : ℝ) (calcification_rate : ℝ) : ℝ :=
  fertile_eggs * (1 - calcification_rate)

noncomputable def hatching_eggs_after_predator (hatching_eggs : ℝ) (predator_rate : ℝ) : ℝ :=
  hatching_eggs * (1 - predator_rate)

noncomputable def hatching_eggs_after_temperature (hatching_eggs : ℝ) (temperature_rate : ℝ) : ℝ :=
  hatching_eggs * (1 - temperature_rate)

open Nat

theorem total_hatched_eggs :
  let g1_total_eggs := 30
  let g2_total_eggs := 40
  let g1_infertility_rate := 0.20
  let g2_infertility_rate := 0.25
  let g1_calcification_rate := 1.0 / 3.0
  let g2_calcification_rate := 0.25
  let predator_rate := 0.10
  let temperature_rate := 0.05
  let g1_fertile := fertile_eggs g1_total_eggs g1_infertility_rate
  let g1_hatch_calcification := hatching_eggs_after_calcification g1_fertile g1_calcification_rate
  let g1_hatch_predator := hatching_eggs_after_predator g1_hatch_calcification predator_rate
  let g1_hatch_temp := hatching_eggs_after_temperature g1_hatch_predator temperature_rate
  let g2_fertile := fertile_eggs g2_total_eggs g2_infertility_rate
  let g2_hatch_calcification := hatching_eggs_after_calcification g2_fertile g2_calcification_rate
  let g2_hatch_predator := hatching_eggs_after_predator g2_hatch_calcification predator_rate
  let g2_hatch_temp := hatching_eggs_after_temperature g2_hatch_predator temperature_rate
  let total_hatched := g1_hatch_temp + g2_hatch_temp
  floor total_hatched = 32 :=
by
  sorry

end NUMINAMATH_GPT_total_hatched_eggs_l2129_212923


namespace NUMINAMATH_GPT_evening_campers_l2129_212965

theorem evening_campers (morning_campers afternoon_campers total_campers : ℕ) (h_morning : morning_campers = 36) (h_afternoon : afternoon_campers = 13) (h_total : total_campers = 98) :
  total_campers - (morning_campers + afternoon_campers) = 49 :=
by
  sorry

end NUMINAMATH_GPT_evening_campers_l2129_212965


namespace NUMINAMATH_GPT_zero_of_f_l2129_212922

noncomputable def f (x : ℝ) : ℝ := (|Real.log x - Real.log 2|) - (1 / 3) ^ x

theorem zero_of_f :
  ∃ x1 x2 : ℝ, x1 < x2 ∧ (f x1 = 0) ∧ (f x2 = 0) ∧
  (1 < x1 ∧ x1 < 2) ∧ (2 < x2) := 
sorry

end NUMINAMATH_GPT_zero_of_f_l2129_212922


namespace NUMINAMATH_GPT_ferris_wheel_seat_calculation_l2129_212959

theorem ferris_wheel_seat_calculation (n k : ℕ) (h1 : n = 4) (h2 : k = 2) : n / k = 2 := 
by
  sorry

end NUMINAMATH_GPT_ferris_wheel_seat_calculation_l2129_212959


namespace NUMINAMATH_GPT_bricks_in_top_half_l2129_212997

theorem bricks_in_top_half (total_rows bottom_rows top_rows bricks_per_bottom_row total_bricks bricks_per_top_row: ℕ) 
  (h_total_rows : total_rows = 10)
  (h_bottom_rows : bottom_rows = 5)
  (h_top_rows : top_rows = 5)
  (h_bricks_per_bottom_row : bricks_per_bottom_row = 12)
  (h_total_bricks : total_bricks = 100)
  (h_bricks_per_top_row : bricks_per_top_row = (total_bricks - bottom_rows * bricks_per_bottom_row) / top_rows) : 
  bricks_per_top_row = 8 := 
by 
  sorry

end NUMINAMATH_GPT_bricks_in_top_half_l2129_212997


namespace NUMINAMATH_GPT_sum_of_numbers_l2129_212913

theorem sum_of_numbers (x y : ℝ) (h1 : y = 4 * x) (h2 : x + y = 45) : x + y = 45 := 
by
  sorry

end NUMINAMATH_GPT_sum_of_numbers_l2129_212913


namespace NUMINAMATH_GPT_percent_errors_l2129_212978

theorem percent_errors (S : ℝ) (hS : S > 0) (Sm : ℝ) (hSm : Sm = 1.25 * S) :
  let P := 4 * S
  let Pm := 4 * Sm
  let A := S^2
  let Am := Sm^2
  let D := S * Real.sqrt 2
  let Dm := Sm * Real.sqrt 2
  let E_P := ((Pm - P) / P) * 100
  let E_A := ((Am - A) / A) * 100
  let E_D := ((Dm - D) / D) * 100
  E_P = 25 ∧ E_A = 56.25 ∧ E_D = 25 :=
by
  sorry

end NUMINAMATH_GPT_percent_errors_l2129_212978


namespace NUMINAMATH_GPT_smallest_part_of_division_l2129_212911

theorem smallest_part_of_division (x : ℝ) (h : 2 * x + (1/2) * x + (1/4) * x = 105) : 
  (1/4) * x = 10.5 :=
sorry

end NUMINAMATH_GPT_smallest_part_of_division_l2129_212911


namespace NUMINAMATH_GPT_workers_time_to_complete_job_l2129_212948

theorem workers_time_to_complete_job (D E Z H k : ℝ) (h1 : 1 / D + 1 / E + 1 / Z + 1 / H = 1 / (D - 8))
  (h2 : 1 / D + 1 / E + 1 / Z + 1 / H = 1 / (E - 2))
  (h3 : 1 / D + 1 / E + 1 / Z + 1 / H = 3 / Z) :
  E = 10 → Z = 3 * (E - 2) → k = 120 / 19 :=
by
  intros hE hZ
  sorry

end NUMINAMATH_GPT_workers_time_to_complete_job_l2129_212948


namespace NUMINAMATH_GPT_lineD_is_parallel_to_line1_l2129_212951

-- Define the lines
def line1 (x y : ℝ) := x - 2 * y + 1 = 0
def lineA (x y : ℝ) := 2 * x - y + 1 = 0
def lineB (x y : ℝ) := 2 * x - 4 * y + 2 = 0
def lineC (x y : ℝ) := 2 * x + 4 * y + 1 = 0
def lineD (x y : ℝ) := 2 * x - 4 * y + 1 = 0

-- Define a function to check parallelism between lines
def are_parallel (f g : ℝ → ℝ → Prop) :=
  ∀ x y : ℝ, (f x y → g x y) ∨ (g x y → f x y)

-- Prove that lineD is parallel to line1
theorem lineD_is_parallel_to_line1 : are_parallel line1 lineD :=
by
  sorry

end NUMINAMATH_GPT_lineD_is_parallel_to_line1_l2129_212951


namespace NUMINAMATH_GPT_total_orchestra_l2129_212998

def percussion_section : ℕ := 4
def brass_section : ℕ := 13
def strings_section : ℕ := 18
def woodwinds_section : ℕ := 10
def keyboards_and_harp_section : ℕ := 3
def maestro : ℕ := 1

theorem total_orchestra (p b s w k m : ℕ) 
  (h_p : p = percussion_section)
  (h_b : b = brass_section)
  (h_s : s = strings_section)
  (h_w : w = woodwinds_section)
  (h_k : k = keyboards_and_harp_section)
  (h_m : m = maestro) :
  p + b + s + w + k + m = 49 := by 
  rw [h_p, h_b, h_s, h_w, h_k, h_m]
  unfold percussion_section brass_section strings_section woodwinds_section keyboards_and_harp_section maestro
  norm_num

end NUMINAMATH_GPT_total_orchestra_l2129_212998


namespace NUMINAMATH_GPT_distinct_complex_roots_A_eq_neg7_l2129_212949

theorem distinct_complex_roots_A_eq_neg7 (x₁ x₂ : ℂ) (A : ℝ) (hx1: x₁ ≠ x₂)
  (h1 : x₁ * (x₁ + 1) = A)
  (h2 : x₂ * (x₂ + 1) = A)
  (h3 : x₁^4 + 3 * x₁^3 + 5 * x₁ = x₂^4 + 3 * x₂^3 + 5 * x₂) : A = -7 := 
sorry

end NUMINAMATH_GPT_distinct_complex_roots_A_eq_neg7_l2129_212949


namespace NUMINAMATH_GPT_louis_age_l2129_212921

variable (L J M : ℕ) -- L for Louis, J for Jerica, and M for Matilda

theorem louis_age : 
  (M = 35) ∧ (M = J + 7) ∧ (J = 2 * L) → L = 14 := 
by 
  intro h 
  sorry

end NUMINAMATH_GPT_louis_age_l2129_212921


namespace NUMINAMATH_GPT_Meghan_scored_20_marks_less_than_Jose_l2129_212956

theorem Meghan_scored_20_marks_less_than_Jose
  (M J A : ℕ)
  (h1 : J = A + 40)
  (h2 : M + J + A = 210)
  (h3 : J = 100 - 10) :
  J - M = 20 :=
by
  -- Skipping the proof
  sorry

end NUMINAMATH_GPT_Meghan_scored_20_marks_less_than_Jose_l2129_212956


namespace NUMINAMATH_GPT_heat_required_l2129_212928

theorem heat_required (m : ℝ) (c₀ : ℝ) (alpha : ℝ) (t₁ t₂ : ℝ) :
  m = 2 ∧ c₀ = 150 ∧ alpha = 0.05 ∧ t₁ = 20 ∧ t₂ = 100 →
  let Δt := t₂ - t₁
  let c_avg := (c₀ * (1 + alpha * t₁) + c₀ * (1 + alpha * t₂)) / 2
  let Q := c_avg * m * Δt
  Q = 96000 := by
  sorry

end NUMINAMATH_GPT_heat_required_l2129_212928


namespace NUMINAMATH_GPT_problem_f_2011_2012_l2129_212918

noncomputable def f : ℝ → ℝ := sorry

theorem problem_f_2011_2012 :
  (∀ x : ℝ, f (-x) = -f x) →
  (∀ x : ℝ, (0 ≤ x ∧ x ≤ 1) → f (1-x) = f (1+x)) →
  (∀ x : ℝ, (0 ≤ x ∧ x ≤ 1) → f x = 2^x - 1) →
  f 2011 + f 2012 = -1 :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_problem_f_2011_2012_l2129_212918


namespace NUMINAMATH_GPT_intersection_of_A_and_B_l2129_212963

namespace ProofProblem

def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 3, 5}
def B : Set ℕ := {1, 2, 4, 5}

theorem intersection_of_A_and_B :
  A ∩ B = {1, 5} := by
  sorry

end ProofProblem

end NUMINAMATH_GPT_intersection_of_A_and_B_l2129_212963


namespace NUMINAMATH_GPT_calculate_f8_f4_l2129_212941

open Real

noncomputable def f : ℝ → ℝ := sorry

axiom odd_function (x : ℝ) : f (-x) = -f x
axiom periodic_function (x : ℝ) : f (x + 5) = f x
axiom f_at_1 : f 1 = 1
axiom f_at_2 : f 2 = 3

theorem calculate_f8_f4 : f 8 - f 4 = -2 := by
  sorry

end NUMINAMATH_GPT_calculate_f8_f4_l2129_212941


namespace NUMINAMATH_GPT_monthly_earnings_l2129_212904

theorem monthly_earnings (savings_per_month : ℤ) (total_needed : ℤ) (total_earned : ℤ)
  (H1 : savings_per_month = 500)
  (H2 : total_needed = 45000)
  (H3 : total_earned = 360000) :
  total_earned / (total_needed / savings_per_month) = 4000 := by
  sorry

end NUMINAMATH_GPT_monthly_earnings_l2129_212904


namespace NUMINAMATH_GPT_leonard_younger_than_nina_by_4_l2129_212926

variable (L N J : ℕ)

-- Conditions based on conditions from the problem
axiom h1 : L = 6
axiom h2 : N = 1 / 2 * J
axiom h3 : L + N + J = 36

-- Statement to prove
theorem leonard_younger_than_nina_by_4 : N - L = 4 :=
by 
  sorry

end NUMINAMATH_GPT_leonard_younger_than_nina_by_4_l2129_212926


namespace NUMINAMATH_GPT_min_value_condition_l2129_212950

theorem min_value_condition
  (a b c d e f g h : ℝ)
  (h1 : a * b * c * d = 16)
  (h2 : e * f * g * h = 36) :
  ∃ x : ℝ, x = (ae)^2 + (bf)^2 + (cg)^2 + (dh)^2 ∧ x ≥ 576 := sorry

end NUMINAMATH_GPT_min_value_condition_l2129_212950


namespace NUMINAMATH_GPT_image_of_center_l2129_212994

def original_center : ℤ × ℤ := (3, -4)

def reflect_x (p : ℤ × ℤ) : ℤ × ℤ := (p.1, -p.2)
def reflect_y (p : ℤ × ℤ) : ℤ × ℤ := (-p.1, p.2)
def translate_down (p : ℤ × ℤ) (d : ℤ) : ℤ × ℤ := (p.1, p.2 - d)

theorem image_of_center :
  (translate_down (reflect_y (reflect_x original_center)) 10) = (-3, -6) :=
by
  sorry

end NUMINAMATH_GPT_image_of_center_l2129_212994


namespace NUMINAMATH_GPT_a_minus_b_a_squared_lt_zero_sufficient_not_necessary_for_a_lt_b_l2129_212910

theorem a_minus_b_a_squared_lt_zero_sufficient_not_necessary_for_a_lt_b
  (a b : ℝ) :
  (∀ a b : ℝ, (a - b) * a ^ 2 < 0 → a < b) ∧ 
  (¬∀ a b : ℝ, a < b → (a - b) * a ^ 2 < 0) :=
sorry

end NUMINAMATH_GPT_a_minus_b_a_squared_lt_zero_sufficient_not_necessary_for_a_lt_b_l2129_212910


namespace NUMINAMATH_GPT_condition_A_condition_B_condition_C_condition_D_correct_answer_l2129_212988

theorem condition_A : ∀ x : ℝ, x^2 + 2 * x - 1 ≠ x * (x + 2) - 1 := sorry

theorem condition_B : ∀ a b : ℝ, (a + b)^2 = a^2 + 2 * a * b + b^2 := sorry

theorem condition_C : ∀ x y : ℝ, x^2 - 4 * y^2 = (x + 2 * y) * (x - 2 * y) := sorry

theorem condition_D : ∀ a b : ℝ, a^2 - a * b - a ≠ a * (a - b) := sorry

theorem correct_answer : ∀ x y : ℝ, (x^2 - 4 * y^2) = (x + 2 * y) * (x - 2 * y) := 
  by 
    exact condition_C

end NUMINAMATH_GPT_condition_A_condition_B_condition_C_condition_D_correct_answer_l2129_212988


namespace NUMINAMATH_GPT_dividend_calculation_l2129_212902

theorem dividend_calculation :
  let divisor := 17
  let quotient := 9
  let remainder := 6
  let dividend := 159
  (divisor * quotient) + remainder = dividend :=
by
  sorry

end NUMINAMATH_GPT_dividend_calculation_l2129_212902


namespace NUMINAMATH_GPT_chord_length_range_l2129_212982

open Real

def chord_length_ge (t : ℝ) : Prop :=
  let r := sqrt 8
  let l := (4 * sqrt 2) / 3
  let d := abs t / sqrt 2
  let s := l / 2
  s ≤ sqrt (r^2 - d^2)

theorem chord_length_range (t : ℝ) : chord_length_ge t ↔ -((8 * sqrt 2) / 3) ≤ t ∧ t ≤ (8 * sqrt 2) / 3 :=
by
  sorry

end NUMINAMATH_GPT_chord_length_range_l2129_212982


namespace NUMINAMATH_GPT_no_opposite_meanings_in_C_l2129_212954

def opposite_meanings (condition : String) : Prop :=
  match condition with
  | "A" => true
  | "B" => true
  | "C" => false
  | "D" => true
  | _   => false

theorem no_opposite_meanings_in_C :
  opposite_meanings "C" = false :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_no_opposite_meanings_in_C_l2129_212954


namespace NUMINAMATH_GPT_point_on_y_axis_l2129_212906

theorem point_on_y_axis (m : ℝ) (M : ℝ × ℝ) (hM : M = (m + 1, m + 3)) (h_on_y_axis : M.1 = 0) : M = (0, 2) :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_point_on_y_axis_l2129_212906


namespace NUMINAMATH_GPT_projectile_reaches_100_feet_l2129_212986

theorem projectile_reaches_100_feet :
  ∃ (t : ℝ), t > 0 ∧ (-16 * t ^ 2 + 80 * t = 100) ∧ (t = 2.5) := by
sorry

end NUMINAMATH_GPT_projectile_reaches_100_feet_l2129_212986


namespace NUMINAMATH_GPT_largest_sphere_radius_on_torus_l2129_212985

theorem largest_sphere_radius_on_torus
  (inner_radius outer_radius : ℝ)
  (torus_center : ℝ × ℝ × ℝ)
  (circle_radius : ℝ)
  (sphere_radius : ℝ)
  (sphere_center : ℝ × ℝ × ℝ) :
  inner_radius = 3 →
  outer_radius = 5 →
  torus_center = (4, 0, 1) →
  circle_radius = 1 →
  sphere_center = (0, 0, sphere_radius) →
  sphere_radius = 4 :=
by
  intros h_inner_radius h_outer_radius h_torus_center h_circle_radius h_sphere_center
  sorry

end NUMINAMATH_GPT_largest_sphere_radius_on_torus_l2129_212985


namespace NUMINAMATH_GPT_exists_infinite_solutions_l2129_212933

theorem exists_infinite_solutions :
  ∃ (x y z : ℤ), (∀ k : ℤ, x = 2 * k ∧ y = 999 - 2 * k ^ 2 ∧ z = 998 - 2 * k ^ 2) ∧ (x ^ 2 + y ^ 2 - z ^ 2 = 1997) :=
by 
  -- The proof should go here
  sorry

end NUMINAMATH_GPT_exists_infinite_solutions_l2129_212933


namespace NUMINAMATH_GPT_average_of_list_l2129_212939

theorem average_of_list (n : ℕ) (h : (2 + 9 + 4 + n + 2 * n) / 5 = 6) : n = 5 := 
by
  sorry

end NUMINAMATH_GPT_average_of_list_l2129_212939


namespace NUMINAMATH_GPT_find_x_l2129_212976

theorem find_x : ∃ x : ℝ, (1 / 3 * ((2 * x + 5) + (8 * x + 3) + (3 * x + 8)) = 5 * x - 10) ∧ x = 23 :=
by
  sorry

end NUMINAMATH_GPT_find_x_l2129_212976


namespace NUMINAMATH_GPT_complete_the_square_l2129_212955

theorem complete_the_square (x : ℝ) (h : x^2 - 4 * x + 3 = 0) : (x - 2)^2 = 1 :=
sorry

end NUMINAMATH_GPT_complete_the_square_l2129_212955


namespace NUMINAMATH_GPT_unique_handshakes_count_l2129_212901

-- Definitions from the conditions
def teams : Nat := 4
def players_per_team : Nat := 2
def total_players : Nat := teams * players_per_team

def handshakes_per_player : Nat := total_players - players_per_team

-- The Lean statement to prove the total number of unique handshakes
theorem unique_handshakes_count : (total_players * handshakes_per_player) / 2 = 24 := 
by
  -- Proof steps would go here
  sorry

end NUMINAMATH_GPT_unique_handshakes_count_l2129_212901


namespace NUMINAMATH_GPT_probability_of_drawing_three_white_balls_l2129_212999

theorem probability_of_drawing_three_white_balls
  (total_balls white_balls black_balls: ℕ)
  (h_total: total_balls = 15)
  (h_white: white_balls = 7)
  (h_black: black_balls = 8)
  (draws: ℕ)
  (h_draws: draws = 3) :
  (Nat.choose white_balls draws / Nat.choose total_balls draws) = (7 / 91) :=
by sorry

end NUMINAMATH_GPT_probability_of_drawing_three_white_balls_l2129_212999


namespace NUMINAMATH_GPT_time_to_cross_bridge_l2129_212947

def length_of_train : ℕ := 250
def length_of_bridge : ℕ := 150
def speed_in_kmhr : ℕ := 72
def speed_in_ms : ℕ := (speed_in_kmhr * 1000) / 3600

theorem time_to_cross_bridge : 
  (length_of_train + length_of_bridge) / speed_in_ms = 20 :=
by
  have total_distance := length_of_train + length_of_bridge
  have speed := speed_in_ms
  sorry

end NUMINAMATH_GPT_time_to_cross_bridge_l2129_212947


namespace NUMINAMATH_GPT_hypotenuse_length_l2129_212992

theorem hypotenuse_length (a c : ℝ) (h_perimeter : 2 * a + c = 36) (h_area : (1 / 2) * a^2 = 24) : c = 4 * Real.sqrt 6 :=
by
  sorry

end NUMINAMATH_GPT_hypotenuse_length_l2129_212992


namespace NUMINAMATH_GPT_least_three_digit_12_heavy_number_l2129_212942

def is_12_heavy (n : ℕ) : Prop :=
  n % 12 > 8

def three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

theorem least_three_digit_12_heavy_number :
  ∃ n, three_digit n ∧ is_12_heavy n ∧ ∀ m, three_digit m ∧ is_12_heavy m → n ≤ m :=
  Exists.intro 105 (by
    sorry)

end NUMINAMATH_GPT_least_three_digit_12_heavy_number_l2129_212942


namespace NUMINAMATH_GPT_factorize_expression_l2129_212934

theorem factorize_expression (x y : ℂ) : (x * y^2 - x = x * (y + 1) * (y - 1)) :=
sorry

end NUMINAMATH_GPT_factorize_expression_l2129_212934


namespace NUMINAMATH_GPT_angle_triple_complement_l2129_212917

-- Let x be the angle in degrees.
-- The angle is triple its complement.
-- We need to prove that x = 67.5.

theorem angle_triple_complement (x : ℝ) (h : x = 3 * (90 - x)) : x = 67.5 := 
by
  sorry

end NUMINAMATH_GPT_angle_triple_complement_l2129_212917


namespace NUMINAMATH_GPT_eggs_left_in_jar_l2129_212914

def eggs_after_removal (original removed : Nat) : Nat :=
  original - removed

theorem eggs_left_in_jar : eggs_after_removal 27 7 = 20 :=
by
  sorry

end NUMINAMATH_GPT_eggs_left_in_jar_l2129_212914


namespace NUMINAMATH_GPT_probability_no_adjacent_green_hats_l2129_212932

-- Step d): Rewrite the math proof problem in a Lean 4 statement.

theorem probability_no_adjacent_green_hats (total_children green_hats : ℕ)
  (hc : total_children = 9) (hg : green_hats = 3) :
  (∃ (p : ℚ), p = 5 / 14) :=
sorry

end NUMINAMATH_GPT_probability_no_adjacent_green_hats_l2129_212932


namespace NUMINAMATH_GPT_f_has_four_distinct_real_roots_l2129_212993

noncomputable def f (x d : ℝ) := x ^ 2 + 4 * x + d

theorem f_has_four_distinct_real_roots (d : ℝ) (h : d = 2) :
  ∃ r1 r2 r3 r4 : ℝ, r1 ≠ r2 ∧ r2 ≠ r3 ∧ r3 ≠ r4 ∧ r1 ≠ r3 ∧ r1 ≠ r4 ∧ r2 ≠ r4 ∧ 
  f (f r1 d) = 0 ∧ f (f r2 d) = 0 ∧ f (f r3 d) = 0 ∧ f (f r4 d) = 0 :=
by
  sorry

end NUMINAMATH_GPT_f_has_four_distinct_real_roots_l2129_212993


namespace NUMINAMATH_GPT_four_digit_numbers_with_one_digit_as_average_l2129_212912

noncomputable def count_valid_four_digit_numbers : Nat := 80

theorem four_digit_numbers_with_one_digit_as_average :
  ∃ n : Nat, n = count_valid_four_digit_numbers ∧ n = 80 := by
  use count_valid_four_digit_numbers
  constructor
  · rfl
  · rfl

end NUMINAMATH_GPT_four_digit_numbers_with_one_digit_as_average_l2129_212912


namespace NUMINAMATH_GPT_sum_of_fractions_l2129_212973

theorem sum_of_fractions : (1/2 + 1/2 + 1/3 + 1/3 + 1/3) = 2 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_sum_of_fractions_l2129_212973


namespace NUMINAMATH_GPT_find_ordered_pair_l2129_212961

noncomputable def ordered_pair (c d : ℝ) := c = 1 ∧ d = -2

theorem find_ordered_pair (c d : ℝ) (h1 : c ≠ 0) (h2 : d ≠ 0) (h3 : ∀ x : ℝ, x^2 + c * x + d = 0 → (x = c ∨ x = d)) : ordered_pair c d :=
by
  sorry

end NUMINAMATH_GPT_find_ordered_pair_l2129_212961


namespace NUMINAMATH_GPT_range_of_a_l2129_212925

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 0 then 9 * x + a^2 / x + 7
  else 9 * x + a^2 / x - 7

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, x ≥ 0 → f a x ≥ a + 1) → a ≤ -8/7 :=
by
  intros h
  -- Detailed proof would go here
  sorry

end NUMINAMATH_GPT_range_of_a_l2129_212925


namespace NUMINAMATH_GPT_three_legged_tables_count_l2129_212908

theorem three_legged_tables_count (x y : ℕ) (h1 : 3 * x + 4 * y = 23) (h2 : 2 ≤ x) (h3 : 2 ≤ y) : x = 5 := 
sorry

end NUMINAMATH_GPT_three_legged_tables_count_l2129_212908


namespace NUMINAMATH_GPT_sum_of_integers_l2129_212953

theorem sum_of_integers (x y : ℕ) (h1 : x > y) (h2 : x - y = 15) (h3 : x * y = 56) : x + y = Real.sqrt 449 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_integers_l2129_212953


namespace NUMINAMATH_GPT_Emma_investment_l2129_212909

-- Define the necessary context and variables
variable (E : ℝ) -- Emma's investment
variable (B : ℝ := 500) -- Briana's investment which is a known constant
variable (ROI_Emma : ℝ := 0.30 * E) -- Emma's return on investment after 2 years
variable (ROI_Briana : ℝ := 0.20 * B) -- Briana's return on investment after 2 years
variable (ROI_difference : ℝ := ROI_Emma - ROI_Briana) -- The difference in their ROI

theorem Emma_investment :
  ROI_difference = 10 → E = 366.67 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_Emma_investment_l2129_212909


namespace NUMINAMATH_GPT_transform_M_eq_l2129_212989

open Matrix

def A : Matrix (Fin 2) (Fin 2) ℚ :=
  ![![0, 1/3], ![1, -2/3]]

def M : Fin 2 → ℚ :=
  ![-1, 1]

theorem transform_M_eq :
  A⁻¹.mulVec M = ![-1, -3] :=
by
  sorry

end NUMINAMATH_GPT_transform_M_eq_l2129_212989


namespace NUMINAMATH_GPT_number_of_square_tiles_l2129_212919

-- A box contains a mix of triangular and square tiles.
-- There are 30 tiles in total with 100 edges altogether.
variable (x y : ℕ) -- where x is the number of triangular tiles and y is the number of square tiles, both must be natural numbers
-- Each triangular tile has 3 edges, and each square tile has 4 edges.

-- Define the conditions
def tile_condition_1 : Prop := x + y = 30
def tile_condition_2 : Prop := 3 * x + 4 * y = 100

-- The goal is to prove the number of square tiles y is 10.
theorem number_of_square_tiles : tile_condition_1 x y → tile_condition_2 x y → y = 10 :=
  by
    intros h1 h2
    sorry

end NUMINAMATH_GPT_number_of_square_tiles_l2129_212919


namespace NUMINAMATH_GPT_helen_owes_more_l2129_212964

noncomputable def future_value (P : ℝ) (r : ℝ) (n : ℕ) (t : ℕ) : ℝ :=
  P * (1 + r / n) ^ (n * t)

noncomputable def future_value_semiannually : ℝ :=
  future_value 8000 0.10 2 3

noncomputable def future_value_annually : ℝ :=
  8000 * (1 + 0.10) ^ 3

noncomputable def difference : ℝ :=
  future_value_semiannually - future_value_annually

theorem helen_owes_more : abs (difference - 72.80) < 0.01 :=
by
  sorry

end NUMINAMATH_GPT_helen_owes_more_l2129_212964


namespace NUMINAMATH_GPT_triangle_angles_l2129_212907

variable (a b c t : ℝ)

def angle_alpha : ℝ := 43

def area_condition (α β : ℝ) : Prop :=
  2 * t = a * b * Real.sqrt (Real.sin α ^ 2 + Real.sin β ^ 2 + Real.sin α * Real.sin β)

theorem triangle_angles (α β γ : ℝ) (hα : α = angle_alpha) (h_area : area_condition a b t α β) :
  α = 43 ∧ β = 17 ∧ γ = 120 := sorry

end NUMINAMATH_GPT_triangle_angles_l2129_212907


namespace NUMINAMATH_GPT_just_passed_students_l2129_212983

theorem just_passed_students (total_students : ℕ) 
  (math_first_division_perc : ℕ) 
  (math_second_division_perc : ℕ)
  (eng_first_division_perc : ℕ)
  (eng_second_division_perc : ℕ)
  (sci_first_division_perc : ℕ)
  (sci_second_division_perc : ℕ) 
  (math_just_passed : ℕ)
  (eng_just_passed : ℕ)
  (sci_just_passed : ℕ) :
  total_students = 500 →
  math_first_division_perc = 35 →
  math_second_division_perc = 48 →
  eng_first_division_perc = 25 →
  eng_second_division_perc = 60 →
  sci_first_division_perc = 40 →
  sci_second_division_perc = 45 →
  math_just_passed = (100 - (math_first_division_perc + math_second_division_perc)) * total_students / 100 →
  eng_just_passed = (100 - (eng_first_division_perc + eng_second_division_perc)) * total_students / 100 →
  sci_just_passed = (100 - (sci_first_division_perc + sci_second_division_perc)) * total_students / 100 →
  math_just_passed = 85 ∧ eng_just_passed = 75 ∧ sci_just_passed = 75 :=
by
  intros ht hf1 hf2 he1 he2 hs1 hs2 hjm hje hjs
  sorry

end NUMINAMATH_GPT_just_passed_students_l2129_212983


namespace NUMINAMATH_GPT_proof_inequality_l2129_212936

variable {a b c : ℝ}

theorem proof_inequality (h : a * b < 0) : a^2 + b^2 + c^2 > 2 * a * b + 2 * b * c + 2 * c * a := by
  sorry

end NUMINAMATH_GPT_proof_inequality_l2129_212936


namespace NUMINAMATH_GPT_inequality_solution_l2129_212920

-- Define the problem statement formally
theorem inequality_solution (x : ℝ)
  (h1 : 2 * x > x + 1)
  (h2 : 4 * x - 1 > 7) :
  x > 2 :=
sorry

end NUMINAMATH_GPT_inequality_solution_l2129_212920


namespace NUMINAMATH_GPT_find_number_l2129_212900

-- Statement of the problem in Lean 4
theorem find_number (n : ℝ) (h : n / 3000 = 0.008416666666666666) : n = 25.25 :=
sorry

end NUMINAMATH_GPT_find_number_l2129_212900


namespace NUMINAMATH_GPT_smallest_b_for_quadratic_factors_l2129_212981

theorem smallest_b_for_quadratic_factors :
  ∃ b : ℕ, (∀ r s : ℤ, (r * s = 1764 → r + s = b) → b = 84) :=
sorry

end NUMINAMATH_GPT_smallest_b_for_quadratic_factors_l2129_212981


namespace NUMINAMATH_GPT_ratio_of_speeds_l2129_212969

theorem ratio_of_speeds (va vb L : ℝ) (h1 : 0 < L) (h2 : 0 < va) (h3 : 0 < vb)
  (h4 : ∀ t : ℝ, t = L / va ↔ t = (L - 0.09523809523809523 * L) / vb) :
  va / vb = 21 / 19 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_speeds_l2129_212969


namespace NUMINAMATH_GPT_smallest_nine_ten_eleven_consecutive_sum_l2129_212979

theorem smallest_nine_ten_eleven_consecutive_sum :
  ∃ n : ℕ, n > 0 ∧ (n % 9 = 0) ∧ (n % 10 = 5) ∧ (n % 11 = 0) ∧ n = 495 :=
by {
  sorry
}

end NUMINAMATH_GPT_smallest_nine_ten_eleven_consecutive_sum_l2129_212979


namespace NUMINAMATH_GPT_total_estate_value_l2129_212962

theorem total_estate_value 
  (estate : ℝ)
  (daughter_share son_share wife_share brother_share nanny_share : ℝ)
  (h1 : daughter_share + son_share = (3/5) * estate)
  (h2 : daughter_share = 5 * son_share / 2)
  (h3 : wife_share = 3 * son_share)
  (h4 : brother_share = daughter_share)
  (h5 : nanny_share = 400) :
  estate = 825 := by
  sorry

end NUMINAMATH_GPT_total_estate_value_l2129_212962


namespace NUMINAMATH_GPT_valerie_money_left_l2129_212975

theorem valerie_money_left
  (small_bulb_cost : ℕ)
  (large_bulb_cost : ℕ)
  (num_small_bulbs : ℕ)
  (num_large_bulbs : ℕ)
  (initial_money : ℕ) :
  small_bulb_cost = 8 →
  large_bulb_cost = 12 →
  num_small_bulbs = 3 →
  num_large_bulbs = 1 →
  initial_money = 60 →
  initial_money - (num_small_bulbs * small_bulb_cost + num_large_bulbs * large_bulb_cost) = 24 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_valerie_money_left_l2129_212975


namespace NUMINAMATH_GPT_find_three_digit_number_l2129_212944

theorem find_three_digit_number (a b c : ℕ) (h1 : a + b + c = 16)
    (h2 : 100 * b + 10 * a + c = 100 * a + 10 * b + c - 360)
    (h3 : 100 * a + 10 * c + b = 100 * a + 10 * b + c + 54) :
    100 * a + 10 * b + c = 628 :=
by
  sorry

end NUMINAMATH_GPT_find_three_digit_number_l2129_212944


namespace NUMINAMATH_GPT_geometric_sequence_a4_a5_sum_l2129_212990

theorem geometric_sequence_a4_a5_sum :
  (∀ n : ℕ, a_n > 0) → (a_3 = 3) → (a_6 = (1 / 9)) → 
  (a_4 + a_5 = (4 / 3)) :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_a4_a5_sum_l2129_212990


namespace NUMINAMATH_GPT_dave_initial_apps_l2129_212935

theorem dave_initial_apps (x : ℕ) (h1 : x - 18 = 5) : x = 23 :=
by {
  -- This is where the proof would go 
  sorry -- The proof is omitted as per instructions
}

end NUMINAMATH_GPT_dave_initial_apps_l2129_212935


namespace NUMINAMATH_GPT_katya_sold_glasses_l2129_212974

-- Definitions based on the conditions specified in the problem
def ricky_sales : ℕ := 9

def tina_sales (K : ℕ) : ℕ := 2 * (K + ricky_sales)

def katya_sales_eq (K : ℕ) : Prop := tina_sales K = K + 26

-- Lean statement to prove Katya sold 8 glasses of lemonade
theorem katya_sold_glasses : ∃ (K : ℕ), katya_sales_eq K ∧ K = 8 :=
by
  sorry

end NUMINAMATH_GPT_katya_sold_glasses_l2129_212974


namespace NUMINAMATH_GPT_collinear_points_x_value_l2129_212995

theorem collinear_points_x_value
  (x : ℝ)
  (h : ∃ m : ℝ, m = (1 - (-4)) / (-1 - 2) ∧ m = (-9 - (-4)) / (x - 2)) :
  x = 5 :=
by
  sorry

end NUMINAMATH_GPT_collinear_points_x_value_l2129_212995


namespace NUMINAMATH_GPT_prime_pairs_divisibility_l2129_212931

theorem prime_pairs_divisibility (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) :
  (p * q) ∣ (p ^ p + q ^ q + 1) ↔ (p = 2 ∧ q = 5) ∨ (p = 5 ∧ q = 2) :=
by
  sorry

end NUMINAMATH_GPT_prime_pairs_divisibility_l2129_212931


namespace NUMINAMATH_GPT_Isabella_speed_is_correct_l2129_212980

-- Definitions based on conditions
def distance_km : ℝ := 17.138
def time_s : ℝ := 38

-- Conversion factor
def conversion_factor : ℝ := 1000

-- Distance in meters
def distance_m : ℝ := distance_km * conversion_factor

-- Correct answer (speed in m/s)
def correct_speed : ℝ := 451

-- Statement to prove
theorem Isabella_speed_is_correct : distance_m / time_s = correct_speed :=
by
  sorry

end NUMINAMATH_GPT_Isabella_speed_is_correct_l2129_212980


namespace NUMINAMATH_GPT_matthews_annual_income_l2129_212930

noncomputable def annual_income (q : ℝ) (I : ℝ) (T : ℝ) : Prop :=
  T = 0.01 * q * 50000 + 0.01 * (q + 3) * (I - 50000) ∧
  T = 0.01 * (q + 0.5) * I → I = 60000

-- Statement of the math proof
theorem matthews_annual_income (q : ℝ) (T : ℝ) :
  ∃ I : ℝ, I = 60000 ∧ annual_income q I T :=
sorry

end NUMINAMATH_GPT_matthews_annual_income_l2129_212930


namespace NUMINAMATH_GPT_larger_triangle_side_length_l2129_212952

theorem larger_triangle_side_length
    (A1 A2 : ℕ) (k : ℤ)
    (h1 : A1 - A2 = 32)
    (h2 : A1 = k^2 * A2)
    (h3 : A2 = 4 ∨ A2 = 8 ∨ A2 = 16)
    (h4 : ((4 : ℤ) * k = 12)) :
    (4 * k) = 12 :=
by sorry

end NUMINAMATH_GPT_larger_triangle_side_length_l2129_212952


namespace NUMINAMATH_GPT_largest_b_value_l2129_212924

open Real

structure Triangle :=
(side_a side_b side_c : ℝ)
(a_pos : 0 < side_a)
(b_pos : 0 < side_b)
(c_pos : 0 < side_c)
(tri_ineq_a : side_a + side_b > side_c)
(tri_ineq_b : side_b + side_c > side_a)
(tri_ineq_c : side_c + side_a > side_b)

noncomputable def inradius (T : Triangle) : ℝ :=
  let s := (T.side_a + T.side_b + T.side_c) / 2
  let A := sqrt (s * (s - T.side_a) * (s - T.side_b) * (s - T.side_c))
  A / s

noncomputable def circumradius (T : Triangle) : ℝ :=
  let A := sqrt (((T.side_a + T.side_b + T.side_c) / 2) * ((T.side_a + T.side_b + T.side_c) / 2 - T.side_a) * ((T.side_a + T.side_b + T.side_c) / 2 - T.side_b) * ((T.side_a + T.side_b + T.side_c) / 2 - T.side_c))
  (T.side_a * T.side_b * T.side_c) / (4 * A)

noncomputable def condition_met (T1 T2 : Triangle) : Prop :=
  (inradius T1 / circumradius T1) = (inradius T2 / circumradius T2)

theorem largest_b_value :
  let T1 := Triangle.mk 8 11 11 (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
  ∃ b > 0, ∃ T2 : Triangle, T2.side_a = b ∧ T2.side_b = 1 ∧ T2.side_c = 1 ∧ b = 14 / 11 ∧ condition_met T1 T2 :=
  sorry

end NUMINAMATH_GPT_largest_b_value_l2129_212924


namespace NUMINAMATH_GPT_students_taking_history_but_not_statistics_l2129_212968

theorem students_taking_history_but_not_statistics :
  ∀ (total_students history_students statistics_students history_or_statistics_both : ℕ),
    total_students = 90 →
    history_students = 36 →
    statistics_students = 32 →
    history_or_statistics_both = 57 →
    history_students - (history_students + statistics_students - history_or_statistics_both) = 25 :=
by intros; sorry

end NUMINAMATH_GPT_students_taking_history_but_not_statistics_l2129_212968


namespace NUMINAMATH_GPT_find_cos_E_floor_l2129_212905

theorem find_cos_E_floor (EF GH EH FG : ℝ) (E G : ℝ) 
  (h1 : EF = 200) 
  (h2 : GH = 200) 
  (h3 : EH ≠ FG) 
  (h4 : EF + GH + EH + FG = 800) 
  (h5 : E = G) : 
  (⌊1000 * Real.cos E⌋ = 1000) := 
by 
  sorry

end NUMINAMATH_GPT_find_cos_E_floor_l2129_212905


namespace NUMINAMATH_GPT_minimum_square_area_l2129_212946

-- Definitions of the given conditions
structure Rectangle where
  width : ℕ
  height : ℕ

def rect1 : Rectangle := { width := 2, height := 4 }
def rect2 : Rectangle := { width := 3, height := 5 }
def circle_diameter : ℕ := 3

-- Statement of the theorem
theorem minimum_square_area :
  ∃ sq_side : ℕ, 
    (sq_side ≥ 5 ∧ sq_side ≥ 7) ∧ 
    sq_side * sq_side = 49 := 
by
  use 7
  have h1 : 7 ≥ 5 := by norm_num
  have h2 : 7 ≥ 7 := by norm_num
  have h3 : 7 * 7 = 49 := by norm_num
  exact ⟨⟨h1, h2⟩, h3⟩

end NUMINAMATH_GPT_minimum_square_area_l2129_212946


namespace NUMINAMATH_GPT_hockey_league_total_games_l2129_212945

theorem hockey_league_total_games 
  (divisions : ℕ)
  (teams_per_division : ℕ)
  (intra_division_games : ℕ)
  (inter_division_games : ℕ) :
  divisions = 2 →
  teams_per_division = 6 →
  intra_division_games = 4 →
  inter_division_games = 2 →
  (divisions * ((teams_per_division * (teams_per_division - 1)) / 2) * intra_division_games) + 
  ((divisions / 2) * (divisions / 2) * teams_per_division * teams_per_division * inter_division_games) = 192 :=
by
  intros h_div h_teams h_intra h_inter
  sorry

end NUMINAMATH_GPT_hockey_league_total_games_l2129_212945


namespace NUMINAMATH_GPT_jessica_allowance_l2129_212943

theorem jessica_allowance (A : ℝ) (h1 : A / 2 + 6 = 11) : A = 10 := by
  sorry

end NUMINAMATH_GPT_jessica_allowance_l2129_212943


namespace NUMINAMATH_GPT_ways_to_distribute_balls_in_boxes_l2129_212938

theorem ways_to_distribute_balls_in_boxes :
  ∃ (num_ways : ℕ), num_ways = 4 ^ 5 := sorry

end NUMINAMATH_GPT_ways_to_distribute_balls_in_boxes_l2129_212938


namespace NUMINAMATH_GPT_dogs_Carly_worked_on_l2129_212929

-- Define the parameters for the problem
def total_nails := 164
def three_legged_dogs := 3
def three_nail_paw_dogs := 2
def extra_nail_paw_dog := 1
def regular_dog_nails := 16
def three_legged_nails := (regular_dog_nails - 4)
def three_nail_paw_nails := (regular_dog_nails - 1)
def extra_nail_paw_nails := (regular_dog_nails + 1)

-- Lean statement to prove the number of dogs Carly worked on today
theorem dogs_Carly_worked_on :
  (3 * three_legged_nails) + (2 * three_nail_paw_nails) + extra_nail_paw_nails 
  = 83 → ((total_nails - 83) / regular_dog_nails ≠ 0) → 5 + 3 + 2 + 1 = 11 :=
by sorry

end NUMINAMATH_GPT_dogs_Carly_worked_on_l2129_212929


namespace NUMINAMATH_GPT_distance_to_hospital_l2129_212916

theorem distance_to_hospital {total_paid base_price price_per_mile : ℝ} (h1 : total_paid = 23) (h2 : base_price = 3) (h3 : price_per_mile = 4) : (total_paid - base_price) / price_per_mile = 5 :=
by
  sorry

end NUMINAMATH_GPT_distance_to_hospital_l2129_212916


namespace NUMINAMATH_GPT_mike_pens_given_l2129_212966

noncomputable def pens_remaining (initial_pens mike_pens : ℕ) : ℕ :=
  2 * (initial_pens + mike_pens) - 19

theorem mike_pens_given 
  (initial_pens : ℕ)
  (mike_pens final_pens : ℕ) 
  (H1 : initial_pens = 7)
  (H2 : final_pens = 39) 
  (H3 : pens_remaining initial_pens mike_pens = final_pens) : 
  mike_pens = 22 := sorry

end NUMINAMATH_GPT_mike_pens_given_l2129_212966


namespace NUMINAMATH_GPT_relationship_between_x_and_y_l2129_212972

theorem relationship_between_x_and_y (m x y : ℝ) (h1 : x = 3 - m) (h2 : y = 2 * m + 1) : 2 * x + y = 7 :=
sorry

end NUMINAMATH_GPT_relationship_between_x_and_y_l2129_212972


namespace NUMINAMATH_GPT_inequality_correctness_l2129_212915

variable (a b : ℝ)
variable (h1 : a < b) (h2 : b < 0)

theorem inequality_correctness : a^2 > ab ∧ ab > b^2 := by
  sorry

end NUMINAMATH_GPT_inequality_correctness_l2129_212915


namespace NUMINAMATH_GPT_find_a1_in_arithmetic_sequence_l2129_212903

theorem find_a1_in_arithmetic_sequence (d n a_n : ℤ) (h_d : d = 2) (h_n : n = 15) (h_a_n : a_n = -10) :
  ∃ a1 : ℤ, a1 = -38 :=
by
  sorry

end NUMINAMATH_GPT_find_a1_in_arithmetic_sequence_l2129_212903


namespace NUMINAMATH_GPT_tan_half_angle_l2129_212977

-- Definition for the given angle in the third quadrant with a given sine value
def angle_in_third_quadrant_and_sin (α : ℝ) (hα : π < α ∧ α < 3 * π / 2) (h_sin : Real.sin α = -24 / 25) : Prop :=
  True

-- The main theorem to prove the given condition implies the result
theorem tan_half_angle (α : ℝ) (hα : π < α ∧ α < 3 * π / 2) (h_sin : Real.sin α = -24 / 25) :
  Real.tan (α / 2) = -4 / 3 :=
by
  sorry

end NUMINAMATH_GPT_tan_half_angle_l2129_212977


namespace NUMINAMATH_GPT_geometric_sequence_arithmetic_median_l2129_212958

theorem geometric_sequence_arithmetic_median 
  (a : ℕ → ℝ) 
  (hpos : ∀ n, 0 < a n) 
  (h_arith : 2 * a 1 + a 2 = 2 * a 3) :
  (a 2017 + a 2016) / (a 2015 + a 2014) = 4 :=
sorry

end NUMINAMATH_GPT_geometric_sequence_arithmetic_median_l2129_212958


namespace NUMINAMATH_GPT_total_cost_calculation_l2129_212967

def total_transportation_cost (x : ℝ) : ℝ :=
  let cost_A_to_C := 20 * x
  let cost_A_to_D := 30 * (240 - x)
  let cost_B_to_C := 24 * (200 - x)
  let cost_B_to_D := 32 * (60 + x)
  cost_A_to_C + cost_A_to_D + cost_B_to_C + cost_B_to_D

theorem total_cost_calculation (x : ℝ) :
  total_transportation_cost x = 13920 - 2 * x := by
  sorry

end NUMINAMATH_GPT_total_cost_calculation_l2129_212967


namespace NUMINAMATH_GPT_problem_pf_qf_geq_f_pq_l2129_212996

variable {R : Type*} [LinearOrderedField R]

theorem problem_pf_qf_geq_f_pq (f : R → R) (a b p q x y : R) (hpq : p + q = 1) :
  (∀ x y, p * f x + q * f y ≥ f (p * x + q * y)) ↔ (0 ≤ p ∧ p ≤ 1) := 
by
  sorry

end NUMINAMATH_GPT_problem_pf_qf_geq_f_pq_l2129_212996


namespace NUMINAMATH_GPT_trapezium_area_l2129_212937

theorem trapezium_area (a b h : ℝ) (ha : a = 20) (hb : b = 18) (hh : h = 17) : 
  (1 / 2 * (a + b) * h) = 323 :=
by
  have ha' : a = 20 := ha
  have hb' : b = 18 := hb
  have hh' : h = 17 := hh
  rw [ha', hb', hh']
  sorry

end NUMINAMATH_GPT_trapezium_area_l2129_212937


namespace NUMINAMATH_GPT_sum_of_prime_factors_eq_28_l2129_212957

-- Define 2310 as a constant
def n : ℕ := 2310

-- Define the prime factors of 2310
def prime_factors : List ℕ := [2, 3, 5, 7, 11]

-- The sum of the prime factors
def sum_prime_factors : ℕ := prime_factors.sum

-- State the theorem
theorem sum_of_prime_factors_eq_28 : sum_prime_factors = 28 :=
by 
  sorry

end NUMINAMATH_GPT_sum_of_prime_factors_eq_28_l2129_212957
