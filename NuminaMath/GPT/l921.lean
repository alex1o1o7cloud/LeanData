import Mathlib

namespace NUMINAMATH_GPT_bread_last_days_is_3_l921_92131

-- Define conditions
def num_members : ℕ := 4
def slices_breakfast : ℕ := 3
def slices_snacks : ℕ := 2
def slices_loaf : ℕ := 12
def num_loaves : ℕ := 5

-- Define the problem statement
def bread_last_days : ℕ :=
  (num_loaves * slices_loaf) / (num_members * (slices_breakfast + slices_snacks))

-- State the theorem to be proved
theorem bread_last_days_is_3 : bread_last_days = 3 :=
  sorry

end NUMINAMATH_GPT_bread_last_days_is_3_l921_92131


namespace NUMINAMATH_GPT_largest_possible_n_l921_92100

open Nat

-- Define arithmetic sequences a_n and b_n with given initial conditions
def arithmetic_seq (a_n : ℕ → ℕ) (b_n : ℕ → ℕ) :=
  a_n 1 = 1 ∧ b_n 1 = 1 ∧ 
  a_n 2 ≤ b_n 2 ∧
  (∃n : ℕ, a_n n * b_n n = 1764)

-- Given the arithmetic sequences defined above, prove that the largest possible value of n is 44
theorem largest_possible_n : 
  ∀ (a_n b_n : ℕ → ℕ), arithmetic_seq a_n b_n →
  ∀ (n : ℕ), (a_n n * b_n n = 1764) → n ≤ 44 :=
sorry

end NUMINAMATH_GPT_largest_possible_n_l921_92100


namespace NUMINAMATH_GPT_inequality_1_inequality_2_inequality_3_inequality_4_l921_92151

noncomputable def triangle_angles (a b c : ℝ) : Prop :=
  a + b + c = Real.pi

theorem inequality_1 (a b c : ℝ) (h : triangle_angles a b c) :
  Real.sin a + Real.sin b + Real.sin c ≤ (3 * Real.sqrt 3 / 2) :=
sorry

theorem inequality_2 (a b c : ℝ) (h : triangle_angles a b c) :
  Real.cos (a / 2) + Real.cos (b / 2) + Real.cos (c / 2) ≤ (3 * Real.sqrt 3 / 2) :=
sorry

theorem inequality_3 (a b c : ℝ) (h : triangle_angles a b c) :
  Real.cos a * Real.cos b * Real.cos c ≤ (1 / 8) :=
sorry

theorem inequality_4 (a b c : ℝ) (h : triangle_angles a b c) :
  Real.sin (2 * a) + Real.sin (2 * b) + Real.sin (2 * c) ≤ Real.sin a + Real.sin b + Real.sin c :=
sorry

end NUMINAMATH_GPT_inequality_1_inequality_2_inequality_3_inequality_4_l921_92151


namespace NUMINAMATH_GPT_ratio_doubled_to_original_l921_92106

theorem ratio_doubled_to_original (x y : ℕ) (h1 : y = 2 * x + 9) (h2 : 3 * y = 57) : 2 * x = 2 * (x / 1) := 
by sorry

end NUMINAMATH_GPT_ratio_doubled_to_original_l921_92106


namespace NUMINAMATH_GPT_smallest_percent_increase_l921_92102

-- Define the values of each question.
def value (n : ℕ) : ℕ :=
  match n with
  | 1  => 150
  | 2  => 300
  | 3  => 450
  | 4  => 600
  | 5  => 800
  | 6  => 1500
  | 7  => 3000
  | 8  => 6000
  | 9  => 12000
  | 10 => 24000
  | 11 => 48000
  | 12 => 96000
  | 13 => 192000
  | 14 => 384000
  | 15 => 768000
  | _ => 0

-- Define the percent increase between two values.
def percent_increase (v1 v2 : ℕ) : ℚ :=
  ((v2 - v1 : ℕ) : ℚ) / v1 * 100 

-- Prove that the smallest percent increase is between question 4 and 5.
theorem smallest_percent_increase :
  percent_increase (value 4) (value 5) = 33.33 := 
by
  sorry

end NUMINAMATH_GPT_smallest_percent_increase_l921_92102


namespace NUMINAMATH_GPT_points_on_parabola_l921_92111

theorem points_on_parabola (t : ℝ) : 
  ∃ a b c : ℝ, ∀ (x y: ℝ), (x, y) = (Real.cos t ^ 2, Real.sin (2 * t)) → y^2 = 4 * x - 4 * x^2 := 
by
  sorry

end NUMINAMATH_GPT_points_on_parabola_l921_92111


namespace NUMINAMATH_GPT_find_z_l921_92104

variable {x y z : ℝ}

theorem find_z (h : (1/x + 1/y = 1/z)) : z = (x * y) / (x + y) :=
  sorry

end NUMINAMATH_GPT_find_z_l921_92104


namespace NUMINAMATH_GPT_rectangle_circle_ratio_l921_92157

theorem rectangle_circle_ratio (r s : ℝ) (h : ∀ r s : ℝ, 2 * r * s - π * r^2 = π * r^2) : s / (2 * r) = π / 2 :=
by
  sorry

end NUMINAMATH_GPT_rectangle_circle_ratio_l921_92157


namespace NUMINAMATH_GPT_percentage_decrease_is_24_l921_92133

-- Define the given constants Rs. 820 and Rs. 1078.95
def current_price : ℝ := 820
def original_price : ℝ := 1078.95

-- Define the percentage decrease P
def percentage_decrease (P : ℝ) : Prop :=
  original_price - (P / 100) * original_price = current_price

-- Prove that percentage decrease P is approximately 24
theorem percentage_decrease_is_24 : percentage_decrease 24 :=
by
  unfold percentage_decrease
  sorry

end NUMINAMATH_GPT_percentage_decrease_is_24_l921_92133


namespace NUMINAMATH_GPT_product_is_zero_l921_92170

def product_series (a : ℤ) : ℤ :=
  (a - 12) * (a - 11) * (a - 10) * (a - 9) * (a - 8) * (a - 7) * (a - 6) * (a - 5) * 
  (a - 4) * (a - 3) * (a - 2) * (a - 1) * a

theorem product_is_zero : product_series 3 = 0 :=
by
  sorry

end NUMINAMATH_GPT_product_is_zero_l921_92170


namespace NUMINAMATH_GPT_no_solutions_for_sin_cos_eq_sqrt3_l921_92143

theorem no_solutions_for_sin_cos_eq_sqrt3 (x : ℝ) (hx : 0 ≤ x ∧ x < 2 * Real.pi) :
  ¬ (Real.sin x + Real.cos x = Real.sqrt 3) :=
by
  sorry

end NUMINAMATH_GPT_no_solutions_for_sin_cos_eq_sqrt3_l921_92143


namespace NUMINAMATH_GPT_bonnets_per_orphanage_correct_l921_92146

-- Definitions for each day's bonnet count
def monday_bonnets := 10
def tuesday_and_wednesday_bonnets := 2 * monday_bonnets
def thursday_bonnets := monday_bonnets + 5
def friday_bonnets := thursday_bonnets - 5
def saturday_bonnets := friday_bonnets - 8
def sunday_bonnets := 3 * saturday_bonnets

-- Total bonnets made in the week
def total_bonnets := 
  monday_bonnets +
  tuesday_and_wednesday_bonnets +
  thursday_bonnets +
  friday_bonnets +
  saturday_bonnets +
  sunday_bonnets

-- The number of orphanages
def orphanages := 10

-- Bonnets sent to each orphanage
def bonnets_per_orphanage := total_bonnets / orphanages

theorem bonnets_per_orphanage_correct :
  bonnets_per_orphanage = 6 :=
by
  sorry

end NUMINAMATH_GPT_bonnets_per_orphanage_correct_l921_92146


namespace NUMINAMATH_GPT_min_value_mn_squared_l921_92160

theorem min_value_mn_squared (a b c m n : ℝ) 
  (h_triangle: a^2 + b^2 = c^2)
  (h_line: a * m + b * n + 2 * c = 0):
  m^2 + n^2 = 4 :=
by
  sorry

end NUMINAMATH_GPT_min_value_mn_squared_l921_92160


namespace NUMINAMATH_GPT_six_x_mod_nine_l921_92166

theorem six_x_mod_nine (x : ℕ) (k : ℕ) (hx : x = 9 * k + 5) : (6 * x) % 9 = 3 :=
by
  sorry

end NUMINAMATH_GPT_six_x_mod_nine_l921_92166


namespace NUMINAMATH_GPT_sum_xyz_l921_92182

theorem sum_xyz (x y z : ℝ) (h : (x - 2)^2 + (y - 3)^2 + (z - 6)^2 = 0) : x + y + z = 11 := 
by
  sorry

end NUMINAMATH_GPT_sum_xyz_l921_92182


namespace NUMINAMATH_GPT_trapezium_area_l921_92167

theorem trapezium_area (a b h : ℝ) (ha : a = 20) (hb : b = 18) (hh : h = 15) : 
  1/2 * (a + b) * h = 285 :=
by {
  sorry
}

end NUMINAMATH_GPT_trapezium_area_l921_92167


namespace NUMINAMATH_GPT_expression_evaluation_valid_l921_92115

theorem expression_evaluation_valid (a : ℝ) (h1 : a = 4) :
  (1 + (4 / (a ^ 2 - 4))) * ((a + 2) / a) = 2 := by
  sorry

end NUMINAMATH_GPT_expression_evaluation_valid_l921_92115


namespace NUMINAMATH_GPT_total_amount_paid_l921_92108

def p1 := 20
def p2 := p1 + 2
def p3 := p2 + 3
def p4 := p3 + 4

theorem total_amount_paid : p1 + p2 + p3 + p4 = 96 :=
by
  sorry

end NUMINAMATH_GPT_total_amount_paid_l921_92108


namespace NUMINAMATH_GPT_remainder_when_divided_by_13_l921_92130

theorem remainder_when_divided_by_13 (N : ℤ) (k : ℤ) (h : N = 39 * k + 17) : 
  N % 13 = 4 :=
by
  sorry

end NUMINAMATH_GPT_remainder_when_divided_by_13_l921_92130


namespace NUMINAMATH_GPT_sum_is_correct_l921_92148

-- Define the variables and conditions
variables (a b c d : ℝ)
variable (x : ℝ)

-- Define the condition
def condition : Prop :=
  a + 1 = x ∧
  b + 2 = x ∧
  c + 3 = x ∧
  d + 4 = x ∧
  a + b + c + d + 5 = x

-- The theorem we need to prove
theorem sum_is_correct (h : condition a b c d x) : a + b + c + d = -10 / 3 :=
  sorry

end NUMINAMATH_GPT_sum_is_correct_l921_92148


namespace NUMINAMATH_GPT_find_divisible_by_3_l921_92190

theorem find_divisible_by_3 (n : ℕ) : 
  (∀ k : ℕ, k ≤ 12 → (3 * k + 12) ≤ n) ∧ 
  (∀ m : ℕ, m ≥ 13 → (3 * m + 12) > n) →
  n = 48 :=
by
  sorry

end NUMINAMATH_GPT_find_divisible_by_3_l921_92190


namespace NUMINAMATH_GPT_remaining_amount_division_l921_92199

-- Definitions
def total_amount : ℕ := 2100
def number_of_participants : ℕ := 8
def amount_already_raised : ℕ := 150

-- Proof problem statement
theorem remaining_amount_division :
  (total_amount - amount_already_raised) / (number_of_participants - 1) = 279 :=
by
  sorry

end NUMINAMATH_GPT_remaining_amount_division_l921_92199


namespace NUMINAMATH_GPT_inequality_solution_exists_l921_92172

theorem inequality_solution_exists (a : ℝ) : 
  ∃ x : ℝ, x > 2 ∧ x > -1 ∧ x > a := 
by
  sorry

end NUMINAMATH_GPT_inequality_solution_exists_l921_92172


namespace NUMINAMATH_GPT_DiagonalsOfShapesBisectEachOther_l921_92153

structure Shape where
  bisect_diagonals : Prop

def is_parallelogram (s : Shape) : Prop := s.bisect_diagonals
def is_rectangle (s : Shape) : Prop := s.bisect_diagonals
def is_rhombus (s : Shape) : Prop := s.bisect_diagonals
def is_square (s : Shape) : Prop := s.bisect_diagonals

theorem DiagonalsOfShapesBisectEachOther (s : Shape) :
  is_parallelogram s ∨ is_rectangle s ∨ is_rhombus s ∨ is_square s → s.bisect_diagonals := by
  sorry

end NUMINAMATH_GPT_DiagonalsOfShapesBisectEachOther_l921_92153


namespace NUMINAMATH_GPT_parallel_lines_condition_l921_92117

theorem parallel_lines_condition (a : ℝ) : 
  (∀ x y : ℝ, ax + y + 1 = 0 ↔ x + ay - 1 = 0) ↔ (a = 1) :=
sorry

end NUMINAMATH_GPT_parallel_lines_condition_l921_92117


namespace NUMINAMATH_GPT_solve_for_x_in_equation_l921_92110

theorem solve_for_x_in_equation (x : ℝ)
  (h : (2 / 7) * (1 / 4) * x = 12) : x = 168 :=
sorry

end NUMINAMATH_GPT_solve_for_x_in_equation_l921_92110


namespace NUMINAMATH_GPT_problem_l921_92142

def op (x y : ℝ) : ℝ := x^2 + y^3

theorem problem (k : ℝ) : op k (op k k) = k^2 + k^6 + 6*k^7 + k^9 :=
by
  sorry

end NUMINAMATH_GPT_problem_l921_92142


namespace NUMINAMATH_GPT_total_boxes_count_l921_92154

theorem total_boxes_count 
    (apples_per_crate : ℕ) (apples_crates : ℕ) 
    (oranges_per_crate : ℕ) (oranges_crates : ℕ) 
    (bananas_per_crate : ℕ) (bananas_crates : ℕ) 
    (rotten_apples_percentage : ℝ) (rotten_oranges_percentage : ℝ) (rotten_bananas_percentage : ℝ)
    (apples_per_box : ℕ) (oranges_per_box : ℕ) (bananas_per_box : ℕ) :
    apples_per_crate = 42 → apples_crates = 12 → 
    oranges_per_crate = 36 → oranges_crates = 15 → 
    bananas_per_crate = 30 → bananas_crates = 18 → 
    rotten_apples_percentage = 0.08 → rotten_oranges_percentage = 0.05 → rotten_bananas_percentage = 0.02 →
    apples_per_box = 10 → oranges_per_box = 12 → bananas_per_box = 15 →
    ∃ total_boxes : ℕ, total_boxes = 126 :=
by sorry

end NUMINAMATH_GPT_total_boxes_count_l921_92154


namespace NUMINAMATH_GPT_product_of_consecutive_integers_l921_92147

theorem product_of_consecutive_integers (l : List ℤ) (h1 : l.length = 2019) (h2 : l.sum = 2019) : l.prod = 0 := 
sorry

end NUMINAMATH_GPT_product_of_consecutive_integers_l921_92147


namespace NUMINAMATH_GPT_factorize_expr_l921_92113

theorem factorize_expr (y : ℝ) : 3 * y ^ 2 - 6 * y + 3 = 3 * (y - 1) ^ 2 :=
by
  sorry

end NUMINAMATH_GPT_factorize_expr_l921_92113


namespace NUMINAMATH_GPT_sequence_conjecture_l921_92180

theorem sequence_conjecture (a : ℕ → ℚ) (h₁ : a 1 = 1) (h₂ : ∀ n, a (n + 1) = a n / (a n + 1)) :
  ∀ n : ℕ, 0 < n → a n = 1 / n := by
  sorry

end NUMINAMATH_GPT_sequence_conjecture_l921_92180


namespace NUMINAMATH_GPT_cows_on_farm_l921_92137

theorem cows_on_farm (weekly_production_per_6_cows : ℕ) 
                     (production_over_5_weeks : ℕ) 
                     (number_of_weeks : ℕ) 
                     (cows : ℕ) :
  weekly_production_per_6_cows = 108 →
  production_over_5_weeks = 2160 →
  number_of_weeks = 5 →
  (cows * (weekly_production_per_6_cows / 6) * number_of_weeks = production_over_5_weeks) →
  cows = 24 :=
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_cows_on_farm_l921_92137


namespace NUMINAMATH_GPT_polynomial_value_l921_92125

theorem polynomial_value (a b : ℝ) : 
  (|a - 2| + (b + 1/2)^2 = 0) → (2 * a * b^2 + a^2 * b) - (3 * a * b^2 + a^2 * b - 1) = 1/2 :=
by
  sorry

end NUMINAMATH_GPT_polynomial_value_l921_92125


namespace NUMINAMATH_GPT_tan_x_tan_y_relation_l921_92192

/-- If 
  (sin x / cos y) + (sin y / cos x) = 2 
  and 
  (cos x / sin y) + (cos y / sin x) = 3, 
  then 
  (tan x / tan y) + (tan y / tan x) = 16 / 3.
 -/
theorem tan_x_tan_y_relation (x y : ℝ)
  (h1 : (Real.sin x / Real.cos y) + (Real.sin y / Real.cos x) = 2)
  (h2 : (Real.cos x / Real.sin y) + (Real.cos y / Real.sin x) = 3) :
  (Real.tan x / Real.tan y) + (Real.tan y / Real.tan x) = 16 / 3 :=
sorry

end NUMINAMATH_GPT_tan_x_tan_y_relation_l921_92192


namespace NUMINAMATH_GPT_simplify_expression_l921_92122

theorem simplify_expression (x : ℝ) : (3 * x + 30) + (150 * x - 45) = 153 * x - 15 :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l921_92122


namespace NUMINAMATH_GPT_maximum_value_of_x_minus_y_l921_92169

noncomputable def max_value_x_minus_y (x y : ℝ) (h : x^2 + y^2 - 4 * x - 2 * y - 4 = 0) : ℝ :=
1 + 3 * Real.sqrt 2

theorem maximum_value_of_x_minus_y (x y : ℝ) (h : x^2 + y^2 - 4 * x - 2 * y - 4 = 0) :
    x - y ≤ max_value_x_minus_y x y h := sorry

end NUMINAMATH_GPT_maximum_value_of_x_minus_y_l921_92169


namespace NUMINAMATH_GPT_factorization_correctness_l921_92164

theorem factorization_correctness :
  (∀ x, x^2 + 2 * x + 1 = (x + 1)^2) ∧
  ¬ (∀ x, x * (x + 1) = x^2 + x) ∧
  ¬ (∀ x y, x^2 + x * y - 3 = x * (x + y) - 3) ∧
  ¬ (∀ x, x^2 + 6 * x + 4 = (x + 3)^2 - 5) :=
by
  sorry

end NUMINAMATH_GPT_factorization_correctness_l921_92164


namespace NUMINAMATH_GPT_second_integer_is_64_l921_92109

theorem second_integer_is_64
  (n : ℤ)
  (h1 : (n - 2) + (n + 2) = 128) :
  n = 64 := 
  sorry

end NUMINAMATH_GPT_second_integer_is_64_l921_92109


namespace NUMINAMATH_GPT_complement_of_intersection_l921_92165

open Set

-- Define the universal set U
def U := @univ ℝ
-- Define the sets M and N
def M : Set ℝ := {x | x >= 2}
def N : Set ℝ := {x | 0 <= x ∧ x < 5}

-- Define M ∩ N
def M_inter_N := M ∩ N

-- Define the complement of M ∩ N with respect to U
def C_U (A : Set ℝ) := Aᶜ

theorem complement_of_intersection :
  C_U M_inter_N = {x : ℝ | x < 2 ∨ x ≥ 5} := 
by 
  sorry

end NUMINAMATH_GPT_complement_of_intersection_l921_92165


namespace NUMINAMATH_GPT_min_value_frac_sq_l921_92127

theorem min_value_frac_sq (x : ℝ) (h : x > 12) : (x^2 / (x - 12)) >= 48 :=
by
  sorry

end NUMINAMATH_GPT_min_value_frac_sq_l921_92127


namespace NUMINAMATH_GPT_jordan_oreos_l921_92163

def oreos (james jordan total : ℕ) : Prop :=
  james = 2 * jordan + 3 ∧
  jordan + james = total

theorem jordan_oreos (J : ℕ) (h : oreos (2 * J + 3) J 36) : J = 11 :=
by
  sorry

end NUMINAMATH_GPT_jordan_oreos_l921_92163


namespace NUMINAMATH_GPT_number_of_ordered_pairs_xy_2007_l921_92162

theorem number_of_ordered_pairs_xy_2007 : 
  ∃ n, n = 6 ∧ (∀ x y : ℕ, x * y = 2007 → x > 0 ∧ y > 0) :=
sorry

end NUMINAMATH_GPT_number_of_ordered_pairs_xy_2007_l921_92162


namespace NUMINAMATH_GPT_expression_in_terms_of_p_q_l921_92179

variables {α β γ δ p q : ℝ}

-- Let α and β be the roots of x^2 - 2px + 1 = 0
axiom root_α_β : ∀ x, (x - α) * (x - β) = x^2 - 2 * p * x + 1

-- Let γ and δ be the roots of x^2 + qx + 2 = 0
axiom root_γ_δ : ∀ x, (x - γ) * (x - δ) = x^2 + q * x + 2

-- Expression to be proved
theorem expression_in_terms_of_p_q :
  (α - γ) * (β - γ) * (α - δ) * (β - δ) = 2 * (p - q) ^ 2 :=
sorry

end NUMINAMATH_GPT_expression_in_terms_of_p_q_l921_92179


namespace NUMINAMATH_GPT_positive_integer_solution_l921_92149

theorem positive_integer_solution (x y : ℕ) (hx : 0 < x) (hy : 0 < y) (h : x^4 = y^2 + 71) :
  x = 6 ∧ y = 35 :=
by
  sorry

end NUMINAMATH_GPT_positive_integer_solution_l921_92149


namespace NUMINAMATH_GPT_transformed_curve_l921_92132

theorem transformed_curve :
  (∀ x y : ℝ, 3*x = x' ∧ 4*y = y' → x^2 + y^2 = 1) ↔ (x'^2 / 9 + y'^2 / 16 = 1) :=
by
  sorry

end NUMINAMATH_GPT_transformed_curve_l921_92132


namespace NUMINAMATH_GPT_surface_area_of_interior_box_l921_92187

def original_sheet_width : ℕ := 40
def original_sheet_length : ℕ := 50
def corner_cut_side : ℕ := 8
def corners_count : ℕ := 4

def area_of_original_sheet : ℕ := original_sheet_width * original_sheet_length
def area_of_one_corner_cut : ℕ := corner_cut_side * corner_cut_side
def total_area_removed : ℕ := corners_count * area_of_one_corner_cut
def area_of_remaining_sheet : ℕ := area_of_original_sheet - total_area_removed

theorem surface_area_of_interior_box : area_of_remaining_sheet = 1744 :=
by
  sorry

end NUMINAMATH_GPT_surface_area_of_interior_box_l921_92187


namespace NUMINAMATH_GPT_fraction_of_earth_habitable_l921_92138

theorem fraction_of_earth_habitable :
  ∀ (earth_surface land_area inhabitable_land_area : ℝ),
    land_area = 1 / 3 → 
    inhabitable_land_area = 1 / 4 → 
    (earth_surface * land_area * inhabitable_land_area) = 1 / 12 :=
  by
    intros earth_surface land_area inhabitable_land_area h_land h_inhabitable
    sorry

end NUMINAMATH_GPT_fraction_of_earth_habitable_l921_92138


namespace NUMINAMATH_GPT_find_n_l921_92173

theorem find_n 
  (a : ℝ := 9 / 15)
  (S1 : ℝ := 15 / (1 - a))
  (b : ℝ := (9 + n) / 15)
  (S2 : ℝ := 3 * S1)
  (hS1 : S1 = 37.5)
  (hS2 : S2 = 112.5)
  (hb : b = 13 / 15)
  (hn : 13 = 9 + n) : 
  n = 4 :=
by
  sorry

end NUMINAMATH_GPT_find_n_l921_92173


namespace NUMINAMATH_GPT_restore_original_price_l921_92107

def price_after_increases (p : ℝ) : ℝ :=
  let p1 := p * 1.10
  let p2 := p1 * 1.10
  let p3 := p2 * 1.05
  p3

theorem restore_original_price (p : ℝ) (h : p = 1) : 
  ∃ x : ℝ, x = 22 ∧ (price_after_increases p) * (1 - x / 100) = 1 := 
by 
  sorry

end NUMINAMATH_GPT_restore_original_price_l921_92107


namespace NUMINAMATH_GPT_min_value_of_f_l921_92129

noncomputable def f (x : ℝ) : ℝ := 3 * x + 12 / x ^ 2

theorem min_value_of_f : ∀ x > 0, f x ≥ 9 ∧ (f x = 9 ↔ x = 2) :=
by
  sorry

end NUMINAMATH_GPT_min_value_of_f_l921_92129


namespace NUMINAMATH_GPT_papi_calot_additional_plants_l921_92126

def initial_plants := 7 * 18

def total_plants := 141

def additional_plants := total_plants - initial_plants

theorem papi_calot_additional_plants : additional_plants = 15 :=
by
  sorry

end NUMINAMATH_GPT_papi_calot_additional_plants_l921_92126


namespace NUMINAMATH_GPT_find_digits_l921_92105

theorem find_digits (A B D E C : ℕ) 
  (hC : C = 9) 
  (hA : 2 < A ∧ A < 4)
  (hB : B = 5)
  (hE : E = 6)
  (hD : D = 0)
  (h_distinct : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧ B ≠ C ∧ B ≠ D ∧ B ≠ E ∧ C ≠ D ∧ C ≠ E ∧ D ≠ E) :
  (A, B, D, E) = (3, 5, 0, 6) := by
  sorry

end NUMINAMATH_GPT_find_digits_l921_92105


namespace NUMINAMATH_GPT_p_sufficient_but_not_necessary_for_q_l921_92181

def p (x : ℝ) : Prop := -1 ≤ x ∧ x ≤ 5
def q (x : ℝ) : Prop := (x - 5) * (x + 1) < 0

theorem p_sufficient_but_not_necessary_for_q :
  (∀ x : ℝ, p x → q x) ∧ ∃ x : ℝ, q x ∧ ¬ p x :=
by
  sorry

end NUMINAMATH_GPT_p_sufficient_but_not_necessary_for_q_l921_92181


namespace NUMINAMATH_GPT_find_k_l921_92112

theorem find_k (x y k : ℝ)
  (h1 : 3 * x + 2 * y = k + 1)
  (h2 : 2 * x + 3 * y = k)
  (h3 : x + y = 3) : k = 7 := sorry

end NUMINAMATH_GPT_find_k_l921_92112


namespace NUMINAMATH_GPT_solve_fraction_equation_l921_92195

theorem solve_fraction_equation (x : ℝ) (h₁ : x ≠ 0) (h₂ : x ≠ -3) :
  (2 / x + x / (x + 3) = 1) ↔ x = 6 := 
by
  sorry

end NUMINAMATH_GPT_solve_fraction_equation_l921_92195


namespace NUMINAMATH_GPT_geometric_progression_solution_l921_92198

theorem geometric_progression_solution 
  (b1 q : ℝ)
  (condition1 : (b1^2 / (1 + q + q^2) = 48 / 7))
  (condition2 : (b1^2 / (1 + q^2) = 144 / 17)) 
  : (b1 = 3 ∨ b1 = -3) ∧ q = 1 / 4 :=
by
  sorry

end NUMINAMATH_GPT_geometric_progression_solution_l921_92198


namespace NUMINAMATH_GPT_three_digit_odd_nums_using_1_2_3_4_5_without_repetition_l921_92116

def three_digit_odd_nums (digits : Finset ℕ) : ℕ :=
  let odd_digits := digits.filter (λ n => n % 2 = 1)
  let num_choices_for_units_place := odd_digits.card
  let remaining_digits := digits \ odd_digits
  let num_choices_for_hundreds_tens_places := remaining_digits.card * (remaining_digits.card - 1)
  num_choices_for_units_place * num_choices_for_hundreds_tens_places

theorem three_digit_odd_nums_using_1_2_3_4_5_without_repetition :
  three_digit_odd_nums {1, 2, 3, 4, 5} = 36 :=
by
  -- Proof is skipped
  sorry

end NUMINAMATH_GPT_three_digit_odd_nums_using_1_2_3_4_5_without_repetition_l921_92116


namespace NUMINAMATH_GPT_fraction_of_Bhupathi_is_point4_l921_92194

def abhinav_and_bhupathi_amounts (A B : ℝ) : Prop :=
  A + B = 1210 ∧ B = 484

theorem fraction_of_Bhupathi_is_point4 (A B : ℝ) (x : ℝ) (h : abhinav_and_bhupathi_amounts A B) :
  (4 / 15) * A = x * B → x = 0.4 :=
by
  sorry

end NUMINAMATH_GPT_fraction_of_Bhupathi_is_point4_l921_92194


namespace NUMINAMATH_GPT_parrots_per_cage_l921_92119

theorem parrots_per_cage (total_birds : ℕ) (num_cages : ℕ) (parakeets_per_cage : ℕ) (total_parrots : ℕ) :
  total_birds = 48 → num_cages = 6 → parakeets_per_cage = 2 → total_parrots = 36 →
  ∀ P : ℕ, (total_parrots = P * num_cages) → P = 6 :=
by
  intros h1 h2 h3 h4 P h5
  subst h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_parrots_per_cage_l921_92119


namespace NUMINAMATH_GPT_relationship_y1_y2_l921_92156

theorem relationship_y1_y2
    (b : ℝ) 
    (y1 y2 : ℝ)
    (h1 : y1 = - (1 / 2) * (-2) + b) 
    (h2 : y2 = - (1 / 2) * 3 + b) : 
    y1 > y2 :=
sorry

end NUMINAMATH_GPT_relationship_y1_y2_l921_92156


namespace NUMINAMATH_GPT_num_solutions_system_eqns_l921_92189

theorem num_solutions_system_eqns :
  ∃ (c : ℕ), 
    (∀ (a1 a2 a3 a4 a5 a6 : ℕ), 
       a1 + 2 * a2 + 3 * a3 + 4 * a4 + 5 * a5 + 6 * a6 = 26 ∧ 
       a1 + a2 + a3 + a4 + a5 + a6 = 5 → 
       (a1, a2, a3, a4, a5, a6) ∈ (solutions : Finset (ℕ × ℕ × ℕ × ℕ × ℕ × ℕ))) ∧
    solutions.card = 5 := sorry

end NUMINAMATH_GPT_num_solutions_system_eqns_l921_92189


namespace NUMINAMATH_GPT_combined_salaries_l921_92178

theorem combined_salaries (A B C D E : ℝ) 
  (hC : C = 11000) 
  (hAverage : (A + B + C + D + E) / 5 = 8200) : 
  A + B + D + E = 30000 := 
by 
  sorry

end NUMINAMATH_GPT_combined_salaries_l921_92178


namespace NUMINAMATH_GPT_platform_length_calc_l921_92175

noncomputable def length_of_platform (V : ℝ) (T : ℝ) (L_train : ℝ) : ℝ :=
  (V * 1000 / 3600) * T - L_train

theorem platform_length_calc (speed : ℝ) (time : ℝ) (length_train : ℝ):
  speed = 72 →
  time = 26 →
  length_train = 280.0416 →
  length_of_platform speed time length_train = 239.9584 := by
  intros
  unfold length_of_platform
  sorry

end NUMINAMATH_GPT_platform_length_calc_l921_92175


namespace NUMINAMATH_GPT_inequality_proof_l921_92174

theorem inequality_proof (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a ≥ b) (h5 : b ≥ c) :
  a + b + c ≤ (a^2 + b^2) / (2 * c) + (b^2 + c^2) / (2 * a) + (c^2 + a^2) / (2 * b) ∧
  (a^2 + b^2) / (2 * c) + (b^2 + c^2) / (2 * a) + (c^2 + a^2) / (2 * b) ≤ (a^3 / (b * c)) + (b^3 / (c * a)) + (c^3 / (a * b)) :=
by
  sorry

end NUMINAMATH_GPT_inequality_proof_l921_92174


namespace NUMINAMATH_GPT_sum_of_areas_squares_l921_92183

theorem sum_of_areas_squares (a : ℝ) : 
  (∑' n : ℕ, (a^2 / 4^n)) = (4 * a^2 / 3) :=
by
  sorry

end NUMINAMATH_GPT_sum_of_areas_squares_l921_92183


namespace NUMINAMATH_GPT_triangle_inequality_l921_92141

-- Let α, β, γ be the angles of a triangle opposite to its sides with lengths a, b, and c, respectively.
variables (α β γ a b c : ℝ)

-- Assume that α, β, γ are positive.
axiom positive_angles : α > 0 ∧ β > 0 ∧ γ > 0
-- Assume that a, b, c are the sides opposite to angles α, β, γ respectively.
axiom positive_sides : a > 0 ∧ b > 0 ∧ c > 0

theorem triangle_inequality :
  a * (1 / β + 1 / γ) + b * (1 / γ + 1 / α) + c * (1 / α + 1 / β) ≥ 
  2 * (a / α + b / β + c / γ) :=
sorry

end NUMINAMATH_GPT_triangle_inequality_l921_92141


namespace NUMINAMATH_GPT_xu_jun_age_l921_92118

variable (x y : ℕ)

def condition1 : Prop := y - 2 = 3 * (x - 2)
def condition2 : Prop := y + 8 = 2 * (x + 8)

theorem xu_jun_age (h1 : condition1 x y) (h2 : condition2 x y) : x = 12 :=
by 
sorry

end NUMINAMATH_GPT_xu_jun_age_l921_92118


namespace NUMINAMATH_GPT_children_exceed_bridge_limit_l921_92197

theorem children_exceed_bridge_limit :
  ∀ (Kelly_weight : ℕ) (Megan_weight : ℕ) (Mike_weight : ℕ),
  Kelly_weight = 34 ∧
  Kelly_weight = (85 * Megan_weight) / 100 ∧
  Mike_weight = Megan_weight + 5 →
  Kelly_weight + Megan_weight + Mike_weight - 100 = 19 :=
by sorry

end NUMINAMATH_GPT_children_exceed_bridge_limit_l921_92197


namespace NUMINAMATH_GPT_total_cakes_correct_l921_92155

-- Define the initial number of full-size cakes
def initial_cakes : ℕ := 350

-- Define the number of additional full-size cakes made
def additional_cakes : ℕ := 125

-- Define the number of half-cakes made
def half_cakes : ℕ := 75

-- Convert half-cakes to full-size cakes, considering only whole cakes
def half_to_full_cakes := (half_cakes / 2)

-- Total full-size cakes calculation
def total_cakes :=
  initial_cakes + additional_cakes + half_to_full_cakes

-- Prove the total number of full-size cakes
theorem total_cakes_correct : total_cakes = 512 :=
by
  -- Skip the proof
  sorry

end NUMINAMATH_GPT_total_cakes_correct_l921_92155


namespace NUMINAMATH_GPT_apple_cost_calculation_l921_92186

theorem apple_cost_calculation
    (original_price : ℝ)
    (price_raise : ℝ)
    (amount_per_person : ℝ)
    (num_people : ℝ) :
  original_price = 1.6 →
  price_raise = 0.25 →
  amount_per_person = 2 →
  num_people = 4 →
  (num_people * amount_per_person * (original_price * (1 + price_raise))) = 16 :=
by
  -- insert the mathematical proof steps/cardinality here
  sorry

end NUMINAMATH_GPT_apple_cost_calculation_l921_92186


namespace NUMINAMATH_GPT_olympic_iberic_sets_containing_33_l921_92139

/-- A set of positive integers is iberic if it is a subset of {2, 3, ..., 2018},
    and whenever m, n are both in the set, gcd(m, n) is also in the set. -/
def is_iberic_set (X : Set ℕ) : Prop :=
  X ⊆ {n | 2 ≤ n ∧ n ≤ 2018} ∧ ∀ m n, m ∈ X → n ∈ X → Nat.gcd m n ∈ X

/-- An iberic set is olympic if it is not properly contained in any other iberic set. -/
def is_olympic_set (X : Set ℕ) : Prop :=
  is_iberic_set X ∧ ∀ Y, is_iberic_set Y → X ⊂ Y → False

/-- The olympic iberic sets containing 33 are exactly {3, 6, 9, ..., 2016} and {11, 22, 33, ..., 2013}. -/
theorem olympic_iberic_sets_containing_33 :
  ∀ X, is_iberic_set X ∧ 33 ∈ X → X = {n | 3 ∣ n ∧ 2 ≤ n ∧ n ≤ 2016} ∨ X = {n | 11 ∣ n ∧ 11 ≤ n ∧ n ≤ 2013} :=
by
  sorry

end NUMINAMATH_GPT_olympic_iberic_sets_containing_33_l921_92139


namespace NUMINAMATH_GPT_complex_identity_l921_92114

variable (i : ℂ)
axiom i_squared : i^2 = -1

theorem complex_identity : 1 + i + i^2 = i :=
by sorry

end NUMINAMATH_GPT_complex_identity_l921_92114


namespace NUMINAMATH_GPT_fractional_part_tiled_l921_92158

def room_length : ℕ := 12
def room_width : ℕ := 20
def number_of_tiles : ℕ := 40
def tile_area : ℕ := 1

theorem fractional_part_tiled :
  (number_of_tiles * tile_area : ℚ) / (room_length * room_width) = 1 / 6 :=
by
  sorry

end NUMINAMATH_GPT_fractional_part_tiled_l921_92158


namespace NUMINAMATH_GPT_XiaoMaHu_correct_calculation_l921_92191

theorem XiaoMaHu_correct_calculation :
  (∃ A B C D : Prop, (A = ((a b : ℝ) → (a - b)^2 = a^2 - b^2)) ∧ 
                   (B = ((a : ℝ) → (-2 * a^3)^2 = 4 * a^6)) ∧ 
                   (C = ((a : ℝ) → a^3 + a^2 = 2 * a^5)) ∧ 
                   (D = ((a : ℝ) → -(a - 1) = -a - 1)) ∧ 
                   (¬A ∧ B ∧ ¬C ∧ ¬D)) :=
sorry

end NUMINAMATH_GPT_XiaoMaHu_correct_calculation_l921_92191


namespace NUMINAMATH_GPT_initial_blue_marbles_l921_92159

theorem initial_blue_marbles (B R : ℕ) 
    (h1 : 3 * B = 5 * R) 
    (h2 : 4 * (B - 10) = R + 25) : 
    B = 19 := 
sorry

end NUMINAMATH_GPT_initial_blue_marbles_l921_92159


namespace NUMINAMATH_GPT_area_of_sector_AOB_l921_92176

-- Definitions for the conditions
def circumference_sector_AOB : Real := 6 -- Circumference of sector AOB
def central_angle_AOB : Real := 1 -- Central angle of sector AOB

-- Theorem stating the area of the sector is 2 cm²
theorem area_of_sector_AOB (C : Real) (θ : Real) (hC : C = circumference_sector_AOB) (hθ : θ = central_angle_AOB) : 
    ∃ S : Real, S = 2 :=
by
  sorry

end NUMINAMATH_GPT_area_of_sector_AOB_l921_92176


namespace NUMINAMATH_GPT_ratio_jl_jm_l921_92188

-- Define the side length of the square NOPQ as s
variable (s : ℝ)

-- Define the length (l) and width (m) of the rectangle JKLM
variable (l m : ℝ)

-- Conditions given in the problem
variable (area_overlap : ℝ)
variable (area_condition1 : area_overlap = 0.25 * s * s)
variable (area_condition2 : area_overlap = 0.40 * l * m)

theorem ratio_jl_jm (h1 : area_overlap = 0.25 * s * s) (h2 : area_overlap = 0.40 * l * m) : l / m = 2 / 5 :=
by
  sorry

end NUMINAMATH_GPT_ratio_jl_jm_l921_92188


namespace NUMINAMATH_GPT_floor_factorial_even_l921_92144

theorem floor_factorial_even (n : ℕ) (hn : n > 0) : 
  Nat.floor ((Nat.factorial (n - 1) : ℝ) / (n * (n + 1))) % 2 = 0 := 
sorry

end NUMINAMATH_GPT_floor_factorial_even_l921_92144


namespace NUMINAMATH_GPT_lazy_worker_days_worked_l921_92120

theorem lazy_worker_days_worked :
  ∃ x : ℕ, 24 * x - 6 * (30 - x) = 0 ∧ x = 6 :=
by
  existsi 6
  sorry

end NUMINAMATH_GPT_lazy_worker_days_worked_l921_92120


namespace NUMINAMATH_GPT_highest_average_speed_interval_l921_92121

theorem highest_average_speed_interval
  (d : ℕ → ℕ)
  (h0 : d 0 = 45)        -- Distance from 0 to 30 minutes
  (h1 : d 1 = 135)       -- Distance from 30 to 60 minutes
  (h2 : d 2 = 255)       -- Distance from 60 to 90 minutes
  (h3 : d 3 = 325) :     -- Distance from 90 to 120 minutes
  (1 / 2) * ((d 2 - d 1 : ℕ) : ℝ) > 
  max ((1 / 2) * ((d 1 - d 0 : ℕ) : ℝ)) 
      (max ((1 / 2) * ((d 3 - d 2 : ℕ) : ℝ))
          ((1 / 2) * ((d 3 - d 1 : ℕ) : ℝ))) :=
by
  sorry

end NUMINAMATH_GPT_highest_average_speed_interval_l921_92121


namespace NUMINAMATH_GPT_wheel_moves_in_one_hour_l921_92145

theorem wheel_moves_in_one_hour
  (rotations_per_minute : ℕ)
  (distance_per_rotation_cm : ℕ)
  (minutes_in_hour : ℕ) :
  rotations_per_minute = 20 →
  distance_per_rotation_cm = 35 →
  minutes_in_hour = 60 →
  let distance_per_rotation_m : ℚ := distance_per_rotation_cm / 100
  let total_rotations_per_hour : ℕ := rotations_per_minute * minutes_in_hour
  let total_distance_in_hour : ℚ := distance_per_rotation_m * total_rotations_per_hour
  total_distance_in_hour = 420 := by
  intros
  sorry

end NUMINAMATH_GPT_wheel_moves_in_one_hour_l921_92145


namespace NUMINAMATH_GPT_lucky_ticket_N123456_l921_92150

def digits : List ℕ := [1, 2, 3, 4, 5, 6]

def is_lucky (digits : List ℕ) : Prop :=
  ∃ f : ℕ → ℕ → ℕ, (f 1 (f (f 2 3) 4) * f 5 6) = 100

theorem lucky_ticket_N123456 : is_lucky digits :=
  sorry

end NUMINAMATH_GPT_lucky_ticket_N123456_l921_92150


namespace NUMINAMATH_GPT_probability_excluded_probability_selected_l921_92168

-- Define the population size and the sample size
def population_size : ℕ := 1005
def sample_size : ℕ := 50
def excluded_count : ℕ := 5

-- Use these values within the theorems
theorem probability_excluded : (excluded_count : ℚ) / (population_size : ℚ) = 5 / 1005 :=
by sorry

theorem probability_selected : (sample_size : ℚ) / (population_size : ℚ) = 50 / 1005 :=
by sorry

end NUMINAMATH_GPT_probability_excluded_probability_selected_l921_92168


namespace NUMINAMATH_GPT_min_value_expression_l921_92185

theorem min_value_expression (x y : ℝ) (hx : 0 < x) (hy : y = Real.sqrt x) :
  ∃ c, c = 2 ∧ ∀ u v : ℝ, 0 < u → v = Real.sqrt u → (u^2 + v^4) / (u * v^2) = c :=
by
  sorry

end NUMINAMATH_GPT_min_value_expression_l921_92185


namespace NUMINAMATH_GPT_employed_females_percentage_l921_92103

def P_total : ℝ := 0.64
def P_males : ℝ := 0.46

theorem employed_females_percentage : 
  ((P_total - P_males) / P_total) * 100 = 28.125 :=
by
  sorry

end NUMINAMATH_GPT_employed_females_percentage_l921_92103


namespace NUMINAMATH_GPT_semicircle_arc_length_l921_92134

theorem semicircle_arc_length (a b : ℝ) (hypotenuse_sum : a + b = 70) (a_eq_30 : a = 30) (b_eq_40 : b = 40) :
  ∃ (R : ℝ), (R = 24) ∧ (π * R = 12 * π) :=
by
  sorry

end NUMINAMATH_GPT_semicircle_arc_length_l921_92134


namespace NUMINAMATH_GPT_spencer_session_duration_l921_92124

-- Definitions of the conditions
def jumps_per_minute : ℕ := 4
def sessions_per_day : ℕ := 2
def total_jumps : ℕ := 400
def total_days : ℕ := 5

-- Calculation target: find the duration of each session
def jumps_per_day : ℕ := total_jumps / total_days
def jumps_per_session : ℕ := jumps_per_day / sessions_per_day
def session_duration := jumps_per_session / jumps_per_minute

theorem spencer_session_duration :
  session_duration = 10 := 
sorry

end NUMINAMATH_GPT_spencer_session_duration_l921_92124


namespace NUMINAMATH_GPT_add_and_simplify_fractions_l921_92193

theorem add_and_simplify_fractions :
  (1 / 462) + (23 / 42) = 127 / 231 :=
by
  sorry

end NUMINAMATH_GPT_add_and_simplify_fractions_l921_92193


namespace NUMINAMATH_GPT_orthocenter_of_ABC_is_correct_l921_92123

structure Point3D where
  x : ℚ
  y : ℚ
  z : ℚ

def A : Point3D := {x := 2, y := 3, z := -1}
def B : Point3D := {x := 6, y := -1, z := 2}
def C : Point3D := {x := 4, y := 5, z := 4}

def orthocenter (A B C : Point3D) : Point3D := {
  x := 101 / 33,
  y := 95 / 33,
  z := 47 / 33
}

theorem orthocenter_of_ABC_is_correct : orthocenter A B C = {x := 101 / 33, y := 95 / 33, z := 47 / 33} :=
  sorry

end NUMINAMATH_GPT_orthocenter_of_ABC_is_correct_l921_92123


namespace NUMINAMATH_GPT_ratio_girls_to_boys_l921_92135

-- Definitions of the conditions
def numGirls : ℕ := 10
def numBoys : ℕ := 20

-- Statement of the proof problem
theorem ratio_girls_to_boys : (numGirls / Nat.gcd numGirls numBoys) = 1 ∧ (numBoys / Nat.gcd numGirls numBoys) = 2 :=
by
  sorry

end NUMINAMATH_GPT_ratio_girls_to_boys_l921_92135


namespace NUMINAMATH_GPT_geom_series_first_term_l921_92140

theorem geom_series_first_term (r : ℚ) (S : ℚ) (a : ℚ) 
  (h_r : r = 1 / 4) 
  (h_S : S = 80)
  : a = 60 :=
by
  sorry

end NUMINAMATH_GPT_geom_series_first_term_l921_92140


namespace NUMINAMATH_GPT_tangent_line_extreme_values_l921_92152

-- Define the function f and its conditions
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x + 2

-- Given conditions
def cond1 (a b : ℝ) : Prop := 3 * a * 2^2 + b = 0
def cond2 (a b : ℝ) : Prop := a * 2^3 + b * 2 + 2 = -14

-- Part 1: Tangent line equation at (1, f(1))
theorem tangent_line (a b : ℝ) (h1 : cond1 a b) (h2 : cond2 a b) : 
  ∃ m c : ℝ, m = -9 ∧ c = 9 ∧ (∀ y : ℝ, y = f a b 1 → 9 * 1 + y = 0) :=
sorry

-- Part 2: Extreme values on [-3, 3]
-- Define critical points and endpoints
def f_value_at (a b x : ℝ) := f a b x

theorem extreme_values (a b : ℝ) (h1 : cond1 a b) (h2 : cond2 a b) :
  ∃ min max : ℝ, min = -14 ∧ max = 18 ∧ 
  f_value_at a b 2 = min ∧ f_value_at a b (-2) = max :=
sorry

end NUMINAMATH_GPT_tangent_line_extreme_values_l921_92152


namespace NUMINAMATH_GPT_binomial_coefficient_third_term_l921_92177

theorem binomial_coefficient_third_term (x a : ℝ) (h : 10 * a^3 * x = 80) : a = 2 :=
by
  sorry

end NUMINAMATH_GPT_binomial_coefficient_third_term_l921_92177


namespace NUMINAMATH_GPT_ones_digit_of_largest_power_of_three_dividing_27_factorial_l921_92136

theorem ones_digit_of_largest_power_of_three_dividing_27_factorial :
  let k := (27 / 3) + (27 / 9) + (27 / 27)
  let x := 3^k
  (x % 10) = 3 := by
  sorry

end NUMINAMATH_GPT_ones_digit_of_largest_power_of_three_dividing_27_factorial_l921_92136


namespace NUMINAMATH_GPT_base_conversion_problem_l921_92184

theorem base_conversion_problem (b : ℕ) (h : b^2 + 2 * b - 25 = 0) : b = 3 :=
sorry

end NUMINAMATH_GPT_base_conversion_problem_l921_92184


namespace NUMINAMATH_GPT_average_salary_rest_l921_92196

theorem average_salary_rest (number_of_workers : ℕ) 
                            (avg_salary_all : ℝ) 
                            (number_of_technicians : ℕ) 
                            (avg_salary_technicians : ℝ) 
                            (rest_workers : ℕ) 
                            (total_salary_all : ℝ) 
                            (total_salary_technicians : ℝ) 
                            (total_salary_rest : ℝ) 
                            (avg_salary_rest : ℝ) 
                            (h1 : number_of_workers = 28)
                            (h2 : avg_salary_all = 8000)
                            (h3 : number_of_technicians = 7)
                            (h4 : avg_salary_technicians = 14000)
                            (h5 : rest_workers = number_of_workers - number_of_technicians)
                            (h6 : total_salary_all = number_of_workers * avg_salary_all)
                            (h7 : total_salary_technicians = number_of_technicians * avg_salary_technicians)
                            (h8 : total_salary_rest = total_salary_all - total_salary_technicians)
                            (h9 : avg_salary_rest = total_salary_rest / rest_workers) :
  avg_salary_rest = 6000 :=
by {
  -- the proof would go here
  sorry
}

end NUMINAMATH_GPT_average_salary_rest_l921_92196


namespace NUMINAMATH_GPT_smallest_k_for_divisibility_l921_92171

theorem smallest_k_for_divisibility (z : ℂ) (hz : z^7 = 1) : ∃ k : ℕ, (∀ m : ℕ, z ^ (m * k) = 1) ∧ k = 84 :=
sorry

end NUMINAMATH_GPT_smallest_k_for_divisibility_l921_92171


namespace NUMINAMATH_GPT_number_of_maple_trees_planted_l921_92128

def before := 53
def after := 64
def planted := after - before

theorem number_of_maple_trees_planted : planted = 11 := by
  sorry

end NUMINAMATH_GPT_number_of_maple_trees_planted_l921_92128


namespace NUMINAMATH_GPT_arithmetic_sequence_suff_nec_straight_line_l921_92101

variable (n : ℕ) (P_n : ℕ → ℝ)

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ m : ℕ, a (m + 1) = a m + d

def lies_on_straight_line (P : ℕ → ℝ) : Prop :=
  ∃ m b, ∀ n, P n = m * n + b

theorem arithmetic_sequence_suff_nec_straight_line
  (h_n : 0 < n)
  (h_arith : arithmetic_sequence P_n) :
  lies_on_straight_line P_n ↔ arithmetic_sequence P_n :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_suff_nec_straight_line_l921_92101


namespace NUMINAMATH_GPT_baron_munchausen_failed_l921_92161

theorem baron_munchausen_failed : 
  ∀ (n : ℕ), 10 ≤ n ∧ n ≤ 99 → ¬∃ (d1 d2 : ℕ), ∃ (k : ℕ), n * 100 + (d1 * 10 + d2) = k^2 := 
by
  intros n hn
  obtain ⟨h10, h99⟩ := hn
  sorry

end NUMINAMATH_GPT_baron_munchausen_failed_l921_92161
