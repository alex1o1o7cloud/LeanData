import Mathlib

namespace NUMINAMATH_GPT_average_number_of_problems_per_day_l811_81150

theorem average_number_of_problems_per_day (P D : ℕ) (hP : P = 161) (hD : D = 7) : (P / D) = 23 :=
  by sorry

end NUMINAMATH_GPT_average_number_of_problems_per_day_l811_81150


namespace NUMINAMATH_GPT_roots_equal_of_quadratic_eq_zero_l811_81110

theorem roots_equal_of_quadratic_eq_zero (a : ℝ) :
  (∃ x : ℝ, (x^2 - a*x + 1) = 0 ∧ (∀ y : ℝ, (y^2 - a*y + 1) = 0 → y = x)) → (a = 2 ∨ a = -2) :=
by
  sorry

end NUMINAMATH_GPT_roots_equal_of_quadratic_eq_zero_l811_81110


namespace NUMINAMATH_GPT_leonardo_nap_duration_l811_81178

theorem leonardo_nap_duration (h : (1 : ℝ) / 5 * 60 = 12) : (1 / 5 : ℝ) * 60 = 12 :=
by 
  exact h

end NUMINAMATH_GPT_leonardo_nap_duration_l811_81178


namespace NUMINAMATH_GPT_intersect_empty_range_of_a_union_subsets_range_of_a_l811_81132

variable {x a : ℝ}

def A (a : ℝ) : Set ℝ := {x | a ≤ x ∧ x ≤ a + 3}
def B : Set ℝ := {x | (x - 6) * (x + 2) > 0}

theorem intersect_empty_range_of_a (h : A a ∩ B = ∅) : -2 ≤ a ∧ a ≤ 3 :=
by
  sorry

theorem union_subsets_range_of_a (h : A a ∪ B = B) : a < -5 ∨ a > 6 :=
by
  sorry

end NUMINAMATH_GPT_intersect_empty_range_of_a_union_subsets_range_of_a_l811_81132


namespace NUMINAMATH_GPT_abs_inequality_holds_l811_81170

theorem abs_inequality_holds (m x : ℝ) (h : -1 ≤ m ∧ m ≤ 6) : 
  |x - 2| + |x + 4| ≥ m^2 - 5 * m :=
sorry

end NUMINAMATH_GPT_abs_inequality_holds_l811_81170


namespace NUMINAMATH_GPT_johnny_future_years_l811_81148

theorem johnny_future_years (x : ℕ) (h1 : 8 + x = 2 * (8 - 3)) : x = 2 :=
by
  sorry

end NUMINAMATH_GPT_johnny_future_years_l811_81148


namespace NUMINAMATH_GPT_rowing_trip_time_l811_81152

theorem rowing_trip_time
  (v_0 : ℝ) -- Rowing speed in still water
  (v_c : ℝ) -- Velocity of current
  (d : ℝ) -- Distance to the place
  (h_v0 : v_0 = 10) -- Given condition that rowing speed is 10 kmph
  (h_vc : v_c = 2) -- Given condition that current speed is 2 kmph
  (h_d : d = 144) -- Given condition that distance is 144 km :
  : (d / (v_0 - v_c) + d / (v_0 + v_c)) = 30 := -- Proving the total round trip time is 30 hours
by
  sorry

end NUMINAMATH_GPT_rowing_trip_time_l811_81152


namespace NUMINAMATH_GPT_base_price_lowered_percentage_l811_81185

theorem base_price_lowered_percentage (P : ℝ) (new_price final_price : ℝ) (x : ℝ)
    (h1 : new_price = P - (x / 100) * P)
    (h2 : final_price = 0.9 * new_price)
    (h3 : final_price = P - (14.5 / 100) * P) :
    x = 5 :=
  sorry

end NUMINAMATH_GPT_base_price_lowered_percentage_l811_81185


namespace NUMINAMATH_GPT_total_amount_pqr_l811_81184

theorem total_amount_pqr (p q r : ℕ) (T : ℕ) 
  (hr : r = 2 / 3 * (T - r))
  (hr_value : r = 1600) : 
  T = 4000 :=
by
  sorry

end NUMINAMATH_GPT_total_amount_pqr_l811_81184


namespace NUMINAMATH_GPT_cos_value_in_second_quadrant_l811_81167

variable (a : ℝ)
variables (h1 : π/2 < a ∧ a < π) (h2 : Real.sin a = 5/13)

theorem cos_value_in_second_quadrant : Real.cos a = -12/13 :=
  sorry

end NUMINAMATH_GPT_cos_value_in_second_quadrant_l811_81167


namespace NUMINAMATH_GPT_polynomial_expansion_sum_constants_l811_81129

theorem polynomial_expansion_sum_constants :
  ∃ (A B C D : ℤ), ((x - 3) * (4 * x ^ 2 + 2 * x - 7) = A * x ^ 3 + B * x ^ 2 + C * x + D) → A + B + C + D = 2 := 
by
  sorry

end NUMINAMATH_GPT_polynomial_expansion_sum_constants_l811_81129


namespace NUMINAMATH_GPT_stratified_sampling_size_l811_81169

theorem stratified_sampling_size (a_ratio b_ratio c_ratio : ℕ) (total_items_A : ℕ) (h_ratio : a_ratio + b_ratio + c_ratio = 10)
  (h_A_ratio : a_ratio = 2) (h_B_ratio : b_ratio = 3) (h_C_ratio : c_ratio = 5) (items_A : total_items_A = 20) : 
  ∃ n : ℕ, n = total_items_A * 5 := 
by {
  -- The proof should go here. Since we only need the statement:
  sorry
}

end NUMINAMATH_GPT_stratified_sampling_size_l811_81169


namespace NUMINAMATH_GPT_value_of_fraction_l811_81183

noncomputable def arithmetic_sequence (a1 a2 : ℝ) : Prop :=
  a2 - a1 = (-4 - (-1)) / (4 - 1)

noncomputable def geometric_sequence (b2 : ℝ) : Prop :=
  b2 * b2 = (-4) * (-1) ∧ b2 < 0

theorem value_of_fraction (a1 a2 b2 : ℝ)
  (h1 : arithmetic_sequence a1 a2)
  (h2 : geometric_sequence b2) :
  (a2 - a1) / b2 = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_value_of_fraction_l811_81183


namespace NUMINAMATH_GPT_f_14_52_l811_81168

def f : ℕ × ℕ → ℕ := sorry

axiom f_xx (x : ℕ) : f (x, x) = x
axiom f_symm (x y : ℕ) : f (x, y) = f (y, x)
axiom f_eq (x y : ℕ) : (x + y) * f (x, y) = y * f (x, x + y)

theorem f_14_52 : f (14, 52) = 364 := sorry

end NUMINAMATH_GPT_f_14_52_l811_81168


namespace NUMINAMATH_GPT_apples_in_basket_l811_81145

noncomputable def total_apples (good_cond: ℕ) (good_ratio: ℝ) := (good_cond : ℝ) / good_ratio

theorem apples_in_basket : total_apples 66 0.88 = 75 :=
by
  sorry

end NUMINAMATH_GPT_apples_in_basket_l811_81145


namespace NUMINAMATH_GPT_find_number_of_non_officers_l811_81102

theorem find_number_of_non_officers
  (avg_salary_all : ℝ)
  (avg_salary_officers : ℝ)
  (avg_salary_non_officers : ℝ)
  (num_officers : ℕ) :
  avg_salary_all = 120 ∧
  avg_salary_officers = 450 ∧
  avg_salary_non_officers = 110 ∧
  num_officers = 15 →
  ∃ N : ℕ, (120 * (15 + N) = 450 * 15 + 110 * N) ∧ N = 495 :=
by
  sorry

end NUMINAMATH_GPT_find_number_of_non_officers_l811_81102


namespace NUMINAMATH_GPT_problem_statement_l811_81181

variable (X Y : ℝ)

theorem problem_statement
  (h1 : 0.18 * X = 0.54 * 1200)
  (h2 : X = 4 * Y) :
  X = 3600 ∧ Y = 900 := by
  sorry

end NUMINAMATH_GPT_problem_statement_l811_81181


namespace NUMINAMATH_GPT_number_of_solid_figures_is_4_l811_81125

def is_solid_figure (shape : String) : Bool :=
  shape = "cone" ∨ shape = "cuboid" ∨ shape = "sphere" ∨ shape = "triangular prism"

def shapes : List String :=
  ["circle", "square", "cone", "cuboid", "line segment", "sphere", "triangular prism", "right-angled triangle"]

def number_of_solid_figures : Nat :=
  (shapes.filter is_solid_figure).length

theorem number_of_solid_figures_is_4 : number_of_solid_figures = 4 :=
  by sorry

end NUMINAMATH_GPT_number_of_solid_figures_is_4_l811_81125


namespace NUMINAMATH_GPT_circle_passing_through_points_l811_81130

theorem circle_passing_through_points :
  ∃ D E F : ℝ, 
    (∀ x y : ℝ, 
      (x = 0 ∧ y = 0 → x^2 + y^2 + D * x + E * y + F = 0) ∧
      (x = 4 ∧ y = 0 → x^2 + y^2 + D * x + E * y + F = 0) ∧
      (x = -1 ∧ y = 1 → x^2 + y^2 + D * x + E * y + F = 0)) ∧
    (x^2 + y^2 + D * x + E * y + F = 0 → x^2 + y^2 - 4 * x - 6 * y = 0) :=
sorry

end NUMINAMATH_GPT_circle_passing_through_points_l811_81130


namespace NUMINAMATH_GPT_roots_quadratic_reciprocal_l811_81135

theorem roots_quadratic_reciprocal (x1 x2 : ℝ) (h1 : x1 + x2 = -8) (h2 : x1 * x2 = 4) :
  (1 / x1) + (1 / x2) = -2 :=
sorry

end NUMINAMATH_GPT_roots_quadratic_reciprocal_l811_81135


namespace NUMINAMATH_GPT_algebraic_expression_value_l811_81111

-- Definitions based on the conditions
variable {a : ℝ}
axiom root_equation : 2 * a^2 + 3 * a - 4 = 0

-- Definition of the problem: Proving that 2a^2 + 3a equals 4.
theorem algebraic_expression_value : 2 * a^2 + 3 * a = 4 :=
by 
  have h : 2 * a^2 + 3 * a - 4 = 0 := root_equation
  have h' : 2 * a^2 + 3 * a = 4 := by sorry
  exact h'

end NUMINAMATH_GPT_algebraic_expression_value_l811_81111


namespace NUMINAMATH_GPT_simplification_at_negative_two_l811_81104

noncomputable def simplify_expression (x : ℚ) : ℚ :=
  ((x^2 - 4*x + 4) / (x^2 - 1)) / ((x^2 - 2*x) / (x + 1)) + (1 / (x - 1))

theorem simplification_at_negative_two :
  ∀ x : ℚ, -2 ≤ x ∧ x ≤ 2 ∧ x ≠ 0 ∧ x ≠ 1 ∧ x ≠ -1 → simplify_expression (-2) = -1 :=
by simp [simplify_expression]; sorry

end NUMINAMATH_GPT_simplification_at_negative_two_l811_81104


namespace NUMINAMATH_GPT_proof_height_difference_l811_81192

noncomputable def height_in_inches_between_ruby_and_xavier : Prop :=
  let janet_height_inches := 62.75
  let inch_to_cm := 2.54
  let janet_height_cm := janet_height_inches * inch_to_cm
  let charlene_height := 1.5 * janet_height_cm
  let pablo_height := charlene_height + 1.85 * 100
  let ruby_height := pablo_height - 0.5
  let xavier_height := charlene_height + 2.13 * 100 - 97.75
  let paul_height := ruby_height + 50
  let height_diff_cm := xavier_height - ruby_height
  let height_diff_inches := height_diff_cm / inch_to_cm
  height_diff_inches = -18.78

theorem proof_height_difference :
  height_in_inches_between_ruby_and_xavier :=
by
  sorry

end NUMINAMATH_GPT_proof_height_difference_l811_81192


namespace NUMINAMATH_GPT_projectile_reaches_40_at_first_time_l811_81173

theorem projectile_reaches_40_at_first_time : ∃ t : ℝ, 0 < t ∧ (40 = -16 * t^2 + 64 * t) ∧ (∀ t' : ℝ, 0 < t' ∧ t' < t → ¬ (40 = -16 * t'^2 + 64 * t')) ∧ t = 0.8 :=
by
  sorry

end NUMINAMATH_GPT_projectile_reaches_40_at_first_time_l811_81173


namespace NUMINAMATH_GPT_median_and_mode_l811_81159

open Set

variable (data_set : List ℝ)
variable (mean : ℝ)

noncomputable def median (l : List ℝ) : ℝ := sorry -- Define medial function
noncomputable def mode (l : List ℝ) : ℝ := sorry -- Define mode function

theorem median_and_mode (x : ℝ) (mean_set : (3 + x + 4 + 5 + 8) / 5 = 5) :
  data_set = [3, 4, 5, 5, 8] ∧ median data_set = 5 ∧ mode data_set = 5 :=
by
  have hx : x = 5 := sorry
  have hdata_set : data_set = [3, 4, 5, 5, 8] := sorry
  have hmedian : median data_set = 5 := sorry
  have hmode : mode data_set = 5 := sorry
  exact ⟨hdata_set, hmedian, hmode⟩

end NUMINAMATH_GPT_median_and_mode_l811_81159


namespace NUMINAMATH_GPT_sin_330_eq_neg_half_l811_81198

theorem sin_330_eq_neg_half : Real.sin (330 * Real.pi / 180) = -1 / 2 :=
by sorry

end NUMINAMATH_GPT_sin_330_eq_neg_half_l811_81198


namespace NUMINAMATH_GPT_triangle_inequality_sides_l811_81172

theorem triangle_inequality_sides {a b c : ℝ} (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (triangle_ineq1 : a + b > c) (triangle_ineq2 : b + c > a) (triangle_ineq3 : c + a > b) : 
  |(a / b) + (b / c) + (c / a) - (b / a) - (c / b) - (a / c)| < 1 :=
  sorry

end NUMINAMATH_GPT_triangle_inequality_sides_l811_81172


namespace NUMINAMATH_GPT_hall_length_width_difference_l811_81149

theorem hall_length_width_difference : 
  ∃ L W : ℝ, W = (1 / 2) * L ∧ L * W = 450 ∧ L - W = 15 :=
sorry

end NUMINAMATH_GPT_hall_length_width_difference_l811_81149


namespace NUMINAMATH_GPT_correct_interval_for_monotonic_decrease_l811_81100

noncomputable def f (x : ℝ) : ℝ := |Real.tan (1 / 2 * x - Real.pi / 6)|

theorem correct_interval_for_monotonic_decrease :
  ∀ k : ℤ, ∃ I : Set ℝ,
    I = Set.Ioc (2 * k * Real.pi - 2 * Real.pi / 3) (2 * k * Real.pi + Real.pi / 3) ∧
    ∀ x y, x ∈ I → y ∈ I → x < y → f y < f x :=
sorry

end NUMINAMATH_GPT_correct_interval_for_monotonic_decrease_l811_81100


namespace NUMINAMATH_GPT_function_values_l811_81157

noncomputable def f (a b c x : ℝ) : ℝ := a * Real.cos x + b * x^2 + c

theorem function_values (a b c : ℝ) : 
  f a b c 1 = 1 ∧ f a b c (-1) = 1 := 
by
  sorry

end NUMINAMATH_GPT_function_values_l811_81157


namespace NUMINAMATH_GPT_exterior_angle_BAC_eq_162_l811_81176

noncomputable def measure_of_angle_BAC : ℝ := 360 - 108 - 90

theorem exterior_angle_BAC_eq_162 :
  measure_of_angle_BAC = 162 := by
  sorry

end NUMINAMATH_GPT_exterior_angle_BAC_eq_162_l811_81176


namespace NUMINAMATH_GPT_complement_supplement_angle_l811_81146

theorem complement_supplement_angle (α : ℝ) : 
  ( 180 - α) = 3 * ( 90 - α ) → α = 45 :=
by 
  sorry

end NUMINAMATH_GPT_complement_supplement_angle_l811_81146


namespace NUMINAMATH_GPT_total_sodas_bought_l811_81179

-- Condition 1: Number of sodas they drank
def sodas_drank : ℕ := 3

-- Condition 2: Number of extra sodas Robin had
def sodas_extras : ℕ := 8

-- Mathematical equivalence we want to prove: Total number of sodas bought by Robin
theorem total_sodas_bought : sodas_drank + sodas_extras = 11 := by
  sorry

end NUMINAMATH_GPT_total_sodas_bought_l811_81179


namespace NUMINAMATH_GPT_original_number_l811_81182

variable (x : ℝ)

theorem original_number (h1 : x - x / 10 = 37.35) : x = 41.5 := by
  sorry

end NUMINAMATH_GPT_original_number_l811_81182


namespace NUMINAMATH_GPT_axis_of_symmetry_parabola_l811_81180

theorem axis_of_symmetry_parabola (x y : ℝ) :
  y = - (1 / 8) * x^2 → y = 2 :=
sorry

end NUMINAMATH_GPT_axis_of_symmetry_parabola_l811_81180


namespace NUMINAMATH_GPT_intersection_infinite_l811_81188

-- Define the equations of the curves
def curve1 (x y : ℝ) : Prop := 2 * x^2 - x * y - y^2 - x - 2 * y - 1 = 0
def curve2 (x y : ℝ) : Prop := 3 * x^2 - 4 * x * y + y^2 - 3 * x + y = 0

-- Theorem statement
theorem intersection_infinite : ∃ (f : ℝ → ℝ), ∀ x, curve1 x (f x) ∧ curve2 x (f x) :=
sorry

end NUMINAMATH_GPT_intersection_infinite_l811_81188


namespace NUMINAMATH_GPT_sum_possible_values_l811_81109

theorem sum_possible_values (x y : ℝ) (h : x * y - x / y^3 - y / x^3 = 4) :
  (x - 2) * (y - 2) = 4 ∨ (x - 2) * (y - 2) = 0 → (4 + 0 = 4) :=
by
  sorry

end NUMINAMATH_GPT_sum_possible_values_l811_81109


namespace NUMINAMATH_GPT_original_square_perimeter_l811_81133

-- Define the problem statement
theorem original_square_perimeter (P_perimeter : ℕ) (hP : P_perimeter = 56) : 
  ∃ sq_perimeter : ℕ, sq_perimeter = 32 := 
by 
  sorry

end NUMINAMATH_GPT_original_square_perimeter_l811_81133


namespace NUMINAMATH_GPT_scalene_triangle_smallest_angle_sum_l811_81166

theorem scalene_triangle_smallest_angle_sum :
  ∀ (A B C : ℝ), A ≠ B ∧ B ≠ C ∧ A ≠ C ∧ A = 45 ∧ C = 135 → (∃ x y : ℝ, x = y ∧ x = 45 ∧ y = 45 ∧ x + y = 90) :=
by
  intros A B C h
  sorry

end NUMINAMATH_GPT_scalene_triangle_smallest_angle_sum_l811_81166


namespace NUMINAMATH_GPT_base5_div_l811_81142

-- Definitions for base 5 numbers
def n1 : ℕ := (2 * 125) + (4 * 25) + (3 * 5) + 4  -- 2434_5 in base 10 is 369
def n2 : ℕ := (1 * 25) + (3 * 5) + 2              -- 132_5 in base 10 is 42
def d  : ℕ := (2 * 5) + 1                          -- 21_5 in base 10 is 11

theorem base5_div (res : ℕ) : res = (122 : ℕ) → (n1 + n2) / d = res :=
by sorry

end NUMINAMATH_GPT_base5_div_l811_81142


namespace NUMINAMATH_GPT_roots_fourth_pow_sum_l811_81191

theorem roots_fourth_pow_sum :
  (∃ p q r : ℂ, (∀ z, (z = p ∨ z = q ∨ z = r) ↔ z^3 - z^2 + 2*z - 3 = 0) ∧ p^4 + q^4 + r^4 = 13) := by
sorry

end NUMINAMATH_GPT_roots_fourth_pow_sum_l811_81191


namespace NUMINAMATH_GPT_correct_growth_rate_equation_l811_81112

-- Define the conditions
def packages_first_day := 200
def packages_third_day := 242

-- Define the average daily growth rate
variable (x : ℝ)

-- State the theorem to prove
theorem correct_growth_rate_equation :
  packages_first_day * (1 + x)^2 = packages_third_day :=
by
  sorry

end NUMINAMATH_GPT_correct_growth_rate_equation_l811_81112


namespace NUMINAMATH_GPT_solve_for_x_l811_81113

theorem solve_for_x (x : ℝ) (h : (7 * x + 3) / (x + 5) - 5 / (x + 5) = 2 / (x + 5)) : x = 4 / 7 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_x_l811_81113


namespace NUMINAMATH_GPT_remaining_download_time_l811_81174

-- Define the relevant quantities
def total_size : ℝ := 1250
def downloaded : ℝ := 310
def download_speed : ℝ := 2.5

-- State the theorem
theorem remaining_download_time : (total_size - downloaded) / download_speed = 376 := by
  -- Proof will be filled in here
  sorry

end NUMINAMATH_GPT_remaining_download_time_l811_81174


namespace NUMINAMATH_GPT_least_number_condition_l811_81189

-- Define the set of divisors as a constant
def divisors : Set ℕ := {1, 2, 3, 4, 5, 6, 8, 15}

-- Define the least number that satisfies the condition
def least_number : ℕ := 125

-- The theorem stating that the least number 125 leaves a remainder of 5 when divided by the given set of numbers
theorem least_number_condition : ∀ d ∈ divisors, least_number % d = 5 :=
by
  sorry

end NUMINAMATH_GPT_least_number_condition_l811_81189


namespace NUMINAMATH_GPT_first_student_time_l811_81123

-- Define the conditions
def num_students := 4
def avg_last_three := 35
def avg_all := 30
def total_time_all := num_students * avg_all
def total_time_last_three := (num_students - 1) * avg_last_three

-- State the theorem
theorem first_student_time : (total_time_all - total_time_last_three) = 15 :=
by
  -- Proof is skipped
  sorry

end NUMINAMATH_GPT_first_student_time_l811_81123


namespace NUMINAMATH_GPT_tan_double_angle_tan_angle_add_pi_div_4_l811_81158

theorem tan_double_angle (α : ℝ) (h : Real.tan α = -2) : Real.tan (2 * α) = 4 / 3 :=
by
  sorry

theorem tan_angle_add_pi_div_4 (α : ℝ) (h : Real.tan α = -2) : Real.tan (2 * α + Real.pi / 4) = -7 :=
by
  sorry

end NUMINAMATH_GPT_tan_double_angle_tan_angle_add_pi_div_4_l811_81158


namespace NUMINAMATH_GPT_option_d_correct_l811_81153

theorem option_d_correct (a b : ℝ) : (a - b)^2 = (b - a)^2 := 
by {
  sorry
}

end NUMINAMATH_GPT_option_d_correct_l811_81153


namespace NUMINAMATH_GPT_complex_magnitude_l811_81165

theorem complex_magnitude (z : ℂ) (i_unit : ℂ := Complex.I) 
  (h : (z - i_unit) * i_unit = 2 + i_unit) : Complex.abs z = Real.sqrt 5 := 
by
  sorry

end NUMINAMATH_GPT_complex_magnitude_l811_81165


namespace NUMINAMATH_GPT_bananas_in_each_group_l811_81117

theorem bananas_in_each_group (total_bananas groups : ℕ) (h1 : total_bananas = 392) (h2 : groups = 196) :
    total_bananas / groups = 2 :=
by
  sorry

end NUMINAMATH_GPT_bananas_in_each_group_l811_81117


namespace NUMINAMATH_GPT_greatest_sum_of_consecutive_integers_product_less_500_l811_81118

theorem greatest_sum_of_consecutive_integers_product_less_500 :
  ∃ n : ℤ, n * (n + 1) < 500 ∧ (n + (n + 1)) = 43 :=
by
  sorry

end NUMINAMATH_GPT_greatest_sum_of_consecutive_integers_product_less_500_l811_81118


namespace NUMINAMATH_GPT_sum_of_b_values_l811_81154

theorem sum_of_b_values (b1 b2 : ℝ) : 
  (∀ x : ℝ, (9 * x^2 + (b1 + 15) * x + 16 = 0 ∨ 9 * x^2 + (b2 + 15) * x + 16 = 0) ∧ 
           (b1 + 15)^2 - 4 * 9 * 16 = 0 ∧ 
           (b2 + 15)^2 - 4 * 9 * 16 = 0) → 
  (b1 + b2) = -30 := 
sorry

end NUMINAMATH_GPT_sum_of_b_values_l811_81154


namespace NUMINAMATH_GPT_n_in_S_implies_n_squared_in_S_l811_81127

-- Definition of the set S
def S : Set ℕ := {n | ∃ a b c d e f : ℕ, a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 ∧ f > 0 ∧ 
                      n - 1 = a^2 + b^2 ∧ n = c^2 + d^2 ∧ n + 1 = e^2 + f^2}

-- The proof goal
theorem n_in_S_implies_n_squared_in_S (n : ℕ) (h : n ∈ S) : n^2 ∈ S :=
by
  sorry

end NUMINAMATH_GPT_n_in_S_implies_n_squared_in_S_l811_81127


namespace NUMINAMATH_GPT_compound_proposition_C_l811_81103

open Real

def p : Prop := ∃ x : ℝ, x - 2 > log x 
def q : Prop := ∀ x : ℝ, sin x < x

theorem compound_proposition_C : p ∧ ¬q :=
by sorry

end NUMINAMATH_GPT_compound_proposition_C_l811_81103


namespace NUMINAMATH_GPT_clara_loses_q_minus_p_l811_81139

def clara_heads_prob : ℚ := 2 / 3
def clara_tails_prob : ℚ := 1 / 3

def ethan_heads_prob : ℚ := 1 / 4
def ethan_tails_prob : ℚ := 3 / 4

def lose_prob_clara : ℚ := clara_heads_prob
def both_tails_prob : ℚ := clara_tails_prob * ethan_tails_prob

noncomputable def total_prob_clara_loses : ℚ :=
  lose_prob_clara + ∑' n : ℕ, (both_tails_prob ^ n) * lose_prob_clara

theorem clara_loses_q_minus_p :
  ∃ (p q : ℕ), Nat.gcd p q = 1 ∧ total_prob_clara_loses = p / q ∧ (q - p = 1) :=
sorry

end NUMINAMATH_GPT_clara_loses_q_minus_p_l811_81139


namespace NUMINAMATH_GPT_determine_number_of_20_pound_boxes_l811_81186

variable (numBoxes : ℕ) (avgWeight : ℕ) (x : ℕ) (y : ℕ)

theorem determine_number_of_20_pound_boxes 
  (h1 : numBoxes = 30) 
  (h2 : avgWeight = 18) 
  (h3 : x + y = 30) 
  (h4 : 10 * x + 20 * y = 540) : 
  y = 24 :=
  by
  sorry

end NUMINAMATH_GPT_determine_number_of_20_pound_boxes_l811_81186


namespace NUMINAMATH_GPT_smallest_positive_integer_cube_ends_in_632_l811_81122

theorem smallest_positive_integer_cube_ends_in_632 :
  ∃ n : ℕ, (n > 0) ∧ (n^3 % 1000 = 632) ∧ ∀ m : ℕ, (m > 0) ∧ (m^3 % 1000 = 632) → n ≤ m := 
sorry

end NUMINAMATH_GPT_smallest_positive_integer_cube_ends_in_632_l811_81122


namespace NUMINAMATH_GPT_oil_truck_radius_l811_81193

theorem oil_truck_radius
  (r_stationary : ℝ) (h_stationary : ℝ) (h_drop : ℝ) 
  (h_truck : ℝ)
  (V_pumped : ℝ) (π : ℝ) (r_truck : ℝ) :
  r_stationary = 100 → h_stationary = 25 → h_drop = 0.064 → h_truck = 10 →
  V_pumped = π * r_stationary^2 * h_drop →
  V_pumped = π * r_truck^2 * h_truck →
  r_truck = 8 := 
by 
  intros r_stationary_eq h_stationary_eq h_drop_eq h_truck_eq V_pumped_eq1 V_pumped_eq2
  sorry

end NUMINAMATH_GPT_oil_truck_radius_l811_81193


namespace NUMINAMATH_GPT_normalize_equation1_normalize_equation2_l811_81138

-- Define the first equation
def equation1 (x y : ℝ) := 2 * x - 3 * y - 10 = 0

-- Define the normalized form of the first equation
def normalized_equation1 (x y : ℝ) := (2 / Real.sqrt 13) * x - (3 / Real.sqrt 13) * y - (10 / Real.sqrt 13) = 0

-- Prove that the normalized form of the first equation is correct
theorem normalize_equation1 (x y : ℝ) (h : equation1 x y) : normalized_equation1 x y := 
sorry

-- Define the second equation
def equation2 (x y : ℝ) := 3 * x + 4 * y = 0

-- Define the normalized form of the second equation
def normalized_equation2 (x y : ℝ) := (3 / 5) * x + (4 / 5) * y = 0

-- Prove that the normalized form of the second equation is correct
theorem normalize_equation2 (x y : ℝ) (h : equation2 x y) : normalized_equation2 x y := 
sorry

end NUMINAMATH_GPT_normalize_equation1_normalize_equation2_l811_81138


namespace NUMINAMATH_GPT_proof_problem_l811_81106

def h (x : ℝ) : ℝ := x^2 - 3 * x + 7
def k (x : ℝ) : ℝ := 2 * x + 4

theorem proof_problem : h (k 3) - k (h 3) = 59 := by
  sorry

end NUMINAMATH_GPT_proof_problem_l811_81106


namespace NUMINAMATH_GPT_cost_of_paving_l811_81143

def length : ℝ := 5.5
def width : ℝ := 4
def rate_per_sq_meter : ℝ := 850

theorem cost_of_paving :
  rate_per_sq_meter * (length * width) = 18700 :=
by
  sorry

end NUMINAMATH_GPT_cost_of_paving_l811_81143


namespace NUMINAMATH_GPT_solve_system_l811_81161

open Classical

theorem solve_system : ∃ t : ℝ, ∀ (x y z : ℝ), 
  (x^2 - 9 * y^2 = 0 ∧ x + y + z = 0) ↔ 
  (x = 3 * t ∧ y = t ∧ z = -4 * t) 
  ∨ (x = -3 * t ∧ y = t ∧ z = 2 * t) := 
by 
  sorry

end NUMINAMATH_GPT_solve_system_l811_81161


namespace NUMINAMATH_GPT_find_k_l811_81134

theorem find_k (k n m : ℕ) (hk : k > 0) (hn : n > 0) (hm : m > 0) 
  (h : (1 / (n ^ 2 : ℝ) + 1 / (m ^ 2 : ℝ)) = (k : ℝ) / (n ^ 2 + m ^ 2)) : k = 4 :=
sorry

end NUMINAMATH_GPT_find_k_l811_81134


namespace NUMINAMATH_GPT_ralph_fewer_pictures_l811_81105

-- Define the number of wild animal pictures Ralph and Derrick have.
def ralph_pictures : ℕ := 26
def derrick_pictures : ℕ := 34

-- The main theorem stating that Ralph has 8 fewer pictures than Derrick.
theorem ralph_fewer_pictures : derrick_pictures - ralph_pictures = 8 := by
  -- The proof is omitted, denoted by 'sorry'.
  sorry

end NUMINAMATH_GPT_ralph_fewer_pictures_l811_81105


namespace NUMINAMATH_GPT_charity_donation_correct_l811_81177

-- Define each donation series for Suzanne, Maria, and James
def suzanne_donation_per_km (n : ℕ) : ℝ :=
  match n with
  |  0     => 10
  | (n+1)  => 2 * suzanne_donation_per_km n

def maria_donation_per_km (n : ℕ) : ℝ :=
  match n with
  |  0     => 15
  | (n+1)  => 1.5 * maria_donation_per_km n

def james_donation_per_km (n : ℕ) : ℝ :=
  match n with
  |  0     => 20
  | (n+1)  => 2 * james_donation_per_km n

-- Total donations after 5 kilometers
def total_donation_suzanne : ℝ := (List.range 5).map suzanne_donation_per_km |>.sum
def total_donation_maria : ℝ := (List.range 5).map maria_donation_per_km |>.sum
def total_donation_james : ℝ := (List.range 5).map james_donation_per_km |>.sum

def total_donation_charity : ℝ :=
  total_donation_suzanne + total_donation_maria + total_donation_james

-- Statement to be proven
theorem charity_donation_correct : total_donation_charity = 1127.81 := by
  sorry

end NUMINAMATH_GPT_charity_donation_correct_l811_81177


namespace NUMINAMATH_GPT_find_b_l811_81163

noncomputable def angle_B : ℝ := 60
noncomputable def c : ℝ := 8
noncomputable def diff_b_a (b a : ℝ) : Prop := b - a = 4

theorem find_b (b a : ℝ) (h₁ : angle_B = 60) (h₂ : c = 8) (h₃ : diff_b_a b a) :
  b = 7 :=
sorry

end NUMINAMATH_GPT_find_b_l811_81163


namespace NUMINAMATH_GPT_find_number_l811_81190

theorem find_number (x : ℤ) (h : 3 * (3 * x) = 18) : x = 2 := 
sorry

end NUMINAMATH_GPT_find_number_l811_81190


namespace NUMINAMATH_GPT_value_increase_factor_l811_81162

theorem value_increase_factor (P S : ℝ) (frac F : ℝ) (hP : P = 200) (hS : S = 240) (hfrac : frac = 0.40) :
  frac * (P * F) = S -> F = 3 := by
  sorry

end NUMINAMATH_GPT_value_increase_factor_l811_81162


namespace NUMINAMATH_GPT_bottles_left_on_shelf_l811_81126

variable (initial_bottles : ℕ)
variable (bottles_jason : ℕ)
variable (bottles_harry : ℕ)

theorem bottles_left_on_shelf (h₁ : initial_bottles = 35) (h₂ : bottles_jason = 5) (h₃ : bottles_harry = bottles_jason + 6) :
  initial_bottles - (bottles_jason + bottles_harry) = 24 := by
  sorry

end NUMINAMATH_GPT_bottles_left_on_shelf_l811_81126


namespace NUMINAMATH_GPT_taxi_fare_distance_l811_81115

theorem taxi_fare_distance (x : ℕ) (h₁ : 8 + 2 * (x - 3) = 20) : x = 9 :=
by {
  sorry
}

end NUMINAMATH_GPT_taxi_fare_distance_l811_81115


namespace NUMINAMATH_GPT_problem1_problem2_l811_81128

-- Problem 1: Prove the expression
theorem problem1 (a b : ℝ) : 
  2 * a * (a - 2 * b) - (2 * a - b) ^ 2 = -2 * a ^ 2 - b ^ 2 := 
sorry

-- Problem 2: Prove the solution to the equation
theorem problem2 (x : ℝ) (h : (x - 1) ^ 3 - 3 = 3 / 8) : 
  x = 5 / 2 := 
sorry

end NUMINAMATH_GPT_problem1_problem2_l811_81128


namespace NUMINAMATH_GPT_part_I_part_II_l811_81101

variable {a b : ℝ}

theorem part_I (h1 : a * b ≠ 0) (h2 : a * b > 0) :
  b / a + a / b ≥ 2 :=
sorry

theorem part_II (h1 : a * b ≠ 0) (h3 : a * b < 0) :
  abs (b / a + a / b) ≥ 2 :=
sorry

end NUMINAMATH_GPT_part_I_part_II_l811_81101


namespace NUMINAMATH_GPT_solve_quadratic_1_solve_quadratic_2_solve_quadratic_3_solve_quadratic_4_solve_quadratic_5_solve_quadratic_6_l811_81140

-- Problem 1: 5x² = 40x
theorem solve_quadratic_1 (x : ℝ) : 5 * x^2 = 40 * x ↔ (x = 0 ∨ x = 8) :=
by sorry

-- Problem 2: 25/9 x² = 100
theorem solve_quadratic_2 (x : ℝ) : (25 / 9) * x^2 = 100 ↔ (x = 6 ∨ x = -6) :=
by sorry

-- Problem 3: 10x = x² + 21
theorem solve_quadratic_3 (x : ℝ) : 10 * x = x^2 + 21 ↔ (x = 7 ∨ x = 3) :=
by sorry

-- Problem 4: x² = 12x + 288
theorem solve_quadratic_4 (x : ℝ) : x^2 = 12 * x + 288 ↔ (x = 24 ∨ x = -12) :=
by sorry

-- Problem 5: x² + 20 1/4 = 11 1/4 x
theorem solve_quadratic_5 (x : ℝ) : x^2 + 81 / 4 = 45 / 4 * x ↔ (x = 9 / 4 ∨ x = 9) :=
by sorry

-- Problem 6: 1/12 x² + 7/12 x = 19
theorem solve_quadratic_6 (x : ℝ) : (1 / 12) * x^2 + (7 / 12) * x = 19 ↔ (x = 12 ∨ x = -19) :=
by sorry

end NUMINAMATH_GPT_solve_quadratic_1_solve_quadratic_2_solve_quadratic_3_solve_quadratic_4_solve_quadratic_5_solve_quadratic_6_l811_81140


namespace NUMINAMATH_GPT_days_in_week_l811_81131

theorem days_in_week {F D : ℕ} (h1 : F = 3 + 11) (h2 : F = 2 * D) : D = 7 :=
by
  sorry

end NUMINAMATH_GPT_days_in_week_l811_81131


namespace NUMINAMATH_GPT_range_of_a_same_side_of_line_l811_81197

theorem range_of_a_same_side_of_line 
  {P Q : ℝ × ℝ} 
  (hP : P = (3, -1)) 
  (hQ : Q = (-1, 2)) 
  (h_side : (3 * a - 3) * (-a + 3) > 0) : 
  a > 1 ∧ a < 3 := 
by 
  sorry

end NUMINAMATH_GPT_range_of_a_same_side_of_line_l811_81197


namespace NUMINAMATH_GPT_total_glass_area_l811_81147

theorem total_glass_area 
  (len₁ len₂ len₃ wid₁ wid₂ wid₃ : ℕ)
  (h₁ : len₁ = 30) (h₂ : wid₁ = 12)
  (h₃ : len₂ = 30) (h₄ : wid₂ = 12)
  (h₅ : len₃ = 20) (h₆ : wid₃ = 12) :
  (len₁ * wid₁ + len₂ * wid₂ + len₃ * wid₃) = 960 := 
by
  sorry

end NUMINAMATH_GPT_total_glass_area_l811_81147


namespace NUMINAMATH_GPT_simplify_polynomial_l811_81107

variable (x : ℝ)

theorem simplify_polynomial : 
  (2 * x^4 + 3 * x^3 - 5 * x + 6) + (-6 * x^4 - 2 * x^3 + 3 * x^2 + 5 * x - 4) = 
  -4 * x^4 + x^3 + 3 * x^2 + 2 :=
by
  sorry

end NUMINAMATH_GPT_simplify_polynomial_l811_81107


namespace NUMINAMATH_GPT_surveys_on_tuesday_l811_81160

theorem surveys_on_tuesday
  (num_surveys_monday: ℕ) -- number of surveys Bart completed on Monday
  (earnings_monday: ℕ) -- earning per survey on Monday
  (total_earnings: ℕ) -- total earnings over the two days
  (earnings_per_survey: ℕ) -- earnings Bart gets per survey
  (monday_earnings_eq : earnings_monday = num_surveys_monday * earnings_per_survey)
  (total_earnings_eq : total_earnings = earnings_monday + (8 : ℕ))
  (earnings_per_survey_eq : earnings_per_survey = 2)
  : ((8 : ℕ) / earnings_per_survey = 4) := sorry

end NUMINAMATH_GPT_surveys_on_tuesday_l811_81160


namespace NUMINAMATH_GPT_problem_statement_l811_81171

noncomputable def f (x : ℝ) : ℝ := x + 1
noncomputable def g (x : ℝ) : ℝ := -x + 1
noncomputable def h (x : ℝ) : ℝ := f x * g x

theorem problem_statement :
  (h (-x) = h x) :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l811_81171


namespace NUMINAMATH_GPT_initial_boys_l811_81114

theorem initial_boys (p : ℝ) (initial_boys : ℝ) (final_boys : ℝ) (final_groupsize : ℝ) : 
  (initial_boys = 0.35 * p) ->
  (final_boys = 0.35 * p - 1) ->
  (final_groupsize = p + 3) ->
  (final_boys / final_groupsize = 0.3) ->
  initial_boys = 13 := 
by
  sorry

end NUMINAMATH_GPT_initial_boys_l811_81114


namespace NUMINAMATH_GPT_hyperbola_property_l811_81120

def hyperbola := {x : ℝ // ∃ y : ℝ, x^2 - y^2 / 8 = 1}

def is_on_left_branch (M : hyperbola) : Prop :=
  M.1 < 0

def focus1 : ℝ := -3
def focus2 : ℝ := 3

def distance (a b : ℝ) : ℝ := abs (a - b)

theorem hyperbola_property (M : hyperbola) (hM : is_on_left_branch M) :
  distance M.1 focus1 + distance focus1 focus2 - distance M.1 focus2 = 4 :=
  sorry

end NUMINAMATH_GPT_hyperbola_property_l811_81120


namespace NUMINAMATH_GPT_fourth_vs_third_difference_l811_81199

def first_competitor_distance : ℕ := 22

def second_competitor_distance : ℕ := first_competitor_distance + 1

def third_competitor_distance : ℕ := second_competitor_distance - 2

def fourth_competitor_distance : ℕ := 24

theorem fourth_vs_third_difference : 
  fourth_competitor_distance - third_competitor_distance = 3 := by
  sorry

end NUMINAMATH_GPT_fourth_vs_third_difference_l811_81199


namespace NUMINAMATH_GPT_room_breadth_is_five_l811_81137

theorem room_breadth_is_five 
  (length : ℝ)
  (height : ℝ)
  (bricks_per_square_meter : ℝ)
  (total_bricks : ℝ)
  (H_length : length = 4)
  (H_height : height = 2)
  (H_bricks_per_square_meter : bricks_per_square_meter = 17)
  (H_total_bricks : total_bricks = 340) 
  : ∃ (breadth : ℝ), breadth = 5 :=
by
  -- we leave the proof as sorry for now
  sorry

end NUMINAMATH_GPT_room_breadth_is_five_l811_81137


namespace NUMINAMATH_GPT_min_value_a_3b_9c_l811_81136

theorem min_value_a_3b_9c (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a * b * c = 27) : 
  a + 3 * b + 9 * c ≥ 27 := 
sorry

end NUMINAMATH_GPT_min_value_a_3b_9c_l811_81136


namespace NUMINAMATH_GPT_isosceles_triangle_perimeter_l811_81195

theorem isosceles_triangle_perimeter (a b : ℕ) (h1 : a = 6) (h2 : b = 13) 
  (triangle_inequality : b + b > a) : 
  (2 * b + a) = 32 := by
  sorry

end NUMINAMATH_GPT_isosceles_triangle_perimeter_l811_81195


namespace NUMINAMATH_GPT_find_d_l811_81196

theorem find_d (a b c d : ℝ) (hac : 0 < a) (hbc : 0 < b) (hcc : 0 < c) (hdc : 0 < d)
  (oscillates : ∀ x, -2 ≤ a * Real.sin (b * x + c) + d ∧ a * Real.sin (b * x + c) + d ≤ 4) :
  d = 1 :=
sorry

end NUMINAMATH_GPT_find_d_l811_81196


namespace NUMINAMATH_GPT_required_cement_l811_81144

def total_material : ℝ := 0.67
def sand : ℝ := 0.17
def dirt : ℝ := 0.33
def cement : ℝ := 0.17

theorem required_cement : cement = total_material - (sand + dirt) := 
by
  sorry

end NUMINAMATH_GPT_required_cement_l811_81144


namespace NUMINAMATH_GPT_simplify_expression_l811_81119

variable (x : ℝ)

theorem simplify_expression : 3 * x + 4 * x^3 + 2 - (7 - 3 * x - 4 * x^3) = 8 * x^3 + 6 * x - 5 := 
by 
  sorry

end NUMINAMATH_GPT_simplify_expression_l811_81119


namespace NUMINAMATH_GPT_arithmetic_sequence_S9_l811_81124

theorem arithmetic_sequence_S9 :
  ∀ {a : ℕ → ℤ} {S : ℕ → ℤ},
  (∀ n : ℕ, S n = (n * (2 * a 1 + (n - 1) * d)) / 2) →
  a 2 = 3 →
  S 4 = 16 →
  S 9 = 81 :=
by
  intro a S h_S h_a2 h_S4
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_S9_l811_81124


namespace NUMINAMATH_GPT_geometric_sequence_sum_l811_81187

theorem geometric_sequence_sum (S : ℕ → ℚ) (n : ℕ) 
  (hS_n : S n = 54) 
  (hS_2n : S (2 * n) = 60) 
  : S (3 * n) = 60 + 2 / 3 := 
sorry

end NUMINAMATH_GPT_geometric_sequence_sum_l811_81187


namespace NUMINAMATH_GPT_find_coordinates_of_M_l811_81108

def point_in_second_quadrant (P : ℝ × ℝ) : Prop :=
  P.1 < 0 ∧ P.2 > 0

def distance_to_x_axis (P : ℝ × ℝ) (d : ℝ) : Prop :=
  abs (P.2) = d

def distance_to_y_axis (P : ℝ × ℝ) (d : ℝ) : Prop :=
  abs (P.1) = d

theorem find_coordinates_of_M :
  ∃ M : ℝ × ℝ, point_in_second_quadrant M ∧ distance_to_x_axis M 5 ∧ distance_to_y_axis M 3 ∧ M = (-3, 5) :=
by
  sorry

end NUMINAMATH_GPT_find_coordinates_of_M_l811_81108


namespace NUMINAMATH_GPT_probability_at_least_one_hit_l811_81116

variable (P₁ P₂ : ℝ)

theorem probability_at_least_one_hit (h₁ : 0 ≤ P₁ ∧ P₁ ≤ 1) (h₂ : 0 ≤ P₂ ∧ P₂ ≤ 1) :
  1 - (1 - P₁) * (1 - P₂) = P₁ + P₂ - P₁ * P₂ :=
by
  sorry

end NUMINAMATH_GPT_probability_at_least_one_hit_l811_81116


namespace NUMINAMATH_GPT_exists_abc_l811_81155

theorem exists_abc (n k : ℕ) (hn : n > 20) (hk : k > 1) (hdiv : k^2 ∣ n) : 
  ∃ (a b c : ℕ), n = a * b + b * c + c * a :=
by
  sorry

end NUMINAMATH_GPT_exists_abc_l811_81155


namespace NUMINAMATH_GPT_sqrt_fraction_addition_l811_81141

theorem sqrt_fraction_addition :
  (Real.sqrt ((25 : ℝ) / 36 + 16 / 9)) = Real.sqrt 89 / 6 := by
  sorry

end NUMINAMATH_GPT_sqrt_fraction_addition_l811_81141


namespace NUMINAMATH_GPT_find_larger_number_l811_81151

theorem find_larger_number (L S : ℕ) (h1 : L - S = 1365) (h2 : L = 6 * S + 15) : L = 1635 := 
by 
  sorry

end NUMINAMATH_GPT_find_larger_number_l811_81151


namespace NUMINAMATH_GPT_problem1_problem2_problem3_problem4_l811_81156

theorem problem1 : (-4.7 : ℝ) + 0.9 = -3.8 := by
  sorry

theorem problem2 : (- (1 / 2) : ℝ) - (-(1 / 3)) = -(1 / 6) := by
  sorry

theorem problem3 : (- (10 / 9) : ℝ) * (- (6 / 10)) = (2 / 3) := by
  sorry

theorem problem4 : (0 : ℝ) * (-5) = 0 := by
  sorry

end NUMINAMATH_GPT_problem1_problem2_problem3_problem4_l811_81156


namespace NUMINAMATH_GPT_find_y_l811_81175

def angle_at_W (RWQ RWT QWR TWQ : ℝ) :=  RWQ + RWT + QWR + TWQ

theorem find_y 
  (RWQ RWT QWR TWQ : ℝ)
  (h1 : RWQ = 90) 
  (h2 : RWT = 3 * y)
  (h3 : QWR = y)
  (h4 : TWQ = 90) 
  (h_sum : angle_at_W RWQ RWT QWR TWQ = 360)  
  : y = 67.5 :=
by
  sorry

end NUMINAMATH_GPT_find_y_l811_81175


namespace NUMINAMATH_GPT_central_cell_value_l811_81164

theorem central_cell_value :
  ∀ (a b c d e f g h i : ℝ),
    (a * b * c = 10) →
    (d * e * f = 10) →
    (g * h * i = 10) →
    (a * d * g = 10) →
    (b * e * h = 10) →
    (c * f * i = 10) →
    (a * b * d * e = 3) →
    (b * c * e * f = 3) →
    (d * e * g * h = 3) →
    (e * f * h * i = 3) →
    e = 0.00081 := 
by 
  intros a b c d e f g h i h1 h2 h3 h4 h5 h6 h7 h8 h9 h10
  sorry

end NUMINAMATH_GPT_central_cell_value_l811_81164


namespace NUMINAMATH_GPT_quadratic_ratio_l811_81121

theorem quadratic_ratio (b c : ℤ) (h : ∀ x : ℤ, x^2 + 1400 * x + 1400 = (x + b) ^ 2 + c) : c / b = -698 :=
sorry

end NUMINAMATH_GPT_quadratic_ratio_l811_81121


namespace NUMINAMATH_GPT_find_z_l811_81194

-- Define the given angles
def angle_ABC : ℝ := 95
def angle_BAC : ℝ := 65

-- Define the angle sum property for triangle ABC
def angle_sum_triangle_ABC (a b : ℝ) : ℝ := 180 - (a + b)

-- Define the angle DCE as equal to angle BCA
def angle_DCE : ℝ := angle_sum_triangle_ABC angle_ABC angle_BAC

-- Define the angle sum property for right triangle CDE
def z (dce : ℝ) : ℝ := 90 - dce

-- State the theorem to be proved
theorem find_z : z angle_DCE = 70 :=
by
  -- Statement for proof is provided
  sorry

end NUMINAMATH_GPT_find_z_l811_81194
