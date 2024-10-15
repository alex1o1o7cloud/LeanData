import Mathlib

namespace NUMINAMATH_GPT_n_squared_plus_d_not_perfect_square_l2081_208198

theorem n_squared_plus_d_not_perfect_square (n d : ℕ) (h1 : n > 0)
  (h2 : d > 0) (h3 : d ∣ 2 * n^2) : ¬ ∃ x : ℕ, n^2 + d = x^2 := 
sorry

end NUMINAMATH_GPT_n_squared_plus_d_not_perfect_square_l2081_208198


namespace NUMINAMATH_GPT_number_of_whole_numbers_in_intervals_l2081_208158

theorem number_of_whole_numbers_in_intervals : 
  let interval_start := (5 / 3 : ℝ)
  let interval_end := 2 * Real.pi
  ∃ n : ℕ, interval_start < ↑n ∧ ↑n < interval_end ∧ (n = 2 ∨ n = 3 ∨ n = 4 ∨ n = 5 ∨ n = 6) ∧ 
  (∀ m : ℕ, interval_start < ↑m ∧ ↑m < interval_end → (m = 2 ∨ m = 3 ∨ m = 4 ∨ m = 5 ∨ m = 6)) :=
sorry

end NUMINAMATH_GPT_number_of_whole_numbers_in_intervals_l2081_208158


namespace NUMINAMATH_GPT_abs_diff_squares_l2081_208190

theorem abs_diff_squares (a b : ℤ) (ha : a = 103) (hb : b = 97) : |a^2 - b^2| = 1200 :=
by
  sorry

end NUMINAMATH_GPT_abs_diff_squares_l2081_208190


namespace NUMINAMATH_GPT_product_of_roots_l2081_208183

open Real

theorem product_of_roots : (sqrt (Real.exp (1 / 4 * log (16)))) * (sqrt (Real.exp (1 / 6 * log (64)))) = 4 :=
by
  -- sorry is used to bypass the actual proof implementation
  sorry

end NUMINAMATH_GPT_product_of_roots_l2081_208183


namespace NUMINAMATH_GPT_reinforcement_correct_l2081_208108

-- Conditions
def initial_men : ℕ := 2000
def initial_days : ℕ := 54
def days_before_reinforcement : ℕ := 18
def days_after_reinforcement : ℕ := 20

-- Define the remaining provisions after 18 days
def provisions_left : ℕ := initial_men * (initial_days - days_before_reinforcement)

-- Define reinforcement
def reinforcement : ℕ := 
  sorry -- placeholder for the definition

-- Theorem to prove
theorem reinforcement_correct :
  reinforcement = 1600 :=
by
  -- Use the given conditions to derive the reinforcement value
  let total_provision := initial_men * initial_days
  let remaining_provision := provisions_left
  let men_after_reinforcement := initial_men + reinforcement
  have h := remaining_provision = men_after_reinforcement * days_after_reinforcement
  sorry -- placeholder for the proof

end NUMINAMATH_GPT_reinforcement_correct_l2081_208108


namespace NUMINAMATH_GPT_larger_number_is_1634_l2081_208178

theorem larger_number_is_1634 (L S : ℤ) (h1 : L - S = 1365) (h2 : L = 6 * S + 20) : L = 1634 := 
sorry

end NUMINAMATH_GPT_larger_number_is_1634_l2081_208178


namespace NUMINAMATH_GPT_largest_integer_n_such_that_n_squared_minus_11n_plus_28_is_negative_l2081_208154

theorem largest_integer_n_such_that_n_squared_minus_11n_plus_28_is_negative :
  ∃ (n : ℤ), (4 < n) ∧ (n < 7) ∧ (n = 6) :=
by
  sorry

end NUMINAMATH_GPT_largest_integer_n_such_that_n_squared_minus_11n_plus_28_is_negative_l2081_208154


namespace NUMINAMATH_GPT_triangle_third_side_length_l2081_208112

theorem triangle_third_side_length
  (AC BC : ℝ)
  (h_a h_b h_c : ℝ)
  (half_sum_heights_eq : (h_a + h_b) / 2 = h_c) :
  AC = 6 → BC = 3 → AB = 4 :=
by
  sorry

end NUMINAMATH_GPT_triangle_third_side_length_l2081_208112


namespace NUMINAMATH_GPT_volume_of_cube_l2081_208173

theorem volume_of_cube (a : ℕ) (h : (a^3 - a = a^3 - 5)) : a^3 = 125 :=
by {
  -- The necessary algebraic manipulation follows
  sorry
}

end NUMINAMATH_GPT_volume_of_cube_l2081_208173


namespace NUMINAMATH_GPT_jill_water_stored_l2081_208119

theorem jill_water_stored (n : ℕ) (h : n = 24) : 
  8 * (1 / 4 : ℝ) + 8 * (1 / 2 : ℝ) + 8 * 1 = 14 :=
by
  sorry

end NUMINAMATH_GPT_jill_water_stored_l2081_208119


namespace NUMINAMATH_GPT_magnitude_of_sum_l2081_208151

variables (a b : ℝ × ℝ)
variables (h1 : a.1 * b.1 + a.2 * b.2 = 0)
variables (h2 : a = (4, 3))
variables (h3 : (b.1 ^ 2 + b.2 ^ 2) = 1)

theorem magnitude_of_sum (a b : ℝ × ℝ) (h1 : a.1 * b.1 + a.2 * b.2 = 0) 
  (h2 : a = (4, 3)) (h3 : (b.1 ^ 2 + b.2 ^ 2) = 1) : 
  (a.1 + 2 * b.1) ^ 2 + (a.2 + 2 * b.2) ^ 2 = 29 :=
by sorry

end NUMINAMATH_GPT_magnitude_of_sum_l2081_208151


namespace NUMINAMATH_GPT_solve_inequality_l2081_208165

theorem solve_inequality (x : ℝ) (h : 3 * x + 4 ≠ 0) :
  3 - 1 / (3 * x + 4) < 5 ↔ -3 / 2 < x :=
by
  sorry

end NUMINAMATH_GPT_solve_inequality_l2081_208165


namespace NUMINAMATH_GPT_initial_processing_capacity_l2081_208176

variable (x y z : ℕ)

-- Conditions
def initial_condition : Prop := x * y = 38880
def after_modernization : Prop := (x + 3) * z = 44800
def capacity_increased : Prop := y < z
def minimum_machines : Prop := x ≥ 20

-- Prove that the initial daily processing capacity y is 1215
theorem initial_processing_capacity
  (h1 : initial_condition x y)
  (h2 : after_modernization x z)
  (h3 : capacity_increased y z)
  (h4 : minimum_machines x) :
  y = 1215 := by
  sorry

end NUMINAMATH_GPT_initial_processing_capacity_l2081_208176


namespace NUMINAMATH_GPT_sin_double_angle_neg_l2081_208146

variable {α : ℝ} {k : ℤ}

-- Condition: α in the fourth quadrant.
def in_fourth_quadrant (α : ℝ) (k : ℤ) : Prop :=
  - (Real.pi / 2) + 2 * k * Real.pi < α ∧ α < 2 * k * Real.pi

-- Goal: Prove sin 2α < 0 given that α is in the fourth quadrant.
theorem sin_double_angle_neg (α : ℝ) (k : ℤ) (h : in_fourth_quadrant α k) : Real.sin (2 * α) < 0 := by
  sorry

end NUMINAMATH_GPT_sin_double_angle_neg_l2081_208146


namespace NUMINAMATH_GPT_total_enemies_l2081_208104

theorem total_enemies (E : ℕ) (h : 8 * (E - 2) = 40) : E = 7 := sorry

end NUMINAMATH_GPT_total_enemies_l2081_208104


namespace NUMINAMATH_GPT_largest_of_three_consecutive_integers_l2081_208142

theorem largest_of_three_consecutive_integers (x : ℤ) 
  (h : x + (x + 1) + (x + 2) = 18) : x + 2 = 7 := 
sorry

end NUMINAMATH_GPT_largest_of_three_consecutive_integers_l2081_208142


namespace NUMINAMATH_GPT_line_through_point_equal_intercepts_l2081_208152

theorem line_through_point_equal_intercepts (P : ℝ × ℝ) (x y a : ℝ) (k : ℝ) 
  (hP : P = (2, 3))
  (hx : x / a + y / a = 1 ∨ (P.fst * k - P.snd = 0)) :
  (x + y - 5 = 0 ∨ 3 * P.fst - 2 * P.snd = 0) := by
  sorry

end NUMINAMATH_GPT_line_through_point_equal_intercepts_l2081_208152


namespace NUMINAMATH_GPT_initial_velocity_l2081_208191

noncomputable def displacement (t : ℝ) : ℝ := 3 * t - t^2

theorem initial_velocity :
  (deriv displacement 0) = 3 :=
by
  sorry

end NUMINAMATH_GPT_initial_velocity_l2081_208191


namespace NUMINAMATH_GPT_roses_left_unsold_l2081_208156

def price_per_rose : ℕ := 4
def initial_roses : ℕ := 13
def total_earned : ℕ := 36

theorem roses_left_unsold : (initial_roses - (total_earned / price_per_rose) = 4) :=
by
  sorry

end NUMINAMATH_GPT_roses_left_unsold_l2081_208156


namespace NUMINAMATH_GPT_range_of_f_when_a_0_range_of_a_for_three_zeros_l2081_208157

noncomputable def f_part1 (x : ℝ) : ℝ :=
if h : x ≤ 0 then 2 ^ x else x ^ 2

theorem range_of_f_when_a_0 : Set.range f_part1 = {y : ℝ | 0 < y} := by
  sorry

noncomputable def f_part2 (a : ℝ) (x : ℝ) : ℝ :=
if h : x ≤ 0 then 2 ^ x - a else x ^ 2 - 3 * a * x + a

def discriminant (a : ℝ) (x : ℝ) : ℝ := (3 * a) ^ 2 - 4 * a

theorem range_of_a_for_three_zeros (a : ℝ) :
  (∀ x : ℝ, f_part2 a x = 0) → (4 / 9 < a ∧ a ≤ 1) := by
  sorry

end NUMINAMATH_GPT_range_of_f_when_a_0_range_of_a_for_three_zeros_l2081_208157


namespace NUMINAMATH_GPT_find_A_l2081_208188

theorem find_A (A B : ℕ) (hcfAB lcmAB : ℕ)
  (hcf_cond : Nat.gcd A B = hcfAB)
  (lcm_cond : Nat.lcm A B = lcmAB)
  (B_val : B = 169)
  (hcf_val : hcfAB = 13)
  (lcm_val : lcmAB = 312) :
  A = 24 :=
by 
  sorry

end NUMINAMATH_GPT_find_A_l2081_208188


namespace NUMINAMATH_GPT_simplify_and_evaluate_expression_l2081_208153

theorem simplify_and_evaluate_expression (x : ℝ) (hx : x = 4) :
  (1 / (x + 2) + 1) / ((x^2 + 6 * x + 9) / (x^2 - 4)) = 2 / 7 :=
by
  sorry

end NUMINAMATH_GPT_simplify_and_evaluate_expression_l2081_208153


namespace NUMINAMATH_GPT_difference_in_dimes_l2081_208139

variables (q : ℝ)

def samantha_quarters : ℝ := 3 * q + 2
def bob_quarters : ℝ := 2 * q + 8
def quarter_to_dimes : ℝ := 2.5

theorem difference_in_dimes :
  quarter_to_dimes * (samantha_quarters q - bob_quarters q) = 2.5 * q - 15 :=
by sorry

end NUMINAMATH_GPT_difference_in_dimes_l2081_208139


namespace NUMINAMATH_GPT_egg_laying_hens_l2081_208193

theorem egg_laying_hens (total_chickens : ℕ) (num_roosters : ℕ) (non_egg_laying_hens : ℕ)
  (h1 : total_chickens = 325)
  (h2 : num_roosters = 28)
  (h3 : non_egg_laying_hens = 20) :
  total_chickens - num_roosters - non_egg_laying_hens = 277 :=
by sorry

end NUMINAMATH_GPT_egg_laying_hens_l2081_208193


namespace NUMINAMATH_GPT_fraction_addition_l2081_208162

variable {a b : ℚ}
variable (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a / b = 3 / 4)

theorem fraction_addition (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a / b = 3 / 4) : (a + b) / b = 7 / 4 :=
  sorry

end NUMINAMATH_GPT_fraction_addition_l2081_208162


namespace NUMINAMATH_GPT_option_B_correct_l2081_208140

theorem option_B_correct (x m : ℕ) : (x^3)^m / (x^m)^2 = x^m := sorry

end NUMINAMATH_GPT_option_B_correct_l2081_208140


namespace NUMINAMATH_GPT_evaluate_nested_radical_l2081_208111

noncomputable def nested_radical (x : ℝ) := x = Real.sqrt (3 - x)

theorem evaluate_nested_radical (x : ℝ) (h : nested_radical x) : 
  x = (Real.sqrt 13 - 1) / 2 :=
by sorry

end NUMINAMATH_GPT_evaluate_nested_radical_l2081_208111


namespace NUMINAMATH_GPT_data_transmission_time_l2081_208160

def packet_size : ℕ := 256
def num_packets : ℕ := 100
def transmission_rate : ℕ := 200
def total_data : ℕ := num_packets * packet_size
def transmission_time_in_seconds : ℚ := total_data / transmission_rate
def transmission_time_in_minutes : ℚ := transmission_time_in_seconds / 60

theorem data_transmission_time :
  transmission_time_in_minutes = 2 :=
  sorry

end NUMINAMATH_GPT_data_transmission_time_l2081_208160


namespace NUMINAMATH_GPT_range_of_x_l2081_208126

section
  variable (f : ℝ → ℝ)

  -- Conditions:
  -- 1. f is an even function
  def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

  -- 2. f is monotonically increasing on [0, +∞)
  def mono_increasing_on_nonneg (f : ℝ → ℝ) : Prop := ∀ x y, 0 ≤ x → x ≤ y → f x ≤ f y

  -- Range of x
  def in_range (x : ℝ) : Prop := (1 : ℝ) / 3 < x ∧ x < (2 : ℝ) / 3

  -- Main statement
  theorem range_of_x (f_is_even : is_even f) (f_is_mono : mono_increasing_on_nonneg f) :
    ∀ x, f (2 * x - 1) < f ((1 : ℝ) / 3) ↔ in_range x := 
  by
    sorry
end

end NUMINAMATH_GPT_range_of_x_l2081_208126


namespace NUMINAMATH_GPT_product_of_factors_l2081_208150

theorem product_of_factors : (2.1 * (53.2 - 0.2) = 111.3) := by
  sorry

end NUMINAMATH_GPT_product_of_factors_l2081_208150


namespace NUMINAMATH_GPT_max_consecutive_sum_l2081_208189

theorem max_consecutive_sum (N a : ℕ) (h : N * (2 * a + N - 1) = 240) : N ≤ 15 :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_max_consecutive_sum_l2081_208189


namespace NUMINAMATH_GPT_price_reduction_equation_l2081_208143

variable (x : ℝ)

theorem price_reduction_equation 
    (original_price : ℝ)
    (final_price : ℝ)
    (two_reductions : original_price * (1 - x) ^ 2 = final_price) :
    100 * (1 - x) ^ 2 = 81 :=
by
  sorry

end NUMINAMATH_GPT_price_reduction_equation_l2081_208143


namespace NUMINAMATH_GPT_num_special_matrices_l2081_208161

open Matrix

theorem num_special_matrices :
  ∃ (M : Matrix (Fin 4) (Fin 4) ℕ), 
    (∀ i j, 1 ≤ M i j ∧ M i j ≤ 16) ∧ 
    (∀ i j, i < j → M i j < M i (j + 1)) ∧ 
    (∀ i j, i < j → M i j < M (i + 1) j) ∧ 
    (∀ i, i < 3 → M i i < M (i + 1) (i + 1)) ∧ 
    (∀ i, i < 3 → M i (3 - i) < M (i + 1) (2 - i)) ∧ 
    (∃ n, n = 144) :=
sorry

end NUMINAMATH_GPT_num_special_matrices_l2081_208161


namespace NUMINAMATH_GPT_product_of_two_equal_numbers_l2081_208115

theorem product_of_two_equal_numbers :
  ∃ (x : ℕ), (5 * 20 = 12 + 22 + 16 + 2 * x) ∧ (x * x = 625) :=
by
  sorry

end NUMINAMATH_GPT_product_of_two_equal_numbers_l2081_208115


namespace NUMINAMATH_GPT_area_of_rectangular_plot_l2081_208133

-- Defining the breadth
def breadth : ℕ := 26

-- Defining the length as thrice the breadth
def length : ℕ := 3 * breadth

-- Defining the area as the product of length and breadth
def area : ℕ := length * breadth

-- The theorem stating the problem to prove
theorem area_of_rectangular_plot : area = 2028 := by
  -- Initial proof step skipped
  sorry

end NUMINAMATH_GPT_area_of_rectangular_plot_l2081_208133


namespace NUMINAMATH_GPT_hypotenuse_length_l2081_208194

theorem hypotenuse_length (a b c : ℝ) (h₀ : a^2 + b^2 + c^2 = 1800) (h₁ : c^2 = a^2 + b^2) : c = 30 :=
by
  sorry

end NUMINAMATH_GPT_hypotenuse_length_l2081_208194


namespace NUMINAMATH_GPT_cost_of_gas_used_l2081_208129

theorem cost_of_gas_used (initial_odometer final_odometer fuel_efficiency cost_per_gallon : ℝ)
  (h₀ : initial_odometer = 82300)
  (h₁ : final_odometer = 82335)
  (h₂ : fuel_efficiency = 22)
  (h₃ : cost_per_gallon = 3.80) :
  (final_odometer - initial_odometer) / fuel_efficiency * cost_per_gallon = 6.04 :=
by
  sorry

end NUMINAMATH_GPT_cost_of_gas_used_l2081_208129


namespace NUMINAMATH_GPT_tangents_from_point_to_circle_l2081_208131

theorem tangents_from_point_to_circle (x y k : ℝ) (
    P : ℝ × ℝ)
    (h₁ : P = (1, -1))
    (circle_eq : x^2 + y^2 + 2*x + 2*y + k = 0)
    (h₂ : P = (1, -1))
    (has_two_tangents : 1^2 + (-1)^2 - k / 2 > 0):
  -2 < k ∧ k < 2 :=
by 
    sorry

end NUMINAMATH_GPT_tangents_from_point_to_circle_l2081_208131


namespace NUMINAMATH_GPT_binom_10_3_l2081_208177

def combination (n k : ℕ) := n.factorial / (k.factorial * (n - k).factorial)

theorem binom_10_3 : combination 10 3 = 120 := by
  sorry

end NUMINAMATH_GPT_binom_10_3_l2081_208177


namespace NUMINAMATH_GPT_solve_quartic_eq_l2081_208147

theorem solve_quartic_eq {x : ℝ} : (x - 4)^4 + (x - 6)^4 = 16 → (x = 4 ∨ x = 6) :=
by
  sorry

end NUMINAMATH_GPT_solve_quartic_eq_l2081_208147


namespace NUMINAMATH_GPT_rowing_distance_l2081_208102

theorem rowing_distance (D : ℝ) : 
  (D / 14 + D / 2 = 120) → D = 210 := by
  sorry

end NUMINAMATH_GPT_rowing_distance_l2081_208102


namespace NUMINAMATH_GPT_find_m_l2081_208196

theorem find_m : ∃ m : ℤ, 2^5 - 7 = 3^3 + m ∧ m = -2 :=
by
  use -2
  sorry

end NUMINAMATH_GPT_find_m_l2081_208196


namespace NUMINAMATH_GPT_delta_max_success_ratio_l2081_208164

theorem delta_max_success_ratio :
  ∃ (x y z w : ℕ),
  (0 < x ∧ x < (7 * y) / 12) ∧
  (0 < z ∧ z < (5 * w) / 8) ∧
  (y + w = 600) ∧
  (35 * x + 28 * z < 4200) ∧
  (x + z = 150) ∧ 
  (x + z) / 600 = 1 / 4 :=
by sorry

end NUMINAMATH_GPT_delta_max_success_ratio_l2081_208164


namespace NUMINAMATH_GPT_find_number_l2081_208123

theorem find_number (N p q : ℝ) (h₁ : N / p = 8) (h₂ : N / q = 18) (h₃ : p - q = 0.2777777777777778) : N = 4 :=
sorry

end NUMINAMATH_GPT_find_number_l2081_208123


namespace NUMINAMATH_GPT_second_man_start_time_l2081_208163

theorem second_man_start_time (P Q : Type) (departure_time_P departure_time_Q meeting_time arrival_time_P arrival_time_Q : ℕ) 
(distance speed : ℝ) (first_man_speed second_man_speed : ℕ → ℝ)
(h1 : departure_time_P = 6) 
(h2 : arrival_time_Q = 10) 
(h3 : arrival_time_P = 12) 
(h4 : meeting_time = 9) 
(h5 : ∀ t, 0 ≤ t ∧ t ≤ 4 → first_man_speed t = distance / 4)
(h6 : ∀ t, second_man_speed t = distance / 4)
(h7 : ∀ t, second_man_speed t * (meeting_time - t) = (3 * distance / 4))
: departure_time_Q = departure_time_P :=
by 
  sorry

end NUMINAMATH_GPT_second_man_start_time_l2081_208163


namespace NUMINAMATH_GPT_trigonometric_expression_evaluation_l2081_208168

theorem trigonometric_expression_evaluation
  (α : ℝ)
  (h1 : Real.tan α = -3 / 4) :
  (3 * Real.sin (α / 2) ^ 2 + 
   2 * Real.sin (α / 2) * Real.cos (α / 2) + 
   Real.cos (α / 2) ^ 2 - 2) / 
  (Real.sin (π / 2 + α) * Real.tan (-3 * π + α) + 
   Real.cos (6 * π - α)) = -7 := 
by 
  sorry
  -- This will skip the proof and ensure the Lean code can be built successfully.

end NUMINAMATH_GPT_trigonometric_expression_evaluation_l2081_208168


namespace NUMINAMATH_GPT_digits_base8_2015_l2081_208136

theorem digits_base8_2015 : ∃ n : Nat, (8^n ≤ 2015 ∧ 2015 < 8^(n+1)) ∧ n + 1 = 4 := 
by 
  sorry

end NUMINAMATH_GPT_digits_base8_2015_l2081_208136


namespace NUMINAMATH_GPT_hyperbola_eccentricity_l2081_208109

theorem hyperbola_eccentricity (a b c : ℚ) (h1 : (c : ℚ) = 5)
  (h2 : (b / a) = 3 / 4) (h3 : c^2 = a^2 + b^2) :
  (c / a : ℚ) = 5 / 4 :=
by
  sorry

end NUMINAMATH_GPT_hyperbola_eccentricity_l2081_208109


namespace NUMINAMATH_GPT_max_alpha_for_2_alpha_divides_3n_plus_1_l2081_208110

theorem max_alpha_for_2_alpha_divides_3n_plus_1 (n : ℕ) (hn : n > 0) : ∃ α : ℕ, (2 ^ α ∣ (3 ^ n + 1)) ∧ ¬ (2 ^ (α + 1) ∣ (3 ^ n + 1)) ∧ α = 1 :=
by
  sorry

end NUMINAMATH_GPT_max_alpha_for_2_alpha_divides_3n_plus_1_l2081_208110


namespace NUMINAMATH_GPT_vector_line_form_to_slope_intercept_l2081_208159

variable (x y : ℝ)

theorem vector_line_form_to_slope_intercept :
  (∀ (x y : ℝ), ((-1) * (x - 3) + 2 * (y + 4) = 0) ↔ (y = (-1/2) * x - 11/2)) :=
by
  sorry

end NUMINAMATH_GPT_vector_line_form_to_slope_intercept_l2081_208159


namespace NUMINAMATH_GPT_find_parameters_l2081_208167

theorem find_parameters (s h : ℝ) :
  (∀ (x y t : ℝ), (x = s + 3 * t) ∧ (y = 2 + h * t) ∧ (y = 5 * x - 7)) → (s = 9 / 5 ∧ h = 15) :=
by
  sorry

end NUMINAMATH_GPT_find_parameters_l2081_208167


namespace NUMINAMATH_GPT_Martha_reading_challenge_l2081_208148

theorem Martha_reading_challenge :
  ∀ x : ℕ,
  (12 + 18 + 14 + 20 + 11 + 13 + 19 + 15 + 17 + x) / 10 = 15 ↔ x = 11 :=
by sorry

end NUMINAMATH_GPT_Martha_reading_challenge_l2081_208148


namespace NUMINAMATH_GPT_binomial_12_10_eq_66_l2081_208103

theorem binomial_12_10_eq_66 :
  Nat.choose 12 10 = 66 := by
  sorry

end NUMINAMATH_GPT_binomial_12_10_eq_66_l2081_208103


namespace NUMINAMATH_GPT_Nina_money_l2081_208197

theorem Nina_money : ∃ (M : ℝ) (W : ℝ), M = 10 * W ∧ M = 14 * (W - 3) ∧ M = 105 :=
by
  sorry

end NUMINAMATH_GPT_Nina_money_l2081_208197


namespace NUMINAMATH_GPT_k_squared_minus_3k_minus_4_l2081_208125

theorem k_squared_minus_3k_minus_4 (a b c d k : ℚ)
  (h₁ : (2 * a) / (b + c + d) = k)
  (h₂ : (2 * b) / (a + c + d) = k)
  (h₃ : (2 * c) / (a + b + d) = k)
  (h₄ : (2 * d) / (a + b + c) = k) :
  k^2 - 3 * k - 4 = -50 / 9 ∨ k^2 - 3 * k - 4 = 6 :=
  sorry

end NUMINAMATH_GPT_k_squared_minus_3k_minus_4_l2081_208125


namespace NUMINAMATH_GPT_fraction_simplification_l2081_208187

theorem fraction_simplification (a b d : ℝ) (h : a^2 + d^2 - b^2 + 2 * a * d ≠ 0) :
  (a^2 + b^2 + d^2 + 2 * b * d) / (a^2 + d^2 - b^2 + 2 * a * d) = (a^2 + (b + d)^2) / ((a + d)^2 + a^2 - b^2) :=
sorry

end NUMINAMATH_GPT_fraction_simplification_l2081_208187


namespace NUMINAMATH_GPT_gunther_cleaning_free_time_l2081_208116

theorem gunther_cleaning_free_time :
  let vacuum := 45
  let dusting := 60
  let mopping := 30
  let bathroom := 40
  let windows := 15
  let brushing_per_cat := 5
  let cats := 4

  let free_time_hours := 4
  let free_time_minutes := 25

  let cleaning_time := vacuum + dusting + mopping + bathroom + windows + (brushing_per_cat * cats)
  let free_time_total := (free_time_hours * 60) + free_time_minutes

  free_time_total - cleaning_time = 55 :=
by
  sorry

end NUMINAMATH_GPT_gunther_cleaning_free_time_l2081_208116


namespace NUMINAMATH_GPT_backup_settings_required_l2081_208172

-- Definitions for the given conditions
def weight_of_silverware_piece : ℕ := 4
def pieces_of_silverware_per_setting : ℕ := 3
def weight_of_plate : ℕ := 12
def plates_per_setting : ℕ := 2
def tables : ℕ := 15
def settings_per_table : ℕ := 8
def total_weight_ounces : ℕ := 5040

-- Statement to prove
theorem backup_settings_required :
  (total_weight_ounces - 
     (tables * settings_per_table) * 
       (pieces_of_silverware_per_setting * weight_of_silverware_piece + 
        plates_per_setting * weight_of_plate)) /
  (pieces_of_silverware_per_setting * weight_of_silverware_piece + 
   plates_per_setting * weight_of_plate) = 20 := 
by sorry

end NUMINAMATH_GPT_backup_settings_required_l2081_208172


namespace NUMINAMATH_GPT_cos_alpha_value_l2081_208179

theorem cos_alpha_value (θ α : Real) (P : Real × Real)
  (hP : P = (-3/5, 4/5))
  (hθ : θ = Real.arccos (-3/5))
  (hαθ : α = θ - Real.pi / 3) :
  Real.cos α = (4 * Real.sqrt 3 - 3) / 10 := 
by 
  sorry

end NUMINAMATH_GPT_cos_alpha_value_l2081_208179


namespace NUMINAMATH_GPT_converse_of_statement_l2081_208166

variables (a b : ℝ)

theorem converse_of_statement :
  (a + b ≤ 2) → (a ≤ 1 ∨ b ≤ 1) :=
by
  sorry

end NUMINAMATH_GPT_converse_of_statement_l2081_208166


namespace NUMINAMATH_GPT_neznaika_mistake_l2081_208114

-- Let's define the conditions
variables {X A Y M E O U : ℕ} -- Represents distinct digits

-- Ascending order of the numbers
variables (XA AY AX OY EM EY MU : ℕ)
  (h1 : XA < AY)
  (h2 : AY < AX)
  (h3 : AX < OY)
  (h4 : OY < EM)
  (h5 : EM < EY)
  (h6 : EY < MU)

-- Identical digits replaced with the same letters
variables (h7 : XA = 10 * X + A)
  (h8 : AY = 10 * A + Y)
  (h9 : AX = 10 * A + X)
  (h10 : OY = 10 * O + Y)
  (h11 : EM = 10 * E + M)
  (h12 : EY = 10 * E + Y)
  (h13 : MU = 10 * M + U)

-- Each letter represents a different digit
variables (h_distinct : X ≠ A ∧ X ≠ Y ∧ X ≠ M ∧ X ≠ E ∧ X ≠ O ∧ X ≠ U ∧
                       A ≠ Y ∧ A ≠ M ∧ A ≠ E ∧ A ≠ O ∧ A ≠ U ∧
                       Y ≠ M ∧ Y ≠ E ∧ Y ≠ O ∧ Y ≠ U ∧
                       M ≠ E ∧ M ≠ O ∧ M ≠ U ∧
                       E ≠ O ∧ E ≠ U ∧
                       O ≠ U)

-- Prove Neznaika made a mistake
theorem neznaika_mistake : false :=
by
  -- Here we'll reach a contradiction, proving false.
  sorry

end NUMINAMATH_GPT_neznaika_mistake_l2081_208114


namespace NUMINAMATH_GPT_allan_plums_l2081_208135

theorem allan_plums (A : ℕ) (h1 : 7 - A = 3) : A = 4 :=
sorry

end NUMINAMATH_GPT_allan_plums_l2081_208135


namespace NUMINAMATH_GPT_expression_eval_l2081_208171

theorem expression_eval (a b c d : ℝ) :
  a * b + c - d = a * (b + c - d) :=
sorry

end NUMINAMATH_GPT_expression_eval_l2081_208171


namespace NUMINAMATH_GPT_oscar_leap_more_than_piper_hop_l2081_208199

noncomputable def difference_leap_hop : ℝ :=
let number_of_poles := 51
let total_distance := 7920 -- in feet
let Elmer_strides_per_gap := 44
let Oscar_leaps_per_gap := 15
let Piper_hops_per_gap := 22
let number_of_gaps := number_of_poles - 1
let Elmer_total_strides := Elmer_strides_per_gap * number_of_gaps
let Oscar_total_leaps := Oscar_leaps_per_gap * number_of_gaps
let Piper_total_hops := Piper_hops_per_gap * number_of_gaps
let Elmer_stride_length := total_distance / Elmer_total_strides
let Oscar_leap_length := total_distance / Oscar_total_leaps
let Piper_hop_length := total_distance / Piper_total_hops
Oscar_leap_length - Piper_hop_length

theorem oscar_leap_more_than_piper_hop :
  difference_leap_hop = 3.36 := by
  sorry

end NUMINAMATH_GPT_oscar_leap_more_than_piper_hop_l2081_208199


namespace NUMINAMATH_GPT_no_symmetry_line_for_exponential_l2081_208175

theorem no_symmetry_line_for_exponential : ¬ ∃ l : ℝ → ℝ, ∀ x : ℝ, (2 ^ x) = l (2 ^ (2 * l x - x)) := 
sorry

end NUMINAMATH_GPT_no_symmetry_line_for_exponential_l2081_208175


namespace NUMINAMATH_GPT_range_of_a_l2081_208107

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, (x < 3) → (4 * a * x + 4 * (a - 3)) ≤ 0) ↔ (0 ≤ a ∧ a ≤ 3 / 4) :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l2081_208107


namespace NUMINAMATH_GPT_terminal_side_third_quadrant_l2081_208132

noncomputable def angle_alpha : ℝ := (7 * Real.pi) / 5

def is_in_third_quadrant (angle : ℝ) : Prop :=
  ∃ k : ℤ, (3 * Real.pi) / 2 < angle + 2 * k * Real.pi ∧ angle + 2 * k * Real.pi < 2 * Real.pi

theorem terminal_side_third_quadrant : is_in_third_quadrant angle_alpha :=
sorry

end NUMINAMATH_GPT_terminal_side_third_quadrant_l2081_208132


namespace NUMINAMATH_GPT_factor_and_divisor_statements_l2081_208128

theorem factor_and_divisor_statements :
  (∃ n : ℕ, 25 = 5 * n) ∧
  ((∃ n : ℕ, 209 = 19 * n) ∧ ¬ (∃ n : ℕ, 63 = 19 * n)) ∧
  (∃ n : ℕ, 180 = 9 * n) :=
by
  sorry

end NUMINAMATH_GPT_factor_and_divisor_statements_l2081_208128


namespace NUMINAMATH_GPT_total_cards_is_56_l2081_208149

-- Let n be the number of Pokemon cards each person has
def n : Nat := 14

-- Let k be the number of people
def k : Nat := 4

-- Total number of Pokemon cards
def total_cards : Nat := n * k

-- Prove that the total number of Pokemon cards is 56
theorem total_cards_is_56 : total_cards = 56 := by
  sorry

end NUMINAMATH_GPT_total_cards_is_56_l2081_208149


namespace NUMINAMATH_GPT_angle_sum_around_point_l2081_208122

theorem angle_sum_around_point (x y : ℝ) (h1 : 0 ≤ x) (h2 : 0 ≤ y) : 
    x + y + 130 = 360 → x + y = 230 := by
  sorry

end NUMINAMATH_GPT_angle_sum_around_point_l2081_208122


namespace NUMINAMATH_GPT_min_value_of_expression_l2081_208134

theorem min_value_of_expression (m n : ℕ) (hm : 0 < m) (hn : 0 < n)
  (hpar : ∀ x y : ℝ, 2 * x + (n - 1) * y - 2 = 0 → ∃ c : ℝ, mx + ny + c = 0) :
  2 * m + n = 9 :=
by
  sorry

end NUMINAMATH_GPT_min_value_of_expression_l2081_208134


namespace NUMINAMATH_GPT_gcd_polynomial_l2081_208137

open Nat

theorem gcd_polynomial (b : ℤ) (hb : 1632 ∣ b) : gcd (b^2 + 11 * b + 30) (b + 6) = 6 := by
  sorry

end NUMINAMATH_GPT_gcd_polynomial_l2081_208137


namespace NUMINAMATH_GPT_calculate_price_l2081_208182

-- Define variables for prices
def sugar_price_in_terms_of_salt (T : ℝ) : ℝ := 2 * T
def rice_price_in_terms_of_salt (T : ℝ) : ℝ := 3 * T
def apple_price : ℝ := 1.50
def pepper_price : ℝ := 1.25

-- Define pricing conditions
def condition_1 (T : ℝ) : Prop :=
  5 * (sugar_price_in_terms_of_salt T) + 3 * T + 2 * (rice_price_in_terms_of_salt T) + 3 * apple_price + 4 * pepper_price = 35

def condition_2 (T : ℝ) : Prop :=
  4 * (sugar_price_in_terms_of_salt T) + 2 * T + 1 * (rice_price_in_terms_of_salt T) + 2 * apple_price + 3 * pepper_price = 24

-- Define final price calculation with discounts
def total_price (T : ℝ) : ℝ :=
  8 * (sugar_price_in_terms_of_salt T) * 0.9 +
  5 * T +
  (rice_price_in_terms_of_salt T + 3 * (rice_price_in_terms_of_salt T - 0.5)) +
  -- adding two free apples to the count
  5 * apple_price +
  6 * pepper_price

-- Main theorem to prove
theorem calculate_price (T : ℝ) (h1 : condition_1 T) (h2 : condition_2 T) :
  total_price T = 55.64 :=
sorry -- proof omitted

end NUMINAMATH_GPT_calculate_price_l2081_208182


namespace NUMINAMATH_GPT_speed_first_hour_l2081_208169

variable (x : ℕ)

-- Definitions based on conditions
def total_distance (x : ℕ) : ℕ := x + 50
def average_speed (x : ℕ) : Prop := (total_distance x) / 2 = 70

-- Theorem statement
theorem speed_first_hour : ∃ x, average_speed x ∧ x = 90 := by
  sorry

end NUMINAMATH_GPT_speed_first_hour_l2081_208169


namespace NUMINAMATH_GPT_calculate_pens_l2081_208121

theorem calculate_pens (P : ℕ) (Students : ℕ) (Pencils : ℕ) (h1 : Students = 40) (h2 : Pencils = 920) (h3 : ∃ k : ℕ, Pencils = Students * k) 
(h4 : ∃ m : ℕ, P = Students * m) : ∃ k : ℕ, P = 40 * k := by
  sorry

end NUMINAMATH_GPT_calculate_pens_l2081_208121


namespace NUMINAMATH_GPT_incorrect_conclusions_l2081_208192

theorem incorrect_conclusions
  (h1 : ∃ (y x : ℝ), (¬∃ a b : ℝ, a < 0 ∧ y = a * x + b) ∧ ∃ a b : ℝ, y = 2.347 * x - 6.423)
  (h2 : ∃ (y x : ℝ), (∃ a b : ℝ, a < 0 ∧ y = a * x + b) ∧ y = -3.476 * x + 5.648)
  (h3 : ∃ (y x : ℝ), (∃ a b : ℝ, a > 0 ∧ y = a * x + b) ∧ y = 5.437 * x + 8.493)
  (h4 : ∃ (y x : ℝ), (¬∃ a b : ℝ, a > 0 ∧ y = a * x + b) ∧ y = -4.326 * x - 4.578) :
  (∃ (y x : ℝ), y = 2.347 * x - 6.423 ∧ (¬∃ a b : ℝ, a < 0 ∧ y = a * x + b)) ∧
  (∃ (y x : ℝ), y = -4.326 * x - 4.578 ∧ (¬∃ a b : ℝ, a > 0 ∧ y = a * x + b)) :=
by {
  sorry
}

end NUMINAMATH_GPT_incorrect_conclusions_l2081_208192


namespace NUMINAMATH_GPT_logic_problem_l2081_208100

variables (p q : Prop)

theorem logic_problem (hnp : ¬ p) (hpq : ¬ (p ∧ q)) : ¬ (p ∨ q) ∨ (p ∨ q) :=
by 
  sorry

end NUMINAMATH_GPT_logic_problem_l2081_208100


namespace NUMINAMATH_GPT_axisymmetric_and_centrally_symmetric_l2081_208195

def Polygon := String

def EquilateralTriangle : Polygon := "EquilateralTriangle"
def Square : Polygon := "Square"
def RegularPentagon : Polygon := "RegularPentagon"
def RegularHexagon : Polygon := "RegularHexagon"

def is_axisymmetric (p : Polygon) : Prop := 
  p = EquilateralTriangle ∨ p = Square ∨ p = RegularPentagon ∨ p = RegularHexagon

def is_centrally_symmetric (p : Polygon) : Prop := 
  p = Square ∨ p = RegularHexagon

theorem axisymmetric_and_centrally_symmetric :
  {p : Polygon | is_axisymmetric p ∧ is_centrally_symmetric p} = {Square, RegularHexagon} :=
by
  sorry

end NUMINAMATH_GPT_axisymmetric_and_centrally_symmetric_l2081_208195


namespace NUMINAMATH_GPT_garden_area_increase_l2081_208155

-- Definitions corresponding to the conditions
def length := 40
def width := 20
def original_perimeter := 2 * (length + width)

-- Definition of the correct answer calculation
def original_area := length * width
def side_length := original_perimeter / 4
def new_area := side_length * side_length
def area_increase := new_area - original_area

-- The statement to be proven
theorem garden_area_increase : area_increase = 100 :=
by sorry

end NUMINAMATH_GPT_garden_area_increase_l2081_208155


namespace NUMINAMATH_GPT_solve_for_x_l2081_208144

-- Let us state and prove that x = 495 / 13 is a solution to the equation 3x + 5 = 500 - (4x + 6x)
theorem solve_for_x (x : ℝ) : 3 * x + 5 = 500 - (4 * x + 6 * x) → x = 495 / 13 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_x_l2081_208144


namespace NUMINAMATH_GPT_degrees_subtraction_l2081_208105

theorem degrees_subtraction :
  (108 * 3600 + 18 * 60 + 25) - (56 * 3600 + 23 * 60 + 32) = (51 * 3600 + 54 * 60 + 53) :=
by sorry

end NUMINAMATH_GPT_degrees_subtraction_l2081_208105


namespace NUMINAMATH_GPT_find_b_l2081_208185

theorem find_b (b x : ℝ) (h₁ : 5 * x + 3 = b * x - 22) (h₂ : x = 5) : b = 10 := 
by 
  sorry

end NUMINAMATH_GPT_find_b_l2081_208185


namespace NUMINAMATH_GPT_combined_age_l2081_208141

variable (m y o : ℕ)

noncomputable def younger_brother_age := 5

noncomputable def older_brother_age_based_on_younger := 3 * younger_brother_age

noncomputable def older_brother_age_based_on_michael (m : ℕ) := 1 + 2 * (m - 1)

theorem combined_age (m y o : ℕ) (h1 : y = younger_brother_age) (h2 : o = older_brother_age_based_on_younger) (h3 : o = older_brother_age_based_on_michael m) :
  y + o + m = 28 := by
  sorry

end NUMINAMATH_GPT_combined_age_l2081_208141


namespace NUMINAMATH_GPT_evaluate_expression_l2081_208124

variable (a : ℕ)

theorem evaluate_expression (h : a = 2) : a^3 * a^4 = 128 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l2081_208124


namespace NUMINAMATH_GPT_solve_equation_l2081_208180

theorem solve_equation :
  ∀ x : ℝ, x ≠ 1 → x ≠ 2 → (x + 1) / (x - 1) = 1 / (x - 2) + 1 → x = 3 := by
  sorry

end NUMINAMATH_GPT_solve_equation_l2081_208180


namespace NUMINAMATH_GPT_distribute_books_l2081_208181

theorem distribute_books : 
  let total_ways := 4^5
  let subtract_one_student_none := 4 * 3^5
  let add_two_students_none := 6 * 2^5
  total_ways - subtract_one_student_none + add_two_students_none = 240 :=
by
  -- Definitions based on conditions in a)
  let total_ways := 4^5
  let subtract_one_student_none := 4 * 3^5
  let add_two_students_none := 6 * 2^5

  -- The final calculation
  have h : total_ways - subtract_one_student_none + add_two_students_none = 240 := by sorry
  exact h

end NUMINAMATH_GPT_distribute_books_l2081_208181


namespace NUMINAMATH_GPT_value_of_expression_l2081_208130

theorem value_of_expression : 30 - 5^2 = 5 := by
  sorry

end NUMINAMATH_GPT_value_of_expression_l2081_208130


namespace NUMINAMATH_GPT_fried_busy_frog_l2081_208106

open ProbabilityTheory

def initial_position : (ℤ × ℤ) := (0, 0)

def possible_moves : List (ℤ × ℤ) := [(0, 0), (1, 0), (0, 1)]

def p (n : ℕ) (pos : ℤ × ℤ) : ℚ :=
  if pos = initial_position then 1 else 0

noncomputable def transition (n : ℕ) (pos : ℤ × ℤ) : ℚ :=
  if pos = (0, 0) then 1/3 * p n (0, 0)
  else if pos = (0, 1) then 1/3 * p n (0, 0) + 1/3 * p n (0, 1)
  else if pos = (1, 0) then 1/3 * p n (0, 0) + 1/3 * p n (1, 0)
  else 0

noncomputable def p_1 (pos : ℤ × ℤ) : ℚ := transition 0 pos

noncomputable def p_2 (pos : ℤ × ℤ) : ℚ := transition 1 pos

noncomputable def p_3 (pos : ℤ × ℤ) : ℚ := transition 2 pos

theorem fried_busy_frog :
  p_3 (0, 0) = 1/27 :=
by
  sorry

end NUMINAMATH_GPT_fried_busy_frog_l2081_208106


namespace NUMINAMATH_GPT_piggy_bank_after_8_weeks_l2081_208118

-- Define initial amount in the piggy bank
def initial_amount : ℝ := 43

-- Define weekly allowance amount
def weekly_allowance : ℝ := 10

-- Define fraction of allowance Jack saves
def saving_fraction : ℝ := 0.5

-- Define number of weeks
def number_of_weeks : ℕ := 8

-- Define weekly savings amount
def weekly_savings : ℝ := saving_fraction * weekly_allowance

-- Define total savings after a given number of weeks
def total_savings (weeks : ℕ) : ℝ := weeks * weekly_savings

-- Define the final amount in the piggy bank after a given number of weeks
def final_amount (weeks : ℕ) : ℝ := initial_amount + total_savings weeks

-- Theorem: Prove that final amount in piggy bank after 8 weeks is $83
theorem piggy_bank_after_8_weeks : final_amount number_of_weeks = 83 := by
  sorry

end NUMINAMATH_GPT_piggy_bank_after_8_weeks_l2081_208118


namespace NUMINAMATH_GPT_arithmetic_sequence_value_l2081_208174

theorem arithmetic_sequence_value 
  (a : ℕ → ℤ) 
  (d : ℤ) 
  (h1 : ∀ n, a n = a 1 + (n - 1) * d)
  (h2 : 4 * a 3 + a 11 - 3 * a 5 = 10) : 
  (1 / 5 * a 4 = 1) := 
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_value_l2081_208174


namespace NUMINAMATH_GPT_people_in_line_l2081_208145

theorem people_in_line (initially_in_line : ℕ) (left_line : ℕ) (after_joined_line : ℕ) 
  (h1 : initially_in_line = 12) (h2 : left_line = 10) (h3 : after_joined_line = 17) : 
  initially_in_line - left_line + 15 = after_joined_line := by
  sorry

end NUMINAMATH_GPT_people_in_line_l2081_208145


namespace NUMINAMATH_GPT_option_D_not_necessarily_true_l2081_208113

variable {a b c : ℝ}

theorem option_D_not_necessarily_true 
  (h1 : c < b)
  (h2 : b < a)
  (h3 : a * c < 0) : ¬((c * b^2 < a * b^2) ↔ (b ≠ 0 ∨ b = 0 ∧ (c * b^2 < a * b^2))) := 
sorry

end NUMINAMATH_GPT_option_D_not_necessarily_true_l2081_208113


namespace NUMINAMATH_GPT_clock_overlap_24_hours_l2081_208184

theorem clock_overlap_24_hours (hour_rotations : ℕ) (minute_rotations : ℕ) 
  (h_hour_rotations: hour_rotations = 2) 
  (h_minute_rotations: minute_rotations = 24) : 
  ∃ (overlaps : ℕ), overlaps = 22 := 
by 
  sorry

end NUMINAMATH_GPT_clock_overlap_24_hours_l2081_208184


namespace NUMINAMATH_GPT_solve_system_of_equations_l2081_208117

theorem solve_system_of_equations : ∃ x y : ℤ, 3 * x - 2 * y = 6 ∧ 2 * x + 3 * y = 17 ∧ x = 4 ∧ y = 3 :=
by
  sorry

end NUMINAMATH_GPT_solve_system_of_equations_l2081_208117


namespace NUMINAMATH_GPT_smallest_positive_angle_terminal_side_eq_l2081_208186

theorem smallest_positive_angle_terminal_side_eq (n : ℤ) :
  (0 ≤ n % 360 ∧ n % 360 < 360) → (∃ k : ℤ, n = -2015 + k * 360 ) → n % 360 = 145 :=
by
  sorry

end NUMINAMATH_GPT_smallest_positive_angle_terminal_side_eq_l2081_208186


namespace NUMINAMATH_GPT_find_pairs_l2081_208138

theorem find_pairs (a b : ℕ) : 
  (∃ (a b : ℕ), 
    (∃ (k₁ k₂ : ℤ), 
      a^2 + b = k₁ * (b^2 - a) ∧ b^2 + a = k₂ * (a^2 - b))) 
      ↔ (a, b) = (1, 2) ∨ (a, b) = (2, 1) ∨ (a, b) = (2, 2) ∨ (a, b) = (2, 3) ∨ (a, b) = (3, 2) ∨ (a, b) = (3, 3) := sorry

end NUMINAMATH_GPT_find_pairs_l2081_208138


namespace NUMINAMATH_GPT_ellipse_properties_l2081_208170

-- Define the ellipse E with its given properties
def is_ellipse (x y : ℝ) : Prop :=
  x^2 / 4 + y^2 / 3 = 1

-- Define properties related to the intersection points and lines
def intersects (l : ℝ → ℝ) (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  l (-1) = 0 ∧ 
  is_ellipse x₁ (l x₁) ∧ 
  is_ellipse x₂ (l x₂) ∧ 
  y₁ = l x₁ ∧ 
  y₂ = l x₂

def perpendicular_lines (l1 l2 : ℝ → ℝ) : Prop :=
  ∀ x, l1 x * l2 x = -1

-- Define the main theorem
theorem ellipse_properties :
  (∀ (x y : ℝ), is_ellipse x y) →
  (∀ (l1 l2 : ℝ → ℝ) 
     (A B C D : ℝ × ℝ),
      intersects l1 A.1 A.2 B.1 B.2 → 
      intersects l2 C.1 C.2 D.1 D.2 → 
      perpendicular_lines l1 l2 → 
      12 * (|A.1 - B.1| + |C.1 - D.1|) = 7 * |A.1 - B.1| * |C.1 - D.1|) :=
by 
  sorry

end NUMINAMATH_GPT_ellipse_properties_l2081_208170


namespace NUMINAMATH_GPT_evaluate_fraction_l2081_208127

variable (a b x : ℝ)
variable (h1 : a ≠ b)
variable (h2 : b ≠ 0)
variable (h3 : x = a / b)

theorem evaluate_fraction :
  (a^2 + b^2) / (a^2 - b^2) = (x^2 + 1) / (x^2 - 1) :=
by
  sorry

end NUMINAMATH_GPT_evaluate_fraction_l2081_208127


namespace NUMINAMATH_GPT_proof_of_diagonal_length_l2081_208101

noncomputable def length_of_diagonal (d : ℝ) : Prop :=
  d^2 = 325 ∧ 17^2 + 36 = 325

theorem proof_of_diagonal_length (d : ℝ) : length_of_diagonal d → d = 5 * Real.sqrt 13 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_proof_of_diagonal_length_l2081_208101


namespace NUMINAMATH_GPT_difference_in_ages_27_l2081_208120

def conditions (a b : ℕ) : Prop :=
  10 * b + a = (1 / 2) * (10 * a + b) + 6 ∧
  10 * a + b + 2 = 5 * (10 * b + a - 4)

theorem difference_in_ages_27 {a b : ℕ} (h : conditions a b) :
  (10 * a + b) - (10 * b + a) = 27 :=
sorry

end NUMINAMATH_GPT_difference_in_ages_27_l2081_208120
