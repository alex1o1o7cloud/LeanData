import Mathlib

namespace NUMINAMATH_GPT_power_function_value_l1191_119101

theorem power_function_value
  (α : ℝ)
  (h : 2^α = Real.sqrt 2) :
  (4 : ℝ) ^ α = 2 :=
by {
  sorry
}

end NUMINAMATH_GPT_power_function_value_l1191_119101


namespace NUMINAMATH_GPT_valid_punching_settings_l1191_119138

theorem valid_punching_settings :
  let total_patterns := 2^9
  let symmetric_patterns := 2^6
  total_patterns - symmetric_patterns = 448 :=
by
  sorry

end NUMINAMATH_GPT_valid_punching_settings_l1191_119138


namespace NUMINAMATH_GPT_simple_interest_is_correct_l1191_119107

def Principal : ℝ := 10000
def Rate : ℝ := 0.09
def Time : ℝ := 1

theorem simple_interest_is_correct :
  Principal * Rate * Time = 900 := by
  sorry

end NUMINAMATH_GPT_simple_interest_is_correct_l1191_119107


namespace NUMINAMATH_GPT_six_box_four_div_three_eight_box_two_div_four_l1191_119139

def fills_middle_zero (d : Nat) : Prop :=
  d < 3

def fills_last_zero (d : Nat) : Prop :=
  (80 + d) % 4 = 0

theorem six_box_four_div_three {d : Nat} : fills_middle_zero d → ((600 + d * 10 + 4) / 3) % 100 / 10 = 0 :=
  sorry

theorem eight_box_two_div_four {d : Nat} : fills_last_zero d → ((800 + d * 10 + 2) / 4) % 10 = 0 :=
  sorry

end NUMINAMATH_GPT_six_box_four_div_three_eight_box_two_div_four_l1191_119139


namespace NUMINAMATH_GPT_gcd_polynomial_997_l1191_119112

theorem gcd_polynomial_997 (b : ℤ) (h : ∃ k : ℤ, b = 997 * k ∧ k % 2 = 1) :
  Int.gcd (3 * b ^ 2 + 17 * b + 31) (b + 7) = 1 := by
  sorry

end NUMINAMATH_GPT_gcd_polynomial_997_l1191_119112


namespace NUMINAMATH_GPT_initial_pairs_l1191_119121

variable (p1 p2 p3 p4 p_initial : ℕ)

def week1_pairs := 12
def week2_pairs := week1_pairs + 4
def week3_pairs := (week1_pairs + week2_pairs) / 2
def week4_pairs := week3_pairs - 3
def total_pairs := 57

theorem initial_pairs :
  let p1 := week1_pairs
  let p2 := week2_pairs
  let p3 := week3_pairs
  let p4 := week4_pairs
  p1 + p2 + p3 + p4 + p_initial = 57 → p_initial = 4 :=
by
  sorry

end NUMINAMATH_GPT_initial_pairs_l1191_119121


namespace NUMINAMATH_GPT_option_b_correct_l1191_119148

theorem option_b_correct (a b m : ℝ) (h1 : a > b) (h2 : b > 0) (h3: a ≠ 1) (h4: b ≠ 1) (h5 : 0 < m) (h6 : m < 1) :
  m^a < m^b :=
sorry

end NUMINAMATH_GPT_option_b_correct_l1191_119148


namespace NUMINAMATH_GPT_third_term_of_geometric_sequence_l1191_119154

theorem third_term_of_geometric_sequence
  (a₁ : ℕ) (a₄ : ℕ)
  (h1 : a₁ = 5)
  (h4 : a₄ = 320) :
  ∃ a₃ : ℕ, a₃ = 80 :=
by
  sorry

end NUMINAMATH_GPT_third_term_of_geometric_sequence_l1191_119154


namespace NUMINAMATH_GPT_increase_in_rectangle_area_l1191_119171

theorem increase_in_rectangle_area (L B : ℝ) :
  let L' := 1.11 * L
  let B' := 1.22 * B
  let original_area := L * B
  let new_area := L' * B'
  let area_increase := new_area - original_area
  let percentage_increase := (area_increase / original_area) * 100
  percentage_increase = 35.42 :=
by
  sorry

end NUMINAMATH_GPT_increase_in_rectangle_area_l1191_119171


namespace NUMINAMATH_GPT_find_a_l1191_119118

-- Define the function f(x)
def f (a : ℚ) (x : ℚ) : ℚ := x^2 + (2 * a + 3) * x + (a^2 + 1)

-- State that the discriminant of f(x) is non-negative
def discriminant_nonnegative (a : ℚ) : Prop :=
  let Δ := (2 * a + 3)^2 - 4 * (a^2 + 1)
  Δ ≥ 0

-- Final statement expressing the final condition on a and the desired result |p| + |q|
theorem find_a (a : ℚ) (p q : ℤ) (h_relprime : Int.gcd p q = 1) (h_eq : a = -5 / 12) (h_abs : p * q = -5 * 12) :
  discriminant_nonnegative a →
  |p| + |q| = 17 :=
by sorry

end NUMINAMATH_GPT_find_a_l1191_119118


namespace NUMINAMATH_GPT_sausages_placement_and_path_length_l1191_119185

variables {a b x y : ℝ} (h1 : 2 * x = y) (h2 : b = 1 / (x + y))
variables (h3 : (x + y) = 8 * a) (h4 : x = 1.4 * y)

theorem sausages_placement_and_path_length (h1 : 2 * x = y) (h2 : b = 1 / (x + y))
(h3 : (x + y) = 8 * a) (h4 : x = 1.4 * y) : 
  x < y ∧ (x / y) = 1.4 :=
by {
  sorry
}

end NUMINAMATH_GPT_sausages_placement_and_path_length_l1191_119185


namespace NUMINAMATH_GPT_min_value_gx2_plus_fx_l1191_119176

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := a * x + b
noncomputable def g (a c : ℝ) (x : ℝ) : ℝ := a * x + c

theorem min_value_gx2_plus_fx (a b c : ℝ) (h_a : a ≠ 0)
    (h_min_fx_gx : ∀ x : ℝ, (f a b x)^2 + g a c x ≥ -6) :
    ∃ x : ℝ, (g a c x)^2 + f a b x = 11/2 := sorry

end NUMINAMATH_GPT_min_value_gx2_plus_fx_l1191_119176


namespace NUMINAMATH_GPT_quadratic_inequality_solution_l1191_119152

theorem quadratic_inequality_solution 
  (a b c : ℝ)
  (h1 : a < 0)
  (h2 : 1 + 2 = b / a)
  (h3 : 1 * 2 = c / a) :
  ∀ x : ℝ, cx^2 + bx + a ≤ 0 ↔ x ≤ -1 ∨ x ≥ -1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_inequality_solution_l1191_119152


namespace NUMINAMATH_GPT_digit_equation_l1191_119179

-- Define the digits for the letters L, O, V, E, and S in base 10.
def digit_L := 4
def digit_O := 3
def digit_V := 7
def digit_E := 8
def digit_S := 6

-- Define the numeral representations.
def LOVE := digit_L * 1000 + digit_O * 100 + digit_V * 10 + digit_E
def EVOL := digit_E * 1000 + digit_V * 100 + digit_O * 10 + digit_L
def SOLVES := digit_S * 100000 + digit_O * 10000 + digit_L * 1000 + digit_V * 100 + digit_E * 10 + digit_S

-- Prove that LOVE + EVOL + LOVE = SOLVES in base 10.
theorem digit_equation :
  LOVE + EVOL + LOVE = SOLVES :=
by
  -- Proof is omitted; include a proper proof in your verification process.
  sorry

end NUMINAMATH_GPT_digit_equation_l1191_119179


namespace NUMINAMATH_GPT_value_of_A_l1191_119136

theorem value_of_A (h p a c k e : ℤ) 
  (H : h = 8)
  (PACK : p + a + c + k = 50)
  (PECK : p + e + c + k = 54)
  (CAKE : c + a + k + e = 40) : 
  a = 25 :=
by 
  sorry

end NUMINAMATH_GPT_value_of_A_l1191_119136


namespace NUMINAMATH_GPT_option_C_cannot_form_right_triangle_l1191_119169

def is_right_triangle_sides (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2

theorem option_C_cannot_form_right_triangle :
  ¬ (is_right_triangle_sides 1.5 2 3) :=
by
  -- This is intentionally left incomplete as per instructions
  sorry

end NUMINAMATH_GPT_option_C_cannot_form_right_triangle_l1191_119169


namespace NUMINAMATH_GPT_acute_angle_at_7_20_is_100_degrees_l1191_119155

theorem acute_angle_at_7_20_is_100_degrees :
  let minute_hand_angle := 4 * 30 -- angle of the minute hand (in degrees)
  let hour_hand_progress := 20 / 60 -- progress of hour hand between 7 and 8
  let hour_hand_angle := 7 * 30 + hour_hand_progress * 30 -- angle of the hour hand (in degrees)

  ∃ angle_acute : ℝ, 
  angle_acute = abs (minute_hand_angle - hour_hand_angle) ∧
  angle_acute = 100 :=
by
  sorry

end NUMINAMATH_GPT_acute_angle_at_7_20_is_100_degrees_l1191_119155


namespace NUMINAMATH_GPT_simplify_sqrt_expression_l1191_119180

theorem simplify_sqrt_expression :
  (Real.sqrt (3 * 5) * Real.sqrt (3^3 * 5^3)) = 225 := 
by 
  sorry

end NUMINAMATH_GPT_simplify_sqrt_expression_l1191_119180


namespace NUMINAMATH_GPT_sum_of_angles_l1191_119140

theorem sum_of_angles (A B C D E F : ℝ)
  (h1 : A + B + C = 180) 
  (h2 : D + E + F = 180) : 
  A + B + C + D + E + F = 360 := 
by 
  sorry

end NUMINAMATH_GPT_sum_of_angles_l1191_119140


namespace NUMINAMATH_GPT_chocolate_bars_count_l1191_119167

theorem chocolate_bars_count (milk_chocolate dark_chocolate almond_chocolate white_chocolate : ℕ)
    (h_milk : milk_chocolate = 25)
    (h_almond : almond_chocolate = 25)
    (h_white : white_chocolate = 25)
    (h_percent : milk_chocolate = almond_chocolate ∧ almond_chocolate = white_chocolate ∧ white_chocolate = dark_chocolate) :
    dark_chocolate = 25 := by
  sorry

end NUMINAMATH_GPT_chocolate_bars_count_l1191_119167


namespace NUMINAMATH_GPT_polynomial_example_properties_l1191_119193

open Polynomial

noncomputable def polynomial_example : Polynomial ℚ :=
- (1 / 2) * (X^2 + X - 1) * (X^2 + 1)

theorem polynomial_example_properties :
  ∃ P : Polynomial ℚ, (X^2 + 1) ∣ P ∧ (X^3 + 1) ∣ (P - 1) :=
by
  use polynomial_example
  -- To complete the proof, one would typically verify the divisibility properties here.
  sorry

end NUMINAMATH_GPT_polynomial_example_properties_l1191_119193


namespace NUMINAMATH_GPT_strawberries_count_l1191_119181

def strawberries_total (J M Z : ℕ) : ℕ :=
  J + M + Z

theorem strawberries_count (J M Z : ℕ) (h1 : J + M = 350) (h2 : M + Z = 250) (h3 : Z = 200) : 
  strawberries_total J M Z = 550 :=
by
  sorry

end NUMINAMATH_GPT_strawberries_count_l1191_119181


namespace NUMINAMATH_GPT_tangent_parallel_to_given_line_l1191_119183

theorem tangent_parallel_to_given_line (a : ℝ) : 
  let y := λ x : ℝ => x^2 + a / x
  let y' := λ x : ℝ => (deriv y) x
  y' 1 = 2 
  → a = 0 := by
  -- y'(1) is the derivative of y at x=1
  sorry

end NUMINAMATH_GPT_tangent_parallel_to_given_line_l1191_119183


namespace NUMINAMATH_GPT_fraction_meaningful_l1191_119146

theorem fraction_meaningful (x : ℝ) : 2 * x - 1 ≠ 0 ↔ x ≠ 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_fraction_meaningful_l1191_119146


namespace NUMINAMATH_GPT_linear_function_result_l1191_119124

variable {R : Type*} [LinearOrderedField R]

noncomputable def linear_function (g : R → R) : Prop :=
  ∃ (a b : R), ∀ x, g x = a * x + b

theorem linear_function_result (g : R → R) (h_lin : linear_function g) (h : g 5 - g 1 = 16) : g 13 - g 1 = 48 :=
  by
  sorry

end NUMINAMATH_GPT_linear_function_result_l1191_119124


namespace NUMINAMATH_GPT_first_term_of_infinite_geometric_series_l1191_119160

theorem first_term_of_infinite_geometric_series (a : ℝ) (r : ℝ) (S : ℝ) 
  (h1 : r = -1/3) 
  (h2 : S = 9) 
  (h3 : S = a / (1 - r)) : a = 12 := 
sorry

end NUMINAMATH_GPT_first_term_of_infinite_geometric_series_l1191_119160


namespace NUMINAMATH_GPT_range_a_f_x_neg_l1191_119105

noncomputable def f (a x : ℝ) : ℝ := x^2 + (2 * a - 1) * x - 3

theorem range_a_f_x_neg (a : ℝ) : 
  (∃ x : ℝ, 1 < x ∧ f a x < 0) → a < 3 / 2 := sorry

end NUMINAMATH_GPT_range_a_f_x_neg_l1191_119105


namespace NUMINAMATH_GPT_find_coordinates_l1191_119141

structure Point :=
  (x : ℝ)
  (y : ℝ)

def A : Point := ⟨2, 3⟩
def B : Point := ⟨4, -3⟩

def satisfiesCondition (A B P : Point) : Prop :=
  2 * (P.x - A.x) = (B.x - P.x) ∧ 2 * (P.y - A.y) = (B.y - P.y)

theorem find_coordinates (P : Point) (h : satisfiesCondition A B P) : 
  P = ⟨6, -9⟩ :=
  sorry

end NUMINAMATH_GPT_find_coordinates_l1191_119141


namespace NUMINAMATH_GPT_find_fraction_l1191_119125

theorem find_fraction (x f : ℝ) (h₁ : x = 140) (h₂ : 0.65 * x = f * x - 21) : f = 0.8 :=
by
  sorry

end NUMINAMATH_GPT_find_fraction_l1191_119125


namespace NUMINAMATH_GPT_radius_of_circle_l1191_119127

theorem radius_of_circle (r : ℝ) (h : π * r^2 = 81 * π) : r = 9 :=
by
  sorry

end NUMINAMATH_GPT_radius_of_circle_l1191_119127


namespace NUMINAMATH_GPT_find_integer_m_l1191_119164

theorem find_integer_m (m : ℤ) :
  (∃! x : ℤ, |2 * x - m| ≤ 1 ∧ x = 2) → m = 4 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_find_integer_m_l1191_119164


namespace NUMINAMATH_GPT_compare_rat_neg_l1191_119109

-- Define the numbers
def a : ℚ := -3 / 4
def b : ℚ := -4 / 5

-- State the theorem
theorem compare_rat_neg : a > b :=
by
  -- Proof to be added here
  sorry

end NUMINAMATH_GPT_compare_rat_neg_l1191_119109


namespace NUMINAMATH_GPT_largest_additional_plates_l1191_119115

theorem largest_additional_plates
  (initial_first_set_size : ℕ)
  (initial_second_set_size : ℕ)
  (initial_third_set_size : ℕ)
  (new_letters : ℕ)
  (constraint : 1 ≤ initial_second_set_size + 1 ∧ 1 ≤ initial_third_set_size + 1)
  (initial_combinations : ℕ)
  (final_combinations1 : ℕ)
  (final_combinations2 : ℕ)
  (additional_combinations : ℕ) :
  initial_first_set_size = 5 →
  initial_second_set_size = 3 →
  initial_third_set_size = 4 →
  new_letters = 4 →
  initial_combinations = initial_first_set_size * initial_second_set_size * initial_third_set_size →
  final_combinations1 = initial_first_set_size * (initial_second_set_size + 2) * (initial_third_set_size + 2) →
  final_combinations2 = (initial_first_set_size + 1) * (initial_second_set_size + 2) * (initial_third_set_size + 1) →
  additional_combinations = max (final_combinations1 - initial_combinations) (final_combinations2 - initial_combinations) →
  additional_combinations = 90 :=
by sorry

end NUMINAMATH_GPT_largest_additional_plates_l1191_119115


namespace NUMINAMATH_GPT_minimum_throws_for_repeated_sum_l1191_119159

theorem minimum_throws_for_repeated_sum :
  let min_sum := 4 * 1
  let max_sum := 4 * 6
  let num_distinct_sums := max_sum - min_sum + 1
  let min_throws := num_distinct_sums + 1
  min_throws = 22 :=
by
  sorry

end NUMINAMATH_GPT_minimum_throws_for_repeated_sum_l1191_119159


namespace NUMINAMATH_GPT_distance_to_school_l1191_119188

theorem distance_to_school : 
  ∀ (d v : ℝ), (d = v * (1 / 3)) → (d = (v + 20) * (1 / 4)) → d = 20 :=
by
  intros d v h1 h2
  sorry

end NUMINAMATH_GPT_distance_to_school_l1191_119188


namespace NUMINAMATH_GPT_ABC_books_sold_eq_4_l1191_119132

/-- "TOP" book cost in dollars --/
def TOP_price : ℕ := 8

/-- "ABC" book cost in dollars --/
def ABC_price : ℕ := 23

/-- Number of "TOP" books sold --/
def TOP_books_sold : ℕ := 13

/-- Difference in earnings in dollars --/
def earnings_difference : ℕ := 12

/-- Prove the number of "ABC" books sold --/
theorem ABC_books_sold_eq_4 (x : ℕ) (h : TOP_books_sold * TOP_price - x * ABC_price = earnings_difference) : x = 4 :=
by
  sorry

end NUMINAMATH_GPT_ABC_books_sold_eq_4_l1191_119132


namespace NUMINAMATH_GPT_count_squares_ending_in_4_l1191_119122

theorem count_squares_ending_in_4 (n : ℕ) : 
  (∀ k : ℕ, (n^2 < 5000) → (n^2 % 10 = 4) → (k ≤ 70)) → 
  (∃ m : ℕ, m = 14) :=
by 
  sorry

end NUMINAMATH_GPT_count_squares_ending_in_4_l1191_119122


namespace NUMINAMATH_GPT_jeans_cost_l1191_119168

theorem jeans_cost (initial_money pizza_cost soda_cost quarter_value after_quarters : ℝ) (quarters_count: ℕ) :
  initial_money = 40 ->
  pizza_cost = 2.75 ->
  soda_cost = 1.50 ->
  quarter_value = 0.25 ->
  quarters_count = 97 ->
  after_quarters = quarters_count * quarter_value ->
  initial_money - (pizza_cost + soda_cost) - after_quarters = 11.50 :=
by
  intros h_initial h_pizza h_soda h_quarter_val h_quarters h_after_quarters
  sorry

end NUMINAMATH_GPT_jeans_cost_l1191_119168


namespace NUMINAMATH_GPT_no_integer_sided_triangle_with_odd_perimeter_1995_l1191_119150

theorem no_integer_sided_triangle_with_odd_perimeter_1995 :
  ¬ ∃ (a b c : ℕ), (a + b + c = 1995) ∧ (∃ (h1 h2 h3 : ℕ), true) :=
by
  sorry

end NUMINAMATH_GPT_no_integer_sided_triangle_with_odd_perimeter_1995_l1191_119150


namespace NUMINAMATH_GPT_points_collinear_sum_l1191_119172

theorem points_collinear_sum (x y : ℝ) :
  ∃ k : ℝ, (x - 1 = 3 * k ∧ 1 = k * (y - 2) ∧ -1 = 2 * k) → 
  x + y = -1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_points_collinear_sum_l1191_119172


namespace NUMINAMATH_GPT_equal_number_of_boys_and_girls_l1191_119102

theorem equal_number_of_boys_and_girls
  (m d M D : ℕ)
  (h1 : (M / m) ≠ (D / d))
  (h2 : (M / m + D / d) / 2 = (M + D) / (m + d)) : m = d :=
sorry

end NUMINAMATH_GPT_equal_number_of_boys_and_girls_l1191_119102


namespace NUMINAMATH_GPT_sequence_difference_l1191_119103

-- Definition of sequences sums
def odd_sum (n : ℕ) : ℕ := (n * n)
def even_sum (n : ℕ) : ℕ := n * (n + 1)

-- Main property to prove
theorem sequence_difference :
  odd_sum 1013 - even_sum 1011 = 3047 :=
by
  -- Definitions and assertions here
  sorry

end NUMINAMATH_GPT_sequence_difference_l1191_119103


namespace NUMINAMATH_GPT_diane_faster_than_rhonda_l1191_119131

theorem diane_faster_than_rhonda :
  ∀ (rhonda_time sally_time diane_time total_time : ℕ), 
  rhonda_time = 24 →
  sally_time = rhonda_time + 2 →
  total_time = 71 →
  total_time = rhonda_time + sally_time + diane_time →
  (rhonda_time - diane_time) = 3 :=
by
  intros rhonda_time sally_time diane_time total_time
  intros h_rhonda h_sally h_total h_sum
  sorry

end NUMINAMATH_GPT_diane_faster_than_rhonda_l1191_119131


namespace NUMINAMATH_GPT_incorrect_proposition_C_l1191_119114

theorem incorrect_proposition_C (a b c d : ℝ) (h_pos: 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d) :
  a^4 + b^4 + c^4 + d^4 = 2 * (a^2 * b^2 + c^2 * d^2) → ¬ (a = b ∧ b = c ∧ c = d) := 
sorry

end NUMINAMATH_GPT_incorrect_proposition_C_l1191_119114


namespace NUMINAMATH_GPT_partition_no_infinite_arith_prog_l1191_119135

theorem partition_no_infinite_arith_prog :
  ∃ (A B : Set ℕ), 
  (∀ n ∈ A, n ∈ B → False) ∧ 
  (∀ (a b : ℕ) (d : ℕ), (a ∈ A ∧ b ∈ A ∧ a ≠ b ∧ (a - b) % d = 0) → False) ∧
  (∀ (a b : ℕ) (d : ℕ), (a ∈ B ∧ b ∈ B ∧ a ≠ b ∧ (a - b) % d = 0) → False) :=
sorry

end NUMINAMATH_GPT_partition_no_infinite_arith_prog_l1191_119135


namespace NUMINAMATH_GPT_max_value_of_function_for_x_lt_0_l1191_119144

noncomputable def f (x : ℝ) : ℝ :=
  x + 4 / x

theorem max_value_of_function_for_x_lt_0 :
  ∀ x : ℝ, x < 0 → f x ≤ -4 ∧ (∃ y : ℝ, f y = -4 ∧ y < 0) := sorry

end NUMINAMATH_GPT_max_value_of_function_for_x_lt_0_l1191_119144


namespace NUMINAMATH_GPT_evaluate_expression_l1191_119137

theorem evaluate_expression :
  1 + (3 / (4 + (5 / (6 + (7 / 8))))) = 85 / 52 := 
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l1191_119137


namespace NUMINAMATH_GPT_rectangle_volume_l1191_119113

theorem rectangle_volume {a b c : ℕ} (h1 : a * b - c * a - b * c = 1) (h2 : c * a = b * c + 1) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : a * b * c = 6 :=
sorry

end NUMINAMATH_GPT_rectangle_volume_l1191_119113


namespace NUMINAMATH_GPT_cistern_filling_time_l1191_119153

/-- Define the rates at which the cistern is filled and emptied -/
def fill_rate := (1 : ℚ) / 3
def empty_rate := (1 : ℚ) / 8

/-- Define the net rate of filling when both taps are open -/
def net_rate := fill_rate - empty_rate

/-- Define the volume of the cistern -/
def cistern_volume := (1 : ℚ)

/-- Compute the time to fill the cistern given the net rate -/
def fill_time := cistern_volume / net_rate

theorem cistern_filling_time :
  fill_time = 4.8 := by
sorry

end NUMINAMATH_GPT_cistern_filling_time_l1191_119153


namespace NUMINAMATH_GPT_matrix_multiplication_comm_l1191_119178

theorem matrix_multiplication_comm {C D : Matrix (Fin 2) (Fin 2) ℝ}
    (h₁ : C + D = C * D)
    (h₂ : C * D = !![5, 1; -2, 4]) :
    (D * C = !![5, 1; -2, 4]) :=
by
  sorry

end NUMINAMATH_GPT_matrix_multiplication_comm_l1191_119178


namespace NUMINAMATH_GPT_arithmetic_sqrt_of_4_l1191_119142

theorem arithmetic_sqrt_of_4 : ∃ x : ℚ, x^2 = 4 ∧ x > 0 → x = 2 :=
by {
  sorry
}

end NUMINAMATH_GPT_arithmetic_sqrt_of_4_l1191_119142


namespace NUMINAMATH_GPT_fill_tank_with_leak_l1191_119192

theorem fill_tank_with_leak (R L T: ℝ)
(h1: R = 1 / 7) (h2: L = 1 / 56) (h3: R - L = 1 / T) : T = 8 := by
  sorry

end NUMINAMATH_GPT_fill_tank_with_leak_l1191_119192


namespace NUMINAMATH_GPT_bridge_length_proof_l1191_119147

noncomputable def length_of_bridge (length_of_train : ℝ) (speed_of_train_km_per_hr : ℝ) (time_to_cross_bridge : ℝ) : ℝ :=
  let speed_of_train_m_per_s := speed_of_train_km_per_hr * (1000 / 3600)
  let total_distance := speed_of_train_m_per_s * time_to_cross_bridge
  total_distance - length_of_train

theorem bridge_length_proof : length_of_bridge 100 75 11.279097672186225 = 135 := by
  simp [length_of_bridge]
  sorry

end NUMINAMATH_GPT_bridge_length_proof_l1191_119147


namespace NUMINAMATH_GPT_minimum_value_when_a_is_1_range_of_a_given_fx_geq_0_l1191_119123

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := 
  Real.log (x + 1) + 2 / (x + 1) + a * x - 2

theorem minimum_value_when_a_is_1 : ∀ x : ℝ, ∃ m : ℝ, 
  (∀ y : ℝ, f y 1 ≥ f x 1) ∧ (f x 1 = m) :=
sorry

theorem range_of_a_given_fx_geq_0 : ∀ a : ℝ, (∀ x : ℝ, 0 ≤ x ∧ x ≤ 2 → 0 ≤ f x a) ↔ 1 ≤ a :=
sorry

end NUMINAMATH_GPT_minimum_value_when_a_is_1_range_of_a_given_fx_geq_0_l1191_119123


namespace NUMINAMATH_GPT_correct_value_division_l1191_119110

theorem correct_value_division (x : ℕ) (h : 9 - x = 3) : 96 / x = 16 :=
by
  sorry

end NUMINAMATH_GPT_correct_value_division_l1191_119110


namespace NUMINAMATH_GPT_circle_area_percentage_increase_l1191_119175

theorem circle_area_percentage_increase (r : ℝ) (h : r > 0) :
  let original_area := (Real.pi * r^2)
  let new_radius := (2.5 * r)
  let new_area := (Real.pi * new_radius^2)
  let increase := new_area - original_area
  let percentage_increase := (increase / original_area) * 100
  percentage_increase = 525 := by
  let original_area := Real.pi * r^2
  let new_radius := 2.5 * r
  let new_area := Real.pi * new_radius^2
  let increase := new_area - original_area
  let percentage_increase := (increase / original_area) * 100
  sorry

end NUMINAMATH_GPT_circle_area_percentage_increase_l1191_119175


namespace NUMINAMATH_GPT_sara_museum_visit_l1191_119191

theorem sara_museum_visit (S : Finset ℕ) (hS : S.card = 6) :
  ∃ count : ℕ, count = 720 ∧ 
  (∀ M A : Finset ℕ, M.card = 3 → A.card = 3 → M ∪ A = S → 
    count = (S.card.choose M.card) * M.card.factorial * A.card.factorial) :=
by
  sorry

end NUMINAMATH_GPT_sara_museum_visit_l1191_119191


namespace NUMINAMATH_GPT_original_number_of_boys_l1191_119126

theorem original_number_of_boys (n : ℕ) (W : ℕ) 
  (h1 : W = n * 35) 
  (h2 : W + 40 = (n + 1) * 36) 
  : n = 4 :=
sorry

end NUMINAMATH_GPT_original_number_of_boys_l1191_119126


namespace NUMINAMATH_GPT_oldest_child_age_l1191_119198

theorem oldest_child_age (x : ℕ) (h : (6 + 8 + x) / 3 = 9) : x = 13 := 
by {
  sorry
}

end NUMINAMATH_GPT_oldest_child_age_l1191_119198


namespace NUMINAMATH_GPT_cost_of_45_roses_l1191_119111

theorem cost_of_45_roses (cost_15_roses : ℕ → ℝ) 
  (h1 : cost_15_roses 15 = 25) 
  (h2 : ∀ (n m : ℕ), cost_15_roses n / n = cost_15_roses m / m )
  (h3 : ∀ (n : ℕ), n > 30 → cost_15_roses n = (1 - 0.10) * cost_15_roses n) :
  cost_15_roses 45 = 67.5 :=
by
  sorry

end NUMINAMATH_GPT_cost_of_45_roses_l1191_119111


namespace NUMINAMATH_GPT_max_distance_to_pole_l1191_119196

noncomputable def max_distance_to_origin (r1 r2 : ℝ) (c1 c2 : ℝ) : ℝ :=
  r1 + r2

theorem max_distance_to_pole (r : ℝ) (c : ℝ) : max_distance_to_origin 2 1 0 0 = 3 := by
  sorry

end NUMINAMATH_GPT_max_distance_to_pole_l1191_119196


namespace NUMINAMATH_GPT_solution_l1191_119162

def system (a b : ℝ) : Prop :=
  (2 * a + b = 3) ∧ (a - b = 1)

theorem solution (a b : ℝ) (h: system a b) : a + 2 * b = 2 :=
by
  cases h with
  | intro h1 h2 => sorry

end NUMINAMATH_GPT_solution_l1191_119162


namespace NUMINAMATH_GPT_hacker_cannot_change_grades_l1191_119187

theorem hacker_cannot_change_grades :
  ¬ ∃ n1 n2 n3 n4 : ℤ,
    2 * n1 + n2 - 2 * n3 - n4 = 27 ∧
    -n1 + 2 * n2 + n3 - 2 * n4 = -27 := by
  sorry

end NUMINAMATH_GPT_hacker_cannot_change_grades_l1191_119187


namespace NUMINAMATH_GPT_triangle_area_l1191_119119

theorem triangle_area (B : Real) (AB AC : Real) 
  (hB : B = Real.pi / 6) 
  (hAB : AB = 2 * Real.sqrt 3)
  (hAC : AC = 2) : 
  let area := 1 / 2 * AB * AC * Real.sin B
  area = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_GPT_triangle_area_l1191_119119


namespace NUMINAMATH_GPT_find_expression_for_a_n_l1191_119170

noncomputable def seq (n : ℕ) : ℕ := sorry
def sumFirstN (n : ℕ) : ℕ := sorry

theorem find_expression_for_a_n (a : ℕ → ℕ)
  (S : ℕ → ℕ)
  (h_pos : ∀ n, 0 < a n)
  (h_arith_seq : ∀ n, S n + 1 = 2 * a n) :
  ∀ n, a n = 2^(n-1) :=
sorry

end NUMINAMATH_GPT_find_expression_for_a_n_l1191_119170


namespace NUMINAMATH_GPT_calculate_expression_l1191_119199

theorem calculate_expression :
  (2 ^ (1/3) * 8 ^ (1/3) + 18 / (3 * 3) - 8 ^ (5/3)) = 2 ^ (4/3) - 30 :=
by
  sorry

end NUMINAMATH_GPT_calculate_expression_l1191_119199


namespace NUMINAMATH_GPT_jordan_rectangle_width_l1191_119133

theorem jordan_rectangle_width (length_carol width_carol length_jordan width_jordan : ℝ)
  (h1: length_carol = 15) (h2: width_carol = 20) (h3: length_jordan = 6)
  (area_equal: length_carol * width_carol = length_jordan * width_jordan) :
  width_jordan = 50 :=
by
  sorry

end NUMINAMATH_GPT_jordan_rectangle_width_l1191_119133


namespace NUMINAMATH_GPT_solution_set_f_x_le_5_l1191_119195

noncomputable def f (x : ℝ) : ℝ :=
if x > 0 then 3 + Real.log x / Real.log 2 else x^2 - x - 1

theorem solution_set_f_x_le_5 : {x : ℝ | f x ≤ 5} = Set.Icc (-2 : ℝ) 4 := by
  sorry

end NUMINAMATH_GPT_solution_set_f_x_le_5_l1191_119195


namespace NUMINAMATH_GPT_sqrt_0_09_eq_0_3_l1191_119174

theorem sqrt_0_09_eq_0_3 : Real.sqrt 0.09 = 0.3 := 
by 
  sorry

end NUMINAMATH_GPT_sqrt_0_09_eq_0_3_l1191_119174


namespace NUMINAMATH_GPT_cost_of_pencils_l1191_119186

open Nat

theorem cost_of_pencils (P : ℕ) : 
  (H : 20 * P + 80 * 3 = 360) → 
  P = 6 :=
by 
  sorry

end NUMINAMATH_GPT_cost_of_pencils_l1191_119186


namespace NUMINAMATH_GPT_arithmetic_sequence_proof_l1191_119149

open Nat

def arithmetic_seq (a : ℕ → ℤ) (d : ℤ) : Prop :=
  a 1 = 2 ∧ (a 2) ^ 2 = (a 1) * (a 5)

def general_formula (a : ℕ → ℤ) (d : ℤ) : Prop :=
  (d = 0 ∧ ∀ n, a n = 2) ∨ (d = 4 ∧ ∀ n, a n = 4 * n - 2)

def sum_seq (a : ℕ → ℤ) (S_n : ℕ → ℤ) (d : ℤ) : Prop :=
  ((∀ n, a n = 2) ∧ (∀ n, S_n n = 2 * n)) ∨ ((∀ n, a n = 4 * n - 2) ∧ (∀ n, S_n n = 4 * n^2 - 2 * n))

theorem arithmetic_sequence_proof :
  ∃ a : ℕ → ℤ, ∃ d : ℤ, arithmetic_seq a d ∧ general_formula a d ∧ ∃ S_n : ℕ → ℤ, sum_seq a S_n d := by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_proof_l1191_119149


namespace NUMINAMATH_GPT_angle_relation_l1191_119106

-- Definitions for the triangle properties and angles.
variables {α : Type*} [LinearOrderedField α]
variables {A B C D E F : α}

-- Definitions stating the properties of the triangles.
def is_isosceles_triangle (a b c : α) : Prop :=
  a = b ∨ b = c ∨ c = a

def triangle_ABC_is_isosceles (AB AC : α) (ABC : α) : Prop :=
  is_isosceles_triangle AB AC ABC

def triangle_DEF_is_isosceles (DE DF : α) (DEF : α) : Prop :=
  is_isosceles_triangle DE DF DEF

-- Condition that gives the specific angle measure in triangle DEF.
def angle_DEF_is_100 (DEF : α) : Prop :=
  DEF = 100

-- The main theorem to prove.
theorem angle_relation (AB AC DE DF DEF a b c : α) :
  triangle_ABC_is_isosceles AB AC (AB + AC) →
  triangle_DEF_is_isosceles DE DF DEF →
  angle_DEF_is_100 DEF →
  a = c :=
by
  -- Assuming the conditions define the angles and state the relationship.
  sorry

end NUMINAMATH_GPT_angle_relation_l1191_119106


namespace NUMINAMATH_GPT_area_smaller_part_l1191_119151

theorem area_smaller_part (A B : ℝ) (h₁ : A + B = 500) (h₂ : B - A = (A + B) / 10) : A = 225 :=
by sorry

end NUMINAMATH_GPT_area_smaller_part_l1191_119151


namespace NUMINAMATH_GPT_find_x_in_average_l1191_119104

theorem find_x_in_average (x : ℝ) :
  (201 + 202 + 204 + 205 + 206 + 209 + 209 + 210 + x) / 9 = 207 → x = 217 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_find_x_in_average_l1191_119104


namespace NUMINAMATH_GPT_triangle_angle_contradiction_l1191_119177

theorem triangle_angle_contradiction (A B C : ℝ) (h1 : 0 < A) (h2 : 0 < B) (h3 : 0 < C) (h4 : A + B + C = 180) :
  A > 60 → B > 60 → C > 60 → false :=
by
  sorry

end NUMINAMATH_GPT_triangle_angle_contradiction_l1191_119177


namespace NUMINAMATH_GPT_violet_prob_l1191_119158

noncomputable def total_candies := 8 + 5 + 9 + 10 + 6

noncomputable def prob_green_first := (8 : ℚ) / total_candies
noncomputable def prob_yellow_second := (10 : ℚ) / (total_candies - 1)
noncomputable def prob_pink_third := (6 : ℚ) / (total_candies - 2)

noncomputable def combined_prob := prob_green_first * prob_yellow_second * prob_pink_third

theorem violet_prob :
  combined_prob = (20 : ℚ) / 2109 := by
    sorry

end NUMINAMATH_GPT_violet_prob_l1191_119158


namespace NUMINAMATH_GPT_annual_growth_rate_equation_l1191_119190

theorem annual_growth_rate_equation
  (initial_capital : ℝ)
  (final_capital : ℝ)
  (n : ℕ)
  (x : ℝ)
  (h1 : initial_capital = 10)
  (h2 : final_capital = 14.4)
  (h3 : n = 2) :
  1000 * (1 + x)^2 = 1440 :=
by
  sorry

end NUMINAMATH_GPT_annual_growth_rate_equation_l1191_119190


namespace NUMINAMATH_GPT_set_D_cannot_form_triangle_l1191_119161

-- Definition for triangle inequality theorem
def triangle_inequality (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

-- Given lengths
def length_1 := 1
def length_2 := 2
def length_3 := 3

-- The proof problem statement
theorem set_D_cannot_form_triangle : ¬ triangle_inequality length_1 length_2 length_3 :=
  by sorry

end NUMINAMATH_GPT_set_D_cannot_form_triangle_l1191_119161


namespace NUMINAMATH_GPT_jar_water_fraction_l1191_119143

theorem jar_water_fraction
  (S L : ℝ)
  (h1 : S = (1 / 5) * S)
  (h2 : S = x * L)
  (h3 : (1 / 5) * S + x * L = (2 / 5) * L) :
  x = (1 / 10) :=
by
  sorry

end NUMINAMATH_GPT_jar_water_fraction_l1191_119143


namespace NUMINAMATH_GPT_probability_diamond_then_ace_l1191_119100

theorem probability_diamond_then_ace :
  let total_cards := 104
  let diamonds := 26
  let aces := 8
  let remaining_cards_after_first_draw := total_cards - 1
  let ace_of_diamonds_prob := (2 : ℚ) / total_cards
  let any_ace_after_ace_of_diamonds := (7 : ℚ) / remaining_cards_after_first_draw
  let combined_prob_ace_of_diamonds_then_any_ace := ace_of_diamonds_prob * any_ace_after_ace_of_diamonds
  let diamond_not_ace_prob := (24 : ℚ) / total_cards
  let any_ace_after_diamond_not_ace := (8 : ℚ) / remaining_cards_after_first_draw
  let combined_prob_diamond_not_ace_then_any_ace := diamond_not_ace_prob * any_ace_after_diamond_not_ace
  let total_prob := combined_prob_ace_of_diamonds_then_any_ace + combined_prob_diamond_not_ace_then_any_ace
  total_prob = (31 : ℚ) / 5308 :=
by
  sorry

end NUMINAMATH_GPT_probability_diamond_then_ace_l1191_119100


namespace NUMINAMATH_GPT_min_max_SX_SY_l1191_119166

theorem min_max_SX_SY (n : ℕ) (hn : 2 ≤ n) (a : Finset ℕ) 
  (ha_sum : Finset.sum a id = 2 * n - 1) :
  ∃ (min_val max_val : ℕ), 
    (min_val = 2 * n - 2) ∧ 
    (max_val = n * (n - 1)) :=
sorry

end NUMINAMATH_GPT_min_max_SX_SY_l1191_119166


namespace NUMINAMATH_GPT_max_marks_mike_l1191_119189

theorem max_marks_mike (pass_percentage : ℝ) (scored_marks : ℝ) (shortfall : ℝ) : 
  pass_percentage = 0.30 → 
  scored_marks = 212 → 
  shortfall = 28 → 
  (scored_marks + shortfall) = 240 → 
  (scored_marks + shortfall) = pass_percentage * (max_marks : ℝ) → 
  max_marks = 800 := 
by 
  intros hp hs hsh hps heq 
  sorry

end NUMINAMATH_GPT_max_marks_mike_l1191_119189


namespace NUMINAMATH_GPT_find_k_l1191_119117

theorem find_k (t k : ℤ) (h1 : t = 35) (h2 : t = 5 * (k - 32) / 9) : k = 95 :=
sorry

end NUMINAMATH_GPT_find_k_l1191_119117


namespace NUMINAMATH_GPT_minimum_parents_needed_l1191_119182

/-- 
Given conditions:
1. There are 30 students going on the excursion.
2. Each car can accommodate 5 people, including the driver.
Prove that the minimum number of parents needed to be invited on the excursion is 8.
-/
theorem minimum_parents_needed (students : ℕ) (car_capacity : ℕ) (drivers_needed : ℕ) 
  (h1 : students = 30) (h2 : car_capacity = 5) (h3 : drivers_needed = 1) 
  : ∃ (parents : ℕ), parents = 8 :=
by
  existsi 8
  sorry

end NUMINAMATH_GPT_minimum_parents_needed_l1191_119182


namespace NUMINAMATH_GPT_digit_A_of_3AA1_divisible_by_9_l1191_119116

theorem digit_A_of_3AA1_divisible_by_9 (A : ℕ) (h : (3 + A + A + 1) % 9 = 0) : A = 7 :=
sorry

end NUMINAMATH_GPT_digit_A_of_3AA1_divisible_by_9_l1191_119116


namespace NUMINAMATH_GPT_apple_cost_l1191_119128

theorem apple_cost (x l q : ℝ) 
  (h1 : 10 * l = 3.62) 
  (h2 : x * l + (33 - x) * q = 11.67)
  (h3 : x * l + (36 - x) * q = 12.48) : 
  x = 30 :=
by
  sorry

end NUMINAMATH_GPT_apple_cost_l1191_119128


namespace NUMINAMATH_GPT_pat_interest_rate_l1191_119194

noncomputable def interest_rate (t : ℝ) : ℝ := 70 / t

theorem pat_interest_rate (r : ℝ) (t : ℝ) (initial_amount : ℝ) (final_amount : ℝ) (years : ℝ) : 
  initial_amount * 2^((years / t)) = final_amount ∧ 
  years = 18 ∧ 
  final_amount = 28000 ∧ 
  initial_amount = 7000 →    
  r = interest_rate 9 := 
by
  sorry

end NUMINAMATH_GPT_pat_interest_rate_l1191_119194


namespace NUMINAMATH_GPT_leq_sum_l1191_119108

open BigOperators

theorem leq_sum (x : Fin 3 → ℝ) (hx_pos : ∀ i, 0 < x i) (hx_sum : ∑ i, x i = 1) :
  (∑ i, 1 / (1 + (x i)^2)) ≤ 27 / 10 :=
sorry

end NUMINAMATH_GPT_leq_sum_l1191_119108


namespace NUMINAMATH_GPT_algebra_expression_value_l1191_119156

theorem algebra_expression_value (x y : ℝ) (h1 : x * y = 3) (h2 : x - y = -2) : x^2 * y - x * y^2 = -6 := 
by
  sorry

end NUMINAMATH_GPT_algebra_expression_value_l1191_119156


namespace NUMINAMATH_GPT_evaluate_expression_l1191_119157

def S (n : ℕ) : ℤ :=
  if n % 2 = 1 then (n + 1) / 2
  else -n / 2

theorem evaluate_expression : S 19 * S 31 + S 48 = 136 :=
by sorry

end NUMINAMATH_GPT_evaluate_expression_l1191_119157


namespace NUMINAMATH_GPT_arithmetic_series_sum_l1191_119197

def a := 5
def l := 20
def n := 16
def S := (n / 2) * (a + l)

theorem arithmetic_series_sum :
  S = 200 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_series_sum_l1191_119197


namespace NUMINAMATH_GPT_wendy_candy_in_each_box_l1191_119145

variable (x : ℕ)

def brother_candy : ℕ := 6
def total_candy : ℕ := 12
def wendy_boxes : ℕ := 2 * x

theorem wendy_candy_in_each_box :
  2 * x + brother_candy = total_candy → x = 3 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_wendy_candy_in_each_box_l1191_119145


namespace NUMINAMATH_GPT_distance_from_y_axis_l1191_119173

theorem distance_from_y_axis (x : ℝ) : abs x = 10 :=
by
  -- Define distances
  let d_x := 5
  let d_y := abs x
  -- Given condition
  have h : d_x = (1 / 2) * d_y := sorry
  -- Use the given condition to prove the required statement
  sorry

end NUMINAMATH_GPT_distance_from_y_axis_l1191_119173


namespace NUMINAMATH_GPT_hyperbola_slope_of_asymptote_positive_value_l1191_119129

noncomputable def hyperbola_slope_of_asymptote (x y : ℝ) : ℝ :=
  if h : (Real.sqrt ((x - 2)^2 + (y - 3)^2) - Real.sqrt ((x - 8)^2 + (y - 3)^2) = 4)
  then (Real.sqrt 5) / 2
  else 0

-- Statement of the mathematically equivalent proof problem
theorem hyperbola_slope_of_asymptote_positive_value :
  ∃ x y : ℝ, (Real.sqrt ((x - 2)^2 + (y - 3)^2) - Real.sqrt ((x - 8)^2 + (y - 3)^2) = 4) ∧
  hyperbola_slope_of_asymptote x y = (Real.sqrt 5) / 2 :=
by
  sorry

end NUMINAMATH_GPT_hyperbola_slope_of_asymptote_positive_value_l1191_119129


namespace NUMINAMATH_GPT_stadium_fee_difference_l1191_119130

theorem stadium_fee_difference :
  let capacity := 2000
  let entry_fee := 20
  let full_fees := capacity * entry_fee
  let three_quarters_fees := (capacity * 3 / 4) * entry_fee
  full_fees - three_quarters_fees = 10000 :=
by
  sorry

end NUMINAMATH_GPT_stadium_fee_difference_l1191_119130


namespace NUMINAMATH_GPT_extreme_values_x_axis_l1191_119120

noncomputable def f (a b c : ℝ) (x : ℝ) : ℝ := x * (a * x^2 + b * x + c)

theorem extreme_values_x_axis (a b c : ℝ) (h1 : a ≠ 0)
  (h2 : ∀ x, f a b c x = x * (a * x^2 + b * x + c))
  (h3 : ∀ x, deriv (f a b c) x = 3 * a * x^2 + 2 * b * x + c)
  (h4 : deriv (f a b c) 1 = 0)
  (h5 : deriv (f a b c) (-1) = 0) :
  b = 0 :=
sorry

end NUMINAMATH_GPT_extreme_values_x_axis_l1191_119120


namespace NUMINAMATH_GPT_find_smallest_n_l1191_119134

open Matrix Complex

noncomputable def rotation_matrix := ![
  ![Real.sqrt 2 / 2, -Real.sqrt 2 / 2],
  ![Real.sqrt 2 / 2, Real.sqrt 2 / 2]
]

def I_2 := (1 : Matrix (Fin 2) (Fin 2) ℝ)

theorem find_smallest_n (A : Matrix (Fin 2) (Fin 2) ℝ) 
  (hA : A = rotation_matrix) : 
  ∃ (n : ℕ), 0 < n ∧ A ^ n = I_2 ∧ ∀ m : ℕ, 0 < m ∧ m < n → A ^ m ≠ I_2 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_smallest_n_l1191_119134


namespace NUMINAMATH_GPT_find_k_value_l1191_119184

theorem find_k_value (k : ℝ) (h : (7 * (-1)^3 - 3 * (-1)^2 + k * -1 + 5 = 0)) :
  k^3 + 2 * k^2 - 11 * k - 85 = -105 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_k_value_l1191_119184


namespace NUMINAMATH_GPT_problem_even_and_monotonically_increasing_l1191_119163

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = f (-x)

def is_monotonically_increasing_on (f : ℝ → ℝ) (I : Set ℝ) : Prop :=
  ∀ x y : ℝ, x ∈ I → y ∈ I → x < y → f x ≤ f y

theorem problem_even_and_monotonically_increasing :
  is_even_function (fun x => Real.exp (|x|)) ∧ is_monotonically_increasing_on (fun x => Real.exp (|x|)) (Set.Ioo 0 1) :=
by
  sorry

end NUMINAMATH_GPT_problem_even_and_monotonically_increasing_l1191_119163


namespace NUMINAMATH_GPT_system_of_equations_solution_l1191_119165

theorem system_of_equations_solution (x y : ℝ) (h1 : 4 * x + 3 * y = 11) (h2 : 4 * x - 3 * y = 5) :
  x = 2 ∧ y = 1 :=
by {
  sorry
}

end NUMINAMATH_GPT_system_of_equations_solution_l1191_119165
