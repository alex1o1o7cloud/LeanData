import Mathlib

namespace NUMINAMATH_GPT_necessary_but_not_sufficient_condition_l1111_111129

variable (a b : ℝ)

theorem necessary_but_not_sufficient_condition : (a > b) → ((a > b) ↔ ((a - b) * b^2 > 0)) :=
sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_condition_l1111_111129


namespace NUMINAMATH_GPT_temperature_on_fourth_day_l1111_111108

theorem temperature_on_fourth_day
  (t₁ t₂ t₃ : ℤ) 
  (avg : ℤ)
  (h₁ : t₁ = -36) 
  (h₂ : t₂ = 13) 
  (h₃ : t₃ = -10) 
  (h₄ : avg = -12) 
  : ∃ t₄ : ℤ, t₄ = -15 :=
by
  sorry

end NUMINAMATH_GPT_temperature_on_fourth_day_l1111_111108


namespace NUMINAMATH_GPT_range_of_a_l1111_111160

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^2 - 4*x + a

theorem range_of_a 
  (f : ℝ → ℝ → ℝ)
  (h : ∀ (x : ℝ), 0 ≤ x ∧ x ≤ 1 → f x a ≥ 0) : 
  3 ≤ a :=
sorry

end NUMINAMATH_GPT_range_of_a_l1111_111160


namespace NUMINAMATH_GPT_equation_pattern_l1111_111144

theorem equation_pattern (n : ℕ) (h : n = 999999) : n^2 = (n + 1) * (n - 1) + 1 :=
by
  sorry

end NUMINAMATH_GPT_equation_pattern_l1111_111144


namespace NUMINAMATH_GPT_range_of_a_no_fixed_points_l1111_111139

def f (x : ℝ) (a : ℝ) : ℝ := x^2 + a * x + 1

theorem range_of_a_no_fixed_points : 
  ∀ a : ℝ, ¬∃ x : ℝ, f x a = x ↔ -1 < a ∧ a < 3 :=
by sorry

end NUMINAMATH_GPT_range_of_a_no_fixed_points_l1111_111139


namespace NUMINAMATH_GPT_volume_in_cubic_meters_l1111_111185

noncomputable def mass_condition : ℝ := 100 -- mass in kg
noncomputable def volume_per_gram : ℝ := 10 -- volume in cubic centimeters per gram
noncomputable def volume_per_kg : ℝ := volume_per_gram * 1000 -- volume in cubic centimeters per kg
noncomputable def mass_in_kg : ℝ := mass_condition

theorem volume_in_cubic_meters (h : mass_in_kg = 100)
    (v_per_kg : volume_per_kg = volume_per_gram * 1000) :
  (mass_in_kg * volume_per_kg) / 1000000 = 1 := by
  sorry

end NUMINAMATH_GPT_volume_in_cubic_meters_l1111_111185


namespace NUMINAMATH_GPT_distinct_roots_of_quadratic_l1111_111169

variable {a b : ℝ}
-- condition: a and b are distinct
variable (h_distinct: a ≠ b)

theorem distinct_roots_of_quadratic (a b : ℝ) (h_distinct : a ≠ b) :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (x + a)*(x + b) = 2*x + a + b :=
by
  sorry

end NUMINAMATH_GPT_distinct_roots_of_quadratic_l1111_111169


namespace NUMINAMATH_GPT_most_frequent_digit_100000_l1111_111120

/- Define the digital root function -/
def digital_root (n : ℕ) : ℕ :=
  if n == 0 then 0 else if n % 9 == 0 then 9 else n % 9

/- Define the problem statement -/
theorem most_frequent_digit_100000 : 
  ∃ digit : ℕ, 
  digit = 1 ∧ (∀ n : ℕ, 1 ≤ n ∧ n ≤ 100000 → ∃ k : ℕ, k = digital_root n ∧ k = digit) →
  digit = 1 :=
sorry

end NUMINAMATH_GPT_most_frequent_digit_100000_l1111_111120


namespace NUMINAMATH_GPT_find_triplets_satisfying_equation_l1111_111115

theorem find_triplets_satisfying_equation :
  ∃ (x y z : ℕ), x ≤ y ∧ y ≤ z ∧ x^3 * (y^3 + z^3) = 2012 * (x * y * z + 2) ∧ (x, y, z) = (2, 251, 252) :=
by
  sorry

end NUMINAMATH_GPT_find_triplets_satisfying_equation_l1111_111115


namespace NUMINAMATH_GPT_line_passes_fixed_point_max_distance_eqn_l1111_111119

-- Definition of the line equation
def line_eq (a b x y : ℝ) : Prop :=
  (2 * a + b) * x + (a + b) * y + a - b = 0

-- Point P
def point_P : ℝ × ℝ :=
  (3, 4)

-- Fixed point that the line passes through
def fixed_point : ℝ × ℝ :=
  (-2, 3)

-- Statement that the line passes through the fixed point
theorem line_passes_fixed_point (a b : ℝ) :
  line_eq a b (-2) 3 :=
sorry

-- Equation of the line when distance from point P to line is maximized
def line_max_distance (a b : ℝ) : Prop :=
  5 * 3 + 4 + 7 = 0

-- Statement that the equation of the line is as given when distance is maximized
theorem max_distance_eqn (a b : ℝ) :
  line_max_distance a b :=
sorry

end NUMINAMATH_GPT_line_passes_fixed_point_max_distance_eqn_l1111_111119


namespace NUMINAMATH_GPT_granger_total_amount_l1111_111141

-- Define the constants for the problem
def cost_spam := 3
def cost_peanut_butter := 5
def cost_bread := 2
def quantity_spam := 12
def quantity_peanut_butter := 3
def quantity_bread := 4

-- Define the total cost calculation
def total_cost := (quantity_spam * cost_spam) + (quantity_peanut_butter * cost_peanut_butter) + (quantity_bread * cost_bread)

-- The theorem we need to prove
theorem granger_total_amount : total_cost = 59 := by
  sorry

end NUMINAMATH_GPT_granger_total_amount_l1111_111141


namespace NUMINAMATH_GPT_no_solutions_to_cubic_sum_l1111_111162

theorem no_solutions_to_cubic_sum (x y z : ℤ) : 
    ¬ (x^3 + y^3 = z^3 + 4) :=
by 
  sorry

end NUMINAMATH_GPT_no_solutions_to_cubic_sum_l1111_111162


namespace NUMINAMATH_GPT_stormi_needs_more_money_to_afford_bicycle_l1111_111150

-- Definitions from conditions
def money_washed_cars : ℕ := 3 * 10
def money_mowed_lawns : ℕ := 2 * 13
def bicycle_cost : ℕ := 80
def total_earnings : ℕ := money_washed_cars + money_mowed_lawns

-- The goal to prove 
theorem stormi_needs_more_money_to_afford_bicycle :
  (bicycle_cost - total_earnings) = 24 := by
  sorry

end NUMINAMATH_GPT_stormi_needs_more_money_to_afford_bicycle_l1111_111150


namespace NUMINAMATH_GPT_cost_of_each_fish_is_four_l1111_111155

-- Definitions according to the conditions
def number_of_fish_given_to_dog := 40
def number_of_fish_given_to_cat := number_of_fish_given_to_dog / 2
def total_fish := number_of_fish_given_to_dog + number_of_fish_given_to_cat
def total_cost := 240
def cost_per_fish := total_cost / total_fish

-- The main statement / theorem that needs to be proved
theorem cost_of_each_fish_is_four :
  cost_per_fish = 4 :=
by
  sorry

end NUMINAMATH_GPT_cost_of_each_fish_is_four_l1111_111155


namespace NUMINAMATH_GPT_ages_of_siblings_l1111_111164

-- Define the variables representing the ages of the siblings
variables (R D S E : ℕ)

-- Define the conditions
def conditions := 
  R = D + 6 ∧ 
  D = S + 8 ∧ 
  E = R - 5 ∧ 
  R + 8 = 2 * (S + 8)

-- Define the statement to be proved
theorem ages_of_siblings (h : conditions R D S E) : 
  R = 20 ∧ D = 14 ∧ S = 6 ∧ E = 15 :=
sorry

end NUMINAMATH_GPT_ages_of_siblings_l1111_111164


namespace NUMINAMATH_GPT_basketball_starting_lineups_l1111_111109

theorem basketball_starting_lineups (n_players n_guards n_forwards n_centers : ℕ)
  (h_players : n_players = 12)
  (h_guards : n_guards = 2)
  (h_forwards : n_forwards = 2)
  (h_centers : n_centers = 1) :
  (Nat.choose n_players n_guards) * (Nat.choose (n_players - n_guards) n_forwards) * (Nat.choose (n_players - n_guards - n_forwards) n_centers) = 23760 := by
  sorry

end NUMINAMATH_GPT_basketball_starting_lineups_l1111_111109


namespace NUMINAMATH_GPT_number_of_sides_of_polygon_l1111_111152

theorem number_of_sides_of_polygon (exterior_angle : ℝ) (h : exterior_angle = 40) : 
  (360 / exterior_angle) = 9 :=
by
  sorry

end NUMINAMATH_GPT_number_of_sides_of_polygon_l1111_111152


namespace NUMINAMATH_GPT_multiplication_of_positive_and_negative_l1111_111195

theorem multiplication_of_positive_and_negative :
  9 * (-3) = -27 := by
  sorry

end NUMINAMATH_GPT_multiplication_of_positive_and_negative_l1111_111195


namespace NUMINAMATH_GPT_arithmetic_sequence_5_7_9_l1111_111199

variable {a : ℕ → ℕ}

theorem arithmetic_sequence_5_7_9 (h : 13 * (a 7) = 39) : a 5 + a 7 + a 9 = 9 := 
sorry

end NUMINAMATH_GPT_arithmetic_sequence_5_7_9_l1111_111199


namespace NUMINAMATH_GPT_profit_without_discount_l1111_111176

theorem profit_without_discount (CP SP_discount SP_without_discount : ℝ) (profit_discount profit_without_discount percent_discount : ℝ)
  (h1 : CP = 100) 
  (h2 : percent_discount = 0.05) 
  (h3 : profit_discount = 0.425) 
  (h4 : SP_discount = CP + profit_discount * CP) 
  (h5 : SP_discount = 142.5)
  (h6 : SP_without_discount = SP_discount / (1 - percent_discount)) : 
  profit_without_discount = ((SP_without_discount - CP) / CP) * 100 := 
by
  sorry

end NUMINAMATH_GPT_profit_without_discount_l1111_111176


namespace NUMINAMATH_GPT_smallest_prime_dividing_large_sum_is_5_l1111_111179

-- Definitions based on the conditions
def large_sum : ℕ := 4^15 + 7^12

-- Prime number checking function
def is_prime (n : ℕ) : Prop := 
  n > 1 ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

-- Check for the smallest prime number dividing the sum
def smallest_prime_dividing_sum (n : ℕ) : ℕ := 
  if n % 2 = 0 then 2 
  else if n % 3 = 0 then 3 
  else if n % 5 = 0 then 5 
  else 2 -- Since 2 is a placeholder, theoretical logic checks can replace this branch

-- Final theorem to prove
theorem smallest_prime_dividing_large_sum_is_5 : smallest_prime_dividing_sum large_sum = 5 := 
  sorry

end NUMINAMATH_GPT_smallest_prime_dividing_large_sum_is_5_l1111_111179


namespace NUMINAMATH_GPT_value_of_f2_l1111_111178

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ := a * x + b * x + 3

theorem value_of_f2 (a b : ℝ) (h1 : f 1 a b = 7) (h2 : f 3 a b = 15) : f 2 a b = 11 :=
by
  sorry

end NUMINAMATH_GPT_value_of_f2_l1111_111178


namespace NUMINAMATH_GPT_integer_coefficient_equation_calculate_expression_l1111_111183

noncomputable def a : ℝ := (Real.sqrt 5 - 1) / 2

theorem integer_coefficient_equation :
  a ^ 2 + a - 1 = 0 :=
sorry

theorem calculate_expression :
  a ^ 3 - 2 * a + 2015 = 2014 :=
sorry

end NUMINAMATH_GPT_integer_coefficient_equation_calculate_expression_l1111_111183


namespace NUMINAMATH_GPT_smallest_possible_a_plus_b_l1111_111191

theorem smallest_possible_a_plus_b :
  ∃ (a b : ℕ), 0 < a ∧ 0 < b ∧ gcd (a + b) 330 = 1 ∧ (b ^ b ∣ a ^ a) ∧ ¬ (b ∣ a) ∧ (a + b = 507) := 
sorry

end NUMINAMATH_GPT_smallest_possible_a_plus_b_l1111_111191


namespace NUMINAMATH_GPT_inequality_proof_l1111_111159

theorem inequality_proof 
  (a b c d : ℝ)
  (a_pos : 0 < a)
  (b_pos : 0 < b)
  (c_pos : 0 < c)
  (d_pos : 0 < d)
  (sum_eq : a + b + c + d = 3) : 
  1 / a^2 + 1 / b^2 + 1 / c^2 + 1 / d^2 ≤ 1 / (a * b * c * d)^2 := 
by 
  sorry

end NUMINAMATH_GPT_inequality_proof_l1111_111159


namespace NUMINAMATH_GPT_six_inch_cube_value_eq_844_l1111_111136

-- Definition of the value of a cube in lean
noncomputable def cube_value (s₁ s₂ : ℕ) (value₁ : ℕ) : ℕ :=
  let volume₁ := s₁ ^ 3
  let volume₂ := s₂ ^ 3
  (value₁ * volume₂) / volume₁

-- Theorem stating the equivalence between the volumes and values.
theorem six_inch_cube_value_eq_844 :
  cube_value 4 6 250 = 844 :=
by
  sorry

end NUMINAMATH_GPT_six_inch_cube_value_eq_844_l1111_111136


namespace NUMINAMATH_GPT_max_marks_set_for_test_l1111_111192

-- Define the conditions according to the problem statement
def passing_percentage : ℝ := 0.70
def student_marks : ℝ := 120
def marks_needed_to_pass : ℝ := 150
def passing_threshold (M : ℝ) : ℝ := passing_percentage * M

-- The maximum marks set for the test
theorem max_marks_set_for_test (M : ℝ) : M = 386 :=
by
  -- Given the conditions
  have h : passing_threshold M = student_marks + marks_needed_to_pass := sorry
  -- Solving for M
  sorry

end NUMINAMATH_GPT_max_marks_set_for_test_l1111_111192


namespace NUMINAMATH_GPT_concentration_after_dilution_l1111_111122

-- Definitions and conditions
def initial_volume : ℝ := 5
def initial_concentration : ℝ := 0.06
def poured_out_volume : ℝ := 1
def added_water_volume : ℝ := 2

-- Theorem statement
theorem concentration_after_dilution : 
  (initial_volume * initial_concentration - poured_out_volume * initial_concentration) / 
  (initial_volume - poured_out_volume + added_water_volume) = 0.04 :=
by 
  sorry

end NUMINAMATH_GPT_concentration_after_dilution_l1111_111122


namespace NUMINAMATH_GPT_HephaestusCharges_l1111_111148

variable (x : ℕ)

theorem HephaestusCharges :
  3 * x + 6 * (12 - x) = 54 -> x = 6 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_HephaestusCharges_l1111_111148


namespace NUMINAMATH_GPT_rhombus_side_length_l1111_111100

theorem rhombus_side_length (total_length : ℕ) (num_sides : ℕ) (h1 : total_length = 32) (h2 : num_sides = 4) :
    total_length / num_sides = 8 :=
by
  -- Proof will be provided here
  sorry

end NUMINAMATH_GPT_rhombus_side_length_l1111_111100


namespace NUMINAMATH_GPT_remainder_of_polynomial_l1111_111149

theorem remainder_of_polynomial (x : ℕ) :
  (x + 1) ^ 2021 % (x ^ 2 + x + 1) = 1 + x ^ 2 := 
by
  sorry

end NUMINAMATH_GPT_remainder_of_polynomial_l1111_111149


namespace NUMINAMATH_GPT_fraction_eq_l1111_111186

theorem fraction_eq : (15.5 / (-0.75) : ℝ) = (-62 / 3) := 
by {
  sorry
}

end NUMINAMATH_GPT_fraction_eq_l1111_111186


namespace NUMINAMATH_GPT_a9_value_l1111_111189

-- Define the sequence
def seq (a : ℕ → ℚ) : Prop :=
  a 1 = 3 ∧ ∀ n, a (n+1) = 1 - (1 / a n)

-- State the theorem
theorem a9_value : ∃ a : ℕ → ℚ, seq a ∧ a 9 = -1/2 :=
by
  sorry

end NUMINAMATH_GPT_a9_value_l1111_111189


namespace NUMINAMATH_GPT_ensure_two_different_colors_ensure_two_yellow_balls_l1111_111140

-- First statement: Ensuring two balls of different colors
theorem ensure_two_different_colors (balls_red balls_white balls_yellow : Nat)
  (hr : balls_red = 10) (hw : balls_white = 10) (hy : balls_yellow = 10) :
  ∃ n, n >= 11 ∧ 
       ∀ draws : Fin n → Fin (balls_red + balls_white + balls_yellow), 
       ∃ i j, draws i ≠ draws j := 
sorry

-- Second statement: Ensuring two yellow balls
theorem ensure_two_yellow_balls (balls_red balls_white balls_yellow : Nat)
  (hr : balls_red = 10) (hw : balls_white = 10) (hy : balls_yellow = 10) :
  ∃ n, n >= 22 ∧
       ∀ draws : Fin n → Fin (balls_red + balls_white + balls_yellow), 
       ∃ i j, (draws i).val - balls_red - balls_white < balls_yellow ∧ 
              (draws j).val - balls_red - balls_white < balls_yellow ∧
              draws i = draws j := 
sorry

end NUMINAMATH_GPT_ensure_two_different_colors_ensure_two_yellow_balls_l1111_111140


namespace NUMINAMATH_GPT_sum_infinite_series_eq_l1111_111198

theorem sum_infinite_series_eq : 
  (∑' n : ℕ, if n > 0 then ((3 * n - 2 : ℝ) / (n * (n + 1) * (n + 3))) else 0) = (7 / 12) :=
by
  sorry

end NUMINAMATH_GPT_sum_infinite_series_eq_l1111_111198


namespace NUMINAMATH_GPT_Marilyn_end_caps_l1111_111143

def starting_caps := 51
def shared_caps := 36
def ending_caps := starting_caps - shared_caps

theorem Marilyn_end_caps : ending_caps = 15 := by
  -- proof omitted
  sorry

end NUMINAMATH_GPT_Marilyn_end_caps_l1111_111143


namespace NUMINAMATH_GPT_condition_sufficient_not_necessary_l1111_111117

theorem condition_sufficient_not_necessary
  (A B C D : Prop)
  (h1 : A → B)
  (h2 : B ↔ C)
  (h3 : C → D) :
  (A → D) ∧ ¬(D → A) :=
by
  sorry

end NUMINAMATH_GPT_condition_sufficient_not_necessary_l1111_111117


namespace NUMINAMATH_GPT_line_BC_eq_l1111_111110

def altitude1 (x y : ℝ) : Prop := x + y = 0
def altitude2 (x y : ℝ) : Prop := 2 * x - 3 * y + 1 = 0
def point_A : ℝ × ℝ := (1, 2)

def line_eq (a b c x y : ℝ) : Prop := a * x + b * y + c = 0

theorem line_BC_eq (x y : ℝ) :
  (∃ b c : ℝ × ℝ, altitude1 b.1 b.2 ∧ altitude2 c.1 c.2 ∧
                   line_eq 2 3 7 b.1 b.2 ∧ line_eq 2 3 7 c.1 c.2 ∧
                   b ≠ c) → 
    line_eq 2 3 7 x y :=
by sorry

end NUMINAMATH_GPT_line_BC_eq_l1111_111110


namespace NUMINAMATH_GPT_overall_average_marks_l1111_111157

theorem overall_average_marks
  (avg_A : ℝ) (n_A : ℕ) (avg_B : ℝ) (n_B : ℕ) (avg_C : ℝ) (n_C : ℕ)
  (h_avg_A : avg_A = 40) (h_n_A : n_A = 12)
  (h_avg_B : avg_B = 60) (h_n_B : n_B = 28)
  (h_avg_C : avg_C = 55) (h_n_C : n_C = 15) :
  ((n_A * avg_A) + (n_B * avg_B) + (n_C * avg_C)) / (n_A + n_B + n_C) = 54.27 := by
  sorry

end NUMINAMATH_GPT_overall_average_marks_l1111_111157


namespace NUMINAMATH_GPT_not_sum_three_nonzero_squares_l1111_111175

-- To state that 8n - 1 is not the sum of three non-zero squares
theorem not_sum_three_nonzero_squares (n : ℕ) :
  ¬ (∃ a b c : ℕ, a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ 8 * n - 1 = a^2 + b^2 + c^2) := by
  sorry

end NUMINAMATH_GPT_not_sum_three_nonzero_squares_l1111_111175


namespace NUMINAMATH_GPT_eggs_volume_correct_l1111_111163

def raw_spinach_volume : ℕ := 40
def cooking_reduction_ratio : ℚ := 0.20
def cream_cheese_volume : ℕ := 6
def total_quiche_volume : ℕ := 18
def cooked_spinach_volume := (raw_spinach_volume : ℚ) * cooking_reduction_ratio
def combined_spinach_and_cream_cheese_volume := cooked_spinach_volume + (cream_cheese_volume : ℚ)
def eggs_volume := (total_quiche_volume : ℚ) - combined_spinach_and_cream_cheese_volume

theorem eggs_volume_correct : eggs_volume = 4 := by
  sorry

end NUMINAMATH_GPT_eggs_volume_correct_l1111_111163


namespace NUMINAMATH_GPT_find_a200_l1111_111132

def seq (a : ℕ → ℕ) : Prop :=
a 1 = 1 ∧ ∀ n ≥ 1, a (n + 1) = a n + 2 * a n / n

theorem find_a200 (a : ℕ → ℕ) (h : seq a) : a 200 = 20100 :=
sorry

end NUMINAMATH_GPT_find_a200_l1111_111132


namespace NUMINAMATH_GPT_min_value_expression_l1111_111126

variable {m n : ℝ}

theorem min_value_expression (hm : m > 0) (hn : n > 0) (hperp : m + n = 1) :
  ∃ (m n : ℝ), (1 / m + 2 / n = 3 + 2 * Real.sqrt 2) :=
by 
  sorry

end NUMINAMATH_GPT_min_value_expression_l1111_111126


namespace NUMINAMATH_GPT_three_digit_oddfactors_count_is_22_l1111_111102

theorem three_digit_oddfactors_count_is_22 :
  let lower_bound := 100
  let upper_bound := 999
  let smallest_square_root := 10
  let largest_square_root := 31
  let count := largest_square_root - smallest_square_root + 1
  ∀ n : ℤ, lower_bound ≤ n ∧ n ≤ upper_bound → (∃ k : ℤ, n = k * k →
  (smallest_square_root ≤ k ∧ k ≤ largest_square_root)) → count = 22 :=
by
  intros lower_bound upper_bound smallest_square_root largest_square_root count n hn h
  have smallest_square_root := 10
  have largest_square_root := 31
  have count := largest_square_root - smallest_square_root + 1
  sorry

end NUMINAMATH_GPT_three_digit_oddfactors_count_is_22_l1111_111102


namespace NUMINAMATH_GPT_solve_for_x_l1111_111167

theorem solve_for_x (x : ℝ) :
  let area_square1 := (2 * x) ^ 2
  let area_square2 := (5 * x) ^ 2
  let area_triangle := 0.5 * (2 * x) * (5 * x)
  (area_square1 + area_square2 + area_triangle = 850) → x = 5 := by
  sorry

end NUMINAMATH_GPT_solve_for_x_l1111_111167


namespace NUMINAMATH_GPT_compute_infinite_series_l1111_111145

noncomputable def infinite_series (c d : ℝ) (hcd : c > d) : ℝ :=
  ∑' n, 1 / (((n - 1 : ℝ) * c - (n - 2 : ℝ) * d) * (n * c - (n - 1 : ℝ) * d))

theorem compute_infinite_series (c d : ℝ) (hcd : c > d) :
  infinite_series c d hcd = 1 / ((c - d) * d) :=
by
  sorry

end NUMINAMATH_GPT_compute_infinite_series_l1111_111145


namespace NUMINAMATH_GPT_only_one_way_to_center_l1111_111172

def is_center {n : ℕ} (grid_size n : ℕ) (coord : ℕ × ℕ) : Prop :=
  coord = (grid_size / 2 + 1, grid_size / 2 + 1)

def count_ways_to_center : ℕ :=
  if h : (1 <= 3 ∧ 3 <= 5) then 1 else 0

theorem only_one_way_to_center : count_ways_to_center = 1 := by
  sorry

end NUMINAMATH_GPT_only_one_way_to_center_l1111_111172


namespace NUMINAMATH_GPT_maximum_bags_of_milk_l1111_111153

theorem maximum_bags_of_milk (bag_cost : ℚ) (promotion : ℕ → ℕ) (total_money : ℚ) 
  (h1 : bag_cost = 2.5) 
  (h2 : promotion 2 = 3) 
  (h3 : total_money = 30) : 
  ∃ n, n = 18 ∧ (total_money >= n * bag_cost - (n / 3) * bag_cost) :=
by
  sorry

end NUMINAMATH_GPT_maximum_bags_of_milk_l1111_111153


namespace NUMINAMATH_GPT_range_of_a_l1111_111118

def f (x a : ℝ) : ℝ := x^2 + a * x

theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, f x a = 0) ∧ (∃ x : ℝ, f (f x a) a = 0) → (0 ≤ a ∧ a < 4) :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l1111_111118


namespace NUMINAMATH_GPT_no_integer_solutions_l1111_111113

theorem no_integer_solutions (x y : ℤ) : 19 * x^3 - 84 * y^2 ≠ 1984 :=
by
  sorry

end NUMINAMATH_GPT_no_integer_solutions_l1111_111113


namespace NUMINAMATH_GPT_parabola_x_intercepts_l1111_111174

theorem parabola_x_intercepts : 
  ∃! x : ℝ, ∃ y : ℝ, y = 0 ∧ x = -3 * y ^ 2 + 2 * y + 3 :=
by 
  sorry

end NUMINAMATH_GPT_parabola_x_intercepts_l1111_111174


namespace NUMINAMATH_GPT_find_C_plus_D_l1111_111114

noncomputable def polynomial_divisible (x : ℝ) (C : ℝ) (D : ℝ) : Prop := 
  ∃ (ω : ℝ), ω^2 + ω + 1 = 0 ∧ ω^104 + C*ω + D = 0

theorem find_C_plus_D (C D : ℝ) : 
  (∃ x : ℝ, polynomial_divisible x C D) → C + D = 2 :=
by
  sorry

end NUMINAMATH_GPT_find_C_plus_D_l1111_111114


namespace NUMINAMATH_GPT_find_c_k_l1111_111170

-- Definitions of the arithmetic and geometric sequences
def a (n d : ℕ) := 1 + (n - 1) * d
def b (n r : ℕ) := r ^ (n - 1)
def c (n d r : ℕ) := a n d + b n r

-- Conditions for the specific problem
theorem find_c_k (k d r : ℕ) (h1 : 1 + (k - 2) * d + r ^ (k - 2) = 150) (h2 : 1 + k * d + r ^ k = 1500) : c k d r = 314 :=
by
  sorry

end NUMINAMATH_GPT_find_c_k_l1111_111170


namespace NUMINAMATH_GPT_field_width_calculation_l1111_111197

theorem field_width_calculation (w : ℝ) (h_length : length = 24) (h_length_width_relation : length = 2 * w - 3) : w = 13.5 :=
by 
  sorry

end NUMINAMATH_GPT_field_width_calculation_l1111_111197


namespace NUMINAMATH_GPT_mildred_weight_is_correct_l1111_111107

noncomputable def carol_weight := 9
noncomputable def mildred_weight := carol_weight + 50

theorem mildred_weight_is_correct : mildred_weight = 59 :=
by 
  -- the proof is omitted
  sorry

end NUMINAMATH_GPT_mildred_weight_is_correct_l1111_111107


namespace NUMINAMATH_GPT_minimum_yellow_balls_l1111_111156

theorem minimum_yellow_balls (g o y : ℕ) :
  (o ≥ (1/3:ℝ) * g) ∧ (o ≤ (1/4:ℝ) * y) ∧ (g + o ≥ 75) → y ≥ 76 :=
sorry

end NUMINAMATH_GPT_minimum_yellow_balls_l1111_111156


namespace NUMINAMATH_GPT_radius_of_circle_l1111_111134

theorem radius_of_circle (P : ℝ) (PQ QR : ℝ) (distance_center_P : ℝ) (r : ℝ) :
  P = 17 ∧ PQ = 12 ∧ QR = 8 ∧ (PQ * (PQ + QR) = (distance_center_P - r) * (distance_center_P + r)) → r = 7 :=
by
  sorry

end NUMINAMATH_GPT_radius_of_circle_l1111_111134


namespace NUMINAMATH_GPT_common_difference_arith_seq_l1111_111116

theorem common_difference_arith_seq (a : ℕ → ℝ) (d : ℝ)
    (h₀ : a 1 + a 5 = 10)
    (h₁ : a 4 = 7)
    (h₂ : ∀ n, a (n + 1) = a n + d) : 
    d = 2 := by
  sorry

end NUMINAMATH_GPT_common_difference_arith_seq_l1111_111116


namespace NUMINAMATH_GPT_fraction_half_way_l1111_111181

theorem fraction_half_way (x y : ℚ) (h1 : x = 3 / 4) (h2 : y = 5 / 6) : (x + y) / 2 = 19 / 24 := by
  sorry

end NUMINAMATH_GPT_fraction_half_way_l1111_111181


namespace NUMINAMATH_GPT_slide_vs_slip_l1111_111171

noncomputable def ladder : Type := sorry

def slide_distance (ladder : ladder) : ℝ := sorry
def slip_distance (ladder : ladder) : ℝ := sorry
def is_right_triangle (ladder : ladder) : Prop := sorry

theorem slide_vs_slip (l : ladder) (h : is_right_triangle l) : slip_distance l > slide_distance l :=
sorry

end NUMINAMATH_GPT_slide_vs_slip_l1111_111171


namespace NUMINAMATH_GPT_number_of_children_at_reunion_l1111_111135

theorem number_of_children_at_reunion (A C : ℕ) 
    (h1 : 3 * A = C)
    (h2 : 2 * A / 3 = 10) : 
  C = 45 :=
by
  sorry

end NUMINAMATH_GPT_number_of_children_at_reunion_l1111_111135


namespace NUMINAMATH_GPT_smallest_degree_of_f_l1111_111142

theorem smallest_degree_of_f (p : Polynomial ℂ) (hp_deg : p.degree < 1992)
  (hp0 : p.eval 0 ≠ 0) (hp1 : p.eval 1 ≠ 0) (hp_1 : p.eval (-1) ≠ 0) :
  ∃ f g : Polynomial ℂ, 
    (Polynomial.derivative^[1992] (p / (X^3 - X))) = f / g ∧ f.degree = 3984 := 
sorry

end NUMINAMATH_GPT_smallest_degree_of_f_l1111_111142


namespace NUMINAMATH_GPT_negation_proposition_l1111_111112

theorem negation_proposition : ¬ (∀ x : ℝ, (1 < x) → x^3 > x^(1/3)) ↔ ∃ x : ℝ, (1 < x) ∧ x^3 ≤ x^(1/3) := by
  sorry

end NUMINAMATH_GPT_negation_proposition_l1111_111112


namespace NUMINAMATH_GPT_find_added_number_l1111_111138

def S₁₅ := 15 * 17
def S₁₆ := 16 * 20
def added_number := S₁₆ - S₁₅

theorem find_added_number : added_number = 65 :=
by
  sorry

end NUMINAMATH_GPT_find_added_number_l1111_111138


namespace NUMINAMATH_GPT_flat_rate_first_night_l1111_111125

-- Definitions of conditions
def total_cost_sarah (f n : ℕ) := f + 3 * n = 210
def total_cost_mark (f n : ℕ) := f + 7 * n = 450

-- Main theorem to be proven
theorem flat_rate_first_night : 
  ∃ f n : ℕ, total_cost_sarah f n ∧ total_cost_mark f n ∧ f = 30 :=
by
  sorry

end NUMINAMATH_GPT_flat_rate_first_night_l1111_111125


namespace NUMINAMATH_GPT_geometric_sequence_analogy_l1111_111173

variables {a_n b_n : ℕ → ℕ} {S T : ℕ → ℕ}

-- Conditions for the arithmetic sequence
def is_arithmetic_sequence_sum (S : ℕ → ℕ) :=
  S 8 - S 4 = 2 * (S 4) ∧ S 12 - S 8 = 2 * (S 8 - S 4)

-- Conditions for the geometric sequence
def is_geometric_sequence_product (T : ℕ → ℕ) :=
  (T 8 / T 4) = (T 4) ∧ (T 12 / T 8) = (T 8 / T 4)

-- Statement of the proof problem
theorem geometric_sequence_analogy
  (h_arithmetic : is_arithmetic_sequence_sum S)
  (h_geometric_nil : is_geometric_sequence_product T) :
  T 4 / T 4 = 1 ∧
  (T 8 / T 4) / (T 8 / T 4) = 1 ∧
  (T 12 / T 8) / (T 12 / T 8) = 1 := 
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_analogy_l1111_111173


namespace NUMINAMATH_GPT_train_speed_l1111_111151

theorem train_speed (length : ℝ) (time : ℝ) (conversion_factor : ℝ)
  (h1 : length = 500) (h2 : time = 5) (h3 : conversion_factor = 3.6) :
  (length / time) * conversion_factor = 360 :=
by
  sorry

end NUMINAMATH_GPT_train_speed_l1111_111151


namespace NUMINAMATH_GPT_hyperbola_foci_distance_l1111_111187

theorem hyperbola_foci_distance :
  let a := Real.sqrt 25
  let b := Real.sqrt 9
  let c := Real.sqrt (a ^ 2 + b ^ 2)
  let distance := 2 * c
  distance = 2 * Real.sqrt 34 :=
by
  let a := Real.sqrt 25
  let b := Real.sqrt 9
  let c := Real.sqrt (a ^ 2 + b ^ 2)
  let distance := 2 * c
  exact sorry

end NUMINAMATH_GPT_hyperbola_foci_distance_l1111_111187


namespace NUMINAMATH_GPT_math_problem_l1111_111161

-- Define constants and conversions from decimal/mixed numbers to fractions
def thirteen_and_three_quarters : ℚ := 55 / 4
def nine_and_sixth : ℚ := 55 / 6
def one_point_two : ℚ := 1.2
def ten_point_three : ℚ := 103 / 10
def eight_and_half : ℚ := 17 / 2
def six_point_eight : ℚ := 34 / 5
def three_and_three_fifths : ℚ := 18 / 5
def five_and_five_sixths : ℚ := 35 / 6
def three_and_two_thirds : ℚ := 11 / 3
def three_and_one_sixth : ℚ := 19 / 6
def fifty_six : ℚ := 56
def twenty_seven_and_sixth : ℚ := 163 / 6

def E : ℚ := 
  ((thirteen_and_three_quarters + nine_and_sixth) * one_point_two) / ((ten_point_three - eight_and_half) * (5 / 9)) + 
  ((six_point_eight - three_and_three_fifths) * five_and_five_sixths) / ((three_and_two_thirds - three_and_one_sixth) * fifty_six) - 
  twenty_seven_and_sixth

theorem math_problem : E = 29 / 3 := by
  sorry

end NUMINAMATH_GPT_math_problem_l1111_111161


namespace NUMINAMATH_GPT_units_digit_of_x4_plus_inv_x4_l1111_111104

theorem units_digit_of_x4_plus_inv_x4 (x : ℝ) (hx : x^2 - 13 * x + 1 = 0) : 
  (x^4 + x⁻¹ ^ 4) % 10 = 7 := sorry

end NUMINAMATH_GPT_units_digit_of_x4_plus_inv_x4_l1111_111104


namespace NUMINAMATH_GPT_inequality_solution_empty_set_l1111_111106

theorem inequality_solution_empty_set : ∀ x : ℝ, ¬ (x * (2 - x) > 3) :=
by
  -- Translate the condition and show that there are no x satisfying the inequality
  sorry

end NUMINAMATH_GPT_inequality_solution_empty_set_l1111_111106


namespace NUMINAMATH_GPT_margo_paired_with_irma_probability_l1111_111177

theorem margo_paired_with_irma_probability :
  let n := 15
  let total_outcomes := n
  let favorable_outcomes := 1
  let probability := favorable_outcomes / total_outcomes
  probability = (1 / 15) :=
by
  let n := 15
  let total_outcomes := n
  let favorable_outcomes := 1
  let probability := favorable_outcomes / total_outcomes
  have h : probability = 1 / 15 := by
    -- skipping the proof details as per instructions
    sorry
  exact h

end NUMINAMATH_GPT_margo_paired_with_irma_probability_l1111_111177


namespace NUMINAMATH_GPT_cardinality_bound_l1111_111154

theorem cardinality_bound {m n : ℕ} (hm : m > 1) (hn : n > 1)
  (S : Finset ℕ) (hS : S.card = n)
  (A : Fin m → Finset ℕ)
  (h : ∀ (x y : ℕ), x ∈ S → y ∈ S → x ≠ y → ∃ i, (x ∈ A i ∧ y ∉ (A i)) ∨ (x ∉ (A i) ∧ y ∈ A i)) :
  n ≤ 2^m :=
sorry

end NUMINAMATH_GPT_cardinality_bound_l1111_111154


namespace NUMINAMATH_GPT_solve_xyz_l1111_111166

theorem solve_xyz (x y z : ℕ) (h1 : x > y) (h2 : y > z) (h3 : z > 0) (h4 : x^2 = y * 2^z + 1) :
  (z ≥ 4 ∧ x = 2^(z-1) + 1 ∧ y = 2^(z-2) + 1) ∨
  (z ≥ 5 ∧ x = 2^(z-1) - 1 ∧ y = 2^(z-2) - 1) ∨
  (z ≥ 3 ∧ x = 2^z - 1 ∧ y = 2^z - 2) :=
sorry

end NUMINAMATH_GPT_solve_xyz_l1111_111166


namespace NUMINAMATH_GPT_fixed_amount_at_least_190_l1111_111128

variable (F S : ℝ)

theorem fixed_amount_at_least_190
  (h1 : S = 7750)
  (h2 : F + 0.04 * S ≥ 500) :
  F ≥ 190 := by
  sorry

end NUMINAMATH_GPT_fixed_amount_at_least_190_l1111_111128


namespace NUMINAMATH_GPT_find_d_l1111_111131

theorem find_d (y d : ℝ) (hy : y > 0) (h : (8 * y) / 20 + (3 * y) / d = 0.7 * y) : d = 10 :=
by
  sorry

end NUMINAMATH_GPT_find_d_l1111_111131


namespace NUMINAMATH_GPT_arithmetic_geometric_sequence_ab_l1111_111146

theorem arithmetic_geometric_sequence_ab :
  ∀ (a l m b n : ℤ), 
    (b < 0) → 
    (2 * a = -10) → 
    (b^2 = 9) → 
    ab = 15 :=
by
  intros a l m b n hb ha hb_eq
  sorry

end NUMINAMATH_GPT_arithmetic_geometric_sequence_ab_l1111_111146


namespace NUMINAMATH_GPT_max_points_of_intersection_l1111_111180

-- Define the conditions
variable {α : Type*} [DecidableEq α]
variable (L : Fin 100 → α → α → Prop) -- Representation of the lines

-- Define property of being parallel
variable (are_parallel : ∀ {n : ℕ}, L (5 * n) = L (5 * n + 5))

-- Define property of passing through point B
variable (passes_through_B : ∀ {n : ℕ}, ∃ P B, L (5 * n - 4) P B)

-- Prove the stated result
theorem max_points_of_intersection : 
  ∃ max_intersections, max_intersections = 4571 :=
by {
  sorry
}

end NUMINAMATH_GPT_max_points_of_intersection_l1111_111180


namespace NUMINAMATH_GPT_triangle_altitude_length_l1111_111127

variable (AB AC BC BA1 AA1 : ℝ)
variable (eq1 : AB = 8)
variable (eq2 : AC = 10)
variable (eq3 : BC = 12)

theorem triangle_altitude_length (h : ∃ AA1, AA1 * AA1 + BA1 * BA1 = 64 ∧ 
                                AA1 * AA1 + (BC - BA1) * (BC - BA1) = 100) :
    BA1 = 4.5 := by
  sorry 

end NUMINAMATH_GPT_triangle_altitude_length_l1111_111127


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_condition_l1111_111111

def p (x : ℝ) : Prop := |4 * x - 3| ≤ 1
def q (x a : ℝ) : Prop := x^2 - (2 * a + 1) * x + a^2 + a ≤ 0

theorem necessary_but_not_sufficient_condition (a : ℝ) :
  (∀ x : ℝ, p x → q x a) ∧ ¬ (∀ x : ℝ, q x a → p x) ↔ (0 ≤ a ∧ a ≤ 1/2) :=
sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_condition_l1111_111111


namespace NUMINAMATH_GPT_vincent_earnings_after_5_days_l1111_111184

def fantasy_book_price : ℕ := 4
def daily_fantasy_books_sold : ℕ := 5
def literature_book_price : ℕ := fantasy_book_price / 2
def daily_literature_books_sold : ℕ := 8
def days : ℕ := 5

def daily_earnings : ℕ :=
  (fantasy_book_price * daily_fantasy_books_sold) +
  (literature_book_price * daily_literature_books_sold)

def total_earnings (d : ℕ) : ℕ :=
  daily_earnings * d

theorem vincent_earnings_after_5_days : total_earnings days = 180 := by
  sorry

end NUMINAMATH_GPT_vincent_earnings_after_5_days_l1111_111184


namespace NUMINAMATH_GPT_price_per_book_sold_l1111_111193

-- Definitions based on the given conditions
def total_books_before_sale : ℕ := 3 * 50
def books_sold : ℕ := 2 * 50
def total_amount_received : ℕ := 500

-- Target statement to be proved
theorem price_per_book_sold :
  (total_amount_received : ℚ) / books_sold = 5 :=
sorry

end NUMINAMATH_GPT_price_per_book_sold_l1111_111193


namespace NUMINAMATH_GPT_two_by_three_grid_count_l1111_111182

noncomputable def valid2x3Grids : Nat :=
  let valid_grids : Nat := 9
  valid_grids

theorem two_by_three_grid_count : valid2x3Grids = 9 := by
  -- Skipping the proof steps, but stating the theorem.
  sorry

end NUMINAMATH_GPT_two_by_three_grid_count_l1111_111182


namespace NUMINAMATH_GPT_num_factors_of_90_multiple_of_6_l1111_111101

def is_factor (m n : ℕ) : Prop := n % m = 0
def is_multiple_of (m n : ℕ) : Prop := n % m = 0

theorem num_factors_of_90_multiple_of_6 : 
  ∃ (count : ℕ), count = 4 ∧ ∀ x, is_factor x 90 → is_multiple_of 6 x → x > 0 :=
sorry

end NUMINAMATH_GPT_num_factors_of_90_multiple_of_6_l1111_111101


namespace NUMINAMATH_GPT_Collin_total_petals_l1111_111147

-- Definitions of the conditions
def initial_flowers_Collin : ℕ := 25
def flowers_Ingrid : ℕ := 33
def petals_per_flower : ℕ := 4
def third_of_flowers_Ingrid : ℕ := flowers_Ingrid / 3

-- Total number of flowers Collin has after receiving from Ingrid
def total_flowers_Collin : ℕ := initial_flowers_Collin + third_of_flowers_Ingrid

-- Total number of petals Collin has
def total_petals_Collin : ℕ := total_flowers_Collin * petals_per_flower

-- The theorem to be proved
theorem Collin_total_petals : total_petals_Collin = 144 := by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_Collin_total_petals_l1111_111147


namespace NUMINAMATH_GPT_RS_plus_ST_l1111_111137

theorem RS_plus_ST {a b c d e : ℕ} 
  (h1 : a = 68) 
  (h2 : b = 10) 
  (h3 : c = 7) 
  (h4 : d = 6) 
  : e = 3 :=
sorry

end NUMINAMATH_GPT_RS_plus_ST_l1111_111137


namespace NUMINAMATH_GPT_general_form_line_eq_line_passes_fixed_point_l1111_111105

-- (Ⅰ) Prove that if m = 1/2 and point P (1/2, 2), the general form equation of line l is 2x - y + 1 = 0
theorem general_form_line_eq (m n : ℝ) (h1 : m = 1/2) (h2 : n = 1 / (1 - m)) (h3 : n = 2) (P : (ℝ × ℝ)) (hP : P = (1/2, 2)) :
  ∃ (a b c : ℝ), a * P.1 + b * P.2 + c = 0 ∧ a = 2 ∧ b = -1 ∧ c = 1 := sorry

-- (Ⅱ) Prove that if point P(m,n) is on the line l0, then the line mx + (n-1)y + n + 5 = 0 passes through a fixed point, coordinates (1,1)
theorem line_passes_fixed_point (m n : ℝ) (h1 : m + 2 * n + 4 = 0) :
  ∀ (x y : ℝ), (m * x + (n - 1) * y + n + 5 = 0) ↔ (x = 1) ∧ (y = 1) := sorry

end NUMINAMATH_GPT_general_form_line_eq_line_passes_fixed_point_l1111_111105


namespace NUMINAMATH_GPT_student_rank_from_left_l1111_111124

theorem student_rank_from_left (total_students rank_from_right rank_from_left : ℕ) 
  (h1 : total_students = 21) 
  (h2 : rank_from_right = 16) 
  (h3 : total_students = rank_from_right + rank_from_left - 1) 
  : rank_from_left = 6 := 
by 
  sorry

end NUMINAMATH_GPT_student_rank_from_left_l1111_111124


namespace NUMINAMATH_GPT_polynomial_remainder_l1111_111194

theorem polynomial_remainder (x : ℤ) :
  let dividend := 3*x^3 - 2*x^2 - 23*x + 60
  let divisor := x - 4
  let quotient := 3*x^2 + 10*x + 17
  let remainder := 128
  dividend = divisor * quotient + remainder :=
by 
  -- proof steps would go here, but we use "sorry" as instructed
  sorry

end NUMINAMATH_GPT_polynomial_remainder_l1111_111194


namespace NUMINAMATH_GPT_find_a3_l1111_111121

noncomputable def geometric_sequence (a : ℕ → ℝ) (r : ℝ) : Prop :=
∀ n, a (n + 1) = a n * r

theorem find_a3 (a : ℕ → ℝ) (r : ℝ)
  (h1 : geometric_sequence a r)
  (h2 : a 0 * a 1 * a 2 * a 3 * a 4 = 32):
  a 2 = 2 :=
sorry

end NUMINAMATH_GPT_find_a3_l1111_111121


namespace NUMINAMATH_GPT_problem_solution_l1111_111133

def otimes (a b : ℚ) : ℚ := (a ^ 3) / (b ^ 2)

theorem problem_solution :
  (otimes (otimes 2 3) 4) - (otimes 2 (otimes 3 4)) = (-2016) / 729 := by
  sorry

end NUMINAMATH_GPT_problem_solution_l1111_111133


namespace NUMINAMATH_GPT_runner_speed_ratio_l1111_111196

theorem runner_speed_ratio (d s u v_f v_s : ℝ) (hs : s ≠ 0) (hu : u ≠ 0)
  (H1 : (v_f + v_s) * s = d) (H2 : (v_f - v_s) * u = v_s * u) :
  v_f / v_s = 2 :=
by
  sorry

end NUMINAMATH_GPT_runner_speed_ratio_l1111_111196


namespace NUMINAMATH_GPT_inverse_proportional_l1111_111168

-- Define the variables and the condition
variables {R : Type*} [CommRing R] {x y k : R}
-- Assuming x and y are non-zero
variables (hx : x ≠ 0) (hy : y ≠ 0)

-- Define the constant product relationship
def product_constant (x y k : R) : Prop := x * y = k

-- The main statement that needs to be proved
theorem inverse_proportional (h : product_constant x y k) : 
  ∃ k, x * y = k :=
by sorry

end NUMINAMATH_GPT_inverse_proportional_l1111_111168


namespace NUMINAMATH_GPT_toms_initial_investment_l1111_111158

theorem toms_initial_investment (t j k : ℕ) (hj_neq_ht : t ≠ j) (hk_neq_ht : t ≠ k) (hj_neq_hk : j ≠ k) 
  (h1 : t + j + k = 1200) 
  (h2 : t - 150 + 3 * j + 3 * k = 1800) : 
  t = 825 := 
sorry

end NUMINAMATH_GPT_toms_initial_investment_l1111_111158


namespace NUMINAMATH_GPT_beads_bracelet_rotational_symmetry_l1111_111123

theorem beads_bracelet_rotational_symmetry :
  let n := 8
  let factorial := Nat.factorial
  (factorial n / n = 5040) := by
  sorry

end NUMINAMATH_GPT_beads_bracelet_rotational_symmetry_l1111_111123


namespace NUMINAMATH_GPT_arithmetic_sequence_50th_term_l1111_111103

theorem arithmetic_sequence_50th_term :
  let a1 := 3
  let d := 2
  let n := 50
  let a_n := a1 + (n - 1) * d
  a_n = 101 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_50th_term_l1111_111103


namespace NUMINAMATH_GPT_second_and_third_finish_job_together_in_8_days_l1111_111130

theorem second_and_third_finish_job_together_in_8_days
  (x y : ℕ)
  (h1 : 1/24 + 1/x + 1/y = 1/6) :
  1/x + 1/y = 1/8 :=
by sorry

end NUMINAMATH_GPT_second_and_third_finish_job_together_in_8_days_l1111_111130


namespace NUMINAMATH_GPT_minimum_apples_l1111_111165

theorem minimum_apples (n : ℕ) (A : ℕ) (h1 : A = 25 * n + 24) (h2 : A > 300) : A = 324 :=
sorry

end NUMINAMATH_GPT_minimum_apples_l1111_111165


namespace NUMINAMATH_GPT_amy_tips_calculation_l1111_111190

theorem amy_tips_calculation 
  (hourly_wage : ℝ) (hours_worked : ℝ) (total_earnings : ℝ) 
  (h_wage : hourly_wage = 2)
  (h_hours : hours_worked = 7)
  (h_total : total_earnings = 23) : 
  total_earnings - (hourly_wage * hours_worked) = 9 := 
sorry

end NUMINAMATH_GPT_amy_tips_calculation_l1111_111190


namespace NUMINAMATH_GPT_salary_reduction_percentage_l1111_111188

theorem salary_reduction_percentage
  (S : ℝ) 
  (h : S * (1 - R / 100) = S / 1.388888888888889): R = 28 :=
sorry

end NUMINAMATH_GPT_salary_reduction_percentage_l1111_111188
