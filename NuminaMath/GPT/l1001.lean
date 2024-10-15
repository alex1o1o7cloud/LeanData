import Mathlib

namespace NUMINAMATH_GPT_soup_options_l1001_100132

-- Define the given conditions
variables (lettuce_types tomato_types olive_types total_options : ℕ)
variable (S : ℕ)

-- State the conditions
theorem soup_options :
  lettuce_types = 2 →
  tomato_types = 3 →
  olive_types = 4 →
  total_options = 48 →
  (lettuce_types * tomato_types * olive_types * S = total_options) →
  S = 2 :=
by
  sorry

end NUMINAMATH_GPT_soup_options_l1001_100132


namespace NUMINAMATH_GPT_smallest_factor_of_36_sum_4_l1001_100106

theorem smallest_factor_of_36_sum_4 : ∃ a b c : ℤ, (a * b * c = 36) ∧ (a + b + c = 4) ∧ (a = -4 ∨ b = -4 ∨ c = -4) :=
by
  sorry

end NUMINAMATH_GPT_smallest_factor_of_36_sum_4_l1001_100106


namespace NUMINAMATH_GPT_quadratic_roots_proof_l1001_100186

theorem quadratic_roots_proof (b c : ℤ) :
  (∀ x : ℤ, x^2 + b * x + c = 0 ↔ (x = 1 ∨ x = -2)) → (b = 1 ∧ c = -2) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_roots_proof_l1001_100186


namespace NUMINAMATH_GPT_div_problem_l1001_100178

variables (A B C : ℝ)

theorem div_problem (h1 : A = (2/3) * B) (h2 : B = (1/4) * C) (h3 : A + B + C = 527) : B = 93 :=
by {
  sorry
}

end NUMINAMATH_GPT_div_problem_l1001_100178


namespace NUMINAMATH_GPT_find_expression_l1001_100155

theorem find_expression (x : ℝ) (h : (1 / Real.cos (2022 * x)) + Real.tan (2022 * x) = 1 / 2022) :
  (1 / Real.cos (2022 * x)) - Real.tan (2022 * x) = 2022 :=
by
  sorry

end NUMINAMATH_GPT_find_expression_l1001_100155


namespace NUMINAMATH_GPT_isosceles_obtuse_triangle_smallest_angle_l1001_100101

theorem isosceles_obtuse_triangle_smallest_angle :
  ∀ (α β γ : ℝ), α = 1.8 * 90 ∧ β = γ ∧ α + β + γ = 180 → β = 9 :=
by
  intros α β γ h
  sorry

end NUMINAMATH_GPT_isosceles_obtuse_triangle_smallest_angle_l1001_100101


namespace NUMINAMATH_GPT_fibonacci_eighth_term_l1001_100147

theorem fibonacci_eighth_term
  (F : ℕ → ℕ)
  (h1 : F 9 = 34)
  (h2 : F 10 = 55)
  (fib : ∀ n, F (n + 2) = F (n + 1) + F n) :
  F 8 = 21 :=
by
  sorry

end NUMINAMATH_GPT_fibonacci_eighth_term_l1001_100147


namespace NUMINAMATH_GPT_find_possible_values_of_a_l1001_100151

noncomputable def P : Set ℝ := {x | x^2 + x - 6 = 0}
noncomputable def Q (a : ℝ) : Set ℝ := {x | a * x + 1 = 0}

theorem find_possible_values_of_a (a : ℝ) (h : Q a ⊆ P) :
  a = 0 ∨ a = -1/2 ∨ a = 1/3 := by
  sorry

end NUMINAMATH_GPT_find_possible_values_of_a_l1001_100151


namespace NUMINAMATH_GPT_min_throws_for_repeated_sum_l1001_100143

theorem min_throws_for_repeated_sum (n : ℕ) (h1 : 2 ≤ n) (h2 : n ≤ 16) : 
  ∃ m, m = 16 ∧ (∀ (k : ℕ), k < 16 → ∃ i < 16, ∃ j < 16, i ≠ j ∧ i + j = k) :=
by
  sorry

end NUMINAMATH_GPT_min_throws_for_repeated_sum_l1001_100143


namespace NUMINAMATH_GPT_negation_proposition_l1001_100130

theorem negation_proposition :
  (∀ x : ℝ, x^2 - x + 3 > 0) ↔ (∃ x : ℝ, x^2 - x + 3 ≤ 0) :=
by sorry

end NUMINAMATH_GPT_negation_proposition_l1001_100130


namespace NUMINAMATH_GPT_jade_transactions_l1001_100127

theorem jade_transactions (mabel anthony cal jade : ℕ) 
    (h1 : mabel = 90) 
    (h2 : anthony = mabel + (10 * mabel / 100)) 
    (h3 : cal = 2 * anthony / 3) 
    (h4 : jade = cal + 18) : 
    jade = 84 := by 
  -- Start with given conditions
  rw [h1] at h2 
  have h2a : anthony = 99 := by norm_num; exact h2 
  rw [h2a] at h3 
  have h3a : cal = 66 := by norm_num; exact h3 
  rw [h3a] at h4 
  norm_num at h4 
  exact h4

end NUMINAMATH_GPT_jade_transactions_l1001_100127


namespace NUMINAMATH_GPT_neutral_equilibrium_l1001_100183

noncomputable def equilibrium_ratio (r h : ℝ) : ℝ := r / h

theorem neutral_equilibrium (r h : ℝ) (k : ℝ) : (equilibrium_ratio r h = k) → (k = Real.sqrt 2) :=
by
  intro h1
  have h1' : (r / h = k) := h1
  sorry

end NUMINAMATH_GPT_neutral_equilibrium_l1001_100183


namespace NUMINAMATH_GPT_solution_set_of_inequality_l1001_100188

theorem solution_set_of_inequality : { x : ℝ | 0 < x ∧ x < 2 } = { x : ℝ | 0 < x ∧ x < 2 } :=
by
  sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l1001_100188


namespace NUMINAMATH_GPT_radius_of_circle_l1001_100137

theorem radius_of_circle :
  ∀ (r : ℝ), (π * r^2 = 2.5 * 2 * π * r) → r = 5 :=
by sorry

end NUMINAMATH_GPT_radius_of_circle_l1001_100137


namespace NUMINAMATH_GPT_runner_overtake_time_l1001_100180

theorem runner_overtake_time
  (L : ℝ)
  (v1 v2 v3 : ℝ)
  (h1 : v1 = v2 + L / 6)
  (h2 : v1 = v3 + L / 10) :
  L / (v3 - v2) = 15 := by
  sorry

end NUMINAMATH_GPT_runner_overtake_time_l1001_100180


namespace NUMINAMATH_GPT_bounded_sequence_l1001_100175

theorem bounded_sequence (a : ℕ → ℕ) (h1 : a 1 = 2) (h2 : a 2 = 2)
  (h_rec : ∀ n : ℕ, a (n + 2) = (a (n + 1) + a n) / Nat.gcd (a n) (a (n + 1))) :
  ∃ M : ℕ, ∀ n : ℕ, a n ≤ M := 
sorry

end NUMINAMATH_GPT_bounded_sequence_l1001_100175


namespace NUMINAMATH_GPT_chris_is_14_l1001_100149

-- Definitions from the given conditions
variables (a b c : ℕ)
variables (h1 : (a + b + c) / 3 = 10)
variables (h2 : c - 4 = a)
variables (h3 : b + 5 = (3 * (a + 5)) / 4)

theorem chris_is_14 (h1 : (a + b + c) / 3 = 10) (h2 : c - 4 = a) (h3 : b + 5 = (3 * (a + 5)) / 4) : c = 14 := 
sorry

end NUMINAMATH_GPT_chris_is_14_l1001_100149


namespace NUMINAMATH_GPT_problem_I3_1_l1001_100115

theorem problem_I3_1 (w x y z : ℝ) (h1 : w * x * y * z = 4) (h2 : w - x * y * z = 3) (h3 : w > 0) : 
  w = 4 :=
by
  sorry

end NUMINAMATH_GPT_problem_I3_1_l1001_100115


namespace NUMINAMATH_GPT_samantha_birth_year_l1001_100169

theorem samantha_birth_year (first_kangaroo_year birth_year kangaroo_freq : ℕ)
  (h_first_kangaroo: first_kangaroo_year = 1991)
  (h_kangaroo_freq: kangaroo_freq = 1)
  (h_samantha_age: ∃ y, y = (first_kangaroo_year + 9 * kangaroo_freq) ∧ 2000 - 14 = y) :
  birth_year = 1986 :=
by sorry

end NUMINAMATH_GPT_samantha_birth_year_l1001_100169


namespace NUMINAMATH_GPT_intersection_M_N_l1001_100195

def M : Set ℤ := {0, 1, 2}
def N : Set ℤ := {x | x^2 - 3*x + 2 ≤ 0}

theorem intersection_M_N:
  M ∩ N = {1, 2} :=
sorry

end NUMINAMATH_GPT_intersection_M_N_l1001_100195


namespace NUMINAMATH_GPT_grove_town_fall_expenditure_l1001_100167

-- Define the expenditures at the end of August and November
def expenditure_end_of_august : ℝ := 3.0
def expenditure_end_of_november : ℝ := 5.5

-- Define the spending during fall months (September, October, November)
def spending_during_fall_months : ℝ := 2.5

-- Statement to be proved
theorem grove_town_fall_expenditure :
  expenditure_end_of_november - expenditure_end_of_august = spending_during_fall_months :=
by
  sorry

end NUMINAMATH_GPT_grove_town_fall_expenditure_l1001_100167


namespace NUMINAMATH_GPT_gcd_max_1001_l1001_100119

theorem gcd_max_1001 (a b : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b = 1001) : 
  ∃ d, d = Nat.gcd a b ∧ d ≤ 143 := 
sorry

end NUMINAMATH_GPT_gcd_max_1001_l1001_100119


namespace NUMINAMATH_GPT_find_b_l1001_100110

variables (U : Set ℝ) (A : Set ℝ) (b : ℝ)

theorem find_b (hU : U = Set.univ)
               (hA : A = {x | 1 ≤ x ∧ x < b})
               (hComplA : U \ A = {x | x < 1 ∨ x ≥ 2}) :
  b = 2 :=
sorry

end NUMINAMATH_GPT_find_b_l1001_100110


namespace NUMINAMATH_GPT_initial_machines_l1001_100114

theorem initial_machines (r : ℝ) (x : ℕ) (h1 : x * 42 * r = 7 * 36 * r) : x = 6 :=
by
  sorry

end NUMINAMATH_GPT_initial_machines_l1001_100114


namespace NUMINAMATH_GPT_greatest_four_digit_divisible_by_conditions_l1001_100108

-- Definitions based on the conditions
def is_divisible_by (a b : ℕ) : Prop := ∃ k, a = k * b

def is_four_digit (n : ℕ) : Prop := n >= 1000 ∧ n < 10000

-- Problem statement: Finding the greatest 4-digit number divisible by 15, 25, 40, and 75
theorem greatest_four_digit_divisible_by_conditions :
  ∃ n, is_four_digit n ∧ is_divisible_by n 15 ∧ is_divisible_by n 25 ∧ is_divisible_by n 40 ∧ is_divisible_by n 75 ∧ n = 9600 :=
  sorry

end NUMINAMATH_GPT_greatest_four_digit_divisible_by_conditions_l1001_100108


namespace NUMINAMATH_GPT_sequence_is_increasing_l1001_100199

theorem sequence_is_increasing (a : ℕ → ℝ) (h : ∀ n : ℕ, a (n + 1) - a n = 2) : ∀ n : ℕ, a (n + 1) > a n :=
by
  intro n
  have h2 : a (n + 1) - a n = 2 := h n
  linarith

end NUMINAMATH_GPT_sequence_is_increasing_l1001_100199


namespace NUMINAMATH_GPT_total_bars_is_7_l1001_100174

variable (x : ℕ)

-- Each chocolate bar costs $3
def cost_per_bar := 3

-- Olivia sold all but 4 bars
def bars_sold (total_bars : ℕ) := total_bars - 4

-- Olivia made $9
def amount_made (total_bars : ℕ) := cost_per_bar * bars_sold total_bars

-- Given conditions
def condition1 (total_bars : ℕ) := amount_made total_bars = 9

-- Proof that the total number of bars is 7
theorem total_bars_is_7 : condition1 x -> x = 7 := by
  sorry

end NUMINAMATH_GPT_total_bars_is_7_l1001_100174


namespace NUMINAMATH_GPT_xiao_wang_exam_grades_l1001_100166

theorem xiao_wang_exam_grades 
  (x y : ℕ) 
  (h1 : (x * y + 98) / (x + 1) = y + 1)
  (h2 : (x * y + 98 + 70) / (x + 2) = y - 1) : 
  x + 2 = 10 ∧ y - 1 = 88 := 
by
  sorry

end NUMINAMATH_GPT_xiao_wang_exam_grades_l1001_100166


namespace NUMINAMATH_GPT_arctan_sum_eq_pi_div_two_l1001_100139

theorem arctan_sum_eq_pi_div_two : Real.arctan (3 / 7) + Real.arctan (7 / 3) = Real.pi / 2 := 
sorry

end NUMINAMATH_GPT_arctan_sum_eq_pi_div_two_l1001_100139


namespace NUMINAMATH_GPT_university_minimum_spend_l1001_100160

def box_length : ℕ := 20
def box_width : ℕ := 20
def box_height : ℕ := 15
def box_cost : ℝ := 1.20
def total_volume : ℝ := 3.06 * (10^6)

def box_volume : ℕ := box_length * box_width * box_height

noncomputable def number_of_boxes : ℕ := Nat.ceil (total_volume / box_volume)
noncomputable def total_cost : ℝ := number_of_boxes * box_cost

theorem university_minimum_spend : total_cost = 612 := by
  sorry

end NUMINAMATH_GPT_university_minimum_spend_l1001_100160


namespace NUMINAMATH_GPT_scaled_system_solution_l1001_100128

theorem scaled_system_solution (a1 b1 c1 a2 b2 c2 x y : ℝ) 
  (h1 : a1 * 8 + b1 * 3 = c1) 
  (h2 : a2 * 8 + b2 * 3 = c2) : 
  4 * a1 * 10 + 3 * b1 * 5 = 5 * c1 ∧ 4 * a2 * 10 + 3 * b2 * 5 = 5 * c2 := 
by 
  sorry

end NUMINAMATH_GPT_scaled_system_solution_l1001_100128


namespace NUMINAMATH_GPT_number_of_integers_x_l1001_100154

def is_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

def is_acute_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 > c^2 ∧ a^2 + c^2 > b^2 ∧ b^2 + c^2 > a^2

def valid_range_x (x : ℝ) : Prop :=
  13 < x ∧ x < 43

def conditions_for_acute_triangle (x : ℝ) : Prop :=
  (x > 28 ∧ x^2 < 1009) ∨ (x ≤ 28 ∧ x > 23.64)

theorem number_of_integers_x (count : ℤ) :
  (∃ (x : ℤ), valid_range_x x ∧ is_triangle 15 28 x ∧ is_acute_triangle 15 28 x ∧ conditions_for_acute_triangle x) →
  count = 8 :=
sorry

end NUMINAMATH_GPT_number_of_integers_x_l1001_100154


namespace NUMINAMATH_GPT_g_of_900_eq_34_l1001_100176

theorem g_of_900_eq_34 (g : ℕ+ → ℝ) 
  (h_mul : ∀ x y : ℕ+, g (x * y) = g x + g y)
  (h_30 : g 30 = 17)
  (h_60 : g 60 = 21) :
  g 900 = 34 :=
sorry

end NUMINAMATH_GPT_g_of_900_eq_34_l1001_100176


namespace NUMINAMATH_GPT_inequality_satisfied_l1001_100197

open Real

theorem inequality_satisfied (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a + b + c = 1) :
  a * sqrt b + b * sqrt c + c * sqrt a ≤ 1 / sqrt 3 :=
sorry

end NUMINAMATH_GPT_inequality_satisfied_l1001_100197


namespace NUMINAMATH_GPT_defective_rate_worker_y_l1001_100118

theorem defective_rate_worker_y (d_x d_y : ℝ) (f_y : ℝ) (total_defective_rate : ℝ) :
  d_x = 0.005 → f_y = 0.8 → total_defective_rate = 0.0074 → 
  (0.2 * d_x + f_y * d_y = total_defective_rate) → d_y = 0.008 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  sorry

end NUMINAMATH_GPT_defective_rate_worker_y_l1001_100118


namespace NUMINAMATH_GPT_rabbit_count_l1001_100157

-- Define the conditions
def original_rabbits : ℕ := 8
def new_rabbits_born : ℕ := 5

-- Define the total rabbits based on the conditions
def total_rabbits : ℕ := original_rabbits + new_rabbits_born

-- The statement to prove that the total number of rabbits is 13
theorem rabbit_count : total_rabbits = 13 :=
by
  -- Proof not needed, hence using sorry
  sorry

end NUMINAMATH_GPT_rabbit_count_l1001_100157


namespace NUMINAMATH_GPT_evaluate_at_5_l1001_100122

def f(x: ℝ) : ℝ := 3 * x^5 - 15 * x^4 + 27 * x^3 - 20 * x^2 - 72 * x + 40

theorem evaluate_at_5 : f 5 = 2515 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_at_5_l1001_100122


namespace NUMINAMATH_GPT_total_grazing_area_l1001_100189

-- Define the dimensions of the field
def field_width : ℝ := 46
def field_height : ℝ := 20

-- Define the length of the rope
def rope_length : ℝ := 17

-- Define the radius and position of the fenced area
def fenced_radius : ℝ := 5
def fenced_distance_x : ℝ := 25
def fenced_distance_y : ℝ := 10

-- Given the conditions, prove the total grazing area
theorem total_grazing_area (field_width field_height rope_length fenced_radius fenced_distance_x fenced_distance_y : ℝ) :
  (π * rope_length^2 / 4) = 227.07 :=
by
  sorry

end NUMINAMATH_GPT_total_grazing_area_l1001_100189


namespace NUMINAMATH_GPT_geometric_sequence_4th_term_is_2_5_l1001_100190

variables (a r : ℝ) (n : ℕ)

def geometric_term (a r : ℝ) (n : ℕ) : ℝ := a * r^(n-1)

theorem geometric_sequence_4th_term_is_2_5 (a r : ℝ)
  (h1 : a = 125) 
  (h2 : geometric_term a r 8 = 72) :
  geometric_term a r 4 = 5 / 2 := 
sorry

end NUMINAMATH_GPT_geometric_sequence_4th_term_is_2_5_l1001_100190


namespace NUMINAMATH_GPT_cone_volume_l1001_100140

noncomputable def volume_of_cone_from_lateral_surface (radius_semicircle : ℝ) 
  (circumference_base : ℝ := 2 * radius_semicircle * Real.pi) 
  (radius_base : ℝ := circumference_base / (2 * Real.pi)) 
  (height_cone : ℝ := Real.sqrt ((radius_semicircle:ℝ) ^ 2 - (radius_base:ℝ) ^ 2)) : ℝ := 
  (1 / 3) * Real.pi * (radius_base ^ 2) * height_cone

theorem cone_volume (h_semicircle : 2 = 2) : volume_of_cone_from_lateral_surface 2 = (Real.sqrt 3) / 3 * Real.pi := 
by
  -- Importing Real.sqrt and Real.pi to bring them into scope
  sorry

end NUMINAMATH_GPT_cone_volume_l1001_100140


namespace NUMINAMATH_GPT_fraction_increase_each_year_l1001_100181

variable (initial_value : ℝ := 57600)
variable (final_value : ℝ := 72900)
variable (years : ℕ := 2)

theorem fraction_increase_each_year :
  ∃ (f : ℝ), initial_value * (1 + f)^years = final_value ∧ f = 0.125 := by
  sorry

end NUMINAMATH_GPT_fraction_increase_each_year_l1001_100181


namespace NUMINAMATH_GPT_probability_of_at_least_40_cents_l1001_100111

-- Definitions for each type of coin and their individual values in cents.
def penny := 1
def nickel := 5
def dime := 10
def quarter := 25
def half_dollar := 50

-- The total value needed for a successful outcome
def minimum_success_value := 40

-- Total number of possible outcomes from flipping 5 coins independently
def total_outcomes := 2^5

-- Count the successful outcomes that result in at least 40 cents
-- This is a placeholder for the actual successful counting method
noncomputable def successful_outcomes := 18

-- Calculate the probability of successful outcomes
noncomputable def probability := (successful_outcomes : ℚ) / total_outcomes

-- Proof statement to show the probability is 9/16
theorem probability_of_at_least_40_cents : probability = 9 / 16 := 
by
  sorry

end NUMINAMATH_GPT_probability_of_at_least_40_cents_l1001_100111


namespace NUMINAMATH_GPT_raisin_cost_fraction_l1001_100141

theorem raisin_cost_fraction
  (R : ℚ) -- cost of a pound of raisins in dollars
  (cost_of_nuts : ℚ)
  (total_cost_raisins : ℚ)
  (total_cost_nuts : ℚ) :
  cost_of_nuts = 3 * R →
  total_cost_raisins = 5 * R →
  total_cost_nuts = 4 * cost_of_nuts →
  (total_cost_raisins / (total_cost_raisins + total_cost_nuts)) = 5 / 17 :=
by
  sorry

end NUMINAMATH_GPT_raisin_cost_fraction_l1001_100141


namespace NUMINAMATH_GPT_ice_cream_flavors_l1001_100113

theorem ice_cream_flavors (F : ℕ) (h1 : F / 4 + F / 2 + 25 = F) : F = 100 :=
by
  sorry

end NUMINAMATH_GPT_ice_cream_flavors_l1001_100113


namespace NUMINAMATH_GPT_no_linear_term_l1001_100182

theorem no_linear_term (m : ℤ) : (∀ (x : ℤ), (x^2 - x + m) * (x - 8) = x^3 - 9*x^2 - 8*m → (8 + m) = 0) → m = -8 :=
by
  sorry

end NUMINAMATH_GPT_no_linear_term_l1001_100182


namespace NUMINAMATH_GPT_cricket_jumps_to_100m_l1001_100171

theorem cricket_jumps_to_100m (x y : ℕ) (h : 9 * x + 8 * y = 100) : x + y = 12 :=
sorry

end NUMINAMATH_GPT_cricket_jumps_to_100m_l1001_100171


namespace NUMINAMATH_GPT_minimum_k_condition_l1001_100144

def is_acute_triangle (a b c : ℕ) : Prop :=
  a * a + b * b > c * c

def any_subset_with_three_numbers_construct_acute_triangle (s : Finset ℕ) : Prop :=
  ∀ t : Finset ℕ, t.card = 3 → 
    (∃ a b c : ℕ, a ∈ t ∧ b ∈ t ∧ c ∈ t ∧ 
      is_acute_triangle a b c ∨
      is_acute_triangle a c b ∨
      is_acute_triangle b c a)

theorem minimum_k_condition (k : ℕ) :
  (∀ s : Finset ℕ, s.card = k → any_subset_with_three_numbers_construct_acute_triangle s) ↔ (k = 29) :=
  sorry

end NUMINAMATH_GPT_minimum_k_condition_l1001_100144


namespace NUMINAMATH_GPT_min_value_of_quadratic_expression_l1001_100120

theorem min_value_of_quadratic_expression (x y z : ℝ) (h : x + y + z = 1) :
  ∃ (u : ℝ), (2 * x^2 + 3 * y^2 + z^2 = u) ∧ u = 6 / 11 :=
sorry

end NUMINAMATH_GPT_min_value_of_quadratic_expression_l1001_100120


namespace NUMINAMATH_GPT_max_sum_of_squares_eq_7_l1001_100146

theorem max_sum_of_squares_eq_7 :
  ∃ (x y : ℤ), (x^2 + y^2 = 25 ∧ x + y = 7) ∧
  (∀ x' y' : ℤ, (x'^2 + y'^2 = 25 → x' + y' ≤ 7)) := by
sorry

end NUMINAMATH_GPT_max_sum_of_squares_eq_7_l1001_100146


namespace NUMINAMATH_GPT_room_area_ratio_l1001_100153

theorem room_area_ratio (total_squares overlapping_squares : ℕ) 
  (h_total : total_squares = 16) 
  (h_overlap : overlapping_squares = 4) : 
  total_squares / overlapping_squares = 4 := 
by 
  sorry

end NUMINAMATH_GPT_room_area_ratio_l1001_100153


namespace NUMINAMATH_GPT_michael_water_left_l1001_100170

theorem michael_water_left :
  let initial_water := 5
  let given_water := (18 / 7 : ℚ) -- using rational number to represent the fractions
  let remaining_water := initial_water - given_water
  remaining_water = 17 / 7 :=
by
  sorry

end NUMINAMATH_GPT_michael_water_left_l1001_100170


namespace NUMINAMATH_GPT_measure_of_angle_B_scalene_triangle_l1001_100161

theorem measure_of_angle_B_scalene_triangle (A B C : ℝ) (hA_gt_0 : A > 0) (hB_gt_0 : B > 0) (hC_gt_0 : C > 0) 
(h_angles_sum : A + B + C = 180) (hB_eq_2A : B = 2 * A) (hC_eq_3A : C = 3 * A) : B = 60 :=
by
  sorry

end NUMINAMATH_GPT_measure_of_angle_B_scalene_triangle_l1001_100161


namespace NUMINAMATH_GPT_regular_price_adult_ticket_l1001_100179

theorem regular_price_adult_ticket : 
  ∀ (concessions_cost_children cost_adult1 cost_adult2 cost_adult3 cost_adult4 cost_adult5
       ticket_cost_child cost_discount1 cost_discount2 cost_discount3 total_cost : ℝ),
  (concessions_cost_children = 3) → 
  (cost_adult1 = 5) → 
  (cost_adult2 = 6) → 
  (cost_adult3 = 7) → 
  (cost_adult4 = 4) → 
  (cost_adult5 = 9) → 
  (ticket_cost_child = 7) → 
  (cost_discount1 = 3) → 
  (cost_discount2 = 2) → 
  (cost_discount3 = 1) → 
  (total_cost = 139) → 
  (∀ A : ℝ, total_cost = 
    (2 * concessions_cost_children + cost_adult1 + cost_adult2 + cost_adult3 + cost_adult4 + cost_adult5) + 
    (2 * ticket_cost_child + (2 * A + (A - cost_discount1) + (A - cost_discount2) + (A - cost_discount3))) → 
    5 * A - 6 = 88 →
    A = 18.80) :=
by
  intros
  sorry

end NUMINAMATH_GPT_regular_price_adult_ticket_l1001_100179


namespace NUMINAMATH_GPT_tricycles_count_l1001_100133

-- Define the conditions
variable (b t s : ℕ)

def total_children := b + t + s = 10
def total_wheels := 2 * b + 3 * t + 2 * s = 29

-- Provide the theorem to prove
theorem tricycles_count (h1 : total_children b t s) (h2 : total_wheels b t s) : t = 9 := 
by
  sorry

end NUMINAMATH_GPT_tricycles_count_l1001_100133


namespace NUMINAMATH_GPT_fishAddedIs15_l1001_100173

-- Define the number of fish Jason starts with
def initialNumberOfFish : ℕ := 6

-- Define the fish counts on each day
def fishOnDay2 := 2 * initialNumberOfFish
def fishOnDay3 := 2 * fishOnDay2 - (1 / 3 : ℚ) * (2 * fishOnDay2)
def fishOnDay4 := 2 * fishOnDay3
def fishOnDay5 := 2 * fishOnDay4 - (1 / 4 : ℚ) * (2 * fishOnDay4)
def fishOnDay6 := 2 * fishOnDay5
def fishOnDay7 := 2 * fishOnDay6

-- Define the total fish on the seventh day after adding some fish
def totalFishOnDay7 := 207

-- Define the number of fish Jason added on the seventh day
def fishAddedOnDay7 := totalFishOnDay7 - fishOnDay7

-- Prove that the number of fish Jason added on the seventh day is 15
theorem fishAddedIs15 : fishAddedOnDay7 = 15 := sorry

end NUMINAMATH_GPT_fishAddedIs15_l1001_100173


namespace NUMINAMATH_GPT_price_after_9_years_decreases_continuously_l1001_100145

theorem price_after_9_years_decreases_continuously (price_current : ℝ) (price_after_9_years : ℝ) :
  (∀ k : ℕ, k % 3 = 0 → price_current = 8100 → price_after_9_years = 2400) :=
sorry

end NUMINAMATH_GPT_price_after_9_years_decreases_continuously_l1001_100145


namespace NUMINAMATH_GPT_number_of_rows_l1001_100131

theorem number_of_rows (total_pencils : ℕ) (pencils_per_row : ℕ) (h1 : total_pencils = 30) (h2 : pencils_per_row = 5) : total_pencils / pencils_per_row = 6 :=
by
  sorry

end NUMINAMATH_GPT_number_of_rows_l1001_100131


namespace NUMINAMATH_GPT_find_min_value_omega_l1001_100164

noncomputable def min_value_ω (ω : ℝ) : Prop :=
  ∀ (f : ℝ → ℝ), (∀ (x : ℝ), f x = 2 * Real.sin (ω * x)) → ω > 0 →
  (∀ (x : ℝ), -Real.pi / 3 ≤ x ∧ x ≤ Real.pi / 4 → f x ≥ -2) →
  ω = 3 / 2

-- The statement to be proved:
theorem find_min_value_omega : ∃ ω : ℝ, min_value_ω ω :=
by
  use 3 / 2
  sorry

end NUMINAMATH_GPT_find_min_value_omega_l1001_100164


namespace NUMINAMATH_GPT_arithmetic_series_first_term_l1001_100196

theorem arithmetic_series_first_term 
  (a d : ℝ)
  (h1 : 50 * (2 * a + 99 * d) = 1800)
  (h2 : 50 * (2 * a + 199 * d) = 6300) :
  a = -26.55 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_series_first_term_l1001_100196


namespace NUMINAMATH_GPT_mapping_problem_l1001_100163

open Set

noncomputable def f₁ (x : ℝ) : ℝ := Real.sqrt x
noncomputable def f₂ (x : ℝ) : ℝ := 1 / x
def f₃ (x : ℝ) : ℝ := x^2 - 2
def f₄ (x : ℝ) : ℝ := x^2

def A₁ : Set ℝ := {1, 4, 9}
def B₁ : Set ℝ := {-3, -2, -1, 1, 2, 3}
def A₂ : Set ℝ := univ
def B₂ : Set ℝ := univ
def A₃ : Set ℝ := univ
def B₃ : Set ℝ := univ
def A₄ : Set ℝ := {-1, 0, 1}
def B₄ : Set ℝ := {-1, 0, 1}

theorem mapping_problem : 
  ¬ (∀ x ∈ A₁, f₁ x ∈ B₁) ∧
  ¬ (∀ x ∈ A₂, x ≠ 0 → f₂ x ∈ B₂) ∧
  (∀ x ∈ A₃, f₃ x ∈ B₃) ∧
  (∀ x ∈ A₄, f₄ x ∈ B₄) :=
by
  sorry

end NUMINAMATH_GPT_mapping_problem_l1001_100163


namespace NUMINAMATH_GPT_triangle_side_range_l1001_100142

theorem triangle_side_range (x : ℝ) (h1 : x > 0) (h2 : x + (x + 1) + (x + 2) ≤ 12) :
  1 < x ∧ x ≤ 3 :=
by
  sorry

end NUMINAMATH_GPT_triangle_side_range_l1001_100142


namespace NUMINAMATH_GPT_sequence_filling_l1001_100156

theorem sequence_filling :
  ∃ (a : Fin 8 → ℕ), 
    a 0 = 20 ∧ 
    a 7 = 16 ∧ 
    (∀ i : Fin 6, a i + a (i+1) + a (i+2) = 100) ∧ 
    (a 1 = 16) ∧ 
    (a 2 = 64) ∧ 
    (a 3 = 20) ∧ 
    (a 4 = 16) ∧ 
    (a 5 = 64) ∧ 
    (a 6 = 20) := 
by
  sorry

end NUMINAMATH_GPT_sequence_filling_l1001_100156


namespace NUMINAMATH_GPT_flight_cost_l1001_100184

theorem flight_cost (ground_school_cost flight_portion_addition total_cost flight_portion_cost: ℕ) 
  (h₁ : ground_school_cost = 325)
  (h₂ : flight_portion_addition = 625)
  (h₃ : flight_portion_cost = ground_school_cost + flight_portion_addition):
  flight_portion_cost = 950 :=
by
  -- placeholder for proofs
  sorry

end NUMINAMATH_GPT_flight_cost_l1001_100184


namespace NUMINAMATH_GPT_min_value_of_expression_l1001_100116

theorem min_value_of_expression (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b = 1) :
  25 ≤ (4 / a) + (9 / b) :=
sorry

end NUMINAMATH_GPT_min_value_of_expression_l1001_100116


namespace NUMINAMATH_GPT_sum_of_polynomials_l1001_100136

theorem sum_of_polynomials (d : ℕ) :
  let expr1 := 15 * d + 17 + 16 * d ^ 2
  let expr2 := 3 * d + 2
  let sum_expr := expr1 + expr2
  let a := 16
  let b := 18
  let c := 19
  sum_expr = a * d ^ 2 + b * d + c ∧ a + b + c = 53 := by
    sorry

end NUMINAMATH_GPT_sum_of_polynomials_l1001_100136


namespace NUMINAMATH_GPT_incorrect_intersection_point_l1001_100125

def linear_function (x : ℝ) : ℝ := -2 * x + 4

theorem incorrect_intersection_point : ¬(linear_function 0 = 4) :=
by {
  /- Proof can be filled here later -/
  sorry
}

end NUMINAMATH_GPT_incorrect_intersection_point_l1001_100125


namespace NUMINAMATH_GPT_relationship_between_M_n_and_N_n_plus_2_l1001_100194

theorem relationship_between_M_n_and_N_n_plus_2 (n : ℕ) (h : 2 ≤ n) :
  let M_n := (n * (n + 1)) / 2 + 1
  let N_n_plus_2 := n + 3
  M_n < N_n_plus_2 :=
by
  sorry

end NUMINAMATH_GPT_relationship_between_M_n_and_N_n_plus_2_l1001_100194


namespace NUMINAMATH_GPT_natasha_can_achieve_plan_l1001_100168

noncomputable def count_ways : Nat :=
  let num_1x1 := 4
  let num_1x2 := 24
  let target := 2021
  6517

theorem natasha_can_achieve_plan (num_1x1 num_1x2 target : Nat) (h1 : num_1x1 = 4) (h2 : num_1x2 = 24) (h3 : target = 2021) :
  count_ways = 6517 :=
by
  sorry

end NUMINAMATH_GPT_natasha_can_achieve_plan_l1001_100168


namespace NUMINAMATH_GPT_janet_saving_l1001_100135

def tile_cost_difference_saving : ℕ :=
  let turquoise_cost_per_tile := 13
  let purple_cost_per_tile := 11
  let area_wall1 := 5 * 8
  let area_wall2 := 7 * 8
  let total_area := area_wall1 + area_wall2
  let tiles_per_square_foot := 4
  let number_of_tiles := total_area * tiles_per_square_foot
  let cost_difference_per_tile := turquoise_cost_per_tile - purple_cost_per_tile
  number_of_tiles * cost_difference_per_tile

theorem janet_saving : tile_cost_difference_saving = 768 := by
  sorry

end NUMINAMATH_GPT_janet_saving_l1001_100135


namespace NUMINAMATH_GPT_lukas_averages_points_l1001_100187

theorem lukas_averages_points (total_points : ℕ) (num_games : ℕ) (average_points : ℕ)
  (h_total: total_points = 60) (h_games: num_games = 5) : average_points = total_points / num_games :=
sorry

end NUMINAMATH_GPT_lukas_averages_points_l1001_100187


namespace NUMINAMATH_GPT_shaded_to_largest_ratio_l1001_100165

theorem shaded_to_largest_ratio :
  let r1 := 1
  let r2 := 2
  let r3 := 3
  let r4 := 4
  let area (r : ℝ) := π * r^2
  let largest_circle_area := area r4
  let innermost_shaded_area := area r1
  let outermost_shaded_area := area r3 - area r2
  let shaded_area := innermost_shaded_area + outermost_shaded_area
  shaded_area / largest_circle_area = 3 / 8 :=
by
  sorry

end NUMINAMATH_GPT_shaded_to_largest_ratio_l1001_100165


namespace NUMINAMATH_GPT_value_of_p_l1001_100105

theorem value_of_p (m n p : ℝ) (h₁ : m = 8 * n + 5) (h₂ : m + 2 = 8 * (n + p) + 5) : p = 1 / 4 :=
by {
  sorry
}

end NUMINAMATH_GPT_value_of_p_l1001_100105


namespace NUMINAMATH_GPT_find_smaller_number_l1001_100177

theorem find_smaller_number (a b : ℕ) (h_ratio : 11 * a = 7 * b) (h_diff : b = a + 16) : a = 28 :=
by
  sorry

end NUMINAMATH_GPT_find_smaller_number_l1001_100177


namespace NUMINAMATH_GPT_find_f_2012_l1001_100109

noncomputable def f : ℝ → ℝ := sorry

axiom f_condition1 : f 1 = 1 / 4
axiom f_condition2 : ∀ x y : ℝ, 4 * f x * f y = f (x + y) + f (x - y)

theorem find_f_2012 : f 2012 = -1 / 4 := 
sorry

end NUMINAMATH_GPT_find_f_2012_l1001_100109


namespace NUMINAMATH_GPT_incorrect_median_l1001_100159

/-- 
Given:
- A stem-and-leaf plot representation.
- Player B's scores are mainly between 30 and 40 points.
- Player B has 13 scores.
Prove:
The judgment "The median score of player B is 28" is incorrect.
-/
theorem incorrect_median (scores : List ℕ) (H_len : scores.length = 13) (H_range : ∀ x ∈ scores, 30 ≤ x ∧ x ≤ 40) 
  (H_median : ∃ median, median = scores.nthLe 6 sorry ∧ median = 28) : False := 
sorry

end NUMINAMATH_GPT_incorrect_median_l1001_100159


namespace NUMINAMATH_GPT_find_fourth_speed_l1001_100172

theorem find_fourth_speed 
  (avg_speed : ℝ)
  (speed1 speed2 speed3 fourth_speed : ℝ)
  (h_avg_speed : avg_speed = 11.52)
  (h_speed1 : speed1 = 6.0)
  (h_speed2 : speed2 = 12.0)
  (h_speed3 : speed3 = 18.0)
  (expected_avg_speed_eq : avg_speed = 4 / ((1 / speed1) + (1 / speed2) + (1 / speed3) + (1 / fourth_speed))) :
  fourth_speed = 2.095 :=
by 
  sorry

end NUMINAMATH_GPT_find_fourth_speed_l1001_100172


namespace NUMINAMATH_GPT_landlord_packages_l1001_100124

def label_packages_required (start1 end1 start2 end2 start3 end3 : ℕ) : ℕ :=
  let digit_count := 1
  let hundreds_first := (end1 - start1 + 1)
  let hundreds_second := (end2 - start2 + 1)
  let hundreds_third := (end3 - start3 + 1)
  let total_hundreds := hundreds_first + hundreds_second + hundreds_third
  
  let tens_first := ((end1 - start1 + 1) / 10) 
  let tens_second := ((end2 - start2 + 1) / 10) 
  let tens_third := ((end3 - start3 + 1) / 10)
  let total_tens := tens_first + tens_second + tens_third

  let units_per_floor := 5
  let total_units := units_per_floor * 3
  
  let total_ones := total_hundreds + total_tens + total_units
  
  let packages_required := total_ones

  packages_required

theorem landlord_packages : label_packages_required 100 150 200 250 300 350 = 198 := 
  by sorry

end NUMINAMATH_GPT_landlord_packages_l1001_100124


namespace NUMINAMATH_GPT_rationalize_denominator_l1001_100138

theorem rationalize_denominator : (14 / Real.sqrt 14) = Real.sqrt 14 := by
  sorry

end NUMINAMATH_GPT_rationalize_denominator_l1001_100138


namespace NUMINAMATH_GPT_dealer_car_ratio_calculation_l1001_100152

theorem dealer_car_ratio_calculation (X Y : ℝ) 
  (cond1 : 1.4 * X = 1.54 * (X + Y) - 1.6 * Y) :
  let a := 3
  let b := 7
  ((X / Y) = (3 / 7) ∧ (11 * a + 13 * b = 124)) :=
by
  sorry

end NUMINAMATH_GPT_dealer_car_ratio_calculation_l1001_100152


namespace NUMINAMATH_GPT_expression_evaluation_l1001_100162

theorem expression_evaluation : |(-7: ℤ)| / ((2 / 3) - (1 / 5)) - (1 / 2) * ((-4)^2) = 7 := by
  sorry

end NUMINAMATH_GPT_expression_evaluation_l1001_100162


namespace NUMINAMATH_GPT_find_other_root_l1001_100104

theorem find_other_root (x y : ℚ) (h : 48 * x^2 - 77 * x + 21 = 0) (hx : x = 3 / 4) : y = 7 / 12 → 48 * y^2 - 77 * y + 21 = 0 := by
  sorry

end NUMINAMATH_GPT_find_other_root_l1001_100104


namespace NUMINAMATH_GPT_dealer_cash_discount_percentage_l1001_100191

-- Definitions of the given conditions
variable (C : ℝ) (n m : ℕ) (profit_p list_ratio : ℝ)
variable (h_n : n = 25) (h_m : m = 20) (h_profit : profit_p = 1.36) (h_list_ratio : list_ratio = 2)

-- The statement we need to prove
theorem dealer_cash_discount_percentage 
  (h_eff_selling_price : (m : ℝ) / n * C = profit_p * C)
  : ((list_ratio * C - (m / n * C)) / (list_ratio * C) * 100 = 60) :=
by
  sorry

end NUMINAMATH_GPT_dealer_cash_discount_percentage_l1001_100191


namespace NUMINAMATH_GPT_factorized_expression_l1001_100102

variable {a b c : ℝ}

theorem factorized_expression :
  ( ((a^2 - b^2)^3 + (b^2 - c^2)^3 + (c^2 - a^2)^3) / 
    ((a - b)^3 + (b - c)^3 + (c - a)^3) ) 
  = (a + b) * (a + c) * (b + c) := 
  sorry

end NUMINAMATH_GPT_factorized_expression_l1001_100102


namespace NUMINAMATH_GPT_calc_2002_sq_minus_2001_mul_2003_l1001_100112

theorem calc_2002_sq_minus_2001_mul_2003 : 2002 ^ 2 - 2001 * 2003 = 1 := 
by
  sorry

end NUMINAMATH_GPT_calc_2002_sq_minus_2001_mul_2003_l1001_100112


namespace NUMINAMATH_GPT_find_tangent_point_and_slope_l1001_100107

theorem find_tangent_point_and_slope :
  ∃ m n : ℝ, (m = 1 ∧ n = Real.exp 1 ∧ 
    (∀ x y : ℝ, y - n = (Real.exp m) * (x - m) → x = 0 ∧ y = 0) ∧ 
    (Real.exp m = Real.exp 1)) :=
sorry

end NUMINAMATH_GPT_find_tangent_point_and_slope_l1001_100107


namespace NUMINAMATH_GPT_sum_of_digits_of_smallest_divisible_is_6_l1001_100148

noncomputable def smallest_divisible (n : ℕ) : ℕ :=
Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm 1 2) 3) 4) 5) 6) 7

def sum_of_digits (n : ℕ) : ℕ :=
n.digits 10 |>.sum

theorem sum_of_digits_of_smallest_divisible_is_6 : sum_of_digits (smallest_divisible 7) = 6 := 
by
  simp [smallest_divisible, sum_of_digits]
  sorry

end NUMINAMATH_GPT_sum_of_digits_of_smallest_divisible_is_6_l1001_100148


namespace NUMINAMATH_GPT_manny_original_marbles_l1001_100129

/-- 
Let total marbles be 120, and the marbles are divided between Mario, Manny, and Mike in the ratio 4:5:6. 
Let x be the number of marbles Manny is left with after giving some marbles to his brother.
Prove that Manny originally had 40 marbles. 
-/
theorem manny_original_marbles (total_marbles : ℕ) (ratio_mario ratio_manny ratio_mike : ℕ)
    (present_marbles : ℕ) (total_parts : ℕ)
    (h_marbles : total_marbles = 120) 
    (h_ratio : ratio_mario = 4 ∧ ratio_manny = 5 ∧ ratio_mike = 6) 
    (h_total_parts : total_parts = ratio_mario + ratio_manny + ratio_mike)
    (h_manny_parts : total_marbles/total_parts * ratio_manny = 40) : 
  present_marbles = 40 := 
sorry

end NUMINAMATH_GPT_manny_original_marbles_l1001_100129


namespace NUMINAMATH_GPT_smallest_sum_BB_b_l1001_100193

theorem smallest_sum_BB_b (B b : ℕ) (hB : 1 ≤ B ∧ B ≤ 4) (hb : b > 6) (h : 31 * B = 4 * b + 4) : B + b = 8 :=
sorry

end NUMINAMATH_GPT_smallest_sum_BB_b_l1001_100193


namespace NUMINAMATH_GPT_quartic_to_quadratic_l1001_100134

-- Defining the statement of the problem
theorem quartic_to_quadratic (a b c x : ℝ) (y : ℝ) :
  a * x^4 + b * x^3 + c * x^2 + b * x + a = 0 →
  y = x + 1 / x →
  ∃ y1 y2, (a * y^2 + b * y + (c - 2 * a) = 0) ∧
           (x^2 - y1 * x + 1 = 0 ∨ x^2 - y2 * x + 1 = 0) :=
by
  sorry

end NUMINAMATH_GPT_quartic_to_quadratic_l1001_100134


namespace NUMINAMATH_GPT_evaluate_g_of_h_l1001_100185

def g (x : ℝ) : ℝ := 3 * x^2 - 4

def h (x : ℝ) : ℝ := 5 * x^3 + 2

theorem evaluate_g_of_h : g (h (-2)) = 4328 := 
by
  sorry

end NUMINAMATH_GPT_evaluate_g_of_h_l1001_100185


namespace NUMINAMATH_GPT_circle_area_eq_pi_div_4_l1001_100198

theorem circle_area_eq_pi_div_4 :
  ∀ (x y : ℝ), 3*x^2 + 3*y^2 - 9*x + 12*y + 27 = 0 -> (π * (1 / 2)^2 = π / 4) :=
by
  sorry

end NUMINAMATH_GPT_circle_area_eq_pi_div_4_l1001_100198


namespace NUMINAMATH_GPT_mean_of_three_added_numbers_l1001_100150

theorem mean_of_three_added_numbers (x y z : ℝ) :
  (∀ (s : ℝ), (s / 7 = 75) → (s + x + y + z) / 10 = 90) → (x + y + z) / 3 = 125 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_mean_of_three_added_numbers_l1001_100150


namespace NUMINAMATH_GPT_red_balls_count_l1001_100123

theorem red_balls_count (r y b : ℕ) (total_balls : ℕ := 15) (prob_neither_red : ℚ := 2/7) :
    y + b = total_balls - r → (15 - r) * (14 - r) = 60 → r = 5 :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_red_balls_count_l1001_100123


namespace NUMINAMATH_GPT_triangle_side_and_altitude_sum_l1001_100117

theorem triangle_side_and_altitude_sum 
(x y : ℕ) (h1 : x < 75) (h2 : y < 28)
(h3 : x * 60 = 75 * 28) (h4 : 100 * y = 75 * 28) : 
x + y = 56 := 
sorry

end NUMINAMATH_GPT_triangle_side_and_altitude_sum_l1001_100117


namespace NUMINAMATH_GPT_hyperbola_range_of_m_l1001_100192

theorem hyperbola_range_of_m (m : ℝ) :
  (∃ x y : ℝ, (x^2) / (1 + m) + (y^2) / (1 - m) = 1) → 
  (m < -1 ∨ m > 1) :=
by 
sorry

end NUMINAMATH_GPT_hyperbola_range_of_m_l1001_100192


namespace NUMINAMATH_GPT_unique_arrangement_l1001_100126

def valid_grid (arrangement : Matrix (Fin 4) (Fin 4) Char) : Prop :=
  (∀ i : Fin 4, (∃ j1 j2 j3 : Fin 4,
    j1 ≠ j2 ∧ j2 ≠ j3 ∧ j1 ≠ j3 ∧
    arrangement i j1 = 'A' ∧
    arrangement i j2 = 'B' ∧
    arrangement i j3 = 'C')) ∧
  (∀ j : Fin 4, (∃ i1 i2 i3 : Fin 4,
    i1 ≠ i2 ∧ i2 ≠ i3 ∧ i1 ≠ i3 ∧
    arrangement i1 j = 'A' ∧
    arrangement i2 j = 'B' ∧
    arrangement i3 j = 'C')) ∧
  (∃ i1 i2 i3 : Fin 4,
    i1 ≠ i2 ∧ i2 ≠ i3 ∧ i1 ≠ i3 ∧
    arrangement i1 i1 = 'A' ∧
    arrangement i2 i2 = 'B' ∧
    arrangement i3 i3 = 'C') ∧
  (∃ i1 i2 i3 : Fin 4,
    i1 ≠ i2 ∧ i2 ≠ i3 ∧ i1 ≠ i3 ∧
    arrangement i1 (Fin.mk (3 - i1.val) sorry) = 'A' ∧
    arrangement i2 (Fin.mk (3 - i2.val) sorry) = 'B' ∧
    arrangement i3 (Fin.mk (3 - i3.val) sorry) = 'C')

def fixed_upper_left (arrangement : Matrix (Fin 4) (Fin 4) Char) : Prop :=
  arrangement 0 0 = 'A'

theorem unique_arrangement : ∃! arrangement : Matrix (Fin 4) (Fin 4) Char,
  valid_grid arrangement ∧ fixed_upper_left arrangement :=
sorry

end NUMINAMATH_GPT_unique_arrangement_l1001_100126


namespace NUMINAMATH_GPT_hammers_in_comparison_group_l1001_100158

theorem hammers_in_comparison_group (H W x : ℝ) (h1 : 2 * H + 2 * W = 1 / 3 * (x * H + 5 * W)) (h2 : W = 2 * H) :
  x = 8 :=
sorry

end NUMINAMATH_GPT_hammers_in_comparison_group_l1001_100158


namespace NUMINAMATH_GPT_proof_problem_l1001_100100

variables {R : Type*} [CommRing R]

-- f is a function from ℝ to ℝ
variable (f : ℝ → ℝ)

-- Definition of an odd function
def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- Variable definitions for the conditions
variables (h_odd : is_odd f)
(h_f1 : f 1 = 1)
(h_period : ∀ x, f (x + 6) = f x + f 3)

-- The proof problem statement
theorem proof_problem : f 2015 + f 2016 = -1 :=
by
  sorry

end NUMINAMATH_GPT_proof_problem_l1001_100100


namespace NUMINAMATH_GPT_greatest_num_consecutive_integers_sum_eq_36_l1001_100121

theorem greatest_num_consecutive_integers_sum_eq_36 :
    ∃ a : ℤ, ∃ N : ℕ, N > 0 ∧ (N = 9) ∧ (N * (2 * a + N - 1) = 72) :=
sorry

end NUMINAMATH_GPT_greatest_num_consecutive_integers_sum_eq_36_l1001_100121


namespace NUMINAMATH_GPT_fence_pole_count_l1001_100103

-- Define the conditions
def path_length : ℕ := 900
def bridge_length : ℕ := 42
def pole_spacing : ℕ := 6

-- Define the goal
def total_poles : ℕ := 286

-- The statement to prove
theorem fence_pole_count :
  let total_length_to_fence := (path_length - bridge_length)
  let poles_per_side := total_length_to_fence / pole_spacing
  let total_poles_needed := 2 * poles_per_side
  total_poles_needed = total_poles :=
by
  sorry

end NUMINAMATH_GPT_fence_pole_count_l1001_100103
