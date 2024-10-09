import Mathlib

namespace farm_horses_cows_difference_l1599_159933

-- Definitions based on provided conditions
def initial_ratio_horses_to_cows (horses cows : ℕ) : Prop := 5 * cows = horses
def transaction (horses cows sold bought : ℕ) : Prop :=
  horses - sold = 5 * cows - 15 ∧ cows + bought = cows + 15

-- Definitions to represent the ratios
def pre_transaction_ratio (horses cows : ℕ) : Prop := initial_ratio_horses_to_cows horses cows
def post_transaction_ratio (horses cows : ℕ) (sold bought : ℕ) : Prop :=
  transaction horses cows sold bought ∧ 7 * (horses - sold) = 17 * (cows + bought)

-- Statement of the theorem
theorem farm_horses_cows_difference :
  ∀ (horses cows : ℕ), 
    pre_transaction_ratio horses cows → 
    post_transaction_ratio horses cows 15 15 →
    (horses - 15) - (cows + 15) = 50 :=
by
  intros horses cows pre_ratio post_ratio
  sorry

end farm_horses_cows_difference_l1599_159933


namespace train_cross_time_l1599_159959

theorem train_cross_time (length_train : ℝ) (length_bridge : ℝ) (speed_kmph : ℝ) : 
  length_train = 100 →
  length_bridge = 150 →
  speed_kmph = 63 →
  (length_train + length_bridge) / (speed_kmph * (1000 / 3600)) = 14.29 :=
by
  sorry

end train_cross_time_l1599_159959


namespace trekking_adults_l1599_159906

theorem trekking_adults
  (A : ℕ)
  (C : ℕ)
  (meal_for_adults : ℕ)
  (meal_for_children : ℕ)
  (remaining_food_children : ℕ) :
  C = 70 →
  meal_for_adults = 70 →
  meal_for_children = 90 →
  remaining_food_children = 72 →
  A - 14 = (meal_for_adults - 14) →
  A = 56 :=
sorry

end trekking_adults_l1599_159906


namespace trees_not_pine_trees_l1599_159945

theorem trees_not_pine_trees
  (total_trees : ℕ)
  (percentage_pine : ℝ)
  (number_pine : ℕ)
  (number_not_pine : ℕ)
  (h_total : total_trees = 350)
  (h_percentage : percentage_pine = 0.70)
  (h_pine : number_pine = percentage_pine * total_trees)
  (h_not_pine : number_not_pine = total_trees - number_pine)
  : number_not_pine = 105 :=
sorry

end trees_not_pine_trees_l1599_159945


namespace f_f_minus_two_l1599_159958

def f (x : ℚ) : ℚ := x⁻¹ + (x⁻¹ / (1 + x⁻¹))

theorem f_f_minus_two : f (f (-2)) = -8 / 3 := by
  sorry

end f_f_minus_two_l1599_159958


namespace chantel_final_bracelets_l1599_159981

-- Definitions of the conditions in Lean
def initial_bracelets_7_days := 7 * 4
def after_school_giveaway := initial_bracelets_7_days - 8
def bracelets_10_days := 10 * 5
def total_after_10_days := after_school_giveaway + bracelets_10_days
def after_soccer_giveaway := total_after_10_days - 12
def crafting_club_bracelets := 4 * 6
def total_after_crafting_club := after_soccer_giveaway + crafting_club_bracelets
def weekend_trip_bracelets := 2 * 3
def total_after_weekend_trip := total_after_crafting_club + weekend_trip_bracelets
def final_total := total_after_weekend_trip - 10

-- Lean statement to prove the final total bracelets
theorem chantel_final_bracelets : final_total = 78 :=
by
  -- Note: The proof is not required, hence the sorry
  sorry

end chantel_final_bracelets_l1599_159981


namespace area_of_rectangle_l1599_159956

-- Define the given conditions
def length : Real := 5.9
def width : Real := 3
def expected_area : Real := 17.7

theorem area_of_rectangle : (length * width) = expected_area := 
by 
  sorry

end area_of_rectangle_l1599_159956


namespace sum_of_squares_of_b_l1599_159910

-- Define the constants
def b1 := 35 / 64
def b2 := 0
def b3 := 21 / 64
def b4 := 0
def b5 := 7 / 64
def b6 := 0
def b7 := 1 / 64

-- The goal is to prove the sum of squares of these constants
theorem sum_of_squares_of_b : 
  (b1 ^ 2 + b2 ^ 2 + b3 ^ 2 + b4 ^ 2 + b5 ^ 2 + b6 ^ 2 + b7 ^ 2) = 429 / 1024 :=
  by
    -- defer the proof
    sorry

end sum_of_squares_of_b_l1599_159910


namespace cistern_filling_time_l1599_159927

/-
Given the following conditions:
- Pipe A fills the cistern in 10 hours.
- Pipe B fills the cistern in 12 hours.
- Exhaust pipe C drains the cistern in 15 hours.
- Exhaust pipe D drains the cistern in 20 hours.

Prove that if all four pipes are opened simultaneously, the cistern will be filled in 15 hours.
-/

theorem cistern_filling_time :
  let rate_A := 1 / 10
  let rate_B := 1 / 12
  let rate_C := -(1 / 15)
  let rate_D := -(1 / 20)
  let combined_rate := rate_A + rate_B + rate_C + rate_D
  let time_to_fill := 1 / combined_rate
  time_to_fill = 15 :=
by 
  sorry

end cistern_filling_time_l1599_159927


namespace polynomial_lt_factorial_l1599_159970

theorem polynomial_lt_factorial (A B C : ℝ) : ∃N : ℕ, ∀n : ℕ, n > N → An^2 + Bn + C < n! := 
by
  sorry

end polynomial_lt_factorial_l1599_159970


namespace find_c_l1599_159952

theorem find_c (a b c : ℚ) (h1 : ∀ y : ℚ, 1 = a * (3 - 1)^2 + b * (3 - 1) + c) (h2 : ∀ y : ℚ, 4 = a * (1)^2 + b * (1) + c)
  (h3 : ∀ y : ℚ, 1 = a * (y - 1)^2 + 4) : c = 13 / 4 :=
by
  sorry

end find_c_l1599_159952


namespace pears_for_apples_l1599_159940

-- Define the costs of apples, oranges, and pears.
variables {cost_apples cost_oranges cost_pears : ℕ}

-- Condition 1: Ten apples cost the same as five oranges
axiom apples_equiv_oranges : 10 * cost_apples = 5 * cost_oranges

-- Condition 2: Three oranges cost the same as four pears
axiom oranges_equiv_pears : 3 * cost_oranges = 4 * cost_pears

-- Theorem: Tyler can buy 13 pears for the price of 20 apples
theorem pears_for_apples : 20 * cost_apples = 13 * cost_pears :=
sorry

end pears_for_apples_l1599_159940


namespace nested_fraction_evaluation_l1599_159997

def nested_expression := 1 / (3 - 1 / (3 - 1 / (3 - 1 / 3)))

theorem nested_fraction_evaluation : nested_expression = 8 / 21 := by
  sorry

end nested_fraction_evaluation_l1599_159997


namespace minimum_box_value_l1599_159932

def is_valid_pair (a b : ℤ) : Prop :=
  a * b = 15 ∧ (a^2 + b^2 ≥ 34)

theorem minimum_box_value :
  ∃ (a b : ℤ), is_valid_pair a b ∧ (∀ (a' b' : ℤ), is_valid_pair a' b' → a^2 + b^2 ≤ a'^2 + b'^2) ∧ a^2 + b^2 = 34 :=
by
  sorry

end minimum_box_value_l1599_159932


namespace contradiction_proof_l1599_159904

theorem contradiction_proof (a b c : ℝ) (h : (a⁻¹ * b⁻¹ * c⁻¹) > 0) : (a ≤ 1) ∧ (b ≤ 1) ∧ (c ≤ 1) → False :=
sorry

end contradiction_proof_l1599_159904


namespace remainder_mod_7_l1599_159990

theorem remainder_mod_7 (n m p : ℕ) 
  (h₁ : n % 4 = 3)
  (h₂ : m % 7 = 5)
  (h₃ : p % 5 = 2) :
  (7 * n + 3 * m - p) % 7 = 6 :=
by
  sorry

end remainder_mod_7_l1599_159990


namespace solution_set_of_inequality_l1599_159937

theorem solution_set_of_inequality :
  { x : ℝ | x^2 + x - 2 ≥ 0 } = { x : ℝ | x ≤ -2 ∨ 1 ≤ x } :=
by
  sorry

end solution_set_of_inequality_l1599_159937


namespace count_numbers_with_digit_2_l1599_159954

def contains_digit_2 (n : Nat) : Prop :=
  n / 100 = 2 ∨ (n / 10 % 10) = 2 ∨ (n % 10) = 2

theorem count_numbers_with_digit_2 (N : Nat) (H : 200 ≤ N ∧ N ≤ 499) : 
  Nat.card {n // 200 ≤ n ∧ n ≤ 499 ∧ contains_digit_2 n} = 138 :=
by
  sorry

end count_numbers_with_digit_2_l1599_159954


namespace John_to_floor_pushups_l1599_159991

theorem John_to_floor_pushups:
  let days_per_week := 5
  let reps_per_day := 1
  let total_reps_per_stage := 15
  let stages := 3 -- number of stages: wall, high elevation, low elevation
  let total_days_needed := stages * total_reps_per_stage
  let total_weeks_needed := total_days_needed / days_per_week
  total_weeks_needed = 9 := by
  -- Here we will define the specifics of the proof later.
  sorry

end John_to_floor_pushups_l1599_159991


namespace park_area_l1599_159916

-- Define the width (w) and length (l) of the park
def width : Float := 11.25
def length : Float := 33.75

-- Define the perimeter and area functions
def perimeter (w l : Float) : Float := 2 * (w + l)
def area (w l : Float) : Float := w * l

-- Provide the conditions
axiom width_is_one_third_length : width = length / 3
axiom perimeter_is_90 : perimeter width length = 90

-- Theorem to prove the area given the conditions
theorem park_area : area width length = 379.6875 := by
  sorry

end park_area_l1599_159916


namespace incorrect_statement_l1599_159946

open Set

theorem incorrect_statement 
  (M : Set ℝ := {x : ℝ | 0 < x ∧ x < 1})
  (N : Set ℝ := {y : ℝ | 0 < y})
  (R : Set ℝ := univ) : M ∪ N ≠ R :=
by
  sorry

end incorrect_statement_l1599_159946


namespace ways_to_write_2020_as_sum_of_twos_and_threes_l1599_159928

def write_as_sum_of_twos_and_threes (n : ℕ) : ℕ :=
  if n % 2 = 0 then (n / 2 + 1) else 0

theorem ways_to_write_2020_as_sum_of_twos_and_threes :
  write_as_sum_of_twos_and_threes 2020 = 337 :=
sorry

end ways_to_write_2020_as_sum_of_twos_and_threes_l1599_159928


namespace find_n_l1599_159931

noncomputable def first_term_1 : ℝ := 12
noncomputable def second_term_1 : ℝ := 4
noncomputable def sum_first_series : ℝ := 18

noncomputable def first_term_2 : ℝ := 12
noncomputable def second_term_2 (n : ℝ) : ℝ := 4 + 2 * n
noncomputable def sum_second_series : ℝ := 90

theorem find_n (n : ℝ) : 
  (first_term_1 = 12) → 
  (second_term_1 = 4) → 
  (sum_first_series = 18) →
  (first_term_2 = 12) →
  (second_term_2 n = 4 + 2 * n) →
  (sum_second_series = 90) →
  (sum_second_series = 5 * sum_first_series) →
  n = 6 :=
by
  intros _ _ _ _ _ _ _
  sorry

end find_n_l1599_159931


namespace cos_pi_over_8_cos_5pi_over_8_l1599_159930

theorem cos_pi_over_8_cos_5pi_over_8 :
  (Real.cos (Real.pi / 8)) * (Real.cos (5 * Real.pi / 8)) = - (Real.sqrt 2 / 4) :=
by
  sorry

end cos_pi_over_8_cos_5pi_over_8_l1599_159930


namespace spherical_to_rectangular_correct_l1599_159993

noncomputable def spherical_to_rectangular (ρ θ φ : ℝ) : ℝ × ℝ × ℝ :=
  (ρ * Real.sin φ * Real.cos θ, ρ * Real.sin φ * Real.sin θ, ρ * Real.cos φ)

theorem spherical_to_rectangular_correct :
  spherical_to_rectangular 4 (Real.pi / 6) (Real.pi / 3) = (3, Real.sqrt 3, 2) := by
  sorry

end spherical_to_rectangular_correct_l1599_159993


namespace sum_of_three_consecutive_integers_l1599_159955

theorem sum_of_three_consecutive_integers (a b c : ℤ) (h1 : a + 1 = b) (h2 : b + 1 = c) (h3 : c = 7) : a + b + c = 18 :=
sorry

end sum_of_three_consecutive_integers_l1599_159955


namespace min_value_of_fraction_l1599_159985

theorem min_value_of_fraction (m n : ℝ) (h1 : 0 < m) (h2 : 0 < n) 
    (h3 : (m * (-3) + n * (-1) + 2 = 0)) 
    (h4 : (m * (-2) + n * 0 + 2 = 0)) : 
    (1 / m + 3 / n) = 6 :=
by
  sorry

end min_value_of_fraction_l1599_159985


namespace second_chapter_pages_l1599_159977

theorem second_chapter_pages (x : ℕ) (h1 : 48 = x + 37) : x = 11 := 
sorry

end second_chapter_pages_l1599_159977


namespace exists_five_consecutive_divisible_by_2014_l1599_159978

theorem exists_five_consecutive_divisible_by_2014 :
  ∃ (a b c d e : ℕ), 53 = a ∧ 54 = b ∧ 55 = c ∧ 56 = d ∧ 57 = e ∧ 100 > a ∧ a + 1 = b ∧ b + 1 = c ∧ c + 1 = d ∧ d + 1 = e ∧ 2014 ∣ (a * b * c * d * e) :=
by 
  sorry

end exists_five_consecutive_divisible_by_2014_l1599_159978


namespace satisfies_equation_l1599_159917

noncomputable def y (x : ℝ) : ℝ := -Real.sqrt (x^4 - x^2)
noncomputable def dy (x : ℝ) : ℝ := x * (1 - 2 * x^2) / Real.sqrt (x^4 - x^2)

theorem satisfies_equation (x : ℝ) (hx : x ≠ 0) : x * y x * dy x - (y x)^2 = x^4 := 
sorry

end satisfies_equation_l1599_159917


namespace correct_calculation_result_l1599_159971

-- Define the conditions in Lean
variable (num : ℤ) (mistake_mult : ℤ) (result : ℤ)
variable (h_mistake : mistake_mult = num * 10) (h_result : result = 50)

-- The statement we want to prove
theorem correct_calculation_result 
  (h_mistake : mistake_mult = num * 10) 
  (h_result : result = 50) 
  (h_num_correct : num = result / 10) :
  (20 / num = 4) := sorry

end correct_calculation_result_l1599_159971


namespace vertical_asymptote_l1599_159921

noncomputable def y (x : ℝ) : ℝ := (3 * x + 1) / (7 * x - 10)

theorem vertical_asymptote (x : ℝ) : (7 * x - 10 = 0) → (x = 10 / 7) :=
by
  intro h
  linarith [h]

#check vertical_asymptote

end vertical_asymptote_l1599_159921


namespace positive_integer_sum_l1599_159913

theorem positive_integer_sum (n : ℤ) :
  (n > 0) ∧
  (∀ stamps_cannot_form : ℤ, (∀ a b c : ℤ, 7 * a + n * b + (n + 2) * c ≠ 120) ↔
  ¬ (0 ≤ 7*a ∧ 7*a ≤ 120 ∧ 0 ≤ n*b ∧ n*b ≤ 120 ∧ 0 ≤ (n + 2)*c ∧ (n + 2)*c ≤ 120)) ∧
  (∀ postage_formed : ℤ, (120 < postage_formed ∧ postage_formed ≤ 125 →
  ∃ a b c : ℤ, 7 * a + n * b + (n + 2) * c = postage_formed)) →
  n = 21 :=
by {
  -- proof omittted 
  sorry
}

end positive_integer_sum_l1599_159913


namespace polynomial_value_at_one_l1599_159964

theorem polynomial_value_at_one
  (a b c : ℝ)
  (h1 : -a - b - c + 1 = 6)
  : a + b + c + 1 = -4 :=
by {
  sorry
}

end polynomial_value_at_one_l1599_159964


namespace corresponding_side_of_larger_triangle_l1599_159920

-- Conditions
variables (A1 A2 : ℕ) (s1 s2 : ℕ)
-- A1 is the area of the larger triangle
-- A2 is the area of the smaller triangle
-- s1 is a side of the smaller triangle = 4 feet
-- s2 is the corresponding side of the larger triangle

-- Given conditions as hypotheses
axiom diff_in_areas : A1 - A2 = 32
axiom ratio_of_areas : A1 = 9 * A2
axiom side_of_smaller_triangle : s1 = 4

-- Theorem to prove the corresponding side of the larger triangle
theorem corresponding_side_of_larger_triangle 
  (h1 : A1 - A2 = 32)
  (h2 : A1 = 9 * A2)
  (h3 : s1 = 4) : 
  s2 = 12 :=
sorry

end corresponding_side_of_larger_triangle_l1599_159920


namespace minimum_omega_l1599_159914

noncomputable section

-- Define the function f and its properties
def f (x : ℝ) (ω : ℝ) (φ : ℝ) : ℝ := Real.cos (ω * x + φ)

-- Assumptions based on the given conditions
variables {ω φ : ℝ}
variables (T : ℝ) (hω_pos : 0 < ω) (hφ_range : 0 < φ ∧ φ < π)
variables (hT : f T ω φ = Real.sqrt 3 / 2)
variables (hx_zero : f (π / 9) ω φ = 0)

-- Prove the minimum value of ω is 3
theorem minimum_omega : ω = 3 := sorry

end minimum_omega_l1599_159914


namespace part_a_part_b_l1599_159934

theorem part_a (a b c d : ℕ) (h : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) : 
  ∃ (a b c d : ℕ), 1 / (a : ℝ) + 1 / (b : ℝ) = 1 / (c : ℝ) + 1 / (d : ℝ) := sorry

theorem part_b (a b c d e : ℕ) (h : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e) : 
  ∃ (a b c d e : ℕ), 1 / (a : ℝ) + 1 / (b : ℝ) = 1 / (c : ℝ) + 1 / (d : ℝ) + 1 / (e : ℝ) := sorry

end part_a_part_b_l1599_159934


namespace violates_properties_l1599_159999

-- Definitions from conditions
variables {a b c m : ℝ}

-- Conclusion to prove
theorem violates_properties :
  (∀ c : ℝ, ac = bc → (c ≠ 0 → a = b)) ∧ (c = 0 → ac = bc) → False :=
sorry

end violates_properties_l1599_159999


namespace sum_of_three_largest_l1599_159961

theorem sum_of_three_largest (n : ℕ) 
  (h1 : n + (n + 1) + (n + 2) = 60) : 
  (n + 2) + (n + 3) + (n + 4) = 66 :=
by
  sorry

end sum_of_three_largest_l1599_159961


namespace determine_B_l1599_159957

open Set

-- Define the universal set U and the sets A and B
variable (U A B : Set ℕ)

-- Definitions based on the problem conditions
def U_def : U = A ∪ B := 
  by sorry

def cond1 : (U = {1, 2, 3, 4, 5, 6, 7}) := 
  by sorry

def cond2 : (A ∩ (U \ B) = {2, 4, 6}) := 
  by sorry

-- The main statement
theorem determine_B (h1 : U = {1, 2, 3, 4, 5, 6, 7}) (h2 : A ∩ (U \ B) = {2, 4, 6}) : B = {1, 3, 5, 7} :=
  by sorry

end determine_B_l1599_159957


namespace find_divisor_l1599_159960

theorem find_divisor :
  ∃ d : ℕ, (d = 859560) ∧ ∃ n : ℕ, (n + 859622) % d = 0 ∧ n = 859560 :=
by
  sorry

end find_divisor_l1599_159960


namespace number_of_children_l1599_159902

theorem number_of_children (crayons_per_child total_crayons : ℕ) (h1 : crayons_per_child = 12) (h2 : total_crayons = 216) : total_crayons / crayons_per_child = 18 :=
by
  have h3 : total_crayons / crayons_per_child = 216 / 12 := by rw [h1, h2]
  norm_num at h3
  exact h3

end number_of_children_l1599_159902


namespace midpoint_between_points_l1599_159918

theorem midpoint_between_points : 
  let (x1, y1, z1) := (2, -3, 5)
  let (x2, y2, z2) := (8, 1, 3)
  (1 / 2 * (x1 + x2), 1 / 2 * (y1 + y2), 1 / 2 * (z1 + z2)) = (5, -1, 4) :=
by
  sorry

end midpoint_between_points_l1599_159918


namespace smallest_six_factors_l1599_159982

theorem smallest_six_factors (n : ℕ) (h : (n = 2 * 3^2)) : n = 18 :=
by {
    sorry -- proof goes here
}

end smallest_six_factors_l1599_159982


namespace mouse_jump_distance_l1599_159963

theorem mouse_jump_distance
  (g : ℕ) 
  (f : ℕ) 
  (m : ℕ)
  (h1 : g = 25)
  (h2 : f = g + 32)
  (h3 : m = f - 26) : 
  m = 31 :=
by
  sorry

end mouse_jump_distance_l1599_159963


namespace bus_ride_difference_l1599_159994

theorem bus_ride_difference :
  ∀ (Oscar_bus Charlie_bus : ℝ),
  Oscar_bus = 0.75 → Charlie_bus = 0.25 → Oscar_bus - Charlie_bus = 0.50 :=
by
  intros Oscar_bus Charlie_bus hOscar hCharlie
  rw [hOscar, hCharlie]
  norm_num

end bus_ride_difference_l1599_159994


namespace quadratic_roots_opposite_l1599_159943

theorem quadratic_roots_opposite (a : ℝ) (h : ∀ x1 x2 : ℝ, 
  (x1 + x2 = 0 ∧ x1 * x2 = a - 1) ∧
  (x1 - (-(x1)) = 0 ∧ x2 - x1 = 0)) :
  a = 0 :=
sorry

end quadratic_roots_opposite_l1599_159943


namespace hcf_of_two_numbers_of_given_conditions_l1599_159974

theorem hcf_of_two_numbers_of_given_conditions :
  ∃ B H, (588 = H * 84) ∧ H = Nat.gcd 588 B ∧ H = 7 :=
by
  use 84, 7
  have h₁ : 588 = 7 * 84 := by sorry
  have h₂ : 7 = Nat.gcd 588 84 := by sorry
  exact ⟨h₁, h₂, rfl⟩

end hcf_of_two_numbers_of_given_conditions_l1599_159974


namespace no_solution_in_positive_rationals_l1599_159979

theorem no_solution_in_positive_rationals (n : ℕ) (hn : n > 0) (x y : ℚ) (hx : x > 0) (hy : y > 0) :
  x + y + (1 / x) + (1 / y) ≠ 3 * n :=
sorry

end no_solution_in_positive_rationals_l1599_159979


namespace area_of_field_l1599_159909

-- Define the variables and conditions
variables {L W : ℝ}

-- Given conditions
def length_side (L : ℝ) : Prop := L = 30
def fencing_equation (L W : ℝ) : Prop := L + 2 * W = 70

-- Prove the area of the field is 600 square feet
theorem area_of_field : length_side L → fencing_equation L W → (L * W = 600) :=
by
  intros hL hF
  rw [length_side, fencing_equation] at *
  sorry

end area_of_field_l1599_159909


namespace no_integer_m_l1599_159935

theorem no_integer_m (n r m : ℕ) (hn : 1 ≤ n) (hr : 2 ≤ r) : 
  ¬ (∃ m : ℕ, n * (n + 1) * (n + 2) = m ^ r) :=
sorry

end no_integer_m_l1599_159935


namespace sqrt_D_rational_sometimes_not_l1599_159988

-- Definitions and conditions
def D (a : ℤ) : ℤ := a^2 + (a + 2)^2 + (a * (a + 2))^2

-- The statement to prove
theorem sqrt_D_rational_sometimes_not (a : ℤ) : ∃ x : ℚ, x = Real.sqrt (D a) ∧ ¬(∃ y : ℤ, x = y) ∨ ∃ y : ℤ, Real.sqrt (D a) = y :=
by 
  sorry

end sqrt_D_rational_sometimes_not_l1599_159988


namespace proof_problem_l1599_159939

noncomputable def otimes (a b : ℝ) : ℝ := a^3 / b^2

theorem proof_problem : ((otimes (otimes 2 3) 4) - otimes 2 (otimes 3 4)) = -224/81 :=
by
  sorry

end proof_problem_l1599_159939


namespace find_number_l1599_159915

theorem find_number (n : ℕ) (h : n + 19 = 47) : n = 28 :=
by {
    sorry
}

end find_number_l1599_159915


namespace point_outside_circle_l1599_159929

theorem point_outside_circle (a b : ℝ) (h : ∃ x y : ℝ, a * x + b * y = 1 ∧ x^2 + y^2 = 1) : a^2 + b^2 > 1 :=
sorry

end point_outside_circle_l1599_159929


namespace students_no_A_l1599_159908

theorem students_no_A
  (total_students : ℕ)
  (A_in_English : ℕ)
  (A_in_math : ℕ)
  (A_in_both : ℕ)
  (total_students_eq : total_students = 40)
  (A_in_English_eq : A_in_English = 10)
  (A_in_math_eq : A_in_math = 18)
  (A_in_both_eq : A_in_both = 6) :
  total_students - ((A_in_English + A_in_math) - A_in_both) = 18 :=
by
  sorry

end students_no_A_l1599_159908


namespace find_k_values_l1599_159923

theorem find_k_values (a b k : ℕ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : b % a = 0) 
  (h₄ : ∀ (m : ℤ), (a : ℤ) = k * (a : ℤ) + m ∧ (8 * (b : ℤ)) = k * (b : ℤ) + m) :
  k = 9 ∨ k = 15 :=
by
  { sorry }

end find_k_values_l1599_159923


namespace common_divisors_9240_13860_l1599_159900

def num_divisors (n : ℕ) : ℕ :=
  -- function to calculate the number of divisors (implementation is not provided here)
  sorry

theorem common_divisors_9240_13860 :
  let d := Nat.gcd 9240 13860
  d = 924 → num_divisors d = 24 := by
  intros d gcd_eq
  rw [gcd_eq]
  sorry

end common_divisors_9240_13860_l1599_159900


namespace ice_rink_rental_fee_l1599_159951

/-!
  # Problem:
  An ice skating rink charges $5 for admission and a certain amount to rent skates. 
  Jill can purchase a new pair of skates for $65. She would need to go to the rink 26 times 
  to justify buying the skates rather than renting a pair. How much does the rink charge to rent skates?
-/

/-- Lean statement of the problem. --/
theorem ice_rink_rental_fee 
  (admission_fee : ℝ) (skates_cost : ℝ) (num_visits : ℕ)
  (total_buying_cost : ℝ) (total_renting_cost : ℝ)
  (rental_fee : ℝ) :
  admission_fee = 5 ∧
  skates_cost = 65 ∧
  num_visits = 26 ∧
  total_buying_cost = skates_cost + (admission_fee * num_visits) ∧
  total_renting_cost = (admission_fee + rental_fee) * num_visits ∧
  total_buying_cost = total_renting_cost →
  rental_fee = 2.50 :=
by
  intros h
  sorry

end ice_rink_rental_fee_l1599_159951


namespace mean_proportional_c_l1599_159980

theorem mean_proportional_c (a b c : ℝ) (h1 : a = 3) (h2 : b = 27) (h3 : c^2 = a * b) : c = 9 := by
  sorry

end mean_proportional_c_l1599_159980


namespace all_propositions_correct_l1599_159948

variable {α : Type*} [AddCommGroup α] [Module ℝ α]

theorem all_propositions_correct (m n : ℝ) (a b : α) (h1 : m ≠ 0) (h2 : a ≠ 0) : 
  (∀ (m : ℝ) (a b : α), m • (a - b) = m • a - m • b) ∧
  (∀ (m n : ℝ) (a : α), (m - n) • a = m • a - n • a) ∧
  (∀ (m : ℝ) (a b : α), m • a = m • b → a = b) ∧
  (∀ (m n : ℝ) (a : α), m • a = n • a → m = n) :=
by {
  sorry
}

end all_propositions_correct_l1599_159948


namespace tangent_line_equation_l1599_159922

noncomputable def f (x : ℝ) : ℝ := Real.log (x + 1)

theorem tangent_line_equation :
  let x1 : ℝ := 1
  let y1 : ℝ := f 1
  ∀ x y : ℝ, 
    (y - y1 = (1 / (x1 + 1)) * (x - x1)) ↔ 
    (x - 2 * y + 2 * Real.log 2 - 1 = 0) :=
by
  sorry

end tangent_line_equation_l1599_159922


namespace value_of_a5_l1599_159941

variable (a : ℕ → ℕ)

-- The initial condition
axiom initial_condition : a 1 = 2

-- The recurrence relation
axiom recurrence_relation : ∀ n : ℕ, n ≠ 0 → n * a (n+1) = 2 * (n + 1) * a n

theorem value_of_a5 : a 5 = 160 := 
sorry

end value_of_a5_l1599_159941


namespace ring_binder_price_l1599_159969

theorem ring_binder_price (x : ℝ) (h1 : 50 + 5 = 55) (h2 : ∀ x, 55 + 3 * (x - 2) = 109) :
  x = 20 :=
by
  sorry

end ring_binder_price_l1599_159969


namespace movie_theater_ticket_cost_l1599_159925

theorem movie_theater_ticket_cost
  (adult_ticket_cost : ℝ)
  (child_ticket_cost : ℝ)
  (total_moviegoers : ℝ)
  (total_amount_paid : ℝ)
  (number_of_adults : ℝ)
  (H_child_ticket_cost : child_ticket_cost = 6.50)
  (H_total_moviegoers : total_moviegoers = 7)
  (H_total_amount_paid : total_amount_paid = 54.50)
  (H_number_of_adults : number_of_adults = 3)
  (H_number_of_children : total_moviegoers - number_of_adults = 4) :
  adult_ticket_cost = 9.50 :=
sorry

end movie_theater_ticket_cost_l1599_159925


namespace fg_equals_gf_l1599_159953

theorem fg_equals_gf (m n p q : ℝ) (h : m + q = n + p) : ∀ x : ℝ, (m * (p * x + q) + n = p * (m * x + n) + q) :=
by sorry

end fg_equals_gf_l1599_159953


namespace solve_equation_l1599_159949

theorem solve_equation (x y z t : ℤ) (h : x^4 - 2*y^4 - 4*z^4 - 8*t^4 = 0) : x = 0 ∧ y = 0 ∧ z = 0 ∧ t = 0 :=
by
  sorry

end solve_equation_l1599_159949


namespace op_correct_l1599_159984

-- Definition of the operation * for non-zero integers
def op (a b : ℤ) : ℚ := (1 : ℚ) / a + (1 / b)

theorem op_correct (a b : ℤ) (ha : a ≠ 0) (hb : b ≠ 0) (h1 : a + b = 12) (h2 : a * b = 32) :
  op a b = 3 / 8 :=
by
  -- Proof, sorry for now
  sorry

end op_correct_l1599_159984


namespace part1_part2_l1599_159911

-- Define the conditions
def cost_price := 30
def initial_selling_price := 40
def initial_sales_volume := 600
def sales_decrease_per_yuan := 10

-- Define the profit calculation function
def profit (selling_price : ℕ) : ℕ :=
  let profit_per_unit := selling_price - cost_price
  let new_sales_volume := initial_sales_volume - sales_decrease_per_yuan * (selling_price - initial_selling_price)
  profit_per_unit * new_sales_volume

-- Statements to prove
theorem part1 :
  profit 50 = 10000 :=
by
  sorry

theorem part2 :
  let max_profit_price := 60
  let max_profit := 12000
  max_profit = (fun price => max (profit price) 0) 60 :=
by
  sorry

end part1_part2_l1599_159911


namespace find_integer_pairs_l1599_159942

theorem find_integer_pairs :
  { (m, n) : ℤ × ℤ | n^3 + m^3 + 231 = n^2 * m^2 + n * m } = {(4, 5), (5, 4)} :=
by
  sorry

end find_integer_pairs_l1599_159942


namespace count_valid_arrangements_l1599_159924

-- Definitions based on conditions
def total_chairs : Nat := 48

def valid_factor_pairs (n : Nat) : List (Nat × Nat) :=
  [ (2, 24), (3, 16), (4, 12), (6, 8), (8, 6), (12, 4), (16, 3), (24, 2) ]

def count_valid_arrays : Nat := valid_factor_pairs total_chairs |>.length

-- The theorem we want to prove
theorem count_valid_arrangements : count_valid_arrays = 8 := 
  by
    -- proof should be provided here
    sorry

end count_valid_arrangements_l1599_159924


namespace average_length_remaining_strings_l1599_159965

theorem average_length_remaining_strings 
  (T1 : ℕ := 6) (avg_length1 : ℕ := 80) 
  (T2 : ℕ := 2) (avg_length2 : ℕ := 70) :
  (6 * avg_length1 - 2 * avg_length2) / 4 = 85 := 
by
  sorry

end average_length_remaining_strings_l1599_159965


namespace negate_existential_l1599_159983

theorem negate_existential :
  ¬ (∃ x0 : ℝ, x0^2 - 2 * x0 + 4 > 0) ↔ ∀ x : ℝ, x^2 - 2 * x + 4 ≤ 0 :=
by
  sorry

end negate_existential_l1599_159983


namespace stadium_breadth_l1599_159968

theorem stadium_breadth (P L B : ℕ) (h1 : P = 800) (h2 : L = 100) :
  2 * (L + B) = P → B = 300 :=
by
  sorry

end stadium_breadth_l1599_159968


namespace range_of_a_l1599_159992

def tangent_perpendicular_to_y_axis (a : ℝ) : Prop :=
  ∃ (x : ℝ), x > 0 ∧ (3 * a * x^2 + 1 / x = 0)

theorem range_of_a : {a : ℝ | tangent_perpendicular_to_y_axis a} = {a : ℝ | a < 0} :=
by
  sorry

end range_of_a_l1599_159992


namespace chocolate_chip_more_than_raisin_l1599_159976

def chocolate_chip_yesterday : ℕ := 19
def chocolate_chip_morning : ℕ := 237
def raisin_cookies : ℕ := 231

theorem chocolate_chip_more_than_raisin : 
  (chocolate_chip_yesterday + chocolate_chip_morning) - raisin_cookies = 25 :=
by 
  sorry

end chocolate_chip_more_than_raisin_l1599_159976


namespace solution_proof_l1599_159926

noncomputable def problem_statement : Prop :=
  let a : ℝ := 0.10
  let b : ℝ := 0.50
  let c : ℝ := 500
  a * (b * c) = 25

theorem solution_proof : problem_statement := by
  sorry

end solution_proof_l1599_159926


namespace acid_solution_l1599_159947

theorem acid_solution (x y : ℝ) (h1 : 0.3 * x + 0.1 * y = 90)
  (h2 : x + y = 600) : x = 150 ∧ y = 450 :=
by
  sorry

end acid_solution_l1599_159947


namespace average_of_second_class_l1599_159987

variable (average1 : ℝ) (average2 : ℝ) (combined_average : ℝ) (n1 : ℕ) (n2 : ℕ)

theorem average_of_second_class
  (h1 : n1 = 25) 
  (h2 : average1 = 40) 
  (h3 : n2 = 30) 
  (h4 : combined_average = 50.90909090909091) 
  (h5 : n1 + n2 = 55) 
  (h6 : n2 * average2 = 55 * combined_average - n1 * average1) :
  average2 = 60 := by
  sorry

end average_of_second_class_l1599_159987


namespace max_value_of_f_l1599_159950

noncomputable def f (x a b : ℝ) := (1 - x^2) * (x^2 + a * x + b)

theorem max_value_of_f (a b : ℝ) (h_symmetric : ∀ x : ℝ, f (-2 - x) a b = f (-2 + x) a b) :
  ∃ x : ℝ, f x a b = 16 := by
  sorry

end max_value_of_f_l1599_159950


namespace john_initial_candies_l1599_159967

theorem john_initial_candies : ∃ x : ℕ, (∃ (x3 : ℕ), x3 = ((x - 2) / 2) ∧ x3 = 6) ∧ x = 14 := by
  sorry

end john_initial_candies_l1599_159967


namespace license_plate_combinations_l1599_159903

def consonants_count := 21
def vowels_count := 5
def digits_count := 10

theorem license_plate_combinations : 
  consonants_count * vowels_count * consonants_count * digits_count * vowels_count = 110250 :=
by
  sorry

end license_plate_combinations_l1599_159903


namespace sarah_boxes_l1599_159901

theorem sarah_boxes (b : ℕ) 
  (h1 : ∀ x : ℕ, x = 7) 
  (h2 : 49 = 7 * b) :
  b = 7 :=
sorry

end sarah_boxes_l1599_159901


namespace scientific_notation_of_600_million_l1599_159996

theorem scientific_notation_of_600_million : 600000000 = 6 * 10^7 := 
sorry

end scientific_notation_of_600_million_l1599_159996


namespace octahedron_vertices_sum_l1599_159905

noncomputable def octahedron_faces_sum (a b c d e f : ℕ) : ℕ :=
  a + b + c + d + e + f

theorem octahedron_vertices_sum (a b c d e f : ℕ) 
  (h : 8 * (octahedron_faces_sum a b c d e f) = 440) : 
  octahedron_faces_sum a b c d e f = 147 :=
by
  sorry

end octahedron_vertices_sum_l1599_159905


namespace molecular_weight_CaO_is_56_08_l1599_159944

-- Define the atomic weights of Calcium and Oxygen
def atomic_weight_Ca := 40.08 -- in g/mol
def atomic_weight_O := 16.00 -- in g/mol

-- Define the molecular weight of the compound
def molecular_weight_CaO := atomic_weight_Ca + atomic_weight_O

-- State the theorem
theorem molecular_weight_CaO_is_56_08 : molecular_weight_CaO = 56.08 :=
by
  -- The proof will be filled in here
  sorry

end molecular_weight_CaO_is_56_08_l1599_159944


namespace man_older_than_son_by_46_l1599_159919

-- Given conditions about the ages
def sonAge : ℕ := 44

def manAge_in_two_years (M : ℕ) : Prop := M + 2 = 2 * (sonAge + 2)

-- The problem to verify
theorem man_older_than_son_by_46 (M : ℕ) (h : manAge_in_two_years M) : M - sonAge = 46 :=
by
  sorry

end man_older_than_son_by_46_l1599_159919


namespace simplify_expression_l1599_159912

theorem simplify_expression (s : ℤ) : 120 * s - 32 * s = 88 * s := by
  sorry

end simplify_expression_l1599_159912


namespace total_distance_of_trip_l1599_159907

theorem total_distance_of_trip (x : ℚ)
  (highway : x / 4 ≤ x)
  (city : 30 ≤ x)
  (country : x / 6 ≤ x)
  (middle_part_fraction : 1 - 1 / 4 - 1 / 6 = 7 / 12) :
  (7 / 12) * x = 30 → x = 360 / 7 :=
by
  sorry

end total_distance_of_trip_l1599_159907


namespace find_amplitude_l1599_159973

-- Conditions
variables (a b c d : ℝ)

theorem find_amplitude
  (h1 : ∀ x, a * Real.sin (b * x + c) + d ≤ 5)
  (h2 : ∀ x, a * Real.sin (b * x + c) + d ≥ -3) :
  a = 4 :=
by 
  sorry

end find_amplitude_l1599_159973


namespace no_two_or_more_consecutive_sum_30_l1599_159966

theorem no_two_or_more_consecutive_sum_30 :
  ∀ (a n : ℕ), n ≥ 2 → (n * (2 * a + n - 1) = 60) → false :=
by
  intro a n hn h
  sorry

end no_two_or_more_consecutive_sum_30_l1599_159966


namespace nomogram_relation_l1599_159986

noncomputable def root_of_eq (x p q : ℝ) : Prop :=
  x^2 + p * x + q = 0

theorem nomogram_relation (x p q : ℝ) (hx : root_of_eq x p q) : 
  q = -x * p - x^2 :=
by 
  sorry

end nomogram_relation_l1599_159986


namespace point_in_fourth_quadrant_l1599_159989

noncomputable def a : ℤ := 2

theorem point_in_fourth_quadrant (x y : ℤ) (h1 : x = a - 1) (h2 : y = a - 3) (h3 : x > 0) (h4 : y < 0) : a = 2 := by
  sorry

end point_in_fourth_quadrant_l1599_159989


namespace sum_cubic_polynomial_l1599_159938

noncomputable def q : ℤ → ℤ := sorry  -- We use a placeholder definition for q

theorem sum_cubic_polynomial :
  q 3 = 2 ∧ q 8 = 22 ∧ q 12 = 10 ∧ q 17 = 32 →
  (q 2 + q 3 + q 4 + q 5 + q 6 + q 7 + q 8 + q 9 + q 10 + q 11 + q 12 + q 13 + q 14 + q 15 + q 16 + q 17 + q 18) = 272 :=
sorry

end sum_cubic_polynomial_l1599_159938


namespace solve_quadratic_equation_l1599_159936

theorem solve_quadratic_equation (x : ℝ) :
  (6 * x^2 - 3 * x - 1 = 2 * x - 2) ↔ (x = 1 / 3 ∨ x = 1 / 2) :=
by sorry

end solve_quadratic_equation_l1599_159936


namespace saving_time_for_downpayment_l1599_159998

def annual_salary : ℚ := 150000
def saving_rate : ℚ := 0.10
def house_cost : ℚ := 450000
def downpayment_rate : ℚ := 0.20

theorem saving_time_for_downpayment : 
  (downpayment_rate * house_cost) / (saving_rate * annual_salary) = 6 :=
by
  sorry

end saving_time_for_downpayment_l1599_159998


namespace red_pencil_count_l1599_159962

-- Definitions for provided conditions
def blue_pencils : ℕ := 20
def ratio : ℕ × ℕ := (5, 3)
def red_pencils (blue : ℕ) (rat : ℕ × ℕ) : ℕ := (blue / rat.fst) * rat.snd

-- Theorem statement
theorem red_pencil_count : red_pencils blue_pencils ratio = 12 := 
by
  sorry

end red_pencil_count_l1599_159962


namespace sum_of_x_and_y_l1599_159975

theorem sum_of_x_and_y :
  ∀ (x y : ℚ), (1/x + 1/y = 4) → (1/x - 1/y = -6) → x + y = -4/5 :=
by
  intros x y h1 h2
  sorry

end sum_of_x_and_y_l1599_159975


namespace solve_for_x_l1599_159972

theorem solve_for_x (x : ℂ) (h : 5 - 2 * I * x = 4 - 5 * I * x) : x = I / 3 :=
by
  sorry

end solve_for_x_l1599_159972


namespace no_square_ends_in_4444_l1599_159995

theorem no_square_ends_in_4444:
  ∀ (a k : ℕ), (a ^ 2 = 1000 * k + 444) → (∃ b m n : ℕ, (b = 500 * n + 38) ∨ (b = 500 * n - 38) → (a = 2 * b) →
  (a ^ 2 ≠ 1000 * m + 4444)) :=
by
  sorry

end no_square_ends_in_4444_l1599_159995
