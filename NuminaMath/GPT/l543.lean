import Mathlib

namespace daily_profit_9080_l543_54340

theorem daily_profit_9080 (num_employees : Nat) (shirts_per_employee_per_day : Nat) (hours_per_shift : Nat) (wage_per_hour : Nat) (bonus_per_shirt : Nat) (shirt_sale_price : Nat) (nonemployee_expenses : Nat) :
  num_employees = 20 →
  shirts_per_employee_per_day = 20 →
  hours_per_shift = 8 →
  wage_per_hour = 12 →
  bonus_per_shirt = 5 →
  shirt_sale_price = 35 →
  nonemployee_expenses = 1000 →
  (num_employees * shirts_per_employee_per_day * shirt_sale_price) - ((num_employees * (hours_per_shift * wage_per_hour + shirts_per_employee_per_day * bonus_per_shirt)) + nonemployee_expenses) = 9080 := 
by
  intros
  sorry

end daily_profit_9080_l543_54340


namespace range_of_a_l543_54361

-- Define the function g(x) = x^3 - 3ax - a
def g (a x : ℝ) : ℝ := x^3 - 3*a*x - a

-- Define the derivative of g(x) which is g'(x) = 3x^2 - 3a
def g' (a x : ℝ) : ℝ := 3*x^2 - 3*a

theorem range_of_a (a : ℝ) : g a 0 * g a 1 < 0 → 0 < a ∧ a < 1 :=
by
  sorry

end range_of_a_l543_54361


namespace intersection_of_sets_l543_54339

theorem intersection_of_sets (A B : Set ℕ) (hA : A = {0, 1, 2}) (hB : B = {1, 2, 3, 4}) :
  A ∩ B = {1, 2} :=
by
  sorry

end intersection_of_sets_l543_54339


namespace remainder_1234_mul_2047_mod_600_l543_54377

theorem remainder_1234_mul_2047_mod_600 : (1234 * 2047) % 600 = 198 := by
  sorry

end remainder_1234_mul_2047_mod_600_l543_54377


namespace water_speed_l543_54300

theorem water_speed (swimmer_speed still_water : ℝ) (distance time : ℝ) (h1 : swimmer_speed = 12) (h2 : distance = 12) (h3 : time = 6) :
  ∃ v : ℝ, v = 10 ∧ distance = (swimmer_speed - v) * time :=
by { sorry }

end water_speed_l543_54300


namespace unique_root_a_b_values_l543_54333

theorem unique_root_a_b_values {a b : ℝ} (h1 : ∀ x, x^2 + a * x + b = 0 ↔ x = 1) : a = -2 ∧ b = 1 := by
  sorry

end unique_root_a_b_values_l543_54333


namespace find_a1_for_geometric_sequence_l543_54319

noncomputable def geometric_sequence := ℕ → ℝ

def is_geometric_sequence (a : geometric_sequence) : Prop :=
  ∃ q, ∀ n, a (n + 1) = a n * q

theorem find_a1_for_geometric_sequence (a : geometric_sequence)
  (h_geom : is_geometric_sequence a)
  (h1 : a 2 * a 5 = 2 * a 3)
  (h2 : (a 4 + a 6) / 2 = 5 / 4) :
  a 1 = 16 ∨ a 1 = -16 :=
sorry

end find_a1_for_geometric_sequence_l543_54319


namespace vector_magnitude_proof_l543_54370

noncomputable def vector_magnitude (v : ℝ × ℝ) : ℝ := 
  Real.sqrt (v.1 ^ 2 + v.2 ^ 2)

theorem vector_magnitude_proof
  (a b c : ℝ × ℝ)
  (h_a : a = (-2, 1))
  (h_b : b = (-2, 3))
  (h_c : ∃ m : ℝ, c = (m, -1) ∧ (m * b.1 + (-1) * b.2 = 0)) :
  vector_magnitude (a.1 - c.1, a.2 - c.2) = Real.sqrt 17 / 2 :=
by
  sorry

end vector_magnitude_proof_l543_54370


namespace sum_of_a_and_b_l543_54312

theorem sum_of_a_and_b {a b : ℝ} (h : a^2 + b^2 + (a*b)^2 = 4*a*b - 1) : a + b = 2 ∨ a + b = -2 :=
sorry

end sum_of_a_and_b_l543_54312


namespace hyperbola_eccentricity_l543_54399

theorem hyperbola_eccentricity (a b m n e : ℝ) (h_a_pos : a > 0) (h_b_pos : b > 0) (h_mn : m * n = 2 / 9)
  (h_hyperbola : ∀ x y : ℝ, (x^2 / a^2) - (y^2 / b^2) = 1) : e = 3 * Real.sqrt 2 / 4 :=
sorry

end hyperbola_eccentricity_l543_54399


namespace common_difference_arithmetic_sequence_l543_54349

theorem common_difference_arithmetic_sequence
  (a : ℕ) (d : ℚ) (n : ℕ) (a_n : ℕ) (S_n : ℕ)
  (h1 : a = 2)
  (h2 : a_n = 20)
  (h3 : S_n = 132)
  (h4 : a_n = a + (n - 1) * d)
  (h5 : S_n = n * (a + a_n) / 2) :
  d = 18 / 11 := sorry

end common_difference_arithmetic_sequence_l543_54349


namespace distance_from_star_l543_54322

def speed_of_light : ℝ := 3 * 10^5 -- km/s
def time_years : ℝ := 4 -- years
def seconds_per_year : ℝ := 3 * 10^7 -- s

theorem distance_from_star :
  let distance := speed_of_light * (time_years * seconds_per_year)
  distance = 3.6 * 10^13 :=
by
  sorry

end distance_from_star_l543_54322


namespace sam_and_erica_money_total_l543_54308

def sam_money : ℕ := 38
def erica_money : ℕ := 53

theorem sam_and_erica_money_total : sam_money + erica_money = 91 :=
by
  -- the proof is not required; hence we skip it
  sorry

end sam_and_erica_money_total_l543_54308


namespace not_perfect_cube_l543_54397

theorem not_perfect_cube (n : ℕ) : ¬ ∃ k : ℕ, k ^ 3 = 2 ^ (2 ^ n) + 1 :=
sorry

end not_perfect_cube_l543_54397


namespace megawheel_seat_capacity_l543_54353

theorem megawheel_seat_capacity (seats people : ℕ) (h1 : seats = 15) (h2 : people = 75) : people / seats = 5 := by
  sorry

end megawheel_seat_capacity_l543_54353


namespace product_of_consecutive_triangular_not_square_infinite_larger_triangular_numbers_square_product_l543_54305

section TriangularNumbers

-- Define triangular numbers
def triangular (n : ℕ) : ℕ := n * (n + 1) / 2

-- Statement 1: The product of two consecutive triangular numbers is not a perfect square
theorem product_of_consecutive_triangular_not_square (n : ℕ) (hn : n > 0) :
  ¬ ∃ m : ℕ, triangular (n - 1) * triangular n = m * m := by
  sorry

-- Statement 2: There exist infinitely many larger triangular numbers such that the product with t_n is a perfect square
theorem infinite_larger_triangular_numbers_square_product (n : ℕ) :
  ∃ᶠ m in at_top, ∃ k : ℕ, triangular n * triangular m = k * k := by
  sorry

end TriangularNumbers

end product_of_consecutive_triangular_not_square_infinite_larger_triangular_numbers_square_product_l543_54305


namespace dry_mixed_fruits_weight_l543_54378

theorem dry_mixed_fruits_weight :
  ∀ (fresh_grapes_weight fresh_apples_weight : ℕ)
    (grapes_water_content fresh_grapes_dry_matter_perc : ℕ)
    (apples_water_content fresh_apples_dry_matter_perc : ℕ),
    fresh_grapes_weight = 400 →
    fresh_apples_weight = 300 →
    grapes_water_content = 65 →
    fresh_grapes_dry_matter_perc = 35 →
    apples_water_content = 84 →
    fresh_apples_dry_matter_perc = 16 →
    (fresh_grapes_weight * fresh_grapes_dry_matter_perc / 100) +
    (fresh_apples_weight * fresh_apples_dry_matter_perc / 100) = 188 := by
  sorry

end dry_mixed_fruits_weight_l543_54378


namespace cost_per_use_correct_l543_54341

-- Definitions based on conditions in the problem
def total_cost : ℕ := 30
def uses_per_week : ℕ := 3
def number_of_weeks : ℕ := 2
def total_uses : ℕ := uses_per_week * number_of_weeks

-- Statement based on the question and correct answer
theorem cost_per_use_correct : (total_cost / total_uses) = 5 := sorry

end cost_per_use_correct_l543_54341


namespace problem_a_problem_b_problem_c_l543_54310

variables {x y z t : ℝ}

-- Variables are positive
axiom pos_x : 0 < x
axiom pos_y : 0 < y
axiom pos_z : 0 < z
axiom pos_t : 0 < t

-- Problem a)
theorem problem_a : x^4 * y^2 * z + y^4 * x^2 * z + y^4 * z^2 * x + z^4 * y^2 * x + x^4 * z^2 * y + z^4 * x^2 * y
  ≥ 2 * (x^3 * y^2 * z^2 + x^2 * y^3 * z^2 + x^2 * y^2 * z^3) :=
sorry

-- Problem b)
theorem problem_b : x^5 + y^5 + z^5 ≥ x^2 * y^2 * z + x^2 * y * z^2 + x * y^2 * z^2 :=
sorry

-- Problem c)
theorem problem_c : x^3 + y^3 + z^3 + t^3 ≥ x * y * z + x * y * t + x * z * t + y * z * t :=
sorry

end problem_a_problem_b_problem_c_l543_54310


namespace naomi_regular_bikes_l543_54357
-- Import necessary libraries

-- Define the condition and the proof problem
theorem naomi_regular_bikes (R C : ℕ) (h1 : C = 11) 
  (h2 : 2 * R + 4 * C = 58) : R = 7 := 
  by 
  -- Include all necessary conditions as assumptions
  have hC : C = 11 := h1
  have htotal : 2 * R + 4 * C = 58 := h2
  -- Skip the proof itself
  sorry

end naomi_regular_bikes_l543_54357


namespace book_original_price_l543_54385

-- Definitions for conditions
def selling_price := 56
def profit_percentage := 75

-- Statement of the theorem
theorem book_original_price : ∃ CP : ℝ, selling_price = CP * (1 + profit_percentage / 100) ∧ CP = 32 :=
by
  sorry

end book_original_price_l543_54385


namespace number_of_false_propositions_is_even_l543_54303

theorem number_of_false_propositions_is_even 
  (P Q : Prop) : 
  ∃ (n : ℕ), (P ∧ ¬P ∧ (¬Q → ¬P) ∧ (Q → P)) = false ∧ n % 2 = 0 := sorry

end number_of_false_propositions_is_even_l543_54303


namespace find_c_l543_54366

/-
Given:
1. c and d are integers.
2. x^2 - x - 1 is a factor of cx^{18} + dx^{17} + x^2 + 1.
Show that c = -1597 under these conditions.

Assume we have the following Fibonacci number definitions:
F_16 = 987,
F_17 = 1597,
F_18 = 2584,
then:
Proof that c = -1597.
-/

noncomputable def fib (n : ℕ) : ℕ :=
  if n = 0 then 0
  else if n = 1 then 1
  else fib (n - 1) + fib (n - 2)

theorem find_c (c d : ℤ) (h1 : c * 2584 + d * 1597 + 1 = 0) (h2 : c * 1597 + d * 987 + 2 = 0) :
  c = -1597 :=
by
  sorry

end find_c_l543_54366


namespace banana_production_total_l543_54346

def banana_production (nearby_island_production : ℕ) (jakies_multiplier : ℕ) : ℕ :=
  nearby_island_production + (jakies_multiplier * nearby_island_production)

theorem banana_production_total
  (nearby_island_production : ℕ)
  (jakies_multiplier : ℕ)
  (h1 : nearby_island_production = 9000)
  (h2 : jakies_multiplier = 10)
  : banana_production nearby_island_production jakies_multiplier = 99000 :=
by
  sorry

end banana_production_total_l543_54346


namespace complex_magnitude_comparison_l543_54306

open Complex

theorem complex_magnitude_comparison :
  let z1 := (5 : ℂ) + (3 : ℂ) * I
  let z2 := (5 : ℂ) + (4 : ℂ) * I
  abs z1 < abs z2 :=
by 
  let z1 := (5 : ℂ) + (3 : ℂ) * I
  let z2 := (5 : ℂ) + (4 : ℂ) * I
  sorry

end complex_magnitude_comparison_l543_54306


namespace number_of_white_balls_l543_54352

theorem number_of_white_balls (x : ℕ) (h1 : 3 + x ≠ 0) (h2 : (3 : ℚ) / (3 + x) = 1 / 5) : x = 12 :=
sorry

end number_of_white_balls_l543_54352


namespace trapezoid_ABCD_BCE_area_l543_54326

noncomputable def triangle_area (a b c : ℝ) (angle_abc : ℝ) : ℝ :=
  1 / 2 * a * b * Real.sin angle_abc

noncomputable def area_of_triangle_BCE (AB DC AD : ℝ) (angle_DAB : ℝ) (area_triangle_DCB : ℝ) : ℝ :=
  let ratio := AB / DC
  (ratio / (1 + ratio)) * area_triangle_DCB

theorem trapezoid_ABCD_BCE_area :
  ∀ (AB DC AD : ℝ) (angle_DAB : ℝ) (area_triangle_DCB : ℝ),
    AB = 30 →
    DC = 24 →
    AD = 3 →
    angle_DAB = Real.pi / 3 →
    area_triangle_DCB = 18 * Real.sqrt 3 →
    area_of_triangle_BCE AB DC AD angle_DAB area_triangle_DCB = 10 * Real.sqrt 3 := 
by
  intros
  sorry

end trapezoid_ABCD_BCE_area_l543_54326


namespace num_words_at_least_one_vowel_l543_54324

-- Definitions based on conditions.
def letters : List Char := ['A', 'B', 'E', 'G', 'H']
def vowels : List Char := ['A', 'E']
def consonants : List Char := ['B', 'G', 'H']

-- The main statement posing the question and answer.
theorem num_words_at_least_one_vowel :
  let total_words := (letters.length) ^ 5
  let consonant_words := (consonants.length) ^ 5
  let result := total_words - consonant_words
  result = 2882 :=
by {
  let total_words := 5 ^ 5
  let consonant_words := 3 ^ 5
  let result := total_words - consonant_words
  have : result = 2882 := by sorry
  exact this
}

end num_words_at_least_one_vowel_l543_54324


namespace moles_of_HCl_formed_l543_54347

-- Define the reaction
def balancedReaction (CH4 Cl2 CH3Cl HCl : ℕ) : Prop :=
  CH4 + Cl2 = CH3Cl + HCl

-- Number of moles given
def molesCH4 := 2
def molesCl2 := 4

-- Theorem statement
theorem moles_of_HCl_formed :
  ∀ CH4 Cl2 CH3Cl HCl : ℕ, balancedReaction CH4 Cl2 CH3Cl HCl →
  CH4 = molesCH4 →
  Cl2 = molesCl2 →
  HCl = 2 := sorry

end moles_of_HCl_formed_l543_54347


namespace total_hooligans_l543_54337

def hooligans_problem (X Y : ℕ) : Prop :=
  (X * Y = 365) ∧ (X + Y = 78 ∨ X + Y = 366)

theorem total_hooligans (X Y : ℕ) (h : hooligans_problem X Y) : X + Y = 78 ∨ X + Y = 366 :=
  sorry

end total_hooligans_l543_54337


namespace edge_length_of_divided_cube_l543_54314

theorem edge_length_of_divided_cube (volume_original_cube : ℕ) (num_divisions : ℕ) (volume_of_one_smaller_cube : ℕ) (edge_length : ℕ) :
  volume_original_cube = 1000 →
  num_divisions = 8 →
  volume_of_one_smaller_cube = volume_original_cube / num_divisions →
  volume_of_one_smaller_cube = edge_length ^ 3 →
  edge_length = 5 :=
by
  sorry

end edge_length_of_divided_cube_l543_54314


namespace inequality_abc_l543_54363

theorem inequality_abc (a b c : ℝ) : a^2 + b^2 + c^2 ≥ a*b + b*c + c*a :=
by
  sorry

end inequality_abc_l543_54363


namespace no_polynomial_exists_l543_54359

open Polynomial

theorem no_polynomial_exists (a b c : ℤ) (h1 : a ≠ b) (h2 : b ≠ c) (h3 : c ≠ a) :
  ¬ ∃ (P : ℤ[X]), P.eval a = b ∧ P.eval b = c ∧ P.eval c = a :=
sorry

end no_polynomial_exists_l543_54359


namespace max_mn_l543_54342

theorem max_mn (a : ℝ) (h₀ : a > 0) (h₁ : a ≠ 1) (m n : ℝ)
  (h₂ : 2 * m + n = 2) : m * n ≤ 4 / 9 :=
by
  sorry

end max_mn_l543_54342


namespace math_problem_proof_l543_54343

noncomputable def ellipse_equation : Prop := 
  let e := (Real.sqrt 2) / 2
  ∃ (a b : ℝ), 0 < a ∧ a > b ∧ e = (Real.sqrt 2) / 2 ∧ 
    (∀ x y, (x^2) / (a^2) + (y^2) / (b^2) = 1 ↔ x^2 / 2 + y^2 = 1)

noncomputable def fixed_point_exist : Prop :=
  let S := (0, 1/3) 
  ∀ k : ℝ, ∃ A B : ℝ × ℝ, 
    let M := (0, 1)
    ( 
        (A.1, A.2) ∈ {P : ℝ × ℝ | (P.1^2) / 2 + P.2^2 = 1} ∧ 
        (B.1, B.2) ∈ {P : ℝ × ℝ | (P.1^2) / 2 + P.2^2 = 1} ∧ 
        (S.2 = k * S.1 - 1 / 3) ∧ 
        ((A.1 - M.1)^2 + (A.2 - M.2)^2) + ((B.1 - M.1)^2 + (B.2 - M.2)^2) = ((A.1 - B.1)^2 + (A.2 - M.2)^2) / 2)

theorem math_problem_proof : ellipse_equation ∧ fixed_point_exist := sorry

end math_problem_proof_l543_54343


namespace walking_distance_l543_54362

theorem walking_distance (a b : ℝ) (h1 : 10 * a + 45 * b = a * 15)
(h2 : x * (a + 9 * b) = 10 * a + 45 * b) : x = 13.5 :=
by
  sorry

end walking_distance_l543_54362


namespace alyssa_bought_224_new_cards_l543_54330

theorem alyssa_bought_224_new_cards
  (initial_cards : ℕ)
  (after_purchase_cards : ℕ)
  (h1 : initial_cards = 676)
  (h2 : after_purchase_cards = 900) :
  after_purchase_cards - initial_cards = 224 :=
by
  -- Placeholder to avoid proof since it's explicitly not required 
  sorry

end alyssa_bought_224_new_cards_l543_54330


namespace sum_of_coordinates_of_center_l543_54325

theorem sum_of_coordinates_of_center (x1 y1 x2 y2 : ℝ) (h1 : (x1, y1) = (7, -6)) (h2 : (x2, y2) = (-1, 4)) :
  let center_x := (x1 + x2) / 2
  let center_y := (y1 + y2) / 2
  center_x + center_y = 2 := by
  sorry

end sum_of_coordinates_of_center_l543_54325


namespace farmer_john_pairs_l543_54375

noncomputable def farmer_john_animals_pairing :
    Nat := 
  let cows := 5
  let pigs := 4
  let horses := 7
  let num_ways_cow_pig_pair := cows * pigs
  let num_ways_horses_remaining := Nat.factorial horses
  num_ways_cow_pig_pair * num_ways_horses_remaining

theorem farmer_john_pairs : farmer_john_animals_pairing = 100800 := 
by
  sorry

end farmer_john_pairs_l543_54375


namespace a_less_than_2_l543_54321

-- Define the quadratic function f(x)
noncomputable def f (x : ℝ) : ℝ := x^2 - 6 * x + 2

-- Define the condition that the inequality f(x) - a > 0 has solutions in the interval [0,5]
def inequality_holds (a : ℝ) : Prop := ∀ (x : ℝ), 0 ≤ x ∧ x ≤ 5 → f x - a > 0

-- Theorem stating that a must be less than 2 to satisfy the above condition
theorem a_less_than_2 : ∀ (a : ℝ), (∃ (x : ℝ), 0 ≤ x ∧ x ≤ 5 ∧ f x - a > 0) → a < 2 := 
sorry

end a_less_than_2_l543_54321


namespace unique_solution_exists_l543_54374

theorem unique_solution_exists :
  ∃ (x y : ℝ), x = -13 / 96 ∧ y = 13 / 40 ∧
    (x / Real.sqrt (x^2 + y^2) - 1/x = 7) ∧
    (y / Real.sqrt (x^2 + y^2) + 1/y = 4) :=
by
  sorry

end unique_solution_exists_l543_54374


namespace volume_surface_ratio_l543_54344

-- Define the structure of the shape
structure Shape where
  center_cube : unit
  surrounding_cubes : Fin 6 -> unit
  top_cube : unit

-- Define the properties for the calculation
def volume (s : Shape) : ℕ := 8
def surface_area (s : Shape) : ℕ := 28
def ratio_volume_surface_area (s : Shape) : ℚ := volume s / surface_area s

-- Main theorem statement
theorem volume_surface_ratio (s : Shape) : ratio_volume_surface_area s = 2 / 7 := sorry

end volume_surface_ratio_l543_54344


namespace coffee_pods_per_box_l543_54380

theorem coffee_pods_per_box (d k : ℕ) (c e : ℝ) (h1 : d = 40) (h2 : k = 3) (h3 : c = 8) (h4 : e = 32) :
  ∃ b : ℕ, b = 30 :=
by
  sorry

end coffee_pods_per_box_l543_54380


namespace tangent_line_at_point_l543_54390

noncomputable def tangent_line_equation (x : ℝ) : Prop :=
  ∀ y : ℝ, y = x * (3 * Real.log x + 1) → (x = 1 ∧ y = 1) → y = 4 * x - 3

theorem tangent_line_at_point : tangent_line_equation 1 :=
sorry

end tangent_line_at_point_l543_54390


namespace evaluate_expression_m_4_evaluate_expression_m_negative_4_l543_54311

variables (a b c d m : ℝ)

theorem evaluate_expression_m_4 (h1 : a + b = 0) (h2 : c * d = 1) (h3 : |m| = 4) (h_m_4 : m = 4) :
  (a + b) / (3 * m) + m^2 - 5 * (c * d) + 6 * m = 35 :=
by sorry

theorem evaluate_expression_m_negative_4 (h1 : a + b = 0) (h2 : c * d = 1) (h3 : |m| = 4) (h_m_negative_4 : m = -4) :
  (a + b) / (3 * m) + m^2 - 5 * (c * d) + 6 * m = -13 :=
by sorry

end evaluate_expression_m_4_evaluate_expression_m_negative_4_l543_54311


namespace complement_union_correct_l543_54358

open Set

variable (U A B : Set ℕ)
variable (hU : U = {0, 1, 2, 3, 4})
variable (hA : A = {0, 1, 2})
variable (hB : B = {2, 3})

theorem complement_union_correct :
  (U \ A) ∪ B = {2, 3, 4} := by
  sorry

end complement_union_correct_l543_54358


namespace addition_problem_l543_54332

theorem addition_problem (F I V N E : ℕ) (h1: F = 8) (h2: I % 2 = 0) 
  (h3: 1 ≤ F ∧ F ≤ 9) (h4: 1 ≤ I ∧ I ≤ 9) (h5: 1 ≤ V ∧ V ≤ 9) 
  (h6: 1 ≤ N ∧ N ≤ 9) (h7: 1 ≤ E ∧ E ≤ 9) 
  (h8: F ≠ I ∧ F ≠ V ∧ F ≠ N ∧ F ≠ E) 
  (h9: I ≠ V ∧ I ≠ N ∧ I ≠ E ∧ V ≠ N ∧ V ≠ E ∧ N ≠ E)
  (h10: 2 * F + 2 * I + 2 * V = 1000 * N + 100 * I + 10 * N + E):
  V = 5 :=
sorry

end addition_problem_l543_54332


namespace smallest_prime_with_composite_reverse_l543_54394

def is_prime (n : Nat) : Prop := 
  n > 1 ∧ ∀ m : Nat, m > 1 ∧ m < n → n % m ≠ 0

def is_composite (n : Nat) : Prop :=
  n > 1 ∧ ∃ m : Nat, m > 1 ∧ m < n ∧ n % m = 0

def reverse_digits (n : Nat) : Nat :=
  let tens := n / 10
  let ones := n % 10
  ones * 10 + tens

theorem smallest_prime_with_composite_reverse :
  ∃ (n : Nat), 10 ≤ n ∧ n < 100 ∧ is_prime n ∧ (n / 10 = 3) ∧ is_composite (reverse_digits n) ∧
  (∀ m : Nat, 10 ≤ m ∧ m < n ∧ (m / 10 = 3) ∧ is_prime m → ¬is_composite (reverse_digits m)) :=
by
  sorry

end smallest_prime_with_composite_reverse_l543_54394


namespace solve_abs_eq_l543_54367

theorem solve_abs_eq (x : ℝ) : (|x + 2| = 3*x - 6) → x = 4 :=
by
  intro h
  sorry

end solve_abs_eq_l543_54367


namespace tower_no_knights_l543_54386

-- Define the problem conditions in Lean

variable {T : Type} -- Type for towers
variable {K : Type} -- Type for knights

variable (towers : Fin 9 → T)
variable (knights : Fin 18 → K)

-- Movement of knights: each knight moves to a neighboring tower every hour (either clockwise or counterclockwise)
variable (moves : K → (T → T))

-- Each knight stands watch at each tower exactly once over the course of the night
variable (stands_watch : ∀ k : K, ∀ t : T, ∃ hour : Fin 9, moves k t = towers hour)

-- Condition: at one time (say hour 1), each tower had at least two knights on watch
variable (time1 : Fin 9 → Fin 9 → ℕ) -- Number of knights at each tower at hour 1
variable (cond1 : ∀ i : Fin 9, 2 ≤ time1 1 i)

-- Condition: at another time (say hour 2), exactly five towers each had exactly one knight on watch
variable (time2 : Fin 9 → Fin 9 → ℕ) -- Number of knights at each tower at hour 2
variable (cond2 : ∃ seq : Fin 5 → Fin 9, (∀ i : Fin 5, time2 2 (seq i) = 1) ∧ ∀ j : Fin 4, i ≠ j → 1 ≠ seq j)

-- Prove: there exists a time when one of the towers had no knights at all
theorem tower_no_knights : ∃ hour : Fin 9, ∃ i : Fin 9, moves (knights i) (towers hour) = towers hour ∧ (∀ knight : K, moves knight (towers hour) ≠ towers hour) :=
sorry

end tower_no_knights_l543_54386


namespace otimes_calculation_l543_54334

def otimes (x y : ℝ) : ℝ := x^2 - 2*y

theorem otimes_calculation (k : ℝ) : otimes k (otimes k k) = -k^2 + 4*k :=
by
  sorry

end otimes_calculation_l543_54334


namespace derivative_of_f_l543_54395

noncomputable def f (x : ℝ) : ℝ := (Real.cos x) / x

theorem derivative_of_f :
  ∀ x ≠ 0, deriv f x = ((-x * Real.sin x - Real.cos x) / (x^2)) := sorry

end derivative_of_f_l543_54395


namespace teamA_fraction_and_sum_l543_54356

def time_to_minutes (t : ℝ) : ℝ := t * 60

def fraction_teamA_worked (m n : ℕ) (h_coprime : Nat.gcd m n = 1) (h_fraction : m = 1 ∧ n = 5) : Prop :=
  (90 - 60) / 150 = m / n

theorem teamA_fraction_and_sum (m n : ℕ) (h_coprime : Nat.gcd m n = 1) (h_fraction : m = 1 ∧ n = 5) :
  90 / 150 = 1 / 5 → m + n = 6 :=
by
  sorry

end teamA_fraction_and_sum_l543_54356


namespace simplify_expr_l543_54355

def expr (y : ℝ) := y * (4 * y^2 - 3) - 6 * (y^2 - 3 * y + 8)

theorem simplify_expr (y : ℝ) : expr y = 4 * y^3 - 6 * y^2 + 15 * y - 48 :=
by
  sorry

end simplify_expr_l543_54355


namespace sqrt_of_16_eq_4_sqrt_of_364_eq_pm19_opposite_of_2_sub_sqrt6_eq_sqrt6_sub_2_l543_54307

theorem sqrt_of_16_eq_4 : Real.sqrt 16 = 4 := 
by sorry

theorem sqrt_of_364_eq_pm19 : Real.sqrt 364 = 19 ∨ Real.sqrt 364 = -19 := 
by sorry

theorem opposite_of_2_sub_sqrt6_eq_sqrt6_sub_2 : -(2 - Real.sqrt 6) = Real.sqrt 6 - 2 := 
by sorry

end sqrt_of_16_eq_4_sqrt_of_364_eq_pm19_opposite_of_2_sub_sqrt6_eq_sqrt6_sub_2_l543_54307


namespace find_a_solve_inequality_l543_54372

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := (2^x + a) / (2^x - 1)

theorem find_a (h : ∀ x : ℝ, f x a = -f (-x) a) : a = 1 := sorry

theorem solve_inequality (x : ℝ) (hx : 0 < x ∧ x < 1) : f x 1 > 3 := sorry

end find_a_solve_inequality_l543_54372


namespace magnitude_of_T_l543_54329

open Complex

noncomputable def i : ℂ := Complex.I

noncomputable def T : ℂ := (1 + i)^19 - (1 - i)^19

theorem magnitude_of_T : Complex.abs T = 1024 := by
  sorry

end magnitude_of_T_l543_54329


namespace total_distance_hopped_l543_54313

def distance_hopped (rate: ℕ) (time: ℕ) : ℕ := rate * time

def spotted_rabbit_distance (time: ℕ) : ℕ :=
  let pattern := [8, 11, 16, 20, 9]
  let full_cycles := time / pattern.length
  let remaining_minutes := time % pattern.length
  let full_cycle_distance := full_cycles * pattern.sum
  let remaining_distance := (List.take remaining_minutes pattern).sum
  full_cycle_distance + remaining_distance

theorem total_distance_hopped :
  distance_hopped 15 12 + distance_hopped 12 12 + distance_hopped 18 12 + distance_hopped 10 12 + spotted_rabbit_distance 12 = 807 :=
by
  sorry

end total_distance_hopped_l543_54313


namespace solution_set_f_neg_x_l543_54351

noncomputable def f (a b x : Real) : Real := (a * x - 1) * (x - b)

theorem solution_set_f_neg_x (a b : Real) (h : ∀ x, f a b x > 0 ↔ -1 < x ∧ x < 3) : 
  ∀ x, f a b (-x) < 0 ↔ x < -3 ∨ x > 1 := 
by
  sorry

end solution_set_f_neg_x_l543_54351


namespace trig_expression_value_l543_54382

theorem trig_expression_value (α : ℝ) (h : Real.tan (Real.pi + α) = 2) : 
  (Real.sin (α - Real.pi) + Real.cos (Real.pi - α)) / (Real.sin (Real.pi + α) - Real.cos (Real.pi - α)) = 3 := 
by
  sorry

end trig_expression_value_l543_54382


namespace reduce_entanglement_l543_54316

/- 
Define a graph structure and required operations as per the given conditions. 
-/
structure Graph (V : Type) :=
  (E : V -> V -> Prop)

def remove_odd_degree_verts (G : Graph V) : Graph V :=
  sorry -- Placeholder for graph reduction logic

def duplicate_graph (G : Graph V) : Graph V :=
  sorry -- Placeholder for graph duplication logic

/--
  Prove that any graph where each vertex can be part of multiple entanglements 
  can be reduced to a state where no two vertices are connected using the given operations.
-/
theorem reduce_entanglement (G : Graph V) : ∃ G', 
  G' = remove_odd_degree_verts (duplicate_graph G) ∧
  (∀ (v1 v2 : V), ¬ G'.E v1 v2) :=
  by
  sorry

end reduce_entanglement_l543_54316


namespace smallest_n_condition_l543_54317

theorem smallest_n_condition :
  ∃ n ≥ 2, ∃ (a : Fin n → ℤ), (Finset.sum Finset.univ a = 1990) ∧ (Finset.univ.prod a = 1990) ∧ (n = 5) :=
by
  sorry

end smallest_n_condition_l543_54317


namespace evaluate_expression_l543_54368

-- Definitions for a and b
def a : Int := 1
def b : Int := -1

theorem evaluate_expression : 
  5 * (3 * a ^ 2 * b - a * b ^ 2) - (a * b ^ 2 + 3 * a ^ 2 * b) + 1 = -17 := by
  -- Simplification steps skipped
  sorry

end evaluate_expression_l543_54368


namespace trigonometric_identity_proof_l543_54360

theorem trigonometric_identity_proof 
  (α : ℝ) 
  (h1 : Real.tan (2 * α) = 3 / 4) 
  (h2 : α ∈ Set.Ioo (-(Real.pi / 2)) (Real.pi / 2))
  (h3 : ∃ x : ℝ, (Real.sin (x + 2) + Real.sin (α - x) - 2 * Real.sin α) = 0) : 
  Real.cos (2 * α) = -4 / 5 ∧ Real.tan (α / 2) = (1 - Real.sqrt 10) / 3 := 
sorry

end trigonometric_identity_proof_l543_54360


namespace school_C_paintings_l543_54345

theorem school_C_paintings
  (A B C : ℕ)
  (h1 : B + C = 41)
  (h2 : A + C = 38)
  (h3 : A + B = 43) : 
  C = 18 :=
by
  sorry

end school_C_paintings_l543_54345


namespace eval_expression_l543_54302

theorem eval_expression :
  (2011 * (2012 * 10001) * (2013 * 100010001)) - (2013 * (2011 * 10001) * (2012 * 100010001)) =
  -2 * 2012 * 2013 * 10001 * 100010001 :=
by
  sorry

end eval_expression_l543_54302


namespace tetrahedron_a_exists_tetrahedron_b_not_exists_l543_54320

/-- Part (a): There exists a tetrahedron with two edges shorter than 1 cm,
    and the other four edges longer than 1 km. -/
theorem tetrahedron_a_exists : 
  ∃ (a b c d : ℝ), 
    ((a < 1 ∧ b < 1 ∧ 1000 < c ∧ 1000 < d ∧ 1000 < (a + c) ∧ 1000 < (b + d)) ∧ 
     a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ d ≠ a) := 
sorry

/-- Part (b): There does not exist a tetrahedron with four edges shorter than 1 cm,
    and the other two edges longer than 1 km. -/
theorem tetrahedron_b_not_exists : 
  ¬ ∃ (a b c d : ℝ), 
    ((a < 1 ∧ b < 1 ∧ c < 1 ∧ d < 1 ∧ 1000 < (a + c) ∧ 1000 < (b + d)) ∧ 
     a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ d ≠ a ) := 
sorry

end tetrahedron_a_exists_tetrahedron_b_not_exists_l543_54320


namespace bankers_discount_l543_54336

theorem bankers_discount {TD S BD : ℝ} (hTD : TD = 66) (hS : S = 429) :
  BD = (TD * S) / (S - TD) → BD = 78 :=
by
  intros h
  rw [hTD, hS] at h
  sorry

end bankers_discount_l543_54336


namespace solve_for_x_l543_54318

theorem solve_for_x (x y : ℝ) (h1 : y = 1 / (4 * x + 2)) (h2 : y = 2) : x = -3 / 8 := by 
  sorry

end solve_for_x_l543_54318


namespace bisection_method_root_exists_bisection_method_next_calculation_l543_54391

noncomputable def f (x : ℝ) : ℝ := x^3 + 3 * x - 1

theorem bisection_method_root_exists :
  (f 0 < 0) → (f 0.5 > 0) → ∃ x0 : ℝ, 0 < x0 ∧ x0 < 0.5 ∧ f x0 = 0 :=
by
  intro h0 h05
  sorry

theorem bisection_method_next_calculation :
  f 0.25 = (0.25)^3 + 3 * 0.25 - 1 :=
by
  calc
    f 0.25 = 0.25^3 + 3 * 0.25 - 1 := rfl

end bisection_method_root_exists_bisection_method_next_calculation_l543_54391


namespace unanswered_questions_count_l543_54381

-- Define the variables: c (correct), w (wrong), u (unanswered)
variables (c w u : ℕ)

-- Define the conditions based on the problem statement.
def total_questions (c w u : ℕ) : Prop := c + w + u = 35
def new_system_score (c u : ℕ) : Prop := 6 * c + 3 * u = 120
def old_system_score (c w : ℕ) : Prop := 5 * c - 2 * w = 55

-- Prove that the number of unanswered questions, u, equals 10
theorem unanswered_questions_count (c w u : ℕ) 
    (h1 : total_questions c w u)
    (h2 : new_system_score c u)
    (h3 : old_system_score c w) : u = 10 :=
by
  sorry

end unanswered_questions_count_l543_54381


namespace average_weight_of_a_and_b_l543_54338

-- Given conditions as Lean definitions
variable (A B C : ℝ)
variable (h1 : (A + B + C) / 3 = 45)
variable (h2 : (B + C) / 2 = 46)
variable (hB : B = 37)

-- The statement we want to prove
theorem average_weight_of_a_and_b : (A + B) / 2 = 40 := by
  sorry

end average_weight_of_a_and_b_l543_54338


namespace exactly_one_gt_one_of_abc_eq_one_l543_54398

theorem exactly_one_gt_one_of_abc_eq_one 
  (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) 
  (h_abc : a * b * c = 1) 
  (h_sum : a + b + c > 1 / a + 1 / b + 1 / c) : 
  (1 < a ∧ b < 1 ∧ c < 1) ∨ (a < 1 ∧ 1 < b ∧ c < 1) ∨ (a < 1 ∧ b < 1 ∧ 1 < c) :=
sorry

end exactly_one_gt_one_of_abc_eq_one_l543_54398


namespace part1_part2_l543_54328

def f (x : ℝ) : ℝ := abs (x - 2)

theorem part1 (x : ℝ) : f x > 4 - abs (x + 1) ↔ x < -3 / 2 ∨ x > 5 / 2 := 
sorry

theorem part2 (a b : ℝ) (ha : 0 < a ∧ a < 1/2) (hb : 0 < b ∧ b < 1/2)
  (h : f (1 / a) + f (2 / b) = 10) : a + b / 2 ≥ 2 / 7 := 
sorry

end part1_part2_l543_54328


namespace appropriate_sampling_method_is_stratified_l543_54327

-- Definition of the problem conditions
def total_students := 500 + 500
def male_students := 500
def female_students := 500
def survey_sample_size := 100

-- The goal is to show that given these conditions, the appropriate sampling method is Stratified sampling method.
theorem appropriate_sampling_method_is_stratified :
  total_students = 1000 ∧
  male_students = 500 ∧
  female_students = 500 ∧
  survey_sample_size = 100 →
  sampling_method = "Stratified" :=
by
  intros h
  sorry

end appropriate_sampling_method_is_stratified_l543_54327


namespace half_abs_diff_squares_l543_54376

theorem half_abs_diff_squares (a b : ℤ) (h₁ : a = 21) (h₂ : b = 17) :
  (|a^2 - b^2| / 2) = 76 :=
by 
  sorry

end half_abs_diff_squares_l543_54376


namespace prime_1011_n_l543_54348

theorem prime_1011_n (n : ℕ) (h : n ≥ 2) : 
  n = 2 ∨ n = 3 ∨ (∀ m : ℕ, m ∣ (n^3 + n + 1) → m = 1 ∨ m = n^3 + n + 1) :=
by sorry

end prime_1011_n_l543_54348


namespace nishita_common_shares_l543_54389

def annual_dividend_preferred_shares (num_preferred_shares : ℕ) (par_value : ℕ) (dividend_rate_preferred : ℕ) : ℕ :=
  (dividend_rate_preferred * par_value * num_preferred_shares) / 100

def annual_dividend_common_shares (total_dividend : ℕ) (dividend_preferred : ℕ) : ℕ :=
  total_dividend - dividend_preferred

def number_of_common_shares (annual_dividend_common : ℕ) (par_value : ℕ) (annual_rate_common : ℕ) : ℕ :=
  annual_dividend_common / ((annual_rate_common * par_value) / 100)

theorem nishita_common_shares (total_annual_dividend : ℕ) (num_preferred_shares : ℕ)
                             (par_value : ℕ) (dividend_rate_preferred : ℕ)
                             (semi_annual_rate_common : ℕ) : 
                             (number_of_common_shares (annual_dividend_common_shares total_annual_dividend 
                             (annual_dividend_preferred_shares num_preferred_shares par_value dividend_rate_preferred)) 
                             par_value (semi_annual_rate_common * 2)) = 3000 :=
by
  -- Provide values specific to the problem
  let total_annual_dividend := 16500
  let num_preferred_shares := 1200
  let par_value := 50
  let dividend_rate_preferred := 10
  let semi_annual_rate_common := 3.5
  sorry

end nishita_common_shares_l543_54389


namespace no_such_functions_l543_54350

open Real

theorem no_such_functions : ¬ ∃ (f g : ℝ → ℝ), (∀ x y : ℝ, f (x^2 + g y) - f (x^2) + g (y) - g (x) ≤ 2 * y) ∧ (∀ x : ℝ, f (x) ≥ x^2) := by
  sorry

end no_such_functions_l543_54350


namespace find_m_l543_54323

theorem find_m (S : ℕ → ℕ) (a : ℕ → ℕ) (m : ℕ) :
  (∀ n, S n = (n * (3 * n - 1)) / 2) →
  (a 1 = 1) →
  (∀ n ≥ 2, a n = S n - S (n - 1)) →
  (a m = 3 * m - 2) →
  (a 4 * a 4 = a 1 * a m) →
  m = 34 :=
by
  intro hS h1 ha1 ha2 hgeom
  sorry

end find_m_l543_54323


namespace new_student_weight_l543_54304

theorem new_student_weight (avg_weight : ℝ) (x : ℝ) :
  (avg_weight * 10 - 120) = ((avg_weight - 6) * 10 + x) → x = 60 :=
by
  intro h
  -- The proof would go here, but it's skipped.
  sorry

end new_student_weight_l543_54304


namespace describe_set_T_l543_54392

-- Define the conditions for the set of points T
def satisfies_conditions (x y : ℝ) : Prop :=
  (x + 3 = 4 ∧ y < 7) ∨ (y - 3 = 4 ∧ x < 1)

-- Define the set T based on the conditions
def set_T := {p : ℝ × ℝ | satisfies_conditions p.1 p.2}

-- Statement to prove the geometric description of the set T
theorem describe_set_T :
  (∃ x y, satisfies_conditions x y) → ∃ p1 p2,
  (p1 = (1, t) ∧ t < 7 → satisfies_conditions 1 t) ∧
  (p2 = (t, 7) ∧ t < 1 → satisfies_conditions t 7) ∧
  (p1 ≠ p2) :=
sorry

end describe_set_T_l543_54392


namespace emily_height_in_cm_l543_54301

theorem emily_height_in_cm 
  (inches_in_foot : ℝ) (cm_in_foot : ℝ) (emily_height_in_inches : ℝ)
  (h_if : inches_in_foot = 12) (h_cf : cm_in_foot = 30.5) (h_ehi : emily_height_in_inches = 62) :
  emily_height_in_inches * (cm_in_foot / inches_in_foot) = 157.6 :=
by
  sorry

end emily_height_in_cm_l543_54301


namespace angle_ABC_is_83_degrees_l543_54371

theorem angle_ABC_is_83_degrees (A B C D K : Type)
  (angle_BAC : Real) (angle_CAD : Real) (angle_ACD : Real)
  (AB AC AD : Real) (angle_ABC : Real) :
  angle_BAC = 60 ∧ angle_CAD = 60 ∧ angle_ACD = 23 ∧ AB + AD = AC → 
  angle_ABC = 83 :=
by
  sorry

end angle_ABC_is_83_degrees_l543_54371


namespace sum_of_coeffs_l543_54309

theorem sum_of_coeffs 
  (a b c d e x : ℝ)
  (h : (729 * x ^ 3 + 8) = (a * x + b) * (c * x ^ 2 + d * x + e)) :
  a + b + c + d + e = 78 :=
sorry

end sum_of_coeffs_l543_54309


namespace sum_of_ab_conditions_l543_54384

theorem sum_of_ab_conditions (a b : ℝ) (h : a^3 + b^3 = 1 - 3 * a * b) : a + b = 1 ∨ a + b = -2 := 
by
  sorry

end sum_of_ab_conditions_l543_54384


namespace greatest_divisor_l543_54379

theorem greatest_divisor (d : ℕ) :
  (1657 % d = 6 ∧ 2037 % d = 5) → d = 127 := by
  sorry

end greatest_divisor_l543_54379


namespace find_constant_a_l543_54393

noncomputable def f (a t : ℝ) : ℝ := (t - 2)^2 - 4 - a

theorem find_constant_a :
  (∃ (a : ℝ),
    (∀ (t : ℝ), -1 ≤ t ∧ t ≤ 1 → |f a t| ≤ 4) ∧ 
    (∃ (t : ℝ), -1 ≤ t ∧ t ≤ 1 ∧ |f a t| = 4)) →
  a = 1 :=
sorry

end find_constant_a_l543_54393


namespace probability_product_positive_is_5_div_9_l543_54365

noncomputable def probability_positive_product : ℚ :=
  let interval := Set.Icc (-30 : ℝ) 15
  let length_interval := 45
  let length_neg := 30
  let length_pos := 15
  let prob_neg := (length_neg : ℚ) / length_interval
  let prob_pos := (length_pos : ℚ) / length_interval
  let prob_product_pos := prob_neg^2 + prob_pos^2
  prob_product_pos

theorem probability_product_positive_is_5_div_9 :
  probability_positive_product = 5 / 9 :=
by
  sorry

end probability_product_positive_is_5_div_9_l543_54365


namespace number_of_divisible_permutations_l543_54331

def digit_list := [1, 3, 1, 1, 5, 2, 1, 5, 2]
def count_permutations (d : List ℕ) (n : ℕ) : ℕ :=
  let fact := Nat.factorial
  let number := fact 8 / (fact 3 * fact 2 * fact 1)
  number

theorem number_of_divisible_permutations : count_permutations digit_list 2 = 3360 := by
  sorry

end number_of_divisible_permutations_l543_54331


namespace fifth_stack_33_l543_54387

def cups_in_fifth_stack (a d : ℕ) : ℕ :=
a + 4 * d

theorem fifth_stack_33 
  (a : ℕ) 
  (d : ℕ) 
  (h_first_stack : a = 17) 
  (h_pattern : d = 4) : 
  cups_in_fifth_stack a d = 33 := by
  sorry

end fifth_stack_33_l543_54387


namespace num_divisible_by_7_in_range_l543_54373

theorem num_divisible_by_7_in_range (n : ℤ) (h : 1 ≤ n ∧ n ≤ 2015)
    : (∃ k, 1 ≤ k ∧ k ≤ 335 ∧ 3 ^ (6 * k) + (6 * k) ^ 3 ≡ 0 [MOD 7]) :=
sorry

end num_divisible_by_7_in_range_l543_54373


namespace squares_in_rectangle_l543_54315

theorem squares_in_rectangle (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) 
  (h4 : a ≤ 1) (h5 : b ≤ 1) (h6 : c ≤ 1) (h7 : a + b + c = 2)  : 
  a + b + c ≤ 2 := sorry

end squares_in_rectangle_l543_54315


namespace factorize_expression_l543_54388

variable (a : ℝ)

theorem factorize_expression : a^3 + 4 * a^2 + 4 * a = a * (a + 2)^2 := by
  sorry

end factorize_expression_l543_54388


namespace reciprocals_of_each_other_l543_54369

theorem reciprocals_of_each_other (a b : ℝ) (h : (a + b)^2 - (a - b)^2 = 4) : a * b = 1 :=
by 
  sorry

end reciprocals_of_each_other_l543_54369


namespace polygon_sides_14_l543_54383

theorem polygon_sides_14 (n : ℕ) (θ : ℝ) 
  (h₀ : (n - 2) * 180 - θ = 2000) :
  n = 14 :=
sorry

end polygon_sides_14_l543_54383


namespace symmetric_point_about_origin_l543_54335

theorem symmetric_point_about_origin (P Q : ℤ × ℤ) (h : P = (-2, -3)) : Q = (2, 3) :=
by
  sorry

end symmetric_point_about_origin_l543_54335


namespace find_values_f_l543_54354

open Real

noncomputable def f (ω A x : ℝ) : ℝ := 2 * sin (ω * x) * cos (ω * x) + 2 * A * (cos (ω * x))^2 - A

theorem find_values_f (θ : ℝ) (A : ℝ) (ω : ℝ) (hA : A > 0) (hω : ω = 1)
  (h1 : π / 6 < θ) (h2 : θ < π / 3) (h3 : f ω A θ = 2 / 3) :
  f ω A (π / 3 - θ) = (1 + 2 * sqrt 6) / 3 :=
  sorry

end find_values_f_l543_54354


namespace line_passes_through_fixed_point_min_area_line_eq_l543_54396

section part_one

variable (m x y : ℝ)

def line_eq := (2 * m + 1) * x + (m + 1) * y - 7 * m - 4

theorem line_passes_through_fixed_point :
  ∀ m, line_eq m 3 1 = 0 :=
sorry

end part_one

section part_two

variable (k x y : ℝ)

def line_eq_l1 (k : ℝ) := y = k * (x - 3) + 1

theorem min_area_line_eq :
  line_eq_l1 (-1/3) x y = (x + 3 * y - 6 = 0) :=
sorry

end part_two

end line_passes_through_fixed_point_min_area_line_eq_l543_54396


namespace volume_of_box_is_correct_l543_54364

def metallic_sheet_initial_length : ℕ := 48
def metallic_sheet_initial_width : ℕ := 36
def square_cut_side_length : ℕ := 8

def box_length : ℕ := metallic_sheet_initial_length - 2 * square_cut_side_length
def box_width : ℕ := metallic_sheet_initial_width - 2 * square_cut_side_length
def box_height : ℕ := square_cut_side_length

def box_volume : ℕ := box_length * box_width * box_height

theorem volume_of_box_is_correct : box_volume = 5120 := by
  sorry

end volume_of_box_is_correct_l543_54364
