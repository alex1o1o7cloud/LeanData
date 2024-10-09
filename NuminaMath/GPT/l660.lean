import Mathlib

namespace block_of_flats_l660_66070

theorem block_of_flats :
  let total_floors := 12
  let half_floors := total_floors / 2
  let apartments_per_half_floor := 6
  let max_residents_per_apartment := 4
  let total_max_residents := 264
  let apartments_on_half_floors := half_floors * apartments_per_half_floor
  ∃ (x : ℝ), 
    4 * (apartments_on_half_floors + half_floors * x) = total_max_residents ->
    x = 5 :=
sorry

end block_of_flats_l660_66070


namespace louie_monthly_payment_l660_66083

noncomputable def compound_interest_payment (P : ℝ) (r : ℝ) (n : ℕ) (t_months : ℕ) : ℝ :=
  let t_years := t_months / 12
  let A := P * (1 + r / ↑n)^(↑n * t_years)
  A / t_months

theorem louie_monthly_payment : compound_interest_payment 1000 0.10 1 3 = 444 :=
by
  sorry

end louie_monthly_payment_l660_66083


namespace quadratic_inequality_solution_set_l660_66057

theorem quadratic_inequality_solution_set (a b c : ℝ) (Δ : ℝ) (hΔ : Δ = b^2 - 4*a*c) :
  (∀ x : ℝ, a*x^2 + b*x + c > 0) ↔ (a > 0 ∧ Δ < 0) := by
  sorry

end quadratic_inequality_solution_set_l660_66057


namespace value_of_b_l660_66009

def g (x : ℝ) : ℝ := 5 * x - 6

theorem value_of_b (b : ℝ) : g b = 0 ↔ b = 6 / 5 :=
by sorry

end value_of_b_l660_66009


namespace num_students_in_section_A_l660_66011

def avg_weight (total_weight : ℕ) (total_students : ℕ) : ℕ :=
  total_weight / total_students

variables (x : ℕ) -- number of students in section A
variables (weight_A : ℕ := 40 * x) -- total weight of section A
variables (students_B : ℕ := 20)
variables (weight_B : ℕ := 20 * 35) -- total weight of section B
variables (total_weight : ℕ := weight_A + weight_B) -- total weight of the whole class
variables (total_students : ℕ := x + students_B) -- total number of students in the class
variables (avg_weight_class : ℕ := avg_weight total_weight total_students)

theorem num_students_in_section_A :
  avg_weight_class = 38 → x = 30 :=
by
-- The proof will go here
sorry

end num_students_in_section_A_l660_66011


namespace appropriate_sampling_methods_l660_66040
-- Import the entire Mathlib library for broader functionality

-- Define the conditions
def community_high_income_families : ℕ := 125
def community_middle_income_families : ℕ := 280
def community_low_income_families : ℕ := 95
def community_total_households : ℕ := community_high_income_families + community_middle_income_families + community_low_income_families

def student_count : ℕ := 12

-- Define the theorem to be proven
theorem appropriate_sampling_methods :
  (community_total_households = 500 → stratified_sampling) ∧
  (student_count = 12 → random_sampling) :=
by sorry

end appropriate_sampling_methods_l660_66040


namespace systematic_sampling_first_group_draw_l660_66007

noncomputable def index_drawn_from_group (x n : ℕ) : ℕ := x + 8 * (n - 1)

theorem systematic_sampling_first_group_draw (k : ℕ) (fifteenth_group : index_drawn_from_group k 15 = 116) :
  index_drawn_from_group k 1 = 4 := 
sorry

end systematic_sampling_first_group_draw_l660_66007


namespace circle_relationship_l660_66099

noncomputable def f : ℝ × ℝ → ℝ := sorry

variables {x y x₁ y₁ x₂ y₂ : ℝ}
variables (h₁ : f (x₁, y₁) = 0) (h₂ : f (x₂, y₂) ≠ 0)

theorem circle_relationship :
  f (x, y) - f (x₁, y₁) - f (x₂, y₂) = 0 ↔ f (x, y) = f (x₂, y₂) :=
sorry

end circle_relationship_l660_66099


namespace square_area_inscribed_in_parabola_l660_66072

-- Declare the parabola equation
def parabola (x : ℝ) : ℝ := x^2 - 10 * x + 20

-- Declare the condition that we have a square inscribed to this parabola.
def is_inscribed_square (side_length : ℝ) : Prop :=
∀ (x : ℝ), (x = 5 - side_length/2 ∨ x = 5 + side_length/2) → (parabola x = 0)

-- Proof goal
theorem square_area_inscribed_in_parabola : ∃ (side_length : ℝ), is_inscribed_square side_length ∧ side_length^2 = 400 :=
by
  sorry

end square_area_inscribed_in_parabola_l660_66072


namespace sector_perimeter_l660_66008

noncomputable def radius : ℝ := 2
noncomputable def central_angle_deg : ℝ := 120
noncomputable def expected_perimeter : ℝ := (4 / 3) * Real.pi + 4

theorem sector_perimeter (r : ℝ) (θ : ℝ) (h_r : r = radius) (h_θ : θ = central_angle_deg) :
    let arc_length := θ / 360 * 2 * Real.pi * r
    let perimeter := arc_length + 2 * r
    perimeter = expected_perimeter :=
by
  -- Skip the proof
  sorry

end sector_perimeter_l660_66008


namespace determine_A_value_l660_66061

noncomputable def solve_for_A (A B C : ℝ) : Prop :=
  (A = 1/16) ↔ 
  (∀ x : ℝ, (1 / ((x + 5) * (x - 3) * (x + 3))) = (A / (x + 5)) + (B / (x - 3)) + (C / (x + 3)))

theorem determine_A_value :
  solve_for_A (1/16) B C :=
by
  sorry

end determine_A_value_l660_66061


namespace probability_of_green_l660_66080

theorem probability_of_green : 
  ∀ (P_red P_orange P_yellow P_green : ℝ), 
    P_red = 0.25 → P_orange = 0.35 → P_yellow = 0.1 → 
    P_red + P_orange + P_yellow + P_green = 1 →
    P_green = 0.3 :=
by
  intros P_red P_orange P_yellow P_green h_red h_orange h_yellow h_total
  sorry

end probability_of_green_l660_66080


namespace sum_of_primes_146_sum_of_primes_99_l660_66015

-- Define what it means to be a prime number
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Prove that there exist tuples of (p1, p2, p3) such that their sum is 146
theorem sum_of_primes_146 :
  ∃ (p1 p2 p3 : ℕ), is_prime p1 ∧ is_prime p2 ∧ is_prime p3 ∧ p1 + p2 + p3 = 146 ∧ p1 ≤ p2 ∧ p2 ≤ p3 :=
sorry

-- Prove that there exist tuples of (p1, p2, p3) such that their sum is 99
theorem sum_of_primes_99 :
  ∃ (p1 p2 p3 : ℕ), is_prime p1 ∧ is_prime p2 ∧ is_prime p3 ∧ p1 + p2 + p3 = 99 ∧ p1 ≤ p2 ∧ p2 ≤ p3 :=
sorry

end sum_of_primes_146_sum_of_primes_99_l660_66015


namespace solve_system_of_equations_l660_66081

theorem solve_system_of_equations (x y z t : ℤ) :
  (3 * x - 2 * y + 4 * z + 2 * t = 19) ∧ (5 * x + 6 * y - 2 * z + 3 * t = 23) →
  (x = 16 * z - 18 * y - 11) ∧ (t = 28 * y - 26 * z + 26) :=
by {
  sorry
}

end solve_system_of_equations_l660_66081


namespace max_value_of_sin2A_tan2B_l660_66012

-- Definitions for the trigonometric functions and angles in triangle ABC
variables {A B C : ℝ}

-- Condition: sin^2 A + sin^2 B = sin^2 C - sqrt 2 * sin A * sin B
def condition (A B C : ℝ) : Prop :=
  (Real.sin A) ^ 2 + (Real.sin B) ^ 2 = (Real.sin C) ^ 2 - Real.sqrt 2 * (Real.sin A) * (Real.sin B)

-- Question: Find the maximum value of sin 2A * tan^2 B
noncomputable def target (A B : ℝ) : ℝ :=
  Real.sin (2 * A) * (Real.tan B) ^ 2

-- The proof statement
theorem max_value_of_sin2A_tan2B (h : condition A B C) : ∃ (max_val : ℝ), max_val = 3 - 2 * Real.sqrt 2 ∧ ∀ (x : ℝ), target A x ≤ max_val := 
sorry

end max_value_of_sin2A_tan2B_l660_66012


namespace polynomial_roots_geometric_progression_q_l660_66006

theorem polynomial_roots_geometric_progression_q :
    ∃ (a r : ℝ), (a ≠ 0) ∧ (r ≠ 0) ∧
    (a + a * r + a * r ^ 2 + a * r ^ 3 = 0) ∧
    (a ^ 4 * r ^ 6 = 16) ∧
    (a ^ 2 + (a * r) ^ 2 + (a * r ^ 2) ^ 2 + (a * r ^ 3) ^ 2 = 16) :=
by
    sorry

end polynomial_roots_geometric_progression_q_l660_66006


namespace remaining_balance_is_correct_l660_66076

def total_price (deposit amount sales_tax_rate discount_rate service_charge P : ℝ) :=
  let sales_tax := sales_tax_rate * P
  let price_after_tax := P + sales_tax
  let discount := discount_rate * price_after_tax
  let price_after_discount := price_after_tax - discount
  let total_price := price_after_discount + service_charge
  total_price

theorem remaining_balance_is_correct (deposit : ℝ) (amount_paid : ℝ) (sales_tax_rate : ℝ) (discount_rate : ℝ) (service_charge : ℝ)
  (P : ℝ) : deposit = 0.10 * P →
         amount_paid = 110 →
         sales_tax_rate = 0.15 →
         discount_rate = 0.05 →
         service_charge = 50 →
         total_price deposit amount_paid sales_tax_rate discount_rate service_charge P - amount_paid = 1141.75 :=
by
  sorry

end remaining_balance_is_correct_l660_66076


namespace largest_integer_value_n_l660_66035

theorem largest_integer_value_n (n : ℤ) : 
  (n^2 - 9 * n + 18 < 0) → n ≤ 5 := sorry

end largest_integer_value_n_l660_66035


namespace problem_statement_l660_66037

open Real

noncomputable def f (x : ℝ) : ℝ := 10^x

theorem problem_statement : f (log 2) * f (log 5) = 10 :=
by {
  -- Note: Proof is omitted as indicated in the procedure.
  sorry
}

end problem_statement_l660_66037


namespace psychiatrist_problem_l660_66065

theorem psychiatrist_problem 
  (x : ℕ)
  (h_total : 4 * 8 + x + (x + 5) = 25)
  : x = 2 := by
  sorry

end psychiatrist_problem_l660_66065


namespace positive_integer_solutions_l660_66027

theorem positive_integer_solutions :
  ∀ (a b c : ℕ), (8 * a - 5 * b)^2 + (3 * b - 2 * c)^2 + (3 * c - 7 * a)^2 = 2 → 
  (a = 3 ∧ b = 5 ∧ c = 7) ∨ (a = 12 ∧ b = 19 ∧ c = 28) :=
by
  sorry

end positive_integer_solutions_l660_66027


namespace pure_imaginary_implies_a_neg_one_l660_66084

theorem pure_imaginary_implies_a_neg_one (a : ℝ) 
  (h_pure_imaginary : ∃ (y : ℝ), z = 0 + y * I) : 
  z = a + 1 - a * I → a = -1 :=
by
  sorry

end pure_imaginary_implies_a_neg_one_l660_66084


namespace f_g_relationship_l660_66060

def f (x : ℝ) : ℝ := 3 * x ^ 2 - x + 1
def g (x : ℝ) : ℝ := 2 * x ^ 2 + x - 1

theorem f_g_relationship (x : ℝ) : f x > g x :=
by
  -- proof goes here
  sorry

end f_g_relationship_l660_66060


namespace QR_value_l660_66018

-- Given conditions for the problem
def QP : ℝ := 15
def sinQ : ℝ := 0.4

-- Define QR based on the given conditions
noncomputable def QR : ℝ := QP / sinQ

-- The theorem to prove that QR = 37.5
theorem QR_value : QR = 37.5 := 
by
  unfold QR QP sinQ
  sorry

end QR_value_l660_66018


namespace value_range_of_f_l660_66088

open Set

def f (x : ℝ) : ℝ := x^2 - 2 * x

theorem value_range_of_f : {y : ℝ | ∃ x ∈ Icc (-2 : ℝ) (2 : ℝ), f x = y} = Icc (-1 : ℝ) 8 := 
by
  sorry

end value_range_of_f_l660_66088


namespace minimum_games_for_80_percent_l660_66021

theorem minimum_games_for_80_percent :
  ∃ N : ℕ, ( ∀ N' : ℕ, (1 + N') / (5 + N') * 100 < 80 → N < N') ∧ (1 + N) / (5 + N) * 100 ≥ 80 :=
sorry

end minimum_games_for_80_percent_l660_66021


namespace first_player_wins_l660_66000

def winning_strategy (m n : ℕ) : Prop :=
  if m = 1 ∧ n = 1 then false else true

theorem first_player_wins (m n : ℕ) :
  winning_strategy m n :=
by
  sorry

end first_player_wins_l660_66000


namespace mat_inverse_sum_l660_66066

theorem mat_inverse_sum (a b c d : ℝ)
  (h1 : -2 * a + 3 * d = 1)
  (h2 : a * c - 12 = 0)
  (h3 : -8 + b * d = 0)
  (h4 : 4 * c - 4 * b = 0)
  (abc : a = 3 * Real.sqrt 2)
  (bb : b = 2 * Real.sqrt 2)
  (cc : c = 2 * Real.sqrt 2)
  (dd : d = (1 + 6 * Real.sqrt 2) / 3) :
  a + b + c + d = 9 * Real.sqrt 2 + 1 / 3 := by
  sorry

end mat_inverse_sum_l660_66066


namespace find_unit_price_B_l660_66079

variable (x : ℕ)

def unit_price_B := x
def unit_price_A := x + 50

theorem find_unit_price_B (h : (2000 / unit_price_A x = 1500 / unit_price_B x)) : unit_price_B x = 150 :=
by
  sorry

end find_unit_price_B_l660_66079


namespace larger_integer_is_72_l660_66090

theorem larger_integer_is_72 (x y : ℤ) (h1 : y = 4 * x) (h2 : (x + 6) * 3 = y) : y = 72 :=
sorry

end larger_integer_is_72_l660_66090


namespace sufficient_but_not_necessary_condition_l660_66025

theorem sufficient_but_not_necessary_condition
  (a b : ℝ) (h : a > b + 1) : (a > b) ∧ ¬ (∀ (a b : ℝ), a > b → a > b + 1) :=
by
  sorry

end sufficient_but_not_necessary_condition_l660_66025


namespace bacon_suggestions_count_l660_66077

def mashed_potatoes_suggestions : ℕ := 324
def tomatoes_suggestions : ℕ := 128
def total_suggestions : ℕ := 826

theorem bacon_suggestions_count :
  total_suggestions - (mashed_potatoes_suggestions + tomatoes_suggestions) = 374 :=
by
  sorry

end bacon_suggestions_count_l660_66077


namespace functional_equation_solution_l660_66051

theorem functional_equation_solution (f : ℝ → ℝ)
  (h : ∀ x, 2 * f (f x) = (x^2 - x) * f x + 4 - 2 * x) :
  f 2 = 2 ∧ (f 1 = 1 ∨ f 1 = 4) :=
sorry

end functional_equation_solution_l660_66051


namespace oliver_boxes_total_l660_66049

theorem oliver_boxes_total (initial_boxes : ℕ := 8) (additional_boxes : ℕ := 6) : initial_boxes + additional_boxes = 14 := 
by 
  sorry

end oliver_boxes_total_l660_66049


namespace decreased_area_of_equilateral_triangle_l660_66029

theorem decreased_area_of_equilateral_triangle 
    (A : ℝ) (hA : A = 100 * Real.sqrt 3) 
    (decrease : ℝ) (hdecrease : decrease = 6) :
    let s := Real.sqrt (4 * A / Real.sqrt 3)
    let s' := s - decrease
    let A' := (s' ^ 2 * Real.sqrt 3) / 4
    A - A' = 51 * Real.sqrt 3 :=
by
  sorry

end decreased_area_of_equilateral_triangle_l660_66029


namespace sum_of_distances_l660_66044

theorem sum_of_distances (P : ℤ × ℤ) (hP : P = (-1, -2)) :
  abs P.1 + abs P.2 = 3 :=
sorry

end sum_of_distances_l660_66044


namespace granola_bars_distribution_l660_66020

theorem granola_bars_distribution
  (total_bars : ℕ)
  (eaten_bars : ℕ)
  (num_children : ℕ)
  (remaining_bars := total_bars - eaten_bars)
  (bars_per_child := remaining_bars / num_children) :
  total_bars = 200 → eaten_bars = 80 → num_children = 6 → bars_per_child = 20 :=
by
  intros h1 h2 h3
  sorry

end granola_bars_distribution_l660_66020


namespace sum_squares_and_products_of_nonneg_reals_l660_66002

theorem sum_squares_and_products_of_nonneg_reals {x y z : ℝ} 
  (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) 
  (h1 : x^2 + y^2 + z^2 = 52) 
  (h2 : x*y + y*z + z*x = 27) : 
  x + y + z = Real.sqrt 106 := 
by 
  sorry

end sum_squares_and_products_of_nonneg_reals_l660_66002


namespace x_eq_1_is_iff_x2_minus_2x_plus_1_eq_0_l660_66014

theorem x_eq_1_is_iff_x2_minus_2x_plus_1_eq_0 :
  (x = 1 ↔ x^2 - 2 * x + 1 = 0) :=
sorry

end x_eq_1_is_iff_x2_minus_2x_plus_1_eq_0_l660_66014


namespace inverse_proportion_quadrants_l660_66067

theorem inverse_proportion_quadrants (x : ℝ) (y : ℝ) (h : y = 6/x) : 
  (x > 0 -> y > 0) ∧ (x < 0 -> y < 0) := 
sorry

end inverse_proportion_quadrants_l660_66067


namespace zeros_of_f_l660_66024

def f (x : ℝ) : ℝ := x^3 - 2 * x^2 - x + 2

theorem zeros_of_f : ∀ x, f x = 0 ↔ x = -1 ∨ x = 1 ∨ x = 2 :=
by {
  sorry
}

end zeros_of_f_l660_66024


namespace product_sum_condition_l660_66046

theorem product_sum_condition (a b c : ℝ) (h1 : a * b * c = 1) (h2 : a + b + c > (1/a) + (1/b) + (1/c)) : 
  (a > 1 ∧ b ≤ 1 ∧ c ≤ 1) ∨ (a ≤ 1 ∧ b > 1 ∧ c ≤ 1) ∨ (a ≤ 1 ∧ b ≤ 1 ∧ c > 1) :=
sorry

end product_sum_condition_l660_66046


namespace f_properties_l660_66030

noncomputable def f : ℚ × ℚ → ℚ := sorry

theorem f_properties :
  (∀ (x y z : ℚ), f (x*y, z) = f (x, z) * f (y, z)) →
  (∀ (x y z : ℚ), f (z, x*y) = f (z, x) * f (z, y)) →
  (∀ (x : ℚ), f (x, 1 - x) = 1) →
  (∀ (x : ℚ), f (x, x) = 1) ∧
  (∀ (x : ℚ), f (x, -x) = 1) ∧
  (∀ (x y : ℚ), f (x, y) * f (y, x) = 1) :=
by
  intros h1 h2 h3
  -- Proof goes here
  sorry

end f_properties_l660_66030


namespace larger_number_l660_66036

theorem larger_number (x y : ℕ) (h1 : x + y = 28) (h2 : x - y = 4) : max x y = 16 := by
  sorry

end larger_number_l660_66036


namespace toys_total_is_240_l660_66001

def number_of_toys_elder : Nat := 60
def number_of_toys_younger (toys_elder : Nat) : Nat := 3 * toys_elder
def total_number_of_toys (toys_elder toys_younger : Nat) : Nat := toys_elder + toys_younger

theorem toys_total_is_240 : total_number_of_toys number_of_toys_elder (number_of_toys_younger number_of_toys_elder) = 240 :=
by
  sorry

end toys_total_is_240_l660_66001


namespace geometric_sequence_problem_l660_66062

theorem geometric_sequence_problem
  (a : ℕ → ℝ)
  (h_pos : ∀ n, a n > 0)
  (h5 : a 5 * a 6 = 3)
  (h9 : a 9 * a 10 = 9) :
  a 7 * a 8 = 3 * Real.sqrt 3 :=
by
  sorry

end geometric_sequence_problem_l660_66062


namespace max_value_expr_l660_66058

open Real

theorem max_value_expr {x y z : ℝ} (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (h : (x / y + y / z + z / x) + (y / x + z / y + x / z) = 9) : 
  (x / y + y / z + z / x) * (y / x + z / y + x / z) = 81 / 4 :=
sorry

end max_value_expr_l660_66058


namespace total_area_painted_is_correct_l660_66052

noncomputable def barn_area_painted (width length height : ℝ) : ℝ :=
  let walls_area := 2 * (width * height + length * height) * 2
  let ceiling_and_roof_area := 2 * (width * length)
  walls_area + ceiling_and_roof_area

theorem total_area_painted_is_correct 
  (width length height : ℝ) 
  (h_w : width = 12) 
  (h_l : length = 15) 
  (h_h : height = 6) 
  : barn_area_painted width length height = 1008 :=
  by
  rw [h_w, h_l, h_h]
  -- Simplify steps omitted
  sorry

end total_area_painted_is_correct_l660_66052


namespace symmetric_poly_roots_identity_l660_66019

variable (a b c : ℝ)

theorem symmetric_poly_roots_identity (h1 : a + b + c = 6) (h2 : ab + bc + ca = 5) (h3 : abc = 1) :
  (1 / a^3) + (1 / b^3) + (1 / c^3) = 38 :=
by
  sorry

end symmetric_poly_roots_identity_l660_66019


namespace find_number_l660_66092

theorem find_number (x : ℝ) :
  (7 * (x + 10) / 5) - 5 = 44 → x = 25 :=
by
  sorry

end find_number_l660_66092


namespace smallest_sum_l660_66023

theorem smallest_sum (a b : ℕ) (h1 : 3^8 * 5^2 = a^b) (h2 : 0 < a) (h3 : 0 < b) : a + b = 407 :=
sorry

end smallest_sum_l660_66023


namespace jonathan_needs_12_bottles_l660_66042

noncomputable def fl_oz_to_liters (fl_oz : ℝ) : ℝ :=
  fl_oz / 33.8

noncomputable def liters_to_ml (liters : ℝ) : ℝ :=
  liters * 1000

noncomputable def num_bottles_needed (ml : ℝ) : ℝ :=
  ml / 150

theorem jonathan_needs_12_bottles :
  num_bottles_needed (liters_to_ml (fl_oz_to_liters 60)) = 12 := 
by
  sorry

end jonathan_needs_12_bottles_l660_66042


namespace klay_to_draymond_ratio_l660_66074

-- Let us define the points earned by each player
def draymond_points : ℕ := 12
def curry_points : ℕ := 2 * draymond_points
def kelly_points : ℕ := 9
def durant_points : ℕ := 2 * kelly_points

-- Total points of the Golden State Team
def total_points_team : ℕ := 69

theorem klay_to_draymond_ratio :
  ∃ klay_points : ℕ,
    klay_points = total_points_team - (draymond_points + curry_points + kelly_points + durant_points) ∧
    klay_points / draymond_points = 1 / 2 :=
by
  sorry

end klay_to_draymond_ratio_l660_66074


namespace volume_and_surface_area_of_prism_l660_66004

theorem volume_and_surface_area_of_prism 
  (a b c : ℝ)
  (h1 : a * b = 24)
  (h2 : b * c = 18)
  (h3 : c * a = 12) :
  (a * b * c = 72) ∧ (2 * (a * b + b * c + c * a) = 108) := by
  sorry

end volume_and_surface_area_of_prism_l660_66004


namespace percentage_of_A_l660_66054

-- Define variables and assumptions
variables (A B : ℕ)
def total_payment := 580
def payment_B := 232

-- Define the proofs of the conditions provided in the problem
axiom total_payment_eq : A + B = total_payment
axiom B_eq : B = payment_B
noncomputable def percentage_paid_to_A := (A / B) * 100

-- Theorem to prove the percentage of the payment to A compared to B
theorem percentage_of_A : percentage_paid_to_A = 150 :=
by
 sorry

end percentage_of_A_l660_66054


namespace sum_of_values_satisfying_eq_l660_66094

theorem sum_of_values_satisfying_eq (x : ℝ) :
  (x^2 - 5 * x + 5 = 16) → ∀ r s : ℝ, (r + s = 5) :=
by
  sorry  -- Proof is omitted, looking to verify the structure only.

end sum_of_values_satisfying_eq_l660_66094


namespace determine_x_l660_66078

variable (n p : ℝ)

-- Definitions based on conditions
def x (n : ℝ) : ℝ := 4 * n
def percentage_condition (n p : ℝ) : Prop := 2 * n + 3 = (p / 100) * 25

-- Statement to be proven
theorem determine_x (h : percentage_condition n p) : x n = 4 * n := by
  sorry

end determine_x_l660_66078


namespace class_average_correct_l660_66069

-- Define the constants as per the problem data
def total_students : ℕ := 30
def students_group_1 : ℕ := 24
def students_group_2 : ℕ := 6
def avg_score_group_1 : ℚ := 85 / 100  -- 85%
def avg_score_group_2 : ℚ := 92 / 100  -- 92%

-- Calculate total scores and averages based on the defined constants
def total_score_group_1 : ℚ := students_group_1 * avg_score_group_1
def total_score_group_2 : ℚ := students_group_2 * avg_score_group_2
def total_class_score : ℚ := total_score_group_1 + total_score_group_2
def class_average : ℚ := total_class_score / total_students

-- Goal: Prove that class_average is 86.4%
theorem class_average_correct : class_average = 86.4 / 100 := sorry

end class_average_correct_l660_66069


namespace trajectory_of_center_line_passes_fixed_point_l660_66032

-- Define the conditions
def pointA : ℝ × ℝ := (4, 0)
def chord_length : ℝ := 8
def pointB : ℝ × ℝ := (-3, 0)
def not_perpendicular_to_x_axis (t : ℝ) : Prop := t ≠ 0
def trajectory_eq (x y : ℝ) : Prop := y^2 = 8 * x
def line_eq (t m y x : ℝ) : Prop := x = t * y + m
def x_axis_angle_bisector (y1 x1 y2 x2 : ℝ) : Prop := (y1 / (x1 + 3)) + (y2 / (x2 + 3)) = 0

-- Prove the trajectory of the center of the moving circle is \( y^2 = 8x \)
theorem trajectory_of_center (x y : ℝ) 
  (H1: (x-4)^2 + y^2 = 4^2 + x^2) 
  (H2: trajectory_eq x y) : 
  trajectory_eq x y := sorry

-- Prove the line passes through the fixed point (3, 0)
theorem line_passes_fixed_point (t m y1 x1 y2 x2 : ℝ) 
  (Ht: not_perpendicular_to_x_axis t)
  (Hsys: ∀ y x, line_eq t m y x → trajectory_eq x y)
  (Hangle: x_axis_angle_bisector y1 x1 y2 x2) : 
  (m = 3) ∧ ∃ y, line_eq t 3 y 3 := sorry

end trajectory_of_center_line_passes_fixed_point_l660_66032


namespace min_value_expression_l660_66034

theorem min_value_expression (θ φ : ℝ) :
  ∃ (θ φ : ℝ), (3 * Real.cos θ + 6 * Real.sin φ - 10)^2 + (3 * Real.sin θ + 6 * Real.cos φ - 18)^2 = 121 :=
sorry

end min_value_expression_l660_66034


namespace tom_overall_profit_l660_66093

def initial_purchase_cost : ℝ := 20 * 3 + 30 * 5 + 15 * 10
def purchase_commission : ℝ := 0.02 * initial_purchase_cost
def total_initial_cost : ℝ := initial_purchase_cost + purchase_commission

def sale_revenue_before_commission : ℝ := 10 * 4 + 20 * 7 + 5 * 12
def sales_commission : ℝ := 0.02 * sale_revenue_before_commission
def total_sales_revenue : ℝ := sale_revenue_before_commission - sales_commission

def remaining_stock_a_value : ℝ := 10 * (3 * 2)
def remaining_stock_b_value : ℝ := 10 * (5 * 1.20)
def remaining_stock_c_value : ℝ := 10 * (10 * 0.90)
def total_remaining_value : ℝ := remaining_stock_a_value + remaining_stock_b_value + remaining_stock_c_value

def overall_profit_or_loss : ℝ := total_sales_revenue + total_remaining_value - total_initial_cost

theorem tom_overall_profit : overall_profit_or_loss = 78 := by
  sorry

end tom_overall_profit_l660_66093


namespace fuel_A_volume_l660_66028

-- Let V_A and V_B be defined as the volumes of fuel A and B respectively.
def V_A : ℝ := sorry
def V_B : ℝ := sorry

-- Given conditions:
axiom h1 : V_A + V_B = 214
axiom h2 : 0.12 * V_A + 0.16 * V_B = 30

-- Prove that the volume of fuel A added, V_A, is 106 gallons.
theorem fuel_A_volume : V_A = 106 := 
by
  sorry

end fuel_A_volume_l660_66028


namespace value_of_c_div_b_l660_66068

theorem value_of_c_div_b (a b c : ℕ) (h1 : a = 0) (h2 : a < b) (h3 : b < c) 
  (h4 : b ≠ a + 1) (h5 : (a + b + c) / 3 = 4 * b) : c / b = 11 :=
by
  sorry

end value_of_c_div_b_l660_66068


namespace find_x_value_l660_66063

theorem find_x_value (A B C x : ℝ) (hA : A = 40) (hB : B = 3 * x) (hC : C = 2 * x) (hSum : A + B + C = 180) : x = 28 :=
by
  sorry

end find_x_value_l660_66063


namespace relationship_among_f_values_l660_66071

variable (f : ℝ → ℝ)
variable (h_even : ∀ x : ℝ, f x = f (-x))
variable (h_decreasing : ∀ x₁ x₂ : ℝ, 0 ≤ x₁ → 0 ≤ x₂ → x₁ ≠ x₂ → (x₁ - x₂) * (f x₁ - f x₂) < 0)

theorem relationship_among_f_values (h₀ : 0 < 2) (h₁ : 2 < 3) :
  f 0 > f (-2) ∧ f (-2) > f 3 :=
by
  sorry

end relationship_among_f_values_l660_66071


namespace max_abs_diff_f_l660_66038

noncomputable def f (x : ℝ) : ℝ := (x + 1) ^ 2 * Real.exp x

theorem max_abs_diff_f {k x1 x2 : ℝ} (hk : -3 ≤ k ∧ k ≤ -1) 
    (hx1 : k ≤ x1 ∧ x1 ≤ k + 2) (hx2 : k ≤ x2 ∧ x2 ≤ k + 2) : 
    |f x1 - f x2| ≤ 4 * Real.exp 1 := 
sorry

end max_abs_diff_f_l660_66038


namespace minimal_withdrawals_proof_l660_66098

-- Defining the conditions
def red_marbles : ℕ := 200
def blue_marbles : ℕ := 300
def green_marbles : ℕ := 400

def max_red_withdrawal_per_time : ℕ := 1
def max_blue_withdrawal_per_time : ℕ := 2
def max_total_withdrawal_per_time : ℕ := 5

-- The target minimal number of withdrawals
def minimal_withdrawals : ℕ := 200

-- Lean statement of the proof problem
theorem minimal_withdrawals_proof :
  ∃ (w : ℕ), w = minimal_withdrawals ∧ 
    (∀ n, n ≤ w →
      (n = 200 ∧ 
       (∀ r b g, r ≤ max_red_withdrawal_per_time ∧ b ≤ max_blue_withdrawal_per_time ∧ (r + b + g) ≤ max_total_withdrawal_per_time))) :=
sorry

end minimal_withdrawals_proof_l660_66098


namespace solve_for_x_l660_66022

theorem solve_for_x : ∃ x : ℝ, 2 * ((x - 1) - (2 * x + 1)) = 6 ∧ x = -5 := by
  use -5
  sorry

end solve_for_x_l660_66022


namespace find_value_of_f_at_1_l660_66039

noncomputable def f (x : ℝ) : ℝ := sorry 

theorem find_value_of_f_at_1 (f : ℝ → ℝ)
  (h : ∀ x : ℝ, 2 * f x - f (- x) = 3 * x + 1) : f 1 = 2 :=
by
  sorry

end find_value_of_f_at_1_l660_66039


namespace find_S9_l660_66096

-- Setting up basic definitions for arithmetic sequence and the sum of its terms
def arithmetic_seq (a d : ℤ) (n : ℕ) : ℤ := a + (n - 1) * d
def sum_arithmetic_seq (a d : ℤ) (n : ℕ) : ℤ := n * (a + arithmetic_seq a d n) / 2

-- Given conditions
variables (a d : ℤ)
axiom h : 2 * arithmetic_seq a d 3 = 3 + a

-- Theorem to prove
theorem find_S9 : sum_arithmetic_seq a d 9 = 27 :=
by {
  sorry
}

end find_S9_l660_66096


namespace min_moves_to_balance_stacks_l660_66010

theorem min_moves_to_balance_stacks :
  let stack1 := 9
  let stack2 := 7
  let stack3 := 5
  let stack4 := 10
  let target := 8
  let total_coins := stack1 + stack2 + stack3 + stack4
  total_coins = 31 →
  ∃ moves, moves = 11 ∧
    (stack1 + 3 * moves = target) ∧
    (stack2 + 3 * moves = target) ∧
    (stack3 + 3 * moves = target) ∧
    (stack4 + 3 * moves = target) :=
sorry

end min_moves_to_balance_stacks_l660_66010


namespace jerry_total_hours_at_field_l660_66087
-- Import the entire necessary library

-- Lean statement of the problem
theorem jerry_total_hours_at_field 
  (games_per_daughter : ℕ)
  (practice_hours_per_game : ℕ)
  (game_duration : ℕ)
  (daughters : ℕ)
  (h1: games_per_daughter = 8)
  (h2: practice_hours_per_game = 4)
  (h3: game_duration = 2)
  (h4: daughters = 2)
 : (game_duration * games_per_daughter * daughters + practice_hours_per_game * games_per_daughter * daughters) = 96 :=
by
  -- Proof not required, so we skip it with sorry
  sorry

end jerry_total_hours_at_field_l660_66087


namespace total_amount_is_20_yuan_60_cents_l660_66043

-- Conditions
def ten_yuan_note : ℕ := 10
def five_yuan_notes : ℕ := 2 * 5
def twenty_cent_coins : ℕ := 3 * 20

-- Total amount calculation
def total_yuan : ℕ := ten_yuan_note + five_yuan_notes
def total_cents : ℕ := twenty_cent_coins

-- Conversion rates
def yuan_per_cent : ℕ := 100
def total_cents_in_yuan : ℕ := total_cents / yuan_per_cent
def remaining_cents : ℕ := total_cents % yuan_per_cent

-- Proof statement
theorem total_amount_is_20_yuan_60_cents : total_yuan = 20 ∧ total_cents_in_yuan = 0 ∧ remaining_cents = 60 :=
by
  sorry

end total_amount_is_20_yuan_60_cents_l660_66043


namespace converse_proposition_l660_66005

-- Define the propositions p and q
variables (p q : Prop)

-- State the problem as a theorem
theorem converse_proposition (p q : Prop) : (q → p) ↔ ¬p → ¬q ∧ ¬q → ¬p ∧ (p → q) := 
by 
  sorry

end converse_proposition_l660_66005


namespace not_all_perfect_squares_l660_66091

noncomputable def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

theorem not_all_perfect_squares (x : ℕ) (hx : x > 0) :
  ¬ (is_perfect_square (2 * x - 1) ∧ is_perfect_square (5 * x - 1) ∧ is_perfect_square (13 * x - 1)) :=
by
  sorry

end not_all_perfect_squares_l660_66091


namespace collinear_c1_c2_l660_66047

def vec3 := (ℝ × ℝ × ℝ)

def a : vec3 := (8, 3, -1)
def b : vec3 := (4, 1, 3)

def c1 : vec3 := (2 * 8 - 4, 2 * 3 - 1, 2 * (-1) - 3) -- (12, 5, -5)
def c2 : vec3 := (2 * 4 - 4 * 8, 2 * 1 - 4 * 3, 2 * 3 - 4 * (-1)) -- (-24, -10, 10)

theorem collinear_c1_c2 : ∃ γ : ℝ, c1 = (γ * -24, γ * -10, γ * 10) :=
  sorry

end collinear_c1_c2_l660_66047


namespace percentage_y_less_than_x_l660_66041

variable (x y : ℝ)
variable (h : x = 12 * y)

theorem percentage_y_less_than_x :
  (11 / 12) * 100 = 91.67 := by
  sorry

end percentage_y_less_than_x_l660_66041


namespace levels_for_blocks_l660_66075

theorem levels_for_blocks (S : ℕ → ℕ) (n : ℕ) (h1 : S n = n * (n + 1)) (h2 : S 10 = 110) : n = 10 :=
by {
  sorry
}

end levels_for_blocks_l660_66075


namespace harmonic_power_identity_l660_66059

open Real

theorem harmonic_power_identity (a b c : ℝ) (n : ℕ) (hn : n % 2 = 1) 
(h : (1 / a + 1 / b + 1 / c) = 1 / (a + b + c)) :
  (1 / (a ^ n) + 1 / (b ^ n) + 1 / (c ^ n) = 1 / (a ^ n + b ^ n + c ^ n)) :=
sorry

end harmonic_power_identity_l660_66059


namespace problem_equivalent_statement_l660_66089

-- Define the operations provided in the problem
inductive Operation
| add
| sub
| mul
| div

open Operation

-- Represents the given equation with the specified operation
def applyOperation (op : Operation) (a b : ℕ) : ℕ :=
  match op with
  | add => a + b
  | sub => a - b
  | mul => a * b
  | div => a / b

theorem problem_equivalent_statement : 
  (∀ (op : Operation), applyOperation op 8 2 - 5 + 7 - (3^2 - 4) ≠ 6) → (¬ ∃ op : Operation, applyOperation op 8 2 = 9) := 
by
  sorry

end problem_equivalent_statement_l660_66089


namespace rachel_fathers_age_when_rachel_is_25_l660_66053

theorem rachel_fathers_age_when_rachel_is_25 (R G M F Y : ℕ) 
  (h1 : R = 12)
  (h2 : G = 7 * R)
  (h3 : M = G / 2)
  (h4 : F = M + 5)
  (h5 : Y = 25 - R) : 
  F + Y = 60 :=
by sorry

end rachel_fathers_age_when_rachel_is_25_l660_66053


namespace clock_rings_in_a_day_l660_66097

-- Define the conditions
def rings_every_3_hours : ℕ := 3
def first_ring : ℕ := 1 -- This is 1 A.M. in our problem
def total_hours_in_day : ℕ := 24

-- Define the theorem
theorem clock_rings_in_a_day (n_rings : ℕ) : 
  (∀ n : ℕ, n_rings = total_hours_in_day / rings_every_3_hours + 1) :=
by
  -- use sorry to skip the proof
  sorry

end clock_rings_in_a_day_l660_66097


namespace problem_1_problem_2_l660_66003

def f (x : ℝ) : ℝ := |x + 1| - |x - 2|

theorem problem_1 : {x : ℝ | f x ≥ 1} = {x : ℝ | x ≥ 1} :=
  sorry

theorem problem_2 (m : ℝ) : (∃ x : ℝ, f x ≥ x^2 - x + m) → m ≤ 5/4 :=
  sorry

end problem_1_problem_2_l660_66003


namespace minimum_questions_two_l660_66056

structure Person :=
  (is_liar : Bool)

structure Decagon :=
  (people : Fin 10 → Person)

def minimumQuestionsNaive (d : Decagon) : Nat :=
  match d with 
  -- add the logic here later
  | _ => sorry

theorem minimum_questions_two (d : Decagon) : minimumQuestionsNaive d = 2 :=
  sorry

end minimum_questions_two_l660_66056


namespace option_d_correct_l660_66026

theorem option_d_correct (x y : ℝ) : -4 * x * y + 3 * x * y = -1 * x * y := 
by {
  sorry
}

end option_d_correct_l660_66026


namespace ratio_of_side_lengths_l660_66095

theorem ratio_of_side_lengths (a b c : ℕ) (h : a * a * b * b = 18 * c * c * 50 * c * c) :
  (12 = 1800000) ->  (15 = 1500) -> (10 > 0):=
by
  sorry

end ratio_of_side_lengths_l660_66095


namespace find_a_l660_66017

def f : ℝ → ℝ := sorry

theorem find_a (x a : ℝ) 
  (h1 : ∀ x, f ((1/2)*x - 1) = 2*x - 5)
  (h2 : f a = 6) : 
  a = 7/4 := 
by 
  sorry

end find_a_l660_66017


namespace part_a_part_b_l660_66031

def N := 10^40

def is_divisor (a b : ℕ) : Prop := b % a = 0

def is_perfect_square (a : ℕ) : Prop := ∃ m : ℕ, m * m = a

def is_perfect_cube (a : ℕ) : Prop := ∃ m : ℕ, m * m * m = a

def is_perfect_power (a : ℕ) : Prop := ∃ (m n : ℕ), n > 1 ∧ a = m^n

def num_divisors_not_square_or_cube (n : ℕ) : ℕ := sorry -- This should calculate the number of divisors that are neither perfect squares nor perfect cubes

def num_divisors_not_in_form_m_n (n : ℕ) : ℕ := sorry -- This should calculate the number of divisors that cannot be represented in the form m^n where n > 1

theorem part_a : num_divisors_not_square_or_cube N = 1093 := by
  sorry

theorem part_b : num_divisors_not_in_form_m_n N = 981 := by
  sorry

end part_a_part_b_l660_66031


namespace int_modulo_l660_66085

theorem int_modulo (n : ℤ) (h1 : 0 ≤ n) (h2 : n < 17) (h3 : 38574 ≡ n [ZMOD 17]) : n = 1 :=
by
  sorry

end int_modulo_l660_66085


namespace solution_set_ineq_l660_66048

theorem solution_set_ineq (x : ℝ) : 
  x * (x + 2) > 0 → abs x < 1 → 0 < x ∧ x < 1 := by
sorry

end solution_set_ineq_l660_66048


namespace simplify_expression_l660_66045

variable {R : Type} [AddCommGroup R] [Module ℤ R]

theorem simplify_expression (a b : R) :
  (25 • a + 70 • b) + (15 • a + 34 • b) - (12 • a + 55 • b) = 28 • a + 49 • b :=
by sorry

end simplify_expression_l660_66045


namespace fractional_eq_solution_range_l660_66016

theorem fractional_eq_solution_range (m : ℝ) : 
  (∃ x : ℝ, (2 * x - m) / (x + 1) = 3 ∧ x > 0) ↔ m < -3 :=
by
  sorry

end fractional_eq_solution_range_l660_66016


namespace small_circle_area_l660_66082

theorem small_circle_area (r R : ℝ) (n : ℕ)
  (h_n : n = 6)
  (h_area_large : π * R^2 = 120)
  (h_relation : r = R / 2) :
  π * r^2 = 40 :=
by
  sorry

end small_circle_area_l660_66082


namespace total_cost_is_9_43_l660_66073

def basketball_game_cost : ℝ := 5.20
def racing_game_cost : ℝ := 4.23
def total_cost : ℝ := basketball_game_cost + racing_game_cost

theorem total_cost_is_9_43 : total_cost = 9.43 := by
  sorry

end total_cost_is_9_43_l660_66073


namespace aram_fraction_of_fine_l660_66033

theorem aram_fraction_of_fine (F : ℝ) (H1 : Joe_paid = (1/4)*F + 3)
  (H2 : Peter_paid = (1/3)*F - 3)
  (H3 : Aram_paid = (1/2)*F - 4)
  (H4 : Joe_paid + Peter_paid + Aram_paid = F) : 
  Aram_paid / F = 5 / 12 := 
sorry

end aram_fraction_of_fine_l660_66033


namespace cubic_inequality_solution_l660_66050

theorem cubic_inequality_solution :
  ∀ x : ℝ, (x + 1) * (x + 2)^2 ≤ 0 ↔ -2 ≤ x ∧ x ≤ -1 := 
by 
  sorry

end cubic_inequality_solution_l660_66050


namespace min_omega_value_l660_66064

noncomputable def f (x : ℝ) (ω : ℝ) (φ : ℝ) := 2 * Real.sin (ω * x + φ)

theorem min_omega_value
  (ω : ℝ) (φ : ℝ)
  (hω : ω > 0)
  (h1 : f (π / 3) ω φ = 0)
  (h2 : f (π / 2) ω φ = 2) :
  ω = 3 :=
sorry

end min_omega_value_l660_66064


namespace find_angle_B_l660_66013

-- Define the parallel lines and angles
variables (l m : ℝ) -- Representing the lines as real numbers for simplicity
variables (A C B : ℝ) -- Representing the angles as real numbers

-- The conditions
def parallel_lines (l m : ℝ) : Prop := l = m
def angle_A (A : ℝ) : Prop := A = 100
def angle_C (C : ℝ) : Prop := C = 60

-- The theorem stating that, given the conditions, the angle B is 120 degrees
theorem find_angle_B (l m : ℝ) (A C B : ℝ) 
  (h1 : parallel_lines l m) 
  (h2 : angle_A A) 
  (h3 : angle_C C) : B = 120 :=
sorry

end find_angle_B_l660_66013


namespace calculate_adults_in_play_l660_66086

theorem calculate_adults_in_play :
  ∃ A : ℕ, (11 * A = 49 + 50) := sorry

end calculate_adults_in_play_l660_66086


namespace larger_number_is_391_l660_66055

theorem larger_number_is_391 (A B : ℕ) 
  (hcf : ∀ n : ℕ, n ∣ A ∧ n ∣ B ↔ n = 23)
  (lcm_factors : ∃ C D : ℕ, lcm A B = 23 * 13 * 17 ∧ C = 13 ∧ D = 17) :
  max A B = 391 :=
sorry

end larger_number_is_391_l660_66055
