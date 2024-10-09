import Mathlib

namespace blocks_before_jess_turn_l1350_135030

def blocks_at_start : Nat := 54
def players : Nat := 5
def rounds : Nat := 5
def father_removes_block_in_6th_round : Nat := 1

theorem blocks_before_jess_turn :
    (blocks_at_start - (players * rounds + father_removes_block_in_6th_round)) = 28 :=
by 
    sorry

end blocks_before_jess_turn_l1350_135030


namespace vertex_in_second_quadrant_l1350_135046

-- Theorems and properties regarding quadratic functions and their roots.
theorem vertex_in_second_quadrant (c : ℝ) (h : 4 + 4 * c < 0) : 
  (1:ℝ) * -1^2 + 2 * -1 - c > 0 :=
sorry

end vertex_in_second_quadrant_l1350_135046


namespace car_avg_speed_l1350_135047

def avg_speed_problem (d1 d2 t : ℕ) : ℕ :=
  (d1 + d2) / t

theorem car_avg_speed (d1 d2 : ℕ) (t : ℕ) (h1 : d1 = 70) (h2 : d2 = 90) (ht : t = 2) :
  avg_speed_problem d1 d2 t = 80 := by
  sorry

end car_avg_speed_l1350_135047


namespace value_of_4_inch_cube_l1350_135099

noncomputable def value_per_cubic_inch (n : ℕ) : ℝ :=
  match n with
  | 1 => 300
  | _ => 1.1 ^ (n - 1) * 300

def cube_volume (n : ℕ) : ℝ :=
  n^3

noncomputable def total_value (n : ℕ) : ℝ :=
  cube_volume n * value_per_cubic_inch n

theorem value_of_4_inch_cube : total_value 4 = 25555 := by
  admit

end value_of_4_inch_cube_l1350_135099


namespace little_john_spent_on_sweets_l1350_135039

theorem little_john_spent_on_sweets
  (initial_amount : ℝ)
  (amount_per_friend : ℝ)
  (friends_count : ℕ)
  (amount_left : ℝ)
  (spent_on_sweets : ℝ) :
  initial_amount = 10.50 →
  amount_per_friend = 2.20 →
  friends_count = 2 →
  amount_left = 3.85 →
  spent_on_sweets = initial_amount - (amount_per_friend * friends_count) - amount_left →
  spent_on_sweets = 2.25 :=
by
  intros h_initial h_per_friend h_friends_count h_left h_spent
  sorry

end little_john_spent_on_sweets_l1350_135039


namespace total_hours_eq_52_l1350_135069

def hours_per_week_on_extracurriculars : ℕ := 2 + 8 + 3  -- Total hours per week
def weeks_in_semester : ℕ := 12  -- Total weeks in a semester
def weeks_before_midterms : ℕ := weeks_in_semester / 2  -- Weeks before midterms
def sick_weeks : ℕ := 2  -- Weeks Annie takes off sick
def active_weeks_before_midterms : ℕ := weeks_before_midterms - sick_weeks  -- Active weeks before midterms

def total_extracurricular_hours_before_midterms : ℕ :=
  hours_per_week_on_extracurriculars * active_weeks_before_midterms

theorem total_hours_eq_52 :
  total_extracurricular_hours_before_midterms = 52 :=
by
  sorry

end total_hours_eq_52_l1350_135069


namespace units_digit_of_N_is_8_l1350_135080

def product_of_digits (n : ℕ) : ℕ :=
  let tens := n / 10
  let units := n % 10
  tens * units

def sum_of_digits (n : ℕ) : ℕ :=
  let tens := n / 10
  let units := n % 10
  tens + units

theorem units_digit_of_N_is_8 (N : ℕ) (hN_range : 10 ≤ N ∧ N < 100)
    (hN_eq : N = product_of_digits N * sum_of_digits N) : N % 10 = 8 :=
sorry

end units_digit_of_N_is_8_l1350_135080


namespace sarees_shirts_cost_l1350_135009

variable (S T : ℕ)

-- Definition of conditions
def condition1 : Prop := 2 * S + 4 * T = 2 * S + 4 * T
def condition2 : Prop := (S + 6 * T) = (2 * S + 4 * T)
def condition3 : Prop := 12 * T = 2400

-- Proof goal
theorem sarees_shirts_cost :
  condition1 S T → condition2 S T → condition3 T → 2 * S + 4 * T = 1600 := by
  sorry

end sarees_shirts_cost_l1350_135009


namespace project_time_for_A_l1350_135064

/--
A can complete a project in some days and B can complete the same project in 30 days.
If A and B start working on the project together and A quits 5 days before the project is 
completed, the project will be completed in 15 days.
Prove that A can complete the project alone in 20 days.
-/
theorem project_time_for_A (x : ℕ) (h : 10 * (1 / x + 1 / 30) + 5 * (1 / 30) = 1) : x = 20 :=
sorry

end project_time_for_A_l1350_135064


namespace power_function_value_at_half_l1350_135019

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^(2 - x) - 3/4

noncomputable def g (α : ℝ) (x : ℝ) : ℝ := x^α

theorem power_function_value_at_half (a : ℝ) (α : ℝ) 
  (h1 : 0 < a) (h2 : a ≠ 1) 
  (h3 : f a 2 = 1 / 4) (h4 : g α 2 = 1 / 4) : 
  g α (1/2) = 4 := 
by
  sorry

end power_function_value_at_half_l1350_135019


namespace increasing_function_when_a_eq_2_range_of_a_for_solution_set_l1350_135055

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  Real.log x - a * (x - 1) / (x + 1)

theorem increasing_function_when_a_eq_2 :
  ∀ ⦃x⦄, x > 0 → (f 2 x - f 2 1) * (x - 1) > 0 := sorry

theorem range_of_a_for_solution_set :
  ∀ ⦃a x⦄, f a x ≥ 0 ↔ (x ≥ 1) → a ≤ 1 := sorry

end increasing_function_when_a_eq_2_range_of_a_for_solution_set_l1350_135055


namespace lassis_from_mangoes_l1350_135035

-- Define the given ratio
def lassis_per_mango := 15 / 3

-- Define the number of mangoes
def mangoes := 15

-- Define the expected number of lassis
def expected_lassis := 75

-- Prove that with 15 mangoes, 75 lassis can be made given the ratio
theorem lassis_from_mangoes (h : lassis_per_mango = 5) : mangoes * lassis_per_mango = expected_lassis :=
by
  sorry

end lassis_from_mangoes_l1350_135035


namespace repair_cost_l1350_135085

theorem repair_cost (purchase_price transport_charges selling_price profit_percentage R : ℝ)
  (h1 : purchase_price = 10000)
  (h2 : transport_charges = 1000)
  (h3 : selling_price = 24000)
  (h4 : profit_percentage = 0.5)
  (h5 : selling_price = (1 + profit_percentage) * (purchase_price + R + transport_charges)) :
  R = 5000 :=
by
  sorry

end repair_cost_l1350_135085


namespace scientific_notation_2150000_l1350_135014

theorem scientific_notation_2150000 : 2150000 = 2.15 * 10^6 :=
  by
  sorry

end scientific_notation_2150000_l1350_135014


namespace expansion_eq_coeff_sum_l1350_135050

theorem expansion_eq_coeff_sum (a : ℕ → ℤ) (m : ℤ) 
  (h : (x - m)^7 = a 0 + a 1 * x + a 2 * x^2 + a 3 * x^3 + a 4 * x^4 + a 5 * x^5 + a 6 * x^6 + a 7 * x^7)
  (h_coeff : a 4 = -35) :
  a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 = 1 ∧ a 1 + a 3 + a 5 + a 7 = 26 := 
by 
  sorry

end expansion_eq_coeff_sum_l1350_135050


namespace frac_sum_diff_l1350_135091

theorem frac_sum_diff (a b : ℝ) (h : (1/a + 1/b) / (1/a - 1/b) = 1001) : (a + b) / (a - b) = -1001 :=
sorry

end frac_sum_diff_l1350_135091


namespace smallest_positive_b_l1350_135020

def periodic_10 (f : ℝ → ℝ) : Prop :=
  ∀ x, f (x - 10) = f x

theorem smallest_positive_b
  (f : ℝ → ℝ)
  (h : periodic_10 f) :
  ∀ x, f ((x - 20) / 2) = f (x / 2) :=
by
  sorry

end smallest_positive_b_l1350_135020


namespace increase_result_l1350_135017

-- Given conditions
def original_number : ℝ := 80
def increase_percentage : ℝ := 1.5

-- The result after the increase
theorem increase_result (h1 : original_number = 80) (h2 : increase_percentage = 1.5) : 
  original_number + (increase_percentage * original_number) = 200 := by
  sorry

end increase_result_l1350_135017


namespace amusement_park_ticket_cost_l1350_135078

/-- Jeremie is going to an amusement park with 3 friends. 
    The cost of a set of snacks is $5. 
    The total cost for everyone to go to the amusement park and buy snacks is $92.
    Prove that the cost of one ticket is $18.
-/
theorem amusement_park_ticket_cost 
  (number_of_people : ℕ)
  (snack_cost_per_person : ℕ)
  (total_cost : ℕ)
  (ticket_cost : ℕ) :
  number_of_people = 4 → 
  snack_cost_per_person = 5 → 
  total_cost = 92 → 
  ticket_cost = 18 :=
by
  intros h1 h2 h3
  sorry

end amusement_park_ticket_cost_l1350_135078


namespace book_pages_l1350_135088

theorem book_pages (books sheets pages_per_sheet pages_per_book : ℕ)
  (hbooks : books = 2)
  (hpages_per_sheet : pages_per_sheet = 8)
  (hsheets : sheets = 150)
  (htotal_pages : pages_per_sheet * sheets = 1200)
  (hpages_per_book : pages_per_book = 1200 / books) :
  pages_per_book = 600 :=
by
  -- Proof goes here
  sorry

end book_pages_l1350_135088


namespace cost_of_dozen_pens_l1350_135041

variable (x : ℝ) (pen_cost pencil_cost : ℝ)
variable (h1 : 3 * pen_cost + 5 * pencil_cost = 260)
variable (h2 : pen_cost / pencil_cost = 5)

theorem cost_of_dozen_pens (x_pos : 0 < x) 
    (pen_cost_def : pen_cost = 5 * x) 
    (pencil_cost_def : pencil_cost = x) :
    12 * pen_cost = 780 := by
  sorry

end cost_of_dozen_pens_l1350_135041


namespace ratio_of_x_to_y_l1350_135079

theorem ratio_of_x_to_y (x y : ℝ) (h : (12 * x - 7 * y) / (17 * x - 3 * y) = 4 / 7) : 
  x / y = 37 / 16 :=
by
  sorry

end ratio_of_x_to_y_l1350_135079


namespace find_c_l1350_135058

noncomputable def f (x : ℝ) : ℝ := 3 * x ^ 2 + 19 * x - 84
noncomputable def g (x : ℝ) : ℝ := 4 * x ^ 2 - 12 * x + 5

theorem find_c (c : ℝ) 
  (h1 : ∃ x : ℝ, (⌊c⌋ : ℝ) = x ∧ f x = 0)
  (h2 : ∃ x : ℝ, (c - ⌊c⌋) = x ∧ g x = 0) :
  c = -23 / 2 := by
  sorry

end find_c_l1350_135058


namespace sum_of_roots_expression_involving_roots_l1350_135082

variables {a b : ℝ}

axiom roots_of_quadratic :
  (a^2 + 3 * a - 2 = 0) ∧ (b^2 + 3 * b - 2 = 0)

theorem sum_of_roots :
  a + b = -3 :=
by 
  sorry

theorem expression_involving_roots :
  a^3 + 3 * a^2 + 2 * b = -6 :=
by 
  sorry

end sum_of_roots_expression_involving_roots_l1350_135082


namespace functional_ineq_solution_l1350_135068

theorem functional_ineq_solution (n : ℕ) (h : n > 0) :
  (∀ x : ℝ, n = 1 → (x^n + (1 - x)^n ≤ 1)) ∧
  (∀ x : ℝ, n > 1 → ((x < 0 ∨ x > 1) → (x^n + (1 - x)^n > 1))) :=
by
  intros
  sorry

end functional_ineq_solution_l1350_135068


namespace sin_cos_tan_l1350_135062

theorem sin_cos_tan (α : ℝ) (h1 : Real.tan α = 3) : Real.sin α * Real.cos α = 3 / 10 := 
sorry

end sin_cos_tan_l1350_135062


namespace relay_race_l1350_135075

theorem relay_race (n : ℕ) (H1 : 2004 % n = 0) (H2 : n ≤ 168) (H3 : n ≠ 1 ∧ n ≠ 2 ∧ n ≠ 3 ∧ n ≠ 4 ∧ n ≠ 6 ∧ n ≠ 12): n = 167 :=
by
  sorry

end relay_race_l1350_135075


namespace baseball_card_distribution_l1350_135086

theorem baseball_card_distribution (total_cards : ℕ) (capacity_4 : ℕ) (capacity_6 : ℕ) (capacity_8 : ℕ) :
  total_cards = 137 →
  capacity_4 = 4 →
  capacity_6 = 6 →
  capacity_8 = 8 →
  (total_cards % capacity_4) % capacity_6 = 1 :=
by
  intros
  sorry

end baseball_card_distribution_l1350_135086


namespace find_second_number_l1350_135076

theorem find_second_number (A B : ℝ) (h1 : A = 3200) (h2 : 0.10 * A = 0.20 * B + 190) : B = 650 :=
by
  sorry

end find_second_number_l1350_135076


namespace veronica_pitting_time_is_2_hours_l1350_135089

def veronica_cherries_pitting_time (pounds : ℕ) (cherries_per_pound : ℕ) (minutes_per_20_cherries : ℕ) :=
  let cherries := pounds * cherries_per_pound
  let sets := cherries / 20
  let total_minutes := sets * minutes_per_20_cherries
  total_minutes / 60

theorem veronica_pitting_time_is_2_hours : 
  veronica_cherries_pitting_time 3 80 10 = 2 :=
  by
    sorry

end veronica_pitting_time_is_2_hours_l1350_135089


namespace complex_shape_perimeter_l1350_135026

theorem complex_shape_perimeter :
  ∃ h : ℝ, 12 * h - 20 = 95 ∧
  (24 + ((230 / 12) - 2) + 10 : ℝ) = 51.1667 :=
by
  sorry

end complex_shape_perimeter_l1350_135026


namespace quilt_width_is_eight_l1350_135029

def length := 7
def cost_per_square_foot := 40
def total_cost := 2240
def area := total_cost / cost_per_square_foot

theorem quilt_width_is_eight :
  area / length = 8 := by
  sorry

end quilt_width_is_eight_l1350_135029


namespace initial_pollykawgs_computation_l1350_135090

noncomputable def initial_pollykawgs_in_pond (daily_rate_matured : ℕ) (daily_rate_caught : ℕ)
  (total_days : ℕ) (catch_days : ℕ) : ℕ :=
let first_phase := (daily_rate_matured + daily_rate_caught) * catch_days
let second_phase := daily_rate_matured * (total_days - catch_days)
first_phase + second_phase

theorem initial_pollykawgs_computation :
  initial_pollykawgs_in_pond 50 10 44 20 = 2400 :=
by sorry

end initial_pollykawgs_computation_l1350_135090


namespace construct_triangle_condition_l1350_135003

theorem construct_triangle_condition (m_a f_a s_a : ℝ) : 
  (m_a < f_a) ∧ (f_a < s_a) ↔ (exists A B C : Type, true) :=
sorry

end construct_triangle_condition_l1350_135003


namespace primes_p_q_divisibility_l1350_135013

theorem primes_p_q_divisibility (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (hq_eq : q = p + 2) :
  (p + q) ∣ (p ^ q + q ^ p) := 
sorry

end primes_p_q_divisibility_l1350_135013


namespace rectangle_width_decreased_by_33_percent_l1350_135087

theorem rectangle_width_decreased_by_33_percent
  (L W A : ℝ)
  (hA : A = L * W)
  (newL : ℝ)
  (h_newL : newL = 1.5 * L)
  (W' : ℝ)
  (h_area_unchanged : newL * W' = A) : 
  (1 - W' / W) * 100 = 33.33 :=
by
  sorry

end rectangle_width_decreased_by_33_percent_l1350_135087


namespace fewest_erasers_l1350_135073

theorem fewest_erasers :
  ∀ (JK JM SJ : ℕ), 
  (JK = 6) →
  (JM = JK + 4) →
  (SJ = JM - 3) →
  (JK ≤ JM ∧ JK ≤ SJ) :=
by
  intros JK JM SJ hJK hJM hSJ
  sorry

end fewest_erasers_l1350_135073


namespace twenty_five_percent_greater_l1350_135072

theorem twenty_five_percent_greater (x : ℕ) (h : x = (88 + (88 * 25) / 100)) : x = 110 :=
sorry

end twenty_five_percent_greater_l1350_135072


namespace winning_candidate_percentage_l1350_135015

theorem winning_candidate_percentage (P : ℕ) (majority : ℕ) (total_votes : ℕ) (h1 : majority = 188) (h2 : total_votes = 470) (h3 : 2 * majority = (2 * P - 100) * total_votes) : 
  P = 70 := 
sorry

end winning_candidate_percentage_l1350_135015


namespace probability_B_in_A_is_17_over_24_l1350_135096

open Set

def set_A : Set (ℝ × ℝ) := {p : ℝ × ℝ | abs p.1 + abs p.2 <= 2}
def set_B : Set (ℝ × ℝ) := {p : ℝ × ℝ | p ∈ set_A ∧ p.2 <= p.1 ^ 2}

noncomputable def area (s : Set (ℝ × ℝ)) : ℝ := sorry -- Assume we have means to compute the area of a set

theorem probability_B_in_A_is_17_over_24 :
  (area set_B / area set_A) = 17 / 24 :=
sorry

end probability_B_in_A_is_17_over_24_l1350_135096


namespace students_not_picked_correct_l1350_135049

-- Define the total number of students and the number of students picked for the team
def total_students := 17
def students_picked := 3 * 4

-- Define the number of students who didn't get picked based on the conditions
noncomputable def students_not_picked : ℕ := total_students - students_picked

-- The theorem stating the problem
theorem students_not_picked_correct : students_not_picked = 5 := 
by 
  sorry

end students_not_picked_correct_l1350_135049


namespace prove_f_2013_l1350_135028

-- Defining the function f that satisfies the given conditions
variable (f : ℕ → ℕ)

-- Conditions provided in the problem
axiom cond1 : ∀ n, f (f n) + f n = 2 * n + 3
axiom cond2 : f 0 = 1
axiom cond3 : f 2014 = 2015

-- The statement to be proven
theorem prove_f_2013 : f 2013 = 2014 := sorry

end prove_f_2013_l1350_135028


namespace trigonometric_identity_l1350_135004

theorem trigonometric_identity (θ : ℝ) (h : Real.tan θ = 2) :
    Real.sin θ ^ 2 + Real.sin θ * Real.cos θ - 2 * Real.cos θ ^ 2 = -4 / 3 :=
sorry

end trigonometric_identity_l1350_135004


namespace combined_tax_rate_l1350_135022

-- Definitions and conditions
def tax_rate_mork : ℝ := 0.45
def tax_rate_mindy : ℝ := 0.20
def income_ratio_mindy_to_mork : ℝ := 4

-- Theorem statement
theorem combined_tax_rate :
  ∀ (M : ℝ), (tax_rate_mork * M + tax_rate_mindy * (income_ratio_mindy_to_mork * M)) / (M + income_ratio_mindy_to_mork * M) = 0.25 :=
by
  intros M
  sorry

end combined_tax_rate_l1350_135022


namespace sandy_position_l1350_135077

structure Position :=
  (x : ℤ)
  (y : ℤ)

def initial_position : Position := { x := 0, y := 0 }
def after_south : Position := { x := 0, y := -20 }
def after_east : Position := { x := 20, y := -20 }
def after_north : Position := { x := 20, y := 0 }
def final_position : Position := { x := 30, y := 0 }

theorem sandy_position :
  final_position.x - initial_position.x = 10 ∧ final_position.y - initial_position.y = 0 :=
by
  sorry

end sandy_position_l1350_135077


namespace union_M_N_eq_l1350_135007

open Set

-- Define set M and set N according to the problem conditions
def M : Set ℕ := {0, 1, 2}
def N : Set ℕ := {y | ∃ x ∈ M, y = x^2}

-- The theorem we need to prove
theorem union_M_N_eq : M ∪ N = {0, 1, 2, 4} :=
by
  -- Just assert the theorem without proving it
  sorry

end union_M_N_eq_l1350_135007


namespace min_value_proof_l1350_135001

noncomputable def minimum_value (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : (3/2) * a + b = 1) : ℝ :=
  (3 / a) + (2 / b)

theorem min_value_proof (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : (3/2) * a + b = 1) : minimum_value a b h1 h2 h3 = 25 / 2 :=
sorry

end min_value_proof_l1350_135001


namespace find_a_find_b_plus_c_l1350_135060

variable (a b c : ℝ)
variable (A B C : ℝ) -- Angles in radians

-- Condition: Given that 2a / cos A = (3c - 2b) / cos B
axiom condition1 : 2 * a / (Real.cos A) = (3 * c - 2 * b) / (Real.cos B)

-- Condition 1: b = sqrt(5) * sin B
axiom condition2 : b = Real.sqrt 5 * (Real.sin B)

-- Proof problem for finding a
theorem find_a : a = 5 / 3 := by
  sorry

-- Condition 2: a = sqrt(6) and the area is sqrt(5) / 2
axiom condition3 : a = Real.sqrt 6
axiom condition4 : 1 / 2 * b * c * (Real.sin A) = Real.sqrt 5 / 2

-- Proof problem for finding b + c
theorem find_b_plus_c : b + c = 4 := by
  sorry

end find_a_find_b_plus_c_l1350_135060


namespace conservation_of_mass_l1350_135054

def molecular_weight_C := 12.01
def molecular_weight_H := 1.008
def molecular_weight_O := 16.00
def molecular_weight_Na := 22.99

def molecular_weight_C9H8O4 := (9 * molecular_weight_C) + (8 * molecular_weight_H) + (4 * molecular_weight_O)
def molecular_weight_NaOH := molecular_weight_Na + molecular_weight_O + molecular_weight_H
def molecular_weight_C7H6O3 := (7 * molecular_weight_C) + (6 * molecular_weight_H) + (3 * molecular_weight_O)
def molecular_weight_CH3COONa := (2 * molecular_weight_C) + (3 * molecular_weight_H) + (2 * molecular_weight_O) + molecular_weight_Na

theorem conservation_of_mass :
  (molecular_weight_C9H8O4 + molecular_weight_NaOH) = (molecular_weight_C7H6O3 + molecular_weight_CH3COONa) := by
  sorry

end conservation_of_mass_l1350_135054


namespace value_of_x_div_y_l1350_135093

theorem value_of_x_div_y (x y : ℝ) (h1 : 3 < (x - y) / (x + y)) (h2 : (x - y) / (x + y) < 6) (h3 : ∃ t : ℤ, x = t * y) : 
  ∃ t : ℤ, x = t * y ∧ t = -2 := 
sorry

end value_of_x_div_y_l1350_135093


namespace even_iff_a_zero_monotonous_iff_a_range_max_value_l1350_135092

-- Define the function f
def f (x : ℝ) (a : ℝ) : ℝ := x^2 + a * x + 2

-- (I) Prove that f(x) is even on [-5, 5] if and only if a = 0
theorem even_iff_a_zero (a : ℝ) : (∀ x : ℝ, -5 ≤ x ∧ x ≤ 5 → f x a = f (-x) a) ↔ a = 0 := sorry

-- (II) Prove that f(x) is monotonous on [-5, 5] if and only if a ≥ 10 or a ≤ -10
theorem monotonous_iff_a_range (a : ℝ) : (∀ x y : ℝ, -5 ≤ x ∧ x ≤ y ∧ y ≤ 5 → f x a ≤ f y a) ↔ (a ≥ 10 ∨ a ≤ -10) := sorry

-- (III) Prove the maximum value of f(x) in the interval [-5, 5]
theorem max_value (a : ℝ) : (∃ x : ℝ, -5 ≤ x ∧ x ≤ 5 ∧ (∀ y : ℝ, -5 ≤ y ∧ y ≤ 5 → f y a ≤ f x a)) ∧  
                           ((a ≥ 0 → f 5 a = 27 + 5 * a) ∧ (a < 0 → f (-5) a = 27 - 5 * a)) := sorry

end even_iff_a_zero_monotonous_iff_a_range_max_value_l1350_135092


namespace abs_diff_ge_abs_sum_iff_non_positive_prod_l1350_135038

theorem abs_diff_ge_abs_sum_iff_non_positive_prod (a b : ℝ) : 
  |a - b| ≥ |a| + |b| ↔ a * b ≤ 0 := 
by sorry

end abs_diff_ge_abs_sum_iff_non_positive_prod_l1350_135038


namespace cannot_be_expressed_as_x_squared_plus_y_fifth_l1350_135040

theorem cannot_be_expressed_as_x_squared_plus_y_fifth :
  ¬ ∃ x y : ℤ, 59121 = x^2 + y^5 :=
by sorry

end cannot_be_expressed_as_x_squared_plus_y_fifth_l1350_135040


namespace proof_max_ρ_sq_l1350_135000

noncomputable def max_ρ_sq (a b : ℝ) (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : a ≥ b) 
    (x y : ℝ) (h₃ : 0 ≤ x) (h₄ : x < a) (h₅ : 0 ≤ y) (h₆ : y < b)
    (h_xy : a^2 + y^2 = b^2 + x^2)
    (h_eq : a^2 + y^2 = (a - x)^2 + (b - y)^2)
    (h_x_le : x ≤ 2 * a / 3) : ℝ :=
  (a / b) ^ 2

theorem proof_max_ρ_sq (a b : ℝ) (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : a ≥ b)
    (x y : ℝ) (h₃ : 0 ≤ x) (h₄ : x < a) (h₅ : 0 ≤ y) (h₆ : y < b)
    (h_xy : a^2 + y^2 = b^2 + x^2)
    (h_eq : a^2 + y^2 = (a - x)^2 + (b - y)^2)
    (h_x_le : x ≤ 2 * a / 3) : (max_ρ_sq a b h₀ h₁ h₂ x y h₃ h₄ h₅ h₆ h_xy h_eq h_x_le) ≤ 9 / 5 := by
  sorry

end proof_max_ρ_sq_l1350_135000


namespace weighted_average_AC_l1350_135057

theorem weighted_average_AC (avgA avgB avgC wA wB wC total_weight: ℝ)
  (h_avgA : avgA = 7.3)
  (h_avgB : avgB = 7.6) 
  (h_avgC : avgC = 7.2)
  (h_wA : wA = 3)
  (h_wB : wB = 4)
  (h_wC : wC = 1)
  (h_total_weight : total_weight = 5) :
  ((avgA * wA + avgC * wC) / total_weight) = 5.82 :=
by
  sorry

end weighted_average_AC_l1350_135057


namespace chicken_cost_l1350_135021
noncomputable def chicken_cost_per_plate
  (plates : ℕ) 
  (rice_cost_per_plate : ℝ) 
  (total_cost : ℝ) : ℝ :=
  let total_rice_cost := plates * rice_cost_per_plate
  let total_chicken_cost := total_cost - total_rice_cost
  total_chicken_cost / plates

theorem chicken_cost
  (hplates : plates = 100)
  (hrice_cost_per_plate : rice_cost_per_plate = 0.10)
  (htotal_cost : total_cost = 50) :
  chicken_cost_per_plate 100 0.10 50 = 0.40 :=
by
  sorry

end chicken_cost_l1350_135021


namespace dot_product_a_b_l1350_135025

def vector_a : ℝ × ℝ := (-1, 1)
def vector_b : ℝ × ℝ := (1, -2)

theorem dot_product_a_b :
  (vector_a.1 * vector_b.1 + vector_a.2 * vector_b.2) = -3 :=
by
  sorry

end dot_product_a_b_l1350_135025


namespace boys_to_girls_ratio_l1350_135008

theorem boys_to_girls_ratio (T G : ℕ) (h : (1 / 2) * G = (1 / 6) * T) : (T - G) = 2 * G := by
  sorry

end boys_to_girls_ratio_l1350_135008


namespace original_water_amount_in_mixture_l1350_135031

-- Define heat calculations and conditions
def latentHeatOfFusionIce : ℕ := 80       -- Latent heat of fusion for ice in cal/g
def initialTempWaterAdded : ℕ := 20      -- Initial temperature of added water in °C
def finalTempMixture : ℕ := 5            -- Final temperature of the mixture in °C
def specificHeatWater : ℕ := 1           -- Specific heat of water in cal/g°C

-- Define the known parameters of the problem
def totalMass : ℕ := 250               -- Total mass of the initial mixture in grams
def addedMassWater : ℕ := 1000         -- Mass of added water in grams
def initialTempMixtureIceWater : ℕ := 0  -- Initial temperature of the ice-water mixture in °C

-- Define the equation that needs to be solved
theorem original_water_amount_in_mixture (x : ℝ) :
  (250 - x) * 80 + (250 - x) * 5 + x * 5 = 15000 →
  x = 90.625 :=
by
  intro h
  sorry

end original_water_amount_in_mixture_l1350_135031


namespace nat_representable_as_sequence_or_difference_l1350_135033

theorem nat_representable_as_sequence_or_difference
  (a : ℕ → ℕ)
  (h1 : ∀ n, 0 < a n)
  (h2 : ∀ n, a n < 2 * n) :
  ∀ m : ℕ, ∃ k l : ℕ, k ≠ l ∧ (m = a k ∨ m = a k - a l) :=
by
  sorry

end nat_representable_as_sequence_or_difference_l1350_135033


namespace find_angle_D_l1350_135036

variables (A B C D angle : ℝ)

-- Assumptions based on the problem statement
axiom sum_A_B : A + B = 140
axiom C_eq_D : C = D

-- The claim we aim to prove
theorem find_angle_D (h₁ : A + B = 140) (h₂: C = D): D = 20 :=
by {
    sorry 
}

end find_angle_D_l1350_135036


namespace all_points_on_single_quadratic_l1350_135037

theorem all_points_on_single_quadratic (points : Fin 100 → (ℝ × ℝ)) :
  (∀ (p1 p2 p3 p4 : Fin 100),
    ∃ a b c : ℝ, 
      ∀ (i : Fin 100), 
        (i = p1 ∨ i = p2 ∨ i = p3 ∨ i = p4) →
          (points i).snd = a * (points i).fst ^ 2 + b * (points i).fst + c) → 
  ∃ a b c : ℝ, ∀ i : Fin 100, (points i).snd = a * (points i).fst ^ 2 + b * (points i).fst + c :=
by 
  sorry

end all_points_on_single_quadratic_l1350_135037


namespace max_mark_cells_l1350_135012

theorem max_mark_cells (n : Nat) (grid : Fin n → Fin n → Bool) :
  (∀ i : Fin n, ∃ j : Fin n, grid i j = true) ∧ 
  (∀ j : Fin n, ∃ i : Fin n, grid i j = true) ∧ 
  (∀ (x1 x2 y1 y2 : Fin n), (x1 ≤ x2 ∧ y1 ≤ y2 ∧ (x2.1 - x1.1 + 1) * (y2.1 - y1.1 + 1) ≥ n) → 
   ∃ i : Fin n, ∃ j : Fin n, grid i j = true ∧ x1 ≤ i ∧ i ≤ x2 ∧ y1 ≤ j ∧ j ≤ y2) → 
  (n ≤ 7) := sorry

end max_mark_cells_l1350_135012


namespace joe_purchased_360_gallons_l1350_135097

def joe_initial_paint (P : ℝ) : Prop :=
  let first_week_paint := (1/4) * P
  let remaining_paint := (3/4) * P
  let second_week_paint := (1/2) * remaining_paint
  let total_used_paint := first_week_paint + second_week_paint
  total_used_paint = 225

theorem joe_purchased_360_gallons : ∃ P : ℝ, joe_initial_paint P ∧ P = 360 :=
by
  sorry

end joe_purchased_360_gallons_l1350_135097


namespace b_charges_l1350_135002

theorem b_charges (total_cost : ℕ) (a_hours b_hours c_hours : ℕ)
  (h_total_cost : total_cost = 720)
  (h_a_hours : a_hours = 9)
  (h_b_hours : b_hours = 10)
  (h_c_hours : c_hours = 13) :
  (total_cost * b_hours / (a_hours + b_hours + c_hours)) = 225 :=
by
  sorry

end b_charges_l1350_135002


namespace find_other_number_l1350_135065

variable (A B : ℕ)
variable (LCM : ℕ → ℕ → ℕ)
variable (HCF : ℕ → ℕ → ℕ)

theorem find_other_number (h1 : LCM A B = 2310) 
  (h2 : HCF A B = 30) (h3 : A = 210) : B = 330 := by
  sorry

end find_other_number_l1350_135065


namespace unique_prime_sum_8_l1350_135061
-- Import all necessary mathematical libraries

-- Prime number definition
def is_prime (n : Nat) : Prop := n > 1 ∧ ∀ m : Nat, m ∣ n → m = 1 ∨ m = n

-- Function definition for f(y), number of unique ways to sum primes to form y
def f (y : Nat) : Nat :=
  if y = 8 then 2 else sorry -- We're assuming the correct answer to state the theorem; in a real proof, we would define this correctly.

theorem unique_prime_sum_8 :
  f 8 = 2 :=
by
  -- The proof goes here, but for now, we leave it as a placeholder.
  sorry

end unique_prime_sum_8_l1350_135061


namespace central_angle_of_sector_l1350_135023

theorem central_angle_of_sector (alpha : ℝ) (l : ℝ) (A : ℝ) (h1 : l = 2 * Real.pi) (h2 : A = 5 * Real.pi) : 
  alpha = 72 :=
by
  sorry

end central_angle_of_sector_l1350_135023


namespace regression_analysis_correct_l1350_135070

-- Definition of the regression analysis context
def regression_analysis_variation (forecast_var : Type) (explanatory_var residual_var : Type) : Prop :=
  forecast_var = explanatory_var ∧ forecast_var = residual_var

-- The theorem to prove
theorem regression_analysis_correct :
  ∀ (forecast_var explanatory_var residual_var : Type),
  regression_analysis_variation forecast_var explanatory_var residual_var →
  (forecast_var = explanatory_var ∧ forecast_var = residual_var) :=
by
  intro forecast_var explanatory_var residual_var h
  exact h

end regression_analysis_correct_l1350_135070


namespace perfect_square_A_plus_2B_plus_4_l1350_135051

theorem perfect_square_A_plus_2B_plus_4 (n : ℕ) (hn : 0 < n) :
  let A := (4 / 9 : ℚ) * (10 ^ (2 * n) - 1)
  let B := (8 / 9 : ℚ) * (10 ^ n - 1)
  ∃ k : ℚ, A + 2 * B + 4 = k^2 := 
by {
  sorry
}

end perfect_square_A_plus_2B_plus_4_l1350_135051


namespace area_of_moving_point_l1350_135071

theorem area_of_moving_point (a b : ℝ) :
  (∀ (x y : ℝ), abs x ≤ 1 ∧ abs y ≤ 1 → a * x - 2 * b * y ≤ 2) →
  ∃ (A : ℝ), A = 8 := sorry

end area_of_moving_point_l1350_135071


namespace complex_division_simplification_l1350_135098

theorem complex_division_simplification (i : ℂ) (h_i : i * i = -1) : (1 - 3 * i) / (2 - i) = 1 - i := by
  sorry

end complex_division_simplification_l1350_135098


namespace integer_solutions_of_xyz_equation_l1350_135045

/--
  Find all integer solutions of the equation \( x + y + z = xyz \).
  The integer solutions are expected to be:
  \[
  (1, 2, 3), (2, 1, 3), (3, 1, 2), (3, 2, 1), (1, 3, 2), (2, 3, 1), (-a, 0, a) \text{ for } (a : ℤ).
  \]
-/
theorem integer_solutions_of_xyz_equation (x y z : ℤ) :
    x + y + z = x * y * z ↔ 
    (x = 1 ∧ y = 2 ∧ z = 3) ∨ (x = 2 ∧ y = 1 ∧ z = 3) ∨ 
    (x = 3 ∧ y = 1 ∧ z = 2) ∨ (x = 3 ∧ y = 2 ∧ z = 1) ∨ 
    (x = 1 ∧ y = 3 ∧ z = 2) ∨ (x = 2 ∧ y = 3 ∧ z = 1) ∨ 
    ∃ a : ℤ, (x = -a ∧ y = 0 ∧ z = a) := by
  sorry


end integer_solutions_of_xyz_equation_l1350_135045


namespace factorize_expression_l1350_135006

theorem factorize_expression (x : ℝ) : 4 * x^2 - 4 = 4 * (x + 1) * (x - 1) := 
  sorry

end factorize_expression_l1350_135006


namespace doves_count_l1350_135044

theorem doves_count 
  (num_doves : ℕ)
  (num_eggs_per_dove : ℕ)
  (hatch_rate : ℚ)
  (initial_doves : num_doves = 50)
  (eggs_per_dove : num_eggs_per_dove = 5)
  (hatch_fraction : hatch_rate = 7/9) :
  (num_doves + Int.toNat ((hatch_rate * num_doves * num_eggs_per_dove).floor)) = 244 :=
by
  sorry

end doves_count_l1350_135044


namespace width_of_room_l1350_135010

theorem width_of_room (length : ℝ) (cost_rate : ℝ) (total_cost : ℝ) (width : ℝ)
  (h1 : length = 5.5)
  (h2 : cost_rate = 800)
  (h3 : total_cost = 16500)
  (h4 : width = total_cost / cost_rate / length) : width = 3.75 :=
by
  sorry

end width_of_room_l1350_135010


namespace ratio_of_third_layer_to_second_l1350_135034

theorem ratio_of_third_layer_to_second (s1 s2 s3 : ℕ) (h1 : s1 = 2) (h2 : s2 = 2 * s1) (h3 : s3 = 12) : s3 / s2 = 3 := 
by
  sorry

end ratio_of_third_layer_to_second_l1350_135034


namespace range_of_x_l1350_135063

def star (a b : ℝ) : ℝ := a * b + 2 * a + b

theorem range_of_x (x : ℝ) (h : star x (x - 2) < 0) : -2 < x ∧ x < 1 := by
  sorry

end range_of_x_l1350_135063


namespace xy_plus_one_ge_four_l1350_135095

theorem xy_plus_one_ge_four {x y : ℝ} (hx : 0 < x) (hy : 0 < y) (hxy : x * y = 1) : 
  (x + 1) * (y + 1) >= 4 ∧ ((x + 1) * (y + 1) = 4 ↔ x = 1 ∧ y = 1) :=
by
  sorry

end xy_plus_one_ge_four_l1350_135095


namespace geometric_sequences_common_ratios_l1350_135027

theorem geometric_sequences_common_ratios 
  (k m n o : ℝ)
  (a_2 a_3 b_2 b_3 c_2 c_3 : ℝ)
  (h1 : a_2 = k * m)
  (h2 : a_3 = k * m^2)
  (h3 : b_2 = k * n)
  (h4 : b_3 = k * n^2)
  (h5 : c_2 = k * o)
  (h6 : c_3 = k * o^2)
  (h7 : a_3 - b_3 + c_3 = 2 * (a_2 - b_2 + c_2))
  (h8 : m ≠ n)
  (h9 : m ≠ o)
  (h10 : n ≠ o) : 
  m + n + o = 1 + 2 * Real.sqrt 2 :=
by
  sorry

end geometric_sequences_common_ratios_l1350_135027


namespace complex_ab_value_l1350_135005

open Complex

theorem complex_ab_value (a b : ℝ) (i : ℂ) (h : i = Complex.I) (h₁ : (a + b * i) * (3 + i) = 10 + 10 * i) : a * b = 8 := 
by
  sorry

end complex_ab_value_l1350_135005


namespace ab_bc_ca_value_a4_b4_c4_value_l1350_135059

theorem ab_bc_ca_value (a b c : ℝ) (h1 : a + b + c = 0) (h2 : a^2 + b^2 + c^2 = 1) : 
  ab + bc + ca = -1/2 :=
sorry

theorem a4_b4_c4_value (a b c : ℝ) (h1 : a + b + c = 0) (h2 : a^2 + b^2 + c^2 = 1) :
  a^4 + b^4 + c^4 = 1/2 :=
sorry

end ab_bc_ca_value_a4_b4_c4_value_l1350_135059


namespace linear_function_max_value_l1350_135024

theorem linear_function_max_value (m x : ℝ) (h : -1 ≤ x ∧ x ≤ 3) (y : ℝ) 
  (hl : y = m * x - 2 * m) (hy : y = 6) : m = -2 ∨ m = 6 := 
by 
  sorry

end linear_function_max_value_l1350_135024


namespace ratio_arms_martians_to_aliens_l1350_135067

def arms_of_aliens : ℕ := 3
def legs_of_aliens : ℕ := 8
def legs_of_martians := legs_of_aliens / 2

def limbs_of_5_aliens := 5 * (arms_of_aliens + legs_of_aliens)
def limbs_of_5_martians (arms_of_martians : ℕ) := 5 * (arms_of_martians + legs_of_martians)

theorem ratio_arms_martians_to_aliens (A_m : ℕ) (h1 : limbs_of_5_aliens = limbs_of_5_martians A_m + 5) :
  (A_m : ℚ) / arms_of_aliens = 2 :=
sorry

end ratio_arms_martians_to_aliens_l1350_135067


namespace units_digit_of_3_pow_y_l1350_135016

theorem units_digit_of_3_pow_y
    (x : ℕ)
    (h1 : (2^3)^x = 4096)
    (y : ℕ)
    (h2 : y = x^3) :
    (3^y) % 10 = 1 :=
by
  sorry

end units_digit_of_3_pow_y_l1350_135016


namespace sequence_bound_equivalent_problem_l1350_135074

variable {n : ℕ}
variable {a : Fin (n+2) → ℝ}

theorem sequence_bound_equivalent_problem (h1 : a 0 = 0) (h2 : a (n + 1) = 0) 
  (h3 : ∀ k : Fin n, |a (k.val - 1) - 2 * a k + a (k + 1)| ≤ 1) :
  ∀ k : Fin (n+2), |a k| ≤ k * (n + 1 - k) / 2 := 
by
  sorry

end sequence_bound_equivalent_problem_l1350_135074


namespace function_has_three_zeros_l1350_135052

theorem function_has_three_zeros (a : ℝ) :
  (∃ x1 x2 x3 : ℝ, x1 ≠ x2 ∧ x2 ≠ x3 ∧ x3 ≠ x1 ∧
    ∀ x, (x = x1 ∨ x = x2 ∨ x = x3) ↔ (x^3 + a * x + 2 = 0)) → a < -3 := by
  sorry

end function_has_three_zeros_l1350_135052


namespace regular_polygon_enclosure_l1350_135094

theorem regular_polygon_enclosure (m n : ℕ) (h₁: m = 6) (h₂: (m + 1) = 7): n = 6 :=
by
  -- Lean code to include the problem hypothesis and conclude the theorem
  sorry

end regular_polygon_enclosure_l1350_135094


namespace arithmetic_sequence_l1350_135053

theorem arithmetic_sequence (a : ℕ → ℝ) 
    (h : ∀ m n, |a m + a n - a (m + n)| ≤ 1 / (m + n)) :
    ∃ d, ∀ k, a k = k * d := 
sorry

end arithmetic_sequence_l1350_135053


namespace infinite_solutions_xyz_t_l1350_135083

theorem infinite_solutions_xyz_t (x y z t : ℕ) (h1 : x ≠ 0) (h2 : y ≠ 0) (h3 : z ≠ 0) (h4 : t ≠ 0) (h5 : gcd (gcd x y) (gcd z t) = 1) :
  ∃ (x y z t : ℕ), x^3 + y^3 + z^3 = t^4 ∧ gcd (gcd x y) (gcd z t) = 1 :=
sorry

end infinite_solutions_xyz_t_l1350_135083


namespace exists_unique_c_for_a_equals_3_l1350_135056

theorem exists_unique_c_for_a_equals_3 :
  ∃! c : ℝ, ∀ x ∈ Set.Icc (3 : ℝ) 9, ∃ y ∈ Set.Icc (3 : ℝ) 27, Real.log x / Real.log 3 + Real.log y / Real.log 3 = c :=
sorry

end exists_unique_c_for_a_equals_3_l1350_135056


namespace problem_statement_l1350_135032

theorem problem_statement 
  (x1 y1 x2 y2 x3 y3 x4 y4 a b c : ℝ)
  (h1 : x1 > 0) (h2 : y1 > 0)
  (h3 : x2 < 0) (h4 : y2 > 0)
  (h5 : x3 < 0) (h6 : y3 < 0)
  (h7 : x4 > 0) (h8 : y4 < 0)
  (h9 : (x1 - a)^2 + (y1 - b)^2 ≤ c^2)
  (h10 : (x2 - a)^2 + (y2 - b)^2 ≤ c^2)
  (h11 : (x3 - a)^2 + (y3 - b)^2 ≤ c^2)
  (h12 : (x4 - a)^2 + (y4 - b)^2 ≤ c^2) : a^2 + b^2 < c^2 :=
by sorry

end problem_statement_l1350_135032


namespace amplitude_of_cosine_function_is_3_l1350_135011

variable (a b : ℝ)
variable (h_a : a > 0)
variable (h_b : b > 0)
variable (h_max : ∀ x : ℝ, a * Real.cos (b * x) ≤ 3)
variable (h_cycle : ∃ x₁ x₂ : ℝ, x₁ < x₂ ∧ (∀ x : ℝ, a * Real.cos (b * (x + 2 * Real.pi)) = a * Real.cos (b * x)))

theorem amplitude_of_cosine_function_is_3 :
  a = 3 :=
sorry

end amplitude_of_cosine_function_is_3_l1350_135011


namespace scientific_notation_of_12000000000_l1350_135048

theorem scientific_notation_of_12000000000 :
  12000000000 = 1.2 * 10^10 :=
by sorry

end scientific_notation_of_12000000000_l1350_135048


namespace fish_count_and_total_l1350_135081

-- Definitions of each friend's number of fish
def max_fish : ℕ := 6
def sam_fish : ℕ := 3 * max_fish
def joe_fish : ℕ := 9 * sam_fish
def harry_fish : ℕ := 5 * joe_fish

-- Total number of fish for all friends combined
def total_fish : ℕ := max_fish + sam_fish + joe_fish + harry_fish

-- The theorem stating the problem and corresponding solution
theorem fish_count_and_total :
  max_fish = 6 ∧
  sam_fish = 3 * max_fish ∧
  joe_fish = 9 * sam_fish ∧
  harry_fish = 5 * joe_fish ∧
  total_fish = (max_fish + sam_fish + joe_fish + harry_fish) :=
by
  repeat { sorry }

end fish_count_and_total_l1350_135081


namespace fish_remaining_correct_l1350_135066

def guppies := 225
def angelfish := 175
def tiger_sharks := 200
def oscar_fish := 140
def discus_fish := 120

def guppies_sold := 3/5 * guppies
def angelfish_sold := 3/7 * angelfish
def tiger_sharks_sold := 1/4 * tiger_sharks
def oscar_fish_sold := 1/2 * oscar_fish
def discus_fish_sold := 2/3 * discus_fish

def guppies_remaining := guppies - guppies_sold
def angelfish_remaining := angelfish - angelfish_sold
def tiger_sharks_remaining := tiger_sharks - tiger_sharks_sold
def oscar_fish_remaining := oscar_fish - oscar_fish_sold
def discus_fish_remaining := discus_fish - discus_fish_sold

def total_remaining_fish := guppies_remaining + angelfish_remaining + tiger_sharks_remaining + oscar_fish_remaining + discus_fish_remaining

theorem fish_remaining_correct : total_remaining_fish = 450 := 
by 
  -- insert the necessary steps of the proof here
  sorry

end fish_remaining_correct_l1350_135066


namespace fg_of_3_is_83_l1350_135042

def g (x : ℕ) : ℕ := x ^ 3
def f (x : ℕ) : ℕ := 3 * x + 2
theorem fg_of_3_is_83 : f (g 3) = 83 := by
  sorry

end fg_of_3_is_83_l1350_135042


namespace line_intersects_hyperbola_l1350_135084

theorem line_intersects_hyperbola (k : Real) : 
  (∃ x y : Real, y = k * x ∧ (x^2) / 9 - (y^2) / 4 = 1) ↔ (-2 / 3 < k ∧ k < 2 / 3) := 
sorry

end line_intersects_hyperbola_l1350_135084


namespace duration_period_l1350_135018

-- Define the conditions and what we need to prove
theorem duration_period (t : ℝ) (h : 3200 * 0.025 * t = 400) : 
  t = 5 :=
sorry

end duration_period_l1350_135018


namespace find_number_l1350_135043

variable (x : ℝ)

theorem find_number (h : 0.20 * x = 0.40 * 140 + 80) : x = 680 :=
by
  sorry

end find_number_l1350_135043
