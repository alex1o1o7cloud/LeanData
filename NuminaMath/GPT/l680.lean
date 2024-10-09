import Mathlib

namespace find_cost_prices_l680_68009

noncomputable def cost_price_per_meter
  (selling_price_per_meter : ℕ) (loss_per_meter : ℕ) : ℕ :=
  selling_price_per_meter + loss_per_meter

theorem find_cost_prices
  (selling_A : ℕ) (meters_A : ℕ) (loss_A : ℕ)
  (selling_B : ℕ) (meters_B : ℕ) (loss_B : ℕ)
  (selling_C : ℕ) (meters_C : ℕ) (loss_C : ℕ)
  (H_A : selling_A = 9000) (H_meters_A : meters_A = 300) (H_loss_A : loss_A = 6)
  (H_B : selling_B = 7000) (H_meters_B : meters_B = 250) (H_loss_B : loss_B = 4)
  (H_C : selling_C = 12000) (H_meters_C : meters_C = 400) (H_loss_C : loss_C = 8) :
  cost_price_per_meter (selling_A / meters_A) loss_A = 36 ∧
  cost_price_per_meter (selling_B / meters_B) loss_B = 32 ∧
  cost_price_per_meter (selling_C / meters_C) loss_C = 38 :=
by {
  sorry
}

end find_cost_prices_l680_68009


namespace negation_of_sin_le_one_l680_68028

theorem negation_of_sin_le_one : (¬ (∀ x : ℝ, Real.sin x ≤ 1)) ↔ (∃ x : ℝ, Real.sin x > 1) :=
by
  sorry

end negation_of_sin_le_one_l680_68028


namespace yan_distance_ratio_l680_68098

theorem yan_distance_ratio 
  (w x y : ℝ)
  (h1 : y / w = x / w + (x + y) / (10 * w)) :
  x / y = 9 / 11 :=
by
  sorry

end yan_distance_ratio_l680_68098


namespace rounding_bounds_l680_68036

theorem rounding_bounds:
  ∃ (max min : ℕ), (∀ x : ℕ, (x >= 1305000) → (x < 1305000) -> false) ∧ 
  (max = 1304999) ∧ 
  (min = 1295000) :=
by
  -- Proof steps would go here
  sorry

end rounding_bounds_l680_68036


namespace integer_part_inequality_l680_68051

theorem integer_part_inequality (x y z : ℝ) (h_pos : x > 0 ∧ y > 0 ∧ z > 0) 
 (h_cond : (x + y + z) * ((1 / x) + (1 / y) + (1 / z)) = (91 / 10)) :
  (⌊(x^3 + y^3 + z^3) * ((1 / x^3) + (1 / y^3) + (1 / z^3))⌋) = 9 :=
by
  -- proof here
  sorry

end integer_part_inequality_l680_68051


namespace largest_pot_cost_l680_68083

noncomputable def cost_of_largest_pot (x : ℝ) : ℝ :=
  x + 5 * 0.15

theorem largest_pot_cost :
  ∃ (x : ℝ), (6 * x + 5 * 0.15 + (0.15 + 2 * 0.15 + 3 * 0.15 + 4 * 0.15 + 5 * 0.15) = 8.85) →
    cost_of_largest_pot x = 1.85 :=
by
  sorry

end largest_pot_cost_l680_68083


namespace hash_difference_l680_68064

def hash (x y : ℕ) : ℤ := x * y - 3 * x + y

theorem hash_difference :
  (hash 8 5) - (hash 5 8) = -12 :=
by
  sorry

end hash_difference_l680_68064


namespace find_R_plus_S_l680_68069

theorem find_R_plus_S (d e R S : ℝ) 
  (h1 : d + 3 = 0)
  (h2 : 7 * d + 3 * e = 0)
  (h3 : R = 3 * d + e + 7)
  (h4 : S = 7 * e) :
  R + S = 54 :=
by
  sorry

end find_R_plus_S_l680_68069


namespace find_fraction_l680_68013

noncomputable def fraction_of_eighths (N : ℝ) (a b : ℝ) : Prop :=
  (3/8) * N * (a/b) = 24

noncomputable def two_fifty_percent (N : ℝ) : Prop :=
  2.5 * N = 199.99999999999997

theorem find_fraction {N a b : ℝ} (h1 : fraction_of_eighths N a b) (h2 : two_fifty_percent N) :
  a/b = 4/5 :=
sorry

end find_fraction_l680_68013


namespace correct_value_wrongly_copied_l680_68049

theorem correct_value_wrongly_copied 
  (mean_initial : ℕ)
  (mean_correct : ℕ)
  (wrong_value : ℕ) 
  (n : ℕ) 
  (initial_mean : mean_initial = 250)
  (correct_mean : mean_correct = 251)
  (wrongly_copied : wrong_value = 135)
  (number_of_values : n = 30) : 
  ∃ x : ℕ, x = 165 := 
by
  use (wrong_value + (mean_correct - mean_initial) * n / n)
  sorry

end correct_value_wrongly_copied_l680_68049


namespace total_baseball_cards_l680_68021
-- Import the broad Mathlib library

-- The conditions stating the number of cards each person has
def melanie_cards : ℕ := 3
def benny_cards : ℕ := 3
def sally_cards : ℕ := 3
def jessica_cards : ℕ := 3

-- The theorem to prove the total number of cards they have is 12
theorem total_baseball_cards : melanie_cards + benny_cards + sally_cards + jessica_cards = 12 := by
  sorry

end total_baseball_cards_l680_68021


namespace candy_last_days_l680_68072

theorem candy_last_days (candy_neighbors candy_sister candy_per_day : ℕ)
  (h1 : candy_neighbors = 5)
  (h2 : candy_sister = 13)
  (h3 : candy_per_day = 9):
  (candy_neighbors + candy_sister) / candy_per_day = 2 :=
by
  sorry

end candy_last_days_l680_68072


namespace find_x_l680_68075

theorem find_x (x y : ℤ) (h1 : x + 2 * y = 100) (h2 : y = 25) : x = 50 :=
by
  sorry

end find_x_l680_68075


namespace find_w_l680_68079

variables {x y : ℚ}

def w : ℚ × ℚ := (-48433 / 975, 2058 / 325)

def vec1 : ℚ × ℚ := (3, 2)
def vec2 : ℚ × ℚ := (3, 4)

def proj (u v : ℚ × ℚ) : ℚ × ℚ :=
  let dot_uv := u.1 * v.1 + u.2 * v.2
  let dot_vv := v.1 * v.1 + v.2 * v.2
  (dot_uv / dot_vv * v.1, dot_uv / dot_vv * v.2)

def p1 : ℚ × ℚ := (47 / 13, 31 / 13)
def p2 : ℚ × ℚ := (85 / 25, 113 / 25)

theorem find_w (hw : w = (x, y)) :
  proj ⟨x, y⟩ vec1 = p1 ∧
  proj ⟨x, y⟩ vec2 = p2 :=
sorry

end find_w_l680_68079


namespace wendy_adds_18_gallons_l680_68059

-- Definitions based on the problem
def truck_tank_capacity : ℕ := 20
def car_tank_capacity : ℕ := 12
def truck_tank_fraction_full : ℚ := 1 / 2
def car_tank_fraction_full : ℚ := 1 / 3

-- Conditions on the amount of gallons currently in the tanks
def truck_current_gallons : ℚ := truck_tank_capacity * truck_tank_fraction_full
def car_current_gallons : ℚ := car_tank_capacity * car_tank_fraction_full

-- Amount of gallons needed to fill up each tank
def truck_gallons_to_add : ℚ := truck_tank_capacity - truck_current_gallons
def car_gallons_to_add : ℚ := car_tank_capacity - car_current_gallons

-- Total gallons needed to fill both tanks
def total_gallons_to_add : ℚ := truck_gallons_to_add + car_gallons_to_add

-- Theorem statement
theorem wendy_adds_18_gallons :
  total_gallons_to_add = 18 := sorry

end wendy_adds_18_gallons_l680_68059


namespace find_added_number_l680_68006

theorem find_added_number 
  (initial_number : ℕ)
  (final_result : ℕ)
  (h : initial_number = 8)
  (h_result : 3 * (2 * initial_number + final_result) = 75) : 
  final_result = 9 := by
  sorry

end find_added_number_l680_68006


namespace calc_exp_l680_68095

open Real

theorem calc_exp (x y : ℝ) : 
  (-(1/3) * (x^2) * y) ^ 3 = -(x^6 * y^3) / 27 := 
  sorry

end calc_exp_l680_68095


namespace general_term_l680_68001

noncomputable def seq (a : ℕ → ℝ) : Prop :=
  a 1 = 1 ∧ ∀ n : ℕ, n > 0 → a (n + 1) = 2 * a n / (2 + a n)

theorem general_term (a : ℕ → ℝ) (h : seq a) : ∀ n : ℕ, n > 0 → a n = 2 / (n + 1) :=
by
sorry

end general_term_l680_68001


namespace sum_of_roots_l680_68046

theorem sum_of_roots : 
  ( ∀ x : ℝ, x^2 - 7*x + 10 = 0 → x = 2 ∨ x = 5 ) → 
  ( 2 + 5 = 7 ) := 
by
  sorry

end sum_of_roots_l680_68046


namespace equation_of_line_l_equations_of_line_m_l680_68030

-- Define the point P and condition for line l
def P := (2, (7 : ℚ)/4)
def l_slope : ℚ := 3 / 4

-- Define the given equation form and conditions for line l
def condition_l (x y : ℚ) : Prop := y - (7 / 4) = (3 / 4) * (x - 2)
def equation_l (x y : ℚ) : Prop := 3 * x - 4 * y = 5

theorem equation_of_line_l :
  ∀ x y : ℚ, condition_l x y → equation_l x y :=
sorry

-- Define the distance condition for line m
def equation_m (x y n : ℚ) : Prop := 3 * x - 4 * y + n = 0
def distance_condition_m (n : ℚ) : Prop := 
  |(-1 + n : ℚ)| / 5 = 3

theorem equations_of_line_m :
  ∃ n : ℚ, distance_condition_m n ∧ (equation_m 2 (7/4) n) ∨ 
            equation_m 2 (7/4) (-14) :=
sorry

end equation_of_line_l_equations_of_line_m_l680_68030


namespace probability_meeting_proof_l680_68040

noncomputable def probability_meeting (arrival_time_paul arrival_time_caroline : ℝ) : Prop :=
  arrival_time_paul ≤ arrival_time_caroline + 1 / 4 ∧ arrival_time_paul ≥ arrival_time_caroline - 1 / 4

theorem probability_meeting_proof :
  ∀ (arrival_time_paul arrival_time_caroline : ℝ)
    (h_paul_range : 0 ≤ arrival_time_paul ∧ arrival_time_paul ≤ 1)
    (h_caroline_range: 0 ≤ arrival_time_caroline ∧ arrival_time_caroline ≤ 1),
  (probability_meeting arrival_time_paul arrival_time_caroline) → 
  ∃ p, p = 7/16 :=
by
  sorry

end probability_meeting_proof_l680_68040


namespace amgm_inequality_proof_l680_68008

noncomputable def amgm_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : Prop :=
  1 < (a / (Real.sqrt (a^2 + b^2))) + (b / (Real.sqrt (b^2 + c^2))) + (c / (Real.sqrt (c^2 + a^2))) 
  ∧ (a / (Real.sqrt (a^2 + b^2))) + (b / (Real.sqrt (b^2 + c^2))) + (c / (Real.sqrt (c^2 + a^2))) 
  ≤ (3 * Real.sqrt 2) / 2

theorem amgm_inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  amgm_inequality a b c ha hb hc := 
sorry

end amgm_inequality_proof_l680_68008


namespace complement_of_M_wrt_U_l680_68074

-- Definitions of the sets U and M as given in the problem
def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def M : Set ℕ := {1, 3, 5}

-- The goal is to show the complement of M w.r.t. U is {2, 4, 6}
theorem complement_of_M_wrt_U :
  (U \ M) = {2, 4, 6} := 
by
  sorry

end complement_of_M_wrt_U_l680_68074


namespace max_value_l680_68057

variable (x y : ℝ)

def condition : Prop := 2 * x ^ 2 + x * y - y ^ 2 = 1

noncomputable def expression : ℝ := (x - 2 * y) / (5 * x ^ 2 - 2 * x * y + 2 * y ^ 2)

theorem max_value : ∀ x y : ℝ, condition x y → expression x y ≤ (Real.sqrt 2) / 4 :=
by
  sorry

end max_value_l680_68057


namespace product_of_digits_l680_68090

theorem product_of_digits (A B : ℕ) (h1 : A + B = 12) (h2 : 8 ∣ (10 * A + B)) : A * B = 32 :=
sorry

end product_of_digits_l680_68090


namespace average_salary_of_employees_l680_68007

theorem average_salary_of_employees (A : ℝ) 
  (h1 : (20 : ℝ) * A + 3400 = 21 * (A + 100)) : 
  A = 1300 := 
by 
  -- proof goes here 
  sorry

end average_salary_of_employees_l680_68007


namespace third_term_is_five_l680_68050

variable {a : ℕ → ℕ}
variable {S : ℕ → ℕ}

-- Suppose S_n = n^2 for n ∈ ℕ*
axiom H1 : ∀ n : ℕ, n > 0 → S n = n * n

-- The relationship a_n = S_n - S_(n-1) for n ≥ 2
axiom H2 : ∀ n : ℕ, n ≥ 2 → a n = S n - S (n - 1)

-- Prove that the third term is 5
theorem third_term_is_five : a 3 = 5 := by
  sorry

end third_term_is_five_l680_68050


namespace trader_allows_discount_l680_68088

-- Definitions for cost price, marked price, and selling price
variable (cp : ℝ)
def mp := cp + 0.12 * cp
def sp := cp - 0.01 * cp

-- The statement to prove
theorem trader_allows_discount :
  mp cp - sp cp = 13 :=
sorry

end trader_allows_discount_l680_68088


namespace problem1_problem2_l680_68077

-- Statement for Problem ①
theorem problem1 
: ( (-1 / 12 - 1 / 36 + 1 / 6) * (-36) = -2) := by
  sorry

-- Statement for Problem ②
theorem problem2
: ((-99 - 11 / 12) * 24 = -2398) := by
  sorry

end problem1_problem2_l680_68077


namespace cone_base_radius_l680_68032

variable (s : ℝ) (A : ℝ) (r : ℝ)

theorem cone_base_radius (h1 : s = 5) (h2 : A = 15 * Real.pi) : r = 3 :=
by
  sorry

end cone_base_radius_l680_68032


namespace derivative_at_pi_div_3_l680_68067

noncomputable def f (x : ℝ) : ℝ := (1 + Real.sqrt 2) * Real.sin x - Real.cos x

theorem derivative_at_pi_div_3 :
  deriv f (π / 3) = (1 / 2) * (1 + Real.sqrt 2 + Real.sqrt 3) :=
by
  sorry

end derivative_at_pi_div_3_l680_68067


namespace apples_added_l680_68038

theorem apples_added (initial_apples added_apples final_apples : ℕ) 
  (h1 : initial_apples = 8) 
  (h2 : final_apples = 13) 
  (h3 : final_apples = initial_apples + added_apples) : 
  added_apples = 5 :=
by
  sorry

end apples_added_l680_68038


namespace find_e_l680_68016

noncomputable def quadratic (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

theorem find_e (r s d e : ℝ) 
  (h1 : quadratic 2 (-4) (-6) r = 0)
  (h2 : quadratic 2 (-4) (-6) s = 0)
  (h3 : r + s = 2) 
  (h4 : r * s = -3)
  (h5 : d = -(r + s - 6))
  (h6 : e = (r - 3) * (s - 3)) : 
  e = 0 :=
sorry

end find_e_l680_68016


namespace part_a_l680_68022

theorem part_a (a x y : ℕ) (h_a_pos : a > 0) (h_x_pos : x > 0) (h_y_pos : y > 0) (h_neq : x ≠ y) :
  (a * x + Nat.gcd a x + Nat.lcm a x) ≠ (a * y + Nat.gcd a y + Nat.lcm a y) := sorry

end part_a_l680_68022


namespace hydrogen_atomic_weight_is_correct_l680_68035

-- Definitions and assumptions based on conditions
def molecular_weight : ℝ := 68
def number_of_hydrogen_atoms : ℕ := 1
def number_of_chlorine_atoms : ℕ := 1
def number_of_oxygen_atoms : ℕ := 2
def atomic_weight_chlorine : ℝ := 35.45
def atomic_weight_oxygen : ℝ := 16.00

-- Definition for the atomic weight of hydrogen to be proved
def atomic_weight_hydrogen (w : ℝ) : Prop :=
  w * number_of_hydrogen_atoms
  + atomic_weight_chlorine * number_of_chlorine_atoms
  + atomic_weight_oxygen * number_of_oxygen_atoms = molecular_weight

-- The theorem to prove the atomic weight of hydrogen
theorem hydrogen_atomic_weight_is_correct : atomic_weight_hydrogen 1.008 :=
by
  unfold atomic_weight_hydrogen
  simp
  sorry

end hydrogen_atomic_weight_is_correct_l680_68035


namespace simplified_fraction_sum_l680_68058

theorem simplified_fraction_sum (n d : ℕ) (h_n : n = 144) (h_d : d = 256) : (9 + 16 = 25) := by
  have h1 : n = 2^4 * 3^2 := by sorry
  have h2 : d = 2^8 := by sorry
  have h3 : (n / gcd n d) = 9 := by sorry
  have h4 : (d / gcd n d) = 16 := by sorry
  exact rfl

end simplified_fraction_sum_l680_68058


namespace analytic_expression_of_f_range_of_k_l680_68078

noncomputable def quadratic_function_minimum (a b : ℝ) : ℝ :=
a * (-1) ^ 2 + b * (-1) + 1

theorem analytic_expression_of_f (a b : ℝ) (ha : quadratic_function_minimum a b = 0)
  (hmin: -1 = -b / (2 * a)) : a = 1 ∧ b = 2 :=
by sorry

theorem range_of_k (k : ℝ) : ∃ k : ℝ, (k ∈ Set.Ici 3 ∨ k = 13 / 4) :=
by sorry

end analytic_expression_of_f_range_of_k_l680_68078


namespace product_PA_PB_eq_nine_l680_68029

theorem product_PA_PB_eq_nine 
  (P A B : ℝ × ℝ) 
  (hP : P = (3, 1)) 
  (h1 : A ≠ B)
  (h2 : ∃ L : ℝ × ℝ → Prop, L P ∧ L A ∧ L B) 
  (h3 : A.fst ^ 2 + A.snd ^ 2 = 1) 
  (h4 : B.fst ^ 2 + B.snd ^ 2 = 1) : 
  |((P.1 - A.1) ^ 2 + (P.2 - A.2) ^ 2)| * |((P.1 - B.1) ^ 2 + (P.2 - B.2) ^ 2)| = 9 := 
sorry

end product_PA_PB_eq_nine_l680_68029


namespace symmetric_line_eq_l680_68065

theorem symmetric_line_eq (x y : ℝ) (h₁ : y = 3 * x + 4) : y = x → y = (1 / 3) * x - (4 / 3) :=
by
  sorry

end symmetric_line_eq_l680_68065


namespace cost_for_23_days_l680_68045

-- Define the cost structure
def costFirstWeek : ℕ → ℝ := λ days => if days <= 7 then days * 18 else 7 * 18
def costAdditionalDays : ℕ → ℝ := λ days => if days > 7 then (days - 7) * 14 else 0

-- Total cost equation
def totalCost (days : ℕ) : ℝ := costFirstWeek days + costAdditionalDays days

-- Declare the theorem to prove
theorem cost_for_23_days : totalCost 23 = 350 := by
  sorry

end cost_for_23_days_l680_68045


namespace solve_inequality_l680_68081

theorem solve_inequality (x : ℝ) : 
  -2 < (x^2 - 18*x + 35) / (x^2 - 4*x + 8) ∧ 
  (x^2 - 18*x + 35) / (x^2 - 4*x + 8) < 2 ↔ 
  3 < x ∧ x < 17 / 3 :=
by
  sorry

end solve_inequality_l680_68081


namespace maddox_theo_equal_profit_l680_68025

-- Definitions based on the problem conditions
def maddox_initial_cost := 10 * 35
def theo_initial_cost := 15 * 30
def maddox_revenue := 10 * 50
def theo_revenue := 15 * 40

-- Define profits based on the revenues and costs
def maddox_profit := maddox_revenue - maddox_initial_cost
def theo_profit := theo_revenue - theo_initial_cost

-- The theorem to be proved
theorem maddox_theo_equal_profit : maddox_profit = theo_profit :=
by
  -- Omitted proof steps
  sorry

end maddox_theo_equal_profit_l680_68025


namespace am_gm_inequality_example_l680_68031

theorem am_gm_inequality_example (x y : ℝ) (hx : x = 16) (hy : y = 64) : 
  (x + y) / 2 ≥ Real.sqrt (x * y) :=
by
  rw [hx, hy]
  sorry

end am_gm_inequality_example_l680_68031


namespace min_area_triangle_ABC_l680_68033

theorem min_area_triangle_ABC :
  let A := (0, 0) 
  let B := (42, 18)
  (∃ p q : ℤ, let C := (p, q) 
              ∃ area : ℝ, area = (1 / 2 : ℝ) * |42 * q - 18 * p| 
              ∧ area = 3) := 
sorry

end min_area_triangle_ABC_l680_68033


namespace largest_among_five_numbers_l680_68053

theorem largest_among_five_numbers :
  max (max (max (max (12345 + 1 / 3579) 
                       (12345 - 1 / 3579))
                   (12345 ^ (1 / 3579)))
               (12345 / (1 / 3579)))
           12345.3579 = 12345 / (1 / 3579) := sorry

end largest_among_five_numbers_l680_68053


namespace find_a_if_lines_perpendicular_l680_68019

theorem find_a_if_lines_perpendicular (a : ℝ) :
  (∀ x, (y1 : ℝ) = a * x - 2 → (y2 : ℝ) = (a + 2) * x + 1 → y1 * y2 = -1) → a = -1 :=
by {
  sorry
}

end find_a_if_lines_perpendicular_l680_68019


namespace cindys_correct_result_l680_68096

-- Explicitly stating the conditions as definitions
def incorrect_operation_result := 260
def x := (incorrect_operation_result / 5) - 7

theorem cindys_correct_result : 5 * x + 7 = 232 :=
by
  -- Placeholder for the proof
  sorry

end cindys_correct_result_l680_68096


namespace total_goals_l680_68091

-- Definitions
def louie_goals_last_match := 4
def louie_previous_goals := 40
def brother_multiplier := 2
def seasons := 3
def games_per_season := 50

-- Total number of goals scored by Louie and his brother
theorem total_goals : (louie_previous_goals + louie_goals_last_match) 
                      + (brother_multiplier * louie_goals_last_match * seasons * games_per_season) 
                      = 1244 :=
by sorry

end total_goals_l680_68091


namespace sqrt_expression_simplification_l680_68093

theorem sqrt_expression_simplification : 
  (Real.sqrt 48 - Real.sqrt 2 * Real.sqrt 6 - Real.sqrt 15 / Real.sqrt 5) = Real.sqrt 3 := 
  by
    sorry

end sqrt_expression_simplification_l680_68093


namespace minimum_value_of_expression_l680_68048

theorem minimum_value_of_expression (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_prod : a * b * c = 3) : 
  a^2 + 8 * a * b + 32 * b^2 + 24 * b * c + 8 * c^2 ≥ 72 :=
sorry

end minimum_value_of_expression_l680_68048


namespace convex_quadrilateral_diagonal_l680_68027

theorem convex_quadrilateral_diagonal (P : ℝ) (d1 d2 : ℝ) (hP : P = 2004) (hd1 : d1 = 1001) :
  (d2 = 1 → False) ∧ 
  (d2 = 2 → True) ∧ 
  (d2 = 1001 → True) :=
by
  sorry

end convex_quadrilateral_diagonal_l680_68027


namespace rent_percentage_l680_68014

-- Define Elaine's earnings last year
def E : ℝ := sorry

-- Define last year's rent expenditure
def rentLastYear : ℝ := 0.20 * E

-- Define this year's earnings
def earningsThisYear : ℝ := 1.35 * E

-- Define this year's rent expenditure
def rentThisYear : ℝ := 0.30 * earningsThisYear

-- Prove the required percentage
theorem rent_percentage : ((rentThisYear / rentLastYear) * 100) = 202.5 := by
  sorry

end rent_percentage_l680_68014


namespace circle_area_l680_68084

theorem circle_area (r : ℝ) (h : 2 * (1 / (2 * π * r)) = r / 2) : π * r^2 = 2 := 
by 
  sorry

end circle_area_l680_68084


namespace unique_solution_of_quadratic_l680_68017

theorem unique_solution_of_quadratic (a : ℝ) :
  (∃! x : ℝ, a * x^2 - 3 * x + 2 = 0) → (a = 0 ∨ a = 9 / 8) :=
by
  sorry

end unique_solution_of_quadratic_l680_68017


namespace expand_product_l680_68068

theorem expand_product (y : ℝ) : 5 * (y - 3) * (y + 10) = 5 * y^2 + 35 * y - 150 :=
by 
  sorry

end expand_product_l680_68068


namespace line_y2_does_not_pass_through_fourth_quadrant_l680_68055

theorem line_y2_does_not_pass_through_fourth_quadrant (k b : ℝ) (h1 : k < 0) (h2 : b > 0) : 
  ¬(∃ x y : ℝ, (y = b * x - k ∧ x > 0 ∧ y < 0)) := 
by 
  sorry

end line_y2_does_not_pass_through_fourth_quadrant_l680_68055


namespace problem_l680_68000

-- Definitions for the arithmetic sequence
def arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

-- Given conditions
def a (n : ℕ) : ℤ := sorry -- Define the arithmetic sequence a_n based on conditions

-- Problem statement
theorem problem : 
  (a 1 = 4) ∧
  (a 2 + a 4 = 4) →
  (∃ d : ℤ, arithmetic_sequence a d ∧ a 10 = -5) :=
by {
  sorry
}

end problem_l680_68000


namespace difference_of_squares_eval_l680_68026

-- Define the conditions
def a : ℕ := 81
def b : ℕ := 49

-- State the corresponding problem and its equivalence
theorem difference_of_squares_eval : (a^2 - b^2) = 4160 := by
  sorry -- Placeholder for the proof

end difference_of_squares_eval_l680_68026


namespace prime_power_minus_l680_68054

theorem prime_power_minus (p : ℕ) (hp : Nat.Prime p) (hps : Nat.Prime (p + 3)) : p ^ 11 - 52 = 1996 := by
  -- this is where the proof would go
  sorry

end prime_power_minus_l680_68054


namespace cos_squared_value_l680_68099

theorem cos_squared_value (x : ℝ) (h : Real.sin (x + π / 6) = 1 / 4) : 
  Real.cos (π / 3 - x) ^ 2 = 1 / 16 := 
sorry

end cos_squared_value_l680_68099


namespace hayley_stickers_l680_68061

theorem hayley_stickers (S F x : ℕ) (hS : S = 72) (hF : F = 9) (hx : x = S / F) : x = 8 :=
by
  sorry

end hayley_stickers_l680_68061


namespace find_P_and_Q_l680_68062

variables {x P Q b c : ℝ}

theorem find_P_and_Q :
  (∃ b c : ℝ, (x^2 + 3 * x + 7) * (x^2 + b * x + c) = x^4 + P * x^2 + Q) →
  (b + 3 = 0) →
  (3 * b + c + 7 = P) →
  (7 * b + 3 * c = 0) →
  (7 * c = Q) →
  P + Q = 54 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end find_P_and_Q_l680_68062


namespace age_of_teacher_l680_68073

theorem age_of_teacher (S : ℕ) (T : Real) (n : ℕ) (average_student_age : Real) (new_average_age : Real) : 
  average_student_age = 14 → 
  new_average_age = 14.66 → 
  n = 45 → 
  S = average_student_age * n → 
  T = 44.7 :=
by
  sorry

end age_of_teacher_l680_68073


namespace line_equation_intercept_twice_x_intercept_l680_68012

theorem line_equation_intercept_twice_x_intercept 
  {x y : ℝ}
  (intersection_point : ∃ (x y : ℝ), 2 * x + y - 8 = 0 ∧ x - 2 * y + 1 = 0) 
  (y_intercept_is_twice_x_intercept : ∃ (a : ℝ), ∀ (x y : ℝ), y = 2 * a ∧ x = a) :
  (∃ (x y : ℝ), 2 * x - 3 * y = 0) ∨ (∃ (x y : ℝ), 2 * x + y - 8 = 0) :=
sorry

end line_equation_intercept_twice_x_intercept_l680_68012


namespace pattyCoinsValue_l680_68092

def totalCoins (q d : ℕ) : Prop := q + d = 30
def originalValue (q d : ℕ) : ℝ := 0.25 * q + 0.10 * d
def swappedValue (q d : ℕ) : ℝ := 0.10 * q + 0.25 * d
def valueIncrease (q : ℕ) : Prop := swappedValue q (30 - q) - originalValue q (30 - q) = 1.20

theorem pattyCoinsValue (q d : ℕ) (h1 : totalCoins q d) (h2 : valueIncrease q) : originalValue q d = 4.65 := 
by
  sorry

end pattyCoinsValue_l680_68092


namespace not_even_not_odd_neither_even_nor_odd_l680_68039

noncomputable def f (x : ℝ) : ℝ := ⌊x⌋ + 1 / 2

theorem not_even (x : ℝ) : f (-x) ≠ f x := sorry
theorem not_odd (x : ℝ) : f (0) ≠ 0 ∨ f (-x) ≠ -f x := sorry

theorem neither_even_nor_odd : ∀ x : ℝ, f (-x) ≠ f x ∧ (f (0) ≠ 0 ∨ f (-x) ≠ -f x) :=
by
  intros x
  exact ⟨not_even x, not_odd x⟩

end not_even_not_odd_neither_even_nor_odd_l680_68039


namespace carol_first_six_l680_68082

-- A formalization of the probabilities involved when Alice, Bob, Carol,
-- and Dave take turns rolling a die, and the process repeats.
def probability_carol_first_six (prob_rolling_six : ℚ) : ℚ := sorry

theorem carol_first_six (prob_rolling_six : ℚ) (h : prob_rolling_six = 1/6) :
  probability_carol_first_six prob_rolling_six = 25 / 91 :=
sorry

end carol_first_six_l680_68082


namespace remainder_of_3_pow_19_mod_5_l680_68056

theorem remainder_of_3_pow_19_mod_5 : (3 ^ 19) % 5 = 2 := by
  have h : 3 ^ 4 % 5 = 1 := by sorry
  sorry

end remainder_of_3_pow_19_mod_5_l680_68056


namespace find_n_positive_integer_l680_68011

theorem find_n_positive_integer:
  ∀ n : ℕ, n > 0 → (∃ k : ℕ, 2^n + 12^n + 2011^n = k^2) ↔ n = 1 := 
by
  sorry

end find_n_positive_integer_l680_68011


namespace percentage_is_12_l680_68004

variable (x : ℝ) (p : ℝ)

-- Given the conditions
def condition_1 : Prop := 0.25 * x = (p / 100) * 1500 - 15
def condition_2 : Prop := x = 660

-- We need to prove that the percentage p is 12
theorem percentage_is_12 (h1 : condition_1 x p) (h2 : condition_2 x) : p = 12 := by
  sorry

end percentage_is_12_l680_68004


namespace find_F_neg_a_l680_68034

-- Definitions of odd functions
def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- Definition of F
def F (f g : ℝ → ℝ) (x : ℝ) := 3 * f x + 5 * g x + 2

theorem find_F_neg_a (f g : ℝ → ℝ) (a : ℝ)
  (hf : is_odd f) (hg : is_odd g) (hFa : F f g a = 3) : F f g (-a) = 1 :=
by
  sorry

end find_F_neg_a_l680_68034


namespace find_number_l680_68023

theorem find_number (x : ℝ) (h : 0.6667 * x + 1 = 0.75 * x) : x = 12 :=
sorry

end find_number_l680_68023


namespace distinct_triangles_in_regular_ngon_l680_68097

theorem distinct_triangles_in_regular_ngon (n : ℕ) (h : n ≥ 3) :
  ∃ t : ℕ, t = n * (n-1) * (n-2) / 6 := 
sorry

end distinct_triangles_in_regular_ngon_l680_68097


namespace seeds_germinated_percentage_l680_68094

theorem seeds_germinated_percentage 
  (n1 n2 : ℕ) 
  (p1 p2 : ℝ) 
  (h1 : n1 = 300)
  (h2 : n2 = 200)
  (h3 : p1 = 0.15)
  (h4 : p2 = 0.35) : 
  ( ( p1 * n1 + p2 * n2 ) / ( n1 + n2 ) ) * 100 = 23 :=
by
  -- Mathematical proof goes here.
  sorry

end seeds_germinated_percentage_l680_68094


namespace find_certain_number_l680_68080

theorem find_certain_number (x : ℕ) (h : (55 * x) % 8 = 7) : x = 1 := 
sorry

end find_certain_number_l680_68080


namespace total_accidents_l680_68002

-- Define the given vehicle counts for the highways
def total_vehicles_A : ℕ := 4 * 10^9
def total_vehicles_B : ℕ := 2 * 10^9
def total_vehicles_C : ℕ := 1 * 10^9

-- Define the accident ratios per highway
def accident_ratio_A : ℕ := 80
def accident_ratio_B : ℕ := 120
def accident_ratio_C : ℕ := 65

-- Define the number of vehicles in millions
def million := 10^6

-- Define the accident calculations per highway
def accidents_A : ℕ := (total_vehicles_A / (100 * million)) * accident_ratio_A
def accidents_B : ℕ := (total_vehicles_B / (200 * million)) * accident_ratio_B
def accidents_C : ℕ := (total_vehicles_C / (50 * million)) * accident_ratio_C

-- Prove the total number of accidents across all highways
theorem total_accidents : accidents_A + accidents_B + accidents_C = 5700 := by
  have : accidents_A = 3200 := by sorry
  have : accidents_B = 1200 := by sorry
  have : accidents_C = 1300 := by sorry
  sorry

end total_accidents_l680_68002


namespace millicent_fraction_books_l680_68037

variable (M H : ℝ)
variable (F : ℝ)

-- Conditions
def harold_has_half_books (M H : ℝ) : Prop := H = (1 / 2) * M
def harold_brings_one_third_books (M H : ℝ) : Prop := (1 / 3) * H = (1 / 6) * M
def new_library_capacity (M F : ℝ) : Prop := (1 / 6) * M + F * M = (5 / 6) * M

-- Target Proof Statement
theorem millicent_fraction_books (M H F : ℝ) 
    (h1 : harold_has_half_books M H) 
    (h2 : harold_brings_one_third_books M H) 
    (h3 : new_library_capacity M F) : 
    F = 2 / 3 :=
sorry

end millicent_fraction_books_l680_68037


namespace complement_of_M_in_U_l680_68089

def U : Set ℝ := Set.univ
def M : Set ℝ := {x | x^2 - 2*x > 0}
def complement_U_M : Set ℝ := {x | 0 ≤ x ∧ x ≤ 2}

theorem complement_of_M_in_U : (U \ M) = complement_U_M :=
by sorry

end complement_of_M_in_U_l680_68089


namespace jeff_cats_count_l680_68085

theorem jeff_cats_count :
  let initial_cats := 20
  let found_monday := 2 + 3
  let found_tuesday := 1 + 2
  let adopted_wednesday := 4 * 2
  let adopted_thursday := 3
  let found_friday := 3
  initial_cats + found_monday + found_tuesday - adopted_wednesday - adopted_thursday + found_friday = 20 := by
  sorry

end jeff_cats_count_l680_68085


namespace calculate_A_share_l680_68005

variable (x : ℝ) (total_gain : ℝ)
variable (h_b_invests : 2 * x)  -- B invests double the amount after 6 months
variable (h_c_invests : 3 * x)  -- C invests thrice the amount after 8 months

/-- Calculate the share of A from the total annual gain -/
theorem calculate_A_share (h_total_gain : total_gain = 18600) :
  let a_investmentMonths := x * 12
  let b_investmentMonths := (2 * x) * 6
  let c_investmentMonths := (3 * x) * 4
  let total_investmentMonths := a_investmentMonths + b_investmentMonths + c_investmentMonths
  let a_share := (a_investmentMonths / total_investmentMonths) * total_gain
  a_share = 6200 :=
by
  sorry

end calculate_A_share_l680_68005


namespace fraction_equiv_subtract_l680_68018

theorem fraction_equiv_subtract (n : ℚ) : (4 - n) / (7 - n) = 3 / 5 → n = 0.5 :=
by
  intros h
  sorry

end fraction_equiv_subtract_l680_68018


namespace problem_1_problem_2_l680_68086

noncomputable def f (x : ℝ) : ℝ := |x - 2|
noncomputable def g (x : ℝ) (m : ℝ) : ℝ := m * |x| - 2

theorem problem_1 : {x : ℝ | f x > 3} = {x : ℝ | x < -1 ∨ x > 5} :=
sorry

theorem problem_2 (m : ℝ) : (∀ x : ℝ, f x ≥ g x m) → m ≤ 1 :=
sorry

end problem_1_problem_2_l680_68086


namespace increase_percentage_when_selfcheckout_broken_l680_68063

-- The problem conditions as variable definitions and declarations
def normal_complaints : ℕ := 120
def short_staffed_increase : ℚ := 1 / 3
def short_staffed_complaints : ℕ := normal_complaints + (normal_complaints / 3)
def total_complaints_three_days : ℕ := 576
def days : ℕ := 3
def both_conditions_complaints : ℕ := total_complaints_three_days / days

-- The theorem that we need to prove
theorem increase_percentage_when_selfcheckout_broken : 
  (both_conditions_complaints - short_staffed_complaints) * 100 / short_staffed_complaints = 20 := 
by
  -- This line sets up that the conclusion is true
  sorry

end increase_percentage_when_selfcheckout_broken_l680_68063


namespace initial_violet_balloons_l680_68044

-- Define initial conditions and variables
def red_balloons := 4
def violet_balloons_lost := 3
def current_violet_balloons := 4

-- Define the theorem we want to prove
theorem initial_violet_balloons (red_balloons : ℕ) (violet_balloons_lost : ℕ) (current_violet_balloons : ℕ) : 
  red_balloons = 4 → violet_balloons_lost = 3 → current_violet_balloons = 4 → (current_violet_balloons + violet_balloons_lost) = 7 :=
by
  intros
  sorry

end initial_violet_balloons_l680_68044


namespace trig_expression_value_l680_68060

theorem trig_expression_value (θ : Real) (h1 : θ > Real.pi) (h2 : θ < 3 * Real.pi / 2) (h3 : Real.tan (2 * θ) = 3 / 4) :
  (2 * Real.cos (θ / 2) ^ 2 + Real.sin θ - 1) / (Real.sqrt 2 * Real.cos (θ + Real.pi / 4)) = 2 := by
  sorry

end trig_expression_value_l680_68060


namespace union_of_sets_l680_68024

def A : Set ℤ := {-1, 0, 1}
def B : Set ℤ := {1, 2}

theorem union_of_sets : A ∪ B = {-1, 0, 1, 2} := 
by
  sorry

end union_of_sets_l680_68024


namespace sin_double_angle_cos_condition_l680_68010

theorem sin_double_angle_cos_condition (x : ℝ) (h : Real.cos (π / 4 - x) = 3 / 5) :
  Real.sin (2 * x) = -7 / 25 :=
sorry

end sin_double_angle_cos_condition_l680_68010


namespace number_of_girls_attending_winter_festival_l680_68041

variables (g b : ℝ)
variables (totalStudents attendFestival: ℝ)

theorem number_of_girls_attending_winter_festival
  (H1 : g + b = 1500)
  (H2 : (3/5) * g + (2/5) * b = 800) :
  (3/5 * g) = 600 :=
sorry

end number_of_girls_attending_winter_festival_l680_68041


namespace y_neither_directly_nor_inversely_proportional_l680_68015

theorem y_neither_directly_nor_inversely_proportional (x y : ℝ) :
  ¬((∃ k : ℝ, x = k * y) ∨ (∃ k : ℝ, x * y = k)) ↔ 2 * x + 3 * y = 6 :=
by 
  sorry

end y_neither_directly_nor_inversely_proportional_l680_68015


namespace find_some_value_l680_68070

theorem find_some_value (m n k : ℝ)
  (h1 : m = n / 6 - 2 / 5)
  (h2 : m + k = (n + 18) / 6 - 2 / 5) : 
  k = 3 :=
sorry

end find_some_value_l680_68070


namespace paul_baseball_cards_l680_68020

-- Define the necessary variables and statements
variable {n : ℕ}

-- State the problem and the proof target
theorem paul_baseball_cards : ∃ k, k = 3 * n + 1 := sorry

end paul_baseball_cards_l680_68020


namespace pushups_count_l680_68087

theorem pushups_count :
  ∀ (David Zachary Hailey : ℕ),
    David = 44 ∧ (David = Zachary + 9) ∧ (Zachary = 2 * Hailey) ∧ (Hailey = 27) →
      (David = 63 ∧ Zachary = 54 ∧ Hailey = 27) :=
by
  intros David Zachary Hailey
  intro conditions
  obtain ⟨hDavid44, hDavid9Zachary, hZachary2Hailey, hHailey27⟩ := conditions
  sorry

end pushups_count_l680_68087


namespace discount_correct_l680_68042

def normal_cost : ℝ := 80
def discount_rate : ℝ := 0.45
def discounted_cost : ℝ := normal_cost - (discount_rate * normal_cost)

theorem discount_correct : discounted_cost = 44 := by
  -- By computation, 0.45 * 80 = 36 and 80 - 36 = 44
  sorry

end discount_correct_l680_68042


namespace x_eq_one_is_sufficient_but_not_necessary_for_x_squared_plus_x_minus_two_eq_zero_l680_68066

theorem x_eq_one_is_sufficient_but_not_necessary_for_x_squared_plus_x_minus_two_eq_zero :
  ∃ (x : ℝ), (x = 1) → (x^2 + x - 2 = 0) ∧ (¬ (∀ (y : ℝ), y^2 + y - 2 = 0 → y = 1)) := by
  sorry

end x_eq_one_is_sufficient_but_not_necessary_for_x_squared_plus_x_minus_two_eq_zero_l680_68066


namespace no_solution_inequalities_l680_68043

theorem no_solution_inequalities (a : ℝ) :
  (¬ ∃ x : ℝ, x > 1 ∧ x < a - 1) → a ≤ 2 :=
by
  intro h
  sorry

end no_solution_inequalities_l680_68043


namespace part1_tangent_circles_part2_chords_l680_68052

theorem part1_tangent_circles (t : ℝ) : 
  t = 1 → 
  ∃ (a b : ℝ), 
    (x + 1)^2 + y^2 = 1 ∨ 
    (x + (2/5))^2 + (y - (9/5))^2 = (1 : ℝ) :=
by
  sorry

theorem part2_chords (t : ℝ) : 
  (∀ (k1 k2 : ℝ), 
    k1 + k2 = -3 * t / 4 ∧ 
    k1 * k2 = (t^2 - 1) / 8 ∧ 
    |k1 - k2| = 3 / 4) → 
    t = 1 ∨ t = -1 :=
by
  sorry

end part1_tangent_circles_part2_chords_l680_68052


namespace domain_of_function_l680_68003

theorem domain_of_function :
  {x : ℝ | x + 3 ≥ 0 ∧ x + 2 ≠ 0} = {x : ℝ | x ≥ -3 ∧ x ≠ -2} :=
by
  sorry

end domain_of_function_l680_68003


namespace spadesuit_eval_l680_68076

def spadesuit (a b : ℤ) := abs (a - b)

theorem spadesuit_eval :
  spadesuit 5 (spadesuit 3 (spadesuit 8 12)) = 4 := 
by
  sorry

end spadesuit_eval_l680_68076


namespace max_value_5x_minus_25x_l680_68071

noncomputable def max_value_of_expression : ℝ :=
  (1 / 4 : ℝ)

theorem max_value_5x_minus_25x :
  ∃ x : ℝ, ∀ y : ℝ, y = 5^x → (5^y - 25^y) ≤ max_value_of_expression :=
sorry

end max_value_5x_minus_25x_l680_68071


namespace jerry_reaches_3_at_some_time_l680_68047

def jerry_reaches_3_probability (n : ℕ) (k : ℕ) : ℚ :=
  -- This function represents the probability that Jerry reaches 3 at some point during n coin tosses
  if n = 7 ∧ k = 3 then (21 / 64 : ℚ) else 0

theorem jerry_reaches_3_at_some_time :
  jerry_reaches_3_probability 7 3 = (21 / 64 : ℚ) :=
sorry

end jerry_reaches_3_at_some_time_l680_68047
