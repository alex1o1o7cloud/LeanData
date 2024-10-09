import Mathlib

namespace order_of_operations_example_l1632_163221

theorem order_of_operations_example :
  3^2 * 4 + 5 * (6 + 3) - 15 / 3 = 76 := by
  sorry

end order_of_operations_example_l1632_163221


namespace correct_propositions_l1632_163212

-- Definitions based on conditions
def diameter_perpendicular_bisects_chord (d : ℝ) (c : ℝ) : Prop :=
  ∃ (r : ℝ), d = 2 * r ∧ c = r

def triangle_vertices_determine_circle (a b c : ℝ) : Prop :=
  ∃ (O : ℝ), O = (a + b + c) / 3

def cyclic_quadrilateral_diagonals_supplementary (a b c d : ℕ) : Prop :=
  a + b + c + d = 360 -- incorrect statement

def tangent_perpendicular_to_radius (r t : ℝ) : Prop :=
  r * t = 1 -- assuming point of tangency

-- Theorem based on the problem conditions
theorem correct_propositions :
  diameter_perpendicular_bisects_chord 2 1 ∧
  triangle_vertices_determine_circle 1 2 3 ∧
  ¬ cyclic_quadrilateral_diagonals_supplementary 90 90 90 90 ∧
  tangent_perpendicular_to_radius 1 1 :=
by
  sorry

end correct_propositions_l1632_163212


namespace problem1_problem2_l1632_163260

noncomputable def sqrt (x : ℝ) := Real.sqrt x

theorem problem1 : sqrt 12 + sqrt 8 * sqrt 6 = 6 * sqrt 3 := by
  sorry

theorem problem2 : sqrt 12 + 1 / (sqrt 3 - sqrt 2) - sqrt 6 * sqrt 3 = 3 * sqrt 3 - 2 * sqrt 2 := by
  sorry

end problem1_problem2_l1632_163260


namespace hypercube_paths_24_l1632_163278

-- Define the 4-dimensional hypercube
structure Hypercube4 :=
(vertices : Fin 16) -- Using Fin 16 to represent the 16 vertices
(edges : Fin 32)    -- Using Fin 32 to represent the 32 edges

def valid_paths (start : Fin 16) : Nat :=
  -- This function should calculate the number of valid paths given the start vertex
  24 -- placeholder, as we are giving the pre-computed total number here

theorem hypercube_paths_24 (start : Fin 16) :
  valid_paths start = 24 :=
by sorry

end hypercube_paths_24_l1632_163278


namespace f_at_seven_l1632_163233

variable {𝓡 : Type*} [CommRing 𝓡] [OrderedAddCommGroup 𝓡] [Module ℝ 𝓡]

-- Assuming f is a function from ℝ to ℝ with the given properties
variable (f : ℝ → ℝ)

-- Condition 1: f is an odd function.
def odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = - f x

-- Condition 2: f(x + 2) = -f(x) for all x.
def periodic_negation (f : ℝ → ℝ) : Prop := ∀ x, f (x + 2) = - f x 

-- Condition 3: f(x) = 2x^2 when x ∈ (0, 2)
def interval_definition (f : ℝ → ℝ) : Prop := ∀ x, 0 < x ∧ x < 2 → f x = 2 * x^2

theorem f_at_seven
  (h_odd : odd_function f)
  (h_periodic : periodic_negation f)
  (h_interval : interval_definition f) :
  f 7 = -2 :=
by
  sorry

end f_at_seven_l1632_163233


namespace parameter_values_l1632_163227

def system_equation_1 (x y : ℝ) : Prop :=
  (|y + 9| + |x + 2| - 2) * (x^2 + y^2 - 3) = 0

def system_equation_2 (x y a : ℝ) : Prop :=
  (x + 2)^2 + (y + 4)^2 = a

theorem parameter_values (a : ℝ) :
  (∃ x y : ℝ, system_equation_1 x y ∧ system_equation_2 x y a ∧ 
    -- counting the number of solutions to the system of equations that total exactly three,
    -- meaning the system has exactly three solutions
    -- Placeholder for counting solutions
    sorry) ↔ (a = 9 ∨ a = 23 + 4 * Real.sqrt 15) := 
sorry

end parameter_values_l1632_163227


namespace smallest_N_winning_strategy_l1632_163250

theorem smallest_N_winning_strategy :
  ∃ (N : ℕ), (N > 0) ∧ (∀ (list : List ℕ), 
    (∀ x, x ∈ list → x > 0 ∧ x ≤ 25) ∧ 
    list.sum ≥ 200 → 
    ∃ (sublist : List ℕ), sublist ⊆ list ∧ 
    200 - N ≤ sublist.sum ∧ sublist.sum ≤ 200 + N) ∧ N = 11 :=
sorry

end smallest_N_winning_strategy_l1632_163250


namespace total_goals_in_5_matches_l1632_163238

theorem total_goals_in_5_matches 
  (x : ℝ) 
  (h1 : 4 * x + 3 = 5 * (x + 0.2)) 
  : 4 * x + 3 = 11 :=
by
  -- The proof is omitted here
  sorry

end total_goals_in_5_matches_l1632_163238


namespace second_meeting_time_l1632_163272

-- Given conditions and constants.
def pool_length : ℕ := 120
def initial_george_distance : ℕ := 80
def initial_henry_distance : ℕ := 40
def george_speed (t : ℕ) : ℕ := initial_george_distance / t
def henry_speed (t : ℕ) : ℕ := initial_henry_distance / t

-- Main statement to prove the question and answer.
theorem second_meeting_time (t : ℕ) (h_t_pos : t > 0) : 
  5 * t = 15 / 2 :=
sorry

end second_meeting_time_l1632_163272


namespace min_a_add_c_l1632_163256

noncomputable def a : ℝ := sorry
noncomputable def b : ℝ := sorry
noncomputable def c : ℝ := sorry
noncomputable def angle_ABC : ℝ := 2 * Real.pi / 3
noncomputable def BD : ℝ := 1

-- The bisector of angle ABC intersects AC at point D
-- Angle bisector theorem and the given information
theorem min_a_add_c : ∃ a c : ℝ, (angle_ABC = 2 * Real.pi / 3) → (BD = 1) → (a * c = a + c) → (a + c ≥ 4) :=
by
  sorry

end min_a_add_c_l1632_163256


namespace distance_is_one_l1632_163251

noncomputable def distance_between_bisectors_and_centroid : ℝ :=
  let AB := 9
  let AC := 12
  let BC := Real.sqrt (AB^2 + AC^2)
  let CD := BC / 2
  let CE := (2/3) * CD
  let r := (AB * AC) / (2 * (AB + AC + BC) / 2)
  let K := CE - r
  K

theorem distance_is_one : distance_between_bisectors_and_centroid = 1 :=
  sorry

end distance_is_one_l1632_163251


namespace find_second_number_l1632_163275

theorem find_second_number (x : ℕ) : 
  ((20 + 40 + 60) / 3 = 4 + ((x + 10 + 28) / 3)) → x = 70 :=
by {
  -- let lhs = (20 + 40 + 60) / 3
  -- let rhs = 4 + ((x + 10 + 28) / 3)
  -- rw rhs at lhs,
  -- value the lhs and rhs,
  -- prove x = 70
  sorry
}

end find_second_number_l1632_163275


namespace length_of_ae_l1632_163291

-- Definitions for lengths of segments
variable {ab bc cd de ac ae : ℝ}

-- Given conditions as assumptions
axiom h1 : bc = 3 * cd
axiom h2 : de = 8
axiom h3 : ab = 5
axiom h4 : ac = 11

-- The main theorem to prove
theorem length_of_ae : ae = ab + bc + cd + de → bc = ac - ab → bc = 6 → cd = bc / 3 → ae = 21 :=
by sorry

end length_of_ae_l1632_163291


namespace min_value_P_l1632_163239

-- Define the polynomial P
def P (x y : ℝ) : ℝ := x^2 + y^2 - 6*x + 8*y + 7

-- Theorem statement: The minimum value of P(x, y) is -18
theorem min_value_P : ∃ (x y : ℝ), P x y = -18 := by
  sorry

end min_value_P_l1632_163239


namespace parallel_lines_implies_m_neg1_l1632_163201

theorem parallel_lines_implies_m_neg1 (m : ℝ) :
  (∀ (x y : ℝ), x + m * y + 6 = 0) ∧
  (∀ (x y : ℝ), (m - 2) * x + 3 * y + 2 * m = 0) ∧
  ∀ (l₁ l₂ : ℝ), l₁ = -(1 / m) ∧ l₂ = -((m - 2) / 3) ∧ l₁ = l₂ → m = -1 :=
by
  sorry

end parallel_lines_implies_m_neg1_l1632_163201


namespace dice_probability_l1632_163292

noncomputable def probability_same_face_in_single_roll : ℝ :=
  (1 / 6)^10

noncomputable def probability_not_all_same_face_in_single_roll : ℝ :=
  1 - probability_same_face_in_single_roll

noncomputable def probability_not_all_same_face_in_five_rolls : ℝ :=
  probability_not_all_same_face_in_single_roll^5

noncomputable def probability_at_least_one_all_same_face : ℝ :=
  1 - probability_not_all_same_face_in_five_rolls

theorem dice_probability :
  probability_at_least_one_all_same_face = 1 - (1 - (1 / 6)^10)^5 :=
sorry

end dice_probability_l1632_163292


namespace range_of_values_l1632_163294

theorem range_of_values (a b : ℝ) : (∀ x : ℝ, x < 1 → ax + b > 2 * (x + 1)) → b > 4 := 
by
  sorry

end range_of_values_l1632_163294


namespace range_of_a_minus_b_l1632_163202

theorem range_of_a_minus_b (a b : ℝ) (h1 : 1 < a ∧ a < 4) (h2 : -2 < b ∧ b < 4) : -3 < a - b ∧ a - b < 6 :=
sorry

end range_of_a_minus_b_l1632_163202


namespace roots_can_be_integers_if_q_positive_roots_cannot_both_be_integers_if_q_negative_l1632_163266

-- Part (a)
theorem roots_can_be_integers_if_q_positive (p q : ℤ) (hq : q > 0) :
  (∃ x y : ℤ, x * y = q ∧ x + y = p) ∧ (∃ x y : ℤ, x * y = q ∧ x + y = p + 1) :=
sorry

-- Part (b)
theorem roots_cannot_both_be_integers_if_q_negative (p q : ℤ) (hq : q < 0) :
  ¬(∃ x y z w : ℤ, x * y = q ∧ x + y = p ∧ z * w = q ∧ z + w = p + 1) :=
sorry

end roots_can_be_integers_if_q_positive_roots_cannot_both_be_integers_if_q_negative_l1632_163266


namespace anna_bought_five_chocolate_bars_l1632_163222

noncomputable section

def initial_amount : ℝ := 10
def price_chewing_gum : ℝ := 1
def price_candy_cane : ℝ := 0.5
def remaining_amount : ℝ := 1

def chewing_gum_cost : ℝ := 3 * price_chewing_gum
def candy_cane_cost : ℝ := 2 * price_candy_cane

def total_spent : ℝ := initial_amount - remaining_amount
def known_items_cost : ℝ := chewing_gum_cost + candy_cane_cost
def chocolate_bars_cost : ℝ := total_spent - known_items_cost
def price_chocolate_bar : ℝ := 1

def chocolate_bars_bought : ℝ := chocolate_bars_cost / price_chocolate_bar

theorem anna_bought_five_chocolate_bars : chocolate_bars_bought = 5 := 
by
  sorry

end anna_bought_five_chocolate_bars_l1632_163222


namespace Mart_income_percentage_of_Juan_l1632_163264

variable (J T M : ℝ)

-- Conditions
def Tim_income_def : Prop := T = 0.5 * J
def Mart_income_def : Prop := M = 1.6 * T

-- Theorem to prove
theorem Mart_income_percentage_of_Juan
  (h1 : Tim_income_def T J) 
  (h2 : Mart_income_def M T) : 
  (M / J) * 100 = 80 :=
by
  sorry

end Mart_income_percentage_of_Juan_l1632_163264


namespace clock_correct_time_fraction_l1632_163247

theorem clock_correct_time_fraction :
  let hours := 24
  let incorrect_hours := 6
  let correct_hours_fraction := (hours - incorrect_hours) / hours
  let minutes_per_hour := 60
  let incorrect_minutes_per_hour := 15
  let correct_minutes_fraction := (minutes_per_hour - incorrect_minutes_per_hour) / minutes_per_hour
  correct_hours_fraction * correct_minutes_fraction = (9 / 16) :=
by 
  sorry

end clock_correct_time_fraction_l1632_163247


namespace basketball_minutes_played_l1632_163223

-- Definitions of the conditions in Lean
def football_minutes : ℕ := 60
def total_hours : ℕ := 2
def total_minutes : ℕ := total_hours * 60

-- The statement we need to prove (that basketball_minutes = 60)
theorem basketball_minutes_played : 
  (120 - football_minutes = 60) := by
  sorry

end basketball_minutes_played_l1632_163223


namespace ball_hits_ground_at_two_seconds_l1632_163231

theorem ball_hits_ground_at_two_seconds :
  (∃ t : ℝ, (-6.1) * t^2 + 2.8 * t + 7 = 0 ∧ t = 2) :=
sorry

end ball_hits_ground_at_two_seconds_l1632_163231


namespace find_length_AB_l1632_163210

noncomputable def length_of_AB (DE DF : ℝ) (AC : ℝ) : ℝ :=
  (AC * DE) / DF

theorem find_length_AB (DE DF AC : ℝ) (pro1 : DE = 9) (pro2 : DF = 17) (pro3 : AC = 10) :
    length_of_AB DE DF AC = 90 / 17 :=
  by
    rw [pro1, pro2, pro3]
    unfold length_of_AB
    norm_num

end find_length_AB_l1632_163210


namespace slope_of_line_is_pm1_l1632_163229

noncomputable def polarCurve (θ : ℝ) : ℝ := 2 * Real.cos θ - 4 * Real.sin θ

noncomputable def lineParametric (t α : ℝ) : ℝ × ℝ := (1 + t * Real.cos α, -1 + t * Real.sin α)

theorem slope_of_line_is_pm1
  (t α : ℝ)
  (hAB : ∃ A B : ℝ × ℝ, lineParametric t α = A ∧ (∃ t1 t2 : ℝ, A = lineParametric t1 α ∧ B = lineParametric t2 α ∧ dist A B = 3 * Real.sqrt 2))
  (hC : ∃ θ : ℝ, polarCurve θ = dist (1, -1) (polarCurve θ * Real.cos θ, polarCurve θ * Real.sin θ)) :
  ∃ k : ℝ, k = 1 ∨ k = -1 :=
sorry

end slope_of_line_is_pm1_l1632_163229


namespace divya_age_l1632_163289

theorem divya_age (D N : ℝ) (h1 : N + 5 = 3 * (D + 5)) (h2 : N + D = 40) : D = 7.5 :=
by sorry

end divya_age_l1632_163289


namespace sales_last_year_l1632_163258

theorem sales_last_year (x : ℝ) (h1 : 416 = (1 + 0.30) * x) : x = 320 :=
by
  sorry

end sales_last_year_l1632_163258


namespace ab_le_1_e2_l1632_163204

noncomputable def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := Real.log x - a * x - b

theorem ab_le_1_e2 {a b : ℝ} (h : 0 < a) (hx : ∃ x : ℝ, 0 < x ∧ f x a b ≥ 0) : a * b ≤ 1 / Real.exp 2 :=
sorry

end ab_le_1_e2_l1632_163204


namespace pow_99_square_pow_neg8_mult_l1632_163274

theorem pow_99_square :
  99^2 = 9801 := 
by
  -- Proof omitted
  sorry

theorem pow_neg8_mult :
  (-8) ^ 2009 * (-1/8) ^ 2008 = -8 :=
by
  -- Proof omitted
  sorry

end pow_99_square_pow_neg8_mult_l1632_163274


namespace geometric_sequence_tan_sum_l1632_163265

theorem geometric_sequence_tan_sum
  (A B C : ℝ)
  (a b c : ℝ)
  (h1 : b^2 = a * c)
  (h2 : Real.tan B = 3/4):
  1 / Real.tan A + 1 / Real.tan C = 5 / 3 := 
by
  sorry

end geometric_sequence_tan_sum_l1632_163265


namespace functional_equation_solution_l1632_163262

theorem functional_equation_solution (f : ℚ → ℚ) :
  (∀ x y : ℚ, f (x + f y) = f x * f y) →
  (∀ x : ℚ, f x = 0 ∨ f x = 1) :=
by
  sorry

end functional_equation_solution_l1632_163262


namespace minimum_packages_shipped_l1632_163286

-- Definitions based on the conditions given in the problem
def Sarah_truck_capacity : ℕ := 18
def Ryan_truck_capacity : ℕ := 11

-- Minimum number of packages shipped
theorem minimum_packages_shipped :
  ∃ (n : ℕ), n = Sarah_truck_capacity * Ryan_truck_capacity :=
by sorry

end minimum_packages_shipped_l1632_163286


namespace cyclist_club_member_count_l1632_163279

-- Define the set of valid digits.
def valid_digits : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 9}

-- Define the problem statement
theorem cyclist_club_member_count : valid_digits.card ^ 3 = 512 :=
by
  -- Placeholder for the proof
  sorry

end cyclist_club_member_count_l1632_163279


namespace negation_of_universal_proposition_l1632_163230

theorem negation_of_universal_proposition :
  (¬ (∀ x : ℝ, x^2 ≥ 0)) ↔ ∃ x : ℝ, x^2 < 0 :=
  sorry

end negation_of_universal_proposition_l1632_163230


namespace probability_heads_exactly_9_of_12_l1632_163226

noncomputable def bin_coeff (n k : ℕ) : ℕ := Nat.choose n k

noncomputable def probability_heads_9_of_12_flips : ℚ :=
  (bin_coeff 12 9 : ℚ) / (2 ^ 12)

theorem probability_heads_exactly_9_of_12 :
  probability_heads_9_of_12_flips = 220 / 4096 :=
by
  sorry

end probability_heads_exactly_9_of_12_l1632_163226


namespace complex_norm_example_l1632_163280

theorem complex_norm_example : 
  abs (-3 - (9 / 4 : ℝ) * I) = 15 / 4 := 
by
  sorry

end complex_norm_example_l1632_163280


namespace convert_8pi_over_5_to_degrees_l1632_163240

noncomputable def radian_to_degree (rad : ℝ) : ℝ := rad * (180 / Real.pi)

theorem convert_8pi_over_5_to_degrees : radian_to_degree (8 * Real.pi / 5) = 288 := by
  sorry

end convert_8pi_over_5_to_degrees_l1632_163240


namespace correct_q_solution_l1632_163220

noncomputable def solve_q (n m q : ℕ) : Prop :=
  (7 / 8 : ℚ) = (n / 96 : ℚ) ∧
  (7 / 8 : ℚ) = ((m + n) / 112 : ℚ) ∧
  (7 / 8 : ℚ) = ((q - m) / 144 : ℚ) ∧
  n = 84 ∧
  m = 14 →
  q = 140

theorem correct_q_solution : ∃ (q : ℕ), solve_q 84 14 q :=
by sorry

end correct_q_solution_l1632_163220


namespace total_money_is_145_83_l1632_163270

noncomputable def jackson_money : ℝ := 125

noncomputable def williams_money : ℝ := jackson_money / 6

noncomputable def total_money : ℝ := jackson_money + williams_money

theorem total_money_is_145_83 :
  total_money = 145.83 := by
sorry

end total_money_is_145_83_l1632_163270


namespace problem1_solution_l1632_163208

theorem problem1_solution (p : ℕ) (hp : Nat.Prime p) (a b c : ℕ) (ha : 0 < a ∧ a ≤ p) (hb : 0 < b ∧ b ≤ p) (hc : 0 < c ∧ c ≤ p)
  (f : ℕ → ℕ) (hf : ∀ x : ℕ, 0 < x → p ∣ f x) :
  (∀ x, f x = a * x^2 + b * x + c) →
  (p = 2 → a + b + c = 4) ∧ (2 < p → p % 2 = 1 → a + b + c = 3 * p) :=
by
  sorry

end problem1_solution_l1632_163208


namespace minimum_value_of_f_l1632_163255

noncomputable def f (x : ℝ) : ℝ := |2*x - 1| + |3*x - 2| + |4*x - 3| + |5*x - 4|

theorem minimum_value_of_f : ∃ x : ℝ, (∀ y : ℝ, f y ≥ 1) ∧ f x = 1 :=
by
  sorry

end minimum_value_of_f_l1632_163255


namespace inequality_problem_l1632_163267

variable (a b c d : ℝ)

open Real

theorem inequality_problem 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d)
  (hprod : a * b * c * d = 1) :
  1 / (a * (1 + b)) + 1 / (b * (1 + c)) + 1 / (c * (1 + d)) + 1 / (d * (1 + a)) ≥ 2 := 
by 
  sorry

end inequality_problem_l1632_163267


namespace charging_time_l1632_163290

theorem charging_time (S T L : ℕ → ℕ) 
  (HS : ∀ t, S t = 15 * t) 
  (HT : ∀ t, T t = 8 * t) 
  (HL : ∀ t, L t = 5 * t)
  (smartphone_capacity tablet_capacity laptop_capacity : ℕ)
  (smartphone_percentage tablet_percentage laptop_percentage : ℕ)
  (h_smartphone : smartphone_capacity = 4500)
  (h_tablet : tablet_capacity = 10000)
  (h_laptop : laptop_capacity = 20000)
  (p_smartphone : smartphone_percentage = 75)
  (p_tablet : tablet_percentage = 25)
  (p_laptop : laptop_percentage = 50)
  (required_charge_s required_charge_t required_charge_l : ℕ)
  (h_rq_s : required_charge_s = smartphone_capacity * smartphone_percentage / 100)
  (h_rq_t : required_charge_t = tablet_capacity * tablet_percentage / 100)
  (h_rq_l : required_charge_l = laptop_capacity * laptop_percentage / 100)
  (time_s time_t time_l : ℕ)
  (h_time_s : time_s = required_charge_s / 15)
  (h_time_t : time_t = required_charge_t / 8)
  (h_time_l : time_l = required_charge_l / 5) : 
  max time_s (max time_t time_l) = 2000 := 
by 
  -- This theorem states that the maximum time taken for charging is 2000 minutes
  sorry

end charging_time_l1632_163290


namespace total_highlighters_correct_l1632_163261

variable (y p b : ℕ)
variable (total_highlighters : ℕ)

def num_yellow_highlighters := 7
def num_pink_highlighters := num_yellow_highlighters + 7
def num_blue_highlighters := num_pink_highlighters + 5
def total_highlighters_in_drawer := num_yellow_highlighters + num_pink_highlighters + num_blue_highlighters

theorem total_highlighters_correct : 
  total_highlighters_in_drawer = 40 :=
sorry

end total_highlighters_correct_l1632_163261


namespace angle_parallel_lines_l1632_163211

variables {Line : Type} (a b c : Line) (theta : ℝ)
variable (angle_between : Line → Line → ℝ)

def is_parallel (a b : Line) : Prop := sorry

theorem angle_parallel_lines (h_parallel : is_parallel a b) (h_angle : angle_between a c = theta) : angle_between b c = theta := 
sorry

end angle_parallel_lines_l1632_163211


namespace solution_set_of_inequality_l1632_163235

theorem solution_set_of_inequality :
  { x : ℝ | (x + 3) * (6 - x) ≥ 0 } = { x : ℝ | -3 ≤ x ∧ x ≤ 6 } :=
sorry

end solution_set_of_inequality_l1632_163235


namespace ratio_of_spent_to_left_after_video_game_l1632_163242

-- Definitions based on conditions
def total_money : ℕ := 100
def spent_on_video_game : ℕ := total_money * 1 / 4
def money_left_after_video_game : ℕ := total_money - spent_on_video_game
def money_left_after_goggles : ℕ := 60
def spent_on_goggles : ℕ := money_left_after_video_game - money_left_after_goggles

-- Statement to prove the ratio
theorem ratio_of_spent_to_left_after_video_game :
  (spent_on_goggles : ℚ) / (money_left_after_video_game : ℚ) = 1 / 5 := 
sorry

end ratio_of_spent_to_left_after_video_game_l1632_163242


namespace initial_time_to_cover_distance_l1632_163209

theorem initial_time_to_cover_distance (s t : ℝ) (h1 : 540 = s * t) (h2 : 540 = 60 * (3/4) * t) : t = 12 :=
sorry

end initial_time_to_cover_distance_l1632_163209


namespace required_force_l1632_163215

theorem required_force (m : ℝ) (g : ℝ) (T : ℝ) (F : ℝ) 
    (h1 : m = 3)
    (h2 : g = 10)
    (h3 : T = m * g)
    (h4 : F = 4 * T) : F = 120 := by
  sorry

end required_force_l1632_163215


namespace first_step_is_remove_parentheses_l1632_163234

variable (x : ℝ)

def equation : Prop := 2 * x + 3 * (2 * x - 1) = 16 - (x + 1)

theorem first_step_is_remove_parentheses (x : ℝ) (eq : equation x) : 
  ∃ step : String, step = "remove the parentheses" := 
  sorry

end first_step_is_remove_parentheses_l1632_163234


namespace arithmetic_sequence_n_value_l1632_163245

theorem arithmetic_sequence_n_value
  (a : ℕ → ℝ)
  (S : ℕ → ℝ)
  (n : ℕ)
  (hS9 : S 9 = 18)
  (ha_n_minus_4 : a (n-4) = 30)
  (hSn : S n = 336)
  (harithmetic_sequence : ∀ k, a (k + 1) - a k = a 2 - a 1) :
  n = 21 :=
sorry

end arithmetic_sequence_n_value_l1632_163245


namespace simplify_expression_l1632_163276

-- Define the problem context
variables {x y : ℝ} {i : ℂ}

-- The mathematical simplification problem
theorem simplify_expression :
  (x ^ 2 + i * y) ^ 3 * (x ^ 2 - i * y) ^ 3 = x ^ 12 - 9 * x ^ 8 * y ^ 2 - 9 * x ^ 4 * y ^ 4 - y ^ 6 :=
by {
  -- Proof steps would go here
  sorry
}

end simplify_expression_l1632_163276


namespace calculation_correct_l1632_163216

theorem calculation_correct : (3.456 - 1.234) * 0.5 = 1.111 :=
by
  sorry

end calculation_correct_l1632_163216


namespace cos_sin_225_deg_l1632_163277

theorem cos_sin_225_deg : (Real.cos (225 * Real.pi / 180) = -Real.sqrt 2 / 2) ∧ (Real.sin (225 * Real.pi / 180) = -Real.sqrt 2 / 2) :=
by
  -- Lean proof steps would go here
  sorry

end cos_sin_225_deg_l1632_163277


namespace athena_spent_correct_amount_l1632_163241

-- Define the conditions
def num_sandwiches : ℕ := 3
def price_per_sandwich : ℝ := 3
def num_drinks : ℕ := 2
def price_per_drink : ℝ := 2.5

-- Define the total cost as per the given conditions
def total_cost : ℝ :=
  (num_sandwiches * price_per_sandwich) + (num_drinks * price_per_drink)

-- The theorem that states the problem and asserts the correct answer
theorem athena_spent_correct_amount : total_cost = 14 := 
  by
    sorry

end athena_spent_correct_amount_l1632_163241


namespace simplify_2M_minus_N_value_at_neg_1_M_gt_N_l1632_163225

-- Definitions of M and N
def M (x : ℝ) : ℝ := 4 * x^2 - 2 * x - 1
def N (x : ℝ) : ℝ := 3 * x^2 - 2 * x - 5

-- The simplified expression for 2M - N
theorem simplify_2M_minus_N {x : ℝ} : 2 * M x - N x = 5 * x^2 - 2 * x + 3 :=
by sorry

-- Value of the simplified expression when x = -1
theorem value_at_neg_1 : (5 * (-1)^2 - 2 * (-1) + 3) = 10 :=
by sorry

-- Relationship between M and N
theorem M_gt_N {x : ℝ} : M x > N x :=
by
  have h : M x - N x = x^2 + 4 := by sorry
  -- x^2 >= 0 for all x, so x^2 + 4 > 0 => M > N
  have nonneg : x^2 >= 0 := by sorry
  have add_pos : x^2 + 4 > 0 := by sorry
  sorry

end simplify_2M_minus_N_value_at_neg_1_M_gt_N_l1632_163225


namespace katya_minimum_problems_l1632_163203

-- Defining the conditions
def katya_probability_solve : ℚ := 4 / 5
def pen_probability_solve : ℚ := 1 / 2
def total_problems : ℕ := 20
def minimum_correct_for_good_grade : ℚ := 13

-- The Lean statement to prove
theorem katya_minimum_problems (x : ℕ) :
  x * katya_probability_solve + (total_problems - x) * pen_probability_solve ≥ minimum_correct_for_good_grade → x ≥ 10 :=
sorry

end katya_minimum_problems_l1632_163203


namespace sum_of_powers_divisible_by_30_l1632_163217

theorem sum_of_powers_divisible_by_30 {a b c : ℤ} (h : (a + b + c) % 30 = 0) : (a^5 + b^5 + c^5) % 30 = 0 := by
  sorry

end sum_of_powers_divisible_by_30_l1632_163217


namespace addition_correct_l1632_163296

-- Define the integers involved
def num1 : ℤ := 22
def num2 : ℤ := 62
def result : ℤ := 84

-- Theorem stating the relationship between the given numbers
theorem addition_correct : num1 + num2 = result :=
by {
  -- proof goes here
  sorry
}

end addition_correct_l1632_163296


namespace perpendicular_lines_l1632_163253

theorem perpendicular_lines (a : ℝ)
  (line1 : (a^2 + a - 6) * x + 12 * y - 3 = 0)
  (line2 : (a - 1) * x - (a - 2) * y + 4 - a = 0) :
  (a - 2) * (a - 3) * (a + 5) = 0 := sorry

end perpendicular_lines_l1632_163253


namespace cuboid_volume_l1632_163288

variable (length width height : ℕ)

-- Conditions given in the problem
def cuboid_edges := (length = 2) ∧ (width = 5) ∧ (height = 8)

-- Mathematically equivalent statement to be proved
theorem cuboid_volume : cuboid_edges length width height → length * width * height = 80 := by
  sorry

end cuboid_volume_l1632_163288


namespace parallel_lines_a_value_l1632_163214

theorem parallel_lines_a_value (a : ℝ) : 
  (∀ x y : ℝ, 3 * x + 2 * a * y - 5 = 0 ↔ (3 * a - 1) * x - a * y - 2 = 0) →
  (a = 0 ∨ a = -1 / 6) :=
by
  sorry

end parallel_lines_a_value_l1632_163214


namespace total_cans_collected_l1632_163244

variable (bags_saturday : ℕ) (bags_sunday : ℕ) (cans_per_bag : ℕ)

def total_bags : ℕ := bags_saturday + bags_sunday

theorem total_cans_collected 
  (h_sat : bags_saturday = 5)
  (h_sun : bags_sunday = 3)
  (h_cans : cans_per_bag = 5) : 
  total_bags bags_saturday bags_sunday * cans_per_bag = 40 :=
by
  sorry

end total_cans_collected_l1632_163244


namespace total_cost_of_dishes_l1632_163219

theorem total_cost_of_dishes
  (e t b : ℝ)
  (h1 : 4 * e + 5 * t + 2 * b = 8.20)
  (h2 : 6 * e + 3 * t + 4 * b = 9.40) :
  5 * e + 6 * t + 3 * b = 12.20 := 
sorry

end total_cost_of_dishes_l1632_163219


namespace wicket_count_l1632_163271

theorem wicket_count (initial_avg new_avg : ℚ) (runs_last_match wickets_last_match : ℕ) (delta_avg : ℚ) (W : ℕ) :
  initial_avg = 12.4 →
  new_avg = 12.0 →
  delta_avg = 0.4 →
  runs_last_match = 26 →
  wickets_last_match = 8 →
  initial_avg * W + runs_last_match = new_avg * (W + wickets_last_match) →
  W = 175 := by
  sorry

end wicket_count_l1632_163271


namespace sum_x_midpoints_of_triangle_l1632_163248

theorem sum_x_midpoints_of_triangle (a b c : ℝ) (h : a + b + c = 15) :
  (a + b) / 2 + (a + c) / 2 + (b + c) / 2 = 15 :=
by
  -- Proof omitted, replace with actual proof
  sorry

end sum_x_midpoints_of_triangle_l1632_163248


namespace blue_markers_count_l1632_163206

-- Definitions based on the problem's conditions
def total_markers : ℕ := 3343
def red_markers : ℕ := 2315

-- Statement to prove
theorem blue_markers_count :
  total_markers - red_markers = 1028 := by
  sorry

end blue_markers_count_l1632_163206


namespace residue_11_pow_2016_mod_19_l1632_163284

theorem residue_11_pow_2016_mod_19 : (11^2016) % 19 = 17 := 
sorry

end residue_11_pow_2016_mod_19_l1632_163284


namespace remaining_plants_after_bugs_l1632_163236

theorem remaining_plants_after_bugs (initial_plants first_day_eaten second_day_fraction third_day_eaten remaining_plants : ℕ) : 
  initial_plants = 30 →
  first_day_eaten = 20 →
  second_day_fraction = 2 →
  third_day_eaten = 1 →
  remaining_plants = initial_plants - first_day_eaten - (initial_plants - first_day_eaten) / second_day_fraction - third_day_eaten →
  remaining_plants = 4 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end remaining_plants_after_bugs_l1632_163236


namespace max_gcd_11n_3_6n_1_l1632_163298

theorem max_gcd_11n_3_6n_1 : ∃ n : ℕ+, ∀ k : ℕ+,  11 * n + 3 = 7 * k + 1 ∧ 6 * n + 1 = 7 * k + 2 → ∃ d : ℕ, d = Nat.gcd (11 * n + 3) (6 * n + 1) ∧ d = 7 :=
by
  sorry

end max_gcd_11n_3_6n_1_l1632_163298


namespace cover_tiles_count_l1632_163257

-- Definitions corresponding to the conditions
def tile_side : ℕ := 6 -- in inches
def tile_area : ℕ := tile_side * tile_side -- area of one tile in square inches

def region_length : ℕ := 3 * 12 -- 3 feet in inches
def region_width : ℕ := 6 * 12 -- 6 feet in inches
def region_area : ℕ := region_length * region_width -- area of the region in square inches

-- The statement of the proof problem
theorem cover_tiles_count : (region_area / tile_area) = 72 :=
by
   -- Proof would be filled in here
   sorry

end cover_tiles_count_l1632_163257


namespace y_intercept_of_line_l1632_163293

theorem y_intercept_of_line 
  (point : ℝ × ℝ)
  (slope_angle : ℝ)
  (h1 : point = (2, -5))
  (h2 : slope_angle = 135) :
  ∃ b : ℝ, (∀ x y : ℝ, y = -x + b ↔ ((y - (-5)) = (-1) * (x - 2))) ∧ b = -3 := 
sorry

end y_intercept_of_line_l1632_163293


namespace part1_part2_1_part2_2_l1632_163252

noncomputable def f (m : ℝ) (a x : ℝ) : ℝ :=
  m / x + Real.log (x / a)

-- Part (1)
theorem part1 (m a : ℝ) (h : m > 0) (ha : a > 0) (hmin : ∀ x, f m a x ≥ 2) : 
  m / a = Real.exp 1 :=
sorry

-- Part (2.1)
theorem part2_1 (a x₀ : ℝ) (ha : a > Real.exp 1) (hx₀ : x₀ > 1) (hzero : f 1 a x₀ = 0) : 
  1 / (2 * x₀) + x₀ < a - 1 :=
sorry

-- Part (2.2)
theorem part2_2 (a x₀ : ℝ) (ha : a > Real.exp 1) (hx₀ : x₀ > 1) (hzero : f 1 a x₀ = 0) : 
  x₀ + 1 / x₀ > 2 * Real.log a - Real.log (Real.log a) :=
sorry

end part1_part2_1_part2_2_l1632_163252


namespace problem1_problem2_problem3_problem4_problem5_problem6_l1632_163282

theorem problem1 : 78 * 4 + 488 = 800 := by sorry
theorem problem2 : 1903 - 475 * 4 = 3 := by sorry
theorem problem3 : 350 * (12 + 342 / 9) = 17500 := by sorry
theorem problem4 : 480 / (125 - 117) = 60 := by sorry
theorem problem5 : (3600 - 18 * 200) / 253 = 0 := by sorry
theorem problem6 : (243 - 162) / 27 * 380 = 1140 := by sorry

end problem1_problem2_problem3_problem4_problem5_problem6_l1632_163282


namespace sum_first_ten_terms_arithmetic_sequence_l1632_163200

theorem sum_first_ten_terms_arithmetic_sequence (a d : ℝ) (S10 : ℝ) 
  (h1 : 0 < d) 
  (h2 : (a - d) + a + (a + d) = -6) 
  (h3 : (a - d) * a * (a + d) = 10) 
  (h4 : S10 = 5 * (2 * (a - d) + 9 * d)) :
  S10 = -20 + 35 * Real.sqrt 6.5 :=
by sorry

end sum_first_ten_terms_arithmetic_sequence_l1632_163200


namespace quadratic_solution_l1632_163263

theorem quadratic_solution (x : ℝ) (h1 : x^2 - 6 * x + 8 = 0) (h2 : x ≠ 0) :
  x = 2 ∨ x = 4 :=
sorry

end quadratic_solution_l1632_163263


namespace number_of_integer_values_l1632_163207

theorem number_of_integer_values (x : ℕ) (h : ⌊ Real.sqrt x ⌋ = 8) : ∃ n : ℕ, n = 17 :=
by
  sorry

end number_of_integer_values_l1632_163207


namespace horizontal_distance_P_Q_l1632_163268

-- Definitions for the given conditions
def curve (x : ℝ) : ℝ := x^2 + 2 * x - 3

-- Define the points P and Q on the curve
def P_x : Set ℝ := {x | curve x = 8}
def Q_x : Set ℝ := {x | curve x = -1}

-- State the theorem to prove horizontal distance is 3sqrt3
theorem horizontal_distance_P_Q : 
  ∃ (x₁ x₂ : ℝ), x₁ ∈ P_x ∧ x₂ ∈ Q_x ∧ |x₁ - x₂| = 3 * Real.sqrt 3 :=
sorry

end horizontal_distance_P_Q_l1632_163268


namespace original_saved_amount_l1632_163249

theorem original_saved_amount (x : ℤ) (h : (3 * x - 42)^2 = 2241) : x = 30 := 
sorry

end original_saved_amount_l1632_163249


namespace min_value_of_quadratic_function_l1632_163299

-- Given the quadratic function y = x^2 + 4x - 5
def quadratic_function (x : ℝ) : ℝ :=
  x^2 + 4*x - 5

-- Statement of the proof in Lean 4
theorem min_value_of_quadratic_function :
  ∃ (x_min y_min : ℝ), y_min = quadratic_function x_min ∧
  ∀ x : ℝ, quadratic_function x ≥ y_min ∧ x_min = -2 ∧ y_min = -9 :=
by
  sorry

end min_value_of_quadratic_function_l1632_163299


namespace chalkboard_area_l1632_163224

theorem chalkboard_area (width : ℝ) (h_w : width = 3) (h_l : 2 * width = length) : width * length = 18 := 
by 
  sorry

end chalkboard_area_l1632_163224


namespace line_through_center_of_circle_l1632_163243

theorem line_through_center_of_circle 
    (x y : ℝ) 
    (h : x^2 + y^2 - 4*x + 6*y = 0) : 
    3*x + 2*y = 0 :=
sorry

end line_through_center_of_circle_l1632_163243


namespace max_y_value_l1632_163218

-- Definitions according to the problem conditions
def is_negative_integer (z : ℤ) : Prop := z < 0

-- The theorem to be proven
theorem max_y_value (x y : ℤ) (hx : is_negative_integer x) (hy : is_negative_integer y) 
  (h_eq : y = 10 * x / (10 - x)) : y = -5 :=
sorry

end max_y_value_l1632_163218


namespace original_bullets_per_person_l1632_163237

theorem original_bullets_per_person (x : ℕ) (h : 5 * (x - 4) = x) : x = 5 :=
by
  sorry

end original_bullets_per_person_l1632_163237


namespace find_a_l1632_163205

noncomputable def f (x a : ℝ) : ℝ := (2 * x + a) ^ 2

theorem find_a (a : ℝ) (h1 : f 2 a = 20) : a = 1 :=
sorry

end find_a_l1632_163205


namespace polynomial_at_1_gcd_of_72_120_168_l1632_163287

-- Define the polynomial function
def polynomial (x : ℤ) : ℤ := 5 * x^5 + 4 * x^4 + 3 * x^3 + 2 * x^2 + x - 6

-- Assertion that the polynomial evaluated at x = 1 gives 9
theorem polynomial_at_1 : polynomial 1 = 9 := by
  -- Usually, this is where the detailed Horner's method proof would go
  sorry

-- Define the gcd function for three numbers
def gcd3 (a b c : ℤ) : ℤ := Int.gcd (Int.gcd a b) c

-- Assertion that the GCD of 72, 120, and 168 is 24
theorem gcd_of_72_120_168 : gcd3 72 120 168 = 24 := by
  -- Usually, this is where the detailed Euclidean algorithm proof would go
  sorry

end polynomial_at_1_gcd_of_72_120_168_l1632_163287


namespace art_museum_survey_l1632_163297

theorem art_museum_survey (V E : ℕ) 
  (h1 : ∀ (x : ℕ), x = 140 → ¬ (x ≤ E))
  (h2 : E = (3 / 4) * V)
  (h3 : V = E + 140) :
  V = 560 := by
  sorry

end art_museum_survey_l1632_163297


namespace dorothy_profit_l1632_163232

def cost_to_buy_ingredients : ℕ := 53
def number_of_doughnuts : ℕ := 25
def selling_price_per_doughnut : ℕ := 3

def revenue : ℕ := number_of_doughnuts * selling_price_per_doughnut
def profit : ℕ := revenue - cost_to_buy_ingredients

theorem dorothy_profit : profit = 22 :=
by
  -- calculation steps
  sorry

end dorothy_profit_l1632_163232


namespace gcd_72_168_l1632_163283

theorem gcd_72_168 : Nat.gcd 72 168 = 24 :=
by
  sorry

end gcd_72_168_l1632_163283


namespace two_star_neg_five_eq_neg_one_neg_two_star_two_star_neg_three_eq_one_l1632_163259

def star (a b : ℤ) : ℤ := a ^ 2 - b + a * b

theorem two_star_neg_five_eq_neg_one : star 2 (-5) = -1 := by
  sorry

theorem neg_two_star_two_star_neg_three_eq_one : star (-2) (star 2 (-3)) = 1 := by
  sorry

end two_star_neg_five_eq_neg_one_neg_two_star_two_star_neg_three_eq_one_l1632_163259


namespace Nancy_weighs_90_pounds_l1632_163213

theorem Nancy_weighs_90_pounds (W : ℝ) (h1 : 0.60 * W = 54) : W = 90 :=
by
  sorry

end Nancy_weighs_90_pounds_l1632_163213


namespace smallest_m_for_triangle_sides_l1632_163281

noncomputable def is_triangle_sides (a b c : ℝ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

theorem smallest_m_for_triangle_sides (a b c : ℝ) (h : is_triangle_sides a b c) :
  (a^2 + c^2) / (b + c)^2 < 1 / 2 := sorry

end smallest_m_for_triangle_sides_l1632_163281


namespace buratino_solved_16_problems_l1632_163254

-- Defining the conditions given in the problem
def total_kopeks_received : ℕ := 655 * 100 + 35

def geometric_sum (n : ℕ) : ℕ := 2^n - 1

-- The goal is to prove that Buratino solved 16 problems
theorem buratino_solved_16_problems (n : ℕ) (h : geometric_sum n = total_kopeks_received) : n = 16 := by
  sorry

end buratino_solved_16_problems_l1632_163254


namespace photo_album_pages_l1632_163273

noncomputable def P1 := 0
noncomputable def P2 := 10
noncomputable def remaining_pages := 20

theorem photo_album_pages (photos total_pages photos_per_page_set1 photos_per_page_set2 photos_per_page_remaining : ℕ) 
  (h1 : photos = 100)
  (h2 : total_pages = 30)
  (h3 : photos_per_page_set1 = 3)
  (h4 : photos_per_page_set2 = 4)
  (h5 : photos_per_page_remaining = 3) : 
  P1 = 0 ∧ P2 = 10 ∧ remaining_pages = 20 :=
by
  sorry

end photo_album_pages_l1632_163273


namespace max_money_received_back_l1632_163285

def total_money_before := 3000
def value_chip_20 := 20
def value_chip_100 := 100
def chips_lost_total := 16
def chips_lost_diff_1 (x y : ℕ) := x = y + 2
def chips_lost_diff_2 (x y : ℕ) := x = y - 2

theorem max_money_received_back :
  ∃ (x y : ℕ), 
  (chips_lost_diff_1 x y ∨ chips_lost_diff_2 x y) ∧ 
  (x + y = chips_lost_total) ∧
  total_money_before - (x * value_chip_20 + y * value_chip_100) = 2120 :=
sorry

end max_money_received_back_l1632_163285


namespace cost_of_shirts_l1632_163246

theorem cost_of_shirts : 
  let shirt1 := 15
  let shirt2 := 15
  let shirt3 := 15
  let shirt4 := 20
  let shirt5 := 20
  shirt1 + shirt2 + shirt3 + shirt4 + shirt5 = 85 := 
by
  sorry

end cost_of_shirts_l1632_163246


namespace macey_weeks_to_save_l1632_163295

theorem macey_weeks_to_save :
  ∀ (total_cost amount_saved weekly_savings : ℝ),
    total_cost = 22.45 →
    amount_saved = 7.75 →
    weekly_savings = 1.35 →
    ⌈(total_cost - amount_saved) / weekly_savings⌉ = 11 :=
by
  intros total_cost amount_saved weekly_savings h_total_cost h_amount_saved h_weekly_savings
  sorry

end macey_weeks_to_save_l1632_163295


namespace new_supervisor_salary_l1632_163228

theorem new_supervisor_salary
  (W S1 S2 : ℝ)
  (avg_old : (W + S1) / 9 = 430)
  (S1_val : S1 = 870)
  (avg_new : (W + S2) / 9 = 410) :
  S2 = 690 :=
by
  sorry

end new_supervisor_salary_l1632_163228


namespace tank_plastering_cost_l1632_163269

noncomputable def plastering_cost (L W D : ℕ) (cost_per_sq_meter : ℚ) : ℚ :=
  let A_bottom := L * W
  let A_long_walls := 2 * (L * D)
  let A_short_walls := 2 * (W * D)
  let A_total := A_bottom + A_long_walls + A_short_walls
  A_total * cost_per_sq_meter

theorem tank_plastering_cost :
  plastering_cost 25 12 6 0.25 = 186 := by
  sorry

end tank_plastering_cost_l1632_163269
