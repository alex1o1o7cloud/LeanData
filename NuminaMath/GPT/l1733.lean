import Mathlib

namespace finding_breadth_and_length_of_floor_l1733_173360

noncomputable def length_of_floor (b : ℝ) := 3 * b
noncomputable def area_of_floor (b : ℝ) := (length_of_floor b) * b

theorem finding_breadth_and_length_of_floor
  (breadth : ℝ)
  (length : ℝ := length_of_floor breadth)
  (area : ℝ := area_of_floor breadth)
  (painting_cost : ℝ)
  (cost_per_sqm : ℝ)
  (h1 : painting_cost = 100)
  (h2 : cost_per_sqm = 2)
  (h3 : area = painting_cost / cost_per_sqm) :
  length = Real.sqrt 150 :=
by
  sorry

end finding_breadth_and_length_of_floor_l1733_173360


namespace diophantine_solution_l1733_173339

theorem diophantine_solution :
  ∃ (x y k : ℤ), 1990 * x - 173 * y = 11 ∧ x = -22 + 173 * k ∧ y = 253 - 1990 * k :=
by {
  sorry
}

end diophantine_solution_l1733_173339


namespace incorrect_observation_value_l1733_173385

-- Definitions stemming from the given conditions
def initial_mean : ℝ := 100
def corrected_mean : ℝ := 99.075
def number_of_observations : ℕ := 40
def correct_observation_value : ℝ := 50

-- Lean theorem statement to prove the incorrect observation value
theorem incorrect_observation_value (initial_mean corrected_mean correct_observation_value : ℝ) (number_of_observations : ℕ) :
  (initial_mean * number_of_observations - corrected_mean * number_of_observations + correct_observation_value) = 87 := 
sorry

end incorrect_observation_value_l1733_173385


namespace jose_profit_share_correct_l1733_173380

-- Definitions for the conditions
def tom_investment : ℕ := 30000
def tom_months : ℕ := 12
def jose_investment : ℕ := 45000
def jose_months : ℕ := 10
def total_profit : ℕ := 36000

-- Capital months calculations
def tom_capital_months : ℕ := tom_investment * tom_months
def jose_capital_months : ℕ := jose_investment * jose_months
def total_capital_months : ℕ := tom_capital_months + jose_capital_months

-- Jose's share of the profit calculation
def jose_share_of_profit : ℕ := (jose_capital_months * total_profit) / total_capital_months

-- The theorem to prove
theorem jose_profit_share_correct : jose_share_of_profit = 20000 := by
  -- This is where the proof steps would go
  sorry

end jose_profit_share_correct_l1733_173380


namespace exists_c_with_same_nonzero_decimal_digits_l1733_173361

theorem exists_c_with_same_nonzero_decimal_digits (m n : ℕ) (hm : 0 < m) (hn : 0 < n) : 
  ∃ (c : ℕ), 0 < c ∧ (∃ (k : ℕ), (c * m) % 10^k = (c * n) % 10^k) := 
sorry

end exists_c_with_same_nonzero_decimal_digits_l1733_173361


namespace g_product_of_roots_l1733_173359

def f (x : ℂ) : ℂ := x^6 + x^3 + 1
def g (x : ℂ) : ℂ := x^2 + 1

theorem g_product_of_roots (x_1 x_2 x_3 x_4 x_5 x_6 : ℂ) 
    (h1 : ∀ x, (x - x_1) * (x - x_2) * (x - x_3) * (x - x_4) * (x - x_5) * (x - x_6) = f x) :
    g x_1 * g x_2 * g x_3 * g x_4 * g x_5 * g x_6 = 1 :=
by 
    sorry

end g_product_of_roots_l1733_173359


namespace deposit_amount_correct_l1733_173390

noncomputable def deposit_amount (initial_amount : ℝ) : ℝ :=
  let first_step := 0.30 * initial_amount
  let second_step := 0.25 * first_step
  0.20 * second_step

theorem deposit_amount_correct :
  deposit_amount 50000 = 750 :=
by
  sorry

end deposit_amount_correct_l1733_173390


namespace find_derivative_l1733_173333

theorem find_derivative (f : ℝ → ℝ) (f' : ℝ → ℝ) (h : ∀ x, f x = 2 * x * f' 1 + Real.log x) : f' 1 = -1 := 
by
  sorry

end find_derivative_l1733_173333


namespace back_wheel_revolutions_l1733_173329

-- Defining relevant distances and conditions
def front_wheel_radius : ℝ := 3 -- radius in feet
def back_wheel_radius : ℝ := 0.5 -- radius in feet
def front_wheel_revolutions : ℕ := 120

-- The target theorem
theorem back_wheel_revolutions :
  let front_wheel_circumference := 2 * Real.pi * front_wheel_radius
  let total_distance := front_wheel_circumference * (front_wheel_revolutions : ℝ)
  let back_wheel_circumference := 2 * Real.pi * back_wheel_radius
  let back_wheel_revs := total_distance / back_wheel_circumference
  back_wheel_revs = 720 :=
by
  sorry

end back_wheel_revolutions_l1733_173329


namespace back_seat_capacity_l1733_173363

def left_seats : Nat := 15
def right_seats : Nat := left_seats - 3
def seats_per_person : Nat := 3
def total_capacity : Nat := 92
def regular_seats_people : Nat := (left_seats + right_seats) * seats_per_person

theorem back_seat_capacity :
  total_capacity - regular_seats_people = 11 :=
by
  sorry

end back_seat_capacity_l1733_173363


namespace smallest_positive_value_l1733_173388

theorem smallest_positive_value (x : ℝ) (hx : x > 0) (h : x / 7 + 2 / (7 * x) = 1) : 
  x = (7 - Real.sqrt 41) / 2 :=
sorry

end smallest_positive_value_l1733_173388


namespace correct_statement_is_D_l1733_173368

axiom three_points_determine_plane : Prop
axiom line_and_point_determine_plane : Prop
axiom quadrilateral_is_planar_figure : Prop
axiom two_intersecting_lines_determine_plane : Prop

theorem correct_statement_is_D : two_intersecting_lines_determine_plane = True := 
by sorry

end correct_statement_is_D_l1733_173368


namespace root_in_interval_l1733_173349

noncomputable def f (x : ℝ) : ℝ := 3^x + 3*x - 8

theorem root_in_interval :
  f 1 < 0 ∧ f 1.5 > 0 ∧ f 1.25 < 0 → ∃ c, 1.25 < c ∧ c < 1.5 ∧ f c = 0 :=
by
  sorry

end root_in_interval_l1733_173349


namespace find_least_x_divisible_by_17_l1733_173382

theorem find_least_x_divisible_by_17 (x k : ℕ) (h : x + 2 = 17 * k) : x = 15 :=
sorry

end find_least_x_divisible_by_17_l1733_173382


namespace final_value_after_three_years_l1733_173381

theorem final_value_after_three_years (X : ℝ) :
  (X - 0.40 * X) * (1 - 0.10) * (1 - 0.20) = 0.432 * X := by
  sorry

end final_value_after_three_years_l1733_173381


namespace five_point_eight_one_million_in_scientific_notation_l1733_173318

theorem five_point_eight_one_million_in_scientific_notation :
  5.81 * 10^6 = 5.81e6 :=
sorry

end five_point_eight_one_million_in_scientific_notation_l1733_173318


namespace find_common_difference_l1733_173364

theorem find_common_difference (a a_n S_n : ℝ) (h1 : a = 3) (h2 : a_n = 50) (h3 : S_n = 318) : 
  ∃ d n, (a + (n - 1) * d = a_n) ∧ (n / 2 * (a + a_n) = S_n) ∧ (d = 47 / 11) := 
by
  sorry

end find_common_difference_l1733_173364


namespace third_derivative_y_l1733_173310

noncomputable def y (x : ℝ) : ℝ := x * Real.cos (x^2)

theorem third_derivative_y (x : ℝ) :
  (deriv^[3] y) x = (8 * x^4 - 6) * Real.sin (x^2) - 24 * x^2 * Real.cos (x^2) :=
by
  sorry

end third_derivative_y_l1733_173310


namespace identify_mathematicians_l1733_173376

def famous_people := List (Nat × String)

def is_mathematician : Nat → Bool
| 1 => false  -- Bill Gates
| 2 => true   -- Gauss
| 3 => false  -- Yuan Longping
| 4 => false  -- Nobel
| 5 => true   -- Chen Jingrun
| 6 => true   -- Hua Luogeng
| 7 => false  -- Gorky
| 8 => false  -- Einstein
| _ => false  -- default case

theorem identify_mathematicians (people : famous_people) : 
  (people.filter (fun (n, _) => is_mathematician n)) = [(2, "Gauss"), (5, "Chen Jingrun"), (6, "Hua Luogeng")] :=
by sorry

end identify_mathematicians_l1733_173376


namespace square_number_increased_decreased_by_five_remains_square_l1733_173394

theorem square_number_increased_decreased_by_five_remains_square :
  ∃ x : ℤ, ∃ u v : ℤ, x^2 + 5 = u^2 ∧ x^2 - 5 = v^2 := by
  sorry

end square_number_increased_decreased_by_five_remains_square_l1733_173394


namespace full_price_ticket_revenue_l1733_173314

theorem full_price_ticket_revenue (f d : ℕ) (p : ℝ) : 
  f + d = 200 → 
  f * p + d * (p / 3) = 3000 → 
  d = 200 - f → 
  (f * p) = 1500 := 
by
  intros h1 h2 h3
  sorry

end full_price_ticket_revenue_l1733_173314


namespace right_triangle_third_side_l1733_173309

theorem right_triangle_third_side (a b : ℝ) (h₁ : a = 3) (h₂ : b = 5) (h₃ : a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2) :
  c = Real.sqrt (a^2 + b^2) ∨ c = Real.sqrt (b^2 - a^2) :=
by 
  sorry

end right_triangle_third_side_l1733_173309


namespace line_tangent_to_circle_perpendicular_l1733_173362

theorem line_tangent_to_circle_perpendicular 
  (l₁ l₂ : String)
  (C : String)
  (h1 : l₂ = "4 * x - 3 * y + 1 = 0")
  (h2 : C = "x^2 + y^2 + 2 * y - 3 = 0") :
  (l₁ = "3 * x + 4 * y + 14 = 0" ∨ l₁ = "3 * x + 4 * y - 6 = 0") :=
by
  sorry

end line_tangent_to_circle_perpendicular_l1733_173362


namespace triangle_area_proof_l1733_173370

noncomputable def cos_fun1 (x : ℝ) : ℝ := 2 * Real.cos (3 * x) + 1
noncomputable def cos_fun2 (x : ℝ) : ℝ := - Real.cos (2 * x)

theorem triangle_area_proof :
  let P := (5 * Real.pi, cos_fun1 (5 * Real.pi))
  let Q := (9 * Real.pi / 2, cos_fun2 (9 * Real.pi / 2))
  let m := (Q.snd - P.snd) / (Q.fst - P.fst)
  let y_intercept := P.snd - m * P.fst
  let y_intercept_point := (0, y_intercept)
  let x_intercept := -y_intercept / m
  let x_intercept_point := (x_intercept, 0)
  let base := x_intercept
  let height := y_intercept
  17 * Real.pi / 4 ≤ P.fst ∧ P.fst ≤ 21 * Real.pi / 4 ∧
  17 * Real.pi / 4 ≤ Q.fst ∧ Q.fst ≤ 21 * Real.pi / 4 ∧
  (P.fst = 5 * Real.pi ∧ Q.fst = 9 * Real.pi / 2) →
  1/2 * base * height = 361 * Real.pi / 8 :=
by
  sorry

end triangle_area_proof_l1733_173370


namespace equilibrium_shift_if_K_changes_l1733_173332

-- Define the equilibrium constant and its relation to temperature
def equilibrium_constant (T : ℝ) : ℝ := sorry

-- Define the conditions
axiom K_related_to_temp (T₁ T₂ : ℝ) (K₁ K₂ : ℝ) :
  equilibrium_constant T₁ = K₁ ∧ equilibrium_constant T₂ = K₂ → T₁ = T₂ ↔ K₁ = K₂

axiom K_constant_with_concentration_change (T : ℝ) (K : ℝ) (c₁ c₂ : ℝ) :
  equilibrium_constant T = K → equilibrium_constant T = K

axiom K_squared_with_stoichiometric_double (T : ℝ) (K : ℝ) :
  equilibrium_constant (2 * T) = K * K

-- Define the problem to be proved
theorem equilibrium_shift_if_K_changes (T₁ T₂ : ℝ) (K₁ K₂ : ℝ) :
  equilibrium_constant T₁ = K₁ ∧ equilibrium_constant T₂ = K₂ → K₁ ≠ K₂ → T₁ ≠ T₂ := 
sorry

end equilibrium_shift_if_K_changes_l1733_173332


namespace sum_of_two_numbers_l1733_173320

theorem sum_of_two_numbers (a b : ℝ) (h1 : a * b = 16) (h2 : (1 / a) = 3 * (1 / b)) (ha : 0 < a) (hb : 0 < b) :
  a + b = 16 * Real.sqrt 3 / 3 :=
by {
  sorry
}

end sum_of_two_numbers_l1733_173320


namespace solution_set_of_quadratic_inequality_l1733_173347

theorem solution_set_of_quadratic_inequality (x : ℝ) : 
  x^2 + 3*x - 4 < 0 ↔ -4 < x ∧ x < 1 :=
sorry

end solution_set_of_quadratic_inequality_l1733_173347


namespace hiker_miles_l1733_173356

-- Defining the conditions as a def
def total_steps (flips : ℕ) (additional_steps : ℕ) : ℕ := flips * 100000 + additional_steps

def steps_per_mile : ℕ := 1500

-- The target theorem to prove the number of miles walked
theorem hiker_miles (flips : ℕ) (additional_steps : ℕ) (s_per_mile : ℕ) 
  (h_flips : flips = 72) (h_additional_steps : additional_steps = 25370) 
  (h_s_per_mile : s_per_mile = 1500) : 
  (total_steps flips additional_steps) / s_per_mile = 4817 :=
by
  -- sorry is used to skip the actual proof
  sorry

end hiker_miles_l1733_173356


namespace range_of_a_l1733_173300

theorem range_of_a (a : ℝ) : 
(∀ x : ℝ, |x - 1| + |x - 3| > a ^ 2 - 2 * a - 1) ↔ -1 < a ∧ a < 3 := 
sorry

end range_of_a_l1733_173300


namespace probability_more_sons_or_daughters_correct_l1733_173396

noncomputable def probability_more_sons_or_daughters : ℚ :=
  let total_combinations := (2 : ℕ) ^ 8
  let equal_sons_daughters := Nat.choose 8 4
  let more_sons_or_daughters := total_combinations - equal_sons_daughters
  more_sons_or_daughters / total_combinations

theorem probability_more_sons_or_daughters_correct :
  probability_more_sons_or_daughters = 93 / 128 := by
  sorry 

end probability_more_sons_or_daughters_correct_l1733_173396


namespace perpendicular_lines_solve_for_a_l1733_173367

theorem perpendicular_lines_solve_for_a :
  ∀ (a : ℝ), 
  ((3 * a + 2) * (5 * a - 2) + (1 - 4 * a) * (a + 4) = 0) → 
  (a = 0 ∨ a = 1) :=
by
  intro a h
  sorry

end perpendicular_lines_solve_for_a_l1733_173367


namespace divisibility_condition_of_exponents_l1733_173307

theorem divisibility_condition_of_exponents (n : ℕ) (h : n ≥ 1) :
  (∀ a b : ℕ, (11 ∣ a^n + b^n) → (11 ∣ a ∧ 11 ∣ b)) ↔ (n % 2 = 0) :=
sorry

end divisibility_condition_of_exponents_l1733_173307


namespace fraction_of_40_l1733_173334

theorem fraction_of_40 : (3 / 4) * 40 = 30 :=
by
  -- We'll add the 'sorry' here to indicate that this is the proof part which is not required.
  sorry

end fraction_of_40_l1733_173334


namespace lines_perpendicular_l1733_173301

theorem lines_perpendicular (A1 B1 C1 A2 B2 C2 : ℝ) (h : A1 * A2 + B1 * B2 = 0) :
  ∃(x y : ℝ), A1 * x + B1 * y + C1 = 0 ∧ A2 * x + B2 * y + C2 = 0 → A1 * A2 + B1 * B2 = 0 :=
by
  sorry

end lines_perpendicular_l1733_173301


namespace hazel_drank_one_cup_l1733_173319

theorem hazel_drank_one_cup (total_cups made_to_crew bike_sold friends_given remaining_cups : ℕ) 
  (H1 : total_cups = 56)
  (H2 : made_to_crew = total_cups / 2)
  (H3 : bike_sold = 18)
  (H4 : friends_given = bike_sold / 2)
  (H5 : remaining_cups = total_cups - (made_to_crew + bike_sold + friends_given)) :
  remaining_cups = 1 := 
sorry

end hazel_drank_one_cup_l1733_173319


namespace determine_ordered_triple_l1733_173351

open Real

theorem determine_ordered_triple (a b c : ℝ) (h₁ : 5 < a) (h₂ : 5 < b) (h₃ : 5 < c) 
  (h₄ : (a + 3)^2 / (b + c - 3) + (b + 6)^2 / (c + a - 6) + (c + 9)^2 / (a + b - 9) = 81) : 
  a = 15 ∧ b = 12 ∧ c = 9 := 
sorry

end determine_ordered_triple_l1733_173351


namespace initial_average_mark_l1733_173348

theorem initial_average_mark (A : ℕ) (A_excluded : ℕ := 20) (A_remaining : ℕ := 90) (n_total : ℕ := 14) (n_excluded : ℕ := 5) :
    (n_total * A = n_excluded * A_excluded + (n_total - n_excluded) * A_remaining) → A = 65 :=
by 
  intros h
  sorry

end initial_average_mark_l1733_173348


namespace problem_statement_l1733_173311

open BigOperators

-- Defining the arithmetic sequence
def a (n : ℕ) : ℕ := n - 1

-- Defining the sequence b_n
def b (n : ℕ) : ℕ :=
if n % 2 = 1 then
  a n + 1
else
  2 ^ a n

-- Defining T_2n as the sum of the first 2n terms of b
def T (n : ℕ) : ℕ :=
(∑ i in Finset.range n, b (2 * i + 1)) +
(∑ i in Finset.range n, b (2 * i + 2))

-- The theorem to be proven
theorem problem_statement (n : ℕ) : 
  a 2 * (a 4 + 1) = a 3 ^ 2 ∧
  T n = n^2 + (2^(2*n+1) - 2) / 3 :=
by
  sorry

end problem_statement_l1733_173311


namespace complement_A_in_U_l1733_173398

def U : Set ℕ := {x | x ≥ 2}
def A : Set ℕ := {x | x^2 ≥ 5}

theorem complement_A_in_U : (U \ A) = {2} := by
  sorry

end complement_A_in_U_l1733_173398


namespace smaller_of_two_numbers_l1733_173305

theorem smaller_of_two_numbers 
  (a b d : ℝ) (h : 0 < a ∧ a < b) (u v : ℝ) 
  (huv : u / v = b / a) (sum_uv : u + v = d) : 
  min u v = (a * d) / (a + b) :=
by
  sorry

end smaller_of_two_numbers_l1733_173305


namespace degrees_to_radians_300_l1733_173330

theorem degrees_to_radians_300:
  (300 * (Real.pi / 180) = 5 * Real.pi / 3) := 
by
  repeat { sorry }

end degrees_to_radians_300_l1733_173330


namespace solve_for_x_l1733_173383

theorem solve_for_x : ∃ x : ℚ, 7 * (4 * x + 3) - 3 = -3 * (2 - 5 * x) + 5 * x / 2 ∧ x = -16 / 7 := by
  sorry

end solve_for_x_l1733_173383


namespace find_ab_l1733_173391

theorem find_ab (A B : Set ℝ) (a b : ℝ) :
  (A = {x | x^2 - 2*x - 3 > 0}) →
  (B = {x | x^2 + a*x + b ≤ 0}) →
  (A ∪ B = Set.univ) → 
  (A ∩ B = {x | 3 < x ∧ x ≤ 4}) →
  a + b = -7 :=
by
  intros
  sorry

end find_ab_l1733_173391


namespace strip_width_l1733_173306

theorem strip_width (w : ℝ) (h_floor : ℝ := 10) (b_floor : ℝ := 8) (area_rug : ℝ := 24) :
  (h_floor - 2 * w) * (b_floor - 2 * w) = area_rug → w = 2 := 
by 
  sorry

end strip_width_l1733_173306


namespace blue_balls_balance_l1733_173386

variables {R B O P : ℝ}

-- Given conditions
def cond1 : 4 * R = 8 * B := sorry
def cond2 : 3 * O = 7 * B := sorry
def cond3 : 8 * B = 6 * P := sorry

-- Proof problem: proving equal balance of 5 red balls, 3 orange balls, and 4 purple balls
theorem blue_balls_balance : 5 * R + 3 * O + 4 * P = (67 / 3) * B :=
by
  sorry

end blue_balls_balance_l1733_173386


namespace number_of_4_digit_numbers_divisible_by_9_l1733_173304

theorem number_of_4_digit_numbers_divisible_by_9 :
  ∃ n : ℕ, (∀ k : ℕ, k ∈ Finset.range n → 1008 + k * 9 ≤ 9999) ∧
           (1008 + (n - 1) * 9 = 9999) ∧
           n = 1000 :=
by
  sorry

end number_of_4_digit_numbers_divisible_by_9_l1733_173304


namespace inequality_a4_b4_c4_l1733_173308

theorem inequality_a4_b4_c4 (a b c : Real) : a^4 + b^4 + c^4 ≥ abc * (a + b + c) := 
by
  sorry

end inequality_a4_b4_c4_l1733_173308


namespace bread_cost_equality_l1733_173373

variable (B : ℝ)
variable (C1 : B + 3 + 2 * B = 9)  -- $3 for butter, 2B for juice, total spent is 9 dollars

theorem bread_cost_equality : B = 2 :=
by
  sorry

end bread_cost_equality_l1733_173373


namespace negation_of_prop_p_l1733_173321

open Classical

variable (p : Prop)

def prop_p := ∀ x : ℝ, x^3 - x^2 + 1 < 0

theorem negation_of_prop_p : ¬prop_p ↔ ∃ x : ℝ, x^3 - x^2 + 1 ≥ 0 := by
  sorry

end negation_of_prop_p_l1733_173321


namespace finished_year_eq_183_l1733_173369

theorem finished_year_eq_183 (x : ℕ) (h1 : x < 200) 
  (h2 : x ^ 13 = 258145266804692077858261512663) : x = 183 :=
sorry

end finished_year_eq_183_l1733_173369


namespace f_x_neg_l1733_173358

noncomputable def f (x : ℝ) : ℝ :=
if x > 0 then x^2 + 1 else -x^2 - 1

theorem f_x_neg (x : ℝ) (h : x < 0) : f x = -x^2 - 1 :=
by
  sorry

end f_x_neg_l1733_173358


namespace abigail_initial_money_l1733_173325

variables (A : ℝ) -- Initial amount of money Abigail had.

-- Conditions
variables (food_rate : ℝ := 0.60) -- 60% spent on food
variables (phone_rate : ℝ := 0.25) -- 25% of the remainder spent on phone bill
variables (entertainment_spent : ℝ := 20) -- $20 spent on entertainment
variables (final_amount : ℝ := 40) -- $40 left after all expenditures

theorem abigail_initial_money :
  (A - (A * food_rate)) * (1 - phone_rate) - entertainment_spent = final_amount → A = 200 :=
by
  intro h
  sorry

end abigail_initial_money_l1733_173325


namespace polygon_area_correct_l1733_173302

noncomputable def polygonArea : ℝ :=
  let x1 := 1
  let y1 := 1
  let x2 := 4
  let y2 := 3
  let x3 := 5
  let y3 := 1
  let x4 := 6
  let y4 := 4
  let x5 := 3
  let y5 := 6
  (1 / 2 : ℝ) * 
  abs ((x1 * y2 + x2 * y3 + x3 * y4 + x4 * y5 + x5 * y1) -
       (y1 * x2 + y2 * x3 + y3 * x4 + y4 * x5 + y5 * x1))

theorem polygon_area_correct : polygonArea = 11.5 := by
  sorry

end polygon_area_correct_l1733_173302


namespace ratio_of_speeds_is_two_l1733_173312

noncomputable def joe_speed : ℝ := 0.266666666667
noncomputable def time : ℝ := 40
noncomputable def total_distance : ℝ := 16

noncomputable def joe_distance : ℝ := joe_speed * time
noncomputable def pete_distance : ℝ := total_distance - joe_distance
noncomputable def pete_speed : ℝ := pete_distance / time

theorem ratio_of_speeds_is_two :
  joe_speed / pete_speed = 2 := by
  sorry

end ratio_of_speeds_is_two_l1733_173312


namespace days_y_worked_l1733_173324

theorem days_y_worked 
  (W : ℝ) 
  (x_days : ℝ) (h1 : x_days = 36)
  (y_days : ℝ) (h2 : y_days = 24)
  (x_remaining_days : ℝ) (h3 : x_remaining_days = 18)
  (d : ℝ) :
  d * (W / y_days) + x_remaining_days * (W / x_days) = W → d = 12 :=
by
  -- Mathematical proof goes here
  sorry

end days_y_worked_l1733_173324


namespace value_of_trig_expr_l1733_173343

theorem value_of_trig_expr : 2 * Real.cos (Real.pi / 12) ^ 2 + 1 = 2 + Real.sqrt 3 / 2 :=
by
  sorry

end value_of_trig_expr_l1733_173343


namespace fraction_given_to_cousin_l1733_173315

theorem fraction_given_to_cousin
  (initial_candies : ℕ)
  (brother_share sister_share : ℕ)
  (eaten_candies left_candies : ℕ)
  (remaining_candies : ℕ)
  (given_to_cousin : ℕ)
  (fraction : ℚ)
  (h1 : initial_candies = 50)
  (h2 : brother_share = 5)
  (h3 : sister_share = 5)
  (h4 : eaten_candies = 12)
  (h5 : left_candies = 18)
  (h6 : initial_candies - brother_share - sister_share = remaining_candies)
  (h7 : remaining_candies - given_to_cousin - eaten_candies = left_candies)
  (h8 : fraction = (given_to_cousin : ℚ) / (remaining_candies : ℚ))
  : fraction = 1 / 4 := 
sorry

end fraction_given_to_cousin_l1733_173315


namespace card_length_l1733_173365

noncomputable def width_card : ℕ := 2
noncomputable def side_poster_board : ℕ := 12
noncomputable def total_cards : ℕ := 24

theorem card_length :
  ∃ (card_length : ℕ),
    (side_poster_board / width_card) * (side_poster_board / card_length) = total_cards ∧ 
    card_length = 3 := by
  sorry

end card_length_l1733_173365


namespace num_supermarkets_us_l1733_173316

noncomputable def num_supermarkets_total : ℕ := 84

noncomputable def us_canada_relationship (C : ℕ) : Prop := C + (C + 10) = num_supermarkets_total

theorem num_supermarkets_us (C : ℕ) (h : us_canada_relationship C) : C + 10 = 47 :=
sorry

end num_supermarkets_us_l1733_173316


namespace probability_of_sequence_123456_l1733_173322

theorem probability_of_sequence_123456 :
  let total_sequences := 66 * 45 * 28 * 15 * 6 * 1     -- Total number of sequences
  let specific_sequences := 1 * 3 * 5 * 7 * 9 * 11        -- Sequences leading to 123456
  specific_sequences / total_sequences = 1 / 720 := by
  let total_sequences := 74919600
  let specific_sequences := 10395
  sorry

end probability_of_sequence_123456_l1733_173322


namespace x_is_perfect_square_l1733_173371

theorem x_is_perfect_square (x y : ℕ) (hxy : x > y) (hdiv : xy ∣ x ^ 2022 + x + y ^ 2) : ∃ n : ℕ, x = n^2 := 
sorry

end x_is_perfect_square_l1733_173371


namespace family_members_l1733_173336

theorem family_members (cost_purify : ℝ) (water_per_person : ℝ) (total_cost : ℝ) 
  (h1 : cost_purify = 1) (h2 : water_per_person = 1 / 2) (h3 : total_cost = 3) : 
  total_cost / (cost_purify * water_per_person) = 6 :=
by
  sorry

end family_members_l1733_173336


namespace smallest_x_for_gx_eq_1024_l1733_173350

noncomputable def g : ℝ → ℝ
  | x => if 2 ≤ x ∧ x ≤ 6 then 2 - |x - 3| else 0

axiom g_property1 : ∀ x : ℝ, 0 < x → g (4 * x) = 4 * g x
axiom g_property2 : ∀ x : ℝ, 2 ≤ x ∧ x ≤ 6 → g x = 2 - |x - 3|
axiom g_2004 : g 2004 = 1024

theorem smallest_x_for_gx_eq_1024 : ∃ x : ℝ, g x = 1024 ∧ ∀ y : ℝ, g y = 1024 → x ≤ y := sorry

end smallest_x_for_gx_eq_1024_l1733_173350


namespace tangent_line_parallel_curve_l1733_173346

def curve (x : ℝ) : ℝ := x^4

def line_parallel_to_curve (l : ℝ → ℝ → Prop) : Prop :=
  ∃ x0 y0 : ℝ, l x0 y0 ∧ curve x0 = y0 ∧ ∀ (x : ℝ), l x (curve x)

theorem tangent_line_parallel_curve :
  ∃ (l : ℝ → ℝ → Prop), line_parallel_to_curve l ∧ ∀ x y, l x y ↔ 8 * x + 16 * y + 3 = 0 :=
by
  sorry

end tangent_line_parallel_curve_l1733_173346


namespace fred_speed_l1733_173397

variable {F : ℝ} -- Fred's speed
variable {T : ℝ} -- Time in hours

-- Conditions
def initial_distance : ℝ := 35
def sam_speed : ℝ := 5
def sam_distance : ℝ := 25
def fred_distance := initial_distance - sam_distance

-- Theorem to prove
theorem fred_speed (h1 : T = sam_distance / sam_speed) (h2 : fred_distance = F * T) :
  F = 2 :=
by
  sorry

end fred_speed_l1733_173397


namespace pencils_count_l1733_173313

theorem pencils_count (P L : ℕ) 
  (h1 : P * 6 = L * 5) 
  (h2 : L = P + 7) : 
  L = 42 :=
by
  sorry

end pencils_count_l1733_173313


namespace angle_expr_correct_l1733_173374

noncomputable def angle_expr : Real :=
  Real.cos (40 * Real.pi / 180) * Real.cos (160 * Real.pi / 180) +
  Real.sin (40 * Real.pi / 180) * Real.sin (20 * Real.pi / 180)

theorem angle_expr_correct : angle_expr = -1 / 2 := 
by 
   sorry

end angle_expr_correct_l1733_173374


namespace purchase_price_l1733_173344

-- Define the context and conditions 
variables (P S : ℝ)
-- Define the conditions
axiom cond1 : S = P + 0.5 * S
axiom cond2 : S - P = 100

-- Define the main theorem
theorem purchase_price : P = 100 :=
by sorry

end purchase_price_l1733_173344


namespace volume_of_one_gram_l1733_173323

theorem volume_of_one_gram (mass_per_cubic_meter : ℕ)
  (kilo_to_grams : ℕ)
  (cubic_meter_to_cubic_centimeters : ℕ)
  (substance_mass : mass_per_cubic_meter = 300)
  (kilo_conv : kilo_to_grams = 1000)
  (cubic_conv : cubic_meter_to_cubic_centimeters = 1000000)
  :
  ∃ v : ℝ, v = cubic_meter_to_cubic_centimeters / (mass_per_cubic_meter * kilo_to_grams) ∧ v = 10 / 3 := 
by 
  sorry

end volume_of_one_gram_l1733_173323


namespace find_c_solution_l1733_173372

theorem find_c_solution {c : ℚ} 
  (h₁ : ∃ x : ℤ, 2 * (x : ℚ)^2 + 17 * x - 55 = 0 ∧ x = ⌊c⌋)
  (h₂ : ∃ x : ℚ, 6 * x^2 - 23 * x + 7 = 0 ∧ 0 ≤ x ∧ x < 1 ∧ x = c - ⌊c⌋) :
  c = -32 / 3 :=
by
  sorry

end find_c_solution_l1733_173372


namespace find_metal_molecular_weight_l1733_173352

noncomputable def molecular_weight_of_metal (compound_mw: ℝ) (oh_mw: ℝ) : ℝ :=
  compound_mw - oh_mw

theorem find_metal_molecular_weight :
  let compound_mw := 171.00
  let oxygen_mw := 16.00
  let hydrogen_mw := 1.01
  let oh_ions := 2
  let oh_mw := oh_ions * (oxygen_mw + hydrogen_mw)
  molecular_weight_of_metal compound_mw oh_mw = 136.98 :=
by
  sorry

end find_metal_molecular_weight_l1733_173352


namespace ellipse_equation_is_correct_line_equation_is_correct_l1733_173389

-- Given conditions
variable (a b e x y : ℝ)
variable (a_pos : 0 < a)
variable (b_pos : 0 < b)
variable (ab_order : b < a)
variable (minor_axis_half_major_axis : 2 * a * (1 / 2) = 2 * b)
variable (right_focus_shortest_distance : a - e = 2 - Real.sqrt 3)
variable (ellipse_equation : a^2 = b^2 + e^2)
variable (m : ℝ)
variable (area_triangle_AOB_is_1 : 1 = 1)

-- Part (I) Prove the equation of ellipse C
theorem ellipse_equation_is_correct :
  (∀ x y : ℝ, (x^2 / 4 + y^2 = 1) ↔ (x^2 / a^2 + y^2 / b^2 = 1)) :=
sorry

-- Part (II) Prove the equation of line l
theorem line_equation_is_correct :
  (∀ x y : ℝ, (y = x + m) ↔ ((y = x + (Real.sqrt 10 / 2)) ∨ (y = x - (Real.sqrt 10 / 2)))) :=
sorry

end ellipse_equation_is_correct_line_equation_is_correct_l1733_173389


namespace largest_divisor_of_difference_of_squares_l1733_173340

theorem largest_divisor_of_difference_of_squares (m n : ℤ) (hm : m % 2 = 1) (hn : n % 2 = 1) (h : n < m) :
  ∃ k, (∀ m n : ℤ, m % 2 = 1 → n % 2 = 1 → n < m → k ∣ (m^2 - n^2)) ∧ (∀ j : ℤ, (∀ m n : ℤ, m % 2 = 1 → n % 2 = 1 → n < m → j ∣ (m^2 - n^2)) → j ≤ k) ∧ k = 8 :=
sorry

end largest_divisor_of_difference_of_squares_l1733_173340


namespace solve_quadratic_for_negative_integer_l1733_173354

theorem solve_quadratic_for_negative_integer (N : ℤ) (h_neg : N < 0) (h_eq : 2 * N^2 + N = 20) : N = -4 :=
sorry

end solve_quadratic_for_negative_integer_l1733_173354


namespace sum_difference_even_odd_l1733_173338

theorem sum_difference_even_odd :
  let x := (100 / 2) * (2 + 200)
  let y := (100 / 2) * (1 + 199)
  x - y = 100 :=
by
  sorry

end sum_difference_even_odd_l1733_173338


namespace abs_inequality_solution_set_l1733_173303

theorem abs_inequality_solution_set (x : ℝ) : (|x - 1| ≥ 5) ↔ (x ≥ 6 ∨ x ≤ -4) := 
by sorry

end abs_inequality_solution_set_l1733_173303


namespace max_k_inequality_l1733_173355

theorem max_k_inequality (a b c : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : 0 < c) :
  ∀ k ≤ 2, ( ( (b - c) ^ 2 * (b + c) / a ) + 
             ( (c - a) ^ 2 * (c + a) / b ) + 
             ( (a - b) ^ 2 * (a + b) / c ) 
             ≥ k * ( a^2 + b^2 + c^2 - a*b - b*c - c*a ) ) :=
by
  sorry

end max_k_inequality_l1733_173355


namespace find_x_l1733_173341

theorem find_x :
  (x : ℝ) →
  (0.40 * 2 = 0.25 * (0.30 * 15 + x)) →
  x = -1.3 :=
by
  intros x h
  sorry

end find_x_l1733_173341


namespace instantaneous_velocity_at_1_l1733_173366

noncomputable def h (t : ℝ) : ℝ := -4.9 * t^2 + 6.5 * t + 10

theorem instantaneous_velocity_at_1 :
  (deriv h 1) = -3.3 :=
by
  sorry

end instantaneous_velocity_at_1_l1733_173366


namespace non_shaded_region_perimeter_l1733_173327

def outer_rectangle_length : ℕ := 12
def outer_rectangle_width : ℕ := 10
def inner_rectangle_length : ℕ := 6
def inner_rectangle_width : ℕ := 2
def shaded_area : ℕ := 116

theorem non_shaded_region_perimeter :
  let total_area := outer_rectangle_length * outer_rectangle_width
  let inner_area := inner_rectangle_length * inner_rectangle_width
  let non_shaded_area := total_area - shaded_area
  non_shaded_area = 4 →
  ∃ width height, width * height = non_shaded_area ∧ 2 * (width + height) = 10 :=
by intros
   sorry

end non_shaded_region_perimeter_l1733_173327


namespace most_appropriate_method_to_solve_4x2_minus_9_eq_0_l1733_173392

theorem most_appropriate_method_to_solve_4x2_minus_9_eq_0 :
  (∀ x : ℤ, 4 * x^2 - 9 = 0 ↔ x = 3 / 2 ∨ x = -3 / 2) → true :=
by
  sorry

end most_appropriate_method_to_solve_4x2_minus_9_eq_0_l1733_173392


namespace solve_sum_of_coefficients_l1733_173395

theorem solve_sum_of_coefficients (a b : ℝ) 
  (h1 : ∀ x, ax^2 - bx + 2 > 0 ↔ -1/2 < x ∧ x < 1/3) : a + b = -10 :=
  sorry

end solve_sum_of_coefficients_l1733_173395


namespace q_is_necessary_but_not_sufficient_for_p_l1733_173377

theorem q_is_necessary_but_not_sufficient_for_p (a : ℝ) : 
  (∀ x : ℝ, (x^2 + 2*x + 1 - a^2 < 0) → (-1 + a < x ∧ x < -1 - a)) → (a < 1) ∧ (¬ (a < 1 → (∀ x : ℝ, (x^2 + 2*x + 1 - a^2 < 0) → (-1 + a < x ∧ x < -1 - a)))) :=
by
  sorry

end q_is_necessary_but_not_sufficient_for_p_l1733_173377


namespace initial_volume_of_solution_l1733_173375

theorem initial_volume_of_solution (V : ℝ) :
  (∀ (init_vol : ℝ), 0.84 * init_vol / (init_vol + 26.9) = 0.58) →
  V = 60 :=
by
  intro h
  sorry

end initial_volume_of_solution_l1733_173375


namespace student_solves_exactly_20_problems_l1733_173342

theorem student_solves_exactly_20_problems :
  (∀ n, 1 ≤ (a : ℕ → ℕ) n) ∧ (∀ k, a (k + 7) ≤ a k + 12) ∧ a 77 ≤ 132 →
  ∃ i j, i < j ∧ a j - a i = 20 := sorry

end student_solves_exactly_20_problems_l1733_173342


namespace complex_modulus_problem_l1733_173345

noncomputable def imaginary_unit : ℂ := Complex.I

theorem complex_modulus_problem (z : ℂ) (h : (1 + Real.sqrt 3 * imaginary_unit)^2 * z = 1 - imaginary_unit^3) :
  Complex.abs z = Real.sqrt 2 / 4 :=
by
  sorry

end complex_modulus_problem_l1733_173345


namespace rectangle_perimeter_eq_l1733_173379

noncomputable def rectangle_perimeter (x y : ℝ) := 2 * (x + y)

theorem rectangle_perimeter_eq (x y a b : ℝ)
  (h_area_rect : x * y = 2450)
  (h_area_ellipse : a * b = 2450)
  (h_foci_distance : x + y = 2 * a)
  (h_diag : x^2 + y^2 = 4 * (a^2 - b^2))
  (h_b : b = Real.sqrt (a^2 - 1225))
  : rectangle_perimeter x y = 120 * Real.sqrt 17 := by
  sorry

end rectangle_perimeter_eq_l1733_173379


namespace closest_perfect_square_to_350_l1733_173378

theorem closest_perfect_square_to_350 : 
  ∃ n : ℤ, n^2 < 350 ∧ 350 < (n + 1)^2 ∧ (350 - n^2 ≤ (n + 1)^2 - 350) ∨ (350 - n^2 ≥ (n + 1)^2 - 350) ∧ 
  (if (350 - n^2 < (n + 1)^2 - 350) then n+1 else n) = 19 := 
by
  sorry

end closest_perfect_square_to_350_l1733_173378


namespace total_amount_correct_l1733_173353

noncomputable def total_amount : ℝ :=
  let nissin_noodles := 24 * 1.80 * 0.80
  let master_kong_tea := 6 * 1.70 * 0.80
  let shanlin_soup := 5 * 3.40
  let shuanghui_sausage := 3 * 11.20 * 0.90
  nissin_noodles + master_kong_tea + shanlin_soup + shuanghui_sausage

theorem total_amount_correct : total_amount = 89.96 := by
  sorry

end total_amount_correct_l1733_173353


namespace necessary_but_not_sufficient_condition_for_purely_imaginary_l1733_173357

theorem necessary_but_not_sufficient_condition_for_purely_imaginary (m : ℂ) :
  (1 - m^2 + (1 + m) * Complex.I = 0 → m = 1) ∧ 
  ((1 - m^2 + (1 + m) * Complex.I = 0 ↔ m = 1) = false) := by
  sorry

end necessary_but_not_sufficient_condition_for_purely_imaginary_l1733_173357


namespace price_reduction_l1733_173337

theorem price_reduction (x y : ℕ) (h1 : (13 - x) * y = 781) (h2 : y ≤ 100) : x = 2 :=
sorry

end price_reduction_l1733_173337


namespace triangle_angle_sum_l1733_173331

theorem triangle_angle_sum (CD CB : ℝ) 
    (isosceles_triangle: CD = CB)
    (interior_pentagon_angle: 108 = 180 * (5 - 2) / 5)
    (interior_triangle_angle: 60 = 180 / 3)
    (triangle_angle_sum: ∀ (a b c : ℝ), a + b + c = 180) :
    mangle_CDB = 6 :=
by
  have x : ℝ := 6
  sorry

end triangle_angle_sum_l1733_173331


namespace distance_x_intercepts_correct_l1733_173326

noncomputable def distance_between_x_intercepts : ℝ :=
  let slope1 : ℝ := 4
  let slope2 : ℝ := -3
  let intercept_point : Prod ℝ ℝ := (8, 20)
  let line1 (x : ℝ) : ℝ := slope1 * (x - intercept_point.1) + intercept_point.2
  let line2 (x : ℝ) : ℝ := slope2 * (x - intercept_point.1) + intercept_point.2
  let x_intercept1 : ℝ := (0 - intercept_point.2) / slope1 + intercept_point.1
  let x_intercept2 : ℝ := (0 - intercept_point.2) / slope2 + intercept_point.1
  abs (x_intercept2 - x_intercept1)

theorem distance_x_intercepts_correct :
  distance_between_x_intercepts = 35 / 3 :=
sorry

end distance_x_intercepts_correct_l1733_173326


namespace phase_shift_right_by_pi_div_3_l1733_173399

noncomputable def graph_shift_right_by_pi_div_3 
  (A : ℝ := 1) 
  (ω : ℝ := 1) 
  (φ : ℝ := - (Real.pi / 3)) 
  (y : ℝ → ℝ := fun x => Real.sin (x - Real.pi / 3)) : 
  Prop :=
  y = fun x => Real.sin (x - (Real.pi / 3))

theorem phase_shift_right_by_pi_div_3 (A : ℝ := 1) (ω : ℝ := 1) (φ : ℝ := - (Real.pi / 3)) :
  graph_shift_right_by_pi_div_3 A ω φ (fun x => Real.sin (x - Real.pi / 3)) :=
sorry

end phase_shift_right_by_pi_div_3_l1733_173399


namespace negation_of_existence_l1733_173317

theorem negation_of_existence : 
  (¬ ∃ x_0 : ℝ, (x_0 + 1 < 0) ∨ (x_0^2 - x_0 > 0)) ↔ ∀ x : ℝ, (x + 1 ≥ 0) ∧ (x^2 - x ≤ 0) := 
by
  sorry

end negation_of_existence_l1733_173317


namespace triangle_acute_angle_sufficient_condition_triangle_acute_angle_not_necessary_condition_l1733_173335

theorem triangle_acute_angle_sufficient_condition
  (a b c : ℝ) (h_triangle : a + b > c ∧ a + c > b ∧ b + c > a) :
  a ≤ (b + c) / 2 → b^2 + c^2 > a^2 :=
sorry

theorem triangle_acute_angle_not_necessary_condition
  (a b c : ℝ) (h_triangle : a + b > c ∧ a + c > b ∧ b + c > a) :
  b^2 + c^2 > a^2 → ¬ (a ≤ (b + c) / 2) :=
sorry

end triangle_acute_angle_sufficient_condition_triangle_acute_angle_not_necessary_condition_l1733_173335


namespace g_neg6_eq_neg1_l1733_173393

def f : ℝ → ℝ := fun x => 4 * x - 6
def g : ℝ → ℝ := fun x => 2 * x^2 + 7 * x - 1

theorem g_neg6_eq_neg1 : g (-6) = -1 := by
  sorry

end g_neg6_eq_neg1_l1733_173393


namespace volume_ratio_of_cubes_l1733_173384

theorem volume_ratio_of_cubes 
  (P_A P_B : ℕ) 
  (h_A : P_A = 40) 
  (h_B : P_B = 64) : 
  (∃ s_A s_B V_A V_B, 
    s_A = P_A / 4 ∧ 
    s_B = P_B / 4 ∧ 
    V_A = s_A^3 ∧ 
    V_B = s_B^3 ∧ 
    (V_A : ℚ) / V_B = 125 / 512) := 
by
  sorry

end volume_ratio_of_cubes_l1733_173384


namespace biscuits_per_guest_correct_l1733_173328

def flour_per_batch : ℚ := 5 / 4
def biscuits_per_batch : ℕ := 9
def flour_needed : ℚ := 5
def guests : ℕ := 18

theorem biscuits_per_guest_correct :
  (flour_needed * biscuits_per_batch / flour_per_batch) / guests = 2 := by
  sorry

end biscuits_per_guest_correct_l1733_173328


namespace perimeter_of_wheel_K_l1733_173387

theorem perimeter_of_wheel_K
  (L_turns_K : 4 / 5 = 1 / (length_of_K / length_of_L))
  (L_turns_M : 6 / 7 = 1 / (length_of_L / length_of_M))
  (M_perimeter : length_of_M = 30) :
  length_of_K = 28 := 
sorry

end perimeter_of_wheel_K_l1733_173387
