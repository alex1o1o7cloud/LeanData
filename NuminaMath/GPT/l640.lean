import Mathlib

namespace number_of_primes_between_30_and_50_l640_64061

/-- 
  Prove that there are exactly 5 prime numbers in the range from 30 to 50. 
  These primes are 31, 37, 41, 43, and 47.
-/
theorem number_of_primes_between_30_and_50 : 
  (Finset.filter Nat.Prime (Finset.range 51)).card - 
  (Finset.filter Nat.Prime (Finset.range 31)).card = 5 := 
by 
  sorry

end number_of_primes_between_30_and_50_l640_64061


namespace find_certain_number_l640_64023

theorem find_certain_number
  (t b c : ℝ)
  (average1 : (t + b + c + 14 + 15) / 5 = 12)
  (average2 : (t + b + c + x) / 4 = 15)
  (x : ℝ) :
  x = 29 :=
by
  sorry

end find_certain_number_l640_64023


namespace equation_B_no_real_solution_l640_64066

theorem equation_B_no_real_solution : ∀ x : ℝ, |3 * x + 1| + 6 ≠ 0 := 
by 
  sorry

end equation_B_no_real_solution_l640_64066


namespace percentage_of_circle_outside_triangle_l640_64006

theorem percentage_of_circle_outside_triangle (A : ℝ)
  (h₁ : 0 < A) -- Total area A is positive
  (A_inter : ℝ) (A_outside_tri : ℝ) (A_total_circle : ℝ)
  (h₂ : A_inter = 0.45 * A)
  (h₃ : A_outside_tri = 0.40 * A)
  (h₄ : A_total_circle = 0.60 * A) :
  100 * (1 - A_inter / A_total_circle) = 25 :=
by
  sorry

end percentage_of_circle_outside_triangle_l640_64006


namespace yoque_payment_months_l640_64063

-- Define the conditions
def monthly_payment : ℝ := 15
def amount_borrowed : ℝ := 150
def total_payment : ℝ := amount_borrowed * 1.1

-- Define the proof problem
theorem yoque_payment_months :
  ∃ (n : ℕ), n * monthly_payment = total_payment :=
by 
  have monthly_payment : ℝ := 15
  have amount_borrowed : ℝ := 150
  have total_payment : ℝ := amount_borrowed * 1.1
  use 11
  sorry

end yoque_payment_months_l640_64063


namespace green_paint_quarts_l640_64064

theorem green_paint_quarts (x : ℕ) (h : 5 * x = 3 * 15) : x = 9 := 
sorry

end green_paint_quarts_l640_64064


namespace dvds_left_l640_64016

-- Define the initial conditions
def owned_dvds : Nat := 13
def sold_dvds : Nat := 6

-- Define the goal
theorem dvds_left (owned_dvds : Nat) (sold_dvds : Nat) : owned_dvds - sold_dvds = 7 :=
by
  sorry

end dvds_left_l640_64016


namespace probability_correct_l640_64018

-- Definitions and conditions
def G : List Char := ['A', 'B', 'C', 'D']

-- Number of favorable arrangements where A is adjacent to B and C
def favorable_arrangements : ℕ := 4  -- ABCD, BCDA, DABC, and CDAB

-- Total possible arrangements of 4 people
def total_arrangements : ℕ := 24  -- 4!

-- Probability calculation
def probability_A_adjacent_B_C : ℚ := favorable_arrangements / total_arrangements

-- Prove that this probability equals 1/6
theorem probability_correct : probability_A_adjacent_B_C = 1 / 6 := by
  sorry

end probability_correct_l640_64018


namespace ensure_mixed_tablets_l640_64089

theorem ensure_mixed_tablets (A B : ℕ) (total : ℕ) (hA : A = 10) (hB : B = 16) (htotal : total = 18) :
  ∃ (a b : ℕ), a + b = total ∧ a ≤ A ∧ b ≤ B ∧ a > 0 ∧ b > 0 :=
by
  sorry

end ensure_mixed_tablets_l640_64089


namespace total_notebooks_distributed_l640_64093

theorem total_notebooks_distributed :
  ∀ (N C : ℕ), 
    (N / C = C / 8) →
    (N = 16 * (C / 2)) →
    N = 512 := 
by
  sorry

end total_notebooks_distributed_l640_64093


namespace pos_real_x_plus_inv_ge_two_l640_64041

theorem pos_real_x_plus_inv_ge_two (x : ℝ) (hx : x > 0) : x + (1 / x) ≥ 2 :=
by
  sorry

end pos_real_x_plus_inv_ge_two_l640_64041


namespace value_of_c_l640_64008

noncomputable def f (x a b c : ℤ) : ℤ := x^3 + a*x^2 + b*x + c

theorem value_of_c (a b c : ℤ) (ha: a ≠ 0) (hb: b ≠ 0) (hc: c ≠ 0)
  (hfa: f a a b c = a^3) (hfb: f b a b c = b^3) : c = 16 := by
    sorry

end value_of_c_l640_64008


namespace unique_solution_for_2_pow_m_plus_1_eq_n_square_l640_64032

theorem unique_solution_for_2_pow_m_plus_1_eq_n_square (m n : ℕ) (hm : 0 < m) (hn : 0 < n) :
  2 ^ m + 1 = n ^ 2 → (m = 3 ∧ n = 3) :=
by {
  sorry
}

end unique_solution_for_2_pow_m_plus_1_eq_n_square_l640_64032


namespace find_num_managers_l640_64005

variable (num_associates : ℕ) (avg_salary_managers avg_salary_associates avg_salary_company : ℚ)
variable (num_managers : ℚ)

-- Define conditions based on given problem
def conditions := 
  num_associates = 75 ∧
  avg_salary_managers = 90000 ∧
  avg_salary_associates = 30000 ∧
  avg_salary_company = 40000

-- Proof problem statement
theorem find_num_managers (h : conditions num_associates avg_salary_managers avg_salary_associates avg_salary_company) :
  num_managers = 15 :=
sorry

end find_num_managers_l640_64005


namespace correctness_of_option_C_l640_64076

-- Define the conditions as hypotheses
variable (x y : ℝ)

def condA : Prop := ∀ x: ℝ, x^3 * x^5 = x^15
def condB : Prop := ∀ x y: ℝ, 2 * x + 3 * y = 5 * x * y
def condC : Prop := ∀ x y: ℝ, 2 * x^2 * (3 * x^2 - 5 * y) = 6 * x^4 - 10 * x^2 * y
def condD : Prop := ∀ x: ℝ, (x - 2)^2 = x^2 - 4

-- State the proof problem is correct
theorem correctness_of_option_C (x y : ℝ) : 2 * x^2 * (3 * x^2 - 5 * y) = 6 * x^4 - 10 * x^2 * y := by
  sorry

end correctness_of_option_C_l640_64076


namespace maria_waist_size_in_cm_l640_64007

noncomputable def waist_size_in_cm (waist_size_inches : ℕ) (extra_inch : ℕ) (inches_per_foot : ℕ) (cm_per_foot : ℕ) : ℚ :=
  let total_inches := waist_size_inches + extra_inch
  let total_feet := (total_inches : ℚ) / inches_per_foot
  total_feet * cm_per_foot

theorem maria_waist_size_in_cm :
  waist_size_in_cm 28 1 12 31 = 74.9 :=
by
  sorry

end maria_waist_size_in_cm_l640_64007


namespace range_of_a_l640_64000

variable (a : ℝ)

def p : Prop :=
  ∃ x : ℝ, x^2 + 2 * a * x + a + 2 = 0

def q : Prop :=
  ∀ x, 1 ≤ x ∧ x ≤ 2 → x^2 - a ≥ 0

theorem range_of_a :
  (p a ∧ q a) → a ≤ -1 := by
  sorry

end range_of_a_l640_64000


namespace cellini_inscription_l640_64095

noncomputable def famous_master_engravings (x: Type) : String :=
  "Эту шкатулку изготовил сын Челлини"

theorem cellini_inscription (x: Type) (created_by_cellini : x) :
  famous_master_engravings x = "Эту шкатулку изготовил сын Челлини" :=
by
  sorry

end cellini_inscription_l640_64095


namespace number_of_blue_pens_minus_red_pens_is_seven_l640_64035

-- Define the problem conditions in Lean
variable (R B K T : ℕ) -- where R is red pens, B is black pens, K is blue pens, T is total pens

-- Define the hypotheses from the problem conditions
def hypotheses :=
  (R = 8) ∧ 
  (B = R + 10) ∧ 
  (T = 41) ∧ 
  (T = R + B + K)

-- Define the theorem we need to prove based on the question and the correct answer
theorem number_of_blue_pens_minus_red_pens_is_seven : 
  hypotheses R B K T → K - R = 7 :=
by 
  intro h
  sorry

end number_of_blue_pens_minus_red_pens_is_seven_l640_64035


namespace find_angle_A_l640_64009

theorem find_angle_A (a b c : ℝ) (A B C : ℝ) (h1 : 2 * Real.sin B = Real.sqrt 3 * b) 
  (h2 : a = 2) (h3 : ∃ area : ℝ, area = Real.sqrt 3 ∧ area = (1 / 2) * b * c * Real.sin A) :
  A = Real.pi / 3 ∧ b = 2 ∧ c = 2 :=
by
  sorry

end find_angle_A_l640_64009


namespace edward_chocolate_l640_64069

theorem edward_chocolate (total_chocolate : ℚ) (num_piles : ℕ) (piles_received_by_Edward : ℕ) :
  total_chocolate = 75 / 7 → num_piles = 5 → piles_received_by_Edward = 2 → 
  (total_chocolate / num_piles) * piles_received_by_Edward = 30 / 7 := 
by
  intros ht hn hp
  sorry

end edward_chocolate_l640_64069


namespace train_speed_l640_64050

theorem train_speed (length_train : ℝ) (time_to_cross : ℝ) (length_bridge : ℝ)
  (h_train : length_train = 100) (h_time : time_to_cross = 12.499)
  (h_bridge : length_bridge = 150) : 
  ((length_train + length_bridge) / time_to_cross * 3.6) = 72 := 
by 
  sorry

end train_speed_l640_64050


namespace range_of_a_l640_64071

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, (0 < x) → (-3^x ≤ a)) ↔ (a ≥ -1) :=
by
  sorry

end range_of_a_l640_64071


namespace symmetry_axis_one_of_cos_2x_minus_sin_2x_l640_64020

noncomputable def symmetry_axis (x : ℝ) : Prop :=
  ∃ k : ℤ, x = (k * Real.pi / 2) - Real.pi / 8

theorem symmetry_axis_one_of_cos_2x_minus_sin_2x :
  symmetry_axis (-Real.pi / 8) :=
by
  use 0
  simp
  sorry

end symmetry_axis_one_of_cos_2x_minus_sin_2x_l640_64020


namespace derivative_of_y_correct_l640_64096

noncomputable def derivative_of_y (x : ℝ) : ℝ :=
  let y := (4^x * (Real.log 4 * Real.sin (4 * x) - 4 * Real.cos (4 * x))) / (16 + (Real.log 4) ^ 2)
  let u := 4^x * (Real.log 4 * Real.sin (4 * x) - 4 * Real.cos (4 * x))
  let v := 16 + (Real.log 4) ^ 2
  let du_dx := (4^x * Real.log 4) * (Real.log 4 * Real.sin (4 * x) - 4 * Real.cos (4 * x)) +
               (4^x) * (4 * Real.log 4 * Real.cos (4 * x) + 16 * Real.sin (4 * x))
  let dv_dx := 0
  (du_dx * v - u * dv_dx) / (v ^ 2)

theorem derivative_of_y_correct (x : ℝ) : derivative_of_y x = 4^x * Real.sin (4 * x) :=
  sorry

end derivative_of_y_correct_l640_64096


namespace find_pairs_l640_64037

theorem find_pairs (a b : ℤ) (ha : a ≥ 1) (hb : b ≥ 1)
  (h1 : (a^2 + b) % (b^2 - a) = 0) 
  (h2 : (b^2 + a) % (a^2 - b) = 0) :
  (a = 2 ∧ b = 2) ∨ (a = 3 ∧ b = 3) ∨ (a = 1 ∧ b = 2) ∨ 
  (a = 2 ∧ b = 1) ∨ (a = 2 ∧ b = 3) ∨ (a = 3 ∧ b = 2) := 
sorry

end find_pairs_l640_64037


namespace solving_linear_equations_count_l640_64025

def total_problems : ℕ := 140
def algebra_percentage : ℝ := 0.40
def algebra_problems := (total_problems : ℝ) * algebra_percentage
def solving_linear_equations_percentage : ℝ := 0.50
def solving_linear_equations_problems := algebra_problems * solving_linear_equations_percentage

theorem solving_linear_equations_count :
  solving_linear_equations_problems = 28 :=
by
  sorry

end solving_linear_equations_count_l640_64025


namespace increasing_function_range_l640_64047

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
if x ≤ 1 then
  -x^2 - a*x - 5
else
  a / x

theorem increasing_function_range (a : ℝ) : 
  (∀ x y : ℝ, x ≤ y → f a x ≤ f a y) ↔ (-3 ≤ a ∧ a ≤ -2) :=
by
  sorry

end increasing_function_range_l640_64047


namespace smallest_positive_b_l640_64002

theorem smallest_positive_b (b : ℕ) : 
  (b % 3 = 2) ∧ 
  (b % 4 = 3) ∧ 
  (b % 5 = 4) ∧ 
  (b % 6 = 5) ↔ 
  b = 59 :=
by
  sorry

end smallest_positive_b_l640_64002


namespace salary_restoration_l640_64097

theorem salary_restoration (S : ℝ) : 
  let reduced_salary := 0.7 * S
  let restore_factor := 1 / 0.7
  let percentage_increase := restore_factor - 1
  percentage_increase * 100 = 42.857 :=
by
  sorry

end salary_restoration_l640_64097


namespace max_distance_m_l640_64010

def circle_eq (x y : ℝ) := x^2 + y^2 - 4*x + 6*y - 3 = 0
def line_eq (m x y : ℝ) := m * x + y + m - 1 = 0
def center_circle (x y : ℝ) := circle_eq x y → (x = 2) ∧ (y = -3)

theorem max_distance_m :
  ∃ m : ℝ, line_eq m (-1) 1 ∧ ∀ x y t u : ℝ, center_circle x y → line_eq m t u → 
  -(4 / 3) * -m = -1 → m = -(3 / 4) :=
sorry

end max_distance_m_l640_64010


namespace tip_calculation_l640_64051

def pizza_price : ℤ := 10
def number_of_pizzas : ℤ := 4
def total_pizza_cost := pizza_price * number_of_pizzas
def bill_given : ℤ := 50
def change_received : ℤ := 5
def total_spent := bill_given - change_received
def tip_given := total_spent - total_pizza_cost

theorem tip_calculation : tip_given = 5 :=
by
  -- skipping the proof
  sorry

end tip_calculation_l640_64051


namespace chess_group_players_l640_64019

theorem chess_group_players (n : ℕ) (H : n * (n - 1) / 2 = 435) : n = 30 :=
by
  sorry

end chess_group_players_l640_64019


namespace urn_contains_specific_balls_after_operations_l640_64048

def initial_red_balls : ℕ := 2
def initial_blue_balls : ℕ := 1
def total_operations : ℕ := 5
def final_red_balls : ℕ := 10
def final_blue_balls : ℕ := 6
def target_probability : ℚ := 16 / 115

noncomputable def urn_proba_result : ℚ := sorry

theorem urn_contains_specific_balls_after_operations :
  urn_proba_result = target_probability := sorry

end urn_contains_specific_balls_after_operations_l640_64048


namespace number_of_pairs_l640_64067

theorem number_of_pairs (f m : ℕ) (n : ℕ) :
  n = 6 →
  (f + m ≤ n) →
  ∃! pairs : ℕ, pairs = 2 :=
by
  intro h1 h2
  sorry

end number_of_pairs_l640_64067


namespace ship_length_l640_64044

theorem ship_length (E S L : ℕ) (h1 : 150 * E = L + 150 * S) (h2 : 90 * E = L - 90 * S) : 
  L = 24 :=
by
  sorry

end ship_length_l640_64044


namespace number_of_boys_l640_64084

theorem number_of_boys (n : ℕ)
  (initial_avg_height : ℕ)
  (incorrect_height : ℕ)
  (correct_height : ℕ)
  (actual_avg_height : ℕ)
  (h1 : initial_avg_height = 184)
  (h2 : incorrect_height = 166)
  (h3 : correct_height = 106)
  (h4 : actual_avg_height = 182)
  (h5 : initial_avg_height * n - (incorrect_height - correct_height) = actual_avg_height * n) :
  n = 30 :=
sorry

end number_of_boys_l640_64084


namespace sum_of_other_endpoint_coordinates_l640_64079

theorem sum_of_other_endpoint_coordinates 
  (A B O : ℝ × ℝ)
  (hA : A = (6, -2)) 
  (hO : O = (3, 5)) 
  (midpoint_formula : (A.1 + B.1) / 2 = O.1 ∧ (A.2 + B.2) / 2 = O.2):
  (B.1 + B.2) = 12 :=
by
  sorry

end sum_of_other_endpoint_coordinates_l640_64079


namespace rhombus_side_length_l640_64073

theorem rhombus_side_length (d1 d2 : ℕ) (h1 : d1 = 24) (h2 : d2 = 70) : 
  ∃ (a : ℕ), a^2 = (d1 / 2)^2 + (d2 / 2)^2 ∧ a = 37 :=
by
  sorry

end rhombus_side_length_l640_64073


namespace corn_height_growth_l640_64026

theorem corn_height_growth (init_height week1_growth week2_growth week3_growth : ℕ)
  (h0 : init_height = 0)
  (h1 : week1_growth = 2)
  (h2 : week2_growth = 2 * week1_growth)
  (h3 : week3_growth = 4 * week2_growth) :
  init_height + week1_growth + week2_growth + week3_growth = 22 :=
by sorry

end corn_height_growth_l640_64026


namespace log_sqrt_defined_in_interval_l640_64013

def defined_interval (x : ℝ) : Prop :=
  ∃ y, y = (5 - x) ∧ y > 0 ∧ (x - 2) ≥ 0

theorem log_sqrt_defined_in_interval {x : ℝ} :
  defined_interval x ↔ (2 < x ∧ x < 5) :=
sorry

end log_sqrt_defined_in_interval_l640_64013


namespace klinker_age_l640_64059

theorem klinker_age (K D : ℕ) (h1 : D = 10) (h2 : K + 15 = 2 * (D + 15)) : K = 35 :=
by
  sorry

end klinker_age_l640_64059


namespace solve_for_x_l640_64077

theorem solve_for_x (x : ℝ) : 0.05 * x + 0.07 * (30 + x) = 15.4 → x = 110.8333333 :=
by
  intro h
  sorry

end solve_for_x_l640_64077


namespace nancy_total_spending_l640_64021

theorem nancy_total_spending :
  let this_month_games := 9
  let this_month_price := 5
  let last_month_games := 8
  let last_month_price := 4
  let next_month_games := 7
  let next_month_price := 6
  let total_cost := (this_month_games * this_month_price) +
                    (last_month_games * last_month_price) +
                    (next_month_games * next_month_price)
  total_cost = 119 :=
by
  sorry

end nancy_total_spending_l640_64021


namespace max_tiles_on_floor_l640_64082

-- Definitions based on the given conditions
def tile_length1 := 35 -- in cm
def tile_length2 := 30 -- in cm
def floor_length := 1000 -- in cm
def floor_width := 210 -- in cm

-- Lean 4 statement for the proof problem
theorem max_tiles_on_floor : 
  (max ((floor_length / tile_length1) * (floor_width / tile_length2))
       ((floor_length / tile_length2) * (floor_width / tile_length1))) = 198 := by
  sorry

end max_tiles_on_floor_l640_64082


namespace fraction_of_number_l640_64046

theorem fraction_of_number (N : ℕ) (hN : N = 180) : 
  (6 + (1 / 2) * (1 / 3) * (1 / 5) * N) = (1 / 25) * N := 
by
  sorry

end fraction_of_number_l640_64046


namespace cubic_solution_l640_64057

theorem cubic_solution (x : ℝ) (h : x^3 + (x + 2)^3 + (x + 4)^3 = (x + 6)^3) : x = 6 :=
by
  sorry

end cubic_solution_l640_64057


namespace two_roots_range_a_l640_64028

noncomputable def piecewise_func (x : ℝ) : ℝ :=
if x ≤ 1 then (1/3) * x + 1 else Real.log x

theorem two_roots_range_a (a : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ piecewise_func x1 = a * x1 ∧ piecewise_func x2 = a * x2) ↔ (1/3 < a ∧ a < 1/Real.exp 1) :=
sorry

end two_roots_range_a_l640_64028


namespace buratino_spent_dollars_l640_64024

theorem buratino_spent_dollars (x y : ℕ) (h1 : x + y = 50) (h2 : 2 * x = 3 * y) : 
  (y * 5 - x * 3) = 10 :=
by
  sorry

end buratino_spent_dollars_l640_64024


namespace school_club_profit_l640_64004

def calculate_profit (bars_bought : ℕ) (cost_per_3_bars : ℚ) (bars_sold : ℕ) (price_per_4_bars : ℚ) : ℚ :=
  let cost_per_bar := cost_per_3_bars / 3
  let total_cost := bars_bought * cost_per_bar
  let price_per_bar := price_per_4_bars / 4
  let total_revenue := bars_sold * price_per_bar
  total_revenue - total_cost

theorem school_club_profit :
  calculate_profit 1200 1.50 1200 2.40 = 120 :=
by sorry

end school_club_profit_l640_64004


namespace remainder_when_divided_by_13_l640_64001

theorem remainder_when_divided_by_13 (N k : ℤ) (h : N = 39 * k + 20) : N % 13 = 7 := by
  sorry

end remainder_when_divided_by_13_l640_64001


namespace min_socks_for_pairs_l640_64083

-- Definitions for conditions
def pairs_of_socks : ℕ := 4
def sizes : ℕ := 2
def colors : ℕ := 2

-- Theorem statement
theorem min_socks_for_pairs : 
  ∃ n, n = 7 ∧ 
  ∀ (socks : ℕ), socks >= pairs_of_socks → socks ≥ 7 :=
sorry

end min_socks_for_pairs_l640_64083


namespace reduction_rate_equation_l640_64070

-- Define the given conditions
def original_price : ℝ := 23
def reduced_price : ℝ := 18.63
def monthly_reduction_rate (x : ℝ) : ℝ := (1 - x) ^ 2

-- Prove that the given equation holds
theorem reduction_rate_equation (x : ℝ) : 
  original_price * monthly_reduction_rate x = reduced_price :=
by
  sorry

end reduction_rate_equation_l640_64070


namespace min_value_fraction_sum_l640_64033

theorem min_value_fraction_sum (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h_eq : 1 = 2 * a + b) :
  (1 / a + 1 / b) ≥ 3 + 2 * Real.sqrt 2 :=
sorry

end min_value_fraction_sum_l640_64033


namespace total_time_pushing_car_l640_64022

theorem total_time_pushing_car :
  let d1 := 3
  let s1 := 6
  let d2 := 3
  let s2 := 3
  let d3 := 4
  let s3 := 8
  let t1 := d1 / s1
  let t2 := d2 / s2
  let t3 := d3 / s3
  (t1 + t2 + t3) = 2 :=
by
  sorry

end total_time_pushing_car_l640_64022


namespace eval_expression_l640_64052

theorem eval_expression : (-3)^5 + 2^(2^3 + 5^2 - 8^2) = -242.999999999535 := by
  sorry

end eval_expression_l640_64052


namespace chessboard_movement_l640_64029

-- Defining the problem as described in the transformed proof problem

theorem chessboard_movement (pieces : Nat) (adjacent_empty_square : Nat → Nat → Bool) (visited_all_squares : Nat → Bool)
  (returns_to_starting_square : Nat → Bool) :
  (∃ (moment : Nat), ∀ (piece : Nat), ¬ returns_to_starting_square piece) :=
by
  -- Here we state that there exists a moment when each piece (checker) is not on its starting square
  sorry

end chessboard_movement_l640_64029


namespace second_box_probability_nth_box_probability_l640_64045

noncomputable def P_A1 : ℚ := 2 / 3
noncomputable def P_A2 : ℚ := 5 / 9
noncomputable def P_An (n : ℕ) : ℚ :=
  1 / 2 * (1 / 3) ^ n + 1 / 2

theorem second_box_probability :
  P_A2 = 5 / 9 := by
  sorry

theorem nth_box_probability (n : ℕ) :
  P_An n = 1 / 2 * (1 / 3) ^ n + 1 / 2 := by
  sorry

end second_box_probability_nth_box_probability_l640_64045


namespace trees_planted_tomorrow_l640_64031

-- Definitions from the conditions
def current_trees := 39
def trees_planted_today := 41
def total_trees := 100

-- Theorem statement matching the proof problem
theorem trees_planted_tomorrow : 
  ∃ (trees_planted_tomorrow : ℕ), current_trees + trees_planted_today + trees_planted_tomorrow = total_trees ∧ trees_planted_tomorrow = 20 := 
by
  sorry

end trees_planted_tomorrow_l640_64031


namespace number_of_ordered_triples_modulo_1000000_l640_64056

def p : ℕ := 2017
def N : ℕ := sorry -- N is the number of ordered triples (a, b, c)

theorem number_of_ordered_triples_modulo_1000000 (N : ℕ) (h : ∀ (a b c : ℕ), 1 ≤ a ∧ a ≤ p * (p - 1) ∧ 1 ≤ b ∧ b ≤ p * (p - 1) ∧ a^b - b^a = p * c → true) : 
  N % 1000000 = 2016 :=
sorry

end number_of_ordered_triples_modulo_1000000_l640_64056


namespace intersection_A_B_l640_64043

def setA : Set ℝ := { x | x^2 - 2*x < 3 }
def setB : Set ℝ := { x | x ≤ 2 }
def setC : Set ℝ := { x | -1 < x ∧ x ≤ 2 }

theorem intersection_A_B :
  (setA ∩ setB) = setC :=
by
  sorry

end intersection_A_B_l640_64043


namespace ashley_friends_ages_correct_sum_l640_64060

noncomputable def ashley_friends_ages_sum : Prop :=
  ∃ (a b c d : ℕ), (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) ∧
                   (1 ≤ a ∧ a ≤ 9 ∧ 1 ≤ b ∧ b ≤ 9 ∧ 1 ≤ c ∧ c ≤ 9 ∧ 1 ≤ d ∧ d ≤ 9) ∧
                   (a * b = 36) ∧ (c * d = 30) ∧ (a + b + c + d = 24)

theorem ashley_friends_ages_correct_sum : ashley_friends_ages_sum := sorry

end ashley_friends_ages_correct_sum_l640_64060


namespace point_B_in_first_quadrant_l640_64055

theorem point_B_in_first_quadrant 
  (a b : ℝ) 
  (h1 : a > 0) 
  (h2 : -b > 0) : 
  (a > 0) ∧ (b > 0) := 
by 
  sorry

end point_B_in_first_quadrant_l640_64055


namespace intersection_A_B_l640_64090

open Set

def set_A : Set ℤ := {x | ∃ k : ℤ, x = 2 * k + 1}
def set_B : Set ℤ := {x | 0 < x ∧ x < 5}

theorem intersection_A_B : set_A ∩ set_B = {1, 3} := 
by 
  sorry

end intersection_A_B_l640_64090


namespace unique_alphabets_count_l640_64081

theorem unique_alphabets_count
  (total_alphabets : ℕ)
  (each_written_times : ℕ)
  (total_written : total_alphabets * each_written_times = 10) :
  total_alphabets = 5 := by
  -- The proof would be filled in here.
  sorry

end unique_alphabets_count_l640_64081


namespace inverse_cos_plus_one_l640_64003

noncomputable def f (x : ℝ) : ℝ := Real.cos x + 1

theorem inverse_cos_plus_one (x : ℝ) (hx : 0 ≤ x ∧ x ≤ 2) :
    f (-(Real.arccos (x - 1))) = x :=
by
  sorry

end inverse_cos_plus_one_l640_64003


namespace complement_of_M_in_U_l640_64015

noncomputable def U : Set ℝ := { x | x^2 - 2 * x - 3 ≤ 0 }
noncomputable def M : Set ℝ := { y | ∃ x, x^2 + y^2 = 1 }

theorem complement_of_M_in_U :
  (U \ M) = { x | 1 < x ∧ x ≤ 3 } :=
by
  sorry

end complement_of_M_in_U_l640_64015


namespace sum_real_imag_parts_l640_64038

noncomputable section

open Complex

theorem sum_real_imag_parts (z : ℂ) (h : z / (1 + 2 * i) = 2 + i) : 
  ((z + 5).re + (z + 5).im) = 0 :=
  by
  sorry

end sum_real_imag_parts_l640_64038


namespace quadratic_minimum_val_l640_64058

theorem quadratic_minimum_val (p q x : ℝ) (hp : p > 0) (hq : q > 0) : 
  (∀ x, x^2 - 2 * p * x + 4 * q ≥ p^2 - 4 * q) := 
by
  sorry

end quadratic_minimum_val_l640_64058


namespace fish_to_rice_value_l640_64099

variable (f l r : ℝ)

theorem fish_to_rice_value (h1 : 5 * f = 3 * l) (h2 : 2 * l = 7 * r) : f = 2.1 * r :=
by
  sorry

end fish_to_rice_value_l640_64099


namespace platform_length_l640_64094

theorem platform_length (length_train : ℝ) (speed_train_kmph : ℝ) (time_sec : ℝ) (length_platform : ℝ) :
  length_train = 1020 → speed_train_kmph = 102 → time_sec = 50 →
  length_platform = (speed_train_kmph * 1000 / 3600) * time_sec - length_train :=
by
  intros
  sorry

end platform_length_l640_64094


namespace increase_productivity_RnD_l640_64091

theorem increase_productivity_RnD :
  let RnD_t := 2640.92
  let ΔAPL_t2 := 0.81
  RnD_t / ΔAPL_t2 = 3260 :=
by
  let RnD_t := 2640.92
  let ΔAPL_t2 := 0.81
  have h : RnD_t / ΔAPL_t2 = 3260 := sorry
  exact h

end increase_productivity_RnD_l640_64091


namespace division_of_5_parts_division_of_7_parts_division_of_8_parts_l640_64088

-- Problem 1: Primary Division of Square into 5 Equal Parts
theorem division_of_5_parts (x : ℝ) (h : x^2 = 1 / 5) : x = Real.sqrt (1 / 5) :=
sorry

-- Problem 2: Primary Division of Square into 7 Equal Parts
theorem division_of_7_parts (x : ℝ) (hx : 196 * x^3 - 294 * x^2 + 128 * x - 15 = 0) : 
  x = (7 + Real.sqrt 19) / 14 :=
sorry

-- Problem 3: Primary Division of Square into 8 Equal Parts
theorem division_of_8_parts (x : ℝ) (hx : 6 * x^2 - 6 * x + 1 = 0) : 
  x = (3 + Real.sqrt 3) / 6 :=
sorry

end division_of_5_parts_division_of_7_parts_division_of_8_parts_l640_64088


namespace range_of_x_l640_64092

theorem range_of_x (x y z : ℝ) (h1 : x ≥ y) (h2 : y ≥ z) (h3 : x + y + z = 1) (h4 : x^2 + y^2 + z^2 = 3) : 1 ≤ x ∧ x ≤ 5 / 3 :=
by
  sorry

end range_of_x_l640_64092


namespace largest_angle_of_triangle_l640_64072

theorem largest_angle_of_triangle
  (a b y : ℝ)
  (h1 : a = 60)
  (h2 : b = 70)
  (h3 : a + b + y = 180) :
  max a (max b y) = b :=
by
  sorry

end largest_angle_of_triangle_l640_64072


namespace stewarts_theorem_l640_64012

theorem stewarts_theorem 
  (a b b₁ a₁ d c : ℝ)
  (h₁ : b * b ≠ 0) 
  (h₂ : a * a ≠ 0) 
  (h₃ : b₁ * b₁ ≠ 0) 
  (h₄ : a₁ * a₁ ≠ 0) 
  (h₅ : d * d ≠ 0) 
  (h₆ : c = a₁ + b₁) :
  b * b * a₁ + a * a * b₁ - d * d * c = a₁ * b₁ * c :=
  sorry

end stewarts_theorem_l640_64012


namespace number_of_families_l640_64049

theorem number_of_families (x : ℕ) (h1 : x + x / 3 = 100) : x = 75 :=
sorry

end number_of_families_l640_64049


namespace max_period_of_function_l640_64053

theorem max_period_of_function (f : ℝ → ℝ) (h1 : ∀ x, f (1 + x) = f (1 - x)) (h2 : ∀ x, f (8 + x) = f (8 - x)) :
  ∃ T > 0, (∀ x, f (x + T) = f x) ∧ (∀ T' > 0, (∀ x, f (x + T') = f x) → T' ≥ 14) ∧ T = 14 :=
sorry

end max_period_of_function_l640_64053


namespace functional_eq_solutions_l640_64085

-- Define the conditions for the problem
def func_equation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x * f y - y * f x) = f (x * y) - x * y

-- Define the two solutions to be proven correct
def f1 (x : ℝ) : ℝ := x
def f2 (x : ℝ) : ℝ := |x|

-- State the main theorem to be proven
theorem functional_eq_solutions (f : ℝ → ℝ) (h : func_equation f) : f = f1 ∨ f = f2 :=
sorry

end functional_eq_solutions_l640_64085


namespace price_of_soda_l640_64086

-- Definitions based on the conditions given in the problem
def initial_amount := 500
def cost_rice := 2 * 20
def cost_wheat_flour := 3 * 25
def remaining_balance := 235
def total_cost := cost_rice + cost_wheat_flour

-- Definition to be proved
theorem price_of_soda : initial_amount - total_cost - remaining_balance = 150 := by
  sorry

end price_of_soda_l640_64086


namespace find_a_range_l640_64027

-- Define propositions p and q
def p (a x : ℝ) : Prop := x^2 - 4 * a * x + 3 * a^2 < 0
def q (x : ℝ) : Prop := (x^2 - x - 6 ≤ 0) ∧ (x^2 + 2 * x - 8 > 0)

-- The main theorem stating the range of a
theorem find_a_range (a : ℝ) (h : ¬(∃ x : ℝ, p a x) → ¬(∃ x : ℝ, q x) ∧ ¬(¬(∃ x : ℝ, q x) → ¬(∃ x : ℝ, p a x))) : 1 < a ∧ a ≤ 2 := sorry

end find_a_range_l640_64027


namespace sophia_lost_pawns_l640_64017

theorem sophia_lost_pawns
    (total_pawns : ℕ := 16)
    (start_pawns_each : ℕ := 8)
    (chloe_lost : ℕ := 1)
    (pawns_left : ℕ := 10)
    (chloe_pawns_left : ℕ := start_pawns_each - chloe_lost) :
    total_pawns = 2 * start_pawns_each → 
    ∃ (sophia_lost : ℕ), sophia_lost = start_pawns_each - (pawns_left - chloe_pawns_left) :=
by 
    intros _ 
    use 5 
    sorry

end sophia_lost_pawns_l640_64017


namespace john_final_price_l640_64030

theorem john_final_price : 
  let goodA_price := 2500
  let goodA_rebate := 0.06 * goodA_price
  let goodA_price_after_rebate := goodA_price - goodA_rebate
  let goodA_sales_tax := 0.10 * goodA_price_after_rebate
  let goodA_final_price := goodA_price_after_rebate + goodA_sales_tax
  
  let goodB_price := 3150
  let goodB_rebate := 0.08 * goodB_price
  let goodB_price_after_rebate := goodB_price - goodB_rebate
  let goodB_sales_tax := 0.12 * goodB_price_after_rebate
  let goodB_final_price := goodB_price_after_rebate + goodB_sales_tax

  let goodC_price := 1000
  let goodC_rebate := 0.05 * goodC_price
  let goodC_price_after_rebate := goodC_price - goodC_rebate
  let goodC_sales_tax := 0.07 * goodC_price_after_rebate
  let goodC_final_price := goodC_price_after_rebate + goodC_sales_tax

  let total_amount := goodA_final_price + goodB_final_price + goodC_final_price

  let special_voucher_discount := 0.03 * total_amount
  let final_price := total_amount - special_voucher_discount
  let rounded_final_price := Float.round final_price

  rounded_final_price = 6642 := by
  sorry

end john_final_price_l640_64030


namespace chocolateBarsPerBox_l640_64036

def numberOfSmallBoxes := 20
def totalChocolateBars := 500

theorem chocolateBarsPerBox : totalChocolateBars / numberOfSmallBoxes = 25 :=
by
  -- Skipping the proof here
  sorry

end chocolateBarsPerBox_l640_64036


namespace calculate_expression_l640_64080

theorem calculate_expression : (3 / 4 - 1 / 8) ^ 5 = 3125 / 32768 :=
by
  sorry

end calculate_expression_l640_64080


namespace correct_parameterization_l640_64042

noncomputable def parametrize_curve (t : ℝ) : ℝ × ℝ :=
  (t, t^2)

theorem correct_parameterization : ∀ t : ℝ, ∃ x y : ℝ, parametrize_curve t = (x, y) ∧ y = x^2 :=
by
  intro t
  use t, t^2
  dsimp [parametrize_curve]
  exact ⟨rfl, rfl⟩

end correct_parameterization_l640_64042


namespace number_of_boxes_l640_64034

-- Definitions based on conditions
def pieces_per_box := 500
def total_pieces := 3000

-- Theorem statement, we need to prove that the number of boxes is 6
theorem number_of_boxes : total_pieces / pieces_per_box = 6 :=
by {
  sorry
}

end number_of_boxes_l640_64034


namespace find_y_l640_64039

theorem find_y (x y : ℝ) (h1 : x = 8) (h2 : x^(3 * y) = 64) : y = 2 / 3 :=
by
  -- Proof omitted
  sorry

end find_y_l640_64039


namespace x_equals_neg_one_l640_64098

theorem x_equals_neg_one
  (a b c : ℝ)
  (h1 : a ≠ 0)
  (h2 : b ≠ 0)
  (h3 : c ≠ 0)
  (h4 : (a + b - c) / c = (a - b + c) / b ∧ (a + b - c) / c = (-a + b + c) / a)
  (x : ℝ)
  (h5 : x = (a + b) * (b + c) * (c + a) / (a * b * c))
  (h6 : x < 0) :
  x = -1 := 
sorry

end x_equals_neg_one_l640_64098


namespace bertha_descendants_no_children_l640_64014

-- Definitions based on the conditions of the problem.
def bertha_daughters : ℕ := 10
def total_descendants : ℕ := 40
def granddaughters : ℕ := total_descendants - bertha_daughters
def daughters_with_children : ℕ := 8
def children_per_daughter_with_children : ℕ := 4
def number_of_granddaughters : ℕ := daughters_with_children * children_per_daughter_with_children
def total_daughters_and_granddaughters : ℕ := bertha_daughters + number_of_granddaughters
def without_children : ℕ := total_daughters_and_granddaughters - daughters_with_children

-- Lean statement to prove the main question given the definitions.
theorem bertha_descendants_no_children : without_children = 34 := by
  -- Placeholder for the proof
  sorry

end bertha_descendants_no_children_l640_64014


namespace appropriate_survey_method_l640_64065

def survey_method_suitability (method : String) (context : String) : Prop :=
  match context, method with
  | "daily floating population of our city", "sampling survey" => true
  | "security checks before passengers board an airplane", "comprehensive survey" => true
  | "killing radius of a batch of shells", "sampling survey" => true
  | "math scores of Class 1 in Grade 7 of a certain school", "census method" => true
  | _, _ => false

theorem appropriate_survey_method :
  survey_method_suitability "census method" "daily floating population of our city" = false ∧
  survey_method_suitability "comprehensive survey" "security checks before passengers board an airplane" = false ∧
  survey_method_suitability "sampling survey" "killing radius of a batch of shells" = false ∧
  survey_method_suitability "census method" "math scores of Class 1 in Grade 7 of a certain school" = true :=
by
  sorry

end appropriate_survey_method_l640_64065


namespace solve_for_x_l640_64068

theorem solve_for_x (x : ℝ) : (2010 + 2 * x) ^ 2 = x ^ 2 → x = -2010 ∨ x = -670 := by
  sorry

end solve_for_x_l640_64068


namespace log2_bounds_l640_64078

noncomputable def log2 (x : ℝ) := Real.log x / Real.log 2

theorem log2_bounds (h1 : 10^3 = 1000) (h2 : 10^4 = 10000) 
  (h3 : 2^10 = 1024) (h4 : 2^11 = 2048) (h5 : 2^12 = 4096) 
  (h6 : 2^13 = 8192) (h7 : 2^14 = 16384) :
  (3 : ℝ) / 10 < log2 10 ∧ log2 10 < (2 : ℝ) / 7 :=
by
  sorry

end log2_bounds_l640_64078


namespace double_bed_heavier_l640_64062

-- Define the problem conditions
variable (S D B : ℝ)
variable (h1 : 5 * S = 50)
variable (h2 : 2 * S + 4 * D + 3 * B = 180)
variable (h3 : 3 * B = 60)

-- Define the goal to prove
theorem double_bed_heavier (S D B : ℝ) (h1 : 5 * S = 50) (h2 : 2 * S + 4 * D + 3 * B = 180) (h3 : 3 * B = 60) : D - S = 15 :=
by
  sorry

end double_bed_heavier_l640_64062


namespace abs_inequality_solution_l640_64074

theorem abs_inequality_solution (x : ℝ) : 
  (|x - 1| < 1) ↔ (0 < x ∧ x < 2) :=
sorry

end abs_inequality_solution_l640_64074


namespace inscribed_polygon_sides_l640_64011

-- We start by defining the conditions of the problem in Lean.
def radius := 1
def side_length_condition (n : ℕ) : Prop :=
  1 < 2 * Real.sin (Real.pi / n) ∧ 2 * Real.sin (Real.pi / n) < Real.sqrt 2

-- Now we state the main theorem.
theorem inscribed_polygon_sides (n : ℕ) (h1 : side_length_condition n) : n = 5 :=
  sorry

end inscribed_polygon_sides_l640_64011


namespace arc_length_of_given_curve_l640_64087

open Real

noncomputable def arc_length (f : ℝ → ℝ) (a b : ℝ) :=
  ∫ x in a..b, sqrt (1 + (deriv f x)^2)

noncomputable def given_function (x : ℝ) : ℝ :=
  arccos (sqrt x) - sqrt (x - x^2) + 4

theorem arc_length_of_given_curve :
  arc_length given_function 0 (1/2) = sqrt 2 :=
by
  sorry

end arc_length_of_given_curve_l640_64087


namespace triangle_area_l640_64075

def point := (ℚ × ℚ)

def vertex1 : point := (3, -3)
def vertex2 : point := (3, 4)
def vertex3 : point := (8, -3)

theorem triangle_area :
  let base := (vertex3.1 - vertex1.1 : ℚ)
  let height := (vertex2.2 - vertex1.2 : ℚ)
  (base * height / 2) = 17.5 :=
by
  sorry

end triangle_area_l640_64075


namespace percentage_increase_proof_l640_64054

def breakfast_calories : ℕ := 500
def shakes_total_calories : ℕ := 3 * 300
def total_daily_calories : ℕ := 3275

noncomputable def percentage_increase_in_calories (P : ℝ) : Prop :=
  let lunch_calories := breakfast_calories * (1 + P / 100)
  let dinner_calories := 2 * lunch_calories
  breakfast_calories + lunch_calories + dinner_calories + shakes_total_calories = total_daily_calories

theorem percentage_increase_proof : percentage_increase_in_calories 125 :=
by
  sorry

end percentage_increase_proof_l640_64054


namespace ways_to_divide_week_l640_64040

-- Define the total number of seconds in a week
def total_seconds_in_week : ℕ := 604800

-- Define the math problem statement
theorem ways_to_divide_week (n m : ℕ) (h : n * m = total_seconds_in_week) (hn : 0 < n) (hm : 0 < m) : 
  (∃ (n_pairs : ℕ), n_pairs = 144) :=
sorry

end ways_to_divide_week_l640_64040
