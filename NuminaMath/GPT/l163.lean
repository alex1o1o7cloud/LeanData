import Mathlib

namespace soccer_league_teams_l163_163899

theorem soccer_league_teams (n : ℕ) (h : n * (n - 1) / 2 = 55) : n = 11 := 
sorry

end soccer_league_teams_l163_163899


namespace factor_x4_minus_64_l163_163792

theorem factor_x4_minus_64 :
  ∀ x : ℝ, (x^4 - 64 = (x - 2 * Real.sqrt 2) * (x + 2 * Real.sqrt 2) * (x^2 + 8)) :=
by
  intro x
  sorry

end factor_x4_minus_64_l163_163792


namespace odd_periodic_function_l163_163180

variable {f : ℝ → ℝ}

-- Given conditions
def odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x
def periodic_function (f : ℝ → ℝ) : Prop := ∀ x, f (x + 1) = -f x

-- Problem statement
theorem odd_periodic_function (h_odd : odd_function f)
  (h_period : periodic_function f) (h_half : f 0.5 = 1) : f 7.5 = -1 :=
sorry

end odd_periodic_function_l163_163180


namespace total_people_l163_163233

-- Definitions of the given conditions
variable (I N B Ne T : ℕ)

-- These variables represent the given conditions
axiom h1 : I = 25
axiom h2 : N = 23
axiom h3 : B = 21
axiom h4 : Ne = 23

-- The theorem we want to prove
theorem total_people : T = 50 :=
by {
  sorry -- We denote the skipping of proof details.
}

end total_people_l163_163233


namespace equality_of_expressions_l163_163140

theorem equality_of_expressions :
  (2^3 ≠ 2 * 3) ∧
  (-(-2)^2 ≠ (-2)^2) ∧
  (-3^2 ≠ 3^2) ∧
  (-2^3 = (-2)^3) :=
by
  sorry

end equality_of_expressions_l163_163140


namespace best_discount_option_l163_163103

-- Define the original price
def original_price : ℝ := 100

-- Define the discount functions for each option
def option_A : ℝ := original_price * (1 - 0.20)
def option_B : ℝ := (original_price * (1 - 0.10)) * (1 - 0.10)
def option_C : ℝ := (original_price * (1 - 0.15)) * (1 - 0.05)
def option_D : ℝ := (original_price * (1 - 0.05)) * (1 - 0.15)

-- Define the theorem stating that option A gives the best price
theorem best_discount_option : option_A ≤ option_B ∧ option_A ≤ option_C ∧ option_A ≤ option_D :=
by {
  sorry
}

end best_discount_option_l163_163103


namespace sum_of_ratio_simplified_l163_163896

theorem sum_of_ratio_simplified (a b c : ℤ) 
  (h1 : (a * a * b) = 75)
  (h2 : (c * c * 2) = 128)
  (h3 : c = 16) :
  a + b + c = 27 := 
by 
  have ha : a = 5 := sorry
  have hb : b = 6 := sorry
  rw [ha, hb, h3]
  norm_num
  exact eq.refl 27

end sum_of_ratio_simplified_l163_163896


namespace sandy_grew_watermelons_l163_163852

-- Definitions for the conditions
def jason_grew_watermelons : ℕ := 37
def total_watermelons : ℕ := 48

-- Define what we want to prove
theorem sandy_grew_watermelons : total_watermelons - jason_grew_watermelons = 11 := by
  sorry

end sandy_grew_watermelons_l163_163852


namespace pond_water_amount_l163_163458

-- Definitions based on the problem conditions
def initial_gallons := 500
def evaporation_rate := 1
def additional_gallons := 10
def days_period := 35
def additional_days_interval := 7

-- Calculations based on the conditions
def total_evaporation := days_period * evaporation_rate
def total_additional_gallons := (days_period / additional_days_interval) * additional_gallons

-- Theorem stating the final amount of water
theorem pond_water_amount : initial_gallons - total_evaporation + total_additional_gallons = 515 := by
  -- Proof is omitted
  sorry

end pond_water_amount_l163_163458


namespace power_root_l163_163471

noncomputable def x : ℝ := 1024 ^ (1 / 5)

theorem power_root (h : 1024 = 2^10) : x = 4 :=
by
  sorry

end power_root_l163_163471


namespace computation_one_computation_two_l163_163329

-- Proof problem (1)
theorem computation_one :
  (-2)^3 + |(-3)| - Real.tan (Real.pi / 4) = -6 := by
  sorry

-- Proof problem (2)
theorem computation_two (a : ℝ) :
  (a + 2)^2 - a * (a - 4) = 8 * a + 4 := by
  sorry

end computation_one_computation_two_l163_163329


namespace number_of_pages_in_each_chapter_l163_163725

variable (x : ℕ)  -- Variable for number of pages in each chapter

-- Definitions based on the problem conditions
def pages_read_before_4_o_clock := 10 * x
def pages_read_at_4_o_clock := 20
def pages_read_after_4_o_clock := 2 * x
def total_pages_read := pages_read_before_4_o_clock x + pages_read_at_4_o_clock + pages_read_after_4_o_clock x

-- The theorem statement
theorem number_of_pages_in_each_chapter (h : total_pages_read x = 500) : x = 40 :=
sorry

end number_of_pages_in_each_chapter_l163_163725


namespace inequality_proof_l163_163964

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (b + c) / a + (c + a) / b + (a + b) / c ≥ ((a ^ 2 + b ^ 2 + c ^ 2) * (a * b + b * c + c * a)) / (a * b * c * (a + b + c)) + 3 := 
by
  -- Adding 'sorry' to indicate the proof is omitted
  sorry

end inequality_proof_l163_163964


namespace min_nS_n_l163_163865

open Function

noncomputable def a (n : ℕ) (a_1 : ℤ) (d : ℤ) : ℤ := a_1 + (n - 1) * d

noncomputable def S (n : ℕ) (a_1 : ℤ) (d : ℤ) : ℤ := n * a_1 + d * n * (n - 1) / 2

theorem min_nS_n (d : ℤ) (h_a7 : ∃ a_1 : ℤ, a 7 a_1 d = 5)
  (h_S5 : ∃ a_1 : ℤ, S 5 a_1 d = -55) :
  ∃ n : ℕ, n > 0 ∧ n * S n a_1 d = -343 :=
by
  sorry

end min_nS_n_l163_163865


namespace xy_equals_18_l163_163984

theorem xy_equals_18 (x y : ℝ) (h : x * (x + y) = x^2 + 18) : x * y = 18 :=
by
  sorry

end xy_equals_18_l163_163984


namespace equality_of_expressions_l163_163141

theorem equality_of_expressions :
  (2^3 ≠ 2 * 3) ∧
  (-(-2)^2 ≠ (-2)^2) ∧
  (-3^2 ≠ 3^2) ∧
  (-2^3 = (-2)^3) :=
by
  sorry

end equality_of_expressions_l163_163141


namespace spatial_quadrilateral_angle_sum_l163_163876

theorem spatial_quadrilateral_angle_sum (A B C D : ℝ) (ABD DBC ADB BDC : ℝ) :
  (A <= ABD + DBC) → (C <= ADB + BDC) → 
  (A + C + B + D <= 360) := 
by
  intros
  sorry

end spatial_quadrilateral_angle_sum_l163_163876


namespace sqrt_factorial_equality_l163_163577

theorem sqrt_factorial_equality : Real.sqrt (4! * 4!) = 24 := 
by
  sorry

end sqrt_factorial_equality_l163_163577


namespace prime_15p_plus_one_l163_163533

open Nat

def is_prime (n : ℕ) : Prop := 
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem prime_15p_plus_one (p q : ℕ) 
  (hp : is_prime p) 
  (hq : q = 15 * p + 1) 
  (hq_prime : is_prime q) :
  q = 31 :=
sorry

end prime_15p_plus_one_l163_163533


namespace probability_both_counterfeit_given_one_counterfeit_l163_163061

-- Conditions
def total_bills := 20
def counterfeit_bills := 5
def selected_bills := 2
def at_least_one_counterfeit := true

-- Definition of events
def eventA := "both selected bills are counterfeit"
def eventB := "at least one of the selected bills is counterfeit"

-- The theorem to prove
theorem probability_both_counterfeit_given_one_counterfeit : 
  at_least_one_counterfeit →
  ( (counterfeit_bills * (counterfeit_bills - 1)) / (total_bills * (total_bills - 1)) ) / 
    ( (counterfeit_bills * (counterfeit_bills - 1) + counterfeit_bills * (total_bills - counterfeit_bills)) / (total_bills * (total_bills - 1)) ) = 2/17 :=
by
  sorry

end probability_both_counterfeit_given_one_counterfeit_l163_163061


namespace eval_p_nested_l163_163527

def p (x y : ℤ) : ℤ :=
  if x ≥ 0 ∧ y ≥ 0 then 2 * x + 3 * y
  else if x < 0 ∧ y < 0 then x ^ 2 - y
  else 4 * x + 2 * y

theorem eval_p_nested :
  p (p 2 (-3)) (p (-4) (-3)) = 61 :=
by
  sorry

end eval_p_nested_l163_163527


namespace prime_square_sum_eq_square_iff_l163_163808

theorem prime_square_sum_eq_square_iff (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q):
  (∃ x y z : ℕ, p^(2*x) + q^(2*y) = z^2) ↔ (p = 2 ∧ q = 3) ∨ (p = 3 ∧ q = 2) := 
sorry

end prime_square_sum_eq_square_iff_l163_163808


namespace minimize_side_length_of_triangle_l163_163084

-- Define a triangle with sides a, b, and c and angle C
structure Triangle :=
  (a b c : ℝ)
  (C : ℝ) -- angle C in radians
  (area : ℝ) -- area of the triangle

-- Define the conditions for the problem
def conditions (T : Triangle) : Prop :=
  T.area > 0 ∧ T.C > 0 ∧ T.C < Real.pi

-- Define the desired result
def min_side_length (T : Triangle) : Prop :=
  T.a = T.b ∧ T.a = Real.sqrt ((2 * T.area) / Real.sin T.C)

-- The theorem to be proven
theorem minimize_side_length_of_triangle (T : Triangle) (h : conditions T) : min_side_length T :=
  sorry

end minimize_side_length_of_triangle_l163_163084


namespace evaluate_expr_correct_l163_163947

def evaluate_expr : Prop :=
  (8 : ℝ) / (4 * 25) = (0.8 : ℝ) / (0.4 * 25)

theorem evaluate_expr_correct : evaluate_expr :=
by
  sorry

end evaluate_expr_correct_l163_163947


namespace find_circle_and_a_l163_163381

-- Definitions of the conditions
def curve (x : ℝ) : ℝ := x ^ 2 - 6 * x + 1

def circle (x y : ℝ) : Prop := (x - 3) ^ 2 + (y - 1) ^ 2 = 9

def line (a : ℝ) (x y : ℝ) : Prop := x - y + a = 0

-- Statement of the proof problem
theorem find_circle_and_a (a : ℝ) :
  (∀ x y, (x = 0 → y = 1 ∨ (y = 0 → (x = 3+2*sqrt 2 ∨ x = 3-2*sqrt 2)))
  → ∀ x1 y1 x2 y2, circle x1 y1 ∧ line a x1 y1 ∧ circle x2 y2 ∧ line a x2 y2
  ∧ (x1 - 0 + y1 - 0 = 0 ∨ x2 - 0 + y2 - 0 = 0) 
  → a = -1) :=
sorry

end find_circle_and_a_l163_163381


namespace sqrt_floor_eq_l163_163276

theorem sqrt_floor_eq (n : ℕ) (hn : 0 < n) :
    ⌊Real.sqrt (4 * n + 1)⌋ = ⌊Real.sqrt (4 * n + 2)⌋ ∧ 
    ⌊Real.sqrt (4 * n + 2)⌋ = ⌊Real.sqrt (4 * n + 3)⌋ ∧ 
    ⌊Real.sqrt (4 * n + 3)⌋ = ⌊Real.sqrt n + Real.sqrt (n + 1)⌋ :=
by
  sorry

end sqrt_floor_eq_l163_163276


namespace ben_last_roll_probability_l163_163369

theorem ben_last_roll_probability :
  let p1 := (3 : ℚ) / 4
  let p2 := (1 : ℚ) / 4
  let P_12th_last := p1^10 * p2
  (P_12th_last).approximate = 0.014 :=
by {
  let p1 := (3 : ℚ) / 4
  let p2 := (1 : ℚ) / 4
  let P_12th_last := p1^10 * p2
  have : P_12th_last = (3^10 : ℚ) / (4^11 : ℚ), by sorry,
  have approx_value : ((3^10 : ℚ) / (4^11 : ℚ)).approximate = 0.014, by sorry,
  exact eq.trans this approx_value
}

end ben_last_roll_probability_l163_163369


namespace sinx_tanx_condition_l163_163222

theorem sinx_tanx_condition (x : ℝ) : (sin x = (Real.sqrt 2) / 2) → (¬ (tan x = 1)) ∧ (tan x = 1 → sin x = (Real.sqrt 2) / 2) :=
by
  sorry

end sinx_tanx_condition_l163_163222


namespace parabola_intersects_x_axis_l163_163287

theorem parabola_intersects_x_axis :
  ∀ m : ℝ, (m^2 - m - 1 = 0) → (-2 * m^2 + 2 * m + 2023 = 2021) :=
by 
intros m hm
/-
  Given condition: m^2 - m - 1 = 0
  We need to show: -2 * m^2 + 2 * m + 2023 = 2021
-/
sorry

end parabola_intersects_x_axis_l163_163287


namespace range_of_m_l163_163066

theorem range_of_m (α : ℝ) (m : ℝ) (h1 : π < α ∧ α < 2 * π ∨ 3 * π < α ∧ α < 4 * π) 
(h2 : Real.sin α = (2 * m - 3) / (4 - m)) : 
  -1 < m ∧ m < (3 : ℝ) / 2 :=
  sorry

end range_of_m_l163_163066


namespace no_real_solution_l163_163794

theorem no_real_solution (x : ℝ) : ¬ (x^3 + 2 * (x + 1)^3 + 3 * (x + 2)^3 = 6 * (x + 4)^3) :=
sorry

end no_real_solution_l163_163794


namespace smallest_possible_N_l163_163761

theorem smallest_possible_N (N : ℕ) (h : ∀ m : ℕ, m ≤ 60 → m % 3 = 0 → ∃ i : ℕ, i < 20 ∧ m = 3 * i + 1 ∧ N = 20) :
    N = 20 :=
by 
  sorry

end smallest_possible_N_l163_163761


namespace trigonometric_expression_value_l163_163204

theorem trigonometric_expression_value (α : ℝ) (h : Real.tan α = 2) :
  (Real.sin α ^ 2 - Real.cos α ^ 2) / (2 * Real.sin α * Real.cos α + Real.cos α ^ 2) = 3 / 5 := 
sorry

end trigonometric_expression_value_l163_163204


namespace vector_coordinates_l163_163212

theorem vector_coordinates :
  let a : ℝ × ℝ := (3, -1)
  let b : ℝ × ℝ := (-1, 2)
  (-3 : ℝ) • a + (-2 : ℝ) • b = (-7, -1) :=
by
  let a : ℝ × ℝ := (3, -1)
  let b : ℝ × ℝ := (-1, 2)
  show (-3 : ℝ) • a + (-2 : ℝ) • b = (-7, -1)
  sorry

end vector_coordinates_l163_163212


namespace sqrt_factorial_product_l163_163599

theorem sqrt_factorial_product :
  Nat.sqrt (Nat.factorial 4 * Nat.factorial 4) = 24 := 
sorry

end sqrt_factorial_product_l163_163599


namespace exist_one_common_ball_l163_163199

theorem exist_one_common_ball (n : ℕ) (h_n : 5 ≤ n) (A : Fin (n+1) → Finset (Fin n))
  (hA_card : ∀ i, (A i).card = 3)
  (h_distinct : ∀ i j, i ≠ j → A i ≠ A j) :
  ∃ (i j : Fin (n+1)), i ≠ j ∧ (A i ∩ A j).card = 1 :=
sorry

end exist_one_common_ball_l163_163199


namespace number_of_cats_l163_163241

theorem number_of_cats (total_animals : ℕ) (dogs : ℕ) (cats : ℕ) 
  (h1 : total_animals = 1212) 
  (h2 : dogs = 567) 
  (h3 : cats = total_animals - dogs) : 
  cats = 645 := 
by 
  sorry

end number_of_cats_l163_163241


namespace total_towels_l163_163431

theorem total_towels (packs : ℕ) (towels_per_pack : ℕ) (h1 : packs = 9) (h2 : towels_per_pack = 3) : packs * towels_per_pack = 27 := by
  sorry

end total_towels_l163_163431


namespace percentage_of_alcohol_in_second_vessel_l163_163925

-- Define the problem conditions
def capacity1 : ℝ := 2
def percentage1 : ℝ := 0.35
def alcohol1 := capacity1 * percentage1

def capacity2 : ℝ := 6 
def percentage2 (x : ℝ) : ℝ := 0.01 * x
def alcohol2 (x : ℝ) := capacity2 * percentage2 x

def total_capacity : ℝ := 8
def final_percentage : ℝ := 0.37
def total_alcohol := total_capacity * final_percentage

theorem percentage_of_alcohol_in_second_vessel (x : ℝ) :
  alcohol1 + alcohol2 x = total_alcohol → x = 37.67 :=
by sorry

end percentage_of_alcohol_in_second_vessel_l163_163925


namespace bus_capacities_rental_plan_l163_163311

variable (x y : ℕ)
variable (m n : ℕ)

theorem bus_capacities :
  3 * x + 2 * y = 195 ∧ 2 * x + 4 * y = 210 → x = 45 ∧ y = 30 :=
by
  sorry

theorem rental_plan :
  7 * m + 3 * n = 20 ∧ m + n ≤ 7 ∧ 65 * m + 45 * n + 30 * (7 - m - n) = 310 →
  m = 2 ∧ n = 2 ∧ 7 - m - n = 3 :=
by
  sorry

end bus_capacities_rental_plan_l163_163311


namespace g_six_g_seven_l163_163417

noncomputable def g : ℝ → ℝ :=
sorry

axiom additivity : ∀ x y : ℝ, g (x + y) = g x + g y
axiom g_three : g 3 = 4

theorem g_six : g 6 = 8 :=
by {
  -- proof steps to be added by the prover
  sorry
}

theorem g_seven : g 7 = 28 / 3 :=
by {
  -- proof steps to be added by the prover
  sorry
}

end g_six_g_seven_l163_163417


namespace edge_length_is_correct_l163_163128

-- Define the given conditions
def volume_material : ℕ := 12 * 18 * 6
def edge_length : ℕ := 3
def number_cubes : ℕ := 48
def volume_cube (e : ℕ) : ℕ := e * e * e

-- Problem statement in Lean:
theorem edge_length_is_correct : volume_material = number_cubes * volume_cube edge_length → edge_length = 3 :=
by
  sorry

end edge_length_is_correct_l163_163128


namespace negation_ln_eq_x_minus_1_l163_163285

theorem negation_ln_eq_x_minus_1 :
  ¬(∃ x : ℝ, 0 < x ∧ Real.log x = x - 1) ↔ ∀ x : ℝ, 0 < x → Real.log x ≠ x - 1 :=
by 
  sorry

end negation_ln_eq_x_minus_1_l163_163285


namespace no_cracked_seashells_l163_163009

theorem no_cracked_seashells (tom_seashells : ℕ) (fred_seashells : ℕ) (total_seashells : ℕ)
  (h1 : tom_seashells = 15) (h2 : fred_seashells = 43) (h3 : total_seashells = 58)
  (h4 : tom_seashells + fred_seashells = total_seashells) : 
  (total_seashells - (tom_seashells + fred_seashells) = 0) :=
by
  sorry

end no_cracked_seashells_l163_163009


namespace slope_of_tangent_l163_163671

theorem slope_of_tangent {x : ℝ} (h : x = 2) : deriv (λ x, 2 * x^2) x = 8 :=
by
  sorry

end slope_of_tangent_l163_163671


namespace find_principal_sum_l163_163149

theorem find_principal_sum 
  (CI SI P : ℝ) 
  (R : ℝ) 
  (T : ℝ) 
  (hCI : CI = 11730) 
  (hSI : SI = 10200) 
  (hT : T = 2) 
  (hCI_formula : CI = P * ((1 + R / 100)^T - 1)) 
  (hSI_formula : SI = (P * R * T) / 100) 
  (h_diff : CI - SI = 1530) :
  P = 34000 := 
by 
  sorry

end find_principal_sum_l163_163149


namespace find_number_of_students_l163_163998

theorem find_number_of_students
  (n : ℕ)
  (average_marks : ℕ → ℚ)
  (wrong_mark_corrected : ℕ → ℕ → ℚ)
  (correct_avg_marks_pred : ℕ → ℚ → Prop)
  (h1 : average_marks n = 60)
  (h2 : wrong_mark_corrected 90 15 = 75)
  (h3 : correct_avg_marks_pred n 57.5) :
  n = 30 :=
sorry

end find_number_of_students_l163_163998


namespace geometric_progression_problem_l163_163654

open Real

theorem geometric_progression_problem
  (a b c r : ℝ)
  (h1 : a = 20)
  (h2 : b = 40)
  (h3 : c = 10)
  (h4 : b = r * a)
  (h5 : c = r * b) :
  (a - (b - c)) - ((a - b) - c) = 20 := by
  sorry

end geometric_progression_problem_l163_163654


namespace total_fish_l163_163856

variable (L A : ℕ)

theorem total_fish (h1 : L = 40) (h2 : A = L + 20) : L + A = 100 := by 
  sorry

end total_fish_l163_163856


namespace kevin_trip_distance_l163_163521

theorem kevin_trip_distance :
  let D := 600
  (∃ T : ℕ, D = 50 * T ∧ D = 75 * (T - 4)) := 
sorry

end kevin_trip_distance_l163_163521


namespace final_quarters_l163_163712

-- Define the initial conditions and transactions
def initial_quarters : ℕ := 760
def first_spent : ℕ := 418
def second_spent : ℕ := 192

-- Define the final amount of quarters Sally should have
theorem final_quarters (initial_quarters first_spent second_spent : ℕ) : initial_quarters - first_spent - second_spent = 150 :=
by
  sorry

end final_quarters_l163_163712


namespace factorial_fraction_eq_zero_l163_163935

theorem factorial_fraction_eq_zero :
  ((5 * (Nat.factorial 7) - 35 * (Nat.factorial 6)) / Nat.factorial 8 = 0) :=
by
  sorry

end factorial_fraction_eq_zero_l163_163935


namespace number_of_second_graders_l163_163130

-- Define the number of kindergartners
def kindergartners : ℕ := 34

-- Define the number of first graders
def first_graders : ℕ := 48

-- Define the total number of students
def total_students : ℕ := 120

-- Define the proof statement
theorem number_of_second_graders : total_students - (kindergartners + first_graders) = 38 := by
  -- omit the proof details
  sorry

end number_of_second_graders_l163_163130


namespace max_possible_n_l163_163625

theorem max_possible_n (n : ℤ) (h : 101 * n ^ 2 ≤ 6400) : n ≤ 7 :=
by {
  sorry
}

end max_possible_n_l163_163625


namespace polynomial_term_equality_l163_163421

theorem polynomial_term_equality (p q : ℝ) (hpq_pos : 0 < p) (hq_pos : 0 < q) 
  (h_sum : p + q = 1) (h_eq : 28 * p^6 * q^2 = 56 * p^5 * q^3) : p = 2 / 3 :=
by
  sorry

end polynomial_term_equality_l163_163421


namespace prob_A1_selected_prob_B1_C1_not_selected_prob_encounter_A1_A2_l163_163429

namespace Probability

-- Definitions for volunteers and language groups
def volunteers := { : Fin 6 } -- A1, A2, B1, B2, C1, C2

def is_french (v : volunteers) : Prop := v = 0 ∨ v = 1
def is_russian (v : volunteers) : Prop := v = 2 ∨ v = 3
def is_english (v : volunteers) : Prop := v = 4 ∨ v = 5

-- Question 1: Probability of selecting A1
def event_A1_selected := λ (group : Fin 3 → volunteers), group 0 = 0

theorem prob_A1_selected : Pr event_A1_selected = 1 / 2 := sorry

-- Question 2: Probability that both B1 and C1 are not selected
def event_B1_C1_not_selected := λ (group : Fin 3 → volunteers),
  ¬ (group 1 = 2 ∧ group 2 = 4)

theorem prob_B1_C1_not_selected : Pr event_B1_C1_not_selected = 3 / 4 := sorry

-- Question 3: Probability of exactly encountering A1 and A2
def on_duty_pairs := ({v : Fin 6} : Set (volunteers × volunteers)) -- all possible pairs

def event_encounter_A1_A2 := (0, 1) ∈ on_duty_pairs

theorem prob_encounter_A1_A2 : Pr event_encounter_A1_A2 = 1 / 15 := sorry

end Probability

end prob_A1_selected_prob_B1_C1_not_selected_prob_encounter_A1_A2_l163_163429


namespace solution_inequality_l163_163390

theorem solution_inequality
  (a a' b b' c : ℝ)
  (h₁ : a ≠ 0)
  (h₂ : a' ≠ 0)
  (h₃ : (c - b) / a > (c - b') / a') :
  (c - b') / a' < (c - b) / a :=
by
  sorry

end solution_inequality_l163_163390


namespace combined_weight_correct_l163_163504

-- Define Jake's present weight
def Jake_weight : ℕ := 196

-- Define the weight loss
def weight_loss : ℕ := 8

-- Define Jake's weight after losing weight
def Jake_weight_after_loss : ℕ := Jake_weight - weight_loss

-- Define the relationship between Jake's weight after loss and his sister's weight
def sister_weight : ℕ := Jake_weight_after_loss / 2

-- Define the combined weight
def combined_weight : ℕ := Jake_weight + sister_weight

-- Prove that the combined weight is 290 pounds
theorem combined_weight_correct : combined_weight = 290 :=
by
  sorry

end combined_weight_correct_l163_163504


namespace no_solution_exists_l163_163108

theorem no_solution_exists (x y : ℝ) :
  ¬(4 * x^2 + 4 * x * y + 19 * y^2 ≤ 2 ∧ x - y ≤ -1) :=
sorry

end no_solution_exists_l163_163108


namespace dividend_percentage_paid_by_company_l163_163312

-- Define the parameters
def faceValue : ℝ := 50
def investmentReturnPercentage : ℝ := 25
def investmentPerShare : ℝ := 37

-- Define the theorem
theorem dividend_percentage_paid_by_company :
  (investmentReturnPercentage / 100 * investmentPerShare / faceValue * 100) = 18.5 :=
by
  -- The proof is omitted
  sorry

end dividend_percentage_paid_by_company_l163_163312


namespace distance_to_airport_l163_163477

theorem distance_to_airport:
  ∃ (d t: ℝ), 
    (d = 35 * (t + 1)) ∧
    (d - 35 = 50 * (t - 1.5)) ∧
    d = 210 := 
by 
  sorry

end distance_to_airport_l163_163477


namespace average_distance_to_sides_l163_163762

open Real

noncomputable def side_length : ℝ := 15
noncomputable def diagonal_distance : ℝ := 9.3
noncomputable def right_turn_distance : ℝ := 3

theorem average_distance_to_sides :
  let d1 := 9.58
  let d2 := 6.58
  let d3 := 5.42
  let d4 := 8.42
  (d1 + d2 + d3 + d4) / 4 = 7.5 :=
by
  let d1 := 9.58
  let d2 := 6.58
  let d3 := 5.42
  let d4 := 8.42
  have h : (d1 + d2 + d3 + d4) / 4 = 7.5
  { sorry }
  exact h

end average_distance_to_sides_l163_163762


namespace carmina_coins_l163_163472

-- Define the conditions related to the problem
variables (n d : ℕ) -- number of nickels and dimes

theorem carmina_coins (h1 : 5 * n + 10 * d = 360) (h2 : 10 * n + 5 * d = 540) : n + d = 60 :=
sorry

end carmina_coins_l163_163472


namespace trams_required_l163_163553

theorem trams_required (initial_trams : ℕ) (initial_interval : ℚ) (reduction_fraction : ℚ) :
  initial_trams = 12 ∧ initial_interval = 5 ∧ reduction_fraction = 1/5 →
  (initial_trams + initial_trams * reduction_fraction - initial_trams) = 3 :=
by
  sorry

end trams_required_l163_163553


namespace circuit_length_is_365_l163_163244

-- Definitions based on given conditions
def runs_morning := 7
def runs_afternoon := 3
def total_distance_week := 25550
def total_runs_day := runs_morning + runs_afternoon
def total_runs_week := total_runs_day * 7

-- Statement of the problem to be proved
theorem circuit_length_is_365 :
  total_distance_week / total_runs_week = 365 :=
sorry

end circuit_length_is_365_l163_163244


namespace original_number_of_cards_l163_163313

-- Declare variables r and b as naturals representing the number of red and black cards, respectively.
variable (r b : ℕ)

-- Assume the probabilities given in the problem.
axiom prob_red : (r : ℝ) / (r + b) = 1 / 3
axiom prob_red_after_add : (r : ℝ) / (r + b + 4) = 1 / 4

-- Define the statement we need to prove.
theorem original_number_of_cards : r + b = 12 :=
by
  -- The proof steps would be here, but we'll use sorry to avoid implementing them.
  sorry

end original_number_of_cards_l163_163313


namespace discount_comparison_l163_163030

noncomputable def final_price (P : ℝ) (d1 d2 d3 : ℝ) : ℝ :=
  P * (1 - d1) * (1 - d2) * (1 - d3)

theorem discount_comparison (P : ℝ) (d11 d12 d13 d21 d22 d23 : ℝ) :
  P = 20000 →
  d11 = 0.25 → d12 = 0.15 → d13 = 0.10 →
  d21 = 0.30 → d22 = 0.10 → d23 = 0.10 →
  final_price P d11 d12 d13 - final_price P d21 d22 d23 = 135 :=
by
  intros
  sorry

end discount_comparison_l163_163030


namespace domain_of_f_f_is_monotonically_increasing_l163_163991

open Real

noncomputable def f (x : ℝ) : ℝ := tan (2 * x - π / 8) + 3

theorem domain_of_f :
  ∀ x, (x ≠ 5 * π / 16 + k * π / 2) := sorry

theorem f_is_monotonically_increasing :
  ∀ x, (π / 16 < x ∧ x < 3 * π / 16 → f x < f (x + ε)) := sorry

end domain_of_f_f_is_monotonically_increasing_l163_163991


namespace range_of_m_l163_163668

theorem range_of_m (m : ℝ) : (∀ x : ℝ, |2009 * x + 1| ≥ |m - 1| - 2) → -1 ≤ m ∧ m ≤ 3 :=
by
  intro h
  sorry

end range_of_m_l163_163668


namespace no_positive_integer_solution_l163_163366

/-- Let \( p \) be a prime greater than 3 and \( x \) be an integer such that \( p \) divides \( x \).
    Then the equation \( x^2 - 1 = y^p \) has no positive integer solutions for \( y \). -/
theorem no_positive_integer_solution {p x y : ℕ} (hp : Nat.Prime p) (hgt : 3 < p) (hdiv : p ∣ x) :
  ¬∃ y : ℕ, (x^2 - 1 = y^p) ∧ (0 < y) :=
by
  sorry

end no_positive_integer_solution_l163_163366


namespace snail_reaches_tree_l163_163765

theorem snail_reaches_tree
  (l1 l2 s : ℝ) 
  (h_l1 : l1 = 4) 
  (h_l2 : l2 = 3) 
  (h_s : s = 40) : 
  ∃ n : ℕ, n = 37 ∧ s - n*(l1 - l2) ≤ l1 :=
  by
    sorry

end snail_reaches_tree_l163_163765


namespace sqrt_factorial_eq_l163_163605

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem sqrt_factorial_eq :
  sqrt (factorial 4 * factorial 4) = 24 :=
by
  have h : factorial 4 = 24 := by
    unfold factorial
    simpa using [factorial, factorial, factorial]
  rw [h, h]
  sorry

end sqrt_factorial_eq_l163_163605


namespace sequence_sum_1234_l163_163689

noncomputable def sequence_sum : ℕ → ℕ
| 0 := 1
| n := if n = 0 then 1 else (n / (n + 1) + 1)

theorem sequence_sum_1234 : (finset.range 1234).sum sequence_sum = 2419 := by
  sorry

end sequence_sum_1234_l163_163689


namespace g_at_3_l163_163282

noncomputable def g : ℝ → ℝ := sorry

axiom g_condition : ∀ x : ℝ, g (3 ^ x) + x * g (3 ^ (-x)) = 2

theorem g_at_3 : g 3 = 0 :=
by
  sorry

end g_at_3_l163_163282


namespace length_of_shorter_side_l163_163632

/-- 
A rectangular plot measuring L meters by 50 meters is to be enclosed by wire fencing. 
If the poles of the fence are kept 5 meters apart, 26 poles will be needed.
What is the length of the shorter side of the rectangular plot?
-/
theorem length_of_shorter_side
(L: ℝ) 
(h1: ∃ L: ℝ, L > 0) -- There's some positive length for the side L
(h2: ∀ distance: ℝ, distance = 5) -- Poles are kept 5 meters apart
(h3: ∀ poles: ℝ, poles = 26) -- 26 poles will be needed
(h4: 125 = 2 * (L + 50)) -- Use the perimeter calculated
: L = 12.5
:= sorry

end length_of_shorter_side_l163_163632


namespace crop_fraction_brought_to_AD_l163_163779

theorem crop_fraction_brought_to_AD
  (AD BC AB CD : ℝ)
  (h : ℝ)
  (angle : ℝ)
  (AD_eq_150 : AD = 150)
  (BC_eq_100 : BC = 100)
  (AB_eq_130 : AB = 130)
  (CD_eq_130 : CD = 130)
  (angle_eq_75 : angle = 75)
  (height_eq : h = (AB / 2) * Real.sin (angle * Real.pi / 180)) -- converting degrees to radians
  (area_trap : ℝ)
  (upper_area : ℝ)
  (total_area_eq : area_trap = (1 / 2) * (AD + BC) * h)
  (upper_area_eq : upper_area = (1 / 2) * (AD + (BC / 2)) * h)
  : (upper_area / area_trap) = 0.8 := 
sorry

end crop_fraction_brought_to_AD_l163_163779


namespace ratio_of_square_sides_sum_l163_163893

theorem ratio_of_square_sides_sum 
  (h : (75:ℚ) / 128 = (75:ℚ) / 128) :
  let a := 5
  let b := 6
  let c := 16
  a + b + c = 27 :=
by
  -- Our goal is to show that the sum of a + b + c equals 27
  let a := 5
  let b := 6
  let c := 16
  have h1 : a + b + c = 27 := sorry
  exact h1

end ratio_of_square_sides_sum_l163_163893


namespace distinguishable_large_equilateral_triangles_l163_163746

-- Definitions based on conditions.
def num_colors : ℕ := 8

def same_color_corners : ℕ := num_colors
def two_same_one_diff_colors : ℕ := num_colors * (num_colors - 1)
def all_diff_colors : ℕ := (num_colors * (num_colors - 1) * (num_colors - 2)) / 6

def corner_configurations : ℕ := same_color_corners + two_same_one_diff_colors + all_diff_colors
def triangle_between_center_and_corner : ℕ := num_colors
def center_triangle : ℕ := num_colors

def total_distinguishable_triangles : ℕ := corner_configurations * triangle_between_center_and_corner * center_triangle

theorem distinguishable_large_equilateral_triangles : total_distinguishable_triangles = 7680 :=
by
  sorry

end distinguishable_large_equilateral_triangles_l163_163746


namespace right_triangle_area_valid_right_triangle_perimeter_valid_l163_163120

-- Define the basic setup for the right triangle problem
def hypotenuse : ℕ := 13
def leg1 : ℕ := 5
def leg2 : ℕ := 12  -- Calculated from Pythagorean theorem, but assumed here as condition

-- Define the calculated area and perimeter based on the above definitions
def area (a b : ℕ) : ℕ := (1 / 2) * a * b
def perimeter (a b c : ℕ) : ℕ := a + b + c

-- State the proof goals
theorem right_triangle_area_valid : area leg1 leg2 = 30 :=
  by sorry

theorem right_triangle_perimeter_valid : perimeter leg1 leg2 hypotenuse = 30 :=
  by sorry

end right_triangle_area_valid_right_triangle_perimeter_valid_l163_163120


namespace find_r_l163_163836

variable (a b c r : ℝ)

theorem find_r (h1 : a * (b - c) / (b * (c - a)) = r)
               (h2 : b * (c - a) / (c * (b - a)) = r)
               (h3 : r > 0) : 
               r = (Real.sqrt 5 - 1) / 2 :=
by
  sorry

end find_r_l163_163836


namespace number_of_terms_in_ap_is_eight_l163_163286

theorem number_of_terms_in_ap_is_eight
  (n : ℕ) (a d : ℝ)
  (even : n % 2 = 0)
  (sum_odd : (n / 2 : ℝ) * (2 * a + (n - 2) * d) = 24)
  (sum_even : (n / 2 : ℝ) * (2 * a + n * d) = 30)
  (last_exceeds_first : (n - 1) * d = 10.5) :
  n = 8 :=
by sorry

end number_of_terms_in_ap_is_eight_l163_163286


namespace water_pressure_on_dam_l163_163642

theorem water_pressure_on_dam :
  let a := 10 -- length of upper base in meters
  let b := 20 -- length of lower base in meters
  let h := 3 -- height in meters
  let ρg := 9810 -- natural constant for water pressure in N/m^3
  let P := ρg * ((a + 2 * b) * h^2 / 6)
  P = 735750 :=
by
  sorry

end water_pressure_on_dam_l163_163642


namespace original_acid_percentage_zero_l163_163755

theorem original_acid_percentage_zero (a w : ℝ) 
  (h1 : (a + 1) / (a + w + 1) = 1 / 4) 
  (h2 : (a + 2) / (a + w + 2) = 2 / 5) : 
  a / (a + w) = 0 := 
by
  sorry

end original_acid_percentage_zero_l163_163755


namespace problem1_problem2_l163_163758

-- Problem (1)
theorem problem1 (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  a^3 + b^3 >= a*b^2 + a^2*b := 
sorry

-- Problem (2)
theorem problem2 (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x + y > 2) : 
  (1 + y) / x < 2 ∨ (1 + x) / y < 2 := 
sorry

end problem1_problem2_l163_163758


namespace probability_of_hitting_exactly_twice_l163_163447

def P_hit_first : ℝ := 0.4
def P_hit_second : ℝ := 0.5
def P_hit_third : ℝ := 0.7

def P_hit_exactly_twice_in_three_shots : ℝ :=
  P_hit_first * P_hit_second * (1 - P_hit_third) +
  (1 - P_hit_first) * P_hit_second * P_hit_third +
  P_hit_first * (1 - P_hit_second) * P_hit_third

theorem probability_of_hitting_exactly_twice :
  P_hit_exactly_twice_in_three_shots = 0.41 := 
by
  sorry

end probability_of_hitting_exactly_twice_l163_163447


namespace jake_third_test_score_l163_163411

theorem jake_third_test_score
  (avg_score_eq_75 : (80 + 90 + third_score + third_score) / 4 = 75)
  (second_score : ℕ := 80 + 10) :
  third_score = 65 :=
by
  sorry

end jake_third_test_score_l163_163411


namespace black_squares_31x31_l163_163647

-- Definitions to express the checkerboard problem conditions
def isCheckerboard (n : ℕ) : Prop := 
  ∀ i j : ℕ,
    i < n → j < n → 
    ((i + j) % 2 = 0 → (i % 2 = 0 ∧ j % 2 = 0) ∨ (i % 2 = 1 ∧ j % 2 = 1))

def blackCornerSquares (n : ℕ) : Prop :=
  ∀ i j : ℕ,
    (i = 0 ∨ i = n - 1) ∧ (j = 0 ∨ j = n - 1) → (i + j) % 2 = 0

-- The main statement to prove
theorem black_squares_31x31 :
  ∃ (n : ℕ) (count : ℕ), n = 31 ∧ isCheckerboard n ∧ blackCornerSquares n ∧ count = 481 := 
by 
  sorry -- Proof to be provided

end black_squares_31x31_l163_163647


namespace sqrt_factorial_multiplication_l163_163616

theorem sqrt_factorial_multiplication : (Real.sqrt ((4! : ℝ) * (4! : ℝ)) = 24) := 
by sorry

end sqrt_factorial_multiplication_l163_163616


namespace unit_circle_sector_arc_length_l163_163234

theorem unit_circle_sector_arc_length (r S l : ℝ) (h1 : r = 1) (h2 : S = 1) (h3 : S = 1 / 2 * l * r) : l = 2 :=
by
  sorry

end unit_circle_sector_arc_length_l163_163234


namespace original_days_to_finish_work_l163_163638

theorem original_days_to_finish_work : 
  ∀ (D : ℕ), 
  (∃ (W : ℕ), 15 * D * W = 25 * (D - 3) * W) → 
  D = 8 :=
by
  intros D h
  sorry

end original_days_to_finish_work_l163_163638


namespace distance_to_airport_l163_163476

theorem distance_to_airport:
  ∃ (d t: ℝ), 
    (d = 35 * (t + 1)) ∧
    (d - 35 = 50 * (t - 1.5)) ∧
    d = 210 := 
by 
  sorry

end distance_to_airport_l163_163476


namespace intersection_of_M_N_equals_0_1_open_interval_l163_163259

def M : Set ℝ := { x | x ≥ 0 }
def N : Set ℝ := { x | x^2 < 1 }

theorem intersection_of_M_N_equals_0_1_open_interval :
  M ∩ N = { x | 0 ≤ x ∧ x < 1 } := 
sorry

end intersection_of_M_N_equals_0_1_open_interval_l163_163259


namespace max_value_x_sub_2z_l163_163075

theorem max_value_x_sub_2z (x y z : ℝ) (h : x^2 + y^2 + z^2 = 16) :
  ∃ m, m = 4 * Real.sqrt 5 ∧ ∀ x y z, x^2 + y^2 + z^2 = 16 → x - 2 * z ≤ m :=
sorry

end max_value_x_sub_2z_l163_163075


namespace keith_picked_p_l163_163519

-- Definitions of the given conditions
def p_j : ℕ := 46  -- Jason's pears
def p_m : ℕ := 12  -- Mike's pears
def p_t : ℕ := 105 -- Total pears picked

-- The theorem statement
theorem keith_picked_p : p_t - (p_j + p_m) = 47 := by
  -- Proof part will be handled later
  sorry

end keith_picked_p_l163_163519


namespace expected_value_of_difference_is_4_point_5_l163_163131

noncomputable def expected_value_difference : ℚ :=
  (2 * 6 / 56 + 3 * 10 / 56 + 4 * 12 / 56 + 5 * 12 / 56 + 6 * 10 / 56 + 7 * 6 / 56)

theorem expected_value_of_difference_is_4_point_5 :
  expected_value_difference = 4.5 := sorry

end expected_value_of_difference_is_4_point_5_l163_163131


namespace compare_f_log_range_of_t_l163_163669

noncomputable def f (x : ℝ) : ℝ := 2 / x

theorem compare_f_log:
  f (Real.log 26 / Real.log 3) < f (Real.log 8 / Real.log 3) ∧
  f (Real.log 8 / Real.log 3) < f (Real.log 3 / Real.log 2) :=
by sorry

theorem range_of_t {t : ℝ} (ht : 0 < t) 
  (h : ∀ x ∈ Icc (2:ℝ) 3, f (t + x^2) + f (1 - x - x^2 - 2^x) > 0) : 
  t < 5 :=
by sorry

end compare_f_log_range_of_t_l163_163669


namespace problem_proof_l163_163695

theorem problem_proof (a b c : ℝ) (h1 : a ≠ b ∧ b ≠ c ∧ c ≠ a) 
  (h2 : a / (b - c) + b / (c - a) + c / (a - b) = 0) : 
  a / (b - c) ^ 2 + b / (c - a) ^ 2 + c / (a - b) ^ 2 = 0 :=
sorry

end problem_proof_l163_163695


namespace theater_revenue_l163_163035

theorem theater_revenue
  (total_seats : ℕ)
  (adult_price : ℕ)
  (child_price : ℕ)
  (child_tickets_sold : ℕ)
  (total_sold_out : total_seats = 80)
  (child_tickets_sold_cond : child_tickets_sold = 63)
  (adult_ticket_price_cond : adult_price = 12)
  (child_ticket_price_cond : child_price = 5)
  : total_seats * adult_price + child_tickets_sold * child_price = 519 :=
by
  -- proof omitted
  sorry

end theater_revenue_l163_163035


namespace prob1_converse_prob1_inverse_prob1_contrapositive_prob2_converse_prob2_inverse_prob2_contrapositive_l163_163620

-- Problem 1: Original proposition converse, inverse, contrapositive
theorem prob1_converse (x y : ℝ) (h : x = 0 ∨ y = 0) : x * y = 0 :=
sorry

theorem prob1_inverse (x y : ℝ) (h : x * y ≠ 0) : x ≠ 0 ∧ y ≠ 0 :=
sorry

theorem prob1_contrapositive (x y : ℝ) (h : x ≠ 0 ∧ y ≠ 0) : x * y ≠ 0 :=
sorry

-- Problem 2: Original proposition converse, inverse, contrapositive
theorem prob2_converse (x y : ℝ) (h : x * y > 0) : x > 0 ∧ y > 0 :=
sorry

theorem prob2_inverse (x y : ℝ) (h : x ≤ 0 ∨ y ≤ 0) : x * y ≤ 0 :=
sorry

theorem prob2_contrapositive (x y : ℝ) (h : x * y ≤ 0) : x ≤ 0 ∨ y ≤ 0 :=
sorry

end prob1_converse_prob1_inverse_prob1_contrapositive_prob2_converse_prob2_inverse_prob2_contrapositive_l163_163620


namespace bob_age_sum_digits_l163_163038

theorem bob_age_sum_digits
  (A B C : ℕ)  -- Define ages for Alice (A), Bob (B), and Carl (C)
  (h1 : C = 2)  -- Carl's age is 2
  (h2 : B = A + 2)  -- Bob is 2 years older than Alice
  (h3 : ∃ n, A = 2 * n ∧ n > 0 ∧ n ≤ 8 )  -- Alice's age is a multiple of Carl's age today, marking the second of the 8 such multiples 
  : ∃ n, (B + n) % (C + n) = 0 ∧ (B + n) = 50 :=  -- Prove that the next time Bob's age is a multiple of Carl's, Bob's age will be 50
sorry

end bob_age_sum_digits_l163_163038


namespace smallest_x_y_sum_l163_163492

theorem smallest_x_y_sum (x y : ℕ) (hx_pos : 0 < x) (hy_pos : 0 < y) (hxy : x ≠ y) (h_fraction : 1/x + 1/y = 1/20) : x + y = 90 :=
sorry

end smallest_x_y_sum_l163_163492


namespace first_investment_percentage_l163_163710

theorem first_investment_percentage :
  let total_inheritance := 4000
  let invested_6_5 := 1800
  let interest_rate_6_5 := 0.065
  let total_interest := 227
  let remaining_investment := total_inheritance - invested_6_5
  let interest_from_6_5 := invested_6_5 * interest_rate_6_5
  let interest_from_remaining := total_interest - interest_from_6_5
  let P := interest_from_remaining / remaining_investment
  P = 0.05 :=
by 
  sorry

end first_investment_percentage_l163_163710


namespace find_c_l163_163545

theorem find_c (x c : ℚ) (h1 : 3 * x + 5 = 1) (h2 : c * x + 15 = 3) : c = 9 :=
by sorry

end find_c_l163_163545


namespace record_expenditure_l163_163078

def income (amount : ℤ) := amount > 0
def expenditure (amount : ℤ) := -amount

theorem record_expenditure : 
  (income 100 = true) ∧ (100 = +100) ∧ (income (expenditure 80) = false) → (expenditure 80 = -80) := 
by sorry

end record_expenditure_l163_163078


namespace simple_interest_rate_l163_163145

theorem simple_interest_rate (P : ℝ) (T : ℝ) (r : ℝ) (h1 : T = 10) (h2 : (3 / 5) * P = (P * r * T) / 100) : r = 6 := by
  sorry

end simple_interest_rate_l163_163145


namespace problem_statement_l163_163961

theorem problem_statement (m n : ℤ) (h : |m - 2| + (n + 1)^2 = 0) : m + n = 1 :=
by sorry

end problem_statement_l163_163961


namespace sum_of_three_numbers_is_seventy_l163_163419

theorem sum_of_three_numbers_is_seventy
  (a b c : ℝ)
  (h1 : a ≤ b ∧ b ≤ c)
  (h2 : (a + b + c) / 3 = a + 20)
  (h3 : (a + b + c) / 3 = c - 30)
  (h4 : b = 10)
  (h5 : a + c = 60) :
  a + b + c = 70 :=
  sorry

end sum_of_three_numbers_is_seventy_l163_163419


namespace work_completion_time_l163_163914

theorem work_completion_time (d : ℕ) (h : d = 9) : 3 * d = 27 := by
  sorry

end work_completion_time_l163_163914


namespace complex_multiplication_example_l163_163451

def imaginary_unit (i : ℂ) : Prop := i^2 = -1

theorem complex_multiplication_example (i : ℂ) (h : imaginary_unit i) :
  (3 + i) * (1 - 2 * i) = 5 - 5 * i := 
by
  sorry

end complex_multiplication_example_l163_163451


namespace sqrt_factorial_mul_factorial_l163_163591

theorem sqrt_factorial_mul_factorial :
  (Real.sqrt (Nat.factorial 4 * Nat.factorial 4) = 24) := by
  sorry

end sqrt_factorial_mul_factorial_l163_163591


namespace circles_positional_relationship_l163_163892

theorem circles_positional_relationship :
  ∃ R r : ℝ, (R * r = 2 ∧ R + r = 3) ∧ 3 = R + r → "externally tangent" = "externally tangent" :=
by
  sorry

end circles_positional_relationship_l163_163892


namespace value_of_expression_l163_163002

-- Conditions
def isOdd (f : ℝ → ℝ) := ∀ x, f (-x) = -f x
def isIncreasingOn (f : ℝ → ℝ) (a b : ℝ) := ∀ x y, a ≤ x → x ≤ y → y ≤ b → f x ≤ f y
def hasMaxOn (f : ℝ → ℝ) (a b : ℝ) (M : ℝ) := ∃ x, a ≤ x ∧ x ≤ b ∧ f x = M
def hasMinOn (f : ℝ → ℝ) (a b : ℝ) (m : ℝ) := ∃ x, a ≤ x ∧ x ≤ b ∧ f x = m

-- Proof statement
theorem value_of_expression (f : ℝ → ℝ) 
  (hf1 : isOdd f)
  (hf2 : isIncreasingOn f 3 7)
  (hf3 : hasMaxOn f 3 6 8)
  (hf4 : hasMinOn f 3 6 (-1)) :
  2 * f (-6) + f (-3) = -15 :=
sorry

end value_of_expression_l163_163002


namespace dessert_distribution_l163_163124

theorem dessert_distribution 
  (mini_cupcakes : ℕ) 
  (donut_holes : ℕ) 
  (total_desserts : ℕ) 
  (students : ℕ) 
  (h1 : mini_cupcakes = 14)
  (h2 : donut_holes = 12) 
  (h3 : students = 13)
  (h4 : total_desserts = mini_cupcakes + donut_holes)
  : total_desserts / students = 2 :=
by sorry

end dessert_distribution_l163_163124


namespace range_of_m_l163_163667

variable (x m : ℝ)

noncomputable def p := abs (x - 1) ≤ 2
noncomputable def q := x^2 - 2 * x + 1 - m^2 ≤ 0
noncomputable def neg_p := abs (x - 1) > 2
noncomputable def neg_q := x^2 - 2 * x + 1 - m^2 > 0

theorem range_of_m (h_m_gt_zero : m > 0) (h_neg_p_necessary : ∀ x, neg_q x m → neg_p x) (h_neg_p_not_sufficient : ∃ x, neg_p x ∧ ¬ neg_q x m) : 
  3 ≤ m := sorry

end range_of_m_l163_163667


namespace min_mn_sum_l163_163119

theorem min_mn_sum :
  ∃ (m n : ℕ), n > m ∧ m ≥ 1 ∧ 
  (1978^n % 1000 = 1978^m % 1000) ∧ (m + n = 106) :=
sorry

end min_mn_sum_l163_163119


namespace smallest_next_smallest_sum_l163_163293

-- Defining the set of numbers as constants
def nums : Set ℕ := {10, 11, 12, 13}

-- Define the smallest number in the set
def smallest : ℕ := 10

-- Define the next smallest number in the set
def next_smallest : ℕ := 11

-- The main theorem statement
theorem smallest_next_smallest_sum : smallest + next_smallest = 21 :=
by 
  sorry

end smallest_next_smallest_sum_l163_163293


namespace total_fish_caught_l163_163858

theorem total_fish_caught (leo_fish : ℕ) (agrey_fish : ℕ) (h1 : leo_fish = 40) (h2 : agrey_fish = leo_fish + 20) :
  leo_fish + agrey_fish = 100 :=
by
  sorry

end total_fish_caught_l163_163858


namespace minimum_n_minus_m_l163_163206

noncomputable def f (x : Real) : Real :=
    (Real.sin x) * (Real.sin (x + Real.pi / 3)) - 1 / 4

theorem minimum_n_minus_m (m n : Real) (h : m < n) 
  (h_domain : ∀ x, m ≤ x ∧ x ≤ n → -1 / 2 ≤ f x ∧ f x ≤ 1 / 4) :
  n - m = 2 * Real.pi / 3 :=
by
  sorry

end minimum_n_minus_m_l163_163206


namespace simplify_polynomial_l163_163719

variable (x : ℝ)

theorem simplify_polynomial :
  (2 * x^10 + 8 * x^9 + 3 * x^8) + (5 * x^12 - x^10 + 2 * x^9 - 5 * x^8 + 4 * x^5 + 6)
  = 5 * x^12 + x^10 + 10 * x^9 - 2 * x^8 + 4 * x^5 + 6 := by
  sorry

end simplify_polynomial_l163_163719


namespace greatest_possible_n_l163_163227

theorem greatest_possible_n (n : ℤ) (h : 101 * n ^ 2 ≤ 8100) : n ≤ 8 :=
by
  -- Intentionally left uncommented.
  sorry

end greatest_possible_n_l163_163227


namespace total_pencils_is_5_l163_163290

-- Define the initial number of pencils and the number of pencils Tim added
def initial_pencils : Nat := 2
def pencils_added_by_tim : Nat := 3

-- Prove the total number of pencils is equal to 5
theorem total_pencils_is_5 : initial_pencils + pencils_added_by_tim = 5 := by
  sorry

end total_pencils_is_5_l163_163290


namespace calculate_nabla_l163_163774

def nabla (a b : ℚ) : ℚ := (a + b) / (1 + a * b)

theorem calculate_nabla : nabla (nabla 2 3) 4 = 11 / 9 :=
by
  sorry

end calculate_nabla_l163_163774


namespace dan_helmet_craters_l163_163942

variable (D S : ℕ)
variable (h1 : D = S + 10)
variable (h2 : D + S + 15 = 75)

theorem dan_helmet_craters : D = 35 := by
  sorry

end dan_helmet_craters_l163_163942


namespace distance_to_office_is_18_l163_163029

-- Definitions given in the problem conditions
variables (x t d : ℝ)
-- Conditions based on the problem statements
axiom speed_condition1 : d = x * t
axiom speed_condition2 : d = (x + 1) * (3 / 4 * t)
axiom speed_condition3 : d = (x - 1) * (t + 3)

-- The mathematical proof statement that needs to be shown
theorem distance_to_office_is_18 :
  d = 18 :=
by
  sorry

end distance_to_office_is_18_l163_163029


namespace cost_of_cheese_without_coupon_l163_163407

theorem cost_of_cheese_without_coupon
    (cost_bread : ℝ := 4.00)
    (cost_meat : ℝ := 5.00)
    (coupon_cheese : ℝ := 1.00)
    (coupon_meat : ℝ := 1.00)
    (cost_sandwich : ℝ := 2.00)
    (num_sandwiches : ℝ := 10)
    (C : ℝ) : 
    (num_sandwiches * cost_sandwich = (cost_bread + (cost_meat - coupon_meat) + cost_meat + (C - coupon_cheese) + C)) → (C = 4.50) :=
by {
    sorry
}

end cost_of_cheese_without_coupon_l163_163407


namespace quadratic_inequality_solution_range_l163_163974

theorem quadratic_inequality_solution_range (a : ℝ) :
  (¬ ∃ x : ℝ, 4 * x^2 + (a - 2) * x + 1 / 4 ≤ 0) ↔ 0 < a ∧ a < 4 :=
by
  sorry

end quadratic_inequality_solution_range_l163_163974


namespace fish_game_teams_l163_163801

noncomputable def number_of_possible_teams (n : ℕ) : ℕ := 
  if n = 6 then 5 else sorry

theorem fish_game_teams : number_of_possible_teams 6 = 5 := by
  unfold number_of_possible_teams
  rfl

end fish_game_teams_l163_163801


namespace reassemble_black_rectangles_into_1x2_rectangle_l163_163850

theorem reassemble_black_rectangles_into_1x2_rectangle
  (x y : ℝ)
  (h1 : 0 < x ∧ x < 2)
  (h2 : 0 < y ∧ y < 2)
  (black_white_equal : 2*x*y - 2*x - 2*y + 2 = 0) :
  (x = 1 ∨ y = 1) →
  ∃ (z : ℝ), z = 1 :=
by
  sorry

end reassemble_black_rectangles_into_1x2_rectangle_l163_163850


namespace at_least_one_not_less_than_one_third_l163_163258

theorem at_least_one_not_less_than_one_third (a b c : ℝ) (h : a + b + c = 1) :
  a ≥ 1/3 ∨ b ≥ 1/3 ∨ c ≥ 1/3 :=
sorry

end at_least_one_not_less_than_one_third_l163_163258


namespace parallelogram_height_l163_163840

theorem parallelogram_height (base height area : ℝ) (h_base : base = 9) (h_area : area = 33.3) (h_formula : area = base * height) : height = 3.7 :=
by
  -- Proof goes here, but currently skipped
  sorry

end parallelogram_height_l163_163840


namespace sqrt_factorial_product_l163_163573

theorem sqrt_factorial_product:
  sqrt ((fact 4) * (fact 4)) = 24 :=
by sorry

end sqrt_factorial_product_l163_163573


namespace smallest_positive_integer_neither_prime_nor_square_no_prime_factor_less_than_50_l163_163138

def is_not_prime (n : ℕ) : Prop := ¬ Prime n

def is_not_square (n : ℕ) : Prop := ∀ m : ℕ, m * m ≠ n

def no_prime_factor_less_than_50 (n : ℕ) : Prop :=
  ∀ p : ℕ, Prime p → p ∣ n → p ≥ 50

theorem smallest_positive_integer_neither_prime_nor_square_no_prime_factor_less_than_50 :
  (∃ n : ℕ, 0 < n ∧ is_not_prime n ∧ is_not_square n ∧ no_prime_factor_less_than_50 n ∧
  (∀ m : ℕ, 0 < m ∧ is_not_prime m ∧ is_not_square m ∧ no_prime_factor_less_than_50 m → n ≤ m)) →
  ∃ n : ℕ, n = 3127 :=
by {
  sorry
}

end smallest_positive_integer_neither_prime_nor_square_no_prime_factor_less_than_50_l163_163138


namespace not_always_divisible_by_40_l163_163220

open Nat

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

theorem not_always_divisible_by_40 (p : ℕ) (hp_prime : is_prime p) (hp_geq7 : p ≥ 7) : ¬ (∀ p : ℕ, is_prime p ∧ p ≥ 7 → 40 ∣ (p^2 - 1)) := 
sorry

end not_always_divisible_by_40_l163_163220


namespace trig_identity_cos_l163_163813

theorem trig_identity_cos (x : ℝ) (h : sin (2 * x + π / 6) = -1 / 3) : cos (π / 3 - 2 * x) = -1 / 3 :=
by
  sorry

end trig_identity_cos_l163_163813


namespace consecutive_page_numbers_sum_l163_163737

theorem consecutive_page_numbers_sum (n : ℕ) (h : n * (n + 1) = 19881) : n + (n + 1) = 283 :=
sorry

end consecutive_page_numbers_sum_l163_163737


namespace jelly_bean_problem_l163_163330

variable (b c : ℕ)

theorem jelly_bean_problem (h1 : b = 3 * c) (h2 : b - 15 = 4 * (c - 15)) : b = 135 :=
sorry

end jelly_bean_problem_l163_163330


namespace tara_dice_probability_divisible_by_8_l163_163112

-- Definitions based on the conditions
def standard_die := {1, 2, 3, 4, 5, 6}
def dice_rolls (n : ℕ) := fin n → standard_die

-- Theorem to prove the probability
theorem tara_dice_probability_divisible_by_8 :
  ∀ (product : ℕ) (rolls : dice_rolls 8), 
  (product = ∏ i, rolls i) →
  (∀ i, rolls i ∈ standard_die) →
  probability (product % 8 = 0) = 1143 / 1152 :=
sorry

end tara_dice_probability_divisible_by_8_l163_163112


namespace total_tickets_sold_l163_163923

-- Define the parameters and conditions
def VIP_ticket_price : ℝ := 45.00
def general_ticket_price : ℝ := 20.00
def total_revenue : ℝ := 7500.00
def tickets_difference : ℕ := 276

-- Define the total number of tickets sold
def total_number_of_tickets (V G : ℕ) : ℕ := V + G

-- The theorem to be proved
theorem total_tickets_sold (V G : ℕ) 
  (h1 : VIP_ticket_price * V + general_ticket_price * G = total_revenue)
  (h2 : V = G - tickets_difference) : 
  total_number_of_tickets V G = 336 :=
by
  sorry

end total_tickets_sold_l163_163923


namespace xiao_ming_valid_paths_final_valid_paths_l163_163932

-- Definitions from conditions
def paths_segments := ('a', 'b', 'c', 'd', 'e', 'f', 'g', 'h')
def initial_paths := 256
def invalid_paths := 64

-- Theorem statement
theorem xiao_ming_valid_paths : initial_paths - invalid_paths = 192 :=
by sorry

theorem final_valid_paths : 192 * 2 = 384 :=
by sorry

end xiao_ming_valid_paths_final_valid_paths_l163_163932


namespace jake_third_test_score_l163_163412

theorem jake_third_test_score
  (avg_score_eq_75 : (80 + 90 + third_score + third_score) / 4 = 75)
  (second_score : ℕ := 80 + 10) :
  third_score = 65 :=
by
  sorry

end jake_third_test_score_l163_163412


namespace equivalence_of_complements_union_l163_163701

open Set

-- Definitions as per the conditions
def U : Set ℝ := univ
def M : Set ℝ := { x | x ≥ 1 }
def N : Set ℝ := { x | 0 ≤ x ∧ x < 5 }
def complement_U (S : Set ℝ) : Set ℝ := U \ S

-- Mathematical statement to be proved
theorem equivalence_of_complements_union :
  (complement_U M ∪ complement_U N) = { x : ℝ | x < 1 ∨ x ≥ 5 } :=
by
  -- Non-trivial proof, hence skipped with sorry
  sorry

end equivalence_of_complements_union_l163_163701


namespace a_2019_value_l163_163498

noncomputable def a_sequence (n : ℕ) : ℝ :=
  if n = 0 then 0  -- not used, a_0 is irrelevant
  else if n = 1 then 1 / 2
  else a_sequence (n - 1) + 1 / (2 ^ (n - 1))

theorem a_2019_value :
  a_sequence 2019 = 3 / 2 - 1 / (2 ^ 2018) :=
by
  sorry

end a_2019_value_l163_163498


namespace car_return_speed_l163_163162

variable (d : ℕ) (r : ℕ)
variable (H0 : d = 180)
variable (H1 : ∀ t1 : ℕ, t1 = d / 90)
variable (H2 : ∀ t2 : ℕ, t2 = d / r)
variable (H3 : ∀ avg_rate : ℕ, avg_rate = 2 * d / (d / 90 + d / r))
variable (H4 : avg_rate = 60)

theorem car_return_speed : r = 45 :=
by sorry

end car_return_speed_l163_163162


namespace flight_cost_l163_163544

theorem flight_cost (ground_school_cost flight_portion_addition total_cost flight_portion_cost: ℕ) 
  (h₁ : ground_school_cost = 325)
  (h₂ : flight_portion_addition = 625)
  (h₃ : flight_portion_cost = ground_school_cost + flight_portion_addition):
  flight_portion_cost = 950 :=
by
  -- placeholder for proofs
  sorry

end flight_cost_l163_163544


namespace find_s_when_t_eq_5_l163_163058

theorem find_s_when_t_eq_5 (s : ℝ) (h : 5 = 8 * s^2 + 2 * s) :
  s = (-1 + Real.sqrt 41) / 8 ∨ s = (-1 - Real.sqrt 41) / 8 :=
by sorry

end find_s_when_t_eq_5_l163_163058


namespace expand_expression_l163_163055

theorem expand_expression (x : ℝ) : (x + 3) * (4 * x - 8) - 2 * x = 4 * x^2 + 2 * x - 24 := by
  sorry

end expand_expression_l163_163055


namespace eval_at_neg_five_l163_163971

def f (x : ℝ) : ℝ := x^2 + 2 * x - 3

theorem eval_at_neg_five : f (-5) = 12 :=
by
  sorry

end eval_at_neg_five_l163_163971


namespace simplify_expression_l163_163328

theorem simplify_expression :
  (5 + 2) * (5^3 + 2^3) * (5^9 + 2^9) * (5^27 + 2^27) * (5^81 + 2^81) = 5^128 - 2^128 :=
by
  sorry

end simplify_expression_l163_163328


namespace min_value_of_diff_l163_163207

noncomputable def f (x : ℝ) : ℝ :=
  sin x * sin (x + π / 3) - 1 / 4

theorem min_value_of_diff {m n : ℝ} (h : m < n) (h_f_range : ∀ x ∈ set.Icc m n, f x ∈ set.Icc (-1 / 2) (1 / 4)) :
  n - m = 2 * π / 3 :=
sorry

end min_value_of_diff_l163_163207


namespace packs_of_tuna_purchased_l163_163326

-- Definitions based on the conditions
def cost_per_pack_of_tuna : ℕ := 2
def cost_per_bottle_of_water : ℤ := (3 / 2)
def total_paid_by_Barbara : ℕ := 56
def money_spent_on_different_goods : ℕ := 40
def number_of_bottles_of_water : ℕ := 4

-- The proposition to prove
theorem packs_of_tuna_purchased :
  ∃ T : ℕ, total_paid_by_Barbara = cost_per_pack_of_tuna * T + cost_per_bottle_of_water * number_of_bottles_of_water + money_spent_on_different_goods ∧ T = 5 :=
by
  sorry

end packs_of_tuna_purchased_l163_163326


namespace ratio_of_eggs_l163_163871

/-- Megan initially had 24 eggs (12 from the store and 12 from her neighbor). She used 6 eggs in total (2 for an omelet and 4 for baking). She set aside 9 eggs for three meals (3 eggs per meal). Finally, Megan divided the remaining 9 eggs by giving 9 to her aunt and keeping 9 for herself. The ratio of the eggs she gave to her aunt to the eggs she kept is 1:1. -/
theorem ratio_of_eggs
  (eggs_bought : ℕ)
  (eggs_from_neighbor : ℕ)
  (eggs_omelet : ℕ)
  (eggs_baking : ℕ)
  (meals : ℕ)
  (eggs_per_meal : ℕ)
  (aunt_got : ℕ)
  (kept_for_meals : ℕ)
  (initial_eggs := eggs_bought + eggs_from_neighbor)
  (used_eggs := eggs_omelet + eggs_baking)
  (remaining_eggs := initial_eggs - used_eggs)
  (assigned_eggs := meals * eggs_per_meal)
  (final_eggs := remaining_eggs - assigned_eggs)
  (ratio : ℚ := aunt_got / kept_for_meals) :
  eggs_bought = 12 ∧
  eggs_from_neighbor = 12 ∧
  eggs_omelet = 2 ∧
  eggs_baking = 4 ∧
  meals = 3 ∧
  eggs_per_meal = 3 ∧
  aunt_got = 9 ∧
  kept_for_meals = assigned_eggs →
  ratio = 1 := by
  sorry

end ratio_of_eggs_l163_163871


namespace initial_weight_of_beef_l163_163922

theorem initial_weight_of_beef (W : ℝ) 
  (stage1 : W' = 0.70 * W) 
  (stage2 : W'' = 0.80 * W') 
  (stage3 : W''' = 0.50 * W'') 
  (final_weight : W''' = 315) : 
  W = 1125 := by 
  sorry

end initial_weight_of_beef_l163_163922


namespace original_money_l163_163040

theorem original_money (M : ℕ) (h1 : 3 * M / 8 ≤ M)
  (h2 : 1 * (M - 3 * M / 8) / 5 ≤ M - 3 * M / 8)
  (h3 : M - 3 * M / 8 - (1 * (M - 3 * M / 8) / 5) = 36) : M = 72 :=
sorry

end original_money_l163_163040


namespace unique_solution_3_pow_x_minus_2_pow_y_eq_7_l163_163056

theorem unique_solution_3_pow_x_minus_2_pow_y_eq_7 :
  ∀ x y : ℕ, (1 ≤ x) → (1 ≤ y) → (3 ^ x - 2 ^ y = 7) → (x = 2 ∧ y = 1) :=
by
  intros x y hx hy hxy
  sorry

end unique_solution_3_pow_x_minus_2_pow_y_eq_7_l163_163056


namespace car_return_speed_l163_163165

theorem car_return_speed (d : ℕ) (r : ℕ) (h₁ : d = 180) (h₂ : (2 * d) / ((d / 90) + (d / r)) = 60) : r = 45 :=
by
  rw [h₁] at h₂
  have h3 : 2 * 180 / ((180 / 90) + (180 / r)) = 60 := h₂
  -- The rest of the proof involves solving for r, but here we only need the statement
  sorry

end car_return_speed_l163_163165


namespace sweets_neither_red_nor_green_l163_163291

theorem sweets_neither_red_nor_green (total_sweets red_sweets green_sweets : ℕ)
  (h1 : total_sweets = 285)
  (h2 : red_sweets = 49)
  (h3 : green_sweets = 59) : total_sweets - (red_sweets + green_sweets) = 177 :=
by
  sorry

end sweets_neither_red_nor_green_l163_163291


namespace tulips_in_daniels_garden_l163_163739
-- Import all necessary libraries

-- Define the problem statement
theorem tulips_in_daniels_garden
  (initial_ratio_tulips_sunflowers : ℚ := 3 / 7)
  (initial_sunflowers : ℕ := 42)
  (additional_sunflowers : ℕ := 14) :
  let total_sunflowers := initial_sunflowers + additional_sunflowers,
      total_tulips := (initial_ratio_tulips_sunflowers * total_sunflowers) : ℕ
  in total_tulips = 24 :=
by
  sorry  -- Proof should be filled in here

end tulips_in_daniels_garden_l163_163739


namespace incorrect_mode_l163_163975

theorem incorrect_mode (data : List ℕ) (hdata : data = [1, 2, 4, 3, 5]) : ¬ (∃ mode, mode = 5 ∧ (data.count mode > 1)) :=
by
  sorry

end incorrect_mode_l163_163975


namespace decagon_ratio_bisect_l163_163939

theorem decagon_ratio_bisect (area_decagon unit_square area_trapezoid : ℕ) 
  (h_area_decagon : area_decagon = 12) 
  (h_bisect : ∃ RS : ℕ, ∃ XR : ℕ, RS * 2 = area_decagon) 
  (below_RS : ∃ base1 base2 height : ℕ, base1 = 3 ∧ base2 = 3 ∧ base1 * height + 1 = 6) 
  : ∃ XR RS : ℕ, RS ≠ 0 ∧ XR / RS = 1 := 
sorry

end decagon_ratio_bisect_l163_163939


namespace solve_system_of_equations_l163_163279

theorem solve_system_of_equations (x y : ℝ) :
  (x^4 + (7/2) * x^2 * y + 2 * y^3 = 0) ∧
  (4 * x^2 + 7 * x * y + 2 * y^3 = 0) →
  (x = 0 ∧ y = 0) ∨ (x = 2 ∧ y = -1) ∨ (x = -11 / 2 ∧ y = -11 / 2) :=
sorry

end solve_system_of_equations_l163_163279


namespace sally_quarters_after_purchases_l163_163713

-- Given conditions
def initial_quarters : ℕ := 760
def first_purchase : ℕ := 418
def second_purchase : ℕ := 192

-- Define the resulting quarters after purchases
def quarters_after_first_purchase (initial : ℕ) (spent : ℕ) : ℕ := initial - spent
def quarters_after_second_purchase (remaining : ℕ) (spent : ℕ) : ℕ := remaining - spent

-- The main statement to be proved
theorem sally_quarters_after_purchases :
  quarters_after_second_purchase (quarters_after_first_purchase initial_quarters first_purchase) second_purchase = 150 :=
by
  unfold quarters_after_first_purchase quarters_after_second_purchase initial_quarters first_purchase second_purchase
  simp
  sorry

end sally_quarters_after_purchases_l163_163713


namespace trig_inequality_l163_163833

theorem trig_inequality (theta : ℝ) (h1 : Real.pi / 4 < theta) (h2 : theta < Real.pi / 2) : 
  Real.cos theta < Real.sin theta ∧ Real.sin theta < Real.tan theta :=
sorry

end trig_inequality_l163_163833


namespace trams_required_l163_163552

theorem trams_required (initial_trams : ℕ) (initial_interval : ℚ) (reduction_fraction : ℚ) :
  initial_trams = 12 ∧ initial_interval = 5 ∧ reduction_fraction = 1/5 →
  (initial_trams + initial_trams * reduction_fraction - initial_trams) = 3 :=
by
  sorry

end trams_required_l163_163552


namespace fixed_monthly_fee_l163_163331

theorem fixed_monthly_fee (x y : ℝ) 
  (h1 : x + 20 * y = 15.20) 
  (h2 : x + 40 * y = 25.20) : 
  x = 5.20 := 
sorry

end fixed_monthly_fee_l163_163331


namespace average_stamps_collected_per_day_l163_163396

theorem average_stamps_collected_per_day :
  let a := 10
  let d := 6
  let n := 6
  let total_sum := (n / 2) * (2 * a + (n - 1) * d)
  let average := total_sum / n
  average = 25 :=
by
  sorry

end average_stamps_collected_per_day_l163_163396


namespace zoo_sea_lions_l163_163933

variable (S P : ℕ)

theorem zoo_sea_lions (h1 : S / P = 4 / 11) (h2 : P = S + 84) : S = 48 := 
sorry

end zoo_sea_lions_l163_163933


namespace problem_solution_l163_163523
open Real

theorem problem_solution (a b c : ℝ) (ha : 0 < a ∧ a < 1) (hb : 0 < b ∧ b < 1) (hc : 0 < c ∧ c < 1) :
  a * (1 - b) ≤ 1 / 4 ∨ b * (1 - c) ≤ 1 / 4 ∨ c * (1 - a) ≤ 1 / 4 :=
by
  sorry

end problem_solution_l163_163523


namespace xy_value_l163_163986

theorem xy_value (x y : ℝ) (h : x * (x + y) = x^2 + 18) : x * y = 18 := 
by
  sorry

end xy_value_l163_163986


namespace committee_membership_l163_163437

theorem committee_membership (n : ℕ) (h1 : 2 * n = 6) (h2 : (n - 1 : ℚ) / 5 = 0.4) : n = 3 := 
sorry

end committee_membership_l163_163437


namespace problem_1_problem_2_l163_163673

def f (x : ℝ) (a : ℝ) : ℝ := |x + 2| - |x + a|

theorem problem_1 (a : ℝ) (h : a = 3) :
  ∀ x, f x a ≤ 1/2 → x ≥ -11/4 := sorry

theorem problem_2 (a : ℝ) :
  (∀ x, f x a ≤ a) → a ≥ 1 := sorry

end problem_1_problem_2_l163_163673


namespace sum_of_fourth_powers_l163_163743

theorem sum_of_fourth_powers (n : ℤ) (h : (n - 2)^2 + n^2 + (n + 2)^2 = 2450) :
  (n - 2)^4 + n^4 + (n + 2)^4 = 1881632 :=
sorry

end sum_of_fourth_powers_l163_163743


namespace inequality_solution_l163_163722

theorem inequality_solution (x : ℝ) (h : x ≠ 4) : (x^2 - 16) / (x - 4) ≤ 0 ↔ x ∈ Set.Iic (-4) :=
by
  sorry

end inequality_solution_l163_163722


namespace length_of_chord_l163_163819

theorem length_of_chord 
  (a : ℝ)
  (h_sym : ∀ (x y : ℝ), (x^2 + y^2 - 2*x + 4*y = 0) → (3*x - a*y - 11 = 0))
  (h_line : 3 * 1 - a * (-2) - 11 = 0)
  (h_midpoint : (1 : ℝ) = (a / 4) ∧ (-1 : ℝ) = (-a / 4)) :
  let r := Real.sqrt 5
  let d := Real.sqrt ((1 - 1)^2 + (-1 + 2)^2)
  (2 * Real.sqrt (r^2 - d^2)) = 4 :=
by {
  -- Variables and assumptions would go here
  sorry
}

end length_of_chord_l163_163819


namespace volume_of_original_cube_l163_163267

theorem volume_of_original_cube (s : ℝ) (h : (s + 2) * (s - 3) * s - s^3 = 26) : s^3 = 343 := 
sorry

end volume_of_original_cube_l163_163267


namespace quadratic_solution_l163_163738

theorem quadratic_solution (a c: ℝ) (h1 : a + c = 7) (h2 : a < c) (h3 : 36 - 4 * a * c = 0) : 
  a = (7 - Real.sqrt 13) / 2 ∧ c = (7 + Real.sqrt 13) / 2 :=
by
  sorry

end quadratic_solution_l163_163738


namespace quadratic_function_is_parabola_l163_163101

theorem quadratic_function_is_parabola (a : ℝ) (b : ℝ) (c : ℝ) :
  ∃ k h, ∀ x, (y = a * (x - h)^2 + k) ∧ a ≠ 0 → (y = 3 * (x - 2)^2 + 6) → (a = 3 ∧ h = 2 ∧ k = 6) → ∀ x, (y = 3 * (x - 2)^2 + 6) := 
by
  sorry

end quadratic_function_is_parabola_l163_163101


namespace weight_order_l163_163873

variables (A B C D : ℝ) -- Representing the weights of objects A, B, C, and D as real numbers.

-- Conditions given in the problem:
axiom eq1 : A + B = C + D
axiom ineq1 : D + A > B + C
axiom ineq2 : B > A + C

-- Proof stating that the weights in ascending order are C < A < B < D.
theorem weight_order (A B C D : ℝ) : C < A ∧ A < B ∧ B < D :=
by
  -- We are not providing the proof steps here.
  sorry

end weight_order_l163_163873


namespace trams_to_add_l163_163555

theorem trams_to_add (initial_trams : ℕ) (initial_interval new_interval : ℤ)
  (reduce_by_fraction : ℤ) (total_distance : ℤ)
  (h1 : initial_trams = 12)
  (h2 : initial_interval = total_distance / initial_trams)
  (h3 : reduce_by_fraction = 5)
  (h4 : new_interval = initial_interval - initial_interval / reduce_by_fraction) :
  initial_trams + (total_distance / new_interval - initial_trams) = 15 :=
by
  sorry

end trams_to_add_l163_163555


namespace sqrt_factorial_product_l163_163574

theorem sqrt_factorial_product:
  sqrt ((fact 4) * (fact 4)) = 24 :=
by sorry

end sqrt_factorial_product_l163_163574


namespace nonneg_real_inequality_l163_163814

theorem nonneg_real_inequality (a b : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) : a^3 + b^3 ≥ Real.sqrt (a * b) * (a^2 + b^2) := 
by
  sorry

end nonneg_real_inequality_l163_163814


namespace max_vector_sum_l163_163512

open Real EuclideanSpace

noncomputable def circle_center : ℝ × ℝ := (3, 0)
noncomputable def radius : ℝ := 2
noncomputable def distance_AB : ℝ := 2 * sqrt 3

theorem max_vector_sum {A B : ℝ × ℝ} 
    (hA_on_circle : dist A circle_center = radius)
    (hB_on_circle : dist B circle_center = radius)
    (hAB_eq : dist A B = distance_AB) :
    (dist (0,0) ((A.1 + B.1, A.2 + B.2))) ≤ 8 :=
by 
  sorry

end max_vector_sum_l163_163512


namespace simplify_expression_l163_163277

variable {R : Type*} [Field R]

theorem simplify_expression (x y z : R) (h₁ : x ≠ 0) (h₂ : y ≠ 0) (h₃ : z ≠ 0) :
  (x + y + z)⁻¹ * (x⁻¹ + y⁻¹ + z⁻¹) = x⁻¹ * y⁻¹ * z⁻¹ :=
sorry

end simplify_expression_l163_163277


namespace parallelogram_angle_B_eq_130_l163_163380

theorem parallelogram_angle_B_eq_130 (A C B D : ℝ) (parallelogram_ABCD : true) 
(angles_sum_A_C : A + C = 100) (A_eq_C : A = C): B = 130 := by
  sorry

end parallelogram_angle_B_eq_130_l163_163380


namespace geometric_sequence_seventh_term_l163_163083

noncomputable def a_7 (a₁ q : ℝ) : ℝ :=
  a₁ * q^6

theorem geometric_sequence_seventh_term :
  a_7 3 (Real.sqrt 2) = 24 :=
by
  sorry

end geometric_sequence_seventh_term_l163_163083


namespace tony_schooling_years_l163_163433

theorem tony_schooling_years:
  let first_degree := 4
  let additional_degrees := 2 * 4
  let graduate_degree := 2
  first_degree + additional_degrees + graduate_degree = 14 :=
by {
  let first_degree := 4
  let additional_degrees := 2 * 4
  let graduate_degree := 2
  show first_degree + additional_degrees + graduate_degree = 14
  sorry
}

end tony_schooling_years_l163_163433


namespace primes_satisfying_equation_l163_163804

theorem primes_satisfying_equation :
  ∀ (p q : ℕ), p.Prime ∧ q.Prime → 
    (∃ x y z : ℕ, p^(2*x) + q^(2*y) = z^2) ↔ 
    (p = 2 ∧ q = 3) ∨ (p = 3 ∧ q = 2) := 
by
  sorry

end primes_satisfying_equation_l163_163804


namespace inequality_holds_l163_163344

theorem inequality_holds (k : ℝ) : (∀ x : ℝ, x^2 + k * x + 1 > 0) ↔ (k > -2 ∧ k < 2) :=
by
  sorry

end inequality_holds_l163_163344


namespace xy_equals_18_l163_163983

theorem xy_equals_18 (x y : ℝ) (h : x * (x + y) = x^2 + 18) : x * y = 18 :=
by
  sorry

end xy_equals_18_l163_163983


namespace ninety_one_square_friendly_unique_square_friendly_l163_163039

-- Given conditions
def square_friendly (c : ℤ) : Prop :=
  ∀ m : ℤ, ∃ n : ℤ, m^2 + 18 * m + c = n^2

-- Part (a)
theorem ninety_one_square_friendly : square_friendly 81 :=
sorry

-- Part (b)
theorem unique_square_friendly (c c' : ℤ) (h_c : square_friendly c) (h_c' : square_friendly c') : c = c' :=
sorry

end ninety_one_square_friendly_unique_square_friendly_l163_163039


namespace sqrt_factorial_equality_l163_163578

theorem sqrt_factorial_equality : Real.sqrt (4! * 4!) = 24 := 
by
  sorry

end sqrt_factorial_equality_l163_163578


namespace problem1_problem2_l163_163493

noncomputable def vec (α : ℝ) (β : ℝ) : ℝ × ℝ × ℝ × ℝ :=
  (Real.cos α, Real.sin α, Real.cos β, -Real.sin β)

theorem problem1 (α β : ℝ) (h1 : 0 < α ∧ α < Real.pi / 2) (h2 : 0 < β ∧ β < Real.pi / 2)
  (h3 : (Real.sqrt ((Real.cos α - Real.cos β) ^ 2 + (Real.sin α + Real.sin β) ^ 2)) = (Real.sqrt 10) / 5) :
  Real.cos (α + β) = 4 / 5 :=
by
  sorry

theorem problem2 (α β : ℝ) (h1 : 0 < α ∧ α < Real.pi / 2) (h2 : 0 < β ∧ β < Real.pi / 2)
  (h3 : Real.cos α = 3 / 5) (h4 : Real.cos (α + β) = 4 / 5) :
  Real.cos β = 24 / 25 :=
by
  sorry

end problem1_problem2_l163_163493


namespace parabola_focus_coordinates_l163_163796

theorem parabola_focus_coordinates (y x : ℝ) (h : y^2 = 8 * x) : (x, y) = (2, 0) :=
sorry

end parabola_focus_coordinates_l163_163796


namespace find_a_l163_163210

def A : Set ℝ := {0, 2}
def B (a : ℝ) : Set ℝ := {1, a ^ 2}

theorem find_a (a : ℝ) (h : A ∪ B a = {0, 1, 2, 4}) : a = 2 ∨ a = -2 :=
by
  sorry

end find_a_l163_163210


namespace assorted_candies_count_l163_163042

theorem assorted_candies_count
  (total_candies : ℕ)
  (chewing_gums : ℕ)
  (chocolate_bars : ℕ)
  (assorted_candies : ℕ) :
  total_candies = 50 →
  chewing_gums = 15 →
  chocolate_bars = 20 →
  assorted_candies = total_candies - (chewing_gums + chocolate_bars) →
  assorted_candies = 15 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  exact h4

end assorted_candies_count_l163_163042


namespace find_quadruples_l163_163656

open Nat

def is_prime (p : ℕ) : Prop := p > 1 ∧ ∀ d, d ∣ p → d = 1 ∨ d = p

theorem find_quadruples (a b p n : ℕ) (hp : is_prime p) (h_ab : a + b ≠ 0) :
  a^3 + b^3 = p^n ↔ (a = 1 ∧ b = 1 ∧ p = 2 ∧ n = 1) ∨
               (a = 1 ∧ b = 2 ∧ p = 3 ∧ n = 2) ∨ 
               (a = 2 ∧ b = 1 ∧ p = 3 ∧ n = 2) ∨
               ∃ (k : ℕ), (a = 2^k ∧ b = 2^k ∧ p = 2 ∧ n = 3*k + 1) ∨ 
                          (a = 2 * 3^k ∧ b = 3^k ∧ p = 3 ∧ n = 3*k + 2) ∨
                          (a = 3^k ∧ b = 2 * 3^k ∧ p = 3 ∧ n = 3*k + 2) := sorry

end find_quadruples_l163_163656


namespace total_amount_paid_l163_163135

theorem total_amount_paid (B : ℕ) (hB : B = 232) (A : ℕ) (hA : A = 3 / 2 * B) :
  A + B = 580 :=
by
  sorry

end total_amount_paid_l163_163135


namespace sqrt_factorial_multiplication_l163_163614

theorem sqrt_factorial_multiplication : (Real.sqrt ((4! : ℝ) * (4! : ℝ)) = 24) := 
by sorry

end sqrt_factorial_multiplication_l163_163614


namespace sum_proper_divisors_81_l163_163442

theorem sum_proper_divisors_81 : (3^0 + 3^1 + 3^2 + 3^3) = 40 :=
by
  sorry

end sum_proper_divisors_81_l163_163442


namespace john_outside_doors_count_l163_163387

theorem john_outside_doors_count 
  (bedroom_doors : ℕ := 3) 
  (cost_outside_door : ℕ := 20) 
  (total_cost : ℕ := 70) 
  (cost_bedroom_door := cost_outside_door / 2) 
  (total_bedroom_cost := bedroom_doors * cost_bedroom_door) 
  (outside_doors := (total_cost - total_bedroom_cost) / cost_outside_door) : 
  outside_doors = 2 :=
by
  sorry

end john_outside_doors_count_l163_163387


namespace accessories_per_doll_l163_163692

theorem accessories_per_doll (n dolls accessories time_per_doll time_per_accessory total_time : ℕ)
  (h0 : dolls = 12000)
  (h1 : time_per_doll = 45)
  (h2 : time_per_accessory = 10)
  (h3 : total_time = 1860000)
  (h4 : time_per_doll + accessories * time_per_accessory = n)
  (h5 : dolls * n = total_time) :
  accessories = 11 :=
by
  sorry

end accessories_per_doll_l163_163692


namespace smallest_positive_period_of_f_f_ge_negative_sqrt_3_in_interval_l163_163972

noncomputable def f (x : Real) : Real :=
  Real.sin x * Real.cos x - Real.sqrt 3 * (Real.sin x) ^ 2

theorem smallest_positive_period_of_f :
  ∃ T > 0, ∀ x, f (x + T) = f x ∧ (∀ T' > 0, ( ∀ x, f (x + T') = f x) → T ≤ T') := by
  sorry

theorem f_ge_negative_sqrt_3_in_interval :
  ∀ x ∈ Set.Icc (-Real.pi / 3) (Real.pi / 6), f x ≥ -Real.sqrt 3 := by
  sorry

end smallest_positive_period_of_f_f_ge_negative_sqrt_3_in_interval_l163_163972


namespace problem_inequality_l163_163489

noncomputable def A (x : ℝ) := (x - 3) ^ 2
noncomputable def B (x : ℝ) := (x - 2) * (x - 4)

theorem problem_inequality (x : ℝ) : A x > B x :=
  by
    sorry

end problem_inequality_l163_163489


namespace elapsed_time_l163_163265

variable (totalDistance : ℕ) (runningSpeed : ℕ) (distanceRemaining : ℕ)

theorem elapsed_time (h1 : totalDistance = 120) (h2 : runningSpeed = 4) (h3 : distanceRemaining = 20) :
  (totalDistance - distanceRemaining) / runningSpeed = 25 := by
sorry

end elapsed_time_l163_163265


namespace chess_positions_after_one_move_each_l163_163675

def number_of_chess_positions (initial_positions : ℕ) (pawn_moves : ℕ) (knight_moves : ℕ) (active_pawns : ℕ) (active_knights : ℕ) : ℕ :=
  let pawn_move_combinations := active_pawns * pawn_moves
  let knight_move_combinations := active_knights * knight_moves
  pawn_move_combinations + knight_move_combinations

theorem chess_positions_after_one_move_each :
  number_of_chess_positions 1 2 2 8 2 * number_of_chess_positions 1 2 2 8 2 = 400 :=
by
  sorry

end chess_positions_after_one_move_each_l163_163675


namespace find_initial_population_l163_163308

-- Define the initial population, conditions and the final population
variable (P : ℕ)

noncomputable def initial_population (P : ℕ) :=
  (0.85 * (0.92 * P) : ℝ) = 3553

theorem find_initial_population (P : ℕ) (h : initial_population P) : P = 4546 := sorry

end find_initial_population_l163_163308


namespace loop_until_correct_l163_163912

-- Define the conditions
def num_iterations := 20

-- Define the loop condition
def loop_condition (i : Nat) : Prop := i > num_iterations

-- Theorem: Proof that the loop should continue until the counter i exceeds 20
theorem loop_until_correct (i : Nat) : loop_condition i := by
  sorry

end loop_until_correct_l163_163912


namespace inverse_negative_exchange_l163_163457

theorem inverse_negative_exchange (f1 f2 f3 f4 : ℝ → ℝ) (hx1 : ∀ x, f1 x = x - (1/x))
  (hx2 : ∀ x, f2 x = x + (1/x)) (hx3 : ∀ x, f3 x = Real.log x)
  (hx4 : ∀ x, f4 x = if 0 < x ∧ x < 1 then x else if x = 1 then 0 else -(1/x)) :
  (∀ x, f1 (1/x) = -f1 x) ∧ (∀ x, f2 (1/x) = -f2 x) ∧ (∀ x, f3 (1/x) = -f3 x) ∧
  (∀ x, f4 (1/x) = -f4 x) ↔ True := by 
  sorry

end inverse_negative_exchange_l163_163457


namespace domain_of_f_l163_163116

noncomputable def f (x : ℝ) : ℝ := 1 / (Real.sqrt (1 - x^2)) + x^0

theorem domain_of_f :
  {x : ℝ | 1 - x^2 > 0 ∧ x ≠ 0} = {x : ℝ | -1 < x ∧ x < 1 ∧ x ≠ 0} :=
by
  sorry

end domain_of_f_l163_163116


namespace ratio_of_areas_l163_163905

theorem ratio_of_areas (r : ℝ) (h1 : r > 0) : 
  let OX := r / 3
  let area_OP := π * r ^ 2
  let area_OX := π * (OX) ^ 2
  (area_OX / area_OP) = 1 / 9 :=
by
  sorry

end ratio_of_areas_l163_163905


namespace sqrt_factorial_multiplication_l163_163613

theorem sqrt_factorial_multiplication : (Real.sqrt ((4! : ℝ) * (4! : ℝ)) = 24) := 
by sorry

end sqrt_factorial_multiplication_l163_163613


namespace sum_proper_divisors_of_81_l163_163445

theorem sum_proper_divisors_of_81 : (∑ i in {0, 1, 2, 3}, 3 ^ i) = 40 := 
by
  sorry

end sum_proper_divisors_of_81_l163_163445


namespace rhombus_diagonals_sum_squares_l163_163080

-- Definition of the rhombus side length condition
def is_rhombus_side_length (side_length : ℝ) : Prop :=
  side_length = 2

-- Lean 4 statement for the proof problem
theorem rhombus_diagonals_sum_squares (side_length : ℝ) (d1 d2 : ℝ) 
  (h : is_rhombus_side_length side_length) :
  side_length = 2 → (d1^2 + d2^2 = 16) :=
by
  sorry

end rhombus_diagonals_sum_squares_l163_163080


namespace vertex_of_parabola_is_correct_l163_163543

theorem vertex_of_parabola_is_correct :
  ∀ x y : ℝ, y = -5 * (x + 2) ^ 2 - 6 → (x = -2 ∧ y = -6) :=
by
  sorry

end vertex_of_parabola_is_correct_l163_163543


namespace chloe_boxes_l163_163045

/-- Chloe was unboxing some of her old winter clothes. She found some boxes of clothing and
inside each box, there were 2 scarves and 6 mittens. Chloe had a total of 32 pieces of
winter clothing. How many boxes of clothing did Chloe find? -/
theorem chloe_boxes (boxes : ℕ) (total_clothing : ℕ) (pieces_per_box : ℕ) :
  pieces_per_box = 8 -> total_clothing = 32 -> total_clothing / pieces_per_box = boxes -> boxes = 4 :=
by
  intros
  sorry

end chloe_boxes_l163_163045


namespace shaded_region_area_l163_163127

open Real

noncomputable def area_of_shaded_region (side : ℝ) (radius : ℝ) : ℝ :=
  let area_square := side ^ 2
  let area_sector := π * radius ^ 2 / 4
  let area_triangle := (1 / 2) * (side / 2) * sqrt ((side / 2) ^ 2 - radius ^ 2)
  area_square - 8 * area_triangle - 4 * area_sector

theorem shaded_region_area (h_side : ℝ) (h_radius : ℝ)
  (h1 : h_side = 8) (h2 : h_radius = 3) :
  area_of_shaded_region h_side h_radius = 64 - 16 * sqrt 7 - 3 * π :=
by
  rw [h1, h2]
  sorry

end shaded_region_area_l163_163127


namespace ad_space_length_l163_163008

theorem ad_space_length 
  (num_companies : ℕ)
  (ads_per_company : ℕ)
  (width : ℝ)
  (cost_per_sq_ft : ℝ)
  (total_cost : ℝ) 
  (H1 : num_companies = 3)
  (H2 : ads_per_company = 10)
  (H3 : width = 5)
  (H4 : cost_per_sq_ft = 60)
  (H5 : total_cost = 108000) :
  ∃ L : ℝ, (num_companies * ads_per_company * width * L * cost_per_sq_ft = total_cost) ∧ (L = 12) :=
by
  sorry

end ad_space_length_l163_163008


namespace sqrt_factorial_product_eq_24_l163_163570

theorem sqrt_factorial_product_eq_24 : (sqrt (fact 4 * fact 4) = 24) :=
by sorry

end sqrt_factorial_product_eq_24_l163_163570


namespace sum_of_midpoints_double_l163_163006

theorem sum_of_midpoints_double (a b c : ℝ) (h : a + b + c = 15) : 
  (a + b) + (a + c) + (b + c) = 30 :=
by
  -- We skip the proof according to the instruction
  sorry

end sum_of_midpoints_double_l163_163006


namespace find_number_l163_163298

theorem find_number (x : ℕ) (h : (9 * x) / 3 = 27) : x = 9 :=
by
  sorry

end find_number_l163_163298


namespace sin_45_eq_sqrt2_div_2_l163_163744

theorem sin_45_eq_sqrt2_div_2 : Real.sin (Real.pi / 4) = (Real.sqrt 2) / 2 := 
  sorry

end sin_45_eq_sqrt2_div_2_l163_163744


namespace average_of_original_set_l163_163541

theorem average_of_original_set (A : ℝ) (h1 : 7 * A = 125 * 7 / 5) : A = 25 := 
sorry

end average_of_original_set_l163_163541


namespace carlotta_total_time_l163_163956

-- Define the main function for calculating total time
def total_time (performance_time practicing_ratio tantrum_ratio : ℕ) : ℕ :=
  performance_time + (performance_time * practicing_ratio) + (performance_time * tantrum_ratio)

-- Define the conditions from the problem
def singing_time := 6
def practicing_per_minute := 3
def tantrums_per_minute := 5

-- The expected total time based on the conditions
def expected_total_time := 54

-- The theorem to prove the equivalence
theorem carlotta_total_time :
  total_time singing_time practicing_per_minute tantrums_per_minute = expected_total_time :=
by
  sorry

end carlotta_total_time_l163_163956


namespace age_difference_between_Mandy_and_sister_l163_163395

variable (Mandy_age Brother_age Sister_age : ℕ)

-- Given conditions
def Mandy_is_3_years_old : Mandy_age = 3 := by sorry
def Brother_is_4_times_older : Brother_age = 4 * Mandy_age := by sorry
def Sister_is_5_years_younger_than_brother : Sister_age = Brother_age - 5 := by sorry

-- Prove the question
theorem age_difference_between_Mandy_and_sister :
  Mandy_age = 3 ∧ Brother_age = 4 * Mandy_age ∧ Sister_age = Brother_age - 5 → Sister_age - Mandy_age = 4 := 
by 
  sorry

end age_difference_between_Mandy_and_sister_l163_163395


namespace min_val_xy_l163_163966

theorem min_val_xy (x y : ℝ) 
  (h : 2 * (Real.cos (x + y - 1))^2 = ((x + 1)^2 + (y - 1)^2 - 2 * x * y) / (x - y + 1)) : 
  xy ≥ (1 / 4) :=
sorry

end min_val_xy_l163_163966


namespace sqrt_factorial_product_eq_24_l163_163572

theorem sqrt_factorial_product_eq_24 : (sqrt (fact 4 * fact 4) = 24) :=
by sorry

end sqrt_factorial_product_eq_24_l163_163572


namespace x_finishes_in_24_days_l163_163757

variable (x y : Type) [Inhabited x] [Inhabited y]

/-- 
  y can finish the work in 16 days,
  y worked for 10 days and left the job,
  x alone needs 9 days to finish the remaining work,
  How many days does x need to finish the work alone?
-/
theorem x_finishes_in_24_days
  (days_y : ℕ := 16)
  (work_done_y : ℕ := 10)
  (work_left_x : ℕ := 9)
  (D_x : ℕ) :
  (1 / days_y : ℚ) * work_done_y + (1 / D_x) * work_left_x = 1 / D_x :=
by
  sorry

end x_finishes_in_24_days_l163_163757


namespace sqrt_factorial_equality_l163_163579

theorem sqrt_factorial_equality : Real.sqrt (4! * 4!) = 24 := 
by
  sorry

end sqrt_factorial_equality_l163_163579


namespace min_value_fraction_sum_l163_163197

open Real

theorem min_value_fraction_sum (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) (h_sum : x + y + z = 2) :
    ∃ m, m = (1 / (x + y) + 1 / (x + z) + 1 / (y + z)) ∧ m = 9/4 :=
by
  sorry

end min_value_fraction_sum_l163_163197


namespace base7_addition_l163_163769

theorem base7_addition : (21 : ℕ) + 254 = 505 :=
by sorry

end base7_addition_l163_163769


namespace determine_g_l163_163650

theorem determine_g (g : ℝ → ℝ) : (∀ x : ℝ, 4 * x^4 + x^3 - 2 * x + 5 + g x = 2 * x^3 - 7 * x^2 + 4) →
  (∀ x : ℝ, g x = -4 * x^4 + x^3 - 7 * x^2 + 2 * x - 1) :=
by
  intro h
  sorry

end determine_g_l163_163650


namespace fraction_meaningful_l163_163117

theorem fraction_meaningful (x : ℝ) : (x ≠ -1) ↔ (∃ k : ℝ, k = 1 / (x + 1)) :=
by
  sorry

end fraction_meaningful_l163_163117


namespace line_intersects_iff_sufficient_l163_163967

noncomputable def sufficient_condition (b : ℝ) : Prop :=
b > 1

noncomputable def condition (b : ℝ) : Prop :=
b > 0

noncomputable def line_intersects_hyperbola (b : ℝ) : Prop :=
b > 2 / 3

theorem line_intersects_iff_sufficient (b : ℝ) (h : condition b) : 
  (sufficient_condition b) → (line_intersects_hyperbola b) ∧ ¬(line_intersects_hyperbola b) → (sufficient_condition b) :=
by {
  sorry
}

end line_intersects_iff_sufficient_l163_163967


namespace johns_total_distance_l163_163386

theorem johns_total_distance :
  let monday := 1700
  let tuesday := monday + 200
  let wednesday := 0.7 * tuesday
  let thursday := 2 * wednesday
  let friday := 3.5 * 1000
  let saturday := 0
  monday + tuesday + wednesday + thursday + friday + saturday = 10090 := 
by
  sorry

end johns_total_distance_l163_163386


namespace gcd_max_value_l163_163428

theorem gcd_max_value (a b : ℕ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : a + b = 1005) : ∃ d, d = Int.gcd a b ∧ d = 335 :=
by {
  sorry
}

end gcd_max_value_l163_163428


namespace right_triangle_inscribed_circle_area_l163_163890

noncomputable def inscribed_circle_area (p c : ℝ) : ℝ :=
  π * (p - c) ^ 2

theorem right_triangle_inscribed_circle_area (p c : ℝ) (h : c > 0) 
  (perimeter : ∃ x y : ℝ, 2 * p = x + y + c ∧ x > 0 ∧ y > 0) :
  inscribed_circle_area p c = π * (p - c) ^ 2 := by 
  sorry

end right_triangle_inscribed_circle_area_l163_163890


namespace probability_divisible_by_3_l163_163778

-- Define the set of numbers
def S : Set ℕ := {2, 3, 5, 6}

-- Define the pairs of numbers whose product is divisible by 3
def valid_pairs : Set (ℕ × ℕ) := {(2, 3), (2, 6), (3, 5), (3, 6), (5, 6)}

-- Define the total number of pairs
def total_pairs := 6

-- Define the number of valid pairs
def valid_pairs_count := 5

-- Prove that the probability of choosing two numbers whose product is divisible by 3 is 5/6
theorem probability_divisible_by_3 : (valid_pairs_count / total_pairs : ℚ) = 5 / 6 := by
  sorry

end probability_divisible_by_3_l163_163778


namespace question1_question2_l163_163353

variable (α : ℝ)

theorem question1 (h1 : (π / 2) < α) (h2 : α < π) (h3 : Real.sin α = 3 / 5) :
    (Real.sin α ^ 2 + Real.sin (2 * α)) / (Real.cos α ^ 2 + Real.cos (2 * α)) = -15 / 23 := by
  sorry

theorem question2 (h1 : (π / 2) < α) (h2 : α < π) (h3 : Real.sin α = 3 / 5) :
    Real.tan (α - 5 * π / 4) = -7 := by
  sorry

end question1_question2_l163_163353


namespace david_distance_to_airport_l163_163475

theorem david_distance_to_airport (t : ℝ) (d : ℝ) :
  (35 * (t + 1) = d) ∧ (d - 35 = 50 * (t - 1.5)) → d = 210 :=
by
  sorry

end david_distance_to_airport_l163_163475


namespace delivery_payment_l163_163708

-- Define the problem conditions and the expected outcome
theorem delivery_payment 
    (deliveries_Oula : ℕ) 
    (deliveries_Tona : ℕ) 
    (difference_in_pay : ℝ) 
    (P : ℝ) 
    (H1 : deliveries_Oula = 96) 
    (H2 : deliveries_Tona = 72) 
    (H3 : difference_in_pay = 2400) :
    96 * P - 72 * P = 2400 → P = 100 :=
by
  intro h1
  sorry

end delivery_payment_l163_163708


namespace unit_price_of_first_batch_minimum_selling_price_l163_163531

-- Proof Problem 1
theorem unit_price_of_first_batch :
  (∃ x : ℝ, (3200 / x) * 2 = 7200 / (x + 10) ∧ x = 80) := 
  sorry

-- Proof Problem 2
theorem minimum_selling_price (x : ℝ) (hx : x = 80) :
  (40 * x + 80 * (x + 10) - 3200 - 7200 + 20 * 0.8 * x ≥ 3520) → 
  (∃ y : ℝ, y ≥ 120) :=
  sorry

end unit_price_of_first_batch_minimum_selling_price_l163_163531


namespace bathroom_square_footage_l163_163020

theorem bathroom_square_footage 
  (tiles_width : ℕ) (tiles_length : ℕ) (tile_size_inch : ℕ)
  (inch_to_foot : ℕ) 
  (h_width : tiles_width = 10) 
  (h_length : tiles_length = 20)
  (h_tile_size : tile_size_inch = 6)
  (h_inch_to_foot : inch_to_foot = 12) :
  let tile_size_foot : ℚ := tile_size_inch / inch_to_foot
  let width_foot : ℚ := tiles_width * tile_size_foot
  let length_foot : ℚ := tiles_length * tile_size_foot
  let area : ℚ := width_foot * length_foot
  area = 50 := 
by
  sorry

end bathroom_square_footage_l163_163020


namespace preimage_of_5_1_is_2_3_l163_163491

-- Define the mapping function
def f (p : ℝ × ℝ) : ℝ × ℝ := (p.1 + p.2, 2*p.1 - p.2)

-- Define the pre-image condition for (5, 1)
theorem preimage_of_5_1_is_2_3 : ∃ p : ℝ × ℝ, f p = (5, 1) ∧ p = (2, 3) :=
by
  -- Here we state that such a point p exists with the required properties.
  sorry

end preimage_of_5_1_is_2_3_l163_163491


namespace triangle_area_zero_vertex_l163_163623

theorem triangle_area_zero_vertex (x1 y1 x2 y2 : ℝ) :
  (1 / 2) * |x1 * y2 - x2 * y1| = 
    abs (1 / 2 * (x1 * y2 - x2 * y1)) := 
sorry

end triangle_area_zero_vertex_l163_163623


namespace carlotta_performance_time_l163_163954

theorem carlotta_performance_time :
  ∀ (s p t : ℕ),  -- s for singing, p for practicing, t for tantrums
  (∀ (n : ℕ), p = 3 * n ∧ t = 5 * n) →
  s = 6 →
  (s + p + t) = 54 :=
by 
  intros s p t h1 h2
  rcases h1 1 with ⟨h3, h4⟩
  sorry

end carlotta_performance_time_l163_163954


namespace units_digit_base6_product_l163_163044

theorem units_digit_base6_product (a b : ℕ) (h1 : a = 168) (h2 : b = 59) : ((a * b) % 6) = 0 := by
  sorry

end units_digit_base6_product_l163_163044


namespace sqrt_factorial_product_l163_163600

theorem sqrt_factorial_product :
  Nat.sqrt (Nat.factorial 4 * Nat.factorial 4) = 24 := 
sorry

end sqrt_factorial_product_l163_163600


namespace sqrt_factorial_multiplication_l163_163615

theorem sqrt_factorial_multiplication : (Real.sqrt ((4! : ℝ) * (4! : ℝ)) = 24) := 
by sorry

end sqrt_factorial_multiplication_l163_163615


namespace part1_part2_l163_163537

theorem part1 : (2 / 9 - 1 / 6 + 1 / 18) * (-18) = -2 := 
by
  sorry

theorem part2 : 54 * (3 / 4 + 1 / 2 - 1 / 4) = 54 := 
by
  sorry

end part1_part2_l163_163537


namespace aunt_li_more_cost_effective_l163_163041

theorem aunt_li_more_cost_effective (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  (100 * a + 100 * b) / 200 ≥ 200 / ((100 / a) + (100 / b)) :=
by
  sorry

end aunt_li_more_cost_effective_l163_163041


namespace unique_solution_to_exponential_poly_equation_l163_163214

noncomputable def polynomial_has_unique_real_solution : Prop :=
  ∃! x : ℝ, (2 : ℝ)^(3 * x + 3) - 3 * (2 : ℝ)^(2 * x + 1) - (2 : ℝ)^x + 1 = 0

theorem unique_solution_to_exponential_poly_equation :
  polynomial_has_unique_real_solution :=
sorry

end unique_solution_to_exponential_poly_equation_l163_163214


namespace kevin_cards_found_l163_163249

theorem kevin_cards_found : ∀ (initial_cards found_cards total_cards : Nat), 
  initial_cards = 7 → 
  total_cards = 54 → 
  total_cards - initial_cards = found_cards →
  found_cards = 47 :=
by
  intros initial_cards found_cards total_cards h1 h2 h3
  rw [h1, h2] at h3
  exact h3


end kevin_cards_found_l163_163249


namespace seashells_given_l163_163273

theorem seashells_given (original_seashells : ℕ) (current_seashells : ℕ) (given_seashells : ℕ) 
  (h1 : original_seashells = 35) 
  (h2 : current_seashells = 17) 
  (h3 : given_seashells = original_seashells - current_seashells) : 
  given_seashells = 18 := 
by 
  sorry

end seashells_given_l163_163273


namespace smallest_a_value_l163_163111

theorem smallest_a_value {a b c : ℝ} :
  (∃ (a b c : ℝ), (∀ x, (a * (x - 1/2)^2 - 5/4 = a * x^2 + b * x + c)) ∧ a > 0 ∧ ∃ n : ℤ, a + b + c = n)
  → (∃ (a : ℝ), a = 1) :=
by
  sorry

end smallest_a_value_l163_163111


namespace students_left_during_year_l163_163082

theorem students_left_during_year (initial_students : ℕ) (new_students : ℕ) (final_students : ℕ) (students_left : ℕ) :
  initial_students = 4 →
  new_students = 42 →
  final_students = 43 →
  students_left = initial_students + new_students - final_students →
  students_left = 3 :=
by
  intro h_initial h_new h_final h_students_left
  rw [h_initial, h_new, h_final] at h_students_left
  exact h_students_left

end students_left_during_year_l163_163082


namespace tulip_price_correct_l163_163263

-- Initial conditions
def first_day_tulips : ℕ := 30
def first_day_roses : ℕ := 20
def second_day_tulips : ℕ := 60
def second_day_roses : ℕ := 40
def third_day_tulips : ℕ := 6
def third_day_roses : ℕ := 16
def rose_price : ℝ := 3
def total_revenue : ℝ := 420

-- Question: What is the price of one tulip?
def tulip_price (T : ℝ) : ℝ :=
    first_day_tulips * T + first_day_roses * rose_price +
    second_day_tulips * T + second_day_roses * rose_price +
    third_day_tulips * T + third_day_roses * rose_price

-- Proof problem statement
theorem tulip_price_correct (T : ℝ) : tulip_price T = total_revenue → T = 2 :=
by
  sorry

end tulip_price_correct_l163_163263


namespace roots_single_circle_or_line_l163_163268

open Complex

theorem roots_single_circle_or_line (a b c d : ℂ) (n : ℕ) (h : n > 0) :
  ∀ z : ℂ, a * (z - b)^n = c * (z - d)^n → 
  (∃ r : ℝ, ∀ z : ℂ, a * (z - b)^n = c * (z - d)^n → |z - b| = r ∨ ∃ m : ℝ, ∀ z : ℂ, a * (z - b)^n = c * (z - d)^n → z = m * (z - d)) :=
by
  sorry

end roots_single_circle_or_line_l163_163268


namespace ratio_of_areas_of_similar_triangles_l163_163348

-- Define the variables and conditions
variables {ABC DEF : Type} 
variables (hABCDEF : Similar ABC DEF) 
variables (perimeterABC perimeterDEF : ℝ)
variables (hpABC : perimeterABC = 3)
variables (hpDEF : perimeterDEF = 1)

-- The theorem statement
theorem ratio_of_areas_of_similar_triangles :
  (perimeterABC / perimeterDEF) ^ 2 = 9 :=
by
  sorry

end ratio_of_areas_of_similar_triangles_l163_163348


namespace expression_value_l163_163446

theorem expression_value 
  (a b c : ℕ) 
  (ha : a = 12) 
  (hb : b = 2) 
  (hc : c = 7) :
  (a - (b - c)) - ((a - b) - c) = 14 := 
by 
  sorry

end expression_value_l163_163446


namespace problem_solution_l163_163015

noncomputable def expression_value : ℝ :=
  ((12.983 * 26) / 200) ^ 3 * Real.log 5 / Real.log 10

theorem problem_solution : expression_value = 3.361 := by
  sorry

end problem_solution_l163_163015


namespace determine_a_for_unique_solution_of_quadratic_l163_163069

theorem determine_a_for_unique_solution_of_quadratic :
  {a : ℝ | ∃! x : ℝ, a * x^2 - 4 * x + 2 = 0} = {0, 2} :=
sorry

end determine_a_for_unique_solution_of_quadratic_l163_163069


namespace sweets_neither_red_nor_green_l163_163292

theorem sweets_neither_red_nor_green (total_sweets : ℕ) (red_sweets : ℕ) (green_sweets : ℕ) 
  (h_total : total_sweets = 285) (h_red : red_sweets = 49) (h_green : green_sweets = 59) :
  total_sweets - (red_sweets + green_sweets) = 177 :=
by 
  rw [h_total, h_red, h_green]
  sorry

end sweets_neither_red_nor_green_l163_163292


namespace remainder_31_31_plus_31_mod_32_l163_163617

theorem remainder_31_31_plus_31_mod_32 : (31 ^ 31 + 31) % 32 = 30 := 
by sorry

end remainder_31_31_plus_31_mod_32_l163_163617


namespace sqrt_factorial_product_l163_163601

def factorial (n : Nat) : Nat :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem sqrt_factorial_product :
  sqrt (factorial 4 * factorial 4) = 24 :=
by
  sorry

end sqrt_factorial_product_l163_163601


namespace log5_15625_eq_6_l163_163789

noncomputable def log5_15625 : ℕ := Real.log 15625 / Real.log 5

theorem log5_15625_eq_6 : log5_15625 = 6 :=
by
  sorry

end log5_15625_eq_6_l163_163789


namespace pyramid_volume_correct_l163_163321

noncomputable def pyramid_volume (A_PQRS A_PQT A_RST: ℝ) (side: ℝ) (height: ℝ) : ℝ :=
  (1 / 3) * A_PQRS * height

theorem pyramid_volume_correct 
  (A_PQRS : ℝ) (A_PQT : ℝ) (A_RST : ℝ) (side : ℝ) (height_PQT : ℝ) (height_RST : ℝ)
  (h_PQT : 2 * A_PQT / side = height_PQT)
  (h_RST : 2 * A_RST / side = height_RST)
  (eq1 : height_PQT^2 + side^2 = height_RST^2 + (side - height_PQT)^2) 
  (eq2 : height_RST^2 = height_PQT^2 + (height_PQT - side)^2)
  : pyramid_volume A_PQRS A_PQT A_RST = 5120 / 3 :=
by
  -- Skipping the proof steps
  sorry

end pyramid_volume_correct_l163_163321


namespace sum_of_proper_divisors_of_81_l163_163440

theorem sum_of_proper_divisors_of_81 : 
  (∑ k in finset.range 4, 3^k) = 40 :=
by
  sorry

end sum_of_proper_divisors_of_81_l163_163440


namespace project_presentation_periods_l163_163025

def students : ℕ := 32
def period_length : ℕ := 40
def presentation_time_per_student : ℕ := 5

theorem project_presentation_periods : 
  (students * presentation_time_per_student) / period_length = 4 := by
  sorry

end project_presentation_periods_l163_163025


namespace profit_percentage_l163_163763

theorem profit_percentage (SP CP : ℤ) (h_SP : SP = 1170) (h_CP : CP = 975) :
  ((SP - CP : ℤ) * 100) / CP = 20 :=
by 
  sorry

end profit_percentage_l163_163763


namespace sqrt_factorial_eq_l163_163608

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem sqrt_factorial_eq :
  sqrt (factorial 4 * factorial 4) = 24 :=
by
  have h : factorial 4 = 24 := by
    unfold factorial
    simpa using [factorial, factorial, factorial]
  rw [h, h]
  sorry

end sqrt_factorial_eq_l163_163608


namespace determine_wholesale_prices_l163_163160

variables (p1 p2 d k : ℝ)
variables (h1 : 1 < k)
variables (prices : Finset ℝ)
variables (p_plus : ℝ → ℝ) (star : ℝ → ℝ)

noncomputable def BakeryPrices : Prop :=
  p_plus p1 = p1 + d ∧ p_plus p2 = p2 + d ∧ 
  star p1 = k * p1 ∧ star p2 = k * p2 ∧ 
  prices = {64, 64, 70, 72} ∧ 
  ∃ (q1 q2 q3 q4 : ℝ), 
    (q1 = p_plus p1 ∨ q1 = star p1 ∨ q1 = p_plus p2 ∨ q1 = star p2) ∧
    (q2 = p_plus p1 ∨ q2 = star p1 ∨ q2 = p_plus p2 ∨ q2 = star p2) ∧
    (q3 = p_plus p1 ∨ q3 = star p1 ∨ q3 = p_plus p2 ∨ q3 = star p2) ∧
    (q4 = p_plus p1 ∨ q4 = star p1 ∨ q4 = p_plus p2 ∨ q4 = star p2) ∧
    prices = {q1, q2, q3, q4}

theorem determine_wholesale_prices (p1 p2 : ℝ) (d k : ℝ)
    (h1 : 1 < k) (prices : Finset ℝ) :
  BakeryPrices p1 p2 d k h1 prices (λ p, p + d) (λ p, k * p) :=
sorry

end determine_wholesale_prices_l163_163160


namespace sqrt_factorial_mul_factorial_eq_l163_163595

theorem sqrt_factorial_mul_factorial_eq :
  Real.sqrt (Nat.factorial 4 * Nat.factorial 4) = 24 := by
sorry

end sqrt_factorial_mul_factorial_eq_l163_163595


namespace general_cosine_identity_l163_163924

theorem general_cosine_identity (α : ℝ) :
  real.cos (α - 120 * real.pi / 180) + real.cos α + real.cos (α + 120 * real.pi / 180) = 0 := by
  sorry

end general_cosine_identity_l163_163924


namespace women_at_each_table_l163_163465

/-- A waiter had 5 tables, each with 3 men and some women, and a total of 40 customers.
    Prove that there are 5 women at each table. -/
theorem women_at_each_table (W : ℕ) (total_customers : ℕ) (men_per_table : ℕ) (tables : ℕ)
  (h1 : total_customers = 40) (h2 : men_per_table = 3) (h3 : tables = 5) :
  (W * tables + men_per_table * tables = total_customers) → (W = 5) :=
by
  sorry

end women_at_each_table_l163_163465


namespace vector_orthogonality_l163_163978

variables (x : ℝ)

def vec_a := (x - 1, 2)
def vec_b := (1, x)

theorem vector_orthogonality :
  (vec_a x).fst * (vec_b x).fst + (vec_a x).snd * (vec_b x).snd = 0 ↔ x = 1 / 3 := by
  sorry

end vector_orthogonality_l163_163978


namespace param_A_valid_param_B_valid_l163_163418

-- Definition of the line equation
def line_eq (x y : ℝ) : Prop := y = 2 * x - 4

-- Parameterization A
def param_A (t : ℝ) : ℝ × ℝ := (2 - t, -2 * t)

-- Parameterization B
def param_B (t : ℝ) : ℝ × ℝ := (5 * t, 10 * t - 4)

-- Theorem to prove that parameterization A satisfies the line equation
theorem param_A_valid (t : ℝ) : line_eq (param_A t).1 (param_A t).2 := by
  sorry

-- Theorem to prove that parameterization B satisfies the line equation
theorem param_B_valid (t : ℝ) : line_eq (param_B t).1 (param_B t).2 := by
  sorry

end param_A_valid_param_B_valid_l163_163418


namespace complement_intersection_l163_163091

def U : Set ℝ := Set.univ
def A : Set ℝ := {x | 3 ≤ x}
def B : Set ℝ := {x | 0 ≤ x ∧ x < 5}

theorem complement_intersection (x : ℝ) : x ∈ (U \ A ∩ B) ↔ (0 ≤ x ∧ x < 3) :=
by {
  sorry
}

end complement_intersection_l163_163091


namespace fraction_addition_l163_163481

theorem fraction_addition :
  (5 / (8 / 13) + 4 / 7) = (487 / 56) := by
  sorry

end fraction_addition_l163_163481


namespace escalator_time_l163_163383

theorem escalator_time
    {d i s : ℝ}
    (h1 : d = 90 * i)
    (h2 : d = 30 * (i + s))
    (h3 : s = 2 * i):
    d / s = 45 := by
  sorry

end escalator_time_l163_163383


namespace James_present_age_l163_163874

variable (D J : ℕ)

theorem James_present_age 
  (h1 : D / J = 6 / 5)
  (h2 : D + 4 = 28) :
  J = 20 := 
by
  sorry

end James_present_age_l163_163874


namespace product_of_roots_l163_163439

theorem product_of_roots : ∃ (x : ℕ), x = 45 ∧ (∃ a b c : ℕ, a ^ 3 = 27 ∧ b ^ 4 = 81 ∧ c ^ 2 = 25 ∧ x = a * b * c) := 
sorry

end product_of_roots_l163_163439


namespace find_digit_for_multiple_of_3_l163_163060

theorem find_digit_for_multiple_of_3 (d : ℕ) (h : d < 10) : 
  (56780 + d) % 3 = 0 ↔ d = 1 :=
by sorry

end find_digit_for_multiple_of_3_l163_163060


namespace probability_sunglasses_to_hat_l163_163685

variable (S H : Finset ℕ) -- S: set of people wearing sunglasses, H: set of people wearing hats
variable (num_S : Nat) (num_H : Nat) (num_SH : Nat)
variable (prob_hat_to_sunglasses : ℚ)

-- Conditions
def condition1 : num_S = 80 := sorry
def condition2 : num_H = 50 := sorry
def condition3 : prob_hat_to_sunglasses = 3 / 5 := sorry
def condition4 : num_SH = (3/5) * 50 := sorry

-- Question: Prove that the probability a person wearing sunglasses is also wearing a hat
theorem probability_sunglasses_to_hat :
  (num_SH : ℚ) / num_S = 3 / 8 :=
sorry

end probability_sunglasses_to_hat_l163_163685


namespace proof_expr1_l163_163305

noncomputable def expr1 : ℝ :=
  (Real.sin (65 * Real.pi / 180) + Real.sin (15 * Real.pi / 180) * Real.sin (10 * Real.pi / 180)) / 
  (Real.sin (25 * Real.pi / 180) - Real.cos (15 * Real.pi / 180) * Real.cos (80 * Real.pi / 180))

theorem proof_expr1 : expr1 = 2 + Real.sqrt 3 :=
by sorry

end proof_expr1_l163_163305


namespace complex_frac_eq_l163_163064

theorem complex_frac_eq (a b : ℝ) (i : ℂ) (h : i^2 = -1)
  (h1 : (1 - i) / (1 + i) = a + b * i) : a - b = 1 :=
by
  sorry

end complex_frac_eq_l163_163064


namespace total_tiles_needed_l163_163170

-- Definitions of the given conditions
def blue_tiles : Nat := 48
def red_tiles : Nat := 32
def additional_tiles_needed : Nat := 20

-- Statement to prove the total number of tiles needed to complete the pool
theorem total_tiles_needed : blue_tiles + red_tiles + additional_tiles_needed = 100 := by
  sorry

end total_tiles_needed_l163_163170


namespace find_initial_speed_l163_163384

-- Definitions for the conditions
def total_distance : ℕ := 800
def time_at_initial_speed : ℕ := 6
def time_at_60_mph : ℕ := 4
def time_at_40_mph : ℕ := 2
def speed_at_60_mph : ℕ := 60
def speed_at_40_mph : ℕ := 40

-- Setting up the equation: total distance covered
def distance_covered (v : ℕ) : ℕ :=
  time_at_initial_speed * v + time_at_60_mph * speed_at_60_mph + time_at_40_mph * speed_at_40_mph

-- Proof problem statement
theorem find_initial_speed : ∃ v : ℕ, distance_covered v = total_distance ∧ v = 80 := by
  existsi 80
  simp [distance_covered, total_distance, time_at_initial_speed, speed_at_60_mph, time_at_40_mph]
  norm_num
  sorry

end find_initial_speed_l163_163384


namespace topsoil_cost_l163_163434

theorem topsoil_cost
  (cost_per_cubic_foot : ℕ)
  (volume_cubic_yards : ℕ)
  (conversion_factor : ℕ)
  (volume_cubic_feet : ℕ := volume_cubic_yards * conversion_factor)
  (total_cost : ℕ := volume_cubic_feet * cost_per_cubic_foot)
  (cost_per_cubic_foot_def : cost_per_cubic_foot = 8)
  (volume_cubic_yards_def : volume_cubic_yards = 8)
  (conversion_factor_def : conversion_factor = 27) :
  total_cost = 1728 := by
  sorry

end topsoil_cost_l163_163434


namespace circle_cartesian_eq_line_intersect_circle_l163_163973

noncomputable def circle_eq := ∀ (ρ θ : ℝ), ρ = 2 * Real.cos θ → (ρ * Real.cos θ - 1) ^ 2 + (ρ * Real.sin θ) ^ 2 = 1

noncomputable def param_eq (t : ℝ) := (x y : ℝ) → x = 1/2 + (Real.sqrt 3)/2 * t ∧ y = 1/2 + 1/2 * t

theorem circle_cartesian_eq : (x - 1) ^ 2 + y ^ 2 = 1 :=
by sorry

theorem line_intersect_circle (x y : ℝ) (t : ℝ) (ρ θ : ℝ) 
  (h1 : ∀ (ρ θ : ℝ), ρ = 2 * Real.cos θ → (ρ * Real.cos θ - 1) ^ 2 + (ρ * Real.sin θ) ^ 2 = 1)
  (h2 : (x y : ℝ) → x = 1/2 + (Real.sqrt 3)/2 * t ∧ y = 1/2 + 1/2 * t) :
  (x - 1) ^ 2 + y ^ 2 = 1 ∧ | (t1 t2 : ℝ) → t1 * t2 = -1/2 | = 1/2 :=
by sorry

end circle_cartesian_eq_line_intersect_circle_l163_163973


namespace sequence_length_l163_163051

theorem sequence_length :
  ∃ n : ℕ, ∀ (a_1 : ℤ) (d : ℤ) (a_n : ℤ), a_1 = -6 → d = 4 → a_n = 50 → a_n = a_1 + (n - 1) * d ∧ n = 15 :=
by
  sorry

end sequence_length_l163_163051


namespace minimum_inequality_l163_163696

theorem minimum_inequality 
  (x_1 x_2 x_3 x_4 : ℝ) 
  (h1 : x_1 > 0) 
  (h2 : x_2 > 0) 
  (h3 : x_3 > 0) 
  (h4 : x_4 > 0) 
  (h_sum : x_1^2 + x_2^2 + x_3^2 + x_4^2 = 4) :
  (x_1 / (1 - x_1^2) + x_2 / (1 - x_2^2) + x_3 / (1 - x_3^2) + x_4 / (1 - x_4^2)) ≥ 6 * Real.sqrt 3 :=
by
  sorry

end minimum_inequality_l163_163696


namespace log_base_5_of_15625_l163_163790

theorem log_base_5_of_15625 : log 5 15625 = 6 :=
by
  -- Given that 15625 is 5^6, we can directly provide this as a known fact.
  have h : 5 ^ 6 = 15625 := by norm_num
  rw [← log_eq_log_of_exp h] -- Use definition of logarithm
  norm_num
  exact sorry

end log_base_5_of_15625_l163_163790


namespace trainer_voice_radius_l163_163636

noncomputable def area_of_heard_voice (r : ℝ) : ℝ := (1/4) * Real.pi * r^2

theorem trainer_voice_radius :
  ∃ r : ℝ, abs (r - 140) < 1 ∧ area_of_heard_voice r = 15393.804002589986 :=
by
  sorry

end trainer_voice_radius_l163_163636


namespace rectangle_perimeter_l163_163179

def relatively_prime (a b : ℕ) : Prop := Nat.gcd a b = 1

theorem rectangle_perimeter
  (a1 a2 a3 a4 a5 a6 a7 a8 a9 l w : ℕ)
  (h1 : a1 + a2 + a3 = a9)
  (h2 : a1 + a2 = a3)
  (h3 : a1 + a3 = a4)
  (h4 : a3 + a4 = a5)
  (h5 : a4 + a5 = a6)
  (h6 : a2 + a3 + a5 = a7)
  (h7 : a2 + a7 = a8)
  (h8 : a1 + a4 + a6 = a9)
  (h9 : a6 + a9 = a7 + a8)
  (h_rel_prime : relatively_prime l w)
  (h_dimensions : l = 61)
  (h_dimensions_w : w = 69) :
  2 * l + 2 * w = 260 := by
  sorry

end rectangle_perimeter_l163_163179


namespace sqrt_factorial_product_eq_24_l163_163571

theorem sqrt_factorial_product_eq_24 : (sqrt (fact 4 * fact 4) = 24) :=
by sorry

end sqrt_factorial_product_eq_24_l163_163571


namespace positive_when_x_negative_l163_163223

theorem positive_when_x_negative (x : ℝ) (h : x < 0) : (x / |x|)^2 > 0 := by
  sorry

end positive_when_x_negative_l163_163223


namespace log5_of_15625_l163_163791

-- Define the logarithm function in base 5
def log_base_5 (n : ℕ) : ℕ := sorry

-- State the theorem with the given condition and conclude the desired result
theorem log5_of_15625 : log_base_5 15625 = 6 :=
by sorry

end log5_of_15625_l163_163791


namespace sqrt_factorial_product_l163_163575

theorem sqrt_factorial_product:
  sqrt ((fact 4) * (fact 4)) = 24 :=
by sorry

end sqrt_factorial_product_l163_163575


namespace second_place_team_wins_l163_163508
open Nat

def points (wins ties : Nat) : Nat :=
  2 * wins + ties

def avg_points (p1 p2 p3 : Nat) : Nat :=
  (p1 + p2 + p3) / 3

def first_place_points := points 12 4
def elsa_team_points := points 8 10

def second_place_wins (w : Nat) : Nat :=
  w

def second_place_points (w : Nat) : Nat :=
  points w 1

theorem second_place_team_wins :
  ∃ (W : Nat), avg_points first_place_points (second_place_points W) elsa_team_points = 27 ∧ W = 13 :=
by sorry

end second_place_team_wins_l163_163508


namespace special_number_is_square_l163_163536

-- Define the special number format
def special_number (n : ℕ) : ℕ :=
  3 * (10^n - 1)/9 + 4

theorem special_number_is_square (n : ℕ) :
  ∃ k : ℕ, k * k = special_number n := by
  sorry

end special_number_is_square_l163_163536


namespace find_f_ln2_l163_163372

noncomputable def f : ℝ → ℝ := sorry

axiom fx_monotonic : Monotone f
axiom fx_condition : ∀ x : ℝ, f (f x + Real.exp x) = 1 - Real.exp 1

theorem find_f_ln2 : f (Real.log 2) = -1 := 
sorry

end find_f_ln2_l163_163372


namespace fraction_addition_l163_163014

theorem fraction_addition (d : ℤ) :
  (6 + 4 * d) / 9 + 3 / 2 = (39 + 8 * d) / 18 := sorry

end fraction_addition_l163_163014


namespace count_yellow_highlighters_l163_163997

-- Definitions of the conditions
def pink_highlighters : ℕ := 9
def blue_highlighters : ℕ := 5
def total_highlighters : ℕ := 22

-- Definition based on the question
def yellow_highlighters : ℕ := total_highlighters - (pink_highlighters + blue_highlighters)

-- The theorem to prove the number of yellow highlighters
theorem count_yellow_highlighters : yellow_highlighters = 8 :=
by
  -- Proof omitted as instructed
  sorry

end count_yellow_highlighters_l163_163997


namespace prime_eq_sol_l163_163811

theorem prime_eq_sol {p q x y z : ℕ} (hp : Nat.Prime p) (hq : Nat.Prime q) :
  (∃ x y z : ℕ, p^(2*x) + q^(2*y) = z^2) ↔ ((p = 2 ∧ q = 3) ∨ (p = 3 ∧ q = 2)) :=
by
  sorry

end prime_eq_sol_l163_163811


namespace negation_is_false_l163_163889

-- Definitions corresponding to the conditions
def prop (x : ℝ) := x > 0 → x^2 > 0

-- Statement of the proof problem in Lean 4
theorem negation_is_false : ¬(∀ x : ℝ, ¬(x > 0 → x^2 > 0)) = false :=
by {
  sorry
}

end negation_is_false_l163_163889


namespace EF_side_length_l163_163243

def square_side_length (n : ℝ) : Prop := n = 10

def distance_parallel_line (d : ℝ) : Prop := d = 6.5

def area_difference (a : ℝ) : Prop := a = 13.8

theorem EF_side_length :
  ∃ (x : ℝ), square_side_length 10 ∧ distance_parallel_line 6.5 ∧ area_difference 13.8 ∧ x = 5.4 :=
sorry

end EF_side_length_l163_163243


namespace minimum_value_of_x_plus_y_l163_163982

noncomputable def minValueSatisfies (x y : ℝ) : Prop :=
  x > 0 ∧ y > 0 ∧ x + y + x * y = 2 → x + y ≥ 2 * Real.sqrt 3 - 2

theorem minimum_value_of_x_plus_y (x y : ℝ) : minValueSatisfies x y :=
by sorry

end minimum_value_of_x_plus_y_l163_163982


namespace ordered_triples_count_l163_163980

theorem ordered_triples_count :
  ∃ (count : ℕ), count = 4 ∧
  (∃ a b c : ℕ,
    a > 0 ∧ b > 0 ∧ c > 0 ∧
    Nat.lcm a b = 90 ∧
    Nat.lcm a c = 980 ∧
    Nat.lcm b c = 630) :=
by
  sorry

end ordered_triples_count_l163_163980


namespace least_positive_integer_reducible_fraction_l163_163341

theorem least_positive_integer_reducible_fraction :
  ∃ (n : ℕ), n > 0 ∧ (∀ m : ℕ, m > 0 → (∃ d : ℕ, d > 1 ∧ d ∣ (m - 10) ∧ d ∣ (9 * m + 11)) ↔ m ≥ n) ∧ n = 111 :=
by
  sorry

end least_positive_integer_reducible_fraction_l163_163341


namespace sum_of_constants_l163_163895

-- Problem statement
theorem sum_of_constants (a b c : ℕ) 
  (h1 : a * a * b = 75) 
  (h2 : c * c = 128) 
  (h3 : ∀ d e f : ℕ, d * sqrt e / f = sqrt 75 / sqrt 128 → d = a ∧ e = b ∧ f = c) :
  a + b + c = 27 := 
sorry

end sum_of_constants_l163_163895


namespace boy_needs_to_sell_75_oranges_to_make_150c_profit_l163_163023

-- Definitions based on the conditions
def cost_per_orange : ℕ := 12 / 4
def sell_price_per_orange : ℕ := 30 / 6
def profit_per_orange : ℕ := sell_price_per_orange - cost_per_orange

-- Problem declaration
theorem boy_needs_to_sell_75_oranges_to_make_150c_profit : 
  (150 / profit_per_orange) = 75 :=
by
  -- Proof will be added here
  sorry

end boy_needs_to_sell_75_oranges_to_make_150c_profit_l163_163023


namespace area_of_square_with_diagonal_l163_163766

theorem area_of_square_with_diagonal (d : ℝ) (s : ℝ) (hsq : d = s * Real.sqrt 2) (hdiagonal : d = 12 * Real.sqrt 2) : 
  s^2 = 144 :=
by
  -- Proof details would go here.
  sorry

end area_of_square_with_diagonal_l163_163766


namespace total_fish_caught_l163_163862

theorem total_fish_caught (leo_fish : ℕ) (agrey_fish : ℕ) 
  (h₁ : leo_fish = 40) (h₂ : agrey_fish = leo_fish + 20) : 
  leo_fish + agrey_fish = 100 := 
by 
  sorry

end total_fish_caught_l163_163862


namespace solve_equation_l163_163425

theorem solve_equation : ∀ (x : ℝ), (x / 2 - 1 = 3) → x = 8 :=
by
  intro x h
  sorry

end solve_equation_l163_163425


namespace prob_two_fours_l163_163403

-- Define the sample space for a fair die
def die_faces : Finset ℕ := {1, 2, 3, 4, 5, 6}

-- The probability of rolling a 4 on a fair die
def prob_rolling_four : ℚ := 1 / 6

-- Probability of two independent events both resulting in rolling a 4
def prob_both_rolling_four : ℚ := (prob_rolling_four) * (prob_rolling_four)

-- Prove that the probability of rolling two 4s in two independent die rolls is 1/36
theorem prob_two_fours : prob_both_rolling_four = 1 / 36 := by
  sorry

end prob_two_fours_l163_163403


namespace solve_for_x_l163_163721

theorem solve_for_x (x : ℚ) (h : 3 / 4 - 1 / x = 1 / 2) : x = 4 :=
sorry

end solve_for_x_l163_163721


namespace tank_capacity_l163_163147

theorem tank_capacity (T : ℚ) (h1 : 0 ≤ T)
  (h2 : 9 + (3 / 4) * T = (9 / 10) * T) : T = 60 :=
sorry

end tank_capacity_l163_163147


namespace proof_math_problem_l163_163168

-- Define the conditions
structure Conditions where
  person1_start_noon : ℕ -- Person 1 starts from Appleminster at 12:00 PM
  person2_start_2pm : ℕ -- Person 2 starts from Boniham at 2:00 PM
  meet_time : ℕ -- They meet at 4:55 PM
  finish_time_simultaneously : Bool -- They finish their journey simultaneously

-- Define the problem
def math_problem (c : Conditions) : Prop :=
  let arrival_time := 7 * 60 -- 7:00 PM in minutes
  c.person1_start_noon = 0 ∧ -- Noon as 0 minutes (12:00 PM)
  c.person2_start_2pm = 120 ∧ -- 2:00 PM as 120 minutes
  c.meet_time = 295 ∧ -- 4:55 PM as 295 minutes
  c.finish_time_simultaneously = true → arrival_time = 420 -- 7:00 PM in minutes

-- Prove the problem statement, skipping actual proof
theorem proof_math_problem (c : Conditions) : math_problem c :=
  by sorry

end proof_math_problem_l163_163168


namespace sum_of_first_1234_terms_l163_163688

-- Define the sequence
def seq : ℕ → ℕ
| 0 := 1
| (n + 1) := if n % (2 + seq n) == 1 then 1 else 2

-- Define the sum of the first n terms of the sequence
def sum_seq (n : ℕ) : ℕ :=
(nat.rec_on n 0 (λ n ih, ih + seq n))

-- Define the given conditions and the correct answer
theorem sum_of_first_1234_terms : sum_seq 1234 = 2419 := 
by sorry

end sum_of_first_1234_terms_l163_163688


namespace prime_square_sum_eq_square_iff_l163_163807

theorem prime_square_sum_eq_square_iff (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q):
  (∃ x y z : ℕ, p^(2*x) + q^(2*y) = z^2) ↔ (p = 2 ∧ q = 3) ∨ (p = 3 ∧ q = 2) := 
sorry

end prime_square_sum_eq_square_iff_l163_163807


namespace seashells_needed_to_reach_target_l163_163706

-- Definitions based on the conditions
def current_seashells : ℕ := 19
def target_seashells : ℕ := 25

-- Statement to prove
theorem seashells_needed_to_reach_target : target_seashells - current_seashells = 6 :=
by
  sorry

end seashells_needed_to_reach_target_l163_163706


namespace Jessica_biking_speed_l163_163853

theorem Jessica_biking_speed
  (swim_distance swim_speed : ℝ)
  (run_distance run_speed : ℝ)
  (bike_distance total_time : ℝ)
  (h1 : swim_distance = 0.5)
  (h2 : swim_speed = 1)
  (h3 : run_distance = 5)
  (h4 : run_speed = 5)
  (h5 : bike_distance = 20)
  (h6 : total_time = 4) :
  bike_distance / (total_time - (swim_distance / swim_speed + run_distance / run_speed)) = 8 :=
by
  -- Proof omitted
  sorry

end Jessica_biking_speed_l163_163853


namespace smallest_ratio_l163_163151

theorem smallest_ratio (r s : ℤ) (h1 : 3 * r ≥ 2 * s - 3) (h2 : 4 * s ≥ r + 12) : 
  (∃ r s, (r : ℚ) / s = 1 / 2) :=
by 
  sorry

end smallest_ratio_l163_163151


namespace car_return_speed_l163_163163

variable (d : ℕ) (r : ℕ)
variable (H0 : d = 180)
variable (H1 : ∀ t1 : ℕ, t1 = d / 90)
variable (H2 : ∀ t2 : ℕ, t2 = d / r)
variable (H3 : ∀ avg_rate : ℕ, avg_rate = 2 * d / (d / 90 + d / r))
variable (H4 : avg_rate = 60)

theorem car_return_speed : r = 45 :=
by sorry

end car_return_speed_l163_163163


namespace tank_empty_time_l163_163300

theorem tank_empty_time (V : ℝ) (r_inlet r_outlet1 r_outlet2 : ℝ) (I : V = 20 * 12^3)
  (r_inlet_val : r_inlet = 5) (r_outlet1_val : r_outlet1 = 9) 
  (r_outlet2_val : r_outlet2 = 8) : 
  (V / ((r_outlet1 + r_outlet2) - r_inlet) = 2880) :=
by
  sorry

end tank_empty_time_l163_163300


namespace arithmetic_sequence_sum_l163_163817

/-- Given an arithmetic sequence {a_n} and the first term a_1 = -2010, 
and given that the average of the first 2009 terms minus the average of the first 2007 terms equals 2,
prove that the sum of the first 2011 terms S_2011 equals 0. -/
theorem arithmetic_sequence_sum (a : ℕ → ℤ) (S : ℕ → ℤ)
  (h_arith_seq : ∃ d, ∀ n, a n = a 1 + (n - 1) * d)
  (h_Sn : ∀ n, S n = n * a 1 + n * (n - 1) / 2 * d)
  (h_a1 : a 1 = -2010)
  (h_avg_diff : (S 2009) / 2009 - (S 2007) / 2007 = 2) :
  S 2011 = 0 := 
sorry

end arithmetic_sequence_sum_l163_163817


namespace c_finishes_work_in_18_days_l163_163226

theorem c_finishes_work_in_18_days (A B C : ℝ) 
  (h1 : A = 1 / 12) 
  (h2 : B = 1 / 9) 
  (h3 : A + B + C = 1 / 4) : 
  1 / C = 18 := 
    sorry

end c_finishes_work_in_18_days_l163_163226


namespace maximum_grade_economics_l163_163917

theorem maximum_grade_economics :
  (∃ (x y : ℝ), x + y = 4.6 ∧ 2.5 * x ≤ 5 ∧ 1.5 * y ≤ 5 ∧
                 let O_mic := 2.5 * x, O_mac := 1.5 * y in
                 let A := 0.25 * O_mic + 0.75 * O_mac,
                     B := 0.75 * O_mic + 0.25 * O_mac in
                 ⌈min A B⌉ = 4)
:= sorry

end maximum_grade_economics_l163_163917


namespace arithmetic_sequence_a8_l163_163513

theorem arithmetic_sequence_a8 (a_1 : ℕ) (S_5 : ℕ) (h_a1 : a_1 = 1) (h_S5 : S_5 = 35) : 
    ∃ a_8 : ℕ, a_8 = 22 :=
by
  sorry

end arithmetic_sequence_a8_l163_163513


namespace man_l163_163629

-- Defining the conditions as variables in Lean
variables (S : ℕ) (M : ℕ)
-- Given conditions
def son_present_age := S = 25
def man_present_age := M = S + 27

-- Goal: the ratio of the man's age to the son's age in two years is 2:1
theorem man's_age_ratio_in_two_years (h1 : son_present_age S) (h2 : man_present_age S M) :
  (M + 2) / (S + 2) = 2 := sorry

end man_l163_163629


namespace contains_K4_l163_163815

noncomputable def G : SimpleGraph (Fin 8) := sorry

axiom subgraph_contains_K3 (S : Finset (Fin 8)) (hS : S.card = 5) : 
  ∃ (H : SimpleGraph S), H.is_subset G ∧ ∃ (T : Finset S), T.card = 3 ∧ H.is_complete T

theorem contains_K4 : ∃ (T : Finset (Fin 8)), T.card = 4 ∧ G.is_complete T :=
by
  -- Reference the conditions we stated in the axioms
  sorry

end contains_K4_l163_163815


namespace maximize_integral_l163_163181
open Real

noncomputable def integral_to_maximize (a b : ℝ) : ℝ :=
  ∫ x in a..b, exp (cos x) * (380 - x - x^2)

theorem maximize_integral :
  ∀ (a b : ℝ), a ≤ b → integral_to_maximize a b ≤ integral_to_maximize (-20) 19 :=
by
  intros a b h
  sorry

end maximize_integral_l163_163181


namespace probability_diamond_or_ace_l163_163920

theorem probability_diamond_or_ace (total_cards : ℕ) (diamonds : ℕ) (aces : ℕ) (jokers : ℕ)
  (not_diamonds_nor_aces : ℕ) (p_not_diamond_nor_ace : ℚ) (p_both_not_diamond_nor_ace : ℚ) : 
  total_cards = 54 →
  diamonds = 13 →
  aces = 4 →
  jokers = 2 →
  not_diamonds_nor_aces = 38 →
  p_not_diamond_nor_ace = 19 / 27 →
  p_both_not_diamond_nor_ace = (19 / 27) ^ 2 →
  1 - p_both_not_diamond_nor_ace = 368 / 729 :=
by 
  intros
  sorry

end probability_diamond_or_ace_l163_163920


namespace symmetric_circle_equation_l163_163950

theorem symmetric_circle_equation :
  ∀ (a b : ℝ), 
    (∀ (x y : ℝ), (x-2)^2 + (y+1)^2 = 4 → y = x + 1) → 
    (∃ x y : ℝ, (x + 2)^2 + (y - 3)^2 = 4) :=
  by
    sorry

end symmetric_circle_equation_l163_163950


namespace range_of_a_l163_163496

noncomputable def f (x a : ℝ) : ℝ := x^2 - 2*a*x + 3

theorem range_of_a (a : ℝ) :
  (∀ x y ∈ (Icc 2 3), f x a ≤ f y a ∨ f y a ≤ f x a) ↔ a ≤ 2 ∨ 3 ≤ a := by
  sorry

end range_of_a_l163_163496


namespace hyperbola_standard_equation_l163_163818

open Real

noncomputable def distance_from_center_to_focus (a b : ℝ) : ℝ := sqrt (a^2 - b^2)

theorem hyperbola_standard_equation (a b c : ℝ)
  (h1 : a > b) (h2 : b > 0)
  (h3 : b = sqrt 3 * c)
  (h4 : a + c = 3 * sqrt 3) :
  ∃ h : a^2 = 12 ∧ b = 3, y^2 / 12 - x^2 / 9 = 1 :=
sorry

end hyperbola_standard_equation_l163_163818


namespace number_of_periods_l163_163027

-- Definitions based on conditions
def students : ℕ := 32
def time_per_student : ℕ := 5
def period_duration : ℕ := 40

-- Theorem stating the equivalent proof problem
theorem number_of_periods :
  (students * time_per_student) / period_duration = 4 :=
sorry

end number_of_periods_l163_163027


namespace fraction_simplification_l163_163646

theorem fraction_simplification :
  (1/2 * 1/3 * 1/4 * 1/5 + 3/2 * 3/4 * 3/5) / (1/2 * 2/3 * 2/5) = 41/8 :=
by
  sorry

end fraction_simplification_l163_163646


namespace least_of_consecutive_odds_l163_163990

noncomputable def average_of_consecutive_odds (n : ℕ) (start : ℤ) : ℤ :=
start + (2 * (n - 1))

theorem least_of_consecutive_odds
    (n : ℕ)
    (mean : ℤ)
    (h : n = 30 ∧ mean = 526) : 
    average_of_consecutive_odds 1 (mean * 2 - (n - 1)) = 497 :=
by
  sorry

end least_of_consecutive_odds_l163_163990


namespace coefficient_sum_zero_l163_163067

theorem coefficient_sum_zero : 
  let p := (Polynomial.X^2 - Polynomial.X - 6)^3 * (Polynomial.X^2 + Polynomial.X - 6)^3 in
  (p.coeff 1) + (p.coeff 5) + (p.coeff 9) = 0 :=
by {
  let p := (Polynomial.X^2 - Polynomial.X - 6)^3 * (Polynomial.X^2 + Polynomial.X - 6)^3,
  sorry
}

end coefficient_sum_zero_l163_163067


namespace complement_intersection_l163_163977

open Set

variable (U : Set ℕ) (A : Set ℕ) (B : Set ℕ)

theorem complement_intersection (hU : U = {2, 3, 6, 8}) (hA : A = {2, 3}) (hB : B = {2, 6, 8}) :
  ((U \ A) ∩ B) = {6, 8} := 
by
  sorry

end complement_intersection_l163_163977


namespace simplify_expression_l163_163716

theorem simplify_expression (x : ℝ) : 3 * x + 6 * x + 9 * x + 12 * x + 15 * x + 18 = 45 * x + 18 :=
by
  sorry

end simplify_expression_l163_163716


namespace sqrt_factorial_product_l163_163603

def factorial (n : Nat) : Nat :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem sqrt_factorial_product :
  sqrt (factorial 4 * factorial 4) = 24 :=
by
  sorry

end sqrt_factorial_product_l163_163603


namespace average_discount_rate_l163_163455

theorem average_discount_rate :
  ∃ x : ℝ, (7200 * (1 - x)^2 = 3528) ∧ x = 0.3 :=
by
  sorry

end average_discount_rate_l163_163455


namespace fourth_term_is_six_l163_163995

-- Definitions from the problem
variables (a d : ℕ)

-- Condition that the sum of the third and fifth terms is 12
def sum_third_fifth_eq_twelve : Prop := (a + 2 * d) + (a + 4 * d) = 12

-- The fourth term of the arithmetic sequence
def fourth_term : ℕ := a + 3 * d

-- The theorem we need to prove
theorem fourth_term_is_six (h : sum_third_fifth_eq_twelve a d) : fourth_term a d = 6 := by
  sorry

end fourth_term_is_six_l163_163995


namespace number_of_adults_l163_163453

-- Define the constants and conditions of the problem.
def children : ℕ := 52
def total_seats : ℕ := 95
def empty_seats : ℕ := 14

-- Define the number of adults and prove it equals 29 given the conditions.
theorem number_of_adults : total_seats - empty_seats - children = 29 :=
by {
  sorry
}

end number_of_adults_l163_163453


namespace quadratic_rewrite_l163_163940

theorem quadratic_rewrite (d e f : ℤ) (h1 : d^2 = 25) (h2 : 2 * d * e = -40) (h3 : e^2 + f = -75) : d * e = -20 := 
by 
  sorry

end quadratic_rewrite_l163_163940


namespace union_of_A_and_B_l163_163699

def A : Set ℕ := {1, 2, 3, 5, 7}
def B : Set ℕ := {3, 4, 5}

theorem union_of_A_and_B : A ∪ B = {1, 2, 3, 4, 5, 7} :=
by sorry

end union_of_A_and_B_l163_163699


namespace total_fish_caught_l163_163860

theorem total_fish_caught (leo_fish : ℕ) (agrey_fish : ℕ) (h1 : leo_fish = 40) (h2 : agrey_fish = leo_fish + 20) :
  leo_fish + agrey_fish = 100 :=
by
  sorry

end total_fish_caught_l163_163860


namespace length_of_jordans_rectangle_l163_163776

theorem length_of_jordans_rectangle
  (carol_length : ℕ) (carol_width : ℕ) (jordan_width : ℕ) (equal_area : (carol_length * carol_width) = (jordan_width * 2)) :
  (2 = 120 / 60) := by
  sorry

end length_of_jordans_rectangle_l163_163776


namespace proof_problem_l163_163488

theorem proof_problem (a b : ℝ) (h1 : a ≥ 0) (h2 : b ≥ 0) (h3 : a^3 + b^3 = 2 * a * b) : a^2 + b^2 ≤ 1 + a * b := 
sorry

end proof_problem_l163_163488


namespace kevin_feeds_each_toad_3_worms_l163_163248

theorem kevin_feeds_each_toad_3_worms
  (num_toads : ℕ) (minutes_per_worm : ℕ) (hours_to_minutes : ℕ) (total_minutes : ℕ)
  (H1 : num_toads = 8)
  (H2 : minutes_per_worm = 15)
  (H3 : hours_to_minutes = 60)
  (H4 : total_minutes = 6 * hours_to_minutes)
  :
  total_minutes / minutes_per_worm / num_toads = 3 :=
sorry

end kevin_feeds_each_toad_3_worms_l163_163248


namespace parallelogram_angle_B_eq_130_l163_163379

theorem parallelogram_angle_B_eq_130 (A C B D : ℝ) (parallelogram_ABCD : true) 
(angles_sum_A_C : A + C = 100) (A_eq_C : A = C): B = 130 := by
  sorry

end parallelogram_angle_B_eq_130_l163_163379


namespace shadow_area_correct_l163_163167

noncomputable def shadow_area (R : ℝ) : ℝ := 3 * Real.pi * R^2

theorem shadow_area_correct (R r d R' : ℝ)
  (h1 : r = (Real.sqrt 3) * R / 2)
  (h2 : d = (3 * R) / 2)
  (h3 : R' = ((3 * R * r) / d)) :
  shadow_area R = Real.pi * R' ^ 2 :=
by
  sorry

end shadow_area_correct_l163_163167


namespace smallest_integer_to_make_multiple_of_five_l163_163907

theorem smallest_integer_to_make_multiple_of_five : 
  ∃ k: ℕ, 0 < k ∧ (726 + k) % 5 = 0 ∧ k = 4 := 
by
  use 4
  sorry

end smallest_integer_to_make_multiple_of_five_l163_163907


namespace find_f_2017_l163_163200

noncomputable def f : ℝ → ℝ := sorry

axiom f_odd : ∀ x : ℝ, f (-x) = -f x
axiom f_period : ∀ x : ℝ, f (x + 2) = -f x
axiom f_neg1 : f (-1) = -3

theorem find_f_2017 : f 2017 = 3 := 
by
  sorry

end find_f_2017_l163_163200


namespace avg_daily_production_n_l163_163916

theorem avg_daily_production_n (n : ℕ) (h₁ : 50 * n + 110 = 55 * (n + 1)) : n = 11 :=
by
  -- Proof omitted
  sorry

end avg_daily_production_n_l163_163916


namespace coefficient_x3_in_product_l163_163478

-- Definitions for the polynomials
def P(x : ℕ → ℕ) : ℕ → ℤ
| 4 => 3
| 3 => 4
| 2 => -2
| 1 => 8
| 0 => -5
| _ => 0

def Q(x : ℕ → ℕ) : ℕ → ℤ
| 3 => 2
| 2 => -7
| 1 => 5
| 0 => -3
| _ => 0

-- Statement of the problem
theorem coefficient_x3_in_product :
  (P 3 * Q 0 + P 2 * Q 1 + P 1 * Q 2) = -78 :=
by
  sorry

end coefficient_x3_in_product_l163_163478


namespace ant_prob_reach_D_after_6_minutes_l163_163175

noncomputable def probability_ant_at_D_after_6_minutes (start_pos end_pos : ℤ × ℤ) (total_moves : ℕ) : ℚ :=
  if start_pos = (-1, -1) ∧ end_pos = (1, 1) ∧ total_moves = 6 then 1 / 8 else 0

theorem ant_prob_reach_D_after_6_minutes :
  probability_ant_at_D_after_6_minutes (-1, -1) (1, 1) 6 = 1 / 8 :=
by sorry

end ant_prob_reach_D_after_6_minutes_l163_163175


namespace largest_multiple_of_7_smaller_than_neg_55_l163_163297

theorem largest_multiple_of_7_smaller_than_neg_55 : ∃ m : ℤ, m % 7 = 0 ∧ m < -55 ∧ ∀ n : ℤ, n % 7 = 0 → n < -55 → n ≤ m :=
sorry

end largest_multiple_of_7_smaller_than_neg_55_l163_163297


namespace ratio_x_y_z_l163_163780

theorem ratio_x_y_z (x y z : ℝ) (h1 : 0.10 * x = 0.20 * y) (h2 : 0.30 * y = 0.40 * z) :
  ∃ k : ℝ, x = 8 * k ∧ y = 4 * k ∧ z = 3 * k :=
by                         
  sorry

end ratio_x_y_z_l163_163780


namespace min_square_side_length_l163_163092

theorem min_square_side_length (s : ℝ) (h : s^2 ≥ 625) : s ≥ 25 :=
sorry

end min_square_side_length_l163_163092


namespace remainder_of_exp_l163_163783

theorem remainder_of_exp (x : ℝ) :
  (x + 1) ^ 2100 % (x^4 - x^2 + 1) = x^2 := 
sorry

end remainder_of_exp_l163_163783


namespace seq_inequality_l163_163351

variable (a : ℕ → ℝ)
variable (n m : ℕ)

-- Conditions
axiom pos_seq (k : ℕ) : a k ≥ 0
axiom add_condition (i j : ℕ) : a (i + j) ≤ a i + a j

-- Statement to prove
theorem seq_inequality (n m : ℕ) (h : m > 0) (h' : n ≥ m) : 
  a n ≤ m * a 1 + ((n : ℝ) / m - 1) * a m := sorry

end seq_inequality_l163_163351


namespace equal_games_per_month_l163_163747

-- Define the given conditions
def total_games : ℕ := 27
def months : ℕ := 3
def games_per_month := total_games / months

-- Proposition that needs to be proven
theorem equal_games_per_month : games_per_month = 9 := 
by
  sorry

end equal_games_per_month_l163_163747


namespace compute_value_l163_163394

open Nat Real

theorem compute_value (A B : ℝ × ℝ) (hA : A = (15, 10)) (hB : B = (-5, 6)) :
  let C : ℝ × ℝ := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
  ∃ (x y : ℝ), C = (x, y) ∧ 2 * x - 4 * y = -22 := by
  sorry

end compute_value_l163_163394


namespace percentage_of_A_l163_163296

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

end percentage_of_A_l163_163296


namespace enclosed_area_of_curve_l163_163729

theorem enclosed_area_of_curve :
  let side_length := 3
  let octagon_area := 2 * (1 + Real.sqrt 2) * side_length^2
  let arc_length := Real.pi
  let arc_angle := Real.pi / 2
  let arc_radius := arc_length / arc_angle
  let sector_area := (arc_angle / (2 * Real.pi)) * Real.pi * arc_radius^2
  let total_sector_area := 12 * sector_area
  let enclosed_area := octagon_area + total_sector_area + 3 * Real.pi
  enclosed_area = 54 + 38.4 * Real.sqrt 2 + 3 * Real.pi :=
by
  -- We will use sorry to indicate the proof is omitted.
  sorry

end enclosed_area_of_curve_l163_163729


namespace determine_squirrel_color_l163_163900

-- Define the types for Squirrel species and the nuts in hollows
inductive Squirrel
| red
| gray

def tells_truth (s : Squirrel) : Prop :=
  s = Squirrel.red

def lies (s : Squirrel) : Prop :=
  s = Squirrel.gray

-- Statements made by the squirrel in front of the second hollow
def statement1 (s : Squirrel) (no_nuts_in_first : Prop) : Prop :=
  tells_truth s → no_nuts_in_first ∧ (lies s → ¬no_nuts_in_first)

def statement2 (s : Squirrel) (nuts_in_either : Prop) : Prop :=
  tells_truth s → nuts_in_either ∧ (lies s → ¬nuts_in_either)

-- Given a squirrel that says the statements and the information about truth and lies
theorem determine_squirrel_color (s : Squirrel) (no_nuts_in_first : Prop) (nuts_in_either : Prop) :
  (statement1 s no_nuts_in_first) ∧ (statement2 s nuts_in_either) → s = Squirrel.red :=
by
  sorry

end determine_squirrel_color_l163_163900


namespace garden_area_l163_163546

theorem garden_area (w l : ℕ) (h1 : l = 3 * w + 30) (h2 : 2 * (w + l) = 780) : 
  w * l = 27000 := 
by 
  sorry

end garden_area_l163_163546


namespace bathroom_square_footage_l163_163022

theorem bathroom_square_footage
  (tiles_width : ℕ)
  (tiles_length : ℕ)
  (tile_size_inches : ℕ)
  (inches_per_foot : ℕ)
  (h1 : tiles_width = 10)
  (h2 : tiles_length = 20)
  (h3 : tile_size_inches = 6)
  (h4 : inches_per_foot = 12)
: (tiles_length * tile_size_inches / inches_per_foot) * (tiles_width * tile_size_inches / inches_per_foot) = 50 := 
by
  sorry

end bathroom_square_footage_l163_163022


namespace smaller_screen_diagonal_l163_163424

/-- The area of a 20-inch square screen is 38 square inches greater than the area
    of a smaller square screen. Prove that the length of the diagonal of the smaller screen is 18 inches. -/
theorem smaller_screen_diagonal (x : ℝ) (d : ℝ) (A₁ A₂ : ℝ)
  (h₀ : d = x * Real.sqrt 2)
  (h₁ : A₁ = 20 * Real.sqrt 2 * 20 * Real.sqrt 2)
  (h₂ : A₂ = x * x)
  (h₃ : A₁ = A₂ + 38) :
  d = 18 :=
by
  sorry

end smaller_screen_diagonal_l163_163424


namespace min_max_value_l163_163963

theorem min_max_value
  (x₁ x₂ x₃ x₄ x₅ : ℝ)
  (h₁ : 0 ≤ x₁) (h₂ : 0 ≤ x₂) (h₃ : 0 ≤ x₃) (h₄ : 0 ≤ x₄) (h₅ : 0 ≤ x₅)
  (h_sum : x₁ + x₂ + x₃ + x₄ + x₅ = 1) :
  (min (max (x₁ + x₂) (max (x₂ + x₃) (max (x₃ + x₄) (x₄ + x₅)))) = 1 / 3) :=
sorry

end min_max_value_l163_163963


namespace intersect_complement_A_B_eq_l163_163976

open Set

variable (U : Set ℝ)
variable (A : Set ℝ)
variable (B : Set ℝ)

noncomputable def complement_A : Set ℝ := U \ A
noncomputable def intersection_complement_A_B : Set ℝ := complement_A U A ∩ B

theorem intersect_complement_A_B_eq : 
  U = univ ∧ A = {x : ℝ | x + 1 < 0} ∧ B = {x : ℝ | x - 3 < 0} →
  intersection_complement_A_B U A B = Icc (-1 : ℝ) 3 :=
by
  intro h
  sorry

end intersect_complement_A_B_eq_l163_163976


namespace number_reduced_by_10_eq_0_09_three_point_two_four_increased_to_three_two_four_zero_l163_163031

theorem number_reduced_by_10_eq_0_09 : ∃ (x : ℝ), x / 10 = 0.09 ∧ x = 0.9 :=
sorry

theorem three_point_two_four_increased_to_three_two_four_zero : ∃ (y : ℝ), 3.24 * y = 3240 ∧ y = 1000 :=
sorry

end number_reduced_by_10_eq_0_09_three_point_two_four_increased_to_three_two_four_zero_l163_163031


namespace mike_total_cards_l163_163529

-- Given conditions
def mike_original_cards : ℕ := 87
def sam_given_cards : ℕ := 13

-- Question equivalence in Lean: Prove that Mike has 100 baseball cards now
theorem mike_total_cards : mike_original_cards + sam_given_cards = 100 :=
by 
  sorry

end mike_total_cards_l163_163529


namespace bounded_sequence_is_constant_two_l163_163793

def is_bounded (l : ℕ → ℕ) := ∃ (M : ℕ), ∀ (n : ℕ), l n ≤ M

def satisfies_condition (a : ℕ → ℕ) : Prop :=
∀ n ≥ 3, a n = (a n.pred + a (n.pred.pred)) / (Nat.gcd (a n.pred) (a (n.pred.pred)))

theorem bounded_sequence_is_constant_two (a : ℕ → ℕ) 
  (h1 : is_bounded a) 
  (h2 : satisfies_condition a) : 
  ∀ n : ℕ, a n = 2 :=
sorry

end bounded_sequence_is_constant_two_l163_163793


namespace period_of_f_cos_theta_l163_163702

open Real

noncomputable def alpha (x : ℝ) : ℝ × ℝ :=
  (sqrt 3 * sin (2 * x), cos x + sin x)

noncomputable def beta (x : ℝ) : ℝ × ℝ :=
  (1, cos x - sin x)

noncomputable def f (x : ℝ) : ℝ :=
  let (α1, α2) := alpha x
  let (β1, β2) := beta x
  α1 * β1 + α2 * β2

theorem period_of_f :
  (∀ x : ℝ, f (x + π) = f x) ∧ (∀ T : ℝ, (T > 0 ∧ ∀ x : ℝ, f (x + T) = f x) → T = π) :=
sorry

theorem cos_theta :
  ∀ θ : ℝ, 0 < θ ∧ θ < π / 2 ∧ f θ = 1 → cos (θ - π / 6) = sqrt 3 / 2 :=
sorry

end period_of_f_cos_theta_l163_163702


namespace value_of_x_minus_y_l163_163216

theorem value_of_x_minus_y (x y : ℝ) 
    (h1 : 3015 * x + 3020 * y = 3025) 
    (h2 : 3018 * x + 3024 * y = 3030) :
    x - y = 11.1167 :=
sorry

end value_of_x_minus_y_l163_163216


namespace difference_of_squares_divisible_by_18_l163_163880

-- Definitions of odd integers.
def is_odd (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k + 1

-- The main theorem stating the equivalence.
theorem difference_of_squares_divisible_by_18 (a b : ℤ) 
  (ha : is_odd a) (hb : is_odd b) : 
  ((3 * a + 2) ^ 2 - (3 * b + 2) ^ 2) % 18 = 0 := 
by
  sorry

end difference_of_squares_divisible_by_18_l163_163880


namespace find_triples_l163_163949

def is_solution (a b c : ℕ) : Prop :=
  a ≤ b ∧ b ≤ c ∧ a^2 + b^2 + c^2 = 2005

theorem find_triples :
  ∃ a b c : ℕ, a > 0 ∧ b > 0 ∧ c > 0 ∧ is_solution a b c :=
sorry

end find_triples_l163_163949


namespace unknown_rate_of_two_towels_l163_163913

theorem unknown_rate_of_two_towels :
  let x := 325
  let known_cost := (3 * 100) + (5 * 150)
  let total_average_price := 170
  let number_of_towels := 10
  known_cost + (2 * x) = total_average_price * number_of_towels :=
by
  let x := 325
  let known_cost := (3 * 100) + (5 * 150)
  let total_average_price := 170
  let number_of_towels := 10
  show known_cost + (2 * x) = total_average_price * number_of_towels
  sorry

end unknown_rate_of_two_towels_l163_163913


namespace prime_eq_sol_l163_163809

theorem prime_eq_sol {p q x y z : ℕ} (hp : Nat.Prime p) (hq : Nat.Prime q) :
  (∃ x y z : ℕ, p^(2*x) + q^(2*y) = z^2) ↔ ((p = 2 ∧ q = 3) ∨ (p = 3 ∧ q = 2)) :=
by
  sorry

end prime_eq_sol_l163_163809


namespace find_a_of_pure_imaginary_z_l163_163364

-- Definition of a pure imaginary number
def pure_imaginary (z : ℂ) : Prop := z.re = 0

-- Main theorem statement
theorem find_a_of_pure_imaginary_z (a : ℝ) (z : ℂ) (hz : pure_imaginary z) (h : (2 - I) * z = 4 + 2 * a * I) : a = 4 :=
by
  sorry

end find_a_of_pure_imaginary_z_l163_163364


namespace marks_in_physics_l163_163332

def marks_in_english : ℝ := 74
def marks_in_mathematics : ℝ := 65
def marks_in_chemistry : ℝ := 67
def marks_in_biology : ℝ := 90
def average_marks : ℝ := 75.6
def number_of_subjects : ℕ := 5

-- We need to show that David's marks in Physics are 82.
theorem marks_in_physics : ∃ (P : ℝ), P = 82 ∧ 
  ((marks_in_english + marks_in_mathematics + P + marks_in_chemistry + marks_in_biology) / number_of_subjects = average_marks) :=
by sorry

end marks_in_physics_l163_163332


namespace number_of_rowers_l163_163150

theorem number_of_rowers (total_coaches : ℕ) (votes_per_coach : ℕ) (votes_per_rower : ℕ) 
  (htotal_coaches : total_coaches = 36) (hvotes_per_coach : votes_per_coach = 5) 
  (hvotes_per_rower : votes_per_rower = 3) : 
  (total_coaches * votes_per_coach) / votes_per_rower = 60 :=
by 
  sorry

end number_of_rowers_l163_163150


namespace solve_triplet_l163_163050

theorem solve_triplet (x y z : ℕ) (h : 2^x * 3^y + 1 = 7^z) :
  (x = 1 ∧ y = 1 ∧ z = 1) ∨ (x = 4 ∧ y = 1 ∧ z = 2) :=
 by sorry

end solve_triplet_l163_163050


namespace triangles_intersection_area_is_zero_l163_163231

-- Define the vertices of the two triangles
def vertex_triangle_1 : Fin 3 → (ℝ × ℝ)
| ⟨0, _⟩ => (0, 2)
| ⟨1, _⟩ => (2, 1)
| ⟨2, _⟩ => (0, 0)

def vertex_triangle_2 : Fin 3 → (ℝ × ℝ)
| ⟨0, _⟩ => (2, 2)
| ⟨1, _⟩ => (0, 1)
| ⟨2, _⟩ => (2, 0)

-- The area of the intersection of the two triangles
def area_intersection (v1 v2 : Fin 3 → (ℝ × ℝ)) : ℝ :=
  0

-- The theorem to prove
theorem triangles_intersection_area_is_zero :
  area_intersection vertex_triangle_1 vertex_triangle_2 = 0 :=
by
  -- Proof is omitted here.
  sorry

end triangles_intersection_area_is_zero_l163_163231


namespace line_passes_point_l163_163281

theorem line_passes_point (k : ℝ) :
  ((1 + 4 * k) * 2 - (2 - 3 * k) * 2 + (2 - 14 * k)) = 0 :=
by
  sorry

end line_passes_point_l163_163281


namespace sqrt_factorial_mul_factorial_l163_163587

theorem sqrt_factorial_mul_factorial (n : ℕ) : 
  n = 4 → sqrt ((nat.factorial n) * (nat.factorial n)) = nat.factorial n :=
by
  intro h
  rw [h, nat.factorial, mul_self_sqrt (nat.factorial_nonneg 4)]

-- Note: While the final "mul_self_sqrt (nat.factorial_nonneg 4)" line is a sketch of the idea,
-- the proof is not complete as requested.

end sqrt_factorial_mul_factorial_l163_163587


namespace problem1_problem2_l163_163759

-- Problem (Ⅰ)
theorem problem1 (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b = 1) :
  (1 + 1 / a) * (1 + 1 / b) ≥ 9 :=
sorry

-- Problem (Ⅱ)
theorem problem2 (a : ℝ) (h1 : ∀ (x : ℝ), x ≥ 1 ↔ |x + 3| - |x - a| ≥ 2) :
  a = 2 :=
sorry

end problem1_problem2_l163_163759


namespace david_distance_to_airport_l163_163474

theorem david_distance_to_airport (t : ℝ) (d : ℝ) :
  (35 * (t + 1) = d) ∧ (d - 35 = 50 * (t - 1.5)) → d = 210 :=
by
  sorry

end david_distance_to_airport_l163_163474


namespace train_stop_time_l163_163450

theorem train_stop_time
  (D : ℝ)
  (h1 : D > 0)
  (T_no_stop : ℝ := D / 300)
  (T_with_stop : ℝ := D / 200)
  (T_stop : ℝ := T_with_stop - T_no_stop):
  T_stop = 6 / 60 := by
    sorry

end train_stop_time_l163_163450


namespace two_digit_numbers_reverse_square_condition_l163_163637

theorem two_digit_numbers_reverse_square_condition :
  ∀ (a b : ℕ), 0 ≤ a ∧ a < 10 ∧ 0 ≤ b ∧ b < 10 →
  (∃ n : ℕ, 10 * a + b + 10 * b + a = n^2) ↔ 
  (10 * a + b = 29 ∨ 10 * a + b = 38 ∨ 10 * a + b = 47 ∨ 10 * a + b = 56 ∨ 
   10 * a + b = 65 ∨ 10 * a + b = 74 ∨ 10 * a + b = 83 ∨ 10 * a + b = 92) :=
by {
  sorry
}

end two_digit_numbers_reverse_square_condition_l163_163637


namespace solve_inequality_l163_163723

theorem solve_inequality (x : ℝ) (h₀ : x ≠ 4) : (x^2 - 16) / (x - 4) ≤ 0 ↔ x ∈ set.Iic (-4) :=
by
  sorry

end solve_inequality_l163_163723


namespace increasing_interval_when_a_neg_increasing_and_decreasing_intervals_when_a_pos_l163_163452

noncomputable def f (a x : ℝ) : ℝ := x - a / x

theorem increasing_interval_when_a_neg {a : ℝ} (h : a < 0) :
  ∀ x : ℝ, x > 0 → f a x > 0 :=
sorry

theorem increasing_and_decreasing_intervals_when_a_pos {a : ℝ} (h : a > 0) :
  (∀ x : ℝ, 0 < x → x < Real.sqrt a → f a x < 0) ∧
  (∀ x : ℝ, x > Real.sqrt a → f a x > 0) :=
sorry

end increasing_interval_when_a_neg_increasing_and_decreasing_intervals_when_a_pos_l163_163452


namespace P2_3_eq_2_3_P1_n_eq_1_n_P2_recurrence_P2_n_eq_2_n_l163_163927

-- Define the problem conditions and questions
def P_1 (n : ℕ) : ℚ := sorry
def P_2 (n : ℕ) : ℚ := sorry

-- Part (a)
theorem P2_3_eq_2_3 : P_2 3 = 2 / 3 := sorry

-- Part (b)
theorem P1_n_eq_1_n (n : ℕ) (h : n ≥ 1): P_1 n = 1 / n := sorry

-- Part (c)
theorem P2_recurrence (n : ℕ) (h : n ≥ 2) : 
  P_2 n = (2 / n) * P_1 (n-1) + ((n-2) / n) * P_2 (n-1) := sorry

-- Part (d)
theorem P2_n_eq_2_n (n : ℕ) (h : n ≥ 1): P_2 n = 2 / n := sorry

end P2_3_eq_2_3_P1_n_eq_1_n_P2_recurrence_P2_n_eq_2_n_l163_163927


namespace project_presentation_periods_l163_163024

def students : ℕ := 32
def period_length : ℕ := 40
def presentation_time_per_student : ℕ := 5

theorem project_presentation_periods : 
  (students * presentation_time_per_student) / period_length = 4 := by
  sorry

end project_presentation_periods_l163_163024


namespace largest_circle_center_is_A_l163_163284

-- Define the given lengths of the pentagon's sides
def AB : ℝ := 16
def BC : ℝ := 14
def CD : ℝ := 17
def DE : ℝ := 13
def AE : ℝ := 14

-- Define the radii of the circles centered at points A, B, C, D, E
variables (R_A R_B R_C R_D R_E : ℝ)

-- Conditions based on the problem statement
def radius_conditions : Prop :=
  R_A + R_B = AB ∧
  R_B + R_C = BC ∧
  R_C + R_D = CD ∧
  R_D + R_E = DE ∧
  R_E + R_A = AE

-- The main theorem to prove
theorem largest_circle_center_is_A (h : radius_conditions R_A R_B R_C R_D R_E) :
  10 ≥ R_A ∧ R_A ≥ R_B ∧ R_A ≥ R_C ∧ R_A ≥ R_D ∧ R_A ≥ R_E :=
by sorry

end largest_circle_center_is_A_l163_163284


namespace Polly_tweets_l163_163073

theorem Polly_tweets :
  let HappyTweets := 18 * 50
  let HungryTweets := 4 * 35
  let WatchingReflectionTweets := 45 * 30
  let SadTweets := 6 * 20
  let PlayingWithToysTweets := 25 * 75
  HappyTweets + HungryTweets + WatchingReflectionTweets + SadTweets + PlayingWithToysTweets = 4385 :=
by
  sorry

end Polly_tweets_l163_163073


namespace simplify_expression_l163_163405

theorem simplify_expression (x y : ℝ) : 
  (x - y) * (x + y) + (x - y) ^ 2 = 2 * x ^ 2 - 2 * x * y :=
sorry

end simplify_expression_l163_163405


namespace trajectory_of_center_C_line_AB_passes_through_fixed_point_l163_163350

-- Given conditions
def circle_E (x y : ℝ) : Prop := x^2 + (y - 1)^2 = 1/4
def line_tangent (y : ℝ) : Prop := y = 1/2

-- Trajectory of the moving circle's center
def trajectory (x y : ℝ) : Prop := x^2 = 4 * y

-- Given a point P
def point_P (m : ℝ) : ℝ × ℝ := (m, -4)

-- Prove (1): The trajectory Γ of the center C of the moving circle is x^2 = 4y
theorem trajectory_of_center_C :
  ∀ (x y : ℝ), (circle_E x y → line_tangent y → trajectory x y) := 
begin
  sorry,
end

-- Prove (2): If two tangents are drawn from P(m, -4) to the curve Γ, 
-- the line AB always passes through the fixed point (0, 4)
theorem line_AB_passes_through_fixed_point :
  ∀ (m x1 y1 x2 y2 : ℝ),
  (trajectory x1 y1) → (trajectory x2 y2) →
  tangent_line (point_P m) (x1, y1) ∧ tangent_line (point_P m) (x2, y2) →
  passes_through (0, 4) (x1, y1) (x2, y2) :=
begin
  sorry,
end


end trajectory_of_center_C_line_AB_passes_through_fixed_point_l163_163350


namespace find_d_l163_163185

noncomputable def equilateral_triangle_side : ℝ := 800

-- Assume points A, B, and C form an equilateral triangle
def ABC_is_equilateral (A B C : Point) : Prop :=
  dist A B = equilateral_triangle_side ∧
  dist B C = equilateral_triangle_side ∧
  dist C A = equilateral_triangle_side

-- Assume Points P and Q such that the given conditions hold
def conditions (A B C P Q O : Point) : Prop :=
  dist P A = dist P B ∧ dist P B = dist P C ∧
  dist Q A = dist Q B ∧ dist Q B = dist Q C ∧
  dist O A = dist O B ∧ dist O B = dist O C ∧
  dist O C = dist O P ∧ dist O P = dist O Q ∧
  ∃ θ : ℝ, θ = 150 ∧ plane_dihedral_angle P A B = θ

-- The main theorem to prove: Given the above conditions, distance d is 800
theorem find_d (A B C P Q O : Point) :
  ABC_is_equilateral A B C → conditions A B C P Q O → dist O A = 800 :=
by
  -- Proof here, using the provided conditions
  sorry

end find_d_l163_163185


namespace jake_third_test_marks_l163_163410

theorem jake_third_test_marks 
  (avg_marks : ℕ)
  (marks_test1 : ℕ)
  (marks_test2 : ℕ)
  (marks_test3 : ℕ)
  (marks_test4 : ℕ)
  (h_avg : avg_marks = 75)
  (h_test1 : marks_test1 = 80)
  (h_test2 : marks_test2 = marks_test1 + 10)
  (h_test3_eq_test4 : marks_test3 = marks_test4)
  (h_total : avg_marks * 4 = marks_test1 + marks_test2 + marks_test3 + marks_test4) : 
  marks_test3 = 65 :=
sorry

end jake_third_test_marks_l163_163410


namespace rectangles_with_trapezoid_area_l163_163848

-- Define the necessary conditions
def small_square_area : ℝ := 1
def total_squares : ℕ := 12
def rows : ℕ := 4
def columns : ℕ := 3
def trapezoid_area : ℝ := 3

-- Statement of the proof problem
theorem rectangles_with_trapezoid_area :
  (∀ rows columns : ℕ, rows * columns = total_squares) →
  (∀ area : ℝ, area = small_square_area) →
  (∀ trapezoid_area : ℝ, trapezoid_area = 3) →
  (rows = 4) →
  (columns = 3) →
  ∃ rectangles : ℕ, rectangles = 10 :=
by
  sorry

end rectangles_with_trapezoid_area_l163_163848


namespace remaining_files_calc_l163_163213

-- Definitions based on given conditions
def music_files : ℕ := 27
def video_files : ℕ := 42
def deleted_files : ℕ := 11

-- Theorem statement to prove the number of remaining files
theorem remaining_files_calc : music_files + video_files - deleted_files = 58 := by
  sorry

end remaining_files_calc_l163_163213


namespace problem1_problem2_l163_163011

theorem problem1 (a b c : ℝ) (h1 : a = 5.42) (h2 : b = 3.75) (h3 : c = 0.58) :
  a - (b - c) = 2.25 :=
by sorry

theorem problem2 (d e f g h : ℝ) (h4 : d = 4 / 5) (h5 : e = 7.7) (h6 : f = 0.8) (h7 : g = 3.3) (h8 : h = 1) :
  d * e + f * g - d = 8 :=
by sorry

end problem1_problem2_l163_163011


namespace complex_number_coordinates_l163_163960

-- Define i as the imaginary unit
def i := Complex.I

-- State the theorem
theorem complex_number_coordinates : (i * (1 - i)).re = 1 ∧ (i * (1 - i)).im = 1 :=
by
  -- Proof would go here
  sorry

end complex_number_coordinates_l163_163960


namespace mandy_reads_books_of_480_pages_l163_163261

def pages_at_age6 : ℕ := 8

def pages_at_age12 (p6 : ℕ) : ℕ := 5 * p6

def pages_at_age20 (p12 : ℕ) : ℕ := 3 * p12

def pages_presently (p20 : ℕ) : ℕ := 4 * p20

theorem mandy_reads_books_of_480_pages :
  let p6 := pages_at_age6,
  let p12 := pages_at_age12 p6,
  let p20 := pages_at_age20 p12,
  let ppresent := pages_presently p20
  in ppresent = 480 :=
by
  sorry

end mandy_reads_books_of_480_pages_l163_163261


namespace roadster_paving_company_cement_usage_l163_163402

theorem roadster_paving_company_cement_usage :
  let L := 10
  let T := 5.1
  L + T = 15.1 :=
by
  -- proof is omitted
  sorry

end roadster_paving_company_cement_usage_l163_163402


namespace least_add_to_divisible_least_subtract_to_divisible_l163_163752

theorem least_add_to_divisible (n : ℤ) (d : ℤ) (r : ℤ) (a : ℤ) : 
  n = 1100 → d = 37 → r = n % d → a = d - r → (n + a) % d = 0 :=
by sorry

theorem least_subtract_to_divisible (n : ℤ) (d : ℤ) (r : ℤ) (s : ℤ) : 
  n = 1100 → d = 37 → r = n % d → s = r → (n - s) % d = 0 :=
by sorry

end least_add_to_divisible_least_subtract_to_divisible_l163_163752


namespace find_k_l163_163847

variable {a : ℕ → ℝ}

def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop := 
  ∀ n, a (n + 1) = a n + d

def sum_of_sequence (S : ℕ → ℝ) (a : ℕ → ℝ) : Prop :=
  ∀ n, S n = n * (a 1 + a n) / 2

theorem find_k (a : ℕ → ℝ) (S : ℕ → ℝ) (d : ℝ) (k : ℕ)
  (h1 : a 2 = -1)
  (h2 : 2 * a 1 + a 3 = -1)
  (h3 : arithmetic_sequence a d)
  (h4 : sum_of_sequence S a)
  (h5 : S k = -99) :
  k = 11 := 
by
  sorry

end find_k_l163_163847


namespace stream_speed_l163_163741

theorem stream_speed (x : ℝ) (hb : ∀ t, t = 48 / (20 + x) → t = 24 / (20 - x)) : x = 20 / 3 :=
by
  have t := hb (48 / (20 + x)) rfl
  sorry

end stream_speed_l163_163741


namespace solve_xy_l163_163343

theorem solve_xy : ∃ x y : ℝ, (x - y = 10 ∧ x^2 + y^2 = 100) ↔ ((x = 0 ∧ y = -10) ∨ (x = 10 ∧ y = 0)) := 
by {
  sorry
}

end solve_xy_l163_163343


namespace david_bike_distance_l163_163945

noncomputable def david_time_hours : ℝ := 2 + 1 / 3
noncomputable def david_speed_mph : ℝ := 6.998571428571427
noncomputable def david_distance : ℝ := 16.33

theorem david_bike_distance :
  david_speed_mph * david_time_hours = david_distance :=
by
  sorry

end david_bike_distance_l163_163945


namespace sin_tan_condition_l163_163221

theorem sin_tan_condition (x : ℝ) (h : Real.sin x = (Real.sqrt 2) / 2) : ¬((∀ x, Real.sin x = (Real.sqrt 2) / 2 → Real.tan x = 1) ∧ (∀ x, Real.tan x = 1 → Real.sin x = (Real.sqrt 2) / 2)) :=
sorry

end sin_tan_condition_l163_163221


namespace halt_duration_l163_163542

theorem halt_duration (avg_speed : ℝ) (distance : ℝ) (start_time end_time : ℝ) (halt_duration : ℝ) :
  avg_speed = 87 ∧ distance = 348 ∧ start_time = 9 ∧ end_time = 13.75 →
  halt_duration = (end_time - start_time) - (distance / avg_speed) → 
  halt_duration = 0.75 :=
by
  sorry

end halt_duration_l163_163542


namespace helen_gas_usage_l163_163830

/--
  Assume:
  - Helen cuts her lawn from March through October.
  - Helen's lawn mower uses 2 gallons of gas every 4th time she cuts the lawn.
  - In March, April, September, and October, Helen cuts her lawn 2 times per month.
  - In May, June, July, and August, Helen cuts her lawn 4 times per month.
  Prove: The total gallons of gas needed for Helen to cut her lawn from March through October equals 12.
-/

theorem helen_gas_usage :
  ∀ (months1 months2 cuts_per_month1 cuts_per_month2 gas_per_4th_cut : ℕ),
  months1 = 4 →
  months2 = 4 →
  cuts_per_month1 = 2 →
  cuts_per_month2 = 4 →
  gas_per_4th_cut = 2 →
  (months1 * cuts_per_month1 + months2 * cuts_per_month2) / 4 * gas_per_4th_cut = 12 :=
by
  intros months1 months2 cuts_per_month1 cuts_per_month2 gas_per_4th_cut
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4, h5]
  calc
    (4 * 2 + 4 * 4) / 4 * 2 = (8 + 16) / 4 * 2 : by rw [mul_add]
                                    ...             = 24 / 4 * 2 : by rw [add_mul]
                                    ...             = 6 * 2       : by norm_num
                                    ...             = 12          : by norm_num

end helen_gas_usage_l163_163830


namespace fraction_equals_decimal_l163_163052

theorem fraction_equals_decimal : (1 / 4 : ℝ) = 0.25 := 
sorry

end fraction_equals_decimal_l163_163052


namespace ellipse_eccentricity_l163_163205

theorem ellipse_eccentricity (x y : ℝ) (h : x^2 / 25 + y^2 / 9 = 1) : 
  let a := 5
  let b := 3
  let c := 4
  let e := c / a
  e = 4 / 5 :=
by
  sorry

end ellipse_eccentricity_l163_163205


namespace hoseok_more_paper_than_minyoung_l163_163093

theorem hoseok_more_paper_than_minyoung : 
  ∀ (initial : ℕ) (minyoung_bought : ℕ) (hoseok_bought : ℕ), 
  initial = 150 →
  minyoung_bought = 32 →
  hoseok_bought = 49 →
  (initial + hoseok_bought) - (initial + minyoung_bought) = 17 :=
by
  intros initial minyoung_bought hoseok_bought h_initial h_min h_hos
  sorry

end hoseok_more_paper_than_minyoung_l163_163093


namespace unique_solution_only_a_is_2_l163_163839

noncomputable def unique_solution_inequality (a : ℝ) : Prop :=
  ∀ (p : ℝ → ℝ), (∀ x, 0 ≤ p x ∧ p x ≤ 1 ∧ p x = x^2 - a * x + a) → 
  ∃! x, p x = 1

theorem unique_solution_only_a_is_2 (a : ℝ) (h : unique_solution_inequality a) : a = 2 :=
sorry

end unique_solution_only_a_is_2_l163_163839


namespace max_boxes_in_large_box_l163_163036

def max_boxes (l_L w_L h_L : ℕ) (l_S w_S h_S : ℕ) : ℕ :=
  (l_L * w_L * h_L) / (l_S * w_S * h_S)

theorem max_boxes_in_large_box :
  let l_L := 8 * 100 -- converted to cm
  let w_L := 7 * 100 -- converted to cm
  let h_L := 6 * 100 -- converted to cm
  let l_S := 4
  let w_S := 7
  let h_S := 6
  max_boxes l_L w_L h_L l_S w_S h_S = 2000000 :=
by {
  let l_L := 800 -- converted to cm
  let w_L := 700 -- converted to cm
  let h_L := 600 -- converted to cm
  let l_S := 4
  let w_S := 7
  let h_S := 6
  trivial
}

end max_boxes_in_large_box_l163_163036


namespace find_number_l163_163482

-- Define the problem constants
def total : ℝ := 1.794
def part1 : ℝ := 0.123
def part2 : ℝ := 0.321
def target : ℝ := 1.350

-- The equivalent proof problem
theorem find_number (x : ℝ) (h : part1 + part2 + x = total) : x = target := by
  -- Proof is intentionally omitted
  sorry

end find_number_l163_163482


namespace graham_crackers_leftover_l163_163870

-- Definitions for the problem conditions
def initial_boxes_graham := 14
def initial_packets_oreos := 15
def initial_ounces_cream_cheese := 36

def boxes_per_cheesecake := 2
def packets_per_cheesecake := 3
def ounces_per_cheesecake := 4

-- Define the statement that needs to be proved
theorem graham_crackers_leftover :
  initial_boxes_graham - (min (initial_boxes_graham / boxes_per_cheesecake) (min (initial_packets_oreos / packets_per_cheesecake) (initial_ounces_cream_cheese / ounces_per_cheesecake)) * boxes_per_cheesecake) = 4 :=
by sorry

end graham_crackers_leftover_l163_163870


namespace arithmetic_progression_sum_squares_l163_163427

theorem arithmetic_progression_sum_squares (a1 a2 a3 : ℚ)
  (h1 : a2 = (a1 + a3) / 2)
  (h2 : a1 + a2 + a3 = 2)
  (h3 : a1^2 + a2^2 + a3^2 = 14/9) :
  (a1 = 1/3 ∧ a2 = 2/3 ∧ a3 = 1) ∨ (a1 = 1 ∧ a2 = 2/3 ∧ a3 = 1/3) :=
sorry

end arithmetic_progression_sum_squares_l163_163427


namespace increasing_geometric_progression_l163_163734

noncomputable def golden_ratio : ℝ := (Real.sqrt 5 + 1) / 2

theorem increasing_geometric_progression (a : ℝ) (ha : 0 < a)
  (h1 : ∃ b c q : ℝ, b = Int.floor a ∧ c = a - b ∧ a = b + c ∧ c = b * q ∧ a = c * q ∧ 1 < q) : 
  a = golden_ratio :=
sorry

end increasing_geometric_progression_l163_163734


namespace trip_drop_probability_l163_163690

-- Definitions
def P_Trip : ℝ := 0.4
def P_Drop_not : ℝ := 0.9

-- Main theorem
theorem trip_drop_probability : ∀ (P_Trip P_Drop_not : ℝ), P_Trip = 0.4 → P_Drop_not = 0.9 → 1 - P_Drop_not = 0.1 :=
by
  intros P_Trip P_Drop_not h1 h2
  rw [h2]
  norm_num

end trip_drop_probability_l163_163690


namespace garden_dimensions_l163_163171

theorem garden_dimensions (l w : ℕ) (h1 : 2 * l + 2 * w = 60) (h2 : l * w = 221) : 
    (l = 17 ∧ w = 13) ∨ (l = 13 ∧ w = 17) :=
sorry

end garden_dimensions_l163_163171


namespace record_expenditure_l163_163079

theorem record_expenditure (income recording expenditure : ℤ) (h : income = 100 ∧ recording = 100) :
  expenditure = -80 ↔ recording - expenditure = income - 80 :=
by
  sorry

end record_expenditure_l163_163079


namespace diagonal_plane_angle_l163_163003

theorem diagonal_plane_angle
  (α : Real)
  (a : Real)
  (plane_square_angle_with_plane : Real)
  (diagonal_plane_angle : Real) 
  (h1 : plane_square_angle_with_plane = α) :
  diagonal_plane_angle = Real.arcsin (Real.sin α / Real.sqrt 2) :=
sorry

end diagonal_plane_angle_l163_163003


namespace find_k_l163_163497

noncomputable def vec_na (x1 k : ℝ) : ℝ × ℝ := (x1 - k/4, 2 * x1^2)
noncomputable def vec_nb (x2 k : ℝ) : ℝ × ℝ := (x2 - k/4, 2 * x2^2)
noncomputable def dot_product (u v : ℝ × ℝ) : ℝ := u.fst * v.fst + u.snd * v.snd

theorem find_k (k : ℝ) (x1 x2 : ℝ) 
  (h1 : x1 + x2 = k / 2) 
  (h2 : x1 * x2 = -1) 
  (h3 : dot_product (vec_na x1 k) (vec_nb x2 k) = 0) : 
  k = 4 * Real.sqrt 3 ∨ k = -4 * Real.sqrt 3 :=
by
  sorry

end find_k_l163_163497


namespace smallest_value_of_3a_plus_2_l163_163217

theorem smallest_value_of_3a_plus_2 (a : ℝ) (h : 8 * a^3 + 6 * a^2 + 7 * a + 5 = 4) :
  3 * a + 2 = 1 / 2 :=
sorry

end smallest_value_of_3a_plus_2_l163_163217


namespace determine_dress_and_notebooks_l163_163749

structure Girl :=
  (name : String)
  (dress_color : String)
  (notebook_color : String)

def colors := ["red", "yellow", "blue"]

def Sveta : Girl := ⟨"Sveta", "red", "red"⟩
def Ira : Girl := ⟨"Ira", "blue", "yellow"⟩
def Tania : Girl := ⟨"Tania", "yellow", "blue"⟩

theorem determine_dress_and_notebooks :
  (Sveta.dress_color = Sveta.notebook_color) ∧
  (¬ Tania.dress_color = "red") ∧
  (¬ Tania.notebook_color = "red") ∧
  (Ira.notebook_color = "yellow") ∧
  (Sveta ∈ [Sveta, Ira, Tania]) ∧
  (Ira ∈ [Sveta, Ira, Tania]) ∧
  (Tania ∈ [Sveta, Ira, Tania]) →
  ([Sveta, Ira, Tania] = 
   [{name := "Sveta", dress_color := "red", notebook_color := "red"},
    {name := "Ira", dress_color := "blue", notebook_color := "yellow"},
    {name := "Tania", dress_color := "yellow", notebook_color := "blue"}])
:=
by
  intro h
  sorry

end determine_dress_and_notebooks_l163_163749


namespace probability_of_event_is_correct_l163_163399

def is_divisible_by (n m : ℕ) : Prop := m ∣ n

/-- Definition of the set from which a, b, and c are chosen -/
def num_set := {n : ℕ | 1 ≤ n ∧ n ≤ 2010}

/-- Definition of the event that abc + ab + a is divisible by 3 -/
def event (a b c : ℕ) : Prop := is_divisible_by (a * b * c + a * b + a) 3

/-- Definition of the probability of the event happening given the set -/
def probability_event : ℚ := 13 / 27

/-- The main theorem -/
theorem probability_of_event_is_correct : 
  (∑' a b c in num_set, indicator (event a b c)) / 
  (∑' a b c in num_set, 1) = probability_event := sorry


end probability_of_event_is_correct_l163_163399


namespace xy_value_l163_163985

theorem xy_value (x y : ℝ) (h : x * (x + y) = x^2 + 18) : x * y = 18 := 
by
  sorry

end xy_value_l163_163985


namespace parabola_intersection_points_l163_163750

theorem parabola_intersection_points :
  (∃ (x y : ℝ), y = 4 * x ^ 2 + 3 * x - 7 ∧ y = 2 * x ^ 2 - 5)
  ↔ ((-2, 3) = (x, y) ∨ (1/2, -4.5) = (x, y)) :=
by
   -- To be proved (proof omitted)
   sorry

end parabola_intersection_points_l163_163750


namespace range_of_m_l163_163349

-- Condition p: The solution set of the inequality x² + mx + 1 < 0 is an empty set
def p (m : ℝ) : Prop :=
  ∀ x : ℝ, x^2 + m * x + 1 ≥ 0

-- Condition q: The function y = 4x² + 4(m-1)x + 3 has no extreme value
def q (m : ℝ) : Prop :=
  ∀ x : ℝ, 12 * x^2 + 4 * (m - 1) ≥ 0

-- Combined condition: "p or q" is true and "p and q" is false
def combined_condition (m : ℝ) : Prop :=
  (p m ∨ q m) ∧ ¬(p m ∧ q m)

-- The range of values for the real number m
theorem range_of_m (m : ℝ) : combined_condition m → (-2 ≤ m ∧ m < 1) ∨ m > 2 :=
sorry

end range_of_m_l163_163349


namespace arithmetic_series_sum_after_multiplication_l163_163178

theorem arithmetic_series_sum_after_multiplication :
  let s : List ℕ := [110, 111, 112, 113, 114, 115, 116, 117, 118, 119]
  3 * s.sum = 3435 := by
  sorry

end arithmetic_series_sum_after_multiplication_l163_163178


namespace cos_identity_l163_163812

theorem cos_identity (x : ℝ) 
  (h : Real.sin (2 * x + (Real.pi / 6)) = -1 / 3) : 
  Real.cos ((Real.pi / 3) - 2 * x) = -1 / 3 :=
sorry

end cos_identity_l163_163812


namespace rectangle_area_l163_163901

theorem rectangle_area (length : ℝ) (width : ℝ) (area : ℝ) 
  (h1 : length = 24) 
  (h2 : width = 0.875 * length) 
  (h3 : area = length * width) : 
  area = 504 := 
by
  sorry

end rectangle_area_l163_163901


namespace sum_of_all_digits_divisible_by_nine_l163_163096

theorem sum_of_all_digits_divisible_by_nine :
  ∀ (A B C D : ℕ),
  (A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D) →
  (A + B + C) % 9 = 0 →
  (B + C + D) % 9 = 0 →
  A + B + C + D = 18 := by
  sorry

end sum_of_all_digits_divisible_by_nine_l163_163096


namespace find_d_l163_163837

theorem find_d (d : ℝ) (h : 3 * (2 - (π / 2)) = 6 + d * π) : d = -3 / 2 :=
by
  sorry

end find_d_l163_163837


namespace value_of_a_minus_3_l163_163773

variable {α : Type*} [Field α] (f : α → α) (a : α)

-- Conditions
variable (h_invertible : Function.Injective f)
variable (h_fa : f a = 3)
variable (h_f3 : f 3 = 6)

-- Statement to prove
theorem value_of_a_minus_3 : a - 3 = -2 :=
by
  sorry

end value_of_a_minus_3_l163_163773


namespace find_m_l163_163510

noncomputable def m_value (a b c d : Int) (Y : Int) : Int :=
  let l1_1 := a + b
  let l1_2 := b + c
  let l1_3 := c + d
  let l2_1 := l1_1 + l1_2
  let l2_2 := l1_2 + l1_3
  let l3 := l2_1 + l2_2
  if l3 = Y then a else 0

theorem find_m : m_value m 6 (-3) 4 20 = 7 := sorry

end find_m_l163_163510


namespace toll_for_18_wheel_truck_l163_163304

-- Define the total number of wheels, wheels on the front axle, 
-- and wheels on each of the other axles.
def total_wheels : ℕ := 18
def front_axle_wheels : ℕ := 2
def other_axle_wheels : ℕ := 4

-- Define the formula for calculating the toll.
def toll_formula (x : ℕ) : ℝ := 2.50 + 0.50 * (x - 2)

-- Calculate the number of other axles.
def calc_other_axles (wheels_left : ℕ) (wheels_per_axle : ℕ) : ℕ :=
wheels_left / wheels_per_axle

-- Statement to prove the final toll is $4.00.
theorem toll_for_18_wheel_truck : toll_formula (
  1 + calc_other_axles (total_wheels - front_axle_wheels) other_axle_wheels
) = 4.00 :=
by sorry

end toll_for_18_wheel_truck_l163_163304


namespace remainder_of_10_pow_23_minus_7_mod_6_l163_163626

theorem remainder_of_10_pow_23_minus_7_mod_6 : ((10 ^ 23 - 7) % 6) = 3 := by
  sorry

end remainder_of_10_pow_23_minus_7_mod_6_l163_163626


namespace triangle_angle_sum_l163_163382

theorem triangle_angle_sum (x : ℝ) (h1 : 70 + 50 + x = 180) : x = 60 := by
  -- proof goes here
  sorry

end triangle_angle_sum_l163_163382


namespace smallest_possible_n_l163_163183

theorem smallest_possible_n (n : ℕ) :
  ∃ n, 17 * n - 3 ≡ 0 [MOD 11] ∧ n = 6 :=
by
  sorry

end smallest_possible_n_l163_163183


namespace sqrt_factorial_mul_factorial_eq_l163_163593

theorem sqrt_factorial_mul_factorial_eq :
  Real.sqrt (Nat.factorial 4 * Nat.factorial 4) = 24 := by
sorry

end sqrt_factorial_mul_factorial_eq_l163_163593


namespace arithmetic_sequence_common_difference_l163_163235

theorem arithmetic_sequence_common_difference (a d : ℕ) (n : ℕ) :
  a = 5 →
  (a + (n - 1) * d = 50) →
  (n * (a + (a + (n - 1) * d)) / 2 = 275) →
  d = 5 := 
by
  intros ha ha_n hs_n
  sorry

end arithmetic_sequence_common_difference_l163_163235


namespace proof_problem_l163_163139

theorem proof_problem
  (x y z : ℤ)
  (h1 : x = 11 * y + 4)
  (h2 : 2 * x = 3 * y * z + 3)
  (h3 : 13 * y - x = 1) :
  z = 8 := by
  sorry

end proof_problem_l163_163139


namespace binomial_theorem_example_l163_163393

theorem binomial_theorem_example 
  (a_0 a_1 a_2 a_3 a_4 a_5 : ℤ)
  (h1 : (2 - 1)^5 = a_0 + a_1 * 1 + a_2 * 1^2 + a_3 * 1^3 + a_4 * 1^4 + a_5 * 1^5)
  (h2 : (2 - (-1))^5 = a_0 - a_1 + a_2 * (-1)^2 - a_3 * (-1)^3 + a_4 * (-1)^4 - a_5 * (-1)^5)
  (h3 : a_5 = -1) :
  (a_0 + a_2 + a_4 : ℤ) / (a_1 + a_3 : ℤ) = -61 / 60 := 
sorry

end binomial_theorem_example_l163_163393


namespace committee_membership_l163_163438

theorem committee_membership (n : ℕ) (h1 : 2 * n = 6) (h2 : (n - 1 : ℚ) / 5 = 0.4) : n = 3 := 
sorry

end committee_membership_l163_163438


namespace xz_less_than_half_l163_163693

theorem xz_less_than_half (x y z : ℝ) (h1 : x ≥ y) (h2 : y ≥ z) (h3 : xy + yz + zx = 1) : x * z < 1 / 2 :=
  sorry

end xz_less_than_half_l163_163693


namespace determine_n_l163_163981

def binom (n k : ℕ) : ℕ := Nat.choose n k

theorem determine_n (n : ℕ) (h1 : binom n 2 + binom n 1 = 6) : n = 3 := 
by
  sorry

end determine_n_l163_163981


namespace range_of_a_l163_163672

noncomputable def f (x a : ℝ) : ℝ := 2 * Real.log x + x^2 - 2 * a * x

theorem range_of_a (a : ℝ) (h₀ : a > 0) 
  (h₁ h₂ : ∃ x₁ x₂ : ℝ, x₁ < x₂ ∧ f x₁ a - f x₂ a ≥ (3/2) - 2 * Real.log 2) : 
  a ≥ (3/2) * Real.sqrt 2 :=
sorry

end range_of_a_l163_163672


namespace largest_area_polygons_l163_163648

-- Define the area of each polygon
def area_P := 4
def area_Q := 6
def area_R := 3 + 3 * (1 / 2)
def area_S := 6 * (1 / 2)
def area_T := 5 + 2 * (1 / 2)

-- Proof of the polygons with the largest area
theorem largest_area_polygons : (area_Q = 6 ∧ area_T = 6) ∧ area_Q ≥ area_P ∧ area_Q ≥ area_R ∧ area_Q ≥ area_S :=
by
  sorry

end largest_area_polygons_l163_163648


namespace intersection_A_B_l163_163090

open Set

-- Definitions of the sets A and B as described in the problem
def setA : Set ℝ := {x | abs x ≤ 1}
def setB : Set ℝ := {x | x ≤ 0}

-- Lean theorem statement to prove the intersection of A and B
theorem intersection_A_B :
  setA ∩ setB = {x | -1 ≤ x ∧ x ≤ 0} := by
  sorry

end intersection_A_B_l163_163090


namespace initial_integer_value_l163_163929

theorem initial_integer_value (x : ℤ) (h : (x + 2) * (x + 2) = x * x - 2016) : x = -505 := 
sorry

end initial_integer_value_l163_163929


namespace bales_stacked_correct_l163_163902

-- Given conditions
def initial_bales : ℕ := 28
def final_bales : ℕ := 82

-- Define the stacking function
def bales_stacked (initial final : ℕ) : ℕ := final - initial

-- Theorem statement we need to prove
theorem bales_stacked_correct : bales_stacked initial_bales final_bales = 54 := by
  sorry

end bales_stacked_correct_l163_163902


namespace log_base_5_of_15625_eq_6_l163_163788

theorem log_base_5_of_15625_eq_6 : log 5 15625 = 6 := 
by {
  have h1 : 5^6 = 15625 := by sorry,
  sorry
}

end log_base_5_of_15625_eq_6_l163_163788


namespace ratio_of_radii_of_circles_l163_163821

theorem ratio_of_radii_of_circles 
  (a b : ℝ) 
  (h1 : a = 6) 
  (h2 : b = 8) 
  (h3 : ∃ (c : ℝ), c = Real.sqrt (a^2 + b^2)) 
  (h4 : ∃ (r R : ℝ), R = c / 2 ∧ r = 24 / (a + b + c)) : R / r = 5 / 2 :=
by
  sorry

end ratio_of_radii_of_circles_l163_163821


namespace part_a_part_b_l163_163307

-- Problem (a)
theorem part_a :
  ¬ ∃ (f : ℝ → ℝ), (∀ x, f x ≠ 0) ∧ (∀ x, 2 * f (f x) = f x ∧ f x ≥ 0) ∧ Differentiable ℝ f :=
sorry

-- Problem (b)
theorem part_b :
  ¬ ∃ (f : ℝ → ℝ), (∀ x, f x ≠ 0) ∧ (∀ x, -1 ≤ 2 * f (f x) ∧ 2 * f (f x) = f x ∧ f x ≤ 1) ∧ Differentiable ℝ f :=
sorry

end part_a_part_b_l163_163307


namespace smallest_prime_factor_of_2939_l163_163908

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def smallest_prime_factor (n : ℕ) (p : ℕ) : Prop :=
  is_prime p ∧ p ∣ n ∧ ∀ q : ℕ, is_prime q ∧ q ∣ n → p ≤ q

theorem smallest_prime_factor_of_2939 : smallest_prime_factor 2939 13 :=
by
  sorry

end smallest_prime_factor_of_2939_l163_163908


namespace sum_of_coefficients_l163_163784

-- Definition of the polynomial
def P (x : ℝ) : ℝ := 5 * (2 * x ^ 9 - 3 * x ^ 6 + 4) - 4 * (x ^ 6 - 5 * x ^ 3 + 6)

-- Theorem stating the sum of the coefficients is 7
theorem sum_of_coefficients : P 1 = 7 := by
  sorry

end sum_of_coefficients_l163_163784


namespace johns_contribution_correct_l163_163224

noncomputable def average_contribution_before : Real := sorry
noncomputable def total_contributions_by_15 : Real := 15 * average_contribution_before
noncomputable def new_average_contribution : Real := 150
noncomputable def johns_contribution : Real := average_contribution_before * 15 + 1377.3

-- The theorem we want to prove
theorem johns_contribution_correct :
  (new_average_contribution = (total_contributions_by_15 + johns_contribution) / 16) ∧
  (new_average_contribution = 2.2 * average_contribution_before) :=
sorry

end johns_contribution_correct_l163_163224


namespace carlotta_performance_time_l163_163953

theorem carlotta_performance_time :
  ∀ (s p t : ℕ),  -- s for singing, p for practicing, t for tantrums
  (∀ (n : ℕ), p = 3 * n ∧ t = 5 * n) →
  s = 6 →
  (s + p + t) = 54 :=
by 
  intros s p t h1 h2
  rcases h1 1 with ⟨h3, h4⟩
  sorry

end carlotta_performance_time_l163_163953


namespace total_animals_count_l163_163563

theorem total_animals_count (a m : ℕ) (h1 : a = 35) (h2 : a + 7 = m) : a + m = 77 :=
by
  sorry

end total_animals_count_l163_163563


namespace neither_sufficient_nor_necessary_condition_l163_163968

theorem neither_sufficient_nor_necessary_condition (a b : ℝ) :
  ¬ ((a < 0 ∧ b < 0) → (a * b * (a - b) > 0)) ∧
  ¬ ((a * b * (a - b) > 0) → (a < 0 ∧ b < 0)) :=
by
  sorry

end neither_sufficient_nor_necessary_condition_l163_163968


namespace sqrt_factorial_product_l163_163582

/-- Define the factorial of 4 -/
def factorial_four : ℕ := 4!

/-- Define the product of the factorial of 4 with itself -/
def product_of_factorials : ℕ := factorial_four * factorial_four

/-- Prove the value of the square root of product_of_factorials is 24 -/
theorem sqrt_factorial_product : Real.sqrt (product_of_factorials) = 24 := by
  have fact_4_eq_24 : factorial_four = 24 := by norm_num
  rw [product_of_factorials, fact_4_eq_24, Nat.mul_self_eq, Real.sqrt_sq]
  norm_num
  exact Nat.zero_le 24

end sqrt_factorial_product_l163_163582


namespace fraction_sum_l163_163228

variable {a b : ℝ}

theorem fraction_sum (ha : a ≠ 0) (hb : b ≠ 0) (hab : a ≠ b) (h1 : a^2 + a - 2007 = 0) (h2 : b^2 + b - 2007 = 0) :
  (1/a + 1/b) = 1/2007 :=
by
  sorry

end fraction_sum_l163_163228


namespace midpoint_sum_eq_six_l163_163012

theorem midpoint_sum_eq_six :
  let x1 := 6
  let y1 := 12
  let x2 := 0
  let y2 := -6
  let midpoint_x := (x1 + x2) / 2 
  let midpoint_y := (y1 + y2) / 2 
  (midpoint_x + midpoint_y) = 6 :=
by
  let x1 := 6
  let y1 := 12
  let x2 := 0
  let y2 := -6
  let midpoint_x := (x1 + x2) / 2 
  let midpoint_y := (y1 + y2) / 2 
  sorry

end midpoint_sum_eq_six_l163_163012


namespace sqrt_factorial_product_l163_163602

def factorial (n : Nat) : Nat :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem sqrt_factorial_product :
  sqrt (factorial 4 * factorial 4) = 24 :=
by
  sorry

end sqrt_factorial_product_l163_163602


namespace reunion_handshakes_l163_163174

/-- 
Given 15 boys at a reunion:
- 5 are left-handed and will only shake hands with other left-handed boys.
- Each boy shakes hands exactly once with each of the others unless they forget.
- Three boys each forget to shake hands with two others.

Prove that the total number of handshakes is 49. 
-/
theorem reunion_handshakes : 
  let total_boys := 15
  let left_handed := 5
  let forgetful_boys := 3
  let forgotten_handshakes_per_boy := 2

  let total_handshakes := total_boys * (total_boys - 1) / 2
  let left_left_handshakes := left_handed * (left_handed - 1) / 2
  let left_right_handshakes := left_handed * (total_boys - left_handed)
  let distinct_forgotten_handshakes := forgetful_boys * forgotten_handshakes_per_boy / 2

  total_handshakes 
    - left_right_handshakes 
    - distinct_forgotten_handshakes
    - left_left_handshakes
  = 49 := 
sorry

end reunion_handshakes_l163_163174


namespace sum_gcd_lcm_l163_163252

theorem sum_gcd_lcm (A B : ℕ) (hA : A = Nat.gcd 10 (Nat.gcd 15 25)) (hB : B = Nat.lcm 10 (Nat.lcm 15 25)) :
  A + B = 155 :=
by
  sorry

end sum_gcd_lcm_l163_163252


namespace sqrt_factorial_eq_l163_163606

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem sqrt_factorial_eq :
  sqrt (factorial 4 * factorial 4) = 24 :=
by
  have h : factorial 4 = 24 := by
    unfold factorial
    simpa using [factorial, factorial, factorial]
  rw [h, h]
  sorry

end sqrt_factorial_eq_l163_163606


namespace minimum_value_of_expression_l163_163389

noncomputable def min_value_expression (a b c : ℝ) : ℝ :=
  a^2 + b^2 + (a + b)^2 + c^2

theorem minimum_value_of_expression (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h_sum : a + b + c = 3) :
  min_value_expression a b c = 9 :=
  sorry

end minimum_value_of_expression_l163_163389


namespace cos_A_value_find_c_l163_163516

theorem cos_A_value (a b c A B C : ℝ) (h : 2 * a * Real.cos A = c * Real.cos B + b * Real.cos C) : 
  Real.cos A = 1 / 2 := 
  sorry

theorem find_c (B C : ℝ) (A : B + C = Real.pi - A) (h1 : 1 = 1) 
  (h2 : Real.cos (B / 2) * Real.cos (B / 2) + Real.cos (C / 2) * Real.cos (C / 2) = 1 + Real.sqrt (3) / 4) : 
  ∃ (c : ℝ), c = 2 * Real.sqrt (3) / 3 ∨ c = Real.sqrt (3) / 3 := 
  sorry

end cos_A_value_find_c_l163_163516


namespace additional_trams_proof_l163_163562

-- Definitions for the conditions
def initial_tram_count : Nat := 12
def total_distance : Nat := 60
def initial_interval : Nat := total_distance / initial_tram_count
def reduced_interval : Nat := initial_interval - (initial_interval / 5)
def final_tram_count : Nat := total_distance / reduced_interval
def additional_trams_needed : Nat := final_tram_count - initial_tram_count

-- The theorem we need to prove
theorem additional_trams_proof : additional_trams_needed = 3 :=
by
  sorry

end additional_trams_proof_l163_163562


namespace min_value_l163_163354

theorem min_value (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 2 * x + 3 * y + 3 * x * y = 6) : 2 * x + 3 * y ≥ 4 :=
sorry

end min_value_l163_163354


namespace fleas_cannot_reach_final_positions_l163_163748

structure Point2D :=
  (x : ℝ)
  (y : ℝ)

def initial_A : Point2D := ⟨0, 0⟩
def initial_B : Point2D := ⟨1, 0⟩
def initial_C : Point2D := ⟨0, 1⟩

def area (A B C : Point2D) : ℝ :=
  0.5 * abs (A.x * (B.y - C.y) + B.x * (C.y - A.y) + C.x * (A.y - B.y))

def final_A : Point2D := ⟨1, 0⟩
def final_B : Point2D := ⟨-1, 0⟩
def final_C : Point2D := ⟨0, 1⟩

theorem fleas_cannot_reach_final_positions : 
    ¬ (∃ (flea_move_sequence : List (Point2D → Point2D)), 
    area initial_A initial_B initial_C = area final_A final_B final_C) :=
by 
  sorry

end fleas_cannot_reach_final_positions_l163_163748


namespace committee_size_l163_163435

theorem committee_size (n : ℕ) (h : 2 * n = 6) (p : ℚ) (h_prob : p = 2/5) : n = 3 :=
by
  -- problem conditions
  have h1 : 2 * n = 6 := h
  have h2 : p = 2/5 := h_prob
  -- skip the proof details
  sorry

end committee_size_l163_163435


namespace maximize_x_plus_y_l163_163192

noncomputable def max_x_plus_y (x y : ℝ) : ℝ :=
  x + y

theorem maximize_x_plus_y :
  ∀ (x y : ℝ), 
  (log (
    (x^2 + y^2) / 2) y ≥ 1 ∧ (x ≠ 0 ∨ y ≠ 0) ∧ (x^2 + y^2 ≠ 2)) →
  max_x_plus_y x y ≤ 1 + Real.sqrt 2 :=
begin
  sorry
end

end maximize_x_plus_y_l163_163192


namespace greatest_power_of_two_factor_l163_163568

theorem greatest_power_of_two_factor (n : ℕ) (h : n = 1000) :
  ∃ k, 2^k ∣ 10^n + 4^(n/2) ∧ k = 1003 :=
by {
  sorry
}

end greatest_power_of_two_factor_l163_163568


namespace four_integers_product_sum_l163_163359

theorem four_integers_product_sum (a b c d : ℕ) (h1 : a * b * c * d = 2002) (h2 : a + b + c + d < 40) :
  (a = 2 ∧ b = 7 ∧ c = 11 ∧ d = 13) ∨ (a = 1 ∧ b = 14 ∧ c = 11 ∧ d = 13) ∨
  (a = 2 ∧ b = 7 ∧ c = 11 ∧ d = 13) ∨ (a = 1 ∧ b = 14 ∧ c = 11 ∧ d = 13) :=
sorry

end four_integers_product_sum_l163_163359


namespace fountain_water_after_25_days_l163_163462

def initial_volume : ℕ := 120
def evaporation_rate : ℕ := 8 / 10 -- Representing 0.8 gallons as 8/10
def rain_addition : ℕ := 5
def days : ℕ := 25
def rain_period : ℕ := 5

-- Calculate the amount of water after 25 days given the above conditions
theorem fountain_water_after_25_days :
  initial_volume + ((days / rain_period) * rain_addition) - (days * evaporation_rate) = 125 :=
by
  sorry

end fountain_water_after_25_days_l163_163462


namespace div_fractions_eq_l163_163136

theorem div_fractions_eq : (3/7) / (5/2) = 6/35 := 
by sorry

end div_fractions_eq_l163_163136


namespace cylindrical_to_rectangular_l163_163781

theorem cylindrical_to_rectangular (r θ z : ℝ) (hr : r = 10) (hθ : θ = 3 * Real.pi / 4) (hz : z = 2) :
    ∃ (x y z' : ℝ), (x = r * Real.cos θ) ∧ (y = r * Real.sin θ) ∧ (z' = z) ∧ (x = -5 * Real.sqrt 2) ∧ (y = 5 * Real.sqrt 2) ∧ (z' = 2) :=
by
  sorry

end cylindrical_to_rectangular_l163_163781


namespace route_Y_is_quicker_l163_163097

noncomputable def route_time (distance : ℝ) (speed : ℝ) : ℝ :=
  distance / speed

def route_X_distance : ℝ := 8
def route_X_speed : ℝ := 40

def route_Y_total_distance : ℝ := 7
def route_Y_construction_distance : ℝ := 1
def route_Y_construction_speed : ℝ := 20
def route_Y_regular_speed_distance : ℝ := 6
def route_Y_regular_speed : ℝ := 50

noncomputable def route_X_time : ℝ :=
  route_time route_X_distance route_X_speed * 60  -- converting to minutes

noncomputable def route_Y_time : ℝ :=
  (route_time route_Y_regular_speed_distance route_Y_regular_speed +
  route_time route_Y_construction_distance route_Y_construction_speed) * 60 -- converting to minutes

theorem route_Y_is_quicker : route_X_time - route_Y_time = 1.8 :=
  by
    sorry

end route_Y_is_quicker_l163_163097


namespace drawing_specific_cards_from_two_decks_l163_163958

def prob_of_drawing_specific_cards (total_cards_deck1 total_cards_deck2 : ℕ) 
  (specific_card1 specific_card2 : ℕ) : ℚ :=
(specific_card1 / total_cards_deck1) * (specific_card2 / total_cards_deck2)

theorem drawing_specific_cards_from_two_decks :
  prob_of_drawing_specific_cards 52 52 1 1 = 1 / 2704 :=
by
  -- The proof can be filled in here
  sorry

end drawing_specific_cards_from_two_decks_l163_163958


namespace tony_schooling_years_l163_163432

theorem tony_schooling_years:
  let first_degree := 4
  let additional_degrees := 2 * 4
  let graduate_degree := 2
  first_degree + additional_degrees + graduate_degree = 14 :=
by {
  let first_degree := 4
  let additional_degrees := 2 * 4
  let graduate_degree := 2
  show first_degree + additional_degrees + graduate_degree = 14
  sorry
}

end tony_schooling_years_l163_163432


namespace sum_proper_divisors_81_l163_163444

theorem sum_proper_divisors_81 : 
  let n := 81,
      proper_divisors := [3^0, 3^1, 3^2, 3^3],
      sum_proper_divisors := proper_divisors.sum 
  in sum_proper_divisors = 40 := 
by
  purely
  let proper_divisors : List Nat := [1, 3, 9, 27]
  let sum_proper_divisors := proper_divisors.sum
  have : sum_proper_divisors = 1 + 3 + 9 + 27 := by rfl
  have : 1 + 3 + 9 + 27 = 40 := by rfl
  show sum_proper_divisors = 40 from this

end sum_proper_divisors_81_l163_163444


namespace puppy_price_l163_163169

theorem puppy_price (P : ℕ) (kittens_price : ℕ) (total_earnings : ℕ) :
  (kittens_price = 2 * 6) → (total_earnings = 17) → (kittens_price + P = total_earnings) → P = 5 :=
by
  intros h1 h2 h3
  sorry

end puppy_price_l163_163169


namespace rectangle_diagonal_length_l163_163288

theorem rectangle_diagonal_length (p : ℝ) (r_lw : ℝ) (l w d : ℝ) 
    (h_p : p = 84) 
    (h_ratio : r_lw = 5 / 2) 
    (h_l : l = 5 * (p / 2) / 7) 
    (h_w : w = 2 * (p / 2) / 7) 
    (h_d : d = Real.sqrt (l ^ 2 + w ^ 2)) :
  d = 2 * Real.sqrt 261 :=
by
  sorry

end rectangle_diagonal_length_l163_163288


namespace intersection_of_asymptotes_l163_163798

noncomputable def f (x : ℝ) : ℝ := (x^2 - 6*x + 8) / (x^2 - 6*x + 9)

theorem intersection_of_asymptotes : f 3 = 1 :=
by sorry

end intersection_of_asymptotes_l163_163798


namespace sum_of_first_3n_terms_l163_163742

-- Define the sums of the geometric sequence
variable (S_n S_2n S_3n : ℕ)

-- Given conditions
variable (h1 : S_n = 48)
variable (h2 : S_2n = 60)

-- The statement we need to prove
theorem sum_of_first_3n_terms (S_n S_2n S_3n : ℕ) (h1 : S_n = 48) (h2 : S_2n = 60) :
  S_3n = 63 := by
  sorry

end sum_of_first_3n_terms_l163_163742


namespace arrange_books_l163_163376

noncomputable def numberOfArrangements : Nat :=
  4 * 3 * 6 * (Nat.factorial 9)

theorem arrange_books :
  numberOfArrangements = 26210880 := by
  sorry

end arrange_books_l163_163376


namespace seq_max_min_terms_l163_163289

noncomputable def a (n: ℕ) : ℝ := 1 / (2^n - 18)

theorem seq_max_min_terms : (∀ (n : ℕ), n > 5 → a 5 > a n) ∧ (∀ (n : ℕ), n ≠ 4 → a 4 < a n) :=
by 
  sorry

end seq_max_min_terms_l163_163289


namespace isosceles_trapezoid_area_l163_163528

-- Defining the problem characteristics
variables {a b c d h θ : ℝ}

-- The area formula for an isosceles trapezoid with given bases and height
theorem isosceles_trapezoid_area (h : ℝ) (c d : ℝ) : 
  (1 / 2) * (c + d) * h = (1 / 2) * (c + d) * h := 
sorry

end isosceles_trapezoid_area_l163_163528


namespace sqrt_factorial_product_l163_163604

def factorial (n : Nat) : Nat :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem sqrt_factorial_product :
  sqrt (factorial 4 * factorial 4) = 24 :=
by
  sorry

end sqrt_factorial_product_l163_163604


namespace Marissa_sister_height_l163_163704

theorem Marissa_sister_height (sunflower_height_feet : ℕ) (height_difference_inches : ℕ) :
  sunflower_height_feet = 6 -> height_difference_inches = 21 -> 
  let sunflower_height_inches := sunflower_height_feet * 12
  let sister_height_inches := sunflower_height_inches - height_difference_inches
  let sister_height_feet := sister_height_inches / 12
  let sister_height_remainder_inches := sister_height_inches % 12
  sister_height_feet = 4 ∧ sister_height_remainder_inches = 3 :=
by
  intros
  sorry

end Marissa_sister_height_l163_163704


namespace fractions_equiv_conditions_l163_163785

theorem fractions_equiv_conditions (x y z : ℝ) (h₁ : 2 * x - z ≠ 0) (h₂ : z ≠ 0) : 
  ((2 * x + y) / (2 * x - z) = y / -z) ↔ (y = -z) :=
by
  sorry

end fractions_equiv_conditions_l163_163785


namespace polar_equation_parabola_l163_163731

/-- Given a polar equation 4 * ρ * (sin(θ / 2))^2 = 5, prove that it represents a parabola in Cartesian coordinates. -/
theorem polar_equation_parabola (ρ θ : ℝ) (h : 4 * ρ * (Real.sin (θ / 2))^ 2 = 5) : 
  ∃ (a : ℝ), a ≠ 0 ∧ (∃ b c : ℝ, ∀ x y : ℝ, (y^2 = a * (x + b)) ∨ (x = c ∨ y = 0)) := 
sorry

end polar_equation_parabola_l163_163731


namespace sneakers_sold_l163_163335

theorem sneakers_sold (total_shoes sandals boots : ℕ) (h1 : total_shoes = 17) (h2 : sandals = 4) (h3 : boots = 11) :
  total_shoes - (sandals + boots) = 2 :=
by
  -- proof steps will be included here
  sorry

end sneakers_sold_l163_163335


namespace prime_square_sum_eq_square_iff_l163_163806

theorem prime_square_sum_eq_square_iff (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q):
  (∃ x y z : ℕ, p^(2*x) + q^(2*y) = z^2) ↔ (p = 2 ∧ q = 3) ∨ (p = 3 ∧ q = 2) := 
sorry

end prime_square_sum_eq_square_iff_l163_163806


namespace find_a_l163_163065

theorem find_a (x a : ℝ) (h₁ : x^2 + x - 6 = 0) :
  (ax + 1 = 0 → (a = -1/2 ∨ a = -1/3) ∧ ax + 1 ≠ 0 ↔ false) := 
by
  sorry

end find_a_l163_163065


namespace if_a_eq_b_then_ac_eq_bc_l163_163910

theorem if_a_eq_b_then_ac_eq_bc (a b c : ℝ) : a = b → ac = bc :=
sorry

end if_a_eq_b_then_ac_eq_bc_l163_163910


namespace partial_fraction_decomposition_l163_163256

noncomputable def polynomial := λ x: ℝ => x^3 - 24 * x^2 + 88 * x - 75

theorem partial_fraction_decomposition
  (p q r A B C : ℝ)
  (hpq : p ≠ q)
  (hpr : p ≠ r)
  (hqr : q ≠ r)
  (hroots : polynomial p = 0 ∧ polynomial q = 0 ∧ polynomial r = 0)
  (hdecomposition: ∀ s: ℝ, s ≠ p ∧ s ≠ q ∧ s ≠ r → 
                      1 / polynomial s = A / (s - p) + B / (s - q) + C / (s - r)) :
  (1 / A + 1 / B + 1 / C = 256) := sorry

end partial_fraction_decomposition_l163_163256


namespace monotonic_range_l163_163495

theorem monotonic_range (a : ℝ) :
  (∀ x y, 2 ≤ x ∧ x ≤ 3 ∧ 2 ≤ y ∧ y ≤ 3 ∧ x < y → (x^2 - 2*a*x + 3) < (y^2 - 2*a*y + 3))
  ∨ (∀ x y, 2 ≤ x ∧ x ≤ 3 ∧ 2 ≤ y ∧ y ≤ 3 ∧ x < y → (x^2 - 2*a*x + 3) > (y^2 - 2*a*y + 3))
  ↔ (a ≤ 2 ∨ a ≥ 3) :=
by
  sorry

end monotonic_range_l163_163495


namespace car_return_speed_l163_163164

theorem car_return_speed (d : ℕ) (r : ℕ) (h₁ : d = 180) (h₂ : (2 * d) / ((d / 90) + (d / r)) = 60) : r = 45 :=
by
  rw [h₁] at h₂
  have h3 : 2 * 180 / ((180 / 90) + (180 / r)) = 60 := h₂
  -- The rest of the proof involves solving for r, but here we only need the statement
  sorry

end car_return_speed_l163_163164


namespace quadrilateral_AD_length_l163_163845

noncomputable def length_AD (AB BC CD : ℝ) (angleB angleC : ℝ) : ℝ :=
  let AE := AB + BC * Real.cos angleC
  let CE := BC * Real.sin angleC
  let DE := CD - CE
  Real.sqrt (AE^2 + DE^2)

theorem quadrilateral_AD_length :
  let AB := 7
  let BC := 10
  let CD := 24
  let angleB := Real.pi / 2 -- 90 degrees in radians
  let angleC := Real.pi / 3 -- 60 degrees in radians
  length_AD AB BC CD angleB angleC = Real.sqrt (795 - 240 * Real.sqrt 3) :=
by
  sorry

end quadrilateral_AD_length_l163_163845


namespace quadratic_function_positive_difference_l163_163118

/-- Given a quadratic function y = ax^2 + bx + c, where the coefficient a
indicates a downward-opening parabola (a < 0) and the y-intercept is positive (c > 0),
prove that the expression (c - a) is always positive. -/
theorem quadratic_function_positive_difference (a b c : ℝ) (h1 : a < 0) (h2 : c > 0) : c - a > 0 := 
by
  sorry

end quadratic_function_positive_difference_l163_163118


namespace duration_of_period_l163_163843

noncomputable def birth_rate : ℕ := 7
noncomputable def death_rate : ℕ := 3
noncomputable def net_increase : ℕ := 172800

theorem duration_of_period : (net_increase / ((birth_rate - death_rate) / 2)) / 3600 = 12 := by
  sorry

end duration_of_period_l163_163843


namespace dan_helmet_craters_l163_163943

namespace HelmetCraters

variables {Dan Daniel Rin : ℕ}

/-- Condition 1: Dan's skateboarding helmet has ten more craters than Daniel's ski helmet. -/
def condition1 (C_d C_daniel : ℕ) : Prop := C_d = C_daniel + 10

/-- Condition 2: Rin's snorkel helmet has 15 more craters than Dan's and Daniel's helmets combined. -/
def condition2 (C_r C_d C_daniel : ℕ) : Prop := C_r = C_d + C_daniel + 15

/-- Condition 3: Rin's helmet has 75 craters. -/
def condition3 (C_r : ℕ) : Prop := C_r = 75

/-- The main theorem: Dan's skateboarding helmet has 35 craters given the conditions. -/
theorem dan_helmet_craters (C_d C_daniel C_r : ℕ) 
    (h1 : condition1 C_d C_daniel) 
    (h2 : condition2 C_r C_d C_daniel) 
    (h3 : condition3 C_r) : C_d = 35 :=
by {
    -- We state that the answer is 35 based on the conditions
    sorry
}

end HelmetCraters

end dan_helmet_craters_l163_163943


namespace functional_equation_l163_163449

noncomputable def f : ℝ → ℝ :=
  sorry

theorem functional_equation (h : ∀ x : ℝ, f x + 3 * f (8 - x) = x) : f 2 = 2 :=
by
  sorry

end functional_equation_l163_163449


namespace hyperbola_range_of_k_l163_163838

theorem hyperbola_range_of_k (k : ℝ) :
  (∃ (x y : ℝ), (x^2)/(k-3) + (y^2)/(k+3) = 1 ∧ 
  (k-3 < 0) ∧ (k+3 > 0)) → (-3 < k ∧ k < 3) :=
by
  sorry

end hyperbola_range_of_k_l163_163838


namespace solve_for_x_l163_163254

def star (a b : ℝ) : ℝ := a * b + 3 * b - 2 * a

theorem solve_for_x (x : ℝ) : star 6 x = 45 ↔ x = 19 / 3 := by
  sorry

end solve_for_x_l163_163254


namespace smallest_number_to_add_quotient_of_resulting_number_l163_163952

theorem smallest_number_to_add (k : ℕ) : 456 ∣ (897326 + k) → k = 242 := 
sorry

theorem quotient_of_resulting_number : (897326 + 242) / 456 = 1968 := 
sorry

end smallest_number_to_add_quotient_of_resulting_number_l163_163952


namespace sum_of_three_numbers_eq_zero_l163_163547

theorem sum_of_three_numbers_eq_zero 
  (a b c : ℝ) 
  (h_sorted : a ≤ b ∧ b ≤ c) 
  (h_median : b = 10) 
  (h_mean_least : (a + b + c) / 3 = a + 20) 
  (h_mean_greatest : (a + b + c) / 3 = c - 10) 
  : a + b + c = 0 := 
by 
  sorry

end sum_of_three_numbers_eq_zero_l163_163547


namespace fraction_of_earth_surface_habitable_for_humans_l163_163506

theorem fraction_of_earth_surface_habitable_for_humans
  (total_land_fraction : ℚ) (habitable_land_fraction : ℚ)
  (h1 : total_land_fraction = 1/3)
  (h2 : habitable_land_fraction = 3/4) :
  (total_land_fraction * habitable_land_fraction) = 1/4 :=
by
  sorry

end fraction_of_earth_surface_habitable_for_humans_l163_163506


namespace large_square_min_side_and_R_max_area_l163_163684

-- Define the conditions
variable (s : ℝ) -- the side length of the larger square
variable (rect_1_side1 rect_1_side2 : ℝ) -- sides of the first rectangle
variable (square_side : ℝ) -- side of the inscribed square
variable (R_area : ℝ) -- area of the rectangle R

-- The known dimensions
axiom h1 : rect_1_side1 = 2
axiom h2 : rect_1_side2 = 4
axiom h3 : square_side = 2
axiom h4 : ∀ x y : ℝ, x > 0 → y > 0 → R_area = x * y -- non-overlapping condition

-- Define the result to be proved
theorem large_square_min_side_and_R_max_area 
  (h_r_fit_1 : rect_1_side1 + square_side ≤ s)
  (h_r_fit_2 : rect_1_side2 + square_side ≤ s)
  (h_R_max_area : R_area = 4)
  : s = 4 ∧ R_area = 4 := 
by 
  sorry

end large_square_min_side_and_R_max_area_l163_163684


namespace primes_satisfying_equation_l163_163805

theorem primes_satisfying_equation :
  ∀ (p q : ℕ), p.Prime ∧ q.Prime → 
    (∃ x y z : ℕ, p^(2*x) + q^(2*y) = z^2) ↔ 
    (p = 2 ∧ q = 3) ∨ (p = 3 ∧ q = 2) := 
by
  sorry

end primes_satisfying_equation_l163_163805


namespace carlotta_total_time_l163_163955

-- Define the main function for calculating total time
def total_time (performance_time practicing_ratio tantrum_ratio : ℕ) : ℕ :=
  performance_time + (performance_time * practicing_ratio) + (performance_time * tantrum_ratio)

-- Define the conditions from the problem
def singing_time := 6
def practicing_per_minute := 3
def tantrums_per_minute := 5

-- The expected total time based on the conditions
def expected_total_time := 54

-- The theorem to prove the equivalence
theorem carlotta_total_time :
  total_time singing_time practicing_per_minute tantrums_per_minute = expected_total_time :=
by
  sorry

end carlotta_total_time_l163_163955


namespace angus_caught_4_more_l163_163640

theorem angus_caught_4_more (
  angus ollie patrick: ℕ
) (
  h1: ollie = angus - 7
) (
  h2: ollie = 5
) (
  h3: patrick = 8
) : (angus - patrick) = 4 := 
sorry

end angus_caught_4_more_l163_163640


namespace traffic_flow_solution_l163_163336

noncomputable def traffic_flow_second_ring : ℕ := 10000
noncomputable def traffic_flow_third_ring (x : ℕ) : Prop := 3 * x - (x + 2000) = 2 * traffic_flow_second_ring

theorem traffic_flow_solution :
  ∃ (x : ℕ), traffic_flow_third_ring x ∧ (x = 11000) ∧ (x + 2000 = 13000) :=
by
  sorry

end traffic_flow_solution_l163_163336


namespace equal_real_roots_iff_c_is_nine_l163_163373

theorem equal_real_roots_iff_c_is_nine (c : ℝ) : (∃ x : ℝ, x^2 + 6 * x + c = 0 ∧ ∃ Δ, Δ = 6^2 - 4 * 1 * c ∧ Δ = 0) ↔ c = 9 :=
by
  sorry

end equal_real_roots_iff_c_is_nine_l163_163373


namespace permutations_of_BANANA_l163_163053

theorem permutations_of_BANANA : 
  let word := ["B", "A", "N", "A", "N", "A"]
  let total_letters := 6
  let repeated_A := 3
  (total_letters.factorial / repeated_A.factorial) = 120 :=
by
  sorry

end permutations_of_BANANA_l163_163053


namespace total_fish_caught_l163_163861

theorem total_fish_caught (leo_fish : ℕ) (agrey_fish : ℕ) 
  (h₁ : leo_fish = 40) (h₂ : agrey_fish = leo_fish + 20) : 
  leo_fish + agrey_fish = 100 := 
by 
  sorry

end total_fish_caught_l163_163861


namespace geometric_series_sum_l163_163644

theorem geometric_series_sum :
  let a := (1 / 4 : ℚ)
  let r := (-1 / 4 : ℚ)
  let n := 6
  let sum := a * ((1 - r ^ n) / (1 - r))
  sum = (4095 / 5120 : ℚ) :=
by
  -- Proof goes here
  sorry

end geometric_series_sum_l163_163644


namespace passes_through_origin_l163_163013

def parabola_A (x : ℝ) : ℝ := x^2 + 1
def parabola_B (x : ℝ) : ℝ := (x + 1)^2
def parabola_C (x : ℝ) : ℝ := x^2 + 2 * x
def parabola_D (x : ℝ) : ℝ := x^2 - x + 1

theorem passes_through_origin : 
  (parabola_A 0 ≠ 0) ∧
  (parabola_B 0 ≠ 0) ∧
  (parabola_C 0 = 0) ∧
  (parabola_D 0 ≠ 0) := 
by 
  sorry

end passes_through_origin_l163_163013


namespace find_integer_pairs_l163_163483

theorem find_integer_pairs :
  ∀ x y : ℤ, x^2 = 2 + 6 * y^2 + y^4 ↔ (x = 3 ∧ y = 1) ∨ (x = -3 ∧ y = 1) ∨ (x = 3 ∧ y = -1) ∨ (x = -3 ∧ y = -1) :=
by {
  sorry
}

end find_integer_pairs_l163_163483


namespace probability_divisible_by_3_l163_163400

-- Defining some basic
variables {a b c : ℕ}
variables {S : Finset ℕ} (hS : S = (Finset.range 2011).filter (λ x, x > 0))

def p_div_by_3 (a b c : ℕ) : ℚ := 
  if 0 < a ∧ a ≤ 2010 ∧ 0 < b ∧ b ≤ 2010 ∧ 0 < c ∧ c ≤ 2010 
  then if (a * b * c + a * b + a) % 3 = 0 then 1 else 0 
  else 0

theorem probability_divisible_by_3 : 
  (Finset.sum S (λ a, Finset.sum S (λ b, Finset.sum S (λ c, p_div_by_3 a b c)))) / (2010 * 2010 * 2010) = 13 / 27 :=
sorry

end probability_divisible_by_3_l163_163400


namespace prime_divides_2_pow_n_minus_n_infinte_times_l163_163269

theorem prime_divides_2_pow_n_minus_n_infinte_times (p : ℕ) (hp : Nat.Prime p) : ∃ᶠ n in at_top, p ∣ 2^n - n :=
sorry

end prime_divides_2_pow_n_minus_n_infinte_times_l163_163269


namespace prime_eq_sol_l163_163810

theorem prime_eq_sol {p q x y z : ℕ} (hp : Nat.Prime p) (hq : Nat.Prime q) :
  (∃ x y z : ℕ, p^(2*x) + q^(2*y) = z^2) ↔ ((p = 2 ∧ q = 3) ∨ (p = 3 ∧ q = 2)) :=
by
  sorry

end prime_eq_sol_l163_163810


namespace triangles_with_positive_area_l163_163676

theorem triangles_with_positive_area :
  let points := {p : ℤ × ℤ | 1 ≤ p.1 ∧ p.1 ≤ 5 ∧ 1 ≤ p.2 ∧ p.2 ≤ 5} in
  ∃ (n : ℕ), n = 2150 ∧ 
    (∃ (triangles : set (ℤ × ℤ) × (ℤ × ℤ) × (ℤ × ℤ)), 
      (∀ t ∈ triangles, 
        t.1 ∈ points ∧ t.2 ∈ points ∧ t.3 ∈ points ∧ 
        (∃ (area : ℚ), area > 0 ∧ 
          area = ((t.2.1 - t.1.1) * (t.3.2 - t.1.2) - (t.3.1 - t.1.1) * (t.2.2 - t.1.2)) / 2)) ∧ 
      ∃ (card_tris : ℕ), card_tris = n) :=
sorry

end triangles_with_positive_area_l163_163676


namespace compare_fractions_l163_163198

variables {a b : ℝ}

theorem compare_fractions (h : a + b > 0) : 
  (a / (b^2)) + (b / (a^2)) ≥ (1 / a) + (1 / b) :=
sorry

end compare_fractions_l163_163198


namespace xy_sum_equal_two_or_minus_two_l163_163987

/-- 
Given the conditions |x| = 3, |y| = 5, and xy < 0, prove that x + y = 2 or x + y = -2. 
-/
theorem xy_sum_equal_two_or_minus_two (x y : ℝ) (hx : |x| = 3) (hy : |y| = 5) (hxy : x * y < 0) : x + y = 2 ∨ x + y = -2 := 
  sorry

end xy_sum_equal_two_or_minus_two_l163_163987


namespace pie_wholesale_price_l163_163154

theorem pie_wholesale_price (p1 p2: ℕ) (d: ℕ) (k: ℕ) (h1: d > 0) (h2: k > 1)
  (price_list: List ℕ)
  (p_plus_1: price_list = [p1 + d, k * p1, p2 + d, k * p2]) 
  (h3: price_list.perm [64, 64, 70, 72]) : 
  p1 = 60 ∧ p2 = 60 := 
sorry

end pie_wholesale_price_l163_163154


namespace no_solution_for_inequalities_l163_163106

theorem no_solution_for_inequalities :
  ¬ ∃ (x y : ℝ), 4 * x^2 + 4 * x * y + 19 * y^2 ≤ 2 ∧ x - y ≤ -1 :=
by
  sorry

end no_solution_for_inequalities_l163_163106


namespace domain_of_f_l163_163115

noncomputable def f (x : ℝ) : ℝ :=  1 / Real.sqrt (1 - x^2) + x^0

theorem domain_of_f (x : ℝ) : (x > -1 ∧ x < 1 ∧ x ≠ 0) ↔ (x ∈ (-1, 0) ∨ x ∈ (0, 1)) :=
by
  sorry

end domain_of_f_l163_163115


namespace min_value_expression_ge_512_l163_163867

noncomputable def min_value_expression (a b c : ℝ) : ℝ :=
  (a^3 + 4*a^2 + a + 1) * (b^3 + 4*b^2 + b + 1) * (c^3 + 4*c^2 + c + 1) / (a * b * c)

theorem min_value_expression_ge_512 {a b c : ℝ} 
  (ha : 0 < a) 
  (hb : 0 < b) 
  (hc : 0 < c) : 
  min_value_expression a b c ≥ 512 :=
by
  sorry

end min_value_expression_ge_512_l163_163867


namespace ratio_of_side_lengths_sum_l163_163894
noncomputable def ratio_of_area := (75 : ℚ) / 128
noncomputable def rationalized_ratio_of_side_lengths := (5 * real.sqrt (6 : ℝ)) / 16
theorem ratio_of_side_lengths_sum :
  ratio_of_area = (75 : ℚ) / 128 →
  rationalized_ratio_of_side_lengths = (5 * real.sqrt (6 : ℝ)) / 16 →
  (5 : ℚ) + 6 + 16 = 27 := by
  intros _ _
  sorry

end ratio_of_side_lengths_sum_l163_163894


namespace flight_distance_l163_163460

theorem flight_distance (D : ℝ) :
  let t_out := D / 300
  let t_return := D / 500
  t_out + t_return = 8 -> D = 1500 :=
by
  intro h
  sorry

end flight_distance_l163_163460


namespace number_of_triangles_l163_163652

theorem number_of_triangles (x y : ℕ) (hx : 1 ≤ x ∧ x ≤ 5) (hy : 1 ≤ y ∧ y ≤ 5) :
  let total_points := 25, total_combinations := Nat.choose total_points 3 in
  let invalid_rows_cols := 100, invalid_diagonals := 24 in
  let valid_triangles := total_combinations - (invalid_rows_cols + invalid_diagonals) in
  valid_triangles = 2176 :=
by
  sorry

end number_of_triangles_l163_163652


namespace min_AB_dot_CD_l163_163686

theorem min_AB_dot_CD (a b : ℝ) (h1 : 0 <= (a - 1)^2 + (b - 3 / 2)^2 - 13/4) :
  ∃ (a b : ℝ), (a-1)^2 + (b - 3 / 2)^2 - 13/4 = 0 :=
by
  sorry

end min_AB_dot_CD_l163_163686


namespace commuting_days_l163_163318

theorem commuting_days 
  (a b c d x : ℕ)
  (cond1 : b + c = 12)
  (cond2 : a + c = 20)
  (cond3 : a + b + 2 * d = 14)
  (cond4 : d = 2) :
  a + b + c + d = 23 := sorry

end commuting_days_l163_163318


namespace power_function_quadrant_IV_l163_163195

theorem power_function_quadrant_IV (a : ℝ) (h : a ∈ ({-1, 1/2, 2, 3} : Set ℝ)) :
  ∀ x : ℝ, x * x^a ≠ -x * (-x^a) := sorry

end power_function_quadrant_IV_l163_163195


namespace value_of_expr_l163_163398

-- Definitions
def operation (a b : ℕ) : ℕ := 4 * a^2 + 4 * b^2 + 8 * a * b

-- The proof statement
theorem value_of_expr (a b : ℕ) (h₀ : operation a b = 100) : (a + b) + 6 = 11 := by
  sorry

end value_of_expr_l163_163398


namespace a_2_correct_l163_163370

noncomputable def a_2_value (a a1 a2 a3 : ℝ) : Prop :=
∀ x : ℝ, x^3 = a + a1 * (x - 2) + a2 * (x - 2)^2 + a3 * (x - 2)^3

theorem a_2_correct (a a1 a2 a3 : ℝ) (h : a_2_value a a1 a2 a3) : a2 = 6 :=
sorry

end a_2_correct_l163_163370


namespace sqrt_factorial_equality_l163_163580

theorem sqrt_factorial_equality : Real.sqrt (4! * 4!) = 24 := 
by
  sorry

end sqrt_factorial_equality_l163_163580


namespace unpainted_unit_cubes_l163_163309

theorem unpainted_unit_cubes (total_units : ℕ) (painted_per_face : ℕ) (painted_edges_adjustment : ℕ) :
  total_units = 216 → painted_per_face = 12 → painted_edges_adjustment = 36 → 
  total_units - (painted_per_face * 6 - painted_edges_adjustment) = 108 :=
by
  intros h_tot_units h_painted_face h_edge_adj
  sorry

end unpainted_unit_cubes_l163_163309


namespace volume_of_pool_l163_163540

variable (P T V C : ℝ)

/-- 
The volume of the pool is given as P * T divided by percentage C.
The question is to prove that the volume V of the pool equals 90000 cubic feet given:
  P: The hose can remove 60 cubic feet per minute.
  T: It takes 1200 minutes to drain the pool.
  C: The pool was at 80% capacity when draining started.
-/
theorem volume_of_pool (h1 : P = 60) 
                       (h2 : T = 1200) 
                       (h3 : C = 0.80) 
                       (h4 : P * T / C = V) :
  V = 90000 := 
sorry

end volume_of_pool_l163_163540


namespace max_value_of_f_l163_163416

noncomputable def f (x : ℝ) : ℝ := x^2 - 2*x + 3

theorem max_value_of_f (a : ℝ) (h : -2 < a ∧ a ≤ 0) : 
  ∀ x ∈ (Set.Icc 0 (a + 2)), f x ≤ 3 :=
sorry

end max_value_of_f_l163_163416


namespace line_equation_l163_163735

theorem line_equation (a b : ℝ) 
  (h1 : -4 = (a + 0) / 2)
  (h2 : 6 = (0 + b) / 2) :
  (∀ x y : ℝ, y = (3 / 2) * (x + 4) → 3 * x - 2 * y + 24 = 0) :=
by
  sorry

end line_equation_l163_163735


namespace apartment_building_count_l163_163456

theorem apartment_building_count 
  (floors_per_building : ℕ) 
  (apartments_per_floor : ℕ) 
  (doors_per_apartment : ℕ) 
  (total_doors_needed : ℕ) 
  (doors_per_building : ℕ) 
  (number_of_buildings : ℕ)
  (h1 : floors_per_building = 12)
  (h2 : apartments_per_floor = 6) 
  (h3 : doors_per_apartment = 7) 
  (h4 : total_doors_needed = 1008) 
  (h5 : doors_per_building = apartments_per_floor * doors_per_apartment * floors_per_building)
  (h6 : number_of_buildings = total_doors_needed / doors_per_building) : 
  number_of_buildings = 2 := 
by 
  rw [h1, h2, h3] at h5 
  rw [h5, h4] at h6 
  exact h6

end apartment_building_count_l163_163456


namespace problem1_problem2_l163_163775

namespace MathProblem

-- Problem 1
theorem problem1 : (π - 2)^0 + (-1)^3 = 0 := by
  sorry

-- Problem 2
variable (m n : ℤ)

theorem problem2 : (3 * m + n) * (m - 2 * n) = 3 * m ^ 2 - 5 * m * n - 2 * n ^ 2 := by
  sorry

end MathProblem

end problem1_problem2_l163_163775


namespace choose_six_with_consecutive_l163_163099

theorem choose_six_with_consecutive (n : ℕ) (h₁ : n = 49) (h₂ : 6 ≤ n) :
  (∑ k in finset.Icc 1 n, k) = (nat.choose 49 6) - (nat.choose 44 6) :=
by sorry

end choose_six_with_consecutive_l163_163099


namespace brown_rabbit_hop_distance_l163_163129

theorem brown_rabbit_hop_distance
  (w : ℕ) (b : ℕ) (t : ℕ)
  (h1 : w = 15)
  (h2 : t = 135)
  (hop_distance_in_5_minutes : w * 5 + b * 5 = t) : 
  b = 12 :=
by
  sorry

end brown_rabbit_hop_distance_l163_163129


namespace subscription_ways_three_households_l163_163979

def num_subscription_ways (n_households : ℕ) (n_newspapers : ℕ) : ℕ :=
  if h : n_households = 3 ∧ n_newspapers = 5 then
    180
  else
    0

theorem subscription_ways_three_households :
  num_subscription_ways 3 5 = 180 :=
by
  unfold num_subscription_ways
  split_ifs
  . rfl
  . contradiction


end subscription_ways_three_households_l163_163979


namespace number_of_ideal_subsets_l163_163392

def is_ideal_subset (p q : ℕ) (S : Set ℕ) : Prop :=
  0 ∈ S ∧ ∀ n ∈ S, n + p ∈ S ∧ n + q ∈ S

theorem number_of_ideal_subsets (p q : ℕ) (hpq : Nat.Coprime p q) :
  ∃ n, n = Nat.choose (p + q) p / (p + q) :=
sorry

end number_of_ideal_subsets_l163_163392


namespace log_bounds_sum_l163_163126

theorem log_bounds_sum : (∀ a b : ℕ, a = 18 ∧ b = 19 → 18 < Real.log 537800 / Real.log 2 ∧ Real.log 537800 / Real.log 2 < 19 → a + b = 37) := 
sorry

end log_bounds_sum_l163_163126


namespace central_angle_of_sector_l163_163884

theorem central_angle_of_sector (r S α : ℝ) (h1 : r = 10) (h2 : S = 100)
  (h3 : S = 1/2 * α * r^2) : α = 2 :=
by
  -- Given radius r and area S, substituting into the formula for the area of the sector,
  -- we derive the central angle α.
  sorry

end central_angle_of_sector_l163_163884


namespace oranges_and_apples_l163_163520

theorem oranges_and_apples (O A : ℕ) (h₁ : 7 * O = 5 * A) (h₂ : O = 28) : A = 20 :=
by {
  sorry
}

end oranges_and_apples_l163_163520


namespace sqrt_factorial_mul_factorial_l163_163590

theorem sqrt_factorial_mul_factorial :
  (Real.sqrt (Nat.factorial 4 * Nat.factorial 4) = 24) := by
  sorry

end sqrt_factorial_mul_factorial_l163_163590


namespace closed_polygonal_chain_exists_l163_163740

theorem closed_polygonal_chain_exists (n m : ℕ) : 
  ((n % 2 = 1 ∨ m % 2 = 1) ↔ 
   ∃ (length : ℕ), length = (n + 1) * (m + 1) ∧ length % 2 = 0) :=
by sorry

end closed_polygonal_chain_exists_l163_163740


namespace bathroom_square_footage_l163_163019

theorem bathroom_square_footage 
  (tiles_width : ℕ) (tiles_length : ℕ) (tile_size_inch : ℕ)
  (inch_to_foot : ℕ) 
  (h_width : tiles_width = 10) 
  (h_length : tiles_length = 20)
  (h_tile_size : tile_size_inch = 6)
  (h_inch_to_foot : inch_to_foot = 12) :
  let tile_size_foot : ℚ := tile_size_inch / inch_to_foot
  let width_foot : ℚ := tiles_width * tile_size_foot
  let length_foot : ℚ := tiles_length * tile_size_foot
  let area : ℚ := width_foot * length_foot
  area = 50 := 
by
  sorry

end bathroom_square_footage_l163_163019


namespace triangle_side_length_l163_163358

theorem triangle_side_length (x : ℝ) (h1 : 6 < x) (h2 : x < 14) : x = 11 :=
by
  sorry

end triangle_side_length_l163_163358


namespace problem_equivalence_l163_163173

theorem problem_equivalence : (7^2 - 3^2)^4 = 2560000 :=
by
  sorry

end problem_equivalence_l163_163173


namespace interest_rate_of_first_account_l163_163299

theorem interest_rate_of_first_account (r : ℝ) 
  (h1 : 7200 = 4000 + 4000)
  (h2 : 4000 * r = 4000 * 0.10) : 
  r = 0.10 :=
sorry

end interest_rate_of_first_account_l163_163299


namespace nth_term_correct_l163_163999

noncomputable def term_in_sequence (n : ℕ) : ℚ :=
  2^n / (2^n + 3)

theorem nth_term_correct (n : ℕ) : term_in_sequence n = 2^n / (2^n + 3) :=
by
  sorry

end nth_term_correct_l163_163999


namespace floor_of_pi_l163_163054

noncomputable def floor_of_pi_eq_three : Prop :=
  ⌊Real.pi⌋ = 3

theorem floor_of_pi : floor_of_pi_eq_three :=
  sorry

end floor_of_pi_l163_163054


namespace intersection_M_P_union_M_P_is_universal_l163_163360
-- Load the relevant libraries

open Set

-- Define the conditions

def U : Set ℝ := univ

def M (m : ℝ) : Set ℝ := {x : ℝ | -1 ≤ x ∧ x ≤ 4 * m - 2}

def P : Set ℝ := {x : ℝ | x > 2 ∨ x ≤ 1}

-- Define the Lean statement for proof problem 1
theorem intersection_M_P (m : ℝ) (h : m = 2) : 
  M m ∩ P = {x : ℝ | -1 ≤ x ∧ x ≤ 1 ∨ 2 < x ∧ x ≤ 4 * 2 - 2} :=
by
  sorry

-- Define the Lean statement for proof problem 2
theorem union_M_P_is_universal (m : ℝ) : 
  (M m ∪ P = univ) ↔ (m ≥ 1) :=
by
  sorry

end intersection_M_P_union_M_P_is_universal_l163_163360


namespace cistern_fill_time_l163_163622

theorem cistern_fill_time (F E : ℝ) (hF : F = 1 / 7) (hE : E = 1 / 9) : (1 / (F - E)) = 31.5 :=
by
  sorry

end cistern_fill_time_l163_163622


namespace total_questions_attempted_l163_163237

theorem total_questions_attempted (C W T : ℕ) (hC : C = 42) (h_score : 4 * C - W = 150) : T = C + W → T = 60 :=
by
  sorry

end total_questions_attempted_l163_163237


namespace kyler_games_won_l163_163095

theorem kyler_games_won (peter_wins peter_losses emma_wins emma_losses kyler_losses : ℕ)
  (h_peter : peter_wins = 5)
  (h_peter_losses : peter_losses = 4)
  (h_emma : emma_wins = 2)
  (h_emma_losses : emma_losses = 5)
  (h_kyler_losses : kyler_losses = 4) : ∃ kyler_wins : ℕ, kyler_wins = 2 :=
by {
  sorry
}

end kyler_games_won_l163_163095


namespace minimum_value_of_a_plus_4b_l163_163202

variable (a b : ℝ)
variable (ha : a > 0)
variable (hb : b > 0)
variable (hgeo : Real.sqrt (a * b) = 2)

theorem minimum_value_of_a_plus_4b : a + 4 * b = 8 := by
  sorry

end minimum_value_of_a_plus_4b_l163_163202


namespace angle_B_in_parallelogram_l163_163378

theorem angle_B_in_parallelogram (ABCD : Parallelogram) (angle_A angle_C : ℝ) 
  (h : angle_A + angle_C = 100) : 
  angle_B = 130 :=
by
  -- Proof omitted
  sorry

end angle_B_in_parallelogram_l163_163378


namespace total_fish_l163_163855

variable (L A : ℕ)

theorem total_fish (h1 : L = 40) (h2 : A = L + 20) : L + A = 100 := by 
  sorry

end total_fish_l163_163855


namespace quadratic_inequality_solution_l163_163655

theorem quadratic_inequality_solution (x : ℝ) : (x^2 + x - 12 > 0) → (x > 3 ∨ x < -4) :=
by
  sorry

end quadratic_inequality_solution_l163_163655


namespace log_comparison_theorem_CauchySchwarz_inequality_theorem_trigonometric_minimum_theorem_l163_163152

noncomputable def log_comparison (n : ℕ) (hn : 0 < n) : Prop := 
  Real.log n / Real.log (n + 1) < Real.log (n + 1) / Real.log (n + 2)

theorem log_comparison_theorem (n : ℕ) (hn : 0 < n) : log_comparison n hn := 
  sorry

def inequality_CauchySchwarz (a b x y : ℝ) : Prop :=
  (a*a + b*b) * (x*x + y*y) ≥ (a*x + b*y) * (a*x + b*y)

theorem CauchySchwarz_inequality_theorem (a b x y : ℝ) : inequality_CauchySchwarz a b x y :=
  sorry

noncomputable def trigonometric_minimum (x : ℝ) : ℝ := 
  (Real.sin x)^2 + (Real.cos x)^2

theorem trigonometric_minimum_theorem : ∀ x : ℝ, trigonometric_minimum x ≥ 9 :=
  sorry

end log_comparison_theorem_CauchySchwarz_inequality_theorem_trigonometric_minimum_theorem_l163_163152


namespace total_fish_caught_l163_163859

theorem total_fish_caught (leo_fish : ℕ) (agrey_fish : ℕ) (h1 : leo_fish = 40) (h2 : agrey_fish = leo_fish + 20) :
  leo_fish + agrey_fish = 100 :=
by
  sorry

end total_fish_caught_l163_163859


namespace find_c_work_rate_l163_163621

variables (A B C : ℚ)   -- Using rational numbers for the work rates

theorem find_c_work_rate (h1 : A + B = 1/3) (h2 : B + C = 1/4) (h3 : C + A = 1/6) : 
  C = 1/24 := 
sorry 

end find_c_work_rate_l163_163621


namespace sqrt_factorial_mul_factorial_l163_163592

theorem sqrt_factorial_mul_factorial :
  (Real.sqrt (Nat.factorial 4 * Nat.factorial 4) = 24) := by
  sorry

end sqrt_factorial_mul_factorial_l163_163592


namespace area_original_is_504_l163_163430

-- Define the sides of the three rectangles
variable (a1 b1 a2 b2 a3 b3 : ℕ)

-- Define the perimeters of the three rectangles
def P1 := 2 * (a1 + b1)
def P2 := 2 * (a2 + b2)
def P3 := 2 * (a3 + b3)

-- Define the conditions given in the problem
axiom P1_equal_P2_plus_20 : P1 = P2 + 20
axiom P2_equal_P3_plus_16 : P2 = P3 + 16

-- Define the calculation for the area of the original rectangle
def area_original := a1 * b1

-- Proof goal: the area of the original rectangle is 504
theorem area_original_is_504 : area_original = 504 := 
sorry

end area_original_is_504_l163_163430


namespace general_formula_seq_arithmetic_l163_163700

variable (a : ℕ → ℝ) (q : ℝ) (S : ℕ → ℝ)

-- Conditions from the problem
axiom sum_condition (n : ℕ) : (1 - q) * S n + q^n = 1
axiom nonzero_q : q * (q - 1) ≠ 0
axiom arithmetic_S : S 3 + S 9 = 2 * S 6

-- Stating the proof goals
theorem general_formula (n : ℕ) : a n = q^(n-1) :=
sorry

theorem seq_arithmetic : a 2 + a 8 = 2 * a 5 :=
sorry

end general_formula_seq_arithmetic_l163_163700


namespace binom_sub_floor_div_prime_l163_163486

theorem binom_sub_floor_div_prime {n p : ℕ} (hp : Nat.Prime p) (hpn : n ≥ p) : 
  p ∣ (Nat.choose n p - (n / p)) :=
sorry

end binom_sub_floor_div_prime_l163_163486


namespace num_ordered_pairs_l163_163831

theorem num_ordered_pairs (M N : ℕ) (hM : M > 0) (hN : N > 0) :
  (M * N = 32) → ∃ (k : ℕ), k = 6 :=
by
  sorry

end num_ordered_pairs_l163_163831


namespace student_desserts_l163_163123

theorem student_desserts (mini_cupcakes : ℕ) (donut_holes : ℕ) (students : ℕ) (equal_distribution : students ≠ 0) :
  mini_cupcakes = 14 → donut_holes = 12 → students = 13 → (mini_cupcakes + donut_holes) / students = 2 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  exact div_eq_of_eq_mul_right (by norm_num : (13 : ℕ) ≠ 0) (by norm_num : 26 = 2 * 13)
  sorry

end student_desserts_l163_163123


namespace storage_house_blocks_needed_l163_163771

noncomputable def volume_of_storage_house
  (L_o : ℕ) (W_o : ℕ) (H_o : ℕ) (T : ℕ) : ℕ :=
  let interior_length := L_o - 2 * T
  let interior_width := W_o - 2 * T
  let interior_height := H_o - T
  let outer_volume := L_o * W_o * H_o
  let interior_volume := interior_length * interior_width * interior_height
  outer_volume - interior_volume

theorem storage_house_blocks_needed :
  volume_of_storage_house 15 12 8 2 = 912 :=
  by
    sorry

end storage_house_blocks_needed_l163_163771


namespace no_solution_exists_l163_163109

theorem no_solution_exists (x y : ℝ) :
  ¬(4 * x^2 + 4 * x * y + 19 * y^2 ≤ 2 ∧ x - y ≤ -1) :=
sorry

end no_solution_exists_l163_163109


namespace find_number_l163_163319

def divisor : ℕ := 22
def quotient : ℕ := 12
def remainder : ℕ := 1
def number : ℕ := (divisor * quotient) + remainder

theorem find_number : number = 265 := by
  sorry

end find_number_l163_163319


namespace pieces_1994_impossible_pieces_1997_possible_l163_163633

def P (n : ℕ) : ℕ := 1 + 4 * n

theorem pieces_1994_impossible : ∀ n : ℕ, P n ≠ 1994 := 
by sorry

theorem pieces_1997_possible : ∃ n : ℕ, P n = 1997 := 
by sorry

end pieces_1994_impossible_pieces_1997_possible_l163_163633


namespace solve_integer_divisibility_l163_163188

theorem solve_integer_divisibility :
  {n : ℕ | n < 589 ∧ 589 ∣ (n^2 + n + 1)} = {49, 216, 315, 482} :=
by
  sorry

end solve_integer_divisibility_l163_163188


namespace option_C_forms_a_set_l163_163909

-- Definition of the criteria for forming a set
def well_defined (criterion : Prop) : Prop := criterion

-- Criteria for option C: all female students in grade one of Jiu Middle School
def grade_one_students_criteria (is_female : Prop) (is_grade_one_student : Prop) : Prop :=
  is_female ∧ is_grade_one_student

-- Proof statement
theorem option_C_forms_a_set :
  ∀ (is_female : Prop) (is_grade_one_student : Prop), well_defined (grade_one_students_criteria is_female is_grade_one_student) :=
  by sorry

end option_C_forms_a_set_l163_163909


namespace f_is_constant_l163_163866

noncomputable def is_const (f : ℤ × ℤ → ℕ) := ∃ c : ℕ, ∀ p : ℤ × ℤ, f p = c

theorem f_is_constant (f : ℤ × ℤ → ℕ) 
  (h : ∀ x y : ℤ, 4 * f (x, y) = f (x - 1, y) + f (x, y + 1) + f (x + 1, y) + f (x, y - 1)) :
  is_const f :=
sorry

end f_is_constant_l163_163866


namespace final_answer_is_correct_l163_163320

-- Define the chosen number
def chosen_number : ℤ := 1376

-- Define the division by 8
def division_result : ℤ := chosen_number / 8

-- Define the final answer
def final_answer : ℤ := division_result - 160

-- Theorem statement
theorem final_answer_is_correct : final_answer = 12 := by
  sorry

end final_answer_is_correct_l163_163320


namespace solve_for_x_l163_163368

theorem solve_for_x (y : ℝ) (x : ℝ) 
  (h : x / (x - 1) = (y^2 + 3 * y - 2) / (y^2 + 3 * y - 3)) : 
  x = (y^2 + 3 * y - 2) / 2 := 
by 
  sorry

end solve_for_x_l163_163368


namespace sqrt_factorial_product_l163_163612

theorem sqrt_factorial_product:
  (Int.sqrt (Nat.factorial 4 * Nat.factorial 4)).toNat = 24 :=
by
  sorry

end sqrt_factorial_product_l163_163612


namespace value_of_p_l163_163505

theorem value_of_p (p q r : ℕ) (h1 : p + q + r = 70) (h2 : p = 2*q) (h3 : q = 3*r) : p = 42 := 
by 
  sorry

end value_of_p_l163_163505


namespace books_initially_l163_163385

theorem books_initially (A B : ℕ) (h1 : A = 3) (h2 : B = (A + 2) + 2) : B = 7 :=
by
  -- Using the given facts, we need to show B = 7
  sorry

end books_initially_l163_163385


namespace find_q_l163_163077

theorem find_q (p q : ℝ) (h : (-2)^3 - 2*(-2)^2 + p*(-2) + q = 0) : 
  q = 16 + 2 * p :=
sorry

end find_q_l163_163077


namespace common_ratio_of_geometric_progression_l163_163733

-- Define the problem conditions
variables {a b c q : ℝ}

-- The sequence a, b, c is a geometric progression
def geometric_progression (a b c : ℝ) (q : ℝ) : Prop :=
  b = a * q ∧ c = a * q^2

-- The sequence 577a, (2020b/7), (c/7) is an arithmetic progression
def arithmetic_progression (x y z : ℝ) : Prop :=
  2 * y = x + z

-- Main theorem statement to prove
theorem common_ratio_of_geometric_progression (h1 : geometric_progression a b c q) 
  (h2 : arithmetic_progression (577 * a) (2020 * b / 7) (c / 7)) 
  (h3 : b < a ∧ c < b) : q = 4039 :=
sorry

end common_ratio_of_geometric_progression_l163_163733


namespace pie_prices_l163_163159

theorem pie_prices 
  (d k : ℕ) (hd : 0 < d) (hk : 1 < k) 
  (p1 p2 : ℕ) 
  (prices : List ℕ) (h_prices : prices = [64, 64, 70, 72]) 
  (hP1 : (p1 + d) ∈ prices)
  (hZ1 : (k * p1) ∈ prices)
  (hP2 : (p2 + d) ∈ prices)
  (hZ2 : (k * p2) ∈ prices)
  (h_different_prices : p1 + d ≠ k * p1 ∧ p2 + d ≠ k * p2) : 
  ∃ p1 p2, p1 ≠ p2 :=
sorry

end pie_prices_l163_163159


namespace triangle_number_arrangement_l163_163918

noncomputable def numbers := [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]

theorem triangle_number_arrangement : 
  ∃ (f : Fin 9 → Fin 9), 
    (numbers[f 0] + numbers[f 1] + numbers[f 2] = 
     numbers[f 3] + numbers[f 4] + numbers[f 5] ∧ 
     numbers[f 3] + numbers[f 4] + numbers[f 5] = 
     numbers[f 6] + numbers[f 7] + numbers[f 8]) :=
sorry

end triangle_number_arrangement_l163_163918


namespace committee_size_l163_163436

theorem committee_size (n : ℕ) (h : 2 * n = 6) (p : ℚ) (h_prob : p = 2/5) : n = 3 :=
by
  -- problem conditions
  have h1 : 2 * n = 6 := h
  have h2 : p = 2/5 := h_prob
  -- skip the proof details
  sorry

end committee_size_l163_163436


namespace find_function_ex_l163_163191

theorem find_function_ex (f : ℝ → ℝ) (a : ℝ) :
  (∀ x y : ℝ, f (f x + y) = 2 * x + f (f y - x)) →
  (∀ x : ℝ, f x = x - a) :=
by
  intros h x
  sorry

end find_function_ex_l163_163191


namespace geometric_sequence_sum_of_first_four_terms_l163_163842

theorem geometric_sequence_sum_of_first_four_terms (a r : ℝ) 
  (h1 : a + a * r = 7) 
  (h2 : a * (1 + r + r^2 + r^3 + r^4 + r^5) = 91) : 
  a * (1 + r + r^2 + r^3) = 32 :=
by
  sorry

end geometric_sequence_sum_of_first_four_terms_l163_163842


namespace pqr_value_l163_163110

theorem pqr_value
  (p q r : ℤ) -- p, q, and r are integers
  (hp : p ≠ 0) (hq : q ≠ 0) (hr : r ≠ 0) -- non-zero condition
  (h1 : p + q + r = 27) -- sum condition
  (h2 : (1 : ℚ) / p + (1 : ℚ) / q + (1 : ℚ) / r + 300 / (p * q * r) = 1) -- equation condition
  : p * q * r = 984 := 
sorry 

end pqr_value_l163_163110


namespace minimum_y_value_y_at_4_eq_6_l163_163835

noncomputable def y (x : ℝ) : ℝ := x + 4 / (x - 2)

theorem minimum_y_value (x : ℝ) (h : x > 2) : y x ≥ 6 :=
sorry

theorem y_at_4_eq_6 : y 4 = 6 :=
sorry

end minimum_y_value_y_at_4_eq_6_l163_163835


namespace least_integer_divisors_l163_163000

theorem least_integer_divisors (n m k : ℕ)
  (h_divisors : 3003 = 3 * 7 * 11 * 13)
  (h_form : n = m * 30 ^ k)
  (h_no_div_30 : ¬(30 ∣ m))
  (h_divisor_count : ∀ (p : ℕ) (h : n = p), (p + 1) * (p + 1) * (p + 1) * (p + 1) = 3003)
  : m + k = 104978 :=
sorry

end least_integer_divisors_l163_163000


namespace divisible_by_4_l163_163059

theorem divisible_by_4 (n m : ℕ) (h1 : n > 0) (h2 : m > 0) (h3 : n^3 + (n + 1)^3 + (n + 2)^3 = m^3) : 4 ∣ n + 1 :=
sorry

end divisible_by_4_l163_163059


namespace combined_mpg_19_l163_163100

theorem combined_mpg_19 (m: ℕ) (h: m = 100) :
  let ray_car_mpg := 50
  let tom_car_mpg := 25
  let jerry_car_mpg := 10
  let ray_gas_used := m / ray_car_mpg
  let tom_gas_used := m / tom_car_mpg
  let jerry_gas_used := m / jerry_car_mpg
  let total_gas_used := ray_gas_used + tom_gas_used + jerry_gas_used
  let total_miles := 3 * m
  let combined_mpg := total_miles * 25 / (4 * m)
  combined_mpg = 19 := 
by {
  sorry
}

end combined_mpg_19_l163_163100


namespace value_of_expression_l163_163062

theorem value_of_expression (a : ℝ) (h : a^2 - 2 * a = 1) : 3 * a^2 - 6 * a - 4 = -1 :=
by
  sorry

end value_of_expression_l163_163062


namespace geometric_sequence_alpha5_eq_three_l163_163515

theorem geometric_sequence_alpha5_eq_three (α : ℕ → ℝ) 
  (h1 : ∀ n : ℕ, α (n + 1) = α n * r) 
  (h2 : α 4 * α 5 * α 6 = 27) : α 5 = 3 := 
by
  sorry

end geometric_sequence_alpha5_eq_three_l163_163515


namespace solve_arithmetic_sequence_l163_163883

theorem solve_arithmetic_sequence (y : ℝ) (h : y > 0) : 
  let a1 := (2 : ℝ)^2
      a2 := y^2
      a3 := (4 : ℝ)^2
  in (a1 + a3) / 2 = a2 → y = Real.sqrt 10 :=
by
  intros a1 a2 a3 H
  have calc1 : a1 = 4 := by norm_num
  have calc2 : a3 = 16 := by norm_num
  rw [calc1, calc2] at H
  have avg_eq : (4 + 16) / 2 = 10 := by norm_num
  rw [avg_eq] at H
  suffices y_pos : y > 0, sorry
  sorry


end solve_arithmetic_sequence_l163_163883


namespace value_of_a_l163_163989

theorem value_of_a (a : ℕ) (h : a^3 = 21 * 49 * 45 * 25) : a = 105 := sorry

end value_of_a_l163_163989


namespace wholesale_price_is_60_l163_163157

variables (p d k : ℕ)
variables (P1 P2 Z1 Z2 : ℕ)

// Conditions from (a)
axiom H1 : P1 = p + d // Baker Plus price day 1
axiom H2 : P2 = p + d // Baker Plus price day 2
axiom H3 : Z1 = k * p // Star price day 1
axiom H4 : Z2 = k * p // Star price day 2
axiom H5 : k > 1 // Star store markup factor greater than 1
axiom H6 : P1 ∈ (set.insert 64 (set.insert 64 (set.insert 70 (set.singleton 72))))
axiom H7 : P2 ∈ (set.insert 64 (set.insert 64 (set.insert 70 (set.singleton 72))))
axiom H8 : Z1 ∈ (set.insert 64 (set.insert 64 (set.insert 70 (set.singleton 72))))
axiom H9 : Z2 ∈ (set.insert 64 (set.insert 64 (set.insert 70 (set.singleton 72))))

theorem wholesale_price_is_60 : p = 60 := 
by {
    // Proof goes here, skipped with sorry
    sorry
}

end wholesale_price_is_60_l163_163157


namespace abs_a_gt_abs_b_l163_163501

variable (a b : Real)

theorem abs_a_gt_abs_b (h1 : a > 0) (h2 : b < 0) (h3 : a + b > 0) : |a| > |b| :=
by
  sorry

end abs_a_gt_abs_b_l163_163501


namespace find_a_l163_163365

variable {a : ℝ}

def A : Set ℝ := {2, 4}
def B (a : ℝ) : Set ℝ := {a, a^2 + 3}

theorem find_a (h : A ∩ (B a) = {2}) : a = 2 :=
by
  sorry

end find_a_l163_163365


namespace min_value_of_cos2_plus_sin_in_interval_l163_163548

noncomputable
def minValueFunction : ℝ :=
  let f : ℝ → ℝ := λ x, (Real.cos x) ^ 2 + Real.sin x
  let interval : Set ℝ := Set.interval (-Real.pi / 4) (Real.pi / 4)
  let minimum_value : ℝ := (1 - Real.sqrt 2) / 2
  if f ∈ continuous_on interval then minimum_value else sorry

theorem min_value_of_cos2_plus_sin_in_interval :
  let f : ℝ → ℝ := λ x, (Real.cos x) ^ 2 + Real.sin x
  let interval : Set ℝ := Set.interval (-Real.pi / 4) (Real.pi / 4)
  has_minimum_on f interval ((1 - Real.sqrt 2) / 2) :=
by
  sorry

end min_value_of_cos2_plus_sin_in_interval_l163_163548


namespace average_eq_instantaneous_velocity_at_t_eq_3_l163_163827

theorem average_eq_instantaneous_velocity_at_t_eq_3
  (S : ℝ → ℝ) (hS : ∀ t, S t = 24 * t - 3 * t^2) :
  (1 / 6) * (S 6 - S 0) = 24 - 6 * 3 :=
by 
  sorry

end average_eq_instantaneous_velocity_at_t_eq_3_l163_163827


namespace simplify_expression_eq_69_l163_163278

theorem simplify_expression_eq_69 : 80 - (5 - (6 + 2 * (7 - 8 - 5))) = 69 := by
  sorry

end simplify_expression_eq_69_l163_163278


namespace f_geq_3e2x_minus_2e3x_l163_163726

-- Define our problem statement in Lean 4
theorem f_geq_3e2x_minus_2e3x (f : ℝ → ℝ) 
    (hf_diff : Differentiable ℝ f)
    (hf_diff2 : Differentiable ℝ (f'))
    (hf0 : f 0 = 1) 
    (hf_prime0 : f' 0 = 0)
    (h_ineq : ∀ x : ℝ, 0 ≤ x → f'' x - 5 * f' x + 6 * f x ≥ 0) :
    ∀ x ≥ 0, f x ≥ 3 * Real.exp (2 * x) - 2 * Real.exp (3 * x) := 
begin
    sorry
end

end f_geq_3e2x_minus_2e3x_l163_163726


namespace findPrincipalAmount_l163_163303

noncomputable def principalAmount (r : ℝ) (t : ℝ) (diff : ℝ) : ℝ :=
  let n := 2 -- compounded semi-annually
  let rate_per_period := (1 + r / n)
  let num_periods := n * t
  (diff / (rate_per_period^num_periods - 1 - r * t))

theorem findPrincipalAmount :
  let r := 0.05
  let t := 3
  let diff := 25
  abs (principalAmount r t diff - 2580.39) < 0.01 := 
by 
  sorry

end findPrincipalAmount_l163_163303


namespace hilton_final_marbles_l163_163070

def initial_marbles : ℕ := 26
def marbles_found : ℕ := 6
def marbles_lost : ℕ := 10
def marbles_from_lori := 2 * marbles_lost

def final_marbles := initial_marbles + marbles_found - marbles_lost + marbles_from_lori

theorem hilton_final_marbles : final_marbles = 42 := sorry

end hilton_final_marbles_l163_163070


namespace sam_morning_run_l163_163714

variable (X : ℝ)
variable (run_miles : ℝ) (walk_miles : ℝ) (bike_miles : ℝ) (total_miles : ℝ)

-- Conditions
def condition1 := walk_miles = 2 * run_miles
def condition2 := bike_miles = 12
def condition3 := total_miles = 18
def condition4 := run_miles + walk_miles + bike_miles = total_miles

-- Proof of the distance Sam ran in the morning
theorem sam_morning_run :
  (condition1 X run_miles walk_miles) →
  (condition2 bike_miles) →
  (condition3 total_miles) →
  (condition4 run_miles walk_miles bike_miles total_miles) →
  run_miles = 2 := by
  sorry

end sam_morning_run_l163_163714


namespace max_value_inequality_l163_163868

theorem max_value_inequality (x y k : ℝ) (hx : 0 < x) (hy : 0 < y) (hk : 0 < k) :
  (kx + y)^2 / (x^2 + y^2) ≤ 2 :=
sorry

end max_value_inequality_l163_163868


namespace equation_of_directrix_l163_163782

theorem equation_of_directrix (x y : ℝ) (h : y^2 = 2 * x) : 
  x = - (1/2) :=
sorry

end equation_of_directrix_l163_163782


namespace additional_trams_proof_l163_163561

-- Definitions for the conditions
def initial_tram_count : Nat := 12
def total_distance : Nat := 60
def initial_interval : Nat := total_distance / initial_tram_count
def reduced_interval : Nat := initial_interval - (initial_interval / 5)
def final_tram_count : Nat := total_distance / reduced_interval
def additional_trams_needed : Nat := final_tram_count - initial_tram_count

-- The theorem we need to prove
theorem additional_trams_proof : additional_trams_needed = 3 :=
by
  sorry

end additional_trams_proof_l163_163561


namespace Clea_Rides_Escalator_Alone_l163_163851

-- Defining the conditions
variables (x y k : ℝ)
def Clea_Walking_Speed := x
def Total_Distance := y = 75 * x
def Time_with_Moving_Escalator := 30 * (x + k) = y
def Escalator_Speed := k = 1.5 * x

-- Stating the proof problem
theorem Clea_Rides_Escalator_Alone : 
  Total_Distance x y → 
  Time_with_Moving_Escalator x y k → 
  Escalator_Speed x k → 
  y / k = 50 :=
by
  intros
  sorry

end Clea_Rides_Escalator_Alone_l163_163851


namespace A_on_curve_slope_at_A_l163_163670

-- Define the function f
def f (x : ℝ) : ℝ := 2 * x ^ 2

-- Define the point A on the curve f
def A : ℝ × ℝ := (2, 8)

-- Define the condition that A is on the curve f
theorem A_on_curve : A.2 = f A.1 := by
  -- * left as a proof placeholder
  sorry

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 4 * x

-- State and prove the main theorem
theorem slope_at_A : (deriv f) 2 = 8 := by
  -- * left as a proof placeholder
  sorry

end A_on_curve_slope_at_A_l163_163670


namespace probability_of_selecting_multiple_l163_163467

open BigOperators

-- Define the set of multipliers
def multiples (n m : ℕ) : Finset ℕ := (Finset.range (n + 1)).filter (λ x => x % m = 0)

-- Calculate the cardinality of the union of multiple sets given a list of divisors
def count_multiples (n : ℕ) (divisors : List ℕ) : ℕ := 
  let sets := divisors.map (multiples n)
  Finset.card (Finset.sup id sets)

-- Our original problem setup
def total_cards : ℕ := 200
def divisors : List ℕ := [2, 3, 5, 7]

-- The main theorem
theorem probability_of_selecting_multiple : Fraction.mk (count_multiples total_cards divisors) total_cards = Fraction.mk 151 200 :=
by
  sorry

end probability_of_selecting_multiple_l163_163467


namespace frood_points_smallest_frood_points_l163_163687

theorem frood_points (n : ℕ) (h : n > 9) : (n * (n + 1)) / 2 > 5 * n :=
by {
  sorry
}

noncomputable def smallest_n : ℕ := 10

theorem smallest_frood_points (m : ℕ) (h : (m * (m + 1)) / 2 > 5 * m) : 10 ≤ m :=
by {
  sorry
}

end frood_points_smallest_frood_points_l163_163687


namespace find_k_l163_163088

noncomputable section

open Polynomial

-- Define the conditions
variables (h k : Polynomial ℚ)
variables (C : k.eval (-1) = 15) (H : h.comp k = h * k) (nonzero_h : h ≠ 0)

-- The goal is to prove k(x) = x^2 + 21x - 35
theorem find_k : k = X^2 + 21 * X - 35 :=
  by sorry

end find_k_l163_163088


namespace dice_sum_10_with_5_prob_l163_163133

open ProbabilityTheory

def six_sided_die := {1, 2, 3, 4, 5, 6}

noncomputable def prob_sum_10_at_least_one_5 : ℚ :=
  Pr (λ (rolls : Fin 3 → ℕ), 
    (∃ i, rolls i = 5) ∧ (rolls 0 + rolls 1 + rolls 2 = 10)) six_sided_die

theorem dice_sum_10_with_5_prob :
  prob_sum_10_at_least_one_5 = 1 / 18 :=
sorry

end dice_sum_10_with_5_prob_l163_163133


namespace sum_proper_divisors_eq_40_l163_163443

def is_proper_divisor (n d : ℕ) : Prop :=
  d ∣ n ∧ d ≠ n

def proper_divisors (n : ℕ) : List ℕ :=
  List.filter (λ d => is_proper_divisor n d) (List.range (n + 1))

def sum_proper_divisors (n : ℕ) : ℕ :=
  (proper_divisors n).sum

theorem sum_proper_divisors_eq_40 : sum_proper_divisors 81 = 40 := sorry

end sum_proper_divisors_eq_40_l163_163443


namespace gp_values_l163_163795

theorem gp_values (p : ℝ) (hp : 0 < p) :
  let a := -p - 12
  let b := 2 * Real.sqrt p
  let c := p - 5
  (b / a = c / b) ↔ p = 4 :=
by
  sorry

end gp_values_l163_163795


namespace base_of_third_term_l163_163503

theorem base_of_third_term (x : ℝ) (some_number : ℝ) :
  625^(-x) + 25^(-2 * x) + some_number^(-4 * x) = 14 → x = 0.25 → some_number = 125 / 1744 :=
by
  intros h1 h2
  sorry

end base_of_third_term_l163_163503


namespace total_texts_received_l163_163049

open Nat 

-- Definition of conditions
def textsBeforeNoon : Nat := 21
def initialTextsAfterNoon : Nat := 2
def doublingTimeHours : Nat := 12

-- Definition to compute the total texts after noon recursively
def textsAfterNoon (n : Nat) : Nat :=
  if n = 0 then initialTextsAfterNoon
  else 2 * textsAfterNoon (n - 1)

-- Definition to sum the geometric series 
def sumGeometricSeries (a r n : Nat) : Nat :=
  if n = 0 then 0
  else a * (1 - r ^ n) / (1 - r)

-- Total text messages Debby received
def totalTextsReceived : Nat :=
  textsBeforeNoon + sumGeometricSeries initialTextsAfterNoon 2 doublingTimeHours

-- Proof statement
theorem total_texts_received: totalTextsReceived = 8211 := 
by 
  sorry

end total_texts_received_l163_163049


namespace min_value_l163_163959

variable (a b c : ℝ)

theorem min_value (h1 : a > b) (h2 : b > c) (h3 : a - c = 5) : 
  (a - b) ^ 2 + (b - c) ^ 2 = 25 / 2 := 
sorry

end min_value_l163_163959


namespace fraction_value_l163_163367

-- Define the variables x and y as real numbers
variables (x y : ℝ)

-- State the theorem
theorem fraction_value (h : 2 * x = -y) : (x * y) / (x^2 - y^2) = 2 / 3 :=
by
  sorry

end fraction_value_l163_163367


namespace number_of_ways_to_arrange_matches_l163_163184

open Nat

theorem number_of_ways_to_arrange_matches :
  (factorial 7) * (2 ^ 3) = 40320 := by
  sorry

end number_of_ways_to_arrange_matches_l163_163184


namespace number_of_words_with_A_is_correct_l163_163887

open Finset

def alphabet : Finset Char := { 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z' }

noncomputable def words_of_length (n : ℕ) : Finset (List Char) :=
  (range (26^n)).Image (λ i, (List.finRange n).map (λ k, alphabet.toList.get! (i / (26^k) % 26)))

noncomputable def words_with_A (n : ℕ) : Finset (List Char) :=
  words_of_length n ∖ (words_of_length n).filter (λ w, ('A' ∉ w))

noncomputable def total_words_with_A : ℕ :=
  ∑ i in range (6), (words_with_A i).card

theorem number_of_words_with_A_is_correct : total_words_with_A = 2202115 :=
  by
    sorry

end number_of_words_with_A_is_correct_l163_163887


namespace tan_neg_two_sin_cos_sum_l163_163834

theorem tan_neg_two_sin_cos_sum (θ : ℝ) (h : Real.tan θ = -2) : 
  Real.sin (2 * θ) + Real.cos (2 * θ) = -7 / 5 :=
by
  sorry

end tan_neg_two_sin_cos_sum_l163_163834


namespace tan_alpha_minus_pi_over_4_l163_163494

noncomputable def alpha : ℝ := sorry
axiom alpha_in_range : -Real.pi / 2 < alpha ∧ alpha < 0
axiom cos_alpha : Real.cos alpha = (Real.sqrt 5) / 5

theorem tan_alpha_minus_pi_over_4 : Real.tan (alpha - Real.pi / 4) = 3 := by
  sorry

end tan_alpha_minus_pi_over_4_l163_163494


namespace mutually_exclusive_and_complementary_l163_163346

noncomputable def bag : Finset (Finset ℕ) :=
  {{0, 1}, {0, 2}, {1, 2}, {0, 3}, {1, 3}, {2, 3}}

def at_least_one_black (s : Finset ℕ) : Prop :=
  ∃ x ∈ s, x = 0 ∨ x = 1

def all_white (s : Finset ℕ) : Prop :=
  ∀ x ∈ s, x = 2 ∨ x = 3

theorem mutually_exclusive_and_complementary :
  (∀ s ∈ bag, at_least_one_black s ∧ all_white s → false) ∧
  ((∀ s ∈ bag, at_least_one_black s ∨ all_white s) ∧ (∀ s ∈ bag, ¬(at_least_one_black s ∧ all_white s))) :=
by
  sorry

end mutually_exclusive_and_complementary_l163_163346


namespace fourth_root_of_207360000_l163_163046

theorem fourth_root_of_207360000 :
  120 ^ 4 = 207360000 :=
sorry

end fourth_root_of_207360000_l163_163046


namespace probability_of_ram_l163_163904

theorem probability_of_ram 
  (P_ravi : ℝ) (P_both : ℝ) 
  (h_ravi : P_ravi = 1 / 5) 
  (h_both : P_both = 0.11428571428571428) : 
  ∃ P_ram : ℝ, P_ram = 0.5714285714285714 :=
by
  sorry

end probability_of_ram_l163_163904


namespace bathroom_square_footage_l163_163021

theorem bathroom_square_footage
  (tiles_width : ℕ)
  (tiles_length : ℕ)
  (tile_size_inches : ℕ)
  (inches_per_foot : ℕ)
  (h1 : tiles_width = 10)
  (h2 : tiles_length = 20)
  (h3 : tile_size_inches = 6)
  (h4 : inches_per_foot = 12)
: (tiles_length * tile_size_inches / inches_per_foot) * (tiles_width * tile_size_inches / inches_per_foot) = 50 := 
by
  sorry

end bathroom_square_footage_l163_163021


namespace find_value_of_a_l163_163301

theorem find_value_of_a (a : ℝ) 
  (h : (2 * a + 16 + 3 * a - 8) / 2 = 69) : a = 26 := 
by
  sorry

end find_value_of_a_l163_163301


namespace sqrt_factorial_mul_factorial_eq_l163_163596

theorem sqrt_factorial_mul_factorial_eq :
  Real.sqrt (Nat.factorial 4 * Nat.factorial 4) = 24 := by
sorry

end sqrt_factorial_mul_factorial_eq_l163_163596


namespace problem_l163_163824

noncomputable def f (x : ℝ) : ℝ := Real.sin x + x - Real.pi / 4
noncomputable def g (x : ℝ) : ℝ := Real.cos x - x + Real.pi / 4

theorem problem (x1 x2 : ℝ) (hx1 : 0 < x1 ∧ x1 < Real.pi / 2) (hx2 : 0 < x2 ∧ x2 < Real.pi / 2) :
  (∃! x, 0 < x ∧ x < Real.pi / 2 ∧ f x = 0) ∧ (∃! x, 0 < x ∧ x < Real.pi / 2 ∧ g x = 0) →
  x1 + x2 = Real.pi / 2 :=
by
  sorry -- Proof goes here

end problem_l163_163824


namespace find_sum_of_squares_of_roots_l163_163362

theorem find_sum_of_squares_of_roots (a b c : ℝ) (h_ab : a < b) (h_bc : b < c)
  (f : ℝ → ℝ) (hf : ∀ x, f x = x^3 - 2 * x^2 - 3 * x + 4)
  (h_eq : f a = f b ∧ f b = f c) :
  a^2 + b^2 + c^2 = 10 :=
sorry

end find_sum_of_squares_of_roots_l163_163362


namespace max_P_l163_163869

noncomputable def P (a b : ℝ) : ℝ :=
  (a^2 + 6*b + 1) / (a^2 + a)

theorem max_P (a b x1 x2 x3 : ℝ) (h1 : a = x1 + x2 + x3) (h2 : a = x1 * x2 * x3) (h3 : ab = x1 * x2 + x2 * x3 + x3 * x1) 
    (hx1 : 0 < x1) (hx2 : 0 < x2) (hx3 : 0 < x3) :
    P a b ≤ (9 + Real.sqrt 3) / 9 := 
sorry

end max_P_l163_163869


namespace remainders_equal_if_difference_divisible_l163_163534

theorem remainders_equal_if_difference_divisible (a b k : ℤ) (h : k ∣ (a - b)) : 
  a % k = b % k :=
sorry

end remainders_equal_if_difference_divisible_l163_163534


namespace number_of_people_ate_pizza_l163_163325

theorem number_of_people_ate_pizza (total_slices initial_slices remaining_slices slices_per_person : ℕ) 
  (h1 : initial_slices = 16) 
  (h2 : remaining_slices = 4) 
  (h3 : slices_per_person = 2) 
  (eaten_slices : total_slices) 
  (h4 : total_slices = initial_slices - remaining_slices) :
    total_slices / slices_per_person = 6 := 
by {
  -- specify the value for eaten_slices to simplify the calculation
  let eaten_slices := initial_slices - remaining_slices,
  have h5 : eaten_slices = 12, from sorry,
  sorry
  }

end number_of_people_ate_pizza_l163_163325


namespace total_fish_l163_163857

variable (L A : ℕ)

theorem total_fish (h1 : L = 40) (h2 : A = L + 20) : L + A = 100 := by 
  sorry

end total_fish_l163_163857


namespace quadratic_expression_value_l163_163203

theorem quadratic_expression_value (a : ℝ)
  (h1 : ∃ x₁ x₂ : ℝ, x₁^2 + 2 * (a - 1) * x₁ + a^2 - 7 * a - 4 = 0 ∧ x₂^2 + 2 * (a - 1) * x₂ + a^2 - 7 * a - 4 = 0)
  (h2 : ∀ x₁ x₂ : ℝ, x₁ * x₂ - 3 * x₁ - 3 * x₂ - 2 = 0) :
  (1 + 4 / (a^2 - 4)) * (a + 2) / a = 2 := 
sorry

end quadratic_expression_value_l163_163203


namespace angle_B_in_parallelogram_l163_163377

theorem angle_B_in_parallelogram (ABCD : Parallelogram) (angle_A angle_C : ℝ) 
  (h : angle_A + angle_C = 100) : 
  angle_B = 130 :=
by
  -- Proof omitted
  sorry

end angle_B_in_parallelogram_l163_163377


namespace volume_and_area_of_pyramid_l163_163272

-- Define the base of the pyramid.
def rect (EF FG : ℕ) : Prop := EF = 10 ∧ FG = 6

-- Define the perpendicular relationships and height of the pyramid.
def pyramid (EF FG PE : ℕ) : Prop := 
  rect EF FG ∧
  PE = 10 ∧ 
  (PE > 0) -- Given conditions include perpendicular properties, implying height is positive.

-- Problem translation: Prove the volume and area calculations.
theorem volume_and_area_of_pyramid (EF FG PE : ℕ) 
  (h1 : rect EF FG) 
  (h2 : PE = 10) : 
  (1 / 3 * EF * FG * PE = 200 ∧ 1 / 2 * EF * FG = 30) := 
by
  sorry

end volume_and_area_of_pyramid_l163_163272


namespace number_of_happy_configurations_is_odd_l163_163526

def S (m n : ℕ) := {p : ℕ × ℕ | 1 ≤ p.1 ∧ p.1 ≤ 2 * m ∧ 1 ≤ p.2 ∧ p.2 ≤ 2 * n}

def happy_configurations (m n : ℕ) : ℕ := 
  sorry -- definition of the number of happy configurations is abstracted for this statement.

theorem number_of_happy_configurations_is_odd (m n : ℕ) (hm : 0 < m) (hn : 0 < n) : 
  happy_configurations m n % 2 = 1 := 
sorry

end number_of_happy_configurations_is_odd_l163_163526


namespace thomas_savings_years_l163_163007

def weekly_allowance : ℕ := 50
def weekly_coffee_shop_earning : ℕ := 9 * 30
def weekly_spending : ℕ := 35
def car_cost : ℕ := 15000
def additional_amount_needed : ℕ := 2000
def weeks_in_a_year : ℕ := 52

def first_year_savings : ℕ := weeks_in_a_year * (weekly_allowance - weekly_spending)
def second_year_savings : ℕ := weeks_in_a_year * (weekly_coffee_shop_earning - weekly_spending)

noncomputable def total_savings_needed : ℕ := car_cost - additional_amount_needed

theorem thomas_savings_years : 
  first_year_savings + second_year_savings = total_savings_needed → 2 = 2 :=
by
  sorry

end thomas_savings_years_l163_163007


namespace expected_value_die_l163_163464

noncomputable def expected_value (P_Star P_Moon : ℚ) (win_Star lose_Moon : ℚ) : ℚ :=
  P_Star * win_Star + P_Moon * lose_Moon

theorem expected_value_die :
  expected_value (2/5) (3/5) 4 (-3) = -1/5 := by
  sorry

end expected_value_die_l163_163464


namespace additional_hours_on_days_without_practice_l163_163337

def total_weekday_homework_hours : ℕ := 2 + 3 + 4 + 3 + 1
def total_weekend_homework_hours : ℕ := 8
def total_homework_hours : ℕ := total_weekday_homework_hours + total_weekend_homework_hours
def total_chore_hours : ℕ := 1 + 1
def total_hours : ℕ := total_homework_hours + total_chore_hours

theorem additional_hours_on_days_without_practice : ∀ (practice_nights : ℕ), 
  (2 ≤ practice_nights ∧ practice_nights ≤ 3) →
  (∃ tuesday_wednesday_thursday_weekend_day_hours : ℕ,
    tuesday_wednesday_thursday_weekend_day_hours = 15) :=
by
  intros practice_nights practice_nights_bounds
  -- Define days without practice in the worst case scenario
  let tuesday_hours := 3
  let wednesday_homework_hours := 4
  let wednesday_chore_hours := 1
  let thursday_hours := 3
  let weekend_day_hours := 4
  let days_without_practice_hours := tuesday_hours + (wednesday_homework_hours + wednesday_chore_hours) + thursday_hours + weekend_day_hours
  use days_without_practice_hours
  -- In the worst case, the total additional hours on days without practice should be 15.
  sorry

end additional_hours_on_days_without_practice_l163_163337


namespace jessica_final_balance_l163_163756

variable (B : ℝ) (withdrawal : ℝ) (deposit : ℝ)

-- Conditions
def condition1 : Prop := withdrawal = (2 / 5) * B
def condition2 : Prop := deposit = (1 / 5) * (B - withdrawal)

-- Proof goal statement
theorem jessica_final_balance (h1 : condition1 B withdrawal)
                             (h2 : condition2 B withdrawal deposit) :
    (B - withdrawal + deposit) = 360 :=
by
  sorry

end jessica_final_balance_l163_163756


namespace tan_identity_l163_163105

theorem tan_identity (α β γ : ℝ) (h : α + β + γ = 45 * π / 180) :
  (1 + Real.tan α) * (1 + Real.tan β) * (1 + Real.tan γ) / (1 + Real.tan α * Real.tan β * Real.tan γ) = 2 :=
by
  sorry

end tan_identity_l163_163105


namespace solve_arithmetic_sequence_l163_163882

-- State the main problem in Lean 4
theorem solve_arithmetic_sequence (y : ℝ) (h : y^2 = (4 + 16) / 2) (hy : y > 0) : y = Real.sqrt 10 := by
  sorry

end solve_arithmetic_sequence_l163_163882


namespace least_positive_integer_x_l163_163797

theorem least_positive_integer_x (x : ℕ) (h1 : x + 3721 ≡ 1547 [MOD 12]) (h2 : x % 2 = 0) : x = 2 :=
sorry

end least_positive_integer_x_l163_163797


namespace binomial_coeff_equal_l163_163153

theorem binomial_coeff_equal (n : ℕ) (h₁ : 6 ≤ n) (h₂ : (n.choose 5) * 3^5 = (n.choose 6) * 3^6) :
  n = 7 := sorry

end binomial_coeff_equal_l163_163153


namespace problem_statement_l163_163657

theorem problem_statement (x : ℝ) (h : x ≠ 2) :
  (x * (x + 1)) / ((x - 2)^2) ≥ 8 ↔ (1 ≤ x ∧ x < 2) ∨ (32/7 < x) :=
by 
  sorry

end problem_statement_l163_163657


namespace find_radius_of_sphere_l163_163635

noncomputable def radius_of_sphere (R : ℝ) : Prop :=
  ∃ a b c : ℝ, 
  (R = |a| ∧ R = |b| ∧ R = |c|) ∧ 
  ((3 - R)^2 + (2 - R)^2 + (1 - R)^2 = R^2)

theorem find_radius_of_sphere : radius_of_sphere (3 + Real.sqrt 2) ∨ radius_of_sphere (3 - Real.sqrt 2) :=
sorry

end find_radius_of_sphere_l163_163635


namespace distance_planes_A_B_l163_163709

noncomputable def distance_between_planes : ℝ :=
  let d1 := 1
  let d2 := 2
  let a := 1
  let b := 1
  let c := 1
  (|d2 - d1|) / (Real.sqrt (a^2 + b^2 + c^2))

theorem distance_planes_A_B :
  let A := fun (x y z : ℝ) => x + y + z = 1
  let B := fun (x y z : ℝ) => x + y + z = 2
  distance_between_planes = 1 / Real.sqrt 3 :=
  by
    -- Proof steps will be here
    sorry

end distance_planes_A_B_l163_163709


namespace maximize_box_volume_l163_163463

-- Define the volume function
def volume (x : ℝ) := (48 - 2 * x)^2 * x

-- Define the constraint on x
def constraint (x : ℝ) := 0 < x ∧ x < 24

-- The theorem stating the side length of the removed square that maximizes the volume
theorem maximize_box_volume : ∃ x : ℝ, constraint x ∧ (∀ y : ℝ, constraint y → volume y ≤ volume 8) :=
by
  sorry

end maximize_box_volume_l163_163463


namespace total_fish_caught_l163_163863

theorem total_fish_caught (leo_fish : ℕ) (agrey_fish : ℕ) 
  (h₁ : leo_fish = 40) (h₂ : agrey_fish = leo_fish + 20) : 
  leo_fish + agrey_fish = 100 := 
by 
  sorry

end total_fish_caught_l163_163863


namespace root_interval_l163_163306

def f (x : ℝ) : ℝ := 5 * x - 7

theorem root_interval : ∃ x, 1 < x ∧ x < 2 ∧ f x = 0 :=
by
  -- Proof steps should be here
  sorry

end root_interval_l163_163306


namespace hyperbola_eccentricity_l163_163630

noncomputable def calculate_eccentricity (a b c x0 y0 : ℝ) : ℝ :=
  c / a

theorem hyperbola_eccentricity :
  ∀ (a b c x0 y0 : ℝ),
    (c = 2) →
    (a^2 + b^2 = 4) →
    (x0 = 3) →
    (y0^2 = 24) →
    (5 = x0 + 2) →
    calculate_eccentricity a b c x0 y0 = 2 := 
by 
  intros a b c x0 y0 h1 h2 h3 h4 h5
  sorry

end hyperbola_eccentricity_l163_163630


namespace rachel_earnings_without_tips_l163_163401

theorem rachel_earnings_without_tips
  (num_people : ℕ) (tip_per_person : ℝ) (total_earnings : ℝ)
  (h1 : num_people = 20)
  (h2 : tip_per_person = 1.25)
  (h3 : total_earnings = 37) :
  total_earnings - (num_people * tip_per_person) = 12 :=
by
  sorry

end rachel_earnings_without_tips_l163_163401


namespace eval_expression_l163_163186

theorem eval_expression : 3 - (-1) + 4 - 5 + (-6) - (-7) + 8 - 9 = 3 := 
  sorry

end eval_expression_l163_163186


namespace value_of_expression_l163_163499

theorem value_of_expression (x : ℝ) : 
  let a := 2000 * x + 2001
  let b := 2000 * x + 2002
  let c := 2000 * x + 2003
  a^2 + b^2 + c^2 - a * b - b * c - c * a = 3 :=
by
  sorry

end value_of_expression_l163_163499


namespace factor_polynomial_l163_163187

theorem factor_polynomial (x : ℝ) : 
    54 * x^4 - 135 * x^8 = -27 * x^4 * (5 * x^4 - 2) := 
by 
  sorry

end factor_polynomial_l163_163187


namespace intersection_of_intervals_l163_163230

theorem intersection_of_intervals :
  let A := {x : ℝ | x < -3}
  let B := {x : ℝ | x > -4}
  A ∩ B = {x : ℝ | -4 < x ∧ x < -3} :=
by
  sorry

end intersection_of_intervals_l163_163230


namespace weight_of_B_l163_163016

-- Definitions for the weights
variables (A B C : ℝ)

-- Conditions from the problem
def avg_ABC : Prop := (A + B + C) / 3 = 45
def avg_AB : Prop := (A + B) / 2 = 40
def avg_BC : Prop := (B + C) / 2 = 43

-- The theorem to prove the weight of B
theorem weight_of_B (h1 : avg_ABC A B C) (h2 : avg_AB A B) (h3 : avg_BC B C) : B = 31 :=
sorry

end weight_of_B_l163_163016


namespace sqrt_factorial_mul_factorial_l163_163586

theorem sqrt_factorial_mul_factorial (n : ℕ) : 
  n = 4 → sqrt ((nat.factorial n) * (nat.factorial n)) = nat.factorial n :=
by
  intro h
  rw [h, nat.factorial, mul_self_sqrt (nat.factorial_nonneg 4)]

-- Note: While the final "mul_self_sqrt (nat.factorial_nonneg 4)" line is a sketch of the idea,
-- the proof is not complete as requested.

end sqrt_factorial_mul_factorial_l163_163586


namespace ratio_of_a_and_b_l163_163388

theorem ratio_of_a_and_b (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0)
  (h3 : (a * Real.sin (Real.pi / 7) + b * Real.cos (Real.pi / 7)) / 
        (a * Real.cos (Real.pi / 7) - b * Real.sin (Real.pi / 7)) = 
        Real.tan (10 * Real.pi / 21)) :
  b / a = Real.sqrt 3 :=
sorry

end ratio_of_a_and_b_l163_163388


namespace arithmetic_mean_of_primes_l163_163189

variable (list : List ℕ) 
variable (primes : List ℕ)
variable (h1 : list = [24, 25, 29, 31, 33])
variable (h2 : primes = [29, 31])

theorem arithmetic_mean_of_primes : (primes.sum / primes.length : ℝ) = 30 := by
  sorry

end arithmetic_mean_of_primes_l163_163189


namespace average_speed_ratio_l163_163338

def eddy_distance := 450 -- distance from A to B in km
def eddy_time := 3 -- time taken by Eddy in hours
def freddy_distance := 300 -- distance from A to C in km
def freddy_time := 4 -- time taken by Freddy in hours

def avg_speed (distance : ℕ) (time : ℕ) : ℕ := distance / time

def eddy_avg_speed := avg_speed eddy_distance eddy_time
def freddy_avg_speed := avg_speed freddy_distance freddy_time

def speed_ratio (speed1 : ℕ) (speed2 : ℕ) : ℕ × ℕ := (speed1 / (gcd speed1 speed2), speed2 / (gcd speed1 speed2))

theorem average_speed_ratio : speed_ratio eddy_avg_speed freddy_avg_speed = (2, 1) :=
by
  sorry

end average_speed_ratio_l163_163338


namespace isosceles_triangle_largest_angle_l163_163238

theorem isosceles_triangle_largest_angle (a b c : ℝ) 
  (h1 : a = b)
  (h2 : c + 50 + 50 = 180) : 
  c = 80 :=
by sorry

end isosceles_triangle_largest_angle_l163_163238


namespace exp_ineq_solution_set_l163_163661

theorem exp_ineq_solution_set (e : ℝ) (h : e = Real.exp 1) :
  {x : ℝ | e^(2*x - 1) < 1} = {x : ℝ | x < 1 / 2} :=
sorry

end exp_ineq_solution_set_l163_163661


namespace total_tiles_l163_163767

theorem total_tiles (n : ℕ) (h : 3 * n - 2 = 55) : n^2 = 361 :=
by
  sorry

end total_tiles_l163_163767


namespace determine_positive_intervals_l163_163479

noncomputable def positive_intervals (x : ℝ) : Prop :=
  (x+1) * (x-1) * (x-3) > 0

theorem determine_positive_intervals :
  ∀ x : ℝ, (positive_intervals x ↔ (x ∈ Set.Ioo (-1 : ℝ) (1 : ℝ) ∨ x ∈ Set.Ioi (3 : ℝ))) :=
by
  sorry

end determine_positive_intervals_l163_163479


namespace determine_wholesale_prices_l163_163161

-- Definitions and Conditions
-- Wholesale prices for days 1 and 2 are defined as p1 and p2
def wholesale_price_day1 : ℝ := p1
def wholesale_price_day2 : ℝ := p2

-- The price increment at "Baker Plus" store is a constant d
def price_increment_baker_plus : ℝ := d
-- The factor at "Star" store is a constant k > 1
def price_factor_star : ℝ := k
-- Given prices for two days: 64, 64, 70, 72 rubles
def given_prices : set ℝ := {64, 64, 70, 72}

-- Prices at the stores
def baker_plus_day1_price := wholesale_price_day1 + price_increment_baker_plus
def star_day1_price := wholesale_price_day1 * price_factor_star

def baker_plus_day2_price := wholesale_price_day2 + price_increment_baker_plus
def star_day2_price := wholesale_price_day2 * price_factor_star

-- Theorem to determine the wholesale prices for each day (p1 and p2)
theorem determine_wholesale_prices
    (baker_plus_day1_price ∈ given_prices)
    (star_day1_price ∈ given_prices)
    (baker_plus_day2_price ∈ given_prices)
    (star_day2_price ∈ given_prices)
    (h1 : baker_plus_day1_price ≠ star_day1_price)
    (h2 : baker_plus_day2_price ≠ star_day2_price) :
  (∃ p1 p2 d k > 1, 
      (p1 + d ∈ given_prices) ∧ (p1 * k ∈ given_prices) ∧
      (p2 + d ∈ given_prices) ∧ (p2 * k ∈ given_prices) ∧
      (p1 ≠ p2)) :=
  sorry

end determine_wholesale_prices_l163_163161


namespace complex_power_sum_l163_163697

noncomputable def z : ℂ := sorry

theorem complex_power_sum (hz : z^2 + z + 1 = 0) : 
  z^101 + z^102 + z^103 + z^104 + z^105 = -2 :=
sorry

end complex_power_sum_l163_163697


namespace avg_age_of_team_is_23_l163_163280

-- Conditions
def captain_age := 24
def wicket_keeper_age := captain_age + 7

def remaining_players_avg_age (team_avg_age : ℝ) := team_avg_age - 1
def total_team_age (team_avg_age : ℝ) := 11 * team_avg_age
def total_remaining_players_age (team_avg_age : ℝ) := 9 * remaining_players_avg_age team_avg_age

-- Proof statement
theorem avg_age_of_team_is_23 (team_avg_age : ℝ) :
  total_team_age team_avg_age = captain_age + wicket_keeper_age + total_remaining_players_age team_avg_age → 
  team_avg_age = 23 :=
by
  sorry

end avg_age_of_team_is_23_l163_163280


namespace expression_in_terms_of_p_q_l163_163864

-- Define the roots and the polynomials conditions
variable (α β γ δ : ℝ)
variable (p q : ℝ)

-- The conditions of the problem
axiom roots_poly1 : α * β = 1 ∧ α + β = -p
axiom roots_poly2 : γ * δ = 1 ∧ γ + δ = -q

theorem expression_in_terms_of_p_q :
  (α - γ) * (β - γ) * (α + δ) * (β + δ) = q^2 - p^2 :=
sorry

end expression_in_terms_of_p_q_l163_163864


namespace number_of_periods_l163_163026

-- Definitions based on conditions
def students : ℕ := 32
def time_per_student : ℕ := 5
def period_duration : ℕ := 40

-- Theorem stating the equivalent proof problem
theorem number_of_periods :
  (students * time_per_student) / period_duration = 4 :=
sorry

end number_of_periods_l163_163026


namespace difference_of_square_of_non_divisible_by_3_l163_163875

theorem difference_of_square_of_non_divisible_by_3 (n : ℕ) (h : ¬ (n % 3 = 0)) : (n^2 - 1) % 3 = 0 :=
sorry

end difference_of_square_of_non_divisible_by_3_l163_163875


namespace chemist_sons_ages_l163_163018

theorem chemist_sons_ages 
    (a b c w : ℕ)
    (h1 : a * b * c = 36)
    (h2 : a + b + c = w)
    (h3 : ∃! x, x = max a (max b c)) :
    (a = 2 ∧ b = 2 ∧ c = 9) ∨ 
    (a = 2 ∧ b = 9 ∧ c = 2) ∨ 
    (a = 9 ∧ b = 2 ∧ c = 2) :=
  sorry

end chemist_sons_ages_l163_163018


namespace sandy_correct_sums_l163_163274

theorem sandy_correct_sums :
  ∃ x y : ℕ, x + y = 30 ∧ 3 * x - 2 * y = 60 ∧ x = 24 :=
by
  sorry

end sandy_correct_sums_l163_163274


namespace transform_polynomial_l163_163076

theorem transform_polynomial (x y : ℝ) 
  (h1 : y = x + 1 / x) 
  (h2 : x^4 - x^3 - 2 * x^2 - x + 1 = 0) : x^2 * (y^2 - y - 4) = 0 :=
sorry

end transform_polynomial_l163_163076


namespace remaining_cubes_count_l163_163881

-- Define the initial number of cubes
def initial_cubes : ℕ := 64

-- Define the holes in the bottom layer
def holes_in_bottom_layer : ℕ := 6

-- Define the number of cubes removed per hole
def cubes_removed_per_hole : ℕ := 3

-- Define the calculation for missing cubes
def missing_cubes : ℕ := holes_in_bottom_layer * cubes_removed_per_hole

-- Define the calculation for remaining cubes
def remaining_cubes : ℕ := initial_cubes - missing_cubes

-- The theorem to prove
theorem remaining_cubes_count : remaining_cubes = 46 := by
  sorry

end remaining_cubes_count_l163_163881


namespace equal_number_of_frogs_after_6_months_l163_163182

theorem equal_number_of_frogs_after_6_months :
  ∃ n : ℕ, 
    n = 6 ∧ 
    (∀ Dn Qn : ℕ, 
      (Dn = 5^(n + 1) ∧ Qn = 3^(n + 5)) → 
      Dn = Qn) :=
by
  sorry

end equal_number_of_frogs_after_6_months_l163_163182


namespace percentage_markup_l163_163420

theorem percentage_markup (selling_price cost_price : ℝ) (h_selling : selling_price = 2000) (h_cost : cost_price = 1250) :
  ((selling_price - cost_price) / cost_price) * 100 = 60 := by
  sorry

end percentage_markup_l163_163420


namespace multiple_of_8_and_12_l163_163406

theorem multiple_of_8_and_12 (x y : ℤ) (hx : ∃ k : ℤ, x = 8 * k) (hy : ∃ k : ℤ, y = 12 * k) :
  (∃ k : ℤ, y = 4 * k) ∧ (∃ k : ℤ, x - y = 4 * k) :=
by
  /- Proof goes here, based on the given conditions -/
  sorry

end multiple_of_8_and_12_l163_163406


namespace least_positive_integer_l163_163906

theorem least_positive_integer (N : ℕ) :
  (N % 11 = 10) ∧
  (N % 12 = 11) ∧
  (N % 13 = 12) ∧
  (N % 14 = 13) ∧
  (N % 15 = 14) ∧
  (N % 16 = 15) →
  N = 720719 :=
by
  sorry

end least_positive_integer_l163_163906


namespace units_digit_n_l163_163342

theorem units_digit_n (m n : ℕ) (h₁ : m * n = 14^8) (hm : m % 10 = 6) : n % 10 = 1 :=
sorry

end units_digit_n_l163_163342


namespace value_at_17pi_over_6_l163_163727

variable (f : Real → Real)

-- Defining the conditions
def period (f : Real → Real) (T : Real) := ∀ x, f (x + T) = f x
def specific_value (f : Real → Real) (x : Real) (v : Real) := f x = v

-- The main theorem statement
theorem value_at_17pi_over_6 : 
  period f (π / 2) →
  specific_value f (π / 3) 1 →
  specific_value f (17 * π / 6) 1 :=
by
  intros h_period h_value
  sorry

end value_at_17pi_over_6_l163_163727


namespace length_of_ae_l163_163448

-- Define the given consecutive points
variables (a b c d e : ℝ)

-- Conditions from the problem
-- 1. Points a, b, c, d, e are 5 consecutive points on a straight line - implicitly assumed on the same line
-- 2. bc = 2 * cd
-- 3. de = 4
-- 4. ab = 5
-- 5. ac = 11

theorem length_of_ae 
  (h1 : b - a = 5) -- ab = 5
  (h2 : c - a = 11) -- ac = 11
  (h3 : c - b = 2 * (d - c)) -- bc = 2 * cd
  (h4 : e - d = 4) -- de = 4
  : (e - a) = 18 := sorry

end length_of_ae_l163_163448


namespace f_g_eq_g_f_iff_n_zero_l163_163524

def f (x n : ℝ) : ℝ := x + n
def g (x q : ℝ) : ℝ := x^2 + q

theorem f_g_eq_g_f_iff_n_zero (x n q : ℝ) : (f (g x q) n = g (f x n) q) ↔ n = 0 := by 
  sorry

end f_g_eq_g_f_iff_n_zero_l163_163524


namespace average_score_for_girls_l163_163926

variable (A a B b : ℕ)
variable (h1 : 71 * A + 76 * a = 74 * (A + a))
variable (h2 : 81 * B + 90 * b = 84 * (B + b))
variable (h3 : 71 * A + 81 * B = 79 * (A + B))

theorem average_score_for_girls
  (h1 : 71 * A + 76 * a = 74 * (A + a))
  (h2 : 81 * B + 90 * b = 84 * (B + b))
  (h3 : 71 * A + 81 * B = 79 * (A + B))
  : (76 * a + 90 * b) / (a + b) = 84 := by
  sorry

end average_score_for_girls_l163_163926


namespace budget_spent_on_research_and_development_l163_163760

theorem budget_spent_on_research_and_development:
  (∀ budget_total : ℝ, budget_total > 0) →
  (∀ transportation : ℝ, transportation = 15) →
  (∃ research_and_development : ℝ, research_and_development ≥ 0) →
  (∀ utilities : ℝ, utilities = 5) →
  (∀ equipment : ℝ, equipment = 4) →
  (∀ supplies : ℝ, supplies = 2) →
  (∀ salaries_degrees : ℝ, salaries_degrees = 234) →
  (∀ total_degrees : ℝ, total_degrees = 360) →
  (∀ percentage_salaries : ℝ, percentage_salaries = (salaries_degrees / total_degrees) * 100) →
  (∀ known_percentages : ℝ, known_percentages = transportation + utilities + equipment + supplies + percentage_salaries) →
  (∀ rnd_percent : ℝ, rnd_percent = 100 - known_percentages) →
  (rnd_percent = 9) :=
  sorry

end budget_spent_on_research_and_development_l163_163760


namespace quadrilateral_bisector_intersection_p_q_r_s_sum_eq_176_l163_163089

structure Point where
  x : ℚ
  y : ℚ

def A : Point := { x := 0, y := 0 }
def B : Point := { x := 2, y := 3 }
def C : Point := { x := 5, y := 4 }
def D : Point := { x := 6, y := 1 }

def line_eq_y_eq_kx_plus_b (k b x : ℚ) : ℚ := k * x + b

def intersects (A : Point) (P : Point × Point) (x y : ℚ) : Prop :=
  ∃ k b, P.1.y = line_eq_y_eq_kx_plus_b k b P.1.x ∧ P.2.y = line_eq_y_eq_kx_plus_b k b P.2.x ∧
         y = line_eq_y_eq_kx_plus_b k b x

theorem quadrilateral_bisector_intersection_p_q_r_s_sum_eq_176 :
  ∃ (p q r s : ℚ), 
    gcd p q = 1 ∧ gcd r s = 1 ∧ intersects A (C, D) (p / q) (r / s) ∧
    (p + q + r + s = 176) :=
sorry

end quadrilateral_bisector_intersection_p_q_r_s_sum_eq_176_l163_163089


namespace primes_satisfying_equation_l163_163803

theorem primes_satisfying_equation :
  ∀ (p q : ℕ), p.Prime ∧ q.Prime → 
    (∃ x y z : ℕ, p^(2*x) + q^(2*y) = z^2) ↔ 
    (p = 2 ∧ q = 3) ∨ (p = 3 ∧ q = 2) := 
by
  sorry

end primes_satisfying_equation_l163_163803


namespace teta_beta_gamma_l163_163208

theorem teta_beta_gamma : 
  ∃ T E T' A B E' T'' A' G A'' M M' A''' A'''' : ℕ, 
  TETA = T * 1000 + E * 100 + T' * 10 + A ∧ 
  BETA = B * 1000 + E' * 100 + T'' * 10 + A' ∧ 
  GAMMA = G * 10000 + A'' * 1000 + M * 100 + M' * 10 + A''' ∧
  TETA + BETA = GAMMA ∧ 
  A = A'''' ∧ E = E' ∧ T = T' ∧ T' = T'' ∧ A = A' ∧ A = A'' ∧ A = A''' ∧ M = M' ∧ 
  T ≠ E ∧ T ≠ A ∧ T ≠ B ∧ T ≠ G ∧ T ≠ M ∧
  E ≠ A ∧ E ≠ B ∧ E ≠ G ∧ E ≠ M ∧
  A ≠ B ∧ A ≠ G ∧ A ≠ M ∧
  B ≠ G ∧ B ≠ M ∧
  G ≠ M ∧
  TETA = 4940 ∧ BETA = 5940 ∧ GAMMA = 10880
  :=
sorry

end teta_beta_gamma_l163_163208


namespace correct_statements_l163_163745

-- Define the propositions p and q
variables (p q : Prop)

-- Define the given statements as logical conditions
def statement1 := (p ∧ q) → (p ∨ q)
def statement2 := ¬(p ∧ q) → (p ∨ q)
def statement3 := (p ∨ q) ↔ ¬¬p
def statement4 := (¬p) → ¬(p ∧ q)

-- Define the proof problem
theorem correct_statements :
  ((statement1 p q) ∧ (¬statement2 p q) ∧ (statement3 p q) ∧ (¬statement4 p q)) :=
by {
  -- Here you would prove that
  -- statement1 is correct,
  -- statement2 is incorrect,
  -- statement3 is correct,
  -- statement4 is incorrect
  sorry
}

end correct_statements_l163_163745


namespace student_scores_correct_answers_l163_163236

variable (c w : ℕ)

theorem student_scores_correct_answers :
  (c + w = 60) ∧ (4 * c - w = 130) → c = 38 :=
by
  intro h
  sorry

end student_scores_correct_answers_l163_163236


namespace transport_cost_is_correct_l163_163408

-- Define the transport cost per kilogram
def transport_cost_per_kg : ℝ := 18000

-- Define the weight of the scientific instrument in kilograms
def weight_kg : ℝ := 0.5

-- Define the discount rate
def discount_rate : ℝ := 0.10

-- Define the cost calculation without the discount
def cost_without_discount : ℝ := weight_kg * transport_cost_per_kg

-- Define the final cost with the discount applied
def discounted_cost : ℝ := cost_without_discount * (1 - discount_rate)

-- The theorem stating that the discounted cost is $8,100
theorem transport_cost_is_correct : discounted_cost = 8100 := by
  sorry

end transport_cost_is_correct_l163_163408


namespace correct_product_l163_163480

-- We define the conditions
def number1 : ℝ := 0.85
def number2 : ℝ := 3.25
def without_decimal_points_prod : ℕ := 27625

-- We state the problem
theorem correct_product (h1 : (85 : ℕ) * (325 : ℕ) = without_decimal_points_prod)
                        (h2 : number1 * number2 * 10000 = (without_decimal_points_prod : ℝ)) :
  number1 * number2 = 2.7625 :=
by sorry

end correct_product_l163_163480


namespace product_of_two_numbers_eq_a_mul_100_a_l163_163682

def product_of_two_numbers (a : ℝ) (b : ℝ) : ℝ := a * b

theorem product_of_two_numbers_eq_a_mul_100_a (a : ℝ) (b : ℝ) (h : a + b = 100) :
    product_of_two_numbers a b = a * (100 - a) :=
by
  sorry

end product_of_two_numbers_eq_a_mul_100_a_l163_163682


namespace samson_fuel_calculation_l163_163535

def total_fuel_needed (main_distance : ℕ) (fuel_rate : ℕ) (hilly_distance : ℕ) (hilly_increase : ℚ)
                      (detours : ℕ) (detour_distance : ℕ) : ℚ :=
  let normal_distance := main_distance - hilly_distance
  let normal_fuel := (fuel_rate / 70) * normal_distance
  let hilly_fuel := (fuel_rate / 70) * hilly_distance * hilly_increase
  let detour_fuel := (fuel_rate / 70) * (detours * detour_distance)
  normal_fuel + hilly_fuel + detour_fuel

theorem samson_fuel_calculation :
  total_fuel_needed 140 10 30 1.2 2 5 = 22.28 :=
by sorry

end samson_fuel_calculation_l163_163535


namespace trams_required_l163_163551

theorem trams_required (initial_trams : ℕ) (initial_interval : ℚ) (reduction_fraction : ℚ) :
  initial_trams = 12 ∧ initial_interval = 5 ∧ reduction_fraction = 1/5 →
  (initial_trams + initial_trams * reduction_fraction - initial_trams) = 3 :=
by
  sorry

end trams_required_l163_163551


namespace range_of_a_l163_163970

noncomputable def f (x : ℝ) (k a : ℝ) :=
  if -1 ≤ x ∧ x < k then
    log 2 (1 - x) + 1
  else if k ≤ x ∧ x ≤ a then
    x ^ 2 - 2 * x + 1
  else
    0
-- our function

theorem range_of_a (k a : ℝ) (hk : -1 ≤ k) (ha : k ≤ a) :
  (∃ (k : ℝ), range f x k a = set.Icc 0 2) ↔ a ∈ set.Ioc (1 / 2) (1 + real.sqrt 2) :=
by
  sorry

end range_of_a_l163_163970


namespace edit_post_time_zero_l163_163877

-- Define the conditions
def total_videos : ℕ := 4
def setup_time : ℕ := 1
def painting_time_per_video : ℕ := 1
def cleanup_time : ℕ := 1
def total_production_time_per_video : ℕ := 3

-- Define the total time spent on setup, painting, and cleanup for one video
def spc_time : ℕ := setup_time + painting_time_per_video + cleanup_time

-- State the theorem to be proven
theorem edit_post_time_zero : (total_production_time_per_video - spc_time) = 0 := by
  sorry

end edit_post_time_zero_l163_163877


namespace min_value_of_squares_l163_163423

variable (a b t : ℝ)

theorem min_value_of_squares (ht : 0 < t) (habt : a + b = t) : 
  a^2 + b^2 ≥ t^2 / 2 := 
by
  sorry

end min_value_of_squares_l163_163423


namespace emily_lives_total_l163_163703

variable (x : ℤ)

def total_lives_after_stages (x : ℤ) : ℤ :=
  let lives_after_stage1 := x + 25
  let lives_after_stage2 := lives_after_stage1 + 24
  let lives_after_stage3 := lives_after_stage2 + 15
  lives_after_stage3

theorem emily_lives_total : total_lives_after_stages x = x + 64 := by
  -- The proof will go here
  sorry

end emily_lives_total_l163_163703


namespace largest_possible_number_of_markers_l163_163872

theorem largest_possible_number_of_markers (n_m n_c : ℕ) 
  (h_m : n_m = 72) (h_c : n_c = 48) : Nat.gcd n_m n_c = 24 :=
by
  sorry

end largest_possible_number_of_markers_l163_163872


namespace solve_quadratic_inequality_l163_163485

-- To express that a real number x is in the interval (0, 2)
def in_interval (x : ℝ) : Prop := 0 < x ∧ x < 2

theorem solve_quadratic_inequality :
  { x : ℝ | x^2 < 2 * x } = { x : ℝ | in_interval x } :=
by
  sorry

end solve_quadratic_inequality_l163_163485


namespace negation_proposition_p_l163_163209

open Classical

variable (n : ℕ)

def proposition_p : Prop := ∃ n : ℕ, 2^n > 100

theorem negation_proposition_p : ¬ proposition_p ↔ ∀ n : ℕ, 2^n ≤ 100 := 
by sorry

end negation_proposition_p_l163_163209


namespace prob_E_given_D_l163_163037

variable (E L D : Event ω) -- Define the events

-- Given conditions:
variable (prob_E : ℙ E = 0.2)
variable (prob_L_given_not_E : ℙ (L | Eᶜ) = 0.25)
variable [IsProbabilityMeasure ℙ]

theorem prob_E_given_D : ℙ (E | D) = 0.5 :=
by
  -- Calculation of total probability of death event
  
  sorry -- Proof is omitted as per instruction.

end prob_E_given_D_l163_163037


namespace digging_project_length_l163_163310

theorem digging_project_length (Length_2 : ℝ) : 
  (100 * 25 * 30) = (75 * Length_2 * 50) → 
  Length_2 = 20 :=
by
  sorry

end digging_project_length_l163_163310


namespace claire_gerbils_l163_163937

variables (G H : ℕ)

-- Claire's total pets
def total_pets : Prop := G + H = 92

-- One-quarter of the gerbils are male
def male_gerbils (G : ℕ) : ℕ := G / 4

-- One-third of the hamsters are male
def male_hamsters (H : ℕ) : ℕ := H / 3

-- Total males are 25
def total_males : Prop := male_gerbils G + male_hamsters H = 25

theorem claire_gerbils : total_pets G H → total_males G H → G = 68 :=
by
  intro h1 h2
  sorry

end claire_gerbils_l163_163937


namespace joan_remaining_balloons_l163_163854

def initial_balloons : ℕ := 9
def lost_balloons : ℕ := 2
def remaining_balloons : ℕ := initial_balloons - lost_balloons

theorem joan_remaining_balloons : remaining_balloons = 7 := by
  sorry

end joan_remaining_balloons_l163_163854


namespace inequality_system_solution_range_l163_163229

theorem inequality_system_solution_range (x m : ℝ) :
  (∃ x : ℝ, (x + 1) / 2 < x / 3 + 1 ∧ x > 3 * m) → m < 1 :=
by
  sorry

end inequality_system_solution_range_l163_163229


namespace number_of_balls_to_remove_l163_163921

theorem number_of_balls_to_remove:
  ∀ (x : ℕ), 120 - x = (48 : ℕ) / (0.75 : ℝ) → x = 56 :=
by sorry

end number_of_balls_to_remove_l163_163921


namespace kevin_found_cards_l163_163250

-- Definitions from the conditions
def initial_cards : ℕ := 7
def final_cards : ℕ := 54

-- The proof goal
theorem kevin_found_cards : final_cards - initial_cards = 47 :=
by
  sorry

end kevin_found_cards_l163_163250


namespace probability_red_prime_green_even_correct_l163_163903

/-- Conditions -/
def dice_sides : set ℕ := {1, 2, 3, 4, 5, 6}
def prime_numbers : set ℕ := {2, 3, 5}
def even_numbers : set ℕ := {2, 4, 6}

noncomputable def probability_red_prime_green_even : ℚ :=
  let total_outcomes := 36 in
  let successful_outcomes_red := (prime_numbers ∩ dice_sides).card in
  let successful_outcomes_green := (even_numbers ∩ dice_sides).card in
  let successful_outcomes := successful_outcomes_red * successful_outcomes_green in
  successful_outcomes / total_outcomes

/-- Theorem statement -/
theorem probability_red_prime_green_even_correct :
  probability_red_prime_green_even = 1 / 4 := by
  sorry

end probability_red_prime_green_even_correct_l163_163903


namespace one_three_digit_cube_divisible_by_16_l163_163071

theorem one_three_digit_cube_divisible_by_16 :
  ∃! (n : ℕ), (100 ≤ n ∧ n < 1000 ∧ ∃ (k : ℕ), n = k^3 ∧ 16 ∣ n) :=
sorry

end one_three_digit_cube_divisible_by_16_l163_163071


namespace rainfall_comparison_l163_163266

-- Define the conditions
def rainfall_mondays (n_mondays : ℕ) (rain_monday : ℝ) : ℝ :=
  n_mondays * rain_monday

def rainfall_tuesdays (n_tuesdays : ℕ) (rain_tuesday : ℝ) : ℝ :=
  n_tuesdays * rain_tuesday

def rainfall_difference (total_monday : ℝ) (total_tuesday : ℝ) : ℝ :=
  total_tuesday - total_monday

-- The proof statement
theorem rainfall_comparison :
  rainfall_difference (rainfall_mondays 13 1.75) (rainfall_tuesdays 16 2.65) = 19.65 := by
  sorry

end rainfall_comparison_l163_163266


namespace loan_payment_period_years_l163_163518

noncomputable def house_cost := 480000
noncomputable def trailer_cost := 120000
noncomputable def monthly_difference := 1500

theorem loan_payment_period_years:
  ∃ N : ℕ, (house_cost = (trailer_cost / N + monthly_difference) * N ∧
            N = 240) →
            N / 12 = 20 :=
sorry

end loan_payment_period_years_l163_163518


namespace total_fuel_usage_is_250_l163_163245

-- Define John's fuel consumption per km
def fuel_consumption_per_km : ℕ := 5

-- Define the distance of the first trip
def distance_trip1 : ℕ := 30

-- Define the distance of the second trip
def distance_trip2 : ℕ := 20

-- Define the fuel usage calculation
def fuel_usage_trip1 := distance_trip1 * fuel_consumption_per_km
def fuel_usage_trip2 := distance_trip2 * fuel_consumption_per_km
def total_fuel_usage := fuel_usage_trip1 + fuel_usage_trip2

-- Prove that the total fuel usage is 250 liters
theorem total_fuel_usage_is_250 : total_fuel_usage = 250 := by
  sorry

end total_fuel_usage_is_250_l163_163245


namespace two_zeros_of_cubic_polynomial_l163_163822

theorem two_zeros_of_cubic_polynomial (m : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ -x1^3 + 3*x1 + m = 0 ∧ -x2^3 + 3*x2 + m = 0) →
  (m = -2 ∨ m = 2) :=
by
  sorry

end two_zeros_of_cubic_polynomial_l163_163822


namespace jordan_travel_distance_heavy_traffic_l163_163691

theorem jordan_travel_distance_heavy_traffic (x : ℝ) (h1 : x / 20 + x / 10 + x / 6 = 7 / 6) : 
  x = 3.7 :=
by
  sorry

end jordan_travel_distance_heavy_traffic_l163_163691


namespace laptop_price_difference_l163_163047

theorem laptop_price_difference :
  let list_price := 59.99
  let tech_bargains_discount := 15
  let budget_bytes_discount_percentage := 0.30
  let tech_bargains_price := list_price - tech_bargains_discount
  let budget_bytes_price := list_price * (1 - budget_bytes_discount_percentage)
  let cheaper_price := min tech_bargains_price budget_bytes_price
  let expensive_price := max tech_bargains_price budget_bytes_price
  (expensive_price - cheaper_price) * 100 = 300 :=
by
  sorry

end laptop_price_difference_l163_163047


namespace eight_times_10x_plus_14pi_l163_163679

theorem eight_times_10x_plus_14pi (x : ℝ) (Q : ℝ) (h : 4 * (5 * x + 7 * π) = Q) : 
  8 * (10 * x + 14 * π) = 4 * Q := 
by {
  sorry  -- proof is omitted
}

end eight_times_10x_plus_14pi_l163_163679


namespace distance_between_lines_is_sqrt2_l163_163114

noncomputable def distance_between_parallel_lines (a b c1 c2 : ℝ) : ℝ :=
  |c1 - c2| / Real.sqrt (a^2 + b^2)

theorem distance_between_lines_is_sqrt2 :
  distance_between_parallel_lines 1 1 (-1) 1 = Real.sqrt 2 := 
by 
  sorry

end distance_between_lines_is_sqrt2_l163_163114


namespace train_length_l163_163459

theorem train_length 
  (speed_jogger_kmph : ℕ)
  (initial_distance_m : ℕ)
  (speed_train_kmph : ℕ)
  (pass_time_s : ℕ)
  (h_speed_jogger : speed_jogger_kmph = 9)
  (h_initial_distance : initial_distance_m = 230)
  (h_speed_train : speed_train_kmph = 45)
  (h_pass_time : pass_time_s = 35) : 
  ∃ length_train_m : ℕ, length_train_m = 580 := sorry

end train_length_l163_163459


namespace simple_interest_rate_l163_163915

variables (P R T SI : ℝ)

theorem simple_interest_rate :
  T = 10 →
  SI = (2 / 5) * P →
  SI = (P * R * T) / 100 →
  R = 4 :=
by
  intros hT hSI hFormula
  sorry

end simple_interest_rate_l163_163915


namespace sam_morning_run_distance_l163_163715

variable (x : ℝ) -- The distance of Sam's morning run in miles

theorem sam_morning_run_distance (h1 : ∀ y, y = 2 * x) (h2 : 12 = 12) (h3 : x + 2 * x + 12 = 18) : x = 2 :=
by sorry

end sam_morning_run_distance_l163_163715


namespace oil_bill_january_l163_163005

-- Declare the constants for January and February oil bills
variables (J F : ℝ)

-- State the conditions
def condition_1 : Prop := F / J = 3 / 2
def condition_2 : Prop := (F + 20) / J = 5 / 3

-- State the theorem based on the conditions and the target statement
theorem oil_bill_january (h1 : condition_1 F J) (h2 : condition_2 F J) : J = 120 :=
by
  sorry

end oil_bill_january_l163_163005


namespace helen_needed_gas_l163_163829

-- Definitions based on conditions
def cuts_per_month_routine_1 : ℕ := 2 -- Cuts per month for March, April, September, October
def cuts_per_month_routine_2 : ℕ := 4 -- Cuts per month for May, June, July, August
def months_routine_1 : ℕ := 4 -- Number of months with routine 1
def months_routine_2 : ℕ := 4 -- Number of months with routine 2
def gas_per_fill : ℕ := 2 -- Gallons of gas used every 4th cut
def cuts_per_fill : ℕ := 4 -- Number of cuts per fill

-- Total number of cuts in routine 1 months
def total_cuts_routine_1 : ℕ := cuts_per_month_routine_1 * months_routine_1

-- Total number of cuts in routine 2 months
def total_cuts_routine_2 : ℕ := cuts_per_month_routine_2 * months_routine_2

-- Total cuts from March to October
def total_cuts : ℕ := total_cuts_routine_1 + total_cuts_routine_2

-- Total fills needed from March to October
def total_fills : ℕ := total_cuts / cuts_per_fill

-- Total gallons of gas needed
def total_gal_of_gas : ℕ := total_fills * gas_per_fill

-- The statement to prove
theorem helen_needed_gas : total_gal_of_gas = 12 :=
by
  -- This would be replaced by our solution steps.
  sorry

end helen_needed_gas_l163_163829


namespace area_of_smaller_circle_l163_163295

noncomputable def radius_smaller_circle : ℝ := sorry
noncomputable def radius_larger_circle : ℝ := 3 * radius_smaller_circle

-- Given: PA = AB = 5
def PA : ℝ := 5
def AB : ℝ := 5

-- Final goal: The area of the smaller circle is 5/3 * π
theorem area_of_smaller_circle (r_s : ℝ) (rsq : r_s^2 = 5 / 3) : (π * r_s^2 = 5/3 * π) :=
by
  exact sorry

end area_of_smaller_circle_l163_163295


namespace initial_girls_count_l163_163664

-- Define the variables
variables (b g : ℕ)

-- Conditions
def condition1 := b = 3 * (g - 20)
def condition2 := 4 * (b - 60) = g - 20

-- Statement of the problem
theorem initial_girls_count
  (h1 : condition1 b g)
  (h2 : condition2 b g) : g = 460 / 11 := 
sorry

end initial_girls_count_l163_163664


namespace gcd_2_pow_1025_sub_1_and_2_pow_1056_sub_1_l163_163751

def a : ℕ := 2^1025 - 1
def b : ℕ := 2^1056 - 1
def answer : ℕ := 2147483647

theorem gcd_2_pow_1025_sub_1_and_2_pow_1056_sub_1 :
  Int.gcd a b = answer := by
  sorry

end gcd_2_pow_1025_sub_1_and_2_pow_1056_sub_1_l163_163751


namespace school_bus_solution_l163_163786

-- Define the capacities
def bus_capacity : Prop := 
  ∃ x y : ℕ, x + y = 75 ∧ 3 * x + 2 * y = 180 ∧ x = 30 ∧ y = 45

-- Define the rental problem
def rental_plans : Prop :=
  ∃ a : ℕ, 6 ≤ a ∧ a ≤ 8 ∧ 
  (30 * a + 45 * (25 - a) ≥ 1000) ∧ 
  (320 * a + 400 * (25 - a) ≤ 9550) ∧ 
  3 = 3

-- The main theorem combines the two aspects
theorem school_bus_solution: bus_capacity ∧ rental_plans := 
  sorry -- Proof omitted

end school_bus_solution_l163_163786


namespace sum_non_solutions_l163_163087

theorem sum_non_solutions (A B C : ℝ) (h : ∀ x, (x + B) * (A * x + 36) = 3 * (x + C) * (x + 9) → x ≠ -12) :
  -12 = -12 := 
sorry

end sum_non_solutions_l163_163087


namespace S_40_value_l163_163422

variable {a : ℕ → ℝ} {S : ℕ → ℝ}

axiom h1 : S 10 = 10
axiom h2 : S 30 = 70

theorem S_40_value : S 40 = 150 :=
by
  -- Conditions
  have h1 : S 10 = 10 := h1
  have h2 : S 30 = 70 := h2
  -- Start proof here
  sorry

end S_40_value_l163_163422


namespace range_of_a_part1_range_of_a_part2_l163_163356

def set_A (x : ℝ) : Prop := -1 < x ∧ x < 6

def set_B (x : ℝ) (a : ℝ) : Prop := (x ≥ 1 + a) ∨ (x ≤ 1 - a)

def condition_1 (a : ℝ) : Prop :=
  (∀ x, set_A x → ¬ set_B x a) → (a ≥ 5)

def condition_2 (a : ℝ) : Prop :=
  (∀ x, (x ≥ 6 ∨ x ≤ -1) → set_B x a) ∧ (∃ x, set_B x a ∧ ¬ (x ≥ 6 ∨ x ≤ -1)) → (0 < a ∧ a ≤ 2)

theorem range_of_a_part1 (a : ℝ) : condition_1 a :=
  sorry

theorem range_of_a_part2 (a : ℝ) : condition_2 a :=
  sorry

end range_of_a_part1_range_of_a_part2_l163_163356


namespace sum_of_three_positives_eq_2002_l163_163678

theorem sum_of_three_positives_eq_2002 : 
  ∃ (n : ℕ), n = 334000 ∧ (∃ (f : ℕ → ℕ → ℕ → Prop), 
    (∀ (A B C : ℕ), f A B C ↔ (0 < A ∧ A ≤ B ∧ B ≤ C ∧ A + B + C = 2002))) := by
  sorry

end sum_of_three_positives_eq_2002_l163_163678


namespace original_price_l163_163086

-- Definitions of conditions
def SalePrice : Float := 70
def DecreasePercentage : Float := 30

-- Statement to prove
theorem original_price (P : Float) (h : 0.70 * P = SalePrice) : P = 100 := by
  sorry

end original_price_l163_163086


namespace total_full_parking_spots_correct_l163_163032

-- Define the number of parking spots on each level
def total_parking_spots (level : ℕ) : ℕ :=
  100 + (level - 1) * 50

-- Define the number of open spots on each level
def open_parking_spots (level : ℕ) : ℕ :=
  if level = 1 then 58
  else if level <= 4 then 58 - 3 * (level - 1)
  else 49 + 10 * (level - 4)

-- Define the number of full parking spots on each level
def full_parking_spots (level : ℕ) : ℕ :=
  total_parking_spots level - open_parking_spots level

-- Sum up the full parking spots on all 7 levels to get the total full spots
def total_full_parking_spots : ℕ :=
  List.sum (List.map full_parking_spots [1, 2, 3, 4, 5, 6, 7])

-- Theorem to prove the total number of full parking spots
theorem total_full_parking_spots_correct : total_full_parking_spots = 1329 :=
by
  sorry

end total_full_parking_spots_correct_l163_163032


namespace sum_of_integers_l163_163134

theorem sum_of_integers (a b c : ℕ) :
  a > 1 → b > 1 → c > 1 →
  a * b * c = 1728 →
  gcd a b = 1 → gcd b c = 1 → gcd a c = 1 →
  a + b + c = 43 :=
by
  intro ha
  intro hb
  intro hc
  intro hproduct
  intro hgcd_ab
  intro hgcd_bc
  intro hgcd_ac
  sorry

end sum_of_integers_l163_163134


namespace right_triangle_sides_l163_163461

theorem right_triangle_sides 
  (a b c : ℝ) 
  (h_right_angle : a^2 + b^2 = c^2) 
  (h_area : (1 / 2) * a * b = 150) 
  (h_perimeter : a + b + c = 60) 
  : (a = 15 ∧ b = 20 ∧ c = 25) ∨ (a = 20 ∧ b = 15 ∧ c = 25) :=
by
  sorry

end right_triangle_sides_l163_163461


namespace greatest_integer_difference_l163_163355

theorem greatest_integer_difference (x y : ℤ) (hx : -6 < (x : ℝ)) (hx2 : (x : ℝ) < -2) (hy : 4 < (y : ℝ)) (hy2 : (y : ℝ) < 10) : 
  ∃ d : ℤ, d = y - x ∧ d = 14 := 
by
  sorry

end greatest_integer_difference_l163_163355


namespace fibonacci_recurrence_l163_163333

theorem fibonacci_recurrence (f : ℕ → ℝ) (a b : ℝ) 
  (h₀ : f 0 = 1) 
  (h₁ : f 1 = 1) 
  (h₂ : ∀ n, f (n + 2) = f (n + 1) + f n)
  (h₃ : a + b = 1) 
  (h₄ : a * b = -1) 
  (h₅ : a > b) 
  : ∀ n, f n = (a ^ (n + 1) - b ^ (n + 1)) / Real.sqrt 5 := by
  sorry

end fibonacci_recurrence_l163_163333


namespace sqrt_factorial_mul_factorial_eq_l163_163594

theorem sqrt_factorial_mul_factorial_eq :
  Real.sqrt (Nat.factorial 4 * Nat.factorial 4) = 24 := by
sorry

end sqrt_factorial_mul_factorial_eq_l163_163594


namespace add_trams_l163_163558

theorem add_trams (total_trams : ℕ) (total_distance : ℝ) (initial_intervals : ℝ) (new_intervals : ℝ) (additional_trams : ℕ) :
  total_trams = 12 → total_distance = 60 → initial_intervals = total_distance / total_trams →
  new_intervals = initial_intervals - (initial_intervals / 5) →
  additional_trams = (total_distance / new_intervals) - total_trams →
  additional_trams = 3 :=
begin
  intros h1 h2 h3 h4 h5,
  sorry
end

end add_trams_l163_163558


namespace g_neither_even_nor_odd_l163_163517

noncomputable def g (x : ℝ) : ℝ := ⌊x⌋ + 1/2 + Real.sin x

theorem g_neither_even_nor_odd : ¬(∀ x, g x = g (-x)) ∧ ¬(∀ x, g x = -g (-x)) := sorry

end g_neither_even_nor_odd_l163_163517


namespace Martiza_study_time_l163_163705

theorem Martiza_study_time :
  ∀ (x : ℕ),
  (30 * x + 30 * 25 = 20 * 60) →
  x = 15 :=
by
  intros x h
  sorry

end Martiza_study_time_l163_163705


namespace solve_system_of_equations_l163_163538

theorem solve_system_of_equations (x y z : ℝ) :
  (x * y + 1 = 2 * z) →
  (y * z + 1 = 2 * x) →
  (z * x + 1 = 2 * y) →
  (x = 1 ∧ y = 1 ∧ z = 1) ∨ 
  ((x = -2 ∧ y = -2 ∧ z = 5/2) ∨
   (x = 5/2 ∧ y = -2 ∧ z = -2) ∨ 
   (x = -2 ∧ y = 5/2 ∧ z = -2)) :=
sorry

end solve_system_of_equations_l163_163538


namespace sqrt_factorial_product_l163_163584

/-- Define the factorial of 4 -/
def factorial_four : ℕ := 4!

/-- Define the product of the factorial of 4 with itself -/
def product_of_factorials : ℕ := factorial_four * factorial_four

/-- Prove the value of the square root of product_of_factorials is 24 -/
theorem sqrt_factorial_product : Real.sqrt (product_of_factorials) = 24 := by
  have fact_4_eq_24 : factorial_four = 24 := by norm_num
  rw [product_of_factorials, fact_4_eq_24, Nat.mul_self_eq, Real.sqrt_sq]
  norm_num
  exact Nat.zero_le 24

end sqrt_factorial_product_l163_163584


namespace part1_minimum_b_over_a_l163_163361

noncomputable def f (x a : ℝ) : ℝ := Real.log x - a * x

-- Prove part 1
theorem part1 (x : ℝ) : (0 < x ∧ x < 1 → (f x 1 / (1/x - 1) > 0)) ∧ (1 < x → (f x 1 / (1/x - 1) < 0)) := sorry

-- Prove part 2
lemma part2 (a b : ℝ) (h : ∀ x > 0, f x a ≤ b - a) (ha : a ≠ 0) : ∃ x > 0, f x a = b - a := sorry

theorem minimum_b_over_a (a : ℝ) (ha : a ≠ 0) (h : ∀ x > 0, f x a ≤ b - a) : b/a ≥ 0 := sorry

end part1_minimum_b_over_a_l163_163361


namespace sqrt_factorial_product_l163_163611

theorem sqrt_factorial_product:
  (Int.sqrt (Nat.factorial 4 * Nat.factorial 4)).toNat = 24 :=
by
  sorry

end sqrt_factorial_product_l163_163611


namespace decreased_cost_proof_l163_163730

def original_cost : ℝ := 200
def percentage_decrease : ℝ := 0.5
def decreased_cost (original_cost : ℝ) (percentage_decrease : ℝ) : ℝ := 
  original_cost - (percentage_decrease * original_cost)

theorem decreased_cost_proof : decreased_cost original_cost percentage_decrease = 100 := 
by { 
  sorry -- Proof is not required
}

end decreased_cost_proof_l163_163730


namespace expression_not_defined_l163_163345

theorem expression_not_defined (x : ℝ) :
    ¬(x^2 - 22*x + 121 = 0) ↔ ¬(x - 11 = 0) :=
by sorry

end expression_not_defined_l163_163345


namespace rebecca_income_percentage_l163_163271

-- Define Rebecca's initial income
def rebecca_initial_income : ℤ := 15000
-- Define Jimmy's income
def jimmy_income : ℤ := 18000
-- Define the increase in Rebecca's income
def rebecca_income_increase : ℤ := 7000

-- Define the new income for Rebecca after increase
def rebecca_new_income : ℤ := rebecca_initial_income + rebecca_income_increase
-- Define the new combined income
def new_combined_income : ℤ := rebecca_new_income + jimmy_income

-- State the theorem to prove that Rebecca's new income is 55% of the new combined income
theorem rebecca_income_percentage : 
  (rebecca_new_income * 100) / new_combined_income = 55 :=
sorry

end rebecca_income_percentage_l163_163271


namespace area_ratio_of_circles_l163_163681

theorem area_ratio_of_circles 
  (CX : ℝ)
  (CY : ℝ)
  (RX RY : ℝ)
  (hX : CX = 2 * π * RX)
  (hY : CY = 2 * π * RY)
  (arc_length_equality : (90 / 360) * CX = (60 / 360) * CY) :
  (π * RX^2) / (π * RY^2) = 9 / 4 :=
by
  sorry

end area_ratio_of_circles_l163_163681


namespace max_ways_to_ascend_and_descend_l163_163132

theorem max_ways_to_ascend_and_descend :
  let east := 2
  let west := 3
  let south := 4
  let north := 1
  let ascend_descend_ways (ascend: ℕ) (n_1 n_2 n_3: ℕ) := ascend * (n_1 + n_2 + n_3)
  (ascend_descend_ways south east west north > ascend_descend_ways east west south north) ∧ 
  (ascend_descend_ways south east west north > ascend_descend_ways west east south north) ∧ 
  (ascend_descend_ways south east west north > ascend_descend_ways north east west south) := sorry

end max_ways_to_ascend_and_descend_l163_163132


namespace probability_Laurent_greater_Chloe_l163_163777

-- Define the problem conditions
def Chloe_distribution : MeasureTheory.Probability.RV ℝ :=
  MeasureTheory.Probability.uniform ω (0, 3000)

def Laurent_distribution : MeasureTheory.Probability.RV ℝ :=
  MeasureTheory.Probability.uniform ω (0, 6000)

-- Define the main theorem we want to prove
theorem probability_Laurent_greater_Chloe :
  @MeasureTheory.probability ℝ ℝ _ Laurent_distribution (λ y, @MeasureTheory.Probability.has_support _ _ Chloe_distribution (λ x, y > x)) = 3 / 4 :=
by sorry -- proof to be done

end probability_Laurent_greater_Chloe_l163_163777


namespace matrix_B_pow48_l163_163251

open Matrix

def B : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![0, 0, 0], ![0, 0, 2], ![0, -2, 0]]

theorem matrix_B_pow48 :
  B ^ 48 = ![![0, 0, 0], ![0, 16^12, 0], ![0, 0, 16^12]] :=
by sorry

end matrix_B_pow48_l163_163251


namespace prove_RoseHasMoney_l163_163102
noncomputable def RoseHasMoney : Prop :=
  let cost_of_paintbrush := 2.40
  let cost_of_paints := 9.20
  let cost_of_easel := 6.50
  let total_cost := cost_of_paintbrush + cost_of_paints + cost_of_easel
  let additional_money_needed := 11
  let money_rose_has := total_cost - additional_money_needed
  money_rose_has = 7.10

theorem prove_RoseHasMoney : RoseHasMoney :=
  sorry

end prove_RoseHasMoney_l163_163102


namespace function_problem_l163_163823

theorem function_problem (f : ℕ → ℝ) (h1 : ∀ p q : ℕ, f (p + q) = f p * f q) (h2 : f 1 = 3) :
  (f (1) ^ 2 + f (2)) / f (1) + (f (2) ^ 2 + f (4)) / f (3) + (f (3) ^ 2 + f (6)) / f (5) + 
  (f (4) ^ 2 + f (8)) / f (7) + (f (5) ^ 2 + f (10)) / f (9) = 30 := by
  sorry

end function_problem_l163_163823


namespace vacation_hours_per_week_l163_163832

open Nat

theorem vacation_hours_per_week :
  let planned_hours_per_week := 25
  let total_weeks := 15
  let total_money_needed := 4500
  let sick_weeks := 3
  let hourly_rate := total_money_needed / (planned_hours_per_week * total_weeks)
  let remaining_weeks := total_weeks - sick_weeks
  let total_hours_needed := total_money_needed / hourly_rate
  let required_hours_per_week := total_hours_needed / remaining_weeks
  required_hours_per_week = 31.25 := by
sorry

end vacation_hours_per_week_l163_163832


namespace inequality_solution_set_l163_163334

noncomputable def f (x : ℝ) : ℝ := (x^2 + 1) / (x + 4)^2

theorem inequality_solution_set :
  {x : ℝ | (x^2 + 1) / (x + 4)^2 ≥ 0} = {x : ℝ | x ≠ -4} :=
by
  sorry

end inequality_solution_set_l163_163334


namespace log2_75_in_terms_of_a_b_l163_163196

noncomputable def log_base2 (x : ℝ) : ℝ := Real.log x / Real.log 2

variables (a b : ℝ)
variables (log2_9_eq_a : log_base2 9 = a)
variables (log2_5_eq_b : log_base2 5 = b)

theorem log2_75_in_terms_of_a_b : log_base2 75 = (1 / 2) * a + 2 * b :=
by sorry

end log2_75_in_terms_of_a_b_l163_163196


namespace initial_candies_l163_163707

theorem initial_candies (x : ℕ) (h1 : x % 4 = 0) (h2 : x / 4 * 3 / 3 * 2 / 2 - 24 ≥ 6) (h3 : x / 4 * 3 / 3 * 2 / 2 - 24 ≤ 9) :
  x = 64 :=
sorry

end initial_candies_l163_163707


namespace triangle_construction_conditions_l163_163048

open Classical

noncomputable def construct_triangle (m_a m_b s_c : ℝ) : Prop :=
  m_a ≤ 2 * s_c ∧ m_b ≤ 2 * s_c

theorem triangle_construction_conditions (m_a m_b s_c : ℝ) :
  construct_triangle m_a m_b s_c ↔ (m_a ≤ 2 * s_c ∧ m_b ≤ 2 * s_c) :=
by
  sorry

end triangle_construction_conditions_l163_163048


namespace parabola_circle_intersection_l163_163549

theorem parabola_circle_intersection :
  (∃ x y : ℝ, y = (x - 2)^2 ∧ x + 1 = (y + 2)^2) →
  (∃ r : ℝ, ∀ x y : ℝ, (y = (x - 2)^2 ∧ x + 1 = (y + 2)^2) →
    (x - 5/2)^2 + (y + 3/2)^2 = r^2 ∧ r^2 = 3/2) :=
by
  intros
  sorry

end parabola_circle_intersection_l163_163549


namespace jake_third_test_marks_l163_163409

theorem jake_third_test_marks 
  (avg_marks : ℕ)
  (marks_test1 : ℕ)
  (marks_test2 : ℕ)
  (marks_test3 : ℕ)
  (marks_test4 : ℕ)
  (h_avg : avg_marks = 75)
  (h_test1 : marks_test1 = 80)
  (h_test2 : marks_test2 = marks_test1 + 10)
  (h_test3_eq_test4 : marks_test3 = marks_test4)
  (h_total : avg_marks * 4 = marks_test1 + marks_test2 + marks_test3 + marks_test4) : 
  marks_test3 = 65 :=
sorry

end jake_third_test_marks_l163_163409


namespace rhombus_diagonal_l163_163415

theorem rhombus_diagonal
  (d1 : ℝ) (d2 : ℝ) (area : ℝ) 
  (h1 : d1 = 17) (h2 : area = 170) 
  (h3 : area = (d1 * d2) / 2) : d2 = 20 :=
by
  sorry

end rhombus_diagonal_l163_163415


namespace selection_methods_l163_163878

theorem selection_methods (females males : Nat) (h_females : females = 3) (h_males : males = 2):
  females + males = 5 := 
  by 
    -- We add sorry here to skip the proof
    sorry

end selection_methods_l163_163878


namespace secret_code_count_l163_163081

noncomputable def number_of_secret_codes (colors slots : ℕ) : ℕ :=
  colors ^ slots

theorem secret_code_count : number_of_secret_codes 9 5 = 59049 := by
  sorry

end secret_code_count_l163_163081


namespace problem1_problem2_l163_163487

theorem problem1 (a b c : ℝ) (hab : a ≠ b) (hbc : b ≠ c) (hca : c ≠ a) :
  abs ((a + b) / (a - b)) + abs ((b + c) / (b - c)) + abs ((c + a) / (c - a)) ≥ 2 :=
sorry

theorem problem2 (a b c : ℝ) (hab : a ≠ b) (hbc : b ≠ c) (hca : c ≠ a) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) :
  abs ((a + b) / (a - b)) + abs ((b + c) / (b - c)) + abs ((c + a) / (c - a)) > 3 :=
sorry

end problem1_problem2_l163_163487


namespace general_formula_a_sum_b_condition_l163_163816

noncomputable def sequence_a (n : ℕ) : ℕ := sorry
noncomputable def sum_a (n : ℕ) : ℕ := sorry

-- Conditions
def a_2_condition : Prop := sequence_a 2 = 4
def sum_condition (n : ℕ) : Prop := 2 * sum_a n = n * sequence_a n + n

-- General formula for the n-th term of the sequence a_n
theorem general_formula_a : 
  (∀ n, sequence_a n = 3 * n - 2) ↔
  (a_2_condition ∧ ∀ n, sum_condition n) :=
sorry

noncomputable def sequence_c (n : ℕ) : ℕ := sorry
noncomputable def sequence_b (n : ℕ) : ℕ := sorry
noncomputable def sum_b (n : ℕ) : ℝ := sorry

-- Geometric sequence condition
def geometric_sequence_condition : Prop :=
  ∀ n, sequence_c n = 4^n

-- Condition for a_n = b_n * c_n
def a_b_c_relation (n : ℕ) : Prop := 
  sequence_a n = sequence_b n * sequence_c n

-- Sum condition T_n < 2/3
theorem sum_b_condition :
  (∀ n, a_b_c_relation n) ∧ geometric_sequence_condition →
  (∀ n, sum_b n < 2 / 3) :=
sorry

end general_formula_a_sum_b_condition_l163_163816


namespace first_five_terms_series_l163_163911

theorem first_five_terms_series (a : ℕ → ℚ) (h : ∀ n, a n = 1 / (n * (n + 1))) :
  (a 1 = 1 / 2) ∧
  (a 2 = 1 / 6) ∧
  (a 3 = 1 / 12) ∧
  (a 4 = 1 / 20) ∧
  (a 5 = 1 / 30) :=
by
  sorry

end first_five_terms_series_l163_163911


namespace symmetric_points_y_axis_l163_163240

theorem symmetric_points_y_axis :
  ∀ (m n : ℝ), (m + 4 = 0) → (n = 3) → (m + n) ^ 2023 = -1 :=
by
  intros m n Hm Hn
  sorry

end symmetric_points_y_axis_l163_163240


namespace find_students_with_equal_homework_hours_l163_163736

theorem find_students_with_equal_homework_hours :
  let Dan := 6
  let Joe := 3
  let Bob := 5
  let Susie := 4
  let Grace := 1
  (Joe + Grace = Dan ∨ Joe + Bob = Dan ∨ Bob + Grace = Dan ∨ Dan + Bob = Dan ∨ Susie + Grace = Dan) → 
  (Bob + Grace = Dan) := 
by 
  intros
  sorry

end find_students_with_equal_homework_hours_l163_163736


namespace problem_f3_is_neg2_l163_163283

theorem problem_f3_is_neg2 (f : ℝ → ℝ) (h1 : ∀ x, f (-x) = -f x) (h2 : ∀ x, f (1 + x) = -f (1 - x)) (h3 : f 1 = 2) : f 3 = -2 :=
sorry

end problem_f3_is_neg2_l163_163283


namespace prob_of_2_digit_in_frac_1_over_7_l163_163891

noncomputable def prob (n : ℕ) : ℚ := (3/2)^(n-1) / (3/2 - 1)

theorem prob_of_2_digit_in_frac_1_over_7 :
  let infinite_series_sum := ∑' n : ℕ, (2/3)^(6 * n + 3)
  ∑' (n : ℕ), prob (6 * n + 3) = 108 / 665 :=
by
  sorry

end prob_of_2_digit_in_frac_1_over_7_l163_163891


namespace sqrt_factorial_mul_factorial_l163_163585

theorem sqrt_factorial_mul_factorial (n : ℕ) : 
  n = 4 → sqrt ((nat.factorial n) * (nat.factorial n)) = nat.factorial n :=
by
  intro h
  rw [h, nat.factorial, mul_self_sqrt (nat.factorial_nonneg 4)]

-- Note: While the final "mul_self_sqrt (nat.factorial_nonneg 4)" line is a sketch of the idea,
-- the proof is not complete as requested.

end sqrt_factorial_mul_factorial_l163_163585


namespace solution_set_ineq_l163_163363

noncomputable def f (x : ℝ) : ℝ := Real.exp x - Real.exp (-x) - 2 * x

theorem solution_set_ineq (x : ℝ) : f (x^2 - 4) + f (3*x) > 0 ↔ x > 1 ∨ x < -4 :=
by sorry

end solution_set_ineq_l163_163363


namespace hangar_length_l163_163639

-- Define the conditions
def num_planes := 7
def length_per_plane := 40 -- in feet

-- Define the main theorem to be proven
theorem hangar_length : num_planes * length_per_plane = 280 := by
  -- Proof omitted with sorry
  sorry

end hangar_length_l163_163639


namespace sqrt_factorial_product_eq_24_l163_163569

theorem sqrt_factorial_product_eq_24 : (sqrt (fact 4 * fact 4) = 24) :=
by sorry

end sqrt_factorial_product_eq_24_l163_163569


namespace candy_box_original_price_l163_163770

theorem candy_box_original_price (P : ℝ) (h₁ : 1.25 * P = 10) : P = 8 := 
sorry

end candy_box_original_price_l163_163770


namespace total_lunch_bill_l163_163275

theorem total_lunch_bill (cost_hotdog cost_salad : ℝ) (h1 : cost_hotdog = 5.36) (h2 : cost_salad = 5.10) : 
  cost_hotdog + cost_salad = 10.46 := 
by 
  sorry

end total_lunch_bill_l163_163275


namespace trams_to_add_l163_163556

theorem trams_to_add (initial_trams : ℕ) (initial_interval new_interval : ℤ)
  (reduce_by_fraction : ℤ) (total_distance : ℤ)
  (h1 : initial_trams = 12)
  (h2 : initial_interval = total_distance / initial_trams)
  (h3 : reduce_by_fraction = 5)
  (h4 : new_interval = initial_interval - initial_interval / reduce_by_fraction) :
  initial_trams + (total_distance / new_interval - initial_trams) = 15 :=
by
  sorry

end trams_to_add_l163_163556


namespace wholesale_prices_l163_163156

-- Definitions for the problem conditions
variable (p1 p2 d k : ℝ)
variable (h_d : d > 0)
variable (h_k : k > 1)
variable (prices : Finset ℝ)
variable (h_prices : prices = {64, 64, 70, 72})

-- The theorem statement to prove
theorem wholesale_prices :
  ∃ p1 p2, (p1 + d ∈ prices ∧ k * p1 ∈ prices) ∧ 
           (p2 + d ∈ prices ∧ k * p2 ∈ prices) ∧ 
           p1 ≠ p2
:= sorry

end wholesale_prices_l163_163156


namespace win_prize_probability_l163_163315

-- Define a condition that represents the probability calculation
def probability (total_outcomes favorable_outcomes : ℕ) : ℚ :=
  favorable_outcomes / total_outcomes

-- Declare the constants as provided in the problem
def total_cards : ℕ := 4
def total_bags : ℕ := 6

-- Calculate the total number of outcomes
def total_outcomes : ℕ := total_cards ^ total_bags

-- Calculate the number of favorable outcomes as explained in the solution
def favorable_outcomes : ℕ :=
  let scenario1 := 4 * Nat.factorial 6 / (Nat.factorial 3 * Nat.factorial 1 * Nat.factorial 1 * Nat.factorial 1)
  let scenario2 := 6 * Nat.factorial 6 / (Nat.factorial 2 * Nat.factorial 2 * Nat.factorial 1 * Nat.factorial 1)
  scenario1 + scenario2

-- Calculate the probability
def computed_probability : ℚ := probability total_outcomes favorable_outcomes

-- Provide the proof statement
theorem win_prize_probability : computed_probability = 195 / 512 :=
  by
  -- Here you would provide the proof of this theorem.
  -- The details of the steps are omitted and replaced with sorry.
  sorry

end win_prize_probability_l163_163315


namespace base_representation_l163_163663

theorem base_representation (b : ℕ) (h₁ : b^2 ≤ 125) (h₂ : 125 < b^3) :
  (∀ b, b = 12 → 125 % b % 2 = 1) → b = 12 := 
by
  sorry

end base_representation_l163_163663


namespace initial_integer_l163_163930

theorem initial_integer (x : ℤ) (h : (x + 2)^2 = x^2 - 2016) : x = -505 :=
by
  sorry

end initial_integer_l163_163930


namespace find_rate_of_interest_l163_163317

def simple_interest (principal : ℕ) (rate : ℕ) (time : ℕ) : ℕ :=
  principal * rate * time / 100

theorem find_rate_of_interest :
  ∀ (R : ℕ),
  simple_interest 5000 R 2 + simple_interest 3000 R 4 = 2640 → R = 12 :=
by
  intros R h
  sorry

end find_rate_of_interest_l163_163317


namespace arithmetic_sequence_common_difference_l163_163113

theorem arithmetic_sequence_common_difference (a : ℕ → ℤ) (d : ℤ) :
  (a 5 = 8) → (a 1 + a 2 + a 3 = 6) → (∀ n : ℕ, a (n + 1) = a 1 + n * d) → d = 2 :=
by
  intros ha5 hsum harr
  sorry

end arithmetic_sequence_common_difference_l163_163113


namespace option_c_correct_l163_163502

theorem option_c_correct (a b : ℝ) (h : a > b) : 2 + a > 2 + b :=
by sorry

end option_c_correct_l163_163502


namespace total_votes_l163_163148

theorem total_votes (V : ℝ) (h1 : 0.60 * V = V - 240) : V = 600 :=
sorry

end total_votes_l163_163148


namespace hats_needed_to_pay_51_l163_163844

def shirt_cost : ℕ := 5
def hat_cost : ℕ := 4
def jeans_cost : ℕ := 10
def total_amount : ℕ := 51
def num_shirts : ℕ := 3
def num_jeans : ℕ := 2

theorem hats_needed_to_pay_51 :
  ∃ (n : ℕ), total_amount = num_shirts * shirt_cost + num_jeans * jeans_cost + n * hat_cost ∧ n = 4 :=
by
  sorry

end hats_needed_to_pay_51_l163_163844


namespace number_of_triples_l163_163500

theorem number_of_triples : 
  {n : ℕ // ∃ a b c : ℝ, a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ ab = c ∧ bc = a ∧ ca = b ∧ n = 4} :=
sorry

end number_of_triples_l163_163500


namespace determine_n_l163_163218

theorem determine_n (n : ℕ) (h : 9^4 = 3^n) : n = 8 :=
by {
  sorry
}

end determine_n_l163_163218


namespace unique_k_for_prime_roots_of_quadratic_l163_163470

/-- Function to check primality of a natural number -/
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

/-- Theorem statement with the given conditions -/
theorem unique_k_for_prime_roots_of_quadratic :
  ∃! k : ℕ, ∃ p q : ℕ, is_prime p ∧ is_prime q ∧ p + q = 50 ∧ p * q = k :=
sorry

end unique_k_for_prime_roots_of_quadratic_l163_163470


namespace parallel_lines_a_l163_163828

theorem parallel_lines_a (a : ℝ) :
  ((∃ k : ℝ, (a + 2) / 6 = k ∧ (a + 3) / (2 * a - 1) = k) ∧ 
   ¬ ((-5 / -5) = ((a + 2) / 6)) ∧ ((a + 3) / (2 * a - 1) = (-5 / -5))) →
  a = -5 / 2 :=
by
  sorry

end parallel_lines_a_l163_163828


namespace inverse_prop_function_through_point_l163_163969

theorem inverse_prop_function_through_point :
  ∀ (f : ℝ → ℝ), (∀ x, f x = k / x) → (f 1 = 2) → (f (-1) = -2) :=
by
  intros f h_inv_prop h_f1
  sorry

end inverse_prop_function_through_point_l163_163969


namespace excircle_opposite_side_b_l163_163085

-- Definition of the terms and assumptions
variables {a b c : ℝ} -- sides of the triangle
variables {r r1 : ℝ}  -- radii of the circles

-- Given conditions
def touches_side_c_and_extensions_of_a_b (r : ℝ) (a b c : ℝ) : Prop :=
  r = (a + b + c) / 2

-- The goal to be proved
theorem excircle_opposite_side_b (a b c : ℝ) (r1 : ℝ) (h1 : touches_side_c_and_extensions_of_a_b r a b c) :
  r1 = (a + c - b) / 2 := 
by
  sorry

end excircle_opposite_side_b_l163_163085


namespace geom_seq_sum_elems_l163_163242

theorem geom_seq_sum_elems (a : ℕ → ℝ) (r : ℝ)
  (h_geom : ∀ n, a (n + 1) = a n * r)
  (h_sum1 : a 1 + a 2 = 40)
  (h_sum2 : a 3 + a 4 = 60) :
  a 7 + a 8 = 135 :=
sorry

end geom_seq_sum_elems_l163_163242


namespace add_trams_l163_163559

theorem add_trams (total_trams : ℕ) (total_distance : ℝ) (initial_intervals : ℝ) (new_intervals : ℝ) (additional_trams : ℕ) :
  total_trams = 12 → total_distance = 60 → initial_intervals = total_distance / total_trams →
  new_intervals = initial_intervals - (initial_intervals / 5) →
  additional_trams = (total_distance / new_intervals) - total_trams →
  additional_trams = 3 :=
begin
  intros h1 h2 h3 h4 h5,
  sorry
end

end add_trams_l163_163559


namespace simplify_parentheses_l163_163618

theorem simplify_parentheses (a b c x y : ℝ) : (3 * a - (2 * a - c) = 3 * a - 2 * a + c) := 
by 
  sorry

end simplify_parentheses_l163_163618


namespace original_numbers_correct_l163_163532

noncomputable def restore_original_numbers : List ℕ :=
  let T : ℕ := 5
  let EL : ℕ := 12
  let EK : ℕ := 19
  let LA : ℕ := 26
  let SS : ℕ := 33
  [T, EL, EK, LA, SS]

theorem original_numbers_correct :
  restore_original_numbers = [5, 12, 19, 26, 33] :=
by
  sorry

end original_numbers_correct_l163_163532


namespace exceeds_alpha_beta_l163_163074

noncomputable def condition (α β p q : ℝ) : Prop :=
  q < 50 ∧ α > 0 ∧ β > 0 ∧ p > 0 ∧ q > 0

theorem exceeds_alpha_beta (α β p q : ℝ) (h : condition α β p q) :
  (1 + p / 100) * (1 - q / 100) > 1 → p > 100 * q / (100 - q) := by
  sorry

end exceeds_alpha_beta_l163_163074


namespace theta_values_satisfy_eq_l163_163072

noncomputable def numSolutions : ℝ := 4 

theorem theta_values_satisfy_eq:
    (∃ n : ℕ, n = 4 ∧
      ∀ θ : ℝ, 
        0 < θ ∧ θ ≤ 2 * Real.pi →
        2 - 4 * Real.cos θ + 3 * Real.sin (2 * θ) = 0) :=
by
  let number_of_solutions := numSolutions
  have h : number_of_solutions = 4, from rfl
  sorry

end theta_values_satisfy_eq_l163_163072


namespace log_sum_equality_l163_163946

noncomputable def evaluate_log_sum : ℝ :=
  3 / (Real.log 1000^4 / Real.log 8) + 4 / (Real.log 1000^4 / Real.log 10)

theorem log_sum_equality :
  evaluate_log_sum = (9 * Real.log 2 / Real.log 10 + 4) / 12 :=
by
  sorry

end log_sum_equality_l163_163946


namespace min_value_reciprocal_sum_l163_163255

theorem min_value_reciprocal_sum (x y z : ℝ) (h_pos : 0 < x ∧ 0 < y ∧ 0 < z) (h_sum : x + y + z = 2) : 
  (∃ c : ℝ, c = (1/x) + (1/y) + (1/z) ∧ c ≥ 9/2) :=
by
  -- proof would go here
  sorry

end min_value_reciprocal_sum_l163_163255


namespace Brandy_caffeine_intake_l163_163001

theorem Brandy_caffeine_intake :
  let weight := 60
  let recommended_limit_per_kg := 2.5
  let tolerance := 50
  let coffee_cups := 2
  let coffee_per_cup := 95
  let energy_drinks := 4
  let caffeine_per_energy_drink := 120
  let max_safe_caffeine := weight * recommended_limit_per_kg + tolerance
  let caffeine_from_coffee := coffee_cups * coffee_per_cup
  let caffeine_from_energy_drinks := energy_drinks * caffeine_per_energy_drink
  let total_caffeine_consumed := caffeine_from_coffee + caffeine_from_energy_drinks
  max_safe_caffeine - total_caffeine_consumed = -470 := 
by
  sorry

end Brandy_caffeine_intake_l163_163001


namespace find_g_at_1_l163_163886

theorem find_g_at_1 (g : ℝ → ℝ) (h : ∀ x, x ≠ 1/2 → g x + g ((2*x + 1)/(1 - 2*x)) = x) : 
  g 1 = 15 / 7 :=
sorry

end find_g_at_1_l163_163886


namespace red_pairs_count_l163_163177

theorem red_pairs_count (students_green : ℕ) (students_red : ℕ) (total_students : ℕ) (total_pairs : ℕ)
(pairs_green_green : ℕ) : 
students_green = 63 →
students_red = 69 →
total_students = 132 →
total_pairs = 66 →
pairs_green_green = 21 →
∃ (pairs_red_red : ℕ), pairs_red_red = 24 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end red_pairs_count_l163_163177


namespace units_digit_of_product_l163_163754

-- Define the three given even composite numbers
def a := 4
def b := 6
def c := 8

-- Define the product of the three numbers
def product := a * b * c

-- State the units digit of the product
theorem units_digit_of_product : product % 10 = 2 :=
by
  -- Proof is skipped here
  sorry

end units_digit_of_product_l163_163754


namespace probability_both_selected_l163_163010

theorem probability_both_selected (P_R : ℚ) (P_V : ℚ) (h1 : P_R = 3 / 7) (h2 : P_V = 1 / 5) :
  P_R * P_V = 3 / 35 :=
by {
  sorry
}

end probability_both_selected_l163_163010


namespace probability_tina_changes_twenty_dollar_bill_l163_163564

theorem probability_tina_changes_twenty_dollar_bill :
  let toys := list.range 10 |>.map (λ n, (n + 1) * 50)
  let favorite_toy_price := 400
  let tina_quarters := 12
  let total_permutations := (10.fact : ℚ)
  let favorable_permutations := (4.fact * 6.fact : ℚ)
  let probability_direct_purchase := favorable_permutations / total_permutations
  let required_change_probability := 1 - probability_direct_purchase
  required_change_probability = (999802 / 1000000 : ℚ) :=
by
  let toys := list.range 10 |>.map (λ n, (n + 1) * 50)
  let favorite_toy_price := 400
  let tina_quarters := 12
  let total_permutations := (10.fact : ℚ)
  let favorable_permutations := (4.fact * 6.fact : ℚ)
  let probability_direct_purchase := favorable_permutations / total_permutations
  let required_change_probability := 1 - probability_direct_purchase
  sorry

end probability_tina_changes_twenty_dollar_bill_l163_163564


namespace additional_trams_proof_l163_163560

-- Definitions for the conditions
def initial_tram_count : Nat := 12
def total_distance : Nat := 60
def initial_interval : Nat := total_distance / initial_tram_count
def reduced_interval : Nat := initial_interval - (initial_interval / 5)
def final_tram_count : Nat := total_distance / reduced_interval
def additional_trams_needed : Nat := final_tram_count - initial_tram_count

-- The theorem we need to prove
theorem additional_trams_proof : additional_trams_needed = 3 :=
by
  sorry

end additional_trams_proof_l163_163560


namespace mindy_tax_rate_proof_l163_163530

noncomputable def mindy_tax_rate (M r : ℝ) : Prop :=
  let Mork_tax := 0.10 * M
  let Mindy_income := 3 * M
  let Mindy_tax := r * Mindy_income
  let Combined_tax_rate := 0.175
  let Combined_tax := Combined_tax_rate * (M + Mindy_income)
  Mork_tax + Mindy_tax = Combined_tax

theorem mindy_tax_rate_proof (M r : ℝ) 
  (h1 : Mork_tax_rate = 0.10) 
  (h2 : mindy_income = 3 * M) 
  (h3 : combined_tax_rate = 0.175) : 
  r = 0.20 := 
sorry

end mindy_tax_rate_proof_l163_163530


namespace solve_for_x_l163_163720

theorem solve_for_x (x : ℝ) (h : (4/7) * (2/5) * x = 8) : x = 35 :=
sorry

end solve_for_x_l163_163720


namespace sum_proper_divisors_81_l163_163441

theorem sum_proper_divisors_81 : 
  let proper_divisors : List ℕ := [1, 3, 9, 27] in
  proper_divisors.sum = 40 :=
by
  sorry

end sum_proper_divisors_81_l163_163441


namespace skt_lineups_l163_163550

theorem skt_lineups :
  let total_progamer_count : ℕ := 111,
      initial_team_size : ℕ := 11,
      lineup_size : ℕ := 5,
      new_progamers_count : ℕ := total_progamer_count - initial_team_size,
      case1_count := Nat.choose initial_team_size lineup_size,
      case2_count := Nat.choose initial_team_size (lineup_size - 1) * new_progamers_count,
      total_unordered_lineups := case1_count + case2_count,
      ordered_lineups := total_unordered_lineups * lineup_size.factorial
  in ordered_lineups = 4015440 :=
by
  sorry

end skt_lineups_l163_163550


namespace team_a_vs_team_b_l163_163509

noncomputable def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (Nat.choose n k) * (p ^ k) * ((1 - p) ^ (n - k))

theorem team_a_vs_team_b (P1 P2 : ℝ) :
  let n_a := 5
  let x_a := 4
  let p_a := 0.5
  let n_b := 5
  let x_b := 3
  let p_b := 1/3
  let P1 := binomial_probability n_a x_a p_a
  let P2 := binomial_probability n_b x_b p_b
  P1 < P2 := by sorry

end team_a_vs_team_b_l163_163509


namespace find_counterfeit_two_weighings_l163_163404

-- defining the variables and conditions
variable (coins : Fin 7 → ℝ)
variable (real_weight : ℝ)
variable (fake_weight : ℝ)
variable (is_counterfeit : Fin 7 → Prop)

-- conditions
axiom counterfeit_weight_diff : ∀ i, is_counterfeit i ↔ (coins i = fake_weight)
axiom consecutive_counterfeits : ∃ (start : Fin 7), ∀ i, (start ≤ i ∧ i < start + 4) → is_counterfeit (i % 7)
axiom weight_diff : fake_weight < real_weight

-- Theorem statement
theorem find_counterfeit_two_weighings : 
  (coins (1 : Fin 7) + coins (2 : Fin 7) = coins (4 : Fin 7) + coins (5 : Fin 7)) →
  is_counterfeit (6 : Fin 7) ∧ is_counterfeit (7 : Fin 7) := 
sorry

end find_counterfeit_two_weighings_l163_163404


namespace circle_properties_l163_163253

theorem circle_properties :
  ∃ (c d s : ℝ), (∀ x y : ℝ, x^2 - 4 * y - 25 = -y^2 + 10 * x + 49 → (x - 5)^2 + (y - 2)^2 = s^2) ∧
  c = 5 ∧ d = 2 ∧ s = Real.sqrt 103 ∧ c + d + s = 7 + Real.sqrt 103 :=
by
  sorry

end circle_properties_l163_163253


namespace bakery_wholesale_price_exists_l163_163158

theorem bakery_wholesale_price_exists :
  ∃ (p1 p2 d k : ℕ), (d > 0) ∧ (k > 1) ∧
  ({p1 + d, p2 + d, k * p1, k * p2} = {64, 64, 70, 72}) :=
by sorry

end bakery_wholesale_price_exists_l163_163158


namespace sqrt_factorial_product_l163_163598

theorem sqrt_factorial_product :
  Nat.sqrt (Nat.factorial 4 * Nat.factorial 4) = 24 := 
sorry

end sqrt_factorial_product_l163_163598


namespace part1_part2_l163_163962

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := x^4 - 4 * x^3 + (3 + m) * x^2 - 12 * x + 12

theorem part1 (m : ℤ) : 
  (∀ x : ℝ, f x m - f (1 - x) m + 4 * x^3 = 0) ↔ (m = 8 ∨ m = 12) := 
sorry

theorem part2 (m : ℝ) : 
  (∀ x : ℝ, f x m ≥ 0) ↔ (4 ≤ m) := 
sorry

end part1_part2_l163_163962


namespace fourth_term_is_six_l163_163996

-- Definitions from the problem
variables (a d : ℕ)

-- Condition that the sum of the third and fifth terms is 12
def sum_third_fifth_eq_twelve : Prop := (a + 2 * d) + (a + 4 * d) = 12

-- The fourth term of the arithmetic sequence
def fourth_term : ℕ := a + 3 * d

-- The theorem we need to prove
theorem fourth_term_is_six (h : sum_third_fifth_eq_twelve a d) : fourth_term a d = 6 := by
  sorry

end fourth_term_is_six_l163_163996


namespace integer_x_cubed_prime_l163_163651

theorem integer_x_cubed_prime (x : ℕ) : 
  (∃ p : ℕ, Prime p ∧ (2^x + x^2 + 25 = p^3)) → x = 6 :=
by
  sorry

end integer_x_cubed_prime_l163_163651


namespace trams_to_add_l163_163554

theorem trams_to_add (initial_trams : ℕ) (initial_interval new_interval : ℤ)
  (reduce_by_fraction : ℤ) (total_distance : ℤ)
  (h1 : initial_trams = 12)
  (h2 : initial_interval = total_distance / initial_trams)
  (h3 : reduce_by_fraction = 5)
  (h4 : new_interval = initial_interval - initial_interval / reduce_by_fraction) :
  initial_trams + (total_distance / new_interval - initial_trams) = 15 :=
by
  sorry

end trams_to_add_l163_163554


namespace log_base_5_of_15625_l163_163787

-- Defining that 5^6 = 15625
theorem log_base_5_of_15625 : log 5 15625 = 6 := 
by {
    -- place the required proof here
    sorry
}

end log_base_5_of_15625_l163_163787


namespace ord_p_factorial_factorial_ratio_is_integer_another_factorial_ratio_is_integer_l163_163666

-- Part 1
theorem ord_p_factorial (n p : ℕ) (hp : p > 1) :
  ∏ ord_p n n! = (n - S_p(n)) / (p - 1) := 
sorry

-- Part 2
theorem factorial_ratio_is_integer (n : ℕ) :
  ∏ n > 0 → ∃ k : ℕ, ( ( (2 * n)! / ((n!) * ((n + 1)!) ) = k ) ) :=
sorry

-- Part 3
theorem another_factorial_ratio_is_integer (m n : ℕ) (Hmn : gcd m (n + 1) = 1) :
  ∃ k : ℕ, ( ( (m * n + n)! / ( (m * n)! * ( (n + 1)! ) ) = k ) ) :=
sorry

end ord_p_factorial_factorial_ratio_is_integer_another_factorial_ratio_is_integer_l163_163666


namespace expression_equality_l163_163142

theorem expression_equality :
  - (2^3) = (-2)^3 :=
by sorry

end expression_equality_l163_163142


namespace total_amount_spent_l163_163247

namespace KeithSpending

def speakers_cost : ℝ := 136.01
def cd_player_cost : ℝ := 139.38
def tires_cost : ℝ := 112.46
def total_cost : ℝ := 387.85

theorem total_amount_spent : speakers_cost + cd_player_cost + tires_cost = total_cost :=
by sorry

end KeithSpending

end total_amount_spent_l163_163247


namespace trigonometric_proof_l163_163645

noncomputable def proof_problem (α β : Real) : Prop :=
  (β = 90 - α) → (Real.sin β = Real.cos α) → 
  (Real.sqrt 3 * Real.sin α + Real.sin β) / Real.sqrt (2 - 2 * Real.cos 100) = 1

-- Statement that incorporates all conditions and concludes the proof problem.
theorem trigonometric_proof :
  proof_problem 20 70 :=
by
  intros h1 h2
  sorry

end trigonometric_proof_l163_163645


namespace sum_diameters_eq_sum_legs_l163_163879

theorem sum_diameters_eq_sum_legs 
  (a b c R r : ℝ)
  (h_right_triangle : a^2 + b^2 = c^2)
  (h_circum_radius : R = c / 2)
  (h_incircle_radius : r = (a + b - c) / 2) :
  2 * R + 2 * r = a + b :=
by 
  sorry

end sum_diameters_eq_sum_legs_l163_163879


namespace space_shuttle_speed_kmh_l163_163323

-- Define the given conditions
def speedInKmPerSecond : ℕ := 4
def secondsInAnHour : ℕ := 3600

-- State the proof problem
theorem space_shuttle_speed_kmh : speedInKmPerSecond * secondsInAnHour = 14400 := by
  sorry

end space_shuttle_speed_kmh_l163_163323


namespace cricket_bat_profit_percentage_correct_football_profit_percentage_correct_l163_163469

noncomputable def cricket_bat_selling_price : ℝ := 850
noncomputable def cricket_bat_profit : ℝ := 215
noncomputable def cricket_bat_cost_price : ℝ := cricket_bat_selling_price - cricket_bat_profit
noncomputable def cricket_bat_profit_percentage : ℝ := (cricket_bat_profit / cricket_bat_cost_price) * 100

noncomputable def football_selling_price : ℝ := 120
noncomputable def football_profit : ℝ := 45
noncomputable def football_cost_price : ℝ := football_selling_price - football_profit
noncomputable def football_profit_percentage : ℝ := (football_profit / football_cost_price) * 100

theorem cricket_bat_profit_percentage_correct :
  |cricket_bat_profit_percentage - 33.86| < 1e-2 :=
by sorry

theorem football_profit_percentage_correct :
  football_profit_percentage = 60 :=
by sorry

end cricket_bat_profit_percentage_correct_football_profit_percentage_correct_l163_163469


namespace sqrt_factorial_product_l163_163583

/-- Define the factorial of 4 -/
def factorial_four : ℕ := 4!

/-- Define the product of the factorial of 4 with itself -/
def product_of_factorials : ℕ := factorial_four * factorial_four

/-- Prove the value of the square root of product_of_factorials is 24 -/
theorem sqrt_factorial_product : Real.sqrt (product_of_factorials) = 24 := by
  have fact_4_eq_24 : factorial_four = 24 := by norm_num
  rw [product_of_factorials, fact_4_eq_24, Nat.mul_self_eq, Real.sqrt_sq]
  norm_num
  exact Nat.zero_le 24

end sqrt_factorial_product_l163_163583


namespace problem1_l163_163718

theorem problem1 :
  0.064^(-1 / 3) - (-1 / 8)^0 + 16^(3 / 4) + 0.25^(1 / 2) = 10 :=
by
  sorry

end problem1_l163_163718


namespace possible_integer_lengths_for_third_side_l163_163225

theorem possible_integer_lengths_for_third_side (x : ℕ) : (8 < x ∧ x < 19) ↔ (4 ≤ x ∧ x ≤ 18) :=
sorry

end possible_integer_lengths_for_third_side_l163_163225


namespace mul_mental_math_l163_163473

theorem mul_mental_math :
  96 * 104 = 9984 := by
  sorry

end mul_mental_math_l163_163473


namespace max_value_x_y_l163_163193

theorem max_value_x_y (x y : ℝ) 
  (h1 : log ((x^2 + y^2) / 2) y ≥ 1) 
  (h2 : (x, y) ≠ (0, 0)) 
  (h3 : x^2 + y^2 ≠ 2) : 
  x + y ≤ 1 + real.sqrt 2 := 
sorry

end max_value_x_y_l163_163193


namespace problem_l163_163849

-- Definitions for angles A, B, C and sides a, b, c of a triangle.
variables {A B C : ℝ} {a b c : ℝ}
-- Given condition
variables (h : a = b * Real.cos C + c * Real.sin B)

-- Triangle inequality and angle conditions
variables (ha : 0 < A) (hb : 0 < B) (hc : 0 < C)
variables (suma : A + B + C = Real.pi)

-- Goal: to prove that under the given condition, angle B is π/4
theorem problem : B = Real.pi / 4 :=
by {
  sorry
}

end problem_l163_163849


namespace gcd_mn_mn_squared_l163_163888

theorem gcd_mn_mn_squared (m n : ℕ) (h : Nat.gcd m n = 1) : ({d : ℕ | d = Nat.gcd (m + n) (m ^ 2 + n ^ 2)} ⊆ {1, 2}) := 
sorry

end gcd_mn_mn_squared_l163_163888


namespace units_digit_of_product_l163_163753

-- Define the three given even composite numbers
def a := 4
def b := 6
def c := 8

-- Define the product of the three numbers
def product := a * b * c

-- State the units digit of the product
theorem units_digit_of_product : product % 10 = 2 :=
by
  -- Proof is skipped here
  sorry

end units_digit_of_product_l163_163753


namespace ratio_of_volumes_of_cones_l163_163627

theorem ratio_of_volumes_of_cones (r θ h1 h2 : ℝ) (hθ : 3 * θ + 4 * θ = 2 * π)
    (hr1 : r₁ = 3 * r / 7) (hr2 : r₂ = 4 * r / 7) :
    let V₁ := (1 / 3) * π * r₁^2 * h1
    let V₂ := (1 / 3) * π * r₂^2 * h2
    V₁ / V₂ = (9 : ℝ) / 16 := by
  sorry

end ratio_of_volumes_of_cones_l163_163627


namespace minimal_positive_sum_circle_integers_l163_163166

-- Definitions based on the conditions in the problem statement
def cyclic_neighbors (l : List Int) (i : ℕ) : Int :=
  l.getD (Nat.mod (i - 1) l.length) 0 + l.getD (Nat.mod (i + 1) l.length) 0

-- Problem statement in Lean: 
theorem minimal_positive_sum_circle_integers :
  ∃ (l : List Int), l.length ≥ 5 ∧ (∀ (i : ℕ), i < l.length → l.getD i 0 ∣ cyclic_neighbors l i) ∧ (0 < l.sum) ∧ l.sum = 2 :=
sorry

end minimal_positive_sum_circle_integers_l163_163166


namespace mutually_exclusive_event_3_l163_163957

def is_odd (n : ℕ) := n % 2 = 1
def is_even (n : ℕ) := n % 2 = 0

def event_1 (a b : ℕ) := 
(is_odd a ∧ is_even b) ∨ (is_even a ∧ is_odd b)

def event_2 (a b : ℕ) := 
is_odd a ∧ is_odd b

def event_3 (a b : ℕ) := 
is_odd a ∧ is_even a ∧ is_odd b ∧ is_even b

def event_4 (a b : ℕ) :=
(is_odd a ∧ is_even b) ∨ (is_even a ∧ is_odd b)

theorem mutually_exclusive_event_3 :
  ∀ a b : ℕ, event_3 a b → ¬ event_1 a b ∧ ¬ event_2 a b ∧ ¬ event_4 a b := by
sorry

end mutually_exclusive_event_3_l163_163957


namespace simplify_expression_l163_163717

variable {x : ℤ}

theorem simplify_expression : 3 * x + 6 * x + 9 * x + 12 * x + 15 * x + 18 = 45 * x + 18 := 
by 
  calc
    3 * x + 6 * x + 9 * x + 12 * x + 15 * x + 18 = (3 + 6 + 9 + 12 + 15) * x + 18 : by ring
    ... = 45 * x + 18 : by norm_num

end simplify_expression_l163_163717


namespace find_angle_D_l163_163539

theorem find_angle_D (A B C D : ℝ) (h1 : A + B = 180) (h2 : C = D) (h3 : A = 50) (h4 : ∃ B_adj, B_adj = 60 ∧ A + B_adj + B = 180) : D = 25 :=
sorry

end find_angle_D_l163_163539


namespace find_matrix_M_l163_163484

-- Given conditions
def vector1 : Fin 2 → ℚ := fun i => if i = 0 then 2 else -1
def vector2 : Fin 2 → ℚ := fun i => if i = 0 then 1 else 3
def output1 : Fin 2 → ℚ := fun i => if i = 0 then 3 else -7
def output2 : Fin 2 → ℚ := fun i => if i = 0 then 10 else 1

-- Solution matrix
def M : Matrix (Fin 2) (Fin 2) ℚ :=
  λ i j => if i = 0 && j = 0 then 19 / 7 else
           if i = 0 && j = 1 then 17 / 7 else
           if i = 1 && j = 0 then -20 / 7 else
           if i = 1 && j = 1 then 9 / 7 else 0

-- Proof statement
theorem find_matrix_M :
    (M.mulVec vector1 = output1) ∧
    (M.mulVec vector2 = output2) := by
  sorry

end find_matrix_M_l163_163484


namespace last_four_digits_pow_product_is_5856_l163_163643

noncomputable def product : ℕ := 301 * 402 * 503 * 604 * 646 * 547 * 448 * 349

theorem last_four_digits_pow_product_is_5856 :
  (product % 10000) ^ 4 % 10000 = 5856 := by
  sorry

end last_four_digits_pow_product_is_5856_l163_163643


namespace claudia_fills_4ounce_glasses_l163_163938

theorem claudia_fills_4ounce_glasses :
  ∀ (total_water : ℕ) (five_ounce_glasses : ℕ) (eight_ounce_glasses : ℕ) 
    (four_ounce_glass_volume : ℕ),
  total_water = 122 →
  five_ounce_glasses = 6 →
  eight_ounce_glasses = 4 →
  four_ounce_glass_volume = 4 →
  (total_water - (five_ounce_glasses * 5 + eight_ounce_glasses * 8)) / four_ounce_glass_volume = 15 :=
by
  intros _ _ _ _ _ _ _ _ 
  sorry

end claudia_fills_4ounce_glasses_l163_163938


namespace no_such_class_exists_l163_163294

theorem no_such_class_exists : ¬ ∃ (b g : ℕ), (3 * b = 5 * g) ∧ (32 < b + g) ∧ (b + g < 40) :=
by {
  -- Proof goes here
  sorry
}

end no_such_class_exists_l163_163294


namespace minimize_distances_l163_163413

/-- Given points P = (6, 7), Q = (3, 4), and R = (0, m),
    find the value of m that minimizes the sum of distances PR and QR. -/
theorem minimize_distances (m : ℝ) :
  let P := (6, 7)
  let Q := (3, 4)
  ∃ m : ℝ, 
    ∀ m' : ℝ, 
    (dist (6, 7) (0, m) + dist (3, 4) (0, m)) ≤ (dist (6, 7) (0, m') + dist (3, 4) (0, m'))
:= ⟨5, sorry⟩

end minimize_distances_l163_163413


namespace largest_number_l163_163144

theorem largest_number
  (a : ℝ) (b : ℝ) (c : ℝ) (d : ℝ) (e : ℝ)
  (ha : a = 0.883) (hb : b = 0.8839) (hc : c = 0.88) (hd : d = 0.839) (he : e = 0.889) :
  e > a ∧ e > b ∧ e > c ∧ e > d :=
by {
  sorry
}

end largest_number_l163_163144


namespace wrongly_entered_mark_l163_163033

theorem wrongly_entered_mark (x : ℝ) : 
  (∀ marks_instead_of_45 number_of_pupils (total_avg_increase : ℝ),
     marks_instead_of_45 = 45 ∧
     number_of_pupils = 44 ∧
     total_avg_increase = 0.5 →
     x = marks_instead_of_45 + total_avg_increase * number_of_pupils) →
  x = 67 :=
by
  intro h
  sorry

end wrongly_entered_mark_l163_163033


namespace new_ratio_books_clothes_l163_163004

theorem new_ratio_books_clothes :
  ∀ (B C E : ℝ), (B = 22.5) → (C = 18) → (E = 9) → (C_new = C - 9) → C_new = 9 → B / C_new = 2.5 :=
by
  intros B C E HB HC HE HCnew Hnew
  sorry

end new_ratio_books_clothes_l163_163004


namespace relationship_between_m_and_n_l163_163352

variable {X_1 X_2 k m n : ℝ}

-- Given conditions
def inverse_proportional_points (X_1 X_2 k : ℝ) (m n : ℝ) : Prop :=
  m = k / X_1 ∧ n = k / X_2 ∧ k > 0 ∧ X_1 < X_2

theorem relationship_between_m_and_n (h : inverse_proportional_points X_1 X_2 k m n) : m > n :=
by
  -- Insert proof here, skipping with sorry
  sorry

end relationship_between_m_and_n_l163_163352


namespace percentage_loss_15_l163_163028

theorem percentage_loss_15
  (sold_at_loss : ℝ)
  (sold_at_profit : ℝ)
  (percentage_profit : ℝ)
  (cost_price : ℝ)
  (percentage_loss : ℝ)
  (H1 : sold_at_loss = 12)
  (H2 : sold_at_profit = 14.823529411764707)
  (H3 : percentage_profit = 5)
  (H4 : cost_price = sold_at_profit / (1 + percentage_profit / 100))
  (H5 : percentage_loss = (cost_price - sold_at_loss) / cost_price * 100) :
  percentage_loss = 15 :=
by
  sorry

end percentage_loss_15_l163_163028


namespace poly_a_roots_poly_b_roots_l163_163057

-- Define the polynomials
def poly_a (x : ℤ) : ℤ := 2 * x ^ 3 - 3 * x ^ 2 - 11 * x + 6
def poly_b (x : ℤ) : ℤ := x ^ 4 + 4 * x ^ 3 - 9 * x ^ 2 - 16 * x + 20

-- Assert the integer roots for poly_a
theorem poly_a_roots : {x : ℤ | poly_a x = 0} = {-2, 3} := sorry

-- Assert the integer roots for poly_b
theorem poly_b_roots : {x : ℤ | poly_b x = 0} = {1, 2, -2, -5} := sorry

end poly_a_roots_poly_b_roots_l163_163057


namespace sin_cos_of_angle_l163_163665

theorem sin_cos_of_angle (a : ℝ) (h₀ : a ≠ 0) :
  ∃ (s c : ℝ), (∃ (k : ℝ), s = k * (8 / 17) ∧ c = -k * (15 / 17) ∧ k = if a > 0 then 1 else -1) :=
by
  sorry

end sin_cos_of_angle_l163_163665


namespace symmetric_pattern_count_l163_163314

noncomputable def number_of_symmetric_patterns (n : ℕ) : ℕ :=
  let regions := 12
  let total_patterns := 2^regions
  total_patterns - 2

theorem symmetric_pattern_count : number_of_symmetric_patterns 8 = 4094 :=
by
  sorry

end symmetric_pattern_count_l163_163314


namespace minimum_value_l163_163659

theorem minimum_value (x : ℝ) (hx : x > 0) : 4 * x^2 + 1 / x^3 ≥ 5 ∧ (4 * x^2 + 1 / x^3 = 5 ↔ x = 1) :=
by {
  sorry
}

end minimum_value_l163_163659


namespace find_range_of_a_l163_163826

noncomputable def f (a : ℝ) : ℝ → ℝ :=
  λ x => a * (x - 2 * Real.exp 1) * Real.log x + 1

def range_of_a (a : ℝ) : Prop :=
  (a < 0 ∨ a > 1 / Real.exp 1)

theorem find_range_of_a (a : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f a x₁ = 0 ∧ f a x₂ = 0) ↔ range_of_a a := by
  sorry

end find_range_of_a_l163_163826


namespace calculate_selling_price_l163_163764

theorem calculate_selling_price (cost_price : ℝ) (profit_percentage : ℝ) (selling_price : ℝ) 
  (h1 : cost_price = 83.33) 
  (h2 : profit_percentage = 20) : 
  selling_price = 100 := by
  sorry

end calculate_selling_price_l163_163764


namespace find_theta_l163_163948

theorem find_theta (θ : Real) : 
  (0 ≤ θ ∧ θ ≤ 2 * Real.pi) →
  (∀ x : Real, 0 ≤ x ∧ x ≤ 1 → 
    x ^ 3 * Real.sin θ + x ^ 2 * Real.cos θ - x * (1 - x) + (1 - x) ^ 2 * Real.sin θ > 0) → 
  Real.sin θ > 0 → 
  Real.cos θ + Real.sin θ > 0 → 
  (Real.pi / 12 < θ ∧ θ < 5 * Real.pi / 12) :=
by
  intro θ_range all_x_condition sin_pos cos_sin_pos
  sorry

end find_theta_l163_163948


namespace first_player_wins_l163_163232

def winning_strategy (m n : ℕ) : Prop :=
  if m = 1 ∧ n = 1 then false else true

theorem first_player_wins (m n : ℕ) :
  winning_strategy m n :=
by
  sorry

end first_player_wins_l163_163232


namespace participants_are_multiple_of_7_l163_163934

theorem participants_are_multiple_of_7 (P : ℕ) (h1 : P % 2 = 0)
  (h2 : ∀ p, p = P / 2 → P + p / 7 = (4 * P) / 7)
  (h3 : (4 * P) / 7 * 7 = 4 * P) : ∃ k : ℕ, P = 7 * k := 
by
  sorry

end participants_are_multiple_of_7_l163_163934


namespace calculate_product_l163_163257

noncomputable def complex_number_r (r : ℂ) : Prop :=
r^6 = 1 ∧ r ≠ 1

theorem calculate_product (r : ℂ) (h₁ : r^6 = 1) (h₂ : r ≠ 1) :
  (r - 1) * (r^2 - 1) * (r^3 - 1) * (r^4 - 1) * (r^5 - 1) = 2 := 
sorry

end calculate_product_l163_163257


namespace correct_range_of_x_l163_163965

variable {x : ℝ}

noncomputable def isosceles_triangle (x y : ℝ) : Prop :=
  let perimeter := 2 * y + x
  let relationship := y = - (1/2) * x + 8
  perimeter = 16 ∧ relationship

theorem correct_range_of_x (x y : ℝ) (h : isosceles_triangle x y) : 0 < x ∧ x < 8 :=
by
  -- The proof of the theorem is omitted
  sorry

end correct_range_of_x_l163_163965


namespace craig_total_commission_correct_l163_163649

-- Define the commission structures
def refrigerator_commission (price : ℝ) : ℝ := 75 + 0.08 * price
def washing_machine_commission (price : ℝ) : ℝ := 50 + 0.10 * price
def oven_commission (price : ℝ) : ℝ := 60 + 0.12 * price

-- Define total sales
def total_refrigerator_sales : ℝ := 5280
def total_washing_machine_sales : ℝ := 2140
def total_oven_sales : ℝ := 4620

-- Define number of appliances sold
def number_of_refrigerators : ℝ := 3
def number_of_washing_machines : ℝ := 4
def number_of_ovens : ℝ := 5

-- Calculate total commissions for each appliance category
def total_refrigerator_commission : ℝ := number_of_refrigerators * refrigerator_commission total_refrigerator_sales
def total_washing_machine_commission : ℝ := number_of_washing_machines * washing_machine_commission total_washing_machine_sales
def total_oven_commission : ℝ := number_of_ovens * oven_commission total_oven_sales

-- Calculate total commission for the week
def total_commission : ℝ := total_refrigerator_commission + total_washing_machine_commission + total_oven_commission

-- Prove that the total commission is as expected
theorem craig_total_commission_correct : total_commission = 5620.20 := 
by
  sorry

end craig_total_commission_correct_l163_163649


namespace dan_helmet_craters_l163_163941

variable (D S : ℕ)
variable (h1 : D = S + 10)
variable (h2 : D + S + 15 = 75)

theorem dan_helmet_craters : D = 35 := by
  sorry

end dan_helmet_craters_l163_163941


namespace door_X_is_inner_sanctuary_l163_163347

  variable (X Y Z W : Prop)
  variable (A B C D E F G H : Prop)
  variable (is_knight : Prop → Prop)

  -- Each statement according to the conditions in the problem.
  variable (stmt_A : X)
  variable (stmt_B : Y ∨ Z)
  variable (stmt_C : is_knight A ∧ is_knight B)
  variable (stmt_D : X ∧ Y)
  variable (stmt_E : X ∧ Y)
  variable (stmt_F : is_knight D ∨ is_knight E)
  variable (stmt_G : is_knight C → is_knight F)
  variable (stmt_H : is_knight G ∧ is_knight H → is_knight A)

  theorem door_X_is_inner_sanctuary :
    is_knight A → is_knight B → is_knight C → is_knight D → is_knight E → is_knight F → is_knight G → is_knight H → X :=
  sorry
  
end door_X_is_inner_sanctuary_l163_163347


namespace multiple_of_C_share_l163_163034

theorem multiple_of_C_share (A B C k : ℝ) : 
  3 * A = k * C ∧ 4 * B = k * C ∧ C = 84 ∧ A + B + C = 427 → k = 7 :=
by
  sorry

end multiple_of_C_share_l163_163034


namespace half_of_number_l163_163919

theorem half_of_number (x : ℝ) (h : (4 / 15 * 5 / 7 * x - 4 / 9 * 2 / 5 * x = 8)) : (1 / 2 * x = 315) :=
sorry

end half_of_number_l163_163919


namespace brandon_cards_l163_163043

theorem brandon_cards (b m : ℕ) 
  (h1 : m = b + 8) 
  (h2 : 14 = m / 2) : 
  b = 20 := by
  sorry

end brandon_cards_l163_163043


namespace largest_N_exists_l163_163802

noncomputable def parabola_properties (a T : ℤ) :=
    (∀ (x y : ℤ), y = a * x * (x - 2 * T) → (x = 0 ∨ x = 2 * T) → y = 0) ∧ 
    (∀ (v : ℤ × ℤ), v = (2 * T + 1, 28) → 28 = a * (2 * T + 1))

theorem largest_N_exists : 
    ∃ (a T : ℤ), T ≠ 0 ∧ (∀ (P : ℤ × ℤ), P = (0, 0) ∨ P = (2 * T, 0) ∨ P = (2 * T + 1, 28)) 
    ∧ (s = T - a * T^2) ∧ s = 60 :=
sorry

end largest_N_exists_l163_163802


namespace common_ratio_geom_arith_prog_l163_163732

theorem common_ratio_geom_arith_prog (a b c q : ℝ) 
  (h1 : b = a * q) 
  (h2 : c = a * q^2)
  (h3 : 2 * (2020 * b / 7) = 577 * a + c / 7) : 
  q = 4039 :=
begin
  -- proof to be filled
  sorry
end

end common_ratio_geom_arith_prog_l163_163732


namespace solution_range_for_m_l163_163375

theorem solution_range_for_m (x m : ℝ) (h₁ : 2 * x - 1 > 3 * (x - 2)) (h₂ : x < m) : m ≥ 5 :=
by {
  sorry
}

end solution_range_for_m_l163_163375


namespace contrapositive_of_odd_even_l163_163885

-- Definitions as conditions
def is_odd (n : ℕ) : Prop := ∃ k, n = 2 * k + 1
def is_even (n : ℕ) : Prop := ∃ k, n = 2 * k

-- Main statement
theorem contrapositive_of_odd_even :
  (∀ a b : ℕ, is_odd a ∧ is_odd b → is_even (a + b)) →
  (∀ a b : ℕ, ¬ is_even (a + b) → ¬ (is_odd a ∧ is_odd b)) := 
by
  intros h a b h1
  sorry

end contrapositive_of_odd_even_l163_163885


namespace shopkeeper_profit_percentage_l163_163322

theorem shopkeeper_profit_percentage (P : ℝ) : (70 / 100) * (1 + P / 100) = 1 → P = 700 / 3 :=
by
  sorry

end shopkeeper_profit_percentage_l163_163322


namespace Karlee_initial_grapes_l163_163246

theorem Karlee_initial_grapes (G S Remaining_Fruits : ℕ)
  (h1 : S = (3 * G) / 5)
  (h2 : Remaining_Fruits = 96)
  (h3 : Remaining_Fruits = (3 * G) / 5 + (9 * G) / 25) :
  G = 100 := by
  -- add proof here
  sorry

end Karlee_initial_grapes_l163_163246


namespace cheapest_store_for_60_balls_l163_163631

def cost_store_A (n : ℕ) (price_per_ball : ℕ) (free_per_10 : ℕ) : ℕ :=
  if n < 10 then n * price_per_ball
  else (n / 10) * 10 * price_per_ball + (n % 10) * price_per_ball * (n / (10 + free_per_10))

def cost_store_B (n : ℕ) (discount : ℕ) (price_per_ball : ℕ) : ℕ :=
  n * (price_per_ball - discount)

def cost_store_C (n : ℕ) (price_per_ball : ℕ) (cashback_threshold cashback_amt : ℕ) : ℕ :=
  let initial_cost := n * price_per_ball
  let cashback := (initial_cost / cashback_threshold) * cashback_amt
  initial_cost - cashback

theorem cheapest_store_for_60_balls
  (price_per_ball discount free_per_10 cashback_threshold cashback_amt : ℕ) :
  cost_store_A 60 price_per_ball free_per_10 = 1250 →
  cost_store_B 60 discount price_per_ball = 1200 →
  cost_store_C 60 price_per_ball cashback_threshold cashback_amt = 1290 →
  min (cost_store_A 60 price_per_ball free_per_10) (min (cost_store_B 60 discount price_per_ball) (cost_store_C 60 price_per_ball cashback_threshold cashback_amt))
  = 1200 :=
by
  sorry

end cheapest_store_for_60_balls_l163_163631


namespace smallest_divisible_12_13_14_l163_163800

theorem smallest_divisible_12_13_14 :
  ∃ n : ℕ, n > 0 ∧ (n % 12 = 0) ∧ (n % 13 = 0) ∧ (n % 14 = 0) ∧ n = 1092 := by
  sorry

end smallest_divisible_12_13_14_l163_163800


namespace find_f_inv_difference_l163_163391

axiom f : ℤ → ℤ
axiom f_inv : ℤ → ℤ
axiom f_has_inverse : ∀ x : ℤ, f_inv (f x) = x ∧ f (f_inv x) = x
axiom f_inverse_conditions : ∀ x : ℤ, f (x + 2) = f_inv (x - 1)

theorem find_f_inv_difference :
  f_inv 2004 - f_inv 1 = 4006 :=
sorry

end find_f_inv_difference_l163_163391


namespace club_supporters_l163_163507

theorem club_supporters (total_members : ℕ) (support_ratio : ℚ) (support_ratio_decimal : ℝ) 
  (membership : total_members = 15) (ratio : support_ratio = 4/5) (decimal_equiv : support_ratio_decimal = 0.8):
  (supporters_needed : ℕ) (members_ratio : ℚ) 
  (members_needed : supporters_needed = (4 * total_members / 5).ceil) 
  (ratio_calc : members_ratio = (supporters_needed : ℚ) / total_members) 
  (rounded_ratio : support_ratio_decimal = members_ratio.to_real):
  supporters_needed = 12 ∧ support_ratio_decimal = 0.8 :=
by
  sorry

end club_supporters_l163_163507


namespace mandy_book_length_l163_163262

theorem mandy_book_length :
  let initial_length := 8
  let initial_age := 6
  let doubled_age := 2 * initial_age
  let length_at_doubled_age := 5 * initial_length
  let later_age := doubled_age + 8
  let length_at_later_age := 3 * length_at_doubled_age
  let final_length := 4 * length_at_later_age
  final_length = 480 :=
by
  sorry

end mandy_book_length_l163_163262


namespace polynomial_example_properties_l163_163270

open Polynomial

noncomputable def polynomial_example : Polynomial ℚ :=
- (1 / 2) * (X^2 + X - 1) * (X^2 + 1)

theorem polynomial_example_properties :
  ∃ P : Polynomial ℚ, (X^2 + 1) ∣ P ∧ (X^3 + 1) ∣ (P - 1) :=
by
  use polynomial_example
  -- To complete the proof, one would typically verify the divisibility properties here.
  sorry

end polynomial_example_properties_l163_163270


namespace sqrt_factorial_mul_factorial_l163_163588

theorem sqrt_factorial_mul_factorial (n : ℕ) : 
  n = 4 → sqrt ((nat.factorial n) * (nat.factorial n)) = nat.factorial n :=
by
  intro h
  rw [h, nat.factorial, mul_self_sqrt (nat.factorial_nonneg 4)]

-- Note: While the final "mul_self_sqrt (nat.factorial_nonneg 4)" line is a sketch of the idea,
-- the proof is not complete as requested.

end sqrt_factorial_mul_factorial_l163_163588


namespace product_of_fraction_l163_163137

theorem product_of_fraction (x : ℚ) (h : x = 17 / 999) : 17 * 999 = 16983 := by sorry

end product_of_fraction_l163_163137


namespace find_BC_length_l163_163371

theorem find_BC_length
  (area : ℝ) (AB AC : ℝ)
  (h_area : area = 10 * Real.sqrt 3)
  (h_AB : AB = 5)
  (h_AC : AC = 8) :
  ∃ BC : ℝ, BC = 7 :=
by
  sorry

end find_BC_length_l163_163371


namespace tangent_line_through_P_l163_163825

theorem tangent_line_through_P (x y : ℝ) :
  (∃ l : ℝ, l = 3*x - 4*y + 5) ∨ (x = 1) :=
by
  sorry

end tangent_line_through_P_l163_163825


namespace fifth_number_selected_l163_163683

-- Define the necessary conditions
def num_students : ℕ := 60
def sample_size : ℕ := 5
def first_selected_number : ℕ := 4
def interval : ℕ := num_students / sample_size

-- Define the proposition to be proved
theorem fifth_number_selected (h1 : 1 ≤ first_selected_number) (h2 : first_selected_number ≤ num_students)
    (h3 : sample_size > 0) (h4 : num_students % sample_size = 0) :
  first_selected_number + 4 * interval = 52 :=
by
  -- Proof omitted
  sorry

end fifth_number_selected_l163_163683


namespace brownies_pieces_count_l163_163466

theorem brownies_pieces_count
  (pan_length pan_width piece_length piece_width : ℕ)
  (h1 : pan_length = 24)
  (h2 : pan_width = 15)
  (h3 : piece_length = 3)
  (h4 : piece_width = 2) :
  (pan_length * pan_width) / (piece_length * piece_width) = 60 :=
by
  sorry

end brownies_pieces_count_l163_163466


namespace complete_the_square_l163_163724

theorem complete_the_square (x : ℝ) (h : x^2 - 4 * x + 3 = 0) : (x - 2)^2 = 1 :=
sorry

end complete_the_square_l163_163724


namespace gcd_of_n13_minus_n_l163_163658

theorem gcd_of_n13_minus_n : 
  ∀ n : ℤ, n ≠ 0 → 2730 ∣ (n ^ 13 - n) :=
by sorry

end gcd_of_n13_minus_n_l163_163658


namespace farey_neighbors_of_half_l163_163846

noncomputable def farey_neighbors (n : ℕ) : List (ℚ) :=
  if n % 2 = 1 then
    [ (n - 1 : ℚ) / (2 * n), (n + 1 : ℚ) / (2 * n) ]
  else
    [ (n - 2 : ℚ) / (2 * (n - 1)), n / (2 * (n - 1)) ]

theorem farey_neighbors_of_half (n : ℕ) (hn : 0 < n) : 
  ∃ a b : ℚ, a ∈ farey_neighbors n ∧ b ∈ farey_neighbors n ∧ 
    (n % 2 = 1 → a = (n - 1 : ℚ) / (2 * n) ∧ b = (n + 1 : ℚ) / (2 * n)) ∧
    (n % 2 = 0 → a = (n - 2 : ℚ) / (2 * (n - 1)) ∧ b = n / (2 * (n - 1))) :=
sorry

end farey_neighbors_of_half_l163_163846


namespace paint_needed_270_statues_l163_163680

theorem paint_needed_270_statues:
  let height_large := 12
  let paint_large := 2
  let height_small := 3
  let num_statues := 270
  let ratio_height := (height_small : ℝ) / (height_large : ℝ)
  let ratio_area := ratio_height ^ 2
  let paint_small := paint_large * ratio_area
  let total_paint := num_statues * paint_small
  total_paint = 33.75 := by
  sorry

end paint_needed_270_statues_l163_163680


namespace find_x_l163_163146

theorem find_x (x : ℝ) (h : 0.35 * 400 = 0.20 * x): x = 700 :=
sorry

end find_x_l163_163146


namespace sqrt_factorial_product_l163_163597

theorem sqrt_factorial_product :
  Nat.sqrt (Nat.factorial 4 * Nat.factorial 4) = 24 := 
sorry

end sqrt_factorial_product_l163_163597


namespace expression_equality_l163_163143

theorem expression_equality :
  - (2^3) = (-2)^3 :=
by sorry

end expression_equality_l163_163143


namespace smallest_m_l163_163525

-- Let n be a positive integer and r be a positive real number less than 1/5000
def valid_r (r : ℝ) : Prop := 0 < r ∧ r < 1 / 5000

def m (n : ℕ) (r : ℝ) := (n + r)^3

theorem smallest_m : (∃ (n : ℕ) (r : ℝ), valid_r r ∧ n ≥ 41 ∧ m n r = 68922) :=
by
  sorry

end smallest_m_l163_163525


namespace logarithmic_expression_max_value_l163_163522

theorem logarithmic_expression_max_value (a b : ℝ) (h1 : a > b) (h2 : b > 1) (h3 : a / b = 3) :
  3 * Real.log (a / b) / Real.log a + 2 * Real.log (b / a) / Real.log b = -4 := 
sorry

end logarithmic_expression_max_value_l163_163522


namespace gardener_cabbages_l163_163316

theorem gardener_cabbages (area_this_year : ℕ) (side_length_this_year : ℕ) (side_length_last_year : ℕ) (area_last_year : ℕ) (additional_cabbages : ℕ) :
  area_this_year = 9801 →
  side_length_this_year = 99 →
  side_length_last_year = side_length_this_year - 1 →
  area_last_year = side_length_last_year * side_length_last_year →
  additional_cabbages = area_this_year - area_last_year →
  additional_cabbages = 197 :=
by
  sorry

end gardener_cabbages_l163_163316


namespace price_difference_pc_sm_l163_163772

-- Definitions based on given conditions
def S : ℕ := 300
def x : ℕ := sorry -- This is what we are trying to find
def PC : ℕ := S + x
def AT : ℕ := S + PC
def total_cost : ℕ := S + PC + AT

-- Theorem to be proved
theorem price_difference_pc_sm (h : total_cost = 2200) : x = 500 :=
by
  -- We would prove the theorem here
  sorry

end price_difference_pc_sm_l163_163772


namespace chalkboard_area_l163_163264

theorem chalkboard_area (width : ℝ) (h₁ : width = 3.5) (length : ℝ) (h₂ : length = 2.3 * width) : 
  width * length = 28.175 :=
by
  rw [h₁, h₂]
  norm_num
  sorry

end chalkboard_area_l163_163264


namespace expand_polynomials_l163_163340

def p (z : ℤ) := 3 * z^3 + 4 * z^2 - 2 * z + 1
def q (z : ℤ) := 2 * z^2 - 3 * z + 5
def r (z : ℤ) := 10 * z^5 - 8 * z^4 + 11 * z^3 + 5 * z^2 - 10 * z + 5

theorem expand_polynomials (z : ℤ) : (p z) * (q z) = r z :=
by sorry

end expand_polynomials_l163_163340


namespace quadratic_root_value_k_l163_163653

theorem quadratic_root_value_k (k : ℝ) :
  (
    ∃ x₁ x₂ : ℝ, x₁ = 3 ∧ x₂ = -4 / 3 ∧
    (∀ x : ℝ, x^2 * k - 8 * x - 18 = 0 ↔ (x = x₁ ∨ x = x₂))
  ) → k = 4.5 :=
by
  sorry

end quadratic_root_value_k_l163_163653


namespace book_cost_in_cny_l163_163397

-- Conditions
def usd_to_nad : ℝ := 7      -- One US dollar to Namibian dollar
def usd_to_cny : ℝ := 6      -- One US dollar to Chinese yuan
def book_cost_nad : ℝ := 168 -- Cost of the book in Namibian dollars

-- Statement to prove
theorem book_cost_in_cny : book_cost_nad * (usd_to_cny / usd_to_nad) = 144 :=
sorry

end book_cost_in_cny_l163_163397


namespace fourth_term_arithmetic_sequence_l163_163993

theorem fourth_term_arithmetic_sequence (a d : ℝ) (h : 2 * a + 2 * d = 12) : a + d = 6 := 
by
  sorry

end fourth_term_arithmetic_sequence_l163_163993


namespace shoe_cost_on_monday_l163_163327

theorem shoe_cost_on_monday 
  (price_thursday : ℝ) 
  (increase_rate : ℝ) 
  (decrease_rate : ℝ) 
  (price_thursday_eq : price_thursday = 40)
  (increase_rate_eq : increase_rate = 0.10)
  (decrease_rate_eq : decrease_rate = 0.10)
  :
  let price_friday := price_thursday * (1 + increase_rate)
  let discount := price_friday * decrease_rate
  let price_monday := price_friday - discount
  price_monday = 39.60 :=
by
  sorry

end shoe_cost_on_monday_l163_163327


namespace length_of_second_train_l163_163454

theorem length_of_second_train
  (length_first_train : ℝ)
  (speed_first_train : ℝ)
  (speed_second_train : ℝ)
  (time_to_cross : ℝ)
  (h1 : length_first_train = 270)
  (h2 : speed_first_train = 120)
  (h3 : speed_second_train = 80)
  (h4 : time_to_cross = 9) :
  ∃ length_second_train : ℝ, length_second_train = 229.95 :=
by
  sorry

end length_of_second_train_l163_163454


namespace sqrt_factorial_product_l163_163610

theorem sqrt_factorial_product:
  (Int.sqrt (Nat.factorial 4 * Nat.factorial 4)).toNat = 24 :=
by
  sorry

end sqrt_factorial_product_l163_163610


namespace quadratic_has_two_distinct_real_roots_l163_163121

theorem quadratic_has_two_distinct_real_roots (k : ℝ) :
  let a := 1
  let b := -(k + 3)
  let c := 2 * k + 1
  let Δ := b^2 - 4 * a * c
  Δ > 0 :=
by
  sorry

end quadratic_has_two_distinct_real_roots_l163_163121


namespace even_function_a_eq_one_l163_163992

noncomputable def f (x a : ℝ) : ℝ := x * Real.log (x + Real.sqrt (a + x ^ 2))

theorem even_function_a_eq_one (a : ℝ) :
  (∀ x : ℝ, f x a = f (-x) a) → a = 1 :=
by
  sorry

end even_function_a_eq_one_l163_163992


namespace f_increasing_intervals_g_range_l163_163068

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin x
noncomputable def g (x : ℝ) : ℝ := (1 + Real.sin x) * f x

theorem f_increasing_intervals : 
  (∀ x, 0 ≤ x → x ≤ Real.pi / 2 → 0 ≤ Real.cos x) ∧ (∀ x, 3 * Real.pi / 2 ≤ x → x ≤ 2 * Real.pi → 0 ≤ Real.cos x) :=
sorry

theorem g_range : 
  ∀ x, 0 ≤ x → x ≤ 2 * Real.pi → -1 / 2 ≤ g x ∧ g x ≤ 4 :=
sorry

end f_increasing_intervals_g_range_l163_163068


namespace hyperbola_equation_l163_163426

variable (a b c : ℝ)

def system_eq1 := (4 / (-3 - c)) = (- a / b)
def system_eq2 := ((c - 3) / 2) * (b / a) = 2
def system_eq3 := a ^ 2 + b ^ 2 = c ^ 2

theorem hyperbola_equation (h1 : system_eq1 a b c) (h2 : system_eq2 a b c) (h3 : system_eq3 a b c) :
  ∃ a b : ℝ, c = 5 ∧ b^2 = 20 ∧ a^2 = 5 ∧ (∀ x y : ℝ, (x ^ 2 / 5) - (y ^ 2 / 20) = 1) :=
  sorry

end hyperbola_equation_l163_163426


namespace add_trams_l163_163557

theorem add_trams (total_trams : ℕ) (total_distance : ℝ) (initial_intervals : ℝ) (new_intervals : ℝ) (additional_trams : ℕ) :
  total_trams = 12 → total_distance = 60 → initial_intervals = total_distance / total_trams →
  new_intervals = initial_intervals - (initial_intervals / 5) →
  additional_trams = (total_distance / new_intervals) - total_trams →
  additional_trams = 3 :=
begin
  intros h1 h2 h3 h4 h5,
  sorry
end

end add_trams_l163_163557


namespace complement_intersection_eq_l163_163017

def U : Set ℕ := {1, 2, 3, 4}
def M : Set ℕ := {1, 2}
def N : Set ℕ := {2, 3}

theorem complement_intersection_eq :
  (U \ (M ∩ N)) = {1, 3, 4} := by
  sorry

end complement_intersection_eq_l163_163017


namespace num_triangles_2164_l163_163677

noncomputable def is_valid_triangle (p1 p2 p3 : ℤ × ℤ) : Prop :=
  let det := (fun (x1 y1 x2 y2 x3 y3 : ℤ) => x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
  det p1.1 p1.2 p2.1 p2.2 p3.1 p3.2 ≠ 0

noncomputable def num_valid_triangles : ℕ :=
  let points := {(x, y) | 1 ≤ x ∧ x ≤ 5 ∧ 1 ≤ y ∧ y ≤ 5}.to_finset.powerset 3
  points.count (λ t, match t.elems with
    | [p1, p2, p3] => is_valid_triangle p1 p2 p3
    | _ => false
  end)

theorem num_triangles_2164 : num_valid_triangles = 2164 := by
  sorry

end num_triangles_2164_l163_163677


namespace arithmetic_seq_property_l163_163514

-- Define the arithmetic sequence {a_n}
def arithmetic_seq (a d : ℤ) (n : ℕ) : ℤ := a + n * d

-- Define the conditions
variable (a d : ℤ)
variable (h1 : arithmetic_seq a d 3 + arithmetic_seq a d 9 + arithmetic_seq a d 15 = 30)

-- Define the statement to be proved
theorem arithmetic_seq_property : 
  arithmetic_seq a d 17 - 2 * arithmetic_seq a d 13 = -10 :=
by
  sorry

end arithmetic_seq_property_l163_163514


namespace sum_of_acute_angles_l163_163468

theorem sum_of_acute_angles (angle1 angle2 angle3 angle4 angle5 angle6 : ℝ)
  (h1 : angle1 = 30) (h2 : angle2 = 30) (h3 : angle3 = 30) (h4 : angle4 = 30) (h5 : angle5 = 30) (h6 : angle6 = 30) :
  (angle1 + angle2 + angle3 + angle4 + angle5 + angle6 + 
  (angle1 + angle2) + (angle2 + angle3) + (angle3 + angle4) + (angle4 + angle5) + (angle5 + angle6)) = 480 :=
  sorry

end sum_of_acute_angles_l163_163468


namespace trigonometric_identity_l163_163820

theorem trigonometric_identity 
  (α : ℝ) 
  (h : Real.tan α = 2 * Real.tan (Real.pi / 5)) :
  (Real.cos (α - 3 * Real.pi / 10)) / (Real.sin (α - Real.pi / 5)) = 3 :=
by
  sorry

end trigonometric_identity_l163_163820


namespace problem_I_problem_II_l163_163260

open Set Real

-- Problem (I)
theorem problem_I (x : ℝ) :
  (|x - 2| ≥ 4 - |x - 1|) ↔ x ∈ Iic (-1/2) ∪ Ici (7/2) :=
by
  sorry

-- Problem (II)
theorem problem_II (m n : ℝ) (hm : 0 < m) (hn : 0 < n) (h : 1/m + 1/2/n = 1) :
  m + 2 * n ≥ 4 :=
by
  sorry

end problem_I_problem_II_l163_163260


namespace area_relationship_l163_163634

theorem area_relationship (P Q R : ℝ) (h_square : 10 * 10 = 100)
  (h_triangle1 : P + R = 50)
  (h_triangle2 : Q + R = 50) :
  P - Q = 0 :=
by
  sorry

end area_relationship_l163_163634


namespace ab_gt_ac_l163_163490

variables {a b c : ℝ}

theorem ab_gt_ac (h1 : c < b) (h2 : b < a) (h3 : a * c < 0) : a * b > a * c :=
sorry

end ab_gt_ac_l163_163490


namespace initial_integer_l163_163931

theorem initial_integer (x : ℤ) (h : (x + 2)^2 = x^2 - 2016) : x = -505 :=
by
  sorry

end initial_integer_l163_163931


namespace solve_for_x_l163_163624

def f (x : ℝ) : ℝ := 2 * x - 3

theorem solve_for_x : ∃ (x : ℝ), 2 * (f x) - 11 = f (x - 2) :=
by
  use 5
  have h1 : f 5 = 2 * 5 - 3 := rfl
  have h2 : f (5 - 2) = 2 * (5 - 2) - 3 := rfl
  simp [f] at *
  exact sorry

end solve_for_x_l163_163624


namespace tangent_line_at_e_intervals_of_monotonicity_l163_163063
open Real

noncomputable def f (x : ℝ) : ℝ := x * log x

theorem tangent_line_at_e :
  ∃ (y : ℝ → ℝ), (∀ x : ℝ, y x = 2 * x - exp 1) ∧ (y (exp 1) = f (exp 1)) ∧ (deriv f (exp 1) = deriv y (exp 1)) :=
sorry

theorem intervals_of_monotonicity :
  (∀ x : ℝ, 0 < x ∧ x < exp (-1) → deriv f x < 0) ∧ (∀ x : ℝ, exp (-1) < x → deriv f x > 0) :=
sorry

end tangent_line_at_e_intervals_of_monotonicity_l163_163063


namespace sqrt_factorial_mul_factorial_l163_163589

theorem sqrt_factorial_mul_factorial :
  (Real.sqrt (Nat.factorial 4 * Nat.factorial 4) = 24) := by
  sorry

end sqrt_factorial_mul_factorial_l163_163589


namespace option_C_correct_l163_163619

theorem option_C_correct (a : ℤ) : (a = 3 → a = a + 1 → a = 4) :=
by {
  sorry
}

end option_C_correct_l163_163619


namespace probability_of_B_l163_163239

-- Define the events and their probabilities according to the problem description
def A₁ := "Event where a red ball is taken from bag A"
def A₂ := "Event where a white ball is taken from bag A"
def A₃ := "Event where a black ball is taken from bag A"
def B := "Event where a red ball is taken from bag B"

-- Types of bags A and B containing balls
structure Bag where
  red : Nat
  white : Nat
  black : Nat

-- Initial bags
def bagA : Bag := ⟨ 3, 2, 5 ⟩
def bagB : Bag := ⟨ 3, 3, 4 ⟩

-- Probabilities of each event in bagA
def P_A₁ : ℚ := 3 / 10
def P_A₂ : ℚ := 2 / 10
def P_A₃ : ℚ := 5 / 10

-- Probability of event B under conditions A₁, A₂, A₃
def P_B_given_A₁ : ℚ := 4 / 11
def P_B_given_A₂ : ℚ := 3 / 11
def P_B_given_A₃ : ℚ := 3 / 11

-- Goal: Prove that the probability of drawing a red ball from bag B (P(B)) is 3/10
theorem probability_of_B : 
  (P_A₁ * P_B_given_A₁ + P_A₂ * P_B_given_A₂ + P_A₃ * P_B_given_A₃) = (3 / 10) :=
by
  -- Placeholder for the proof
  sorry

end probability_of_B_l163_163239


namespace sqrt_factorial_product_l163_163581

/-- Define the factorial of 4 -/
def factorial_four : ℕ := 4!

/-- Define the product of the factorial of 4 with itself -/
def product_of_factorials : ℕ := factorial_four * factorial_four

/-- Prove the value of the square root of product_of_factorials is 24 -/
theorem sqrt_factorial_product : Real.sqrt (product_of_factorials) = 24 := by
  have fact_4_eq_24 : factorial_four = 24 := by norm_num
  rw [product_of_factorials, fact_4_eq_24, Nat.mul_self_eq, Real.sqrt_sq]
  norm_num
  exact Nat.zero_le 24

end sqrt_factorial_product_l163_163581


namespace cube_surface_area_l163_163628

theorem cube_surface_area (side_length : ℝ) (h : side_length = 8) : 6 * side_length^2 = 384 :=
by
  rw [h]
  sorry

end cube_surface_area_l163_163628


namespace vector_sum_to_zero_l163_163936

variable {V : Type}
variable [AddCommGroup V] [Module ℝ V] {A B C : V}

theorem vector_sum_to_zero (AB BC CA : V) (hAB : AB = B - A) (hBC : BC = C - B) (hCA : CA = A - C) :
  AB + BC + CA = 0 := by
  sorry

end vector_sum_to_zero_l163_163936


namespace solve_inequality_l163_163660

theorem solve_inequality (x : ℝ) :
  (2 * x - 1) / (3 * x + 1) > 0 ↔ x < -1/3 ∨ x > 1/2 :=
  sorry

end solve_inequality_l163_163660


namespace evaluate_expression_l163_163339

theorem evaluate_expression : -30 + 5 * (9 / (3 + 3)) = -22.5 := sorry

end evaluate_expression_l163_163339


namespace inequality_problem_l163_163219

-- Given a < b < 0, we want to prove a^2 > ab > b^2
theorem inequality_problem (a b : ℝ) (h : a < b ∧ b < 0) : a^2 > ab ∧ ab > b^2 :=
sorry

end inequality_problem_l163_163219


namespace problem1_problem2_l163_163104

-- Sub-problem 1
theorem problem1 (x y : ℝ) (h1 : 9 * x + 10 * y = 1810) (h2 : 11 * x + 8 * y = 1790) : 
  x - y = -10 := 
sorry

-- Sub-problem 2
theorem problem2 (x y : ℝ) (h1 : 2 * x + 2.5 * y = 1200) (h2 : 1000 * x + 900 * y = 530000) :
  x = 350 ∧ y = 200 := 
sorry

end problem1_problem2_l163_163104


namespace pyramid_base_edge_length_l163_163728

theorem pyramid_base_edge_length (height : ℝ) (radius : ℝ) (side_len : ℝ) :
  height = 4 ∧ radius = 3 →
  side_len = (12 * Real.sqrt 14) / 7 :=
by
  intros h
  rcases h with ⟨h1, h2⟩
  sorry

end pyramid_base_edge_length_l163_163728


namespace tan_triple_angle_l163_163988

theorem tan_triple_angle (θ : ℝ) (h : Real.tan θ = 1/3) : Real.tan (3 * θ) = 13/9 :=
by
  sorry

end tan_triple_angle_l163_163988


namespace hexagon_transformations_l163_163172

-- Definitions for transformations
def R_a : equiv.perm (fin 6) := equiv.perm.rotate (fin 6) 1
def R_b : equiv.perm (fin 6) := R_a ^ 2
def H : equiv.perm (fin 6) :=
equiv.of_bijective
  (λ (i : fin 6), (if i.val < 3 then 5 - i.val else 5 - i.val))
  ⟨by
  { intros x y h,
    dsimp at *,
    split_ifs at h with h1 h2,
    { exact h },
    { exact h },
    { exact h },
    { exact h }, },
  by
  { intro y,
    dsimp,
    split_ifs with h1,
    { use 5 - y.val,
      simp [fin.ext_iff] },
    { use 5 - y.val,
      simp [fin.ext_iff], } }⟩
def V : equiv.perm (fin 6) :=
equiv.of_bijective
  (λ (i : fin 6), (if i.val % 2 == 1 then (i.val - 1) else (i.val + 1)))
  ⟨by
  { intros x y h,
    dsimp at *,
    split_ifs at h with h1 h2,
    { exact h },
    { exact h },
    { exact h },
    { exact h }, },
  by
  { intro y,
    dsimp,
    split_ifs with h1,
    { use (y.val + 1),
      simp [fin.ext_iff] },
    { use (y.val - 1),
      simp [fin.ext_iff], } }⟩

def transformations : list (equiv.perm (fin 6)) := [R_a, R_b, H, V]

def count_sequences : ℕ := (4 : ℕ)^24

theorem hexagon_transformations : (count_sequences = 4^24) :=
by sorry

end hexagon_transformations_l163_163172


namespace dan_helmet_craters_l163_163944

namespace HelmetCraters

variables {Dan Daniel Rin : ℕ}

/-- Condition 1: Dan's skateboarding helmet has ten more craters than Daniel's ski helmet. -/
def condition1 (C_d C_daniel : ℕ) : Prop := C_d = C_daniel + 10

/-- Condition 2: Rin's snorkel helmet has 15 more craters than Dan's and Daniel's helmets combined. -/
def condition2 (C_r C_d C_daniel : ℕ) : Prop := C_r = C_d + C_daniel + 15

/-- Condition 3: Rin's helmet has 75 craters. -/
def condition3 (C_r : ℕ) : Prop := C_r = 75

/-- The main theorem: Dan's skateboarding helmet has 35 craters given the conditions. -/
theorem dan_helmet_craters (C_d C_daniel C_r : ℕ) 
    (h1 : condition1 C_d C_daniel) 
    (h2 : condition2 C_r C_d C_daniel) 
    (h3 : condition3 C_r) : C_d = 35 :=
by {
    -- We state that the answer is 35 based on the conditions
    sorry
}

end HelmetCraters

end dan_helmet_craters_l163_163944


namespace sqrt_factorial_product_l163_163609

theorem sqrt_factorial_product:
  (Int.sqrt (Nat.factorial 4 * Nat.factorial 4)).toNat = 24 :=
by
  sorry

end sqrt_factorial_product_l163_163609


namespace households_with_only_bike_l163_163302

theorem households_with_only_bike
  (N : ℕ) (H_no_car_or_bike : ℕ) (H_car_bike : ℕ) (H_car : ℕ)
  (hN : N = 90)
  (h_no_car_or_bike : H_no_car_or_bike = 11)
  (h_car_bike : H_car_bike = 16)
  (h_car : H_car = 44) :
  ∃ (H_bike_only : ℕ), H_bike_only = 35 :=
by {
  sorry
}

end households_with_only_bike_l163_163302


namespace ratio_of_square_sides_l163_163897

theorem ratio_of_square_sides (a b c : ℕ) (h : ratio_area = (75 / 128) := sorry) (ratio_area :
 ratio_side = sqrt (75 / 128) := sorry) : a + b + c = 27 :=
begin
  sorry
end

end ratio_of_square_sides_l163_163897


namespace ratio_proof_l163_163374

-- Definitions and conditions
variables {A B C : ℕ}

-- Given condition: A : B : C = 3 : 2 : 5
def ratio_cond (A B C : ℕ) := 3 * B = 2 * A ∧ 5 * B = 2 * C

-- Theorem statement
theorem ratio_proof (h : ratio_cond A B C) : (2 * A + 3 * B) / (A + 5 * C) = 3 / 7 :=
by sorry

end ratio_proof_l163_163374


namespace fourth_term_arithmetic_sequence_l163_163994

theorem fourth_term_arithmetic_sequence (a d : ℝ) (h : 2 * a + 2 * d = 12) : a + d = 6 := 
by
  sorry

end fourth_term_arithmetic_sequence_l163_163994


namespace sqrt_factorial_eq_l163_163607

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem sqrt_factorial_eq :
  sqrt (factorial 4 * factorial 4) = 24 :=
by
  have h : factorial 4 = 24 := by
    unfold factorial
    simpa using [factorial, factorial, factorial]
  rw [h, h]
  sorry

end sqrt_factorial_eq_l163_163607


namespace pen_price_first_day_l163_163768

theorem pen_price_first_day (x y : ℕ) 
  (h1 : x * y = (x - 1) * (y + 100)) 
  (h2 : x * y = (x + 2) * (y - 100)) : x = 4 :=
by
  sorry

end pen_price_first_day_l163_163768


namespace train_seat_count_l163_163324

theorem train_seat_count (t : ℝ) (h1 : 0.20 * t = 0.2 * t)
  (h2 : 0.60 * t = 0.6 * t) (h3 : 30 + 0.20 * t + 0.60 * t = t) : t = 150 :=
by
  sorry

end train_seat_count_l163_163324


namespace solution_set_l163_163122

def within_bounds (x : ℝ) : Prop := |2 * x + 1| < 1

theorem solution_set : {x : ℝ | within_bounds x} = {x : ℝ | -1 < x ∧ x < 0} :=
by
  sorry

end solution_set_l163_163122


namespace sqrt_factorial_product_l163_163576

theorem sqrt_factorial_product:
  sqrt ((fact 4) * (fact 4)) = 24 :=
by sorry

end sqrt_factorial_product_l163_163576


namespace cyclic_quadrilaterals_count_l163_163567

theorem cyclic_quadrilaterals_count :
  ∃ n : ℕ, n = 568 ∧
  ∀ (a b c d : ℕ), 
    a + b + c + d = 32 ∧
    a ≤ b ∧ b ≤ c ∧ c ≤ d ∧
    a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧
    (a + b + c > d) ∧ (b + c + d > a) ∧ (c + d + a > b) ∧ (d + a + b > c) ∧
    (c - a)^2 + (d - b)^2 = (c + a)^2 + (d + b)^2
      → n = 568 := 
sorry

end cyclic_quadrilaterals_count_l163_163567


namespace correct_polynomial_multiplication_l163_163566

theorem correct_polynomial_multiplication (a b : ℤ) (x : ℝ)
  (h1 : 2 * b - 3 * a = 11)
  (h2 : 2 * b + a = -9) :
  (2 * x + a) * (3 * x + b) = 6 * x^2 - 19 * x + 10 := by
  sorry

end correct_polynomial_multiplication_l163_163566


namespace perpendicular_condition_l163_163190

theorem perpendicular_condition (a : ℝ) :
  (2 * a * x + (a - 1) * y + 2 = 0) ∧ ((a + 1) * x + 3 * a * y + 3 = 0) →
  (a = 1/5 ↔ ∃ x y: ℝ, ((- (2 * a / (a - 1))) * (-(a + 1) / (3 * a)) = -1)) :=
by
  sorry

end perpendicular_condition_l163_163190


namespace collinear_a_b_l163_163211

-- Define the vectors a and b
def a : ℝ × ℝ := (-1, 2)
def b : ℝ × ℝ := (1, -2)

-- Definition of collinearity of vectors
def collinear (u v : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, u = (k * v.1, k * v.2) ∨ v = (k * u.1, k * u.2)

-- Statement to prove
theorem collinear_a_b : collinear a b :=
by
  sorry

end collinear_a_b_l163_163211


namespace gcd_computation_l163_163194

theorem gcd_computation (a b : ℕ) (h₁ : a = 7260) (h₂ : b = 540) : 
  Nat.gcd a b - 12 + 5 = 53 :=
by
  rw [h₁, h₂]
  sorry

end gcd_computation_l163_163194


namespace remainder_of_6_pow_50_mod_215_l163_163951

theorem remainder_of_6_pow_50_mod_215 :
  (6 ^ 50) % 215 = 36 := 
sorry

end remainder_of_6_pow_50_mod_215_l163_163951


namespace no_solution_for_inequalities_l163_163107

theorem no_solution_for_inequalities :
  ¬ ∃ (x y : ℝ), 4 * x^2 + 4 * x * y + 19 * y^2 ≤ 2 ∧ x - y ≤ -1 :=
by
  sorry

end no_solution_for_inequalities_l163_163107


namespace john_share_l163_163711

theorem john_share
  (total_amount : ℝ)
  (john_ratio jose_ratio binoy_ratio : ℝ)
  (total_amount_eq : total_amount = 6000)
  (ratios_eq : john_ratio = 2 ∧ jose_ratio = 4 ∧ binoy_ratio = 6) :
  (john_ratio / (john_ratio + jose_ratio + binoy_ratio)) * total_amount = 1000 :=
by
  -- Here we would derive the proof, but just use sorry for the moment.
  sorry

end john_share_l163_163711


namespace ratio_upstream_downstream_l163_163125

noncomputable def ratio_time_upstream_to_downstream
  (V_b V_s : ℕ) (T_u T_d : ℕ) : ℕ :=
(V_b + V_s) / (V_b - V_s)

theorem ratio_upstream_downstream
  (V_b V_s : ℕ) (hVb : V_b = 48) (hVs : V_s = 16) (T_u T_d : ℕ)
  (hT : ratio_time_upstream_to_downstream V_b V_s T_u T_d = 2) :
  T_u / T_d = 2 := by
  sorry

end ratio_upstream_downstream_l163_163125


namespace schedule_problem_l163_163215

def num_schedule_ways : Nat :=
  -- total ways to pick 3 out of 6 periods and arrange 3 courses
  let total_ways := Nat.choose 6 3 * Nat.factorial 3
  -- at least two consecutive courses (using Principle of Inclusion and Exclusion)
  let two_consecutive := 5 * 6 * 4
  let three_consecutive := 4 * 6
  let invalid_ways := two_consecutive + three_consecutive
  total_ways - invalid_ways

theorem schedule_problem (h : num_schedule_ways = 24) : num_schedule_ways = 24 := by {
  exact h
}

end schedule_problem_l163_163215


namespace trapezoid_area_division_l163_163511

theorem trapezoid_area_division (AD BC MN : ℝ) (h₁ : AD = 4) (h₂ : BC = 3)
  (h₃ : MN > 0) (area_ratio : ∃ (S_ABMD S_MBCN : ℝ), MN/BC = (S_ABMD + S_MBCN)/(S_ABMD) ∧ (S_ABMD/S_MBCN = 2/5)) :
  MN = Real.sqrt 14 :=
by
  sorry

end trapezoid_area_division_l163_163511


namespace area_of_rhombus_l163_163414

-- Defining the lengths of the diagonals
variable (d1 d2 : ℝ)
variable (d1_eq : d1 = 15)
variable (d2_eq : d2 = 20)

-- Goal is to prove the area given the diagonal lengths
theorem area_of_rhombus (d1 d2 : ℝ) (d1_eq : d1 = 15) (d2_eq : d2 = 20) : 
  (d1 * d2) / 2 = 150 := 
by
  -- Using the given conditions for the proof
  sorry

end area_of_rhombus_l163_163414


namespace find_wholesale_prices_l163_163155

-- Definitions based on conditions
def bakerPlusPrice (p : ℕ) (d : ℕ) : ℕ := p + d
def starPrice (p : ℕ) (k : ℚ) : ℕ := (p * k).toNat

-- The tuple of prices given
def prices : List ℕ := [64, 64, 70, 72]

-- Statement of the problem
theorem find_wholesale_prices (d : ℕ) (k : ℚ) (h_d : d > 0) (h_k : k > 1 ∧ k.denom = 1):
  ∃ p1 p2, 
  (p1 ∈ prices ∧ bakerPlusPrice p1 d ∈ prices ∧ starPrice p1 k ∈ prices) ∧
  (p2 ∈ prices ∧ bakerPlusPrice p2 d ∈ prices ∧ starPrice p2 k ∈ prices) :=
sorry

end find_wholesale_prices_l163_163155


namespace third_side_length_l163_163357

def lengths (a b : ℕ) : Prop :=
a = 4 ∧ b = 10

def triangle_inequality (a b c : ℕ) : Prop :=
a + b > c ∧ abs (a - b) < c

theorem third_side_length (x : ℕ) (h1 : lengths 4 10) (h2 : triangle_inequality 4 10 x) : x = 11 :=
sorry

end third_side_length_l163_163357


namespace find_next_score_l163_163641

def scores := [95, 85, 75, 65, 90]
def current_avg := (95 + 85 + 75 + 65 + 90) / 5
def target_avg := current_avg + 4

theorem find_next_score (s : ℕ) (h : (95 + 85 + 75 + 65 + 90 + s) / 6 = target_avg) : s = 106 :=
by
  -- Proof steps here
  sorry

end find_next_score_l163_163641


namespace incorrect_selection_method_l163_163841

-- Define the instance of the problem.
def classOfStudents : Type := Fin 50 -- Finite type representing the class of 50 students.
def classPresident : classOfStudents := 0 -- Assume 0 is the class president.
def vicePresident : classOfStudents := 1 -- Assume 1 is the vice president.

-- Define the selection condition, i.e., at least one of the class president or vice president must be chosen.
def condition (selected : Finset classOfStudents) : Prop :=
  classPresident ∈ selected ∨ vicePresident ∈ selected ∧ selected.card = 5

-- Define the combination selection function for verification.
def combination (N K : ℕ) : ℕ := Nat.choose N K

-- The statement to be proved: Option C (C_{2}^{1}C_{49}^{4}) is incorrect.
theorem incorrect_selection_method :
  ¬ (combination 2 1 * combination 49 4 = combination 50 5 - combination 48 5 ∧
     combination 2 1 * combination 48 4 + combination 2 2 * combination 48 3 ∧
     combination 2 1 * combination 49 4 - combination 48 3) := 
sorry

end incorrect_selection_method_l163_163841


namespace sum_of_fractions_limit_one_l163_163098

theorem sum_of_fractions_limit_one :
  (∑' (a : ℕ), ∑' (b : ℕ), (1 : ℝ) / ((a + 1) : ℝ) ^ (b + 1)) = 1 := 
sorry

end sum_of_fractions_limit_one_l163_163098


namespace principal_amount_l163_163799

theorem principal_amount
(SI : ℝ) (R : ℝ) (T : ℝ) (P : ℝ)
(h₀ : SI = 800)
(h₁ : R = 0.08)
(h₂ : T = 1)
(h₃ : SI = P * R * T) : P = 10000 :=
by
  sorry

end principal_amount_l163_163799


namespace quadratic_root_shift_l163_163698

theorem quadratic_root_shift (d e : ℝ) :
  (∀ r s : ℝ, (r^2 - 2 * r + 0.5 = 0) → (r-3)^2 + (r-3) * (s-3) * d + e = 0) → e = 3.5 := 
by
  intros
  sorry

end quadratic_root_shift_l163_163698


namespace initial_integer_value_l163_163928

theorem initial_integer_value (x : ℤ) (h : (x + 2) * (x + 2) = x * x - 2016) : x = -505 := 
sorry

end initial_integer_value_l163_163928


namespace hillary_descending_rate_is_1000_l163_163674

-- Definitions from the conditions
def base_to_summit_distance : ℕ := 5000
def hillary_departure_time : ℕ := 6
def hillary_climbing_rate : ℕ := 800
def eddy_climbing_rate : ℕ := 500
def hillary_stop_distance_from_summit : ℕ := 1000
def hillary_and_eddy_pass_time : ℕ := 12

-- Derived definitions
def hillary_climbing_time : ℕ := (base_to_summit_distance - hillary_stop_distance_from_summit) / hillary_climbing_rate
def hillary_stop_time : ℕ := hillary_departure_time + hillary_climbing_time
def eddy_climbing_time_at_pass : ℕ := hillary_and_eddy_pass_time - hillary_departure_time
def eddy_climbed_distance : ℕ := eddy_climbing_rate * eddy_climbing_time_at_pass
def hillary_distance_descended_at_pass : ℕ := (base_to_summit_distance - hillary_stop_distance_from_summit) - eddy_climbed_distance
def hillary_descending_time : ℕ := hillary_and_eddy_pass_time - hillary_stop_time 

def hillary_descending_rate : ℕ := hillary_distance_descended_at_pass / hillary_descending_time

-- Statement to prove
theorem hillary_descending_rate_is_1000 : hillary_descending_rate = 1000 := 
by
  sorry

end hillary_descending_rate_is_1000_l163_163674


namespace polynomial_divisibility_l163_163662

theorem polynomial_divisibility (m : ℤ) : (4 * m + 5) ^ 2 - 9 ∣ 8 := by
  sorry

end polynomial_divisibility_l163_163662


namespace difference_between_numbers_l163_163898

open Int

theorem difference_between_numbers (A B : ℕ) 
  (h1 : A + B = 1812) 
  (h2 : A = 7 * B + 4) : 
  A - B = 1360 :=
by
  sorry

end difference_between_numbers_l163_163898


namespace oreo_shop_purchases_l163_163176

theorem oreo_shop_purchases :
  let oreos := 7 in
  let milks := 4 in
  let total_flavors := oreos + milks in
  ∃ (total_ways : ℕ), 
    total_ways = 
      (nat.choose total_flavors 4)
      + (nat.choose total_flavors 3 * oreos)
      + (nat.choose total_flavors 2 * (nat.choose oreos 2 + oreos))
      + (nat.choose total_flavors 1 * (nat.choose oreos 3 + (oreos * (oreos - 1) * 3) / 2 + oreos))
      + (nat.choose oreos 4 + nat.choose oreos 2 + (oreos * (oreos - 1)) + oreos)
  ∧ total_ways = 4978 :=
by
  sorry

end oreo_shop_purchases_l163_163176


namespace cost_combination_exists_l163_163094

/-!
Given:
- Nadine spent a total of $105.
- The table costs $34.
- The mirror costs $15.
- The lamp costs $6.
- The total cost of the 2 chairs and 3 decorative vases is $50.

Prove:
- There are multiple combinations of individual chair cost (C) and individual vase cost (V) such that 2 * C + 3 * V = 50.
-/

theorem cost_combination_exists :
  ∃ (C V : ℝ), 2 * C + 3 * V = 50 :=
by {
  sorry
}

end cost_combination_exists_l163_163094


namespace roots_of_cubic_eq_sum_l163_163694

namespace MathProof

open Real

theorem roots_of_cubic_eq_sum :
  ∀ a b c : ℝ, 
  (Polynomial.eval a (Polynomial.C 4 * Polynomial.X ^ 3 + Polynomial.C 2023 * Polynomial.X + Polynomial.C 4012) = 0) ∧
  (Polynomial.eval b (Polynomial.C 4 * Polynomial.X ^ 3 + Polynomial.C 2023 * Polynomial.X + Polynomial.C 4012) = 0) ∧
  (Polynomial.eval c (Polynomial.C 4 * Polynomial.X ^ 3 + Polynomial.C 2023 * Polynomial.X + Polynomial.C 4012) = 0) 
  → (a + b)^3 + (b + c)^3 + (c + a)^3 = 3009 :=
by
  intros a b c h
  sorry

end MathProof

end roots_of_cubic_eq_sum_l163_163694


namespace slope_of_line_through_points_l163_163201

theorem slope_of_line_through_points 
  (t : ℝ) 
  (x y : ℝ) 
  (h1 : 3 * x + 4 * y = 12 * t + 6) 
  (h2 : 2 * x + 3 * y = 8 * t - 1) : 
  ∃ m b : ℝ, (∀ t : ℝ, y = m * x + b) ∧ m = 0 :=
by 
  sorry

end slope_of_line_through_points_l163_163201


namespace circles_C_D_intersect_prob_l163_163565

open MeasureTheory

-- Define the distribution of C_X and D_X
noncomputable def uniform_dist (a b : ℝ) : Measure ℝ := 
  MeasureTheory.measure (Set.Icc a b)

-- Integration bounds and Probability calculation
noncomputable def probability_intersect_C_D : ℝ :=
  ∫ x in 0..3, 
    (min (4 : ℝ) (x + sqrt 5) - max (1 : ℝ) (x - sqrt 5)) / 3

-- Main theorem
theorem circles_C_D_intersect_prob :
  probability_intersect_C_D = (* insert the correct answer here *) :=
begin
  sorry
end

end circles_C_D_intersect_prob_l163_163565
