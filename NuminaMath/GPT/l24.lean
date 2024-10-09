import Mathlib

namespace arith_geo_seq_prop_l24_2490

theorem arith_geo_seq_prop (a1 a2 b1 b2 b3 : ℝ)
  (arith_seq_condition : 1 + 2 * (a1 - 1) = a2)
  (geo_seq_condition1 : b1 * b3 = 4)
  (geo_seq_condition2 : b1 > 0)
  (geo_seq_condition3 : 1 * b1 * b2 * b3 * 4 = (b1 * b3 * -4)) :
  (a2 - a1) / b2 = 1/2 :=
by
  sorry

end arith_geo_seq_prop_l24_2490


namespace nine_chapters_problem_l24_2415

theorem nine_chapters_problem (n x : ℤ) (h1 : 8 * n = x + 3) (h2 : 7 * n = x - 4) :
  (x + 3) / 8 = (x - 4) / 7 :=
  sorry

end nine_chapters_problem_l24_2415


namespace ambiguous_times_l24_2493

theorem ambiguous_times (h m : ℝ) : 
  (∃ k l : ℕ, 0 ≤ k ∧ k < 12 ∧ 0 ≤ l ∧ l < 12 ∧ 
              (12 * h = k * 360 + m) ∧ 
              (12 * m = l * 360 + h) ∧
              k ≠ l) → 
  (∃ n : ℕ, n = 132) := 
sorry

end ambiguous_times_l24_2493


namespace Felicity_family_store_visits_l24_2422

theorem Felicity_family_store_visits
  (lollipop_stick : ℕ := 1)
  (fort_total_sticks : ℕ := 400)
  (fort_completion_percent : ℕ := 60)
  (weeks_collected : ℕ := 80)
  (sticks_collected : ℕ := (fort_total_sticks * fort_completion_percent) / 100)
  (store_visits_per_week : ℕ := sticks_collected / weeks_collected) :
  store_visits_per_week = 3 := by
  sorry

end Felicity_family_store_visits_l24_2422


namespace part1_part2_l24_2441

variable (m x : ℝ)

-- Condition: mx - 3 > 2x + m
def inequality1 := m * x - 3 > 2 * x + m

-- Part (1) Condition: x < (m + 3) / (m - 2)
def solution_set_part1 := x < (m + 3) / (m - 2)

-- Part (2) Condition: 2x - 1 > 3 - x
def inequality2 := 2 * x - 1 > 3 - x

theorem part1 (h : ∀ x, inequality1 m x → solution_set_part1 m x) : m < 2 :=
sorry

theorem part2 (h1 : ∀ x, inequality1 m x ↔ inequality2 x) : m = 17 :=
sorry

end part1_part2_l24_2441


namespace coordinate_equation_solution_l24_2454

theorem coordinate_equation_solution (x y : ℝ) :
  2 * x^2 + y^2 + 3 * x * y + 3 * x + y = 2 →
  (y = -x - 2) ∨ (y = -2 * x + 1) :=
by
  sorry

end coordinate_equation_solution_l24_2454


namespace hexagon_angles_l24_2478

theorem hexagon_angles
  (AB CD EF BC DE FA : ℝ)
  (F A B C D E : Type*)
  (FAB ABC EFA CDE : ℝ)
  (h1 : AB = CD)
  (h2 : AB = EF)
  (h3 : BC = DE)
  (h4 : BC = FA)
  (h5 : FAB + ABC = 240)
  (h6 : FAB + EFA = 240) :
  FAB + CDE = 240 :=
sorry

end hexagon_angles_l24_2478


namespace length_of_AD_l24_2438

theorem length_of_AD (AB BC CD DE : ℝ) (right_angle_B right_angle_C : Prop) :
  AB = 6 → BC = 7 → CD = 25 → DE = 15 → AD = Real.sqrt 274 :=
by
  intros
  sorry

end length_of_AD_l24_2438


namespace value_of_n_l24_2488

theorem value_of_n (n : ℕ) : (1 / 5 : ℝ) ^ n * (1 / 4 : ℝ) ^ 18 = 1 / (2 * (10 : ℝ) ^ 35) → n = 35 :=
by
  intro h
  sorry

end value_of_n_l24_2488


namespace tory_sells_grandmother_l24_2408

theorem tory_sells_grandmother (G : ℕ)
    (total_goal : ℕ) (sold_to_uncle : ℕ) (sold_to_neighbor : ℕ) (remaining_to_sell : ℕ)
    (h_goal : total_goal = 50) (h_sold_to_uncle : sold_to_uncle = 7)
    (h_sold_to_neighbor : sold_to_neighbor = 5) (h_remaining_to_sell : remaining_to_sell = 26) :
    (G + sold_to_uncle + sold_to_neighbor + remaining_to_sell = total_goal) → G = 12 :=
by
    intros h
    -- Proof goes here
    sorry

end tory_sells_grandmother_l24_2408


namespace profit_rate_is_five_percent_l24_2412

theorem profit_rate_is_five_percent (cost_price selling_price : ℝ) (hx : 1.1 * cost_price - 10 = 210) : 
  (selling_price = 1.1 * cost_price) → 
  (selling_price - cost_price) / cost_price * 100 = 5 :=
by
  sorry

end profit_rate_is_five_percent_l24_2412


namespace simplify_abs_neg_pow_sub_l24_2407

theorem simplify_abs_neg_pow_sub (a b : ℤ) (h : a = 4) (h' : b = 6) : 
  (|-(a ^ 2) - b| = 22) := 
by
  sorry

end simplify_abs_neg_pow_sub_l24_2407


namespace total_inflation_over_two_years_real_interest_rate_over_two_years_l24_2496

section FinancialCalculations

-- Define the known conditions
def annual_inflation_rate : ℚ := 0.025
def nominal_interest_rate : ℚ := 0.06

-- Prove the total inflation rate over two years equals 5.0625%
theorem total_inflation_over_two_years :
  ((1 + annual_inflation_rate)^2 - 1) * 100 = 5.0625 :=
sorry

-- Prove the real interest rate over two years equals 6.95%
theorem real_interest_rate_over_two_years :
  ((1 + nominal_interest_rate) * (1 + nominal_interest_rate) / (1 + (annual_inflation_rate * annual_inflation_rate)) - 1) * 100 = 6.95 :=
sorry

end FinancialCalculations

end total_inflation_over_two_years_real_interest_rate_over_two_years_l24_2496


namespace reciprocal_of_neg_one_div_2023_l24_2459

theorem reciprocal_of_neg_one_div_2023 : 1 / (-1 / (2023 : ℤ)) = -2023 := sorry

end reciprocal_of_neg_one_div_2023_l24_2459


namespace a3_pm_2b3_not_div_by_37_l24_2442

theorem a3_pm_2b3_not_div_by_37 {a b : ℤ} (ha : ¬ (37 ∣ a)) (hb : ¬ (37 ∣ b)) :
  ¬ (37 ∣ (a^3 + 2 * b^3)) ∧ ¬ (37 ∣ (a^3 - 2 * b^3)) :=
  sorry

end a3_pm_2b3_not_div_by_37_l24_2442


namespace quadratic_inequality_solution_l24_2401

theorem quadratic_inequality_solution
  (x : ℝ) :
  -2 * x^2 + x < -3 ↔ x ∈ Set.Iio (-1) ∪ Set.Ioi (3 / 2) := by
  sorry

end quadratic_inequality_solution_l24_2401


namespace divisibility_by_5_l24_2450

theorem divisibility_by_5 (x y : ℤ) : (x^2 - 2 * x * y + 2 * y^2) % 5 = 0 ∨ (x^2 + 2 * x * y + 2 * y^2) % 5 = 0 ↔ (x % 5 = 0 ∧ y % 5 = 0) ∨ (x % 5 ≠ 0 ∧ y % 5 ≠ 0) := 
by
  sorry

end divisibility_by_5_l24_2450


namespace inequality_solution_l24_2439

theorem inequality_solution (x : ℝ)
  (h : ∀ x, x^2 + 2 * x + 7 > 0) :
  (x - 3) / (x^2 + 2 * x + 7) ≥ 0 ↔ x ∈ Set.Ici 3 :=
by
  sorry

end inequality_solution_l24_2439


namespace simplify_expression_l24_2435

theorem simplify_expression
  (cos : ℝ → ℝ)
  (sin : ℝ → ℝ)
  (cos_double_angle : ∀ x, cos (2 * x) = cos x * cos x - sin x * sin x)
  (sin_double_angle : ∀ x, sin (2 * x) = 2 * sin x * cos x)
  (sin_cofunction : ∀ x, sin (Real.pi / 2 - x) = cos x) :
  (cos 5 * cos 5 - sin 5 * sin 5) / (sin 40 * cos 40) = 2 := by
  sorry

end simplify_expression_l24_2435


namespace find_m_direct_proportion_l24_2445

theorem find_m_direct_proportion (m : ℝ) (h1 : m + 2 ≠ 0) (h2 : |m| - 1 = 1) : m = 2 :=
sorry

end find_m_direct_proportion_l24_2445


namespace problem_proof_l24_2420

theorem problem_proof (x : ℝ) (hx : x + 1/x = 7) : (x - 3)^2 + 49/((x - 3)^2) = 23 := by
  sorry

end problem_proof_l24_2420


namespace smallest_lambda_inequality_l24_2433

theorem smallest_lambda_inequality 
  (a b c d : ℝ) (h_pos : ∀ x ∈ [a, b, c, d], 0 < x) (h_sum : a + b + c + d = 4) :
  5 * (a*b + a*c + a*d + b*c + b*d + c*d) ≤ 8 * (a*b*c*d) + 12 :=
sorry

end smallest_lambda_inequality_l24_2433


namespace x_squared_inverse_y_fourth_l24_2460

theorem x_squared_inverse_y_fourth (x y : ℝ) (k : ℝ) (h₁ : x = 8) (h₂ : y = 2) (h₃ : (x^2) * (y^4) = k) : x^2 = 4 :=
by
  sorry

end x_squared_inverse_y_fourth_l24_2460


namespace problem_statement_l24_2499

noncomputable def f (x : ℝ) : ℝ :=
if x ≥ 0 then x * (x + 4) else x * (x - 4)

theorem problem_statement (a : ℝ) (h : f a > f (8 - a)) : 4 < a :=
by sorry

end problem_statement_l24_2499


namespace andy_questions_wrong_l24_2482

variable (a b c d : ℕ)

theorem andy_questions_wrong
  (h1 : a + b = c + d)
  (h2 : a + d = b + c + 6)
  (h3 : c = 7)
  (h4 : d = 9) :
  a = 10 :=
by {
  sorry  -- Proof would go here
}

end andy_questions_wrong_l24_2482


namespace train_crossing_time_l24_2448

theorem train_crossing_time 
    (length : ℝ) (speed_kmph : ℝ) 
    (conversion_factor: ℝ) (speed_mps: ℝ) 
    (time : ℝ) :
  length = 400 ∧ speed_kmph = 144 ∧ conversion_factor = 1000 / 3600 ∧ speed_mps = speed_kmph * conversion_factor ∧ time = length / speed_mps → time = 10 := 
by 
  sorry

end train_crossing_time_l24_2448


namespace division_problem_l24_2432

theorem division_problem : (5 * 8) / 10 = 4 := by
  sorry

end division_problem_l24_2432


namespace trapezoids_not_necessarily_congruent_l24_2410

-- Define trapezoid structure
structure Trapezoid (α : Type) [LinearOrderedField α] :=
(base1 base2 side1 side2 diag1 diag2 : α) -- sides and diagonals
(angle1 angle2 angle3 angle4 : α)        -- internal angles

-- Conditions about given trapezoids
variables {α : Type} [LinearOrderedField α]
variables (T1 T2 : Trapezoid α)

-- The condition that corresponding angles of the trapezoids are equal
def equal_angles := 
  T1.angle1 = T2.angle1 ∧ T1.angle2 = T2.angle2 ∧ 
  T1.angle3 = T2.angle3 ∧ T1.angle4 = T2.angle4

-- The condition that diagonals of the trapezoids are equal
def equal_diagonals := 
  T1.diag1 = T2.diag1 ∧ T1.diag2 = T2.diag2

-- The statement to prove
theorem trapezoids_not_necessarily_congruent :
  equal_angles T1 T2 ∧ equal_diagonals T1 T2 → ¬ (T1 = T2) := by
  sorry

end trapezoids_not_necessarily_congruent_l24_2410


namespace clarence_oranges_l24_2486

def initial_oranges := 5
def oranges_from_joyce := 3
def total_oranges := initial_oranges + oranges_from_joyce

theorem clarence_oranges : total_oranges = 8 :=
  by
  sorry

end clarence_oranges_l24_2486


namespace recurring_fraction_difference_l24_2471

theorem recurring_fraction_difference :
  let x := (36 / 99 : ℚ)
  let y := (36 / 100 : ℚ)
  x - y = (1 / 275 : ℚ) :=
by
  sorry

end recurring_fraction_difference_l24_2471


namespace calculation_correct_l24_2458

theorem calculation_correct : 2 * (3 ^ 2) ^ 4 = 13122 := by
  sorry

end calculation_correct_l24_2458


namespace number_representation_correct_l24_2469

-- Conditions: 5 in both the tenths and hundredths places, 0 in remaining places.
def number : ℝ := 50.05

theorem number_representation_correct :
  number = 50.05 :=
by 
  -- The proof will show that the definition satisfies the condition.
  sorry

end number_representation_correct_l24_2469


namespace find_b_l24_2449

theorem find_b (b : ℚ) (h : ∃ c : ℚ, (3 * x + c)^2 = 9 * x^2 + 27 * x + b) : b = 81 / 4 := 
sorry

end find_b_l24_2449


namespace sofia_total_cost_l24_2423

def shirt_cost : ℕ := 7
def shoes_cost : ℕ := shirt_cost + 3
def two_shirts_cost : ℕ := 2 * shirt_cost
def total_clothes_cost : ℕ := two_shirts_cost + shoes_cost
def bag_cost : ℕ := total_clothes_cost / 2
def total_cost : ℕ := two_shirts_cost + shoes_cost + bag_cost

theorem sofia_total_cost : total_cost = 36 := by
  sorry

end sofia_total_cost_l24_2423


namespace multiplication_value_l24_2400

theorem multiplication_value (x : ℝ) (h : (2.25 / 3) * x = 9) : x = 12 :=
by
  sorry

end multiplication_value_l24_2400


namespace circles_intersect_condition_l24_2417

theorem circles_intersect_condition (a : ℝ) (ha : a > 0) : 
  (∃ x y : ℝ, x^2 + y^2 = 1 ∧ (x - a)^2 + y^2 = 16) ↔ 3 < a ∧ a < 5 :=
by sorry

end circles_intersect_condition_l24_2417


namespace solve_math_problem_l24_2404

-- Math problem definition
def math_problem (A : ℝ) : Prop :=
  (0 < A ∧ A < (Real.pi / 2)) ∧ (Real.cos A = 3 / 5) →
  Real.sin (2 * A) = 24 / 25

-- Example theorem statement in Lean
theorem solve_math_problem (A : ℝ) : math_problem A :=
sorry

end solve_math_problem_l24_2404


namespace baseball_team_groups_l24_2440

theorem baseball_team_groups
  (new_players : ℕ) 
  (returning_players : ℕ)
  (players_per_group : ℕ)
  (total_players : ℕ := new_players + returning_players) :
  new_players = 48 → 
  returning_players = 6 → 
  players_per_group = 6 → 
  total_players / players_per_group = 9 :=
by
  intros h1 h2 h3
  subst h1
  subst h2
  subst h3
  sorry

end baseball_team_groups_l24_2440


namespace even_numbers_average_19_l24_2426

theorem even_numbers_average_19 (n : ℕ) (h1 : (n / 2) * (2 + 2 * n) / n = 19) : n = 18 :=
by {
  sorry
}

end even_numbers_average_19_l24_2426


namespace range_of_a_l24_2424

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, -2 < x ∧ x < -1 → (a + x) * (1 + x) < 0) → a > 2 :=
by
  sorry

end range_of_a_l24_2424


namespace train_crosses_lamp_post_in_30_seconds_l24_2470

open Real

/-- Prove that given a train that crosses a 2500 m long bridge in 120 s and has a length of
    833.33 m, it takes the train 30 seconds to cross a lamp post. -/
theorem train_crosses_lamp_post_in_30_seconds (L_train : ℝ) (L_bridge : ℝ) (T_bridge : ℝ) (T_lamp_post : ℝ)
  (hL_train : L_train = 833.33)
  (hL_bridge : L_bridge = 2500)
  (hT_bridge : T_bridge = 120)
  (ht : T_lamp_post = (833.33 / ((833.33 + 2500) / 120))) :
  T_lamp_post = 30 :=
by
  sorry

end train_crosses_lamp_post_in_30_seconds_l24_2470


namespace number_of_teams_l24_2402

theorem number_of_teams (n : ℕ) (h : (n * (n - 1)) / 2 = 21) : n = 7 :=
sorry

end number_of_teams_l24_2402


namespace range_of_a_l24_2481

theorem range_of_a (a : ℝ) (x : ℝ) :
  (¬(x > a) →¬(x^2 + 2*x - 3 > 0)) → (a ≥ 1 ) :=
by
  intro h
  sorry

end range_of_a_l24_2481


namespace correct_exponent_calculation_l24_2425

theorem correct_exponent_calculation : 
(∀ (a b : ℝ), (a + b)^2 ≠ a^2 + b^2) ∧
(∀ (a : ℝ), a^9 / a^3 ≠ a^3) ∧
(∀ (a b : ℝ), (ab)^3 = a^3 * b^3) ∧
(∀ (a : ℝ), (a^5)^2 ≠ a^7) :=
by 
  sorry

end correct_exponent_calculation_l24_2425


namespace exists_rational_non_integer_a_not_exists_rational_non_integer_b_l24_2472

-- Define rational non-integer numbers
def is_rational_non_integer (x : ℚ) : Prop := ¬(∃ (z : ℤ), x = z)

-- (a) Proof for existance of rational non-integer numbers y and x such that 19x + 8y, 8x + 3y are integers
theorem exists_rational_non_integer_a :
  ∃ x y : ℚ, is_rational_non_integer x ∧ is_rational_non_integer y ∧ (∃ a b : ℤ, 19 * x + 8 * y = a ∧ 8 * x + 3 * y = b) :=
sorry

-- (b) Proof for non-existance of rational non-integer numbers y and x such that 19x² + 8y², 8x² + 3y² are integers
theorem not_exists_rational_non_integer_b :
  ¬ ∃ x y : ℚ, is_rational_non_integer x ∧ is_rational_non_integer y ∧ (∃ m n : ℤ, 19 * x^2 + 8 * y^2 = m ∧ 8 * x^2 + 3 * y^2 = n) :=
sorry

end exists_rational_non_integer_a_not_exists_rational_non_integer_b_l24_2472


namespace correct_parameterizations_of_line_l24_2451

theorem correct_parameterizations_of_line :
  ∀ (t : ℝ),
    (∀ (x y : ℝ), ((x = 5/3) ∧ (y = 0) ∨ (x = 0) ∧ (y = -5) ∨ (x = -5/3) ∧ (y = 0) ∨ 
                   (x = 1) ∧ (y = -2) ∨ (x = -2) ∧ (y = -11)) → 
                   y = 3 * x - 5) ∧
    (∀ (a b : ℝ), ((a = 1) ∧ (b = 3) ∨ (a = 3) ∧ (b = 1) ∨ (a = -1) ∧ (b = -3) ∨
                   (a = 1/3) ∧ (b = 1)) → 
                   b = 3 * a) →
    -- Check only Options D and E
    ((x = 1) → (y = -2) → (a = 1/3) → (b = 1) → y = 3 * x - 5 ∧ b = 3 * a) ∨
    ((x = -2) → (y = -11) → (a = 1/3) → (b = 1) → y = 3 * x - 5 ∧ b = 3 * a) :=
by
  sorry

end correct_parameterizations_of_line_l24_2451


namespace smallest_possible_k_l24_2413

def infinite_increasing_seq (a : ℕ → ℕ) : Prop :=
∀ n, a n < a (n + 1)

def divisible_by_1005_or_1006 (a : ℕ) : Prop :=
a % 1005 = 0 ∨ a % 1006 = 0

def not_divisible_by_97 (a : ℕ) : Prop :=
a % 97 ≠ 0

def diff_less_than_k (a : ℕ → ℕ) (k : ℕ) : Prop :=
∀ n, (a (n + 1) - a n) ≤ k

theorem smallest_possible_k :
  ∀ (a : ℕ → ℕ), infinite_increasing_seq a →
  (∀ n, divisible_by_1005_or_1006 (a n)) →
  (∀ n, not_divisible_by_97 (a n)) →
  (∃ k, diff_less_than_k a k) →
  (∃ k, k = 2010 ∧ diff_less_than_k a k) :=
by
  sorry

end smallest_possible_k_l24_2413


namespace surface_area_of_circumscribed_sphere_l24_2468

theorem surface_area_of_circumscribed_sphere :
  let a := 2
  let AD := Real.sqrt (a^2 - (a/2)^2)
  let r := Real.sqrt (1 + 1 + AD^2) / 2
  4 * Real.pi * r^2 = 5 * Real.pi := by
  sorry

end surface_area_of_circumscribed_sphere_l24_2468


namespace kim_min_pours_l24_2428

-- Define the initial conditions
def initial_volume (V : ℝ) : ℝ := V
def pour (V : ℝ) : ℝ := 0.9 * V

-- Define the remaining volume after n pours
def remaining_volume (V : ℝ) (n : ℕ) : ℝ := V * (0.9)^n

-- State the problem: After 7 pours, the remaining volume is less than half the initial volume
theorem kim_min_pours (V : ℝ) (hV : V > 0) : remaining_volume V 7 < V / 2 :=
by
  -- Because the proof is not required, we use sorry
  sorry

end kim_min_pours_l24_2428


namespace regression_equation_is_correct_l24_2463

theorem regression_equation_is_correct 
  (linear_corr : ∃ (f : ℝ → ℝ), ∀ (x : ℝ), ∃ (y : ℝ), y = f x)
  (mean_b : ℝ)
  (mean_x : ℝ)
  (mean_y : ℝ)
  (mean_b_eq : mean_b = 0.51)
  (mean_x_eq : mean_x = 61.75)
  (mean_y_eq : mean_y = 38.14) : 
  mean_y = mean_b * mean_x + 6.65 :=
sorry

end regression_equation_is_correct_l24_2463


namespace total_cost_of_coat_l24_2467

def original_price : ℝ := 150
def sale_discount : ℝ := 0.25
def additional_discount : ℝ := 10
def sales_tax : ℝ := 0.10

theorem total_cost_of_coat :
  let sale_price := original_price * (1 - sale_discount)
  let price_after_discount := sale_price - additional_discount
  let final_price := price_after_discount * (1 + sales_tax)
  final_price = 112.75 :=
by
  -- sorry for the actual proof
  sorry

end total_cost_of_coat_l24_2467


namespace four_mutually_acquainted_l24_2416

theorem four_mutually_acquainted (G : SimpleGraph (Fin 9)) 
  (h : ∀ (s : Finset (Fin 9)), s.card = 3 → ∃ (u v : Fin 9), u ∈ s ∧ v ∈ s ∧ G.Adj u v) :
  ∃ (s : Finset (Fin 9)), s.card = 4 ∧ ∀ (u v : Fin 9), u ∈ s → v ∈ s → G.Adj u v :=
by
  sorry

end four_mutually_acquainted_l24_2416


namespace cost_price_books_l24_2430

def cost_of_type_A (cost_A cost_B : ℝ) : Prop :=
  cost_A = cost_B + 15

def quantity_equal (cost_A cost_B : ℝ) : Prop :=
  675 / cost_A = 450 / cost_B

theorem cost_price_books (cost_A cost_B : ℝ) (h1 : cost_of_type_A cost_A cost_B) (h2 : quantity_equal cost_A cost_B) : 
  cost_A = 45 ∧ cost_B = 30 :=
by
  -- Proof omitted
  sorry

end cost_price_books_l24_2430


namespace difference_between_numbers_l24_2487

theorem difference_between_numbers : 
  ∃ (a : ℕ), a + 10 * a = 30000 → 9 * a = 24543 := 
by 
  sorry

end difference_between_numbers_l24_2487


namespace reciprocal_of_mixed_num_l24_2419

-- Define the fraction representation of the mixed number -1 1/2
def mixed_num_to_improper (a : ℚ) : ℚ := -3/2

-- Define the reciprocal function
def reciprocal (x : ℚ) : ℚ := 1 / x

-- Prove the statement
theorem reciprocal_of_mixed_num : reciprocal (mixed_num_to_improper (-1.5)) = -2/3 :=
by
  -- skip proof
  sorry

end reciprocal_of_mixed_num_l24_2419


namespace area_of_yard_proof_l24_2484

def area_of_yard (L W : ℕ) : ℕ :=
  L * W

theorem area_of_yard_proof (L W : ℕ) (hL : L = 40) (hFence : 2 * W + L = 52) : 
  area_of_yard L W = 240 := 
by 
  sorry

end area_of_yard_proof_l24_2484


namespace card_probability_l24_2414

-- Definitions of the conditions
def is_multiple (n d : ℕ) : Prop := d ∣ n

def count_multiples (d m : ℕ) : ℕ := (m / d)

def multiples_in_range (n : ℕ) : ℕ := 
  count_multiples 2 n + count_multiples 3 n + count_multiples 5 n
  - count_multiples 6 n - count_multiples 10 n - count_multiples 15 n 
  + count_multiples 30 n

def probability_of_multiples_in_range (n : ℕ) : ℚ := 
  multiples_in_range n / n 

-- Proof statement
theorem card_probability (n : ℕ) (h : n = 120) : probability_of_multiples_in_range n = 11 / 15 :=
  sorry

end card_probability_l24_2414


namespace original_number_of_men_l24_2485

theorem original_number_of_men (x : ℕ) (h : 10 * x = 7 * (x + 10)) : x = 24 := 
by 
  -- Add your proof here 
  sorry

end original_number_of_men_l24_2485


namespace star_4_3_l24_2446

def star (a b : ℤ) : ℤ := a^2 - a * b + b^2

theorem star_4_3 : star 4 3 = 13 :=
by
  sorry

end star_4_3_l24_2446


namespace dennis_initial_money_l24_2453

def initial_money (shirt_cost: ℕ) (ten_dollar_bills: ℕ) (loose_coins: ℕ) : ℕ :=
  shirt_cost + (10 * ten_dollar_bills) + loose_coins

theorem dennis_initial_money : initial_money 27 2 3 = 50 :=
by 
  -- Here would go the proof steps based on the solution steps identified before
  sorry

end dennis_initial_money_l24_2453


namespace total_birds_in_store_l24_2429

def num_bird_cages := 4
def parrots_per_cage := 8
def parakeets_per_cage := 2
def birds_per_cage := parrots_per_cage + parakeets_per_cage
def total_birds := birds_per_cage * num_bird_cages

theorem total_birds_in_store : total_birds = 40 :=
  by sorry

end total_birds_in_store_l24_2429


namespace complete_the_square_l24_2489

theorem complete_the_square (x : ℝ) : 
  (x^2 - 8 * x + 10 = 0) → 
  ((x - 4)^2 = 6) :=
sorry

end complete_the_square_l24_2489


namespace machine_produces_one_item_in_40_seconds_l24_2465

theorem machine_produces_one_item_in_40_seconds :
  (60 * 1) / 90 * 60 = 40 :=
by
  sorry

end machine_produces_one_item_in_40_seconds_l24_2465


namespace fraction_of_180_l24_2473

theorem fraction_of_180 : (1 / 2) * (1 / 3) * (1 / 6) * 180 = 5 := by
  sorry

end fraction_of_180_l24_2473


namespace joe_paid_4_more_than_jenny_l24_2437

theorem joe_paid_4_more_than_jenny
  (total_plain_pizza_cost : ℕ := 12) 
  (total_slices : ℕ := 12)
  (additional_cost_per_mushroom_slice : ℕ := 1) -- 0.50 dollars represented in integer (value in cents or minimal currency unit)
  (mushroom_slices : ℕ := 4) 
  (plain_slices := total_slices - mushroom_slices) -- Calculate plain slices.
  (total_additional_cost := mushroom_slices * additional_cost_per_mushroom_slice)
  (total_pizza_cost := total_plain_pizza_cost + total_additional_cost)
  (plain_slice_cost := total_plain_pizza_cost / total_slices)
  (mushroom_slice_cost := plain_slice_cost + additional_cost_per_mushroom_slice) 
  (joe_mushroom_slices := mushroom_slices) 
  (joe_plain_slices := 3) 
  (jenny_plain_slices := plain_slices - joe_plain_slices) 
  (joe_paid := (joe_mushroom_slices * mushroom_slice_cost) + (joe_plain_slices * plain_slice_cost))
  (jenny_paid := jenny_plain_slices * plain_slice_cost) : 
  joe_paid - jenny_paid = 4 := 
by {
  -- Here, we define the steps we used to calculate the cost.
  sorry -- Proof skipped as per instructions.
}

end joe_paid_4_more_than_jenny_l24_2437


namespace inequality_proof_l24_2479

theorem inequality_proof (x y z : ℝ) (hx : 1 ≤ x) (hy : 1 ≤ y) (hz : 1 ≤ z) 
  (h : 1 / (x^2 - 1) + 1 / (y^2 - 1) + 1 / (z^2 - 1) = 1) :
  1 / (x + 1) + 1 / (y + 1) + 1 / (z + 1) ≤ 1 := 
by {
  sorry
}

end inequality_proof_l24_2479


namespace arithmetic_sequence_sum_l24_2436

-- Let {a_n} be an arithmetic sequence.
-- Define Sn as the sum of the first n terms.
def is_arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n+1) = a n + d

def S (a : ℕ → ℤ) (n : ℕ) : ℤ :=
  (n * (a 0 + a (n-1))) / 2

theorem arithmetic_sequence_sum
  (a : ℕ → ℤ)
  (h_arith : is_arithmetic_sequence a)
  (h_condition : 2 * a 6 = a 7 + 5) :
  S a 11 = 55 :=
sorry

end arithmetic_sequence_sum_l24_2436


namespace inverse_proportion_function_range_m_l24_2497

theorem inverse_proportion_function_range_m
  (x1 x2 y1 y2 m : ℝ)
  (h_func_A : y1 = (5 * m - 2) / x1)
  (h_func_B : y2 = (5 * m - 2) / x2)
  (h_x : x1 < x2)
  (h_x_neg : x2 < 0)
  (h_y : y1 < y2) :
  m < 2 / 5 :=
sorry

end inverse_proportion_function_range_m_l24_2497


namespace sum_of_three_numbers_is_71_point_5_l24_2427

noncomputable def sum_of_three_numbers (a b c : ℝ) : ℝ :=
a + b + c

theorem sum_of_three_numbers_is_71_point_5 (a b c : ℝ) (h1 : a + b = 35) (h2 : b + c = 48) (h3 : c + a = 60) :
  sum_of_three_numbers a b c = 71.5 :=
by
  unfold sum_of_three_numbers
  sorry

end sum_of_three_numbers_is_71_point_5_l24_2427


namespace diff_squares_of_roots_l24_2434

theorem diff_squares_of_roots : ∀ α β : ℝ, (α * β = 6) ∧ (α + β = 5) -> (α - β)^2 = 1 := by
  sorry

end diff_squares_of_roots_l24_2434


namespace no_integer_pairs_satisfy_equation_l24_2495

theorem no_integer_pairs_satisfy_equation :
  ∀ (m n : ℤ), m^3 + 6 * m^2 + 5 * m ≠ 27 * n^3 + 27 * n^2 + 9 * n + 1 :=
by
  intros m n
  sorry

end no_integer_pairs_satisfy_equation_l24_2495


namespace correct_average_of_10_numbers_l24_2464

theorem correct_average_of_10_numbers
  (incorrect_avg : ℕ)
  (n : ℕ)
  (incorrect_read : ℕ)
  (correct_read : ℕ)
  (incorrect_total_sum : ℕ) :
  incorrect_avg = 19 →
  n = 10 →
  incorrect_read = 26 →
  correct_read = 76 →
  incorrect_total_sum = incorrect_avg * n →
  (correct_total_sum : ℕ) = incorrect_total_sum - incorrect_read + correct_read →
  (correct_avg : ℕ) = correct_total_sum / n →
  correct_avg = 24 :=
by
  intros
  sorry

end correct_average_of_10_numbers_l24_2464


namespace arithmetic_sequence_sum_l24_2461

variable {a : ℕ → ℝ} 

-- Condition: Arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d, ∀ n, a (n + 1) = a n + d

-- Condition: Given sum of specific terms in the sequence
def given_condition (a : ℕ → ℝ) : Prop :=
  a 2 + a 10 = 16

-- Problem: Proving the correct answer
theorem arithmetic_sequence_sum (a : ℕ → ℝ) (h1 : is_arithmetic_sequence a) (h2 : given_condition a) :
  a 4 + a 6 + a 8 = 24 :=
by
  sorry

end arithmetic_sequence_sum_l24_2461


namespace group_size_increase_by_4_l24_2474

theorem group_size_increase_by_4
    (N : ℕ)
    (weight_old : ℕ)
    (weight_new : ℕ)
    (average_increase : ℕ)
    (weight_increase_diff : ℕ)
    (h1 : weight_old = 55)
    (h2 : weight_new = 87)
    (h3 : average_increase = 4)
    (h4 : weight_increase_diff = weight_new - weight_old)
    (h5 : average_increase * N = weight_increase_diff) :
    N = 8 :=
by
  sorry

end group_size_increase_by_4_l24_2474


namespace find_odd_number_between_30_and_50_with_remainder_2_when_divided_by_7_l24_2498

def isOdd (n : ℕ) : Prop := n % 2 = 1
def isInRange (n : ℕ) : Prop := 30 ≤ n ∧ n ≤ 50
def hasRemainderTwo (n : ℕ) : Prop := n % 7 = 2

theorem find_odd_number_between_30_and_50_with_remainder_2_when_divided_by_7 :
  ∃ n : ℕ, isInRange n ∧ isOdd n ∧ hasRemainderTwo n ∧ n = 37 :=
by
  sorry

end find_odd_number_between_30_and_50_with_remainder_2_when_divided_by_7_l24_2498


namespace necessary_but_not_sufficient_not_sufficient_l24_2494

def P (x : ℝ) : Prop := x < 1
def Q (x : ℝ) : Prop := (x + 2) * (x - 1) < 0

theorem necessary_but_not_sufficient (x : ℝ) : P x → Q x := by
  intro hx
  sorry

theorem not_sufficient (x : ℝ) : ¬(Q x → P x) := by
  intro hq
  sorry

end necessary_but_not_sufficient_not_sufficient_l24_2494


namespace outstanding_consumer_installment_credit_l24_2455

-- Given conditions
def total_consumer_installment_credit (C : ℝ) : Prop :=
  let automobile_installment_credit := 0.36 * C
  let automobile_finance_credit := 75
  let total_automobile_credit := 2 * automobile_finance_credit
  automobile_installment_credit = total_automobile_credit

-- Theorem to prove
theorem outstanding_consumer_installment_credit : ∃ (C : ℝ), total_consumer_installment_credit C ∧ C = 416.67 := 
by
  sorry

end outstanding_consumer_installment_credit_l24_2455


namespace trigonometric_identity_proof_l24_2480

variable (α : Real)

theorem trigonometric_identity_proof (h1 : Real.tan α = 4 / 3) (h2 : 0 < α ∧ α < Real.pi / 2) :
  Real.sin (Real.pi + α) + Real.cos (Real.pi - α) = -7 / 5 :=
by
  sorry

end trigonometric_identity_proof_l24_2480


namespace isosceles_triangle_CBD_supplement_l24_2444

/-- Given an isosceles triangle ABC with AC = BC and angle C = 50 degrees,
    and point D such that angle CBD is supplementary to angle ABC,
    prove that angle CBD is 115 degrees. -/
theorem isosceles_triangle_CBD_supplement 
  (A B C D : Type) [DecidableEq A] [DecidableEq B] [DecidableEq C] [DecidableEq D]
  (AC BC : ℝ) (angleBAC angleABC angleC angleCBD : ℝ)
  (isosceles : AC = BC)
  (angle_C_eq : angleC = 50)
  (supplement : angleCBD = 180 - angleABC) :
  angleCBD = 115 :=
sorry

end isosceles_triangle_CBD_supplement_l24_2444


namespace simplify_and_evaluate_expression_l24_2421

theorem simplify_and_evaluate_expression (x : ℝ) (h : x = 3) : 
  ((x^2 / (x - 2) - x - 2) / (4 * x / (x^2 - 4))) = (5 : ℝ) / 3 := 
by
  sorry

end simplify_and_evaluate_expression_l24_2421


namespace triangle_DOE_area_l24_2492

theorem triangle_DOE_area
  (area_ABC : ℝ)
  (DO : ℝ) (OB : ℝ)
  (EO : ℝ) (OA : ℝ)
  (h_area_ABC : area_ABC = 1)
  (h_DO_OB : DO / OB = 1 / 3)
  (h_EO_OA : EO / OA = 4 / 5)
  : (1 / 4) * (4 / 9) * area_ABC = 11 / 135 := 
by 
  sorry

end triangle_DOE_area_l24_2492


namespace year_2049_is_Jisi_l24_2452

-- Define Heavenly Stems
def HeavenlyStems : List String := ["Jia", "Yi", "Bing", "Ding", "Wu", "Ji", "Geng", "Xin", "Ren", "Gui"]

-- Define Earthly Branches
def EarthlyBranches : List String := ["Zi", "Chou", "Yin", "Mao", "Chen", "Si", "Wu", "Wei", "Shen", "You", "Xu", "Hai"]

-- Define the indices of Ding (丁) and You (酉) based on 2017
def Ding_index : Nat := 3
def You_index : Nat := 9

-- Define the year difference
def year_difference : Nat := 2049 - 2017

-- Calculate the indices for the Heavenly Stem and Earthly Branch in 2049
def HeavenlyStem_index_2049 : Nat := (Ding_index + year_difference) % 10
def EarthlyBranch_index_2049 : Nat := (You_index + year_difference) % 12

theorem year_2049_is_Jisi : 
  HeavenlyStems[HeavenlyStem_index_2049]? = some "Ji" ∧ EarthlyBranches[EarthlyBranch_index_2049]? = some "Si" :=
by
  sorry

end year_2049_is_Jisi_l24_2452


namespace largest_multiple_of_7_l24_2466

def repeated_188 (k : Nat) : ℕ := (List.replicate k 188).foldr (λ x acc => x * 1000 + acc) 0

theorem largest_multiple_of_7 :
  ∃ n, n = repeated_188 100 ∧ ∃ m, m ≤ 303 ∧ m ≥ 0 ∧ m ≠ 300 ∧ (repeated_188 m % 7 = 0 → n ≥ repeated_188 m) :=
by
  sorry

end largest_multiple_of_7_l24_2466


namespace trig_sum_identity_l24_2491

theorem trig_sum_identity :
  Real.sin (47 * Real.pi / 180) * Real.cos (43 * Real.pi / 180) 
  + Real.sin (137 * Real.pi / 180) * Real.sin (43 * Real.pi / 180) = 1 :=
by
  sorry

end trig_sum_identity_l24_2491


namespace joao_speed_l24_2462

theorem joao_speed (d : ℝ) (v1 : ℝ) (t1 t2 : ℝ) (h1 : v1 = 10) (h2 : t1 = 6 / 60) (h3 : t2 = 8 / 60) : 
  d = v1 * t1 → d = 10 * (6 / 60) → (d / t2) = 7.5 := 
by
  sorry

end joao_speed_l24_2462


namespace sugar_amount_first_week_l24_2411

theorem sugar_amount_first_week (s : ℕ → ℕ) (h : s 4 = 3) (h_rec : ∀ n, s (n + 1) = s n / 2) : s 1 = 24 :=
by
  sorry

end sugar_amount_first_week_l24_2411


namespace min_max_expr_l24_2475

noncomputable def expr (a b c : ℝ) : ℝ :=
  (1 / (a^2 + 1) + 1 / (b^2 + 1) + 1 / (c^2 + 1)) *
  (a^2 / (a^2 + 1) + b^2 / (b^2 + 1) + c^2 / (c^2 + 1))

theorem min_max_expr (a b c : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) (h_cond : a * b + b * c + c * a = 1) :
  27 / 16 ≤ expr a b c ∧ expr a b c ≤ 2 :=
sorry

end min_max_expr_l24_2475


namespace complex_multiplication_l24_2406

theorem complex_multiplication (i : ℂ) (h : i^2 = -1) : i * (2 - i) = 1 + 2 * i := by
  sorry

end complex_multiplication_l24_2406


namespace line_ellipse_common_points_l24_2483

theorem line_ellipse_common_points (m : ℝ) : (m ≥ 1 ∧ m ≠ 5) ↔ (∀ k : ℝ, ∃ x y : ℝ, y = k * x + 1 ∧ (x^2 / 5) + (y^2 / m) = 1) :=
by 
  sorry

end line_ellipse_common_points_l24_2483


namespace number_of_people_going_on_trip_l24_2405

theorem number_of_people_going_on_trip
  (bags_per_person : ℕ)
  (weight_per_bag : ℕ)
  (total_luggage_capacity : ℕ)
  (additional_capacity : ℕ)
  (bags_per_additional_capacity : ℕ)
  (h1 : bags_per_person = 5)
  (h2 : weight_per_bag = 50)
  (h3 : total_luggage_capacity = 6000)
  (h4 : additional_capacity = 90) :
  (total_luggage_capacity + (bags_per_additional_capacity * weight_per_bag)) / (weight_per_bag * bags_per_person) = 42 := 
by
  simp [h1, h2, h3, h4]
  repeat { sorry }

end number_of_people_going_on_trip_l24_2405


namespace solve_for_T_l24_2418

theorem solve_for_T : ∃ T : ℝ, (3 / 4) * (1 / 6) * T = (2 / 5) * (1 / 4) * 200 ∧ T = 80 :=
by
  use 80
  -- The proof part is omitted as instructed
  sorry

end solve_for_T_l24_2418


namespace inequality_solution_set_nonempty_range_l24_2456

theorem inequality_solution_set_nonempty_range (a : ℝ) :
  (∃ x : ℝ, (a^2 - 4) * x^2 + (a + 2) * x - 1 ≥ 0) ↔ (a ≤ -2 ∨ a ≥ 6 / 5) :=
by
  -- Proof is omitted
  sorry

end inequality_solution_set_nonempty_range_l24_2456


namespace identity_proof_l24_2457

theorem identity_proof (x y : ℝ) (h1 : x + y = 5 / 11) (h2 : x - y = 1 / 55) : x^2 - y^2 = 1 / 121 :=
by 
  sorry

end identity_proof_l24_2457


namespace subset_bound_l24_2443

theorem subset_bound {m n k : ℕ} (h1 : m ≥ n) (h2 : n > 1) 
  (F : Fin k → Finset (Fin m)) 
  (hF : ∀ i j, i < j → (F i ∩ F j).card ≤ 1) 
  (hcard : ∀ i, (F i).card = n) : 
  k ≤ (m * (m - 1)) / (n * (n - 1)) :=
sorry

end subset_bound_l24_2443


namespace infinite_bad_numbers_l24_2403

-- Define types for natural numbers
variables {a b : ℕ}

-- The theorem statement
theorem infinite_bad_numbers (a b : ℕ) : ∃ᶠ (n : ℕ) in at_top, n > 0 ∧ ¬ (n^b + 1 ∣ a^n + 1) :=
sorry

end infinite_bad_numbers_l24_2403


namespace initial_nickels_l24_2409

theorem initial_nickels (quarters : ℕ) (initial_nickels : ℕ) (borrowed_nickels : ℕ) (current_nickels : ℕ) 
  (H1 : initial_nickels = 87) (H2 : borrowed_nickels = 75) (H3 : current_nickels = 12) : 
  initial_nickels = current_nickels + borrowed_nickels := 
by 
  -- proof steps go here
  sorry

end initial_nickels_l24_2409


namespace eighteenth_prime_l24_2476

-- Define the necessary statements
def isPrime (n : ℕ) : Prop := sorry

def primeSeq (n : ℕ) : ℕ :=
  if n = 0 then
    2
  else if n = 1 then
    3
  else
    -- Function to generate the n-th prime number
    sorry

theorem eighteenth_prime :
  primeSeq 17 = 67 := by
  sorry

end eighteenth_prime_l24_2476


namespace total_marks_l24_2431

variable (E S M : Nat)

-- Given conditions
def thrice_as_many_marks_in_English_as_in_Science := E = 3 * S
def ratio_of_marks_in_English_and_Maths            := M = 4 * E
def marks_in_Science                               := S = 17

-- Proof problem statement
theorem total_marks (h1 : E = 3 * S) (h2 : M = 4 * E) (h3 : S = 17) :
  E + S + M = 272 :=
by
  sorry

end total_marks_l24_2431


namespace find_x_l24_2477

theorem find_x (x : ℝ) (h1 : x > 0) (h2 : 1/2 * (2 * x) * x = 72) : x = 6 * Real.sqrt 2 :=
by
  sorry

end find_x_l24_2477


namespace ratio_of_third_to_second_l24_2447

-- Assume we have three numbers (a, b, c) where
-- 1. b = 2 * a
-- 2. c = k * b
-- 3. (a + b + c) / 3 = 165
-- 4. a = 45

theorem ratio_of_third_to_second (a b c k : ℝ) (h1 : b = 2 * a) (h2 : c = k * b) 
  (h3 : (a + b + c) / 3 = 165) (h4 : a = 45) : k = 4 := by 
  sorry

end ratio_of_third_to_second_l24_2447
