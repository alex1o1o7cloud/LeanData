import Mathlib

namespace NUMINAMATH_GPT_minimum_value_expression_l381_38135

theorem minimum_value_expression (x : ℝ) (h : -3 < x ∧ x < 2) :
  ∃ y, y = (x^2 + 4 * x + 5) / (2 * x + 6) ∧ y = 3 / 4 :=
by
  sorry

end NUMINAMATH_GPT_minimum_value_expression_l381_38135


namespace NUMINAMATH_GPT_valid_elixir_combinations_l381_38155

theorem valid_elixir_combinations :
  let herbs := 4
  let crystals := 6
  let incompatible_herbs := 3
  let incompatible_crystals := 2
  let total_combinations := herbs * crystals
  let incompatible_combinations := incompatible_herbs * incompatible_crystals
  total_combinations - incompatible_combinations = 18 :=
by
  sorry

end NUMINAMATH_GPT_valid_elixir_combinations_l381_38155


namespace NUMINAMATH_GPT_Xiao_Ming_max_notebooks_l381_38115

-- Definitions of the given conditions
def total_yuan : ℝ := 30
def total_books : ℕ := 30
def notebook_cost : ℝ := 4
def exercise_book_cost : ℝ := 0.4

-- Definition of the variables used in the inequality
def x (max_notebooks : ℕ) : ℝ := max_notebooks
def exercise_books (max_notebooks : ℕ) : ℝ := total_books - x max_notebooks

-- Definition of the total cost inequality
def total_cost (max_notebooks : ℕ) : ℝ :=
  x max_notebooks * notebook_cost + exercise_books max_notebooks * exercise_book_cost

theorem Xiao_Ming_max_notebooks (max_notebooks : ℕ) : total_cost max_notebooks ≤ total_yuan → max_notebooks ≤ 5 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_Xiao_Ming_max_notebooks_l381_38115


namespace NUMINAMATH_GPT_solve_system_of_inequalities_l381_38184

theorem solve_system_of_inequalities {x : ℝ} :
  (x + 3 ≥ 2) ∧ (2 * (x + 4) > 4 * x + 2) ↔ (-1 ≤ x ∧ x < 3) :=
by
  sorry

end NUMINAMATH_GPT_solve_system_of_inequalities_l381_38184


namespace NUMINAMATH_GPT_find_f_three_l381_38197

variable {α : Type*} [LinearOrderedField α]

def f (a b c x : α) := a * x^5 - b * x^3 + c * x - 3

theorem find_f_three (a b c : α) (h : f a b c (-3) = 7) : f a b c 3 = -13 :=
by sorry

end NUMINAMATH_GPT_find_f_three_l381_38197


namespace NUMINAMATH_GPT_allan_balloons_l381_38144

def initial_balloons : ℕ := 5
def additional_balloons : ℕ := 3
def total_balloons : ℕ := initial_balloons + additional_balloons

theorem allan_balloons :
  total_balloons = 8 :=
sorry

end NUMINAMATH_GPT_allan_balloons_l381_38144


namespace NUMINAMATH_GPT_radian_measure_of_central_angle_l381_38125

-- Given conditions
variables (l r : ℝ)
variables (h1 : (1 / 2) * l * r = 1)
variables (h2 : 2 * r + l = 4)

-- The theorem to prove
theorem radian_measure_of_central_angle (l r : ℝ) (h1 : (1 / 2) * l * r = 1) (h2 : 2 * r + l = 4) : 
  l / r = 2 :=
by 
  -- Proof steps are not provided as per the requirement
  sorry

end NUMINAMATH_GPT_radian_measure_of_central_angle_l381_38125


namespace NUMINAMATH_GPT_perimeter_of_excircle_opposite_leg_l381_38156

noncomputable def perimeter_of_right_triangle (a varrho_a : ℝ) : ℝ :=
  2 * varrho_a * a / (2 * varrho_a - a)

theorem perimeter_of_excircle_opposite_leg
  (a varrho_a : ℝ) (h_a_pos : 0 < a) (h_varrho_a_pos : 0 < varrho_a) :
  (perimeter_of_right_triangle a varrho_a = 2 * varrho_a * a / (2 * varrho_a - a)) :=
by
  sorry

end NUMINAMATH_GPT_perimeter_of_excircle_opposite_leg_l381_38156


namespace NUMINAMATH_GPT_berengere_contribution_l381_38147

noncomputable def exchange_rate : ℝ := (1.5 : ℝ)
noncomputable def pastry_cost_euros : ℝ := (8 : ℝ)
noncomputable def lucas_money_cad : ℝ := (10 : ℝ)
noncomputable def lucas_money_euros : ℝ := lucas_money_cad / exchange_rate

theorem berengere_contribution :
  pastry_cost_euros - lucas_money_euros = (4 / 3 : ℝ) :=
by
  sorry

end NUMINAMATH_GPT_berengere_contribution_l381_38147


namespace NUMINAMATH_GPT_determine_a_l381_38146

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
if x < 1 then 2 * x + a else -x - 2 * a

theorem determine_a (a : ℝ) (h : a ≠ 0) (h_eq : f a (1 - a) = f a (1 + a)) : a = -3/4 :=
by
  sorry

end NUMINAMATH_GPT_determine_a_l381_38146


namespace NUMINAMATH_GPT_problem1_problem2_l381_38166

theorem problem1 (α : ℝ) (h : Real.tan α = -2) : 
  (Real.sin α + 5 * Real.cos α) / (-2 * Real.cos α + Real.sin α) = -3 / 4 :=
sorry

theorem problem2 (α : ℝ) (h : Real.tan α = -2) :
  Real.sin (α - 5 * Real.pi) * Real.sin (3 * Real.pi / 2 - α) = -2 / 5 :=
sorry

end NUMINAMATH_GPT_problem1_problem2_l381_38166


namespace NUMINAMATH_GPT_initial_apples_9_l381_38177

def initial_apple_count (picked : ℕ) (remaining : ℕ) : ℕ :=
  picked + remaining

theorem initial_apples_9 (picked : ℕ) (remaining : ℕ) :
  picked = 2 → remaining = 7 → initial_apple_count picked remaining = 9 := by
sorry

end NUMINAMATH_GPT_initial_apples_9_l381_38177


namespace NUMINAMATH_GPT_work_completion_days_l381_38104

theorem work_completion_days (a b c : ℝ) :
  (1/a) = 1/90 → (1/b) = 1/45 → (1/a + 1/b + 1/c) = 1/5 → c = 6 :=
by
  intros ha hb habc
  sorry

end NUMINAMATH_GPT_work_completion_days_l381_38104


namespace NUMINAMATH_GPT_simplified_expression_evaluation_l381_38193

def expression (x y : ℝ) : ℝ :=
  3 * (x^2 - 2 * x^2 * y) - 3 * x^2 + 2 * y - 2 * (x^2 * y + y)

def x := 1/2
def y := -3

theorem simplified_expression_evaluation : expression x y = 6 :=
  sorry

end NUMINAMATH_GPT_simplified_expression_evaluation_l381_38193


namespace NUMINAMATH_GPT_find_S_l381_38138

variable (R S T c : ℝ)
variable (h1 : R = c * (S^2 / T^2))
variable (c_value : c = 8)
variable (h2 : R = 2) (h3 : T = 2) (h4 : S = 1)
variable (R_new : R = 50) (T_new : T = 5)

theorem find_S : S = 12.5 := by
  sorry

end NUMINAMATH_GPT_find_S_l381_38138


namespace NUMINAMATH_GPT_product_modulo_6_l381_38179

theorem product_modulo_6 :
  (2017 * 2018 * 2019 * 2020) % 6 = 0 :=
by
  -- Conditions provided:
  have h1 : 2017 ≡ 5 [MOD 6] := by sorry
  have h2 : 2018 ≡ 0 [MOD 6] := by sorry
  have h3 : 2019 ≡ 1 [MOD 6] := by sorry
  have h4 : 2020 ≡ 2 [MOD 6] := by sorry
  -- Proof of the theorem:
  sorry

end NUMINAMATH_GPT_product_modulo_6_l381_38179


namespace NUMINAMATH_GPT_sample_size_calculation_l381_38194

theorem sample_size_calculation : 
  ∀ (high_school_students junior_high_school_students sampled_high_school_students n : ℕ), 
  high_school_students = 3500 →
  junior_high_school_students = 1500 →
  sampled_high_school_students = 70 →
  n = (3500 + 1500) * 70 / 3500 →
  n = 100 :=
by
  intros high_school_students junior_high_school_students sampled_high_school_students n
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_sample_size_calculation_l381_38194


namespace NUMINAMATH_GPT_tom_watches_movies_total_duration_l381_38127

-- Define the running times for each movie
def M := 120
def A := M - 30
def B := A + 10
def D := 2 * B - 20

-- Define the number of times Tom watches each movie
def watch_B := 2
def watch_A := 3
def watch_M := 1
def watch_D := 4

-- Calculate the total time spent watching each movie
def total_time_B := watch_B * B
def total_time_A := watch_A * A
def total_time_M := watch_M * M
def total_time_D := watch_D * D

-- Calculate the total duration Tom spends watching these movies in a week
def total_duration := total_time_B + total_time_A + total_time_M + total_time_D

-- The statement to prove
theorem tom_watches_movies_total_duration :
  total_duration = 1310 := 
by
  sorry

end NUMINAMATH_GPT_tom_watches_movies_total_duration_l381_38127


namespace NUMINAMATH_GPT_math_problem_l381_38196

theorem math_problem (x y : ℝ) (h : (x + 2 * y) ^ 3 + x ^ 3 + 2 * x + 2 * y = 0) : x + y - 1 = -1 := 
sorry

end NUMINAMATH_GPT_math_problem_l381_38196


namespace NUMINAMATH_GPT_at_least_one_not_less_than_two_l381_38141

theorem at_least_one_not_less_than_two
  (x y z : ℝ)
  (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  let a := x + 1 / y
  let b := y + 1 / z
  let c := z + 1 / x
  a >= 2 ∨ b >= 2 ∨ c >= 2 := 
sorry

end NUMINAMATH_GPT_at_least_one_not_less_than_two_l381_38141


namespace NUMINAMATH_GPT_find_a5_l381_38137

open Nat

def increasing_seq (a : Nat → Nat) : Prop :=
  ∀ m n : Nat, m < n → a m < a n

theorem find_a5
  (a : Nat → Nat)
  (h1 : ∀ n : Nat, a (a n) = 3 * n)
  (h2 : increasing_seq a)
  (h3 : ∀ n : Nat, a n > 0) :
  a 5 = 8 :=
by
  sorry

end NUMINAMATH_GPT_find_a5_l381_38137


namespace NUMINAMATH_GPT_probability_of_two_red_shoes_is_0_1332_l381_38101

def num_red_shoes : ℕ := 4
def num_green_shoes : ℕ := 6
def total_shoes : ℕ := num_red_shoes + num_green_shoes

def probability_first_red_shoe : ℚ := num_red_shoes / total_shoes
def remaining_red_shoes_after_first_draw : ℕ := num_red_shoes - 1
def remaining_shoes_after_first_draw : ℕ := total_shoes - 1
def probability_second_red_shoe : ℚ := remaining_red_shoes_after_first_draw / remaining_shoes_after_first_draw

def probability_two_red_shoes : ℚ := probability_first_red_shoe * probability_second_red_shoe

theorem probability_of_two_red_shoes_is_0_1332 : probability_two_red_shoes = 1332 / 10000 :=
by
  sorry

end NUMINAMATH_GPT_probability_of_two_red_shoes_is_0_1332_l381_38101


namespace NUMINAMATH_GPT_quadratic_inequality_solution_l381_38153

theorem quadratic_inequality_solution (a : ℝ) (h1 : ∀ x : ℝ, ax^2 + (a + 1) * x + 1 ≥ 0) : a = 1 := by
  sorry

end NUMINAMATH_GPT_quadratic_inequality_solution_l381_38153


namespace NUMINAMATH_GPT_part1_part2_l381_38188

-- Define m as a positive integer greater than or equal to 2
def m (k : ℕ) := k ≥ 2

-- Part 1: Existential statement for x_i's
theorem part1 (m : ℕ) (h : m ≥ 2) :
  ∃ (x : ℕ → ℤ),
    ∀ i, 1 ≤ i ∧ i ≤ m →
    x i * x (m + i) = x (i + 1) * x (m + i - 1) + 1 := by
  sorry

-- Part 2: Infinite sequence y_k
theorem part2 (x : ℕ → ℤ) (m : ℕ) (h : m ≥ 2) :
  (∀ i, 1 ≤ i ∧ i ≤ m → x i * x (m + i) = x (i + 1) * x (m + i - 1) + 1) →
  ∃ (y : ℤ → ℤ),
    (∀ k : ℤ, y k * y (m + k) = y (k + 1) * y (m + k - 1) + 1) ∧
    (∀ i, 1 ≤ i ∧ i ≤ 2 * m → y i = x i) := by
  sorry

end NUMINAMATH_GPT_part1_part2_l381_38188


namespace NUMINAMATH_GPT_final_statement_l381_38131

variable (f : ℝ → ℝ)

-- Conditions
axiom even_function : ∀ x, f (x) = f (-x)
axiom periodic_minus_one : ∀ x, f (x + 1) = -f (x)
axiom increasing_on_neg_one_to_zero : ∀ x y, -1 ≤ x ∧ x < y ∧ y ≤ 0 → f (x) < f (y)

-- Statement
theorem final_statement :
  (∀ x, f (x + 2) = f (x)) ∧
  (¬ (∀ x, 0 ≤ x ∧ x ≤ 1 → f (x) < f (x + 1))) ∧
  (∀ x y, 1 ≤ x ∧ x < y ∧ y ≤ 2 → f (x) < f (y)) ∧
  (f (2) = f (0)) :=
by
  sorry

end NUMINAMATH_GPT_final_statement_l381_38131


namespace NUMINAMATH_GPT_marie_saves_money_in_17_days_l381_38183

noncomputable def number_of_days_needed (cash_register_cost revenue tax_rate costs : ℝ) : ℕ := 
  let net_revenue := revenue / (1 + tax_rate) 
  let daily_profit := net_revenue - costs
  Nat.ceil (cash_register_cost / daily_profit)

def marie_problem_conditions : Prop := 
  let bread_daily_revenue := 40 * 2
  let bagels_daily_revenue := 20 * 1.5
  let cakes_daily_revenue := 6 * 12
  let muffins_daily_revenue := 10 * 3
  let daily_revenue := bread_daily_revenue + bagels_daily_revenue + cakes_daily_revenue + muffins_daily_revenue
  let fixed_daily_costs := 20 + 2 + 80 + 30
  fixed_daily_costs = 132 ∧ daily_revenue = 212 ∧ 8 / 100 = 0.08

theorem marie_saves_money_in_17_days : marie_problem_conditions → number_of_days_needed 1040 212 0.08 132 = 17 := 
by 
  intro h
  -- Proof goes here.
  sorry

end NUMINAMATH_GPT_marie_saves_money_in_17_days_l381_38183


namespace NUMINAMATH_GPT_ellipse_area_constant_l381_38180

def ellipse_equation (a b : ℝ) (x y : ℝ) : Prop := (x^2 / a^2) + (y^2 / b^2) = 1

def passes_through (x_a y_a x_b y_b : ℝ) (p : ℝ × ℝ) : Prop := 
  p.1 = x_a ∧ p.2 = y_a ∨ p.1 = x_b ∧ p.2 = y_b

def area_ABNM_constant (x y : ℝ) : Prop :=
  let x_0 := x;
  let y_0 := y;
  let y_M := -2 * y_0 / (x_0 - 2);
  let BM := 1 + 2 * y_0 / (x_0 - 2);
  let x_N := - x_0 / (y_0 - 1);
  let AN := 2 + x_0 / (y_0 - 1);
  (1 / 2) * AN * BM = 2

theorem ellipse_area_constant :
  ∀ (a b : ℝ), (a = 2 ∧ b = 1) → 
  (∀ (x y : ℝ), 
    ellipse_equation a b x y → 
    passes_through 2 0 0 1 (x, y) → 
    (x < 0 ∧ y < 0) →
    area_ABNM_constant x y) :=
by
  intros
  sorry

end NUMINAMATH_GPT_ellipse_area_constant_l381_38180


namespace NUMINAMATH_GPT_red_button_probability_l381_38105

/-
Mathematical definitions derived from the problem:
Initial setup:
- Jar A has 6 red buttons and 10 blue buttons.
- Same number of red and blue buttons are removed. Jar A retains 3/4 of original buttons.
- Calculate the final number of red buttons in Jar A and B, and determine the probability both selected buttons are red.
-/
theorem red_button_probability :
  let initial_red := 6
  let initial_blue := 10
  let total_buttons := initial_red + initial_blue
  let removal_fraction := 3 / 4
  let final_buttons := (3 / 4 : ℚ) * total_buttons
  let removed_buttons := total_buttons - final_buttons
  let removed_each_color := removed_buttons / 2
  let final_red_A := initial_red - removed_each_color
  let final_red_B := removed_each_color
  let prob_red_A := final_red_A / final_buttons
  let prob_red_B := final_red_B / removed_buttons
  prob_red_A * prob_red_B = 1 / 6 :=
by
  sorry

end NUMINAMATH_GPT_red_button_probability_l381_38105


namespace NUMINAMATH_GPT_geometric_progression_terms_l381_38191

theorem geometric_progression_terms 
  (q b4 S_n : ℚ) 
  (hq : q = 1/3) 
  (hb4 : b4 = 1/54) 
  (hS : S_n = 121/162) 
  (b1 : ℚ) 
  (hb1 : b1 = b4 * q^3)
  (Sn : ℚ) 
  (hSn : Sn = b1 * (1 - q^5) / (1 - q)) : 
  ∀ (n : ℕ), S_n = Sn → n = 5 :=
by
  intro n hn
  sorry

end NUMINAMATH_GPT_geometric_progression_terms_l381_38191


namespace NUMINAMATH_GPT_dance_relationship_l381_38189

theorem dance_relationship (b g : ℕ) 
  (h1 : ∀ i, 1 ≤ i ∧ i ≤ b → i = 1 → ∃ m, m = 7)
  (h2 : b = g - 6) 
  : 7 + (b - 1) = g := 
by
  sorry

end NUMINAMATH_GPT_dance_relationship_l381_38189


namespace NUMINAMATH_GPT_bracelet_pairing_impossible_l381_38122

/--
Elizabeth has 100 different bracelets, and each day she wears three of them to school. 
Prove that it is impossible for any pair of bracelets to appear together on her wrist exactly once.
-/
theorem bracelet_pairing_impossible : 
  (∃ (bracelet_set : Finset (Finset (Fin 100))), 
    (∀ (a b : Fin 100), a ≠ b → ∃ t ∈ bracelet_set, {a, b} ⊆ t) ∧ (∀ t ∈ bracelet_set, t.card = 3) ∧ (bracelet_set.card * 3 / 2 ≠ 99)) :=
sorry

end NUMINAMATH_GPT_bracelet_pairing_impossible_l381_38122


namespace NUMINAMATH_GPT_min_turns_for_route_l381_38107

-- Define the number of parallel and intersecting streets
def num_parallel_streets := 10
def num_intersecting_streets := 10

-- Define the grid as a product of these two numbers
def num_intersections := num_parallel_streets * num_intersecting_streets

-- Define the minimum number of turns necessary for a closed bus route passing through all intersections
def min_turns (grid_size : Nat) : Nat :=
  if grid_size = num_intersections then 20 else 0

-- The main theorem statement
theorem min_turns_for_route : min_turns num_intersections = 20 :=
  sorry

end NUMINAMATH_GPT_min_turns_for_route_l381_38107


namespace NUMINAMATH_GPT_solution_set_quadratic_inequality_l381_38172

theorem solution_set_quadratic_inequality (a b : ℝ) (h1 : a < 0)
    (h2 : ∀ x, ax^2 - bx - 1 > 0 ↔ -1/2 < x ∧ x < -1/3) :
    ∀ x, x^2 - b*x - a ≥ 0 ↔ x ≥ 3 ∨ x ≤ 2 := 
by
  sorry

end NUMINAMATH_GPT_solution_set_quadratic_inequality_l381_38172


namespace NUMINAMATH_GPT_treasure_chest_coins_l381_38169

theorem treasure_chest_coins (hours : ℕ) (coins_per_hour : ℕ) (total_coins : ℕ) :
  hours = 8 → coins_per_hour = 25 → total_coins = hours * coins_per_hour → total_coins = 200 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end NUMINAMATH_GPT_treasure_chest_coins_l381_38169


namespace NUMINAMATH_GPT_proof_P_l381_38110

-- Define the universal set U
def U : Set ℕ := {1, 2, 3, 4, 5}

-- Define the complement of P in U
def CU_P : Set ℕ := {4, 5}

-- Define the set P as the difference between U and CU_P
def P : Set ℕ := U \ CU_P

-- Prove that P = {1, 2, 3}
theorem proof_P :
  P = {1, 2, 3} :=
by
  sorry

end NUMINAMATH_GPT_proof_P_l381_38110


namespace NUMINAMATH_GPT_football_cost_l381_38102

theorem football_cost (cost_shorts cost_shoes money_have money_need : ℝ)
  (h_shorts : cost_shorts = 2.40)
  (h_shoes : cost_shoes = 11.85)
  (h_have : money_have = 10)
  (h_need : money_need = 8) :
  (money_have + money_need - (cost_shorts + cost_shoes) = 3.75) :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_football_cost_l381_38102


namespace NUMINAMATH_GPT_a5_value_S8_value_l381_38174

-- Definitions based on the conditions
def seq (n : ℕ) : ℕ :=
if n = 0 then 0
else if n = 1 then 1
else 2 * seq (n - 1)

noncomputable def S (n : ℕ) : ℕ :=
(1 - 2^n) / (1 - 2)

-- Proof statements
theorem a5_value : seq 5 = 16 := sorry

theorem S8_value : S 8 = 255 := sorry

end NUMINAMATH_GPT_a5_value_S8_value_l381_38174


namespace NUMINAMATH_GPT_bernold_wins_game_l381_38158

/-- A game is played on a 2007 x 2007 grid. Arnold's move consists of taking a 2 x 2 square,
 and Bernold's move consists of taking a 1 x 1 square. They alternate turns with Arnold starting.
  When Arnold can no longer move, Bernold takes all remaining squares. The goal is to prove that 
  Bernold can always win the game by ensuring that Arnold cannot make enough moves to win. --/
theorem bernold_wins_game (N : ℕ) (hN : N = 2007) :
  let admissible_points := (N - 1) * (N - 1)
  let arnold_moves_needed := (N / 2) * (N / 2 + 1) / 2 + 1
  admissible_points < arnold_moves_needed :=
by
  let admissible_points := 2006 * 2006
  let arnold_moves_needed := 1003 * 1004 / 2 + 1
  exact sorry

end NUMINAMATH_GPT_bernold_wins_game_l381_38158


namespace NUMINAMATH_GPT_count_integer_values_l381_38152

theorem count_integer_values (x : ℕ) (h : 3 < Real.sqrt x ∧ Real.sqrt x < 5) : 
  ∃! n, (n = 15) ∧ ∀ k, (3 < Real.sqrt k ∧ Real.sqrt k < 5) → (k ≥ 10 ∧ k ≤ 24) :=
by
  sorry

end NUMINAMATH_GPT_count_integer_values_l381_38152


namespace NUMINAMATH_GPT_min_area_quadrilateral_l381_38111

theorem min_area_quadrilateral
  (S_AOB S_COD : ℝ) (h₁ : S_AOB = 4) (h₂ : S_COD = 9) :
  ∃ S_BOC S_AOD, S_AOB + S_COD + S_BOC + S_AOD = 25 :=
by
  sorry

end NUMINAMATH_GPT_min_area_quadrilateral_l381_38111


namespace NUMINAMATH_GPT_find_a_l381_38185

theorem find_a (a b : ℤ) (h1 : a > b) (h2 : b > 0) (h3 : a + b + a * b = 152) : a = 50 := 
by 
  sorry

end NUMINAMATH_GPT_find_a_l381_38185


namespace NUMINAMATH_GPT_price_per_acre_is_1863_l381_38116

-- Define the conditions
def totalAcres : ℕ := 4
def numLots : ℕ := 9
def pricePerLot : ℤ := 828
def totalRevenue : ℤ := numLots * pricePerLot
def totalCost (P : ℤ) : ℤ := totalAcres * P

-- The proof problem: Prove that the price per acre P is 1863
theorem price_per_acre_is_1863 (P : ℤ) (h : totalCost P = totalRevenue) : P = 1863 :=
by
  sorry

end NUMINAMATH_GPT_price_per_acre_is_1863_l381_38116


namespace NUMINAMATH_GPT_new_avg_weight_l381_38154

theorem new_avg_weight (A B C D E : ℝ) (h1 : (A + B + C) / 3 = 84) (h2 : A = 78) 
(h3 : (B + C + D + E) / 4 = 79) (h4 : E = D + 6) : 
(A + B + C + D) / 4 = 80 :=
by
  sorry

end NUMINAMATH_GPT_new_avg_weight_l381_38154


namespace NUMINAMATH_GPT_maria_ann_age_problem_l381_38187

theorem maria_ann_age_problem
  (M A : ℕ)
  (h1 : M = 7)
  (h2 : M = A - 3) :
  ∃ Y : ℕ, 7 - Y = 1 / 2 * (10 - Y) := by
  sorry

end NUMINAMATH_GPT_maria_ann_age_problem_l381_38187


namespace NUMINAMATH_GPT_eval_sqrt4_8_pow12_l381_38163

-- Define the fourth root of 8
def fourthRootOfEight : ℝ := 8 ^ (1 / 4)

-- Define the original expression
def expr := (fourthRootOfEight) ^ 12

-- The theorem to prove
theorem eval_sqrt4_8_pow12: expr = 512 := by
  sorry

end NUMINAMATH_GPT_eval_sqrt4_8_pow12_l381_38163


namespace NUMINAMATH_GPT_S_n_expression_l381_38168

/-- 
  Given a sequence of positive terms {a_n} with sum of the first n terms represented as S_n,
  and given a_1 = 2, and given the relationship 
  S_{n+1}(S_{n+1} - 3^n) = S_n(S_n + 3^n), prove that S_{2023} = (3^2023 + 1) / 2.
-/
theorem S_n_expression
  (a : ℕ → ℕ) (S : ℕ → ℕ)
  (ha1 : a 1 = 2)
  (hr : ∀ n, S (n + 1) * (S (n + 1) - 3^n) = S n * (S n + 3^n)) :
  S 2023 = (3^2023 + 1) / 2 :=
sorry

end NUMINAMATH_GPT_S_n_expression_l381_38168


namespace NUMINAMATH_GPT_undefined_hydrogen_production_l381_38170

-- Define the chemical species involved as follows:
structure ChemQty where
  Ethane : ℕ
  Oxygen : ℕ
  CarbonDioxide : ℕ
  Water : ℕ

-- Balanced reaction equation
def balanced_reaction : ChemQty :=
  { Ethane := 2, Oxygen := 7, CarbonDioxide := 4, Water := 6 }

-- Given conditions as per problem scenario
def initial_state : ChemQty :=
  { Ethane := 1, Oxygen := 2, CarbonDioxide := 0, Water := 0 }

-- The statement reflecting the unclear result of the reaction under the given conditions.
theorem undefined_hydrogen_production :
  initial_state.Oxygen < balanced_reaction.Oxygen / balanced_reaction.Ethane * initial_state.Ethane →
  ∃ water_products : ℕ, water_products ≤ 6 * initial_state.Ethane / 2 := 
by
  -- Due to incomplete reaction
  sorry

end NUMINAMATH_GPT_undefined_hydrogen_production_l381_38170


namespace NUMINAMATH_GPT_third_side_length_is_six_l381_38100

-- Defining the lengths of the sides of the triangle
def side1 : ℕ := 2
def side2 : ℕ := 6

-- Defining that the third side is an even number between 4 and 8
def is_even (x : ℕ) : Prop := x % 2 = 0
def valid_range (x : ℕ) : Prop := 4 < x ∧ x < 8

-- Stating the theorem
theorem third_side_length_is_six (x : ℕ) (h1 : is_even x) (h2 : valid_range x) : x = 6 :=
by
  sorry

end NUMINAMATH_GPT_third_side_length_is_six_l381_38100


namespace NUMINAMATH_GPT_additional_time_proof_l381_38108

-- Given the charging rate of the battery and the additional time required to reach a percentage
noncomputable def charging_rate := 20 / 60
noncomputable def initial_time := 60
noncomputable def additional_time := 150

-- Define the total time required to reach a certain percentage
noncomputable def total_time := initial_time + additional_time

-- The proof statement to verify the additional time required beyond the initial 60 minutes
theorem additional_time_proof : total_time - initial_time = additional_time := sorry

end NUMINAMATH_GPT_additional_time_proof_l381_38108


namespace NUMINAMATH_GPT_solve_for_k_l381_38117

def sameLine (p1 p2 p3 : ℝ × ℝ) : Prop :=
  (p2.2 - p1.2) * (p3.1 - p1.1) = (p3.2 - p1.2) * (p2.1 - p1.1)

theorem solve_for_k :
  (sameLine (3, 10) (1, k) (-7, 2)) → k = 8.4 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_k_l381_38117


namespace NUMINAMATH_GPT_total_cakes_served_l381_38106

def weekday_cakes_lunch : Nat := 6 + 8 + 10
def weekday_cakes_dinner : Nat := 9 + 7 + 5 + 13
def weekday_cakes_total : Nat := weekday_cakes_lunch + weekday_cakes_dinner

def weekend_cakes_lunch : Nat := 2 * (6 + 8 + 10)
def weekend_cakes_dinner : Nat := 2 * (9 + 7 + 5 + 13)
def weekend_cakes_total : Nat := weekend_cakes_lunch + weekend_cakes_dinner

def total_weekday_cakes : Nat := 5 * weekday_cakes_total
def total_weekend_cakes : Nat := 2 * weekend_cakes_total

def total_week_cakes : Nat := total_weekday_cakes + total_weekend_cakes

theorem total_cakes_served : total_week_cakes = 522 := by
  sorry

end NUMINAMATH_GPT_total_cakes_served_l381_38106


namespace NUMINAMATH_GPT_circle_C2_equation_line_l_equation_l381_38134

-- Proof problem 1: Finding the equation of C2
theorem circle_C2_equation (C1_center_x C1_center_y : ℝ) (A_x A_y : ℝ) 
  (C2_center_x : ℝ) (C1_radius : ℝ) :
  C1_center_x = 6 ∧ C1_center_y = 7 ∧ C1_radius = 5 →
  A_x = 2 ∧ A_y = 4 →
  C2_center_x = 6 →
  (∀ y : ℝ, ((y - C1_center_y = C1_radius + (C1_radius + (y - C1_center_y)))) →
    (x - C2_center_x)^2 + (y - C2_center_y)^2 = 1) :=
sorry

-- Proof problem 2: Finding the equation of the line l
theorem line_l_equation (O_x O_y A_x A_y : ℝ) 
  (C1_center_x C1_center_y : ℝ) 
  (A_BC_dist : ℝ) :
  O_x = 0 ∧ O_y = 0 →
  A_x = 2 ∧ A_y = 4 →
  C1_center_x = 6 ∧ C1_center_y = 7 →
  A_BC_dist = 2 * (25^(1 / 2)) →
  ((2 : ℝ)*x - y + 5 = 0 ∨ (2 : ℝ)*x - y - 15 = 0) :=
sorry

end NUMINAMATH_GPT_circle_C2_equation_line_l_equation_l381_38134


namespace NUMINAMATH_GPT_max_ballpoint_pens_l381_38124

theorem max_ballpoint_pens (x y z : ℕ) (hx : x + y + z = 15)
  (hy : 10 * x + 40 * y + 60 * z = 500) (hz : x ≥ 1 ∧ y ≥ 1 ∧ z ≥ 1) :
  x ≤ 6 :=
sorry

end NUMINAMATH_GPT_max_ballpoint_pens_l381_38124


namespace NUMINAMATH_GPT_length_of_FD_l381_38143

theorem length_of_FD
  (ABCD_is_square : ∀ (A B C D : ℝ), A = 8 ∧ B = 8 ∧ C = 8 ∧ D = 8)
  (E_midpoint_AD : ∀ (A D E : ℝ), E = (A + D) / 2)
  (F_on_BD : ∀ (B D F E : ℝ), B = 8 ∧ F = 3 ∧ D = 8 ∧ E = 4):
  ∃ (FD : ℝ), FD = 3 := by
  sorry

end NUMINAMATH_GPT_length_of_FD_l381_38143


namespace NUMINAMATH_GPT_chessboard_disk_cover_l381_38114

noncomputable def chessboardCoveredSquares : ℕ :=
  let D : ℝ := 1 -- assuming D is a positive real number; actual value irrelevant as it gets cancelled in the comparison
  let grid_size : ℕ := 8
  let total_squares : ℕ := grid_size * grid_size
  let boundary_squares : ℕ := 28 -- pre-calculated in the insides steps
  let interior_squares : ℕ := total_squares - boundary_squares
  let non_covered_corners : ℕ := 4
  interior_squares - non_covered_corners

theorem chessboard_disk_cover : chessboardCoveredSquares = 32 := sorry

end NUMINAMATH_GPT_chessboard_disk_cover_l381_38114


namespace NUMINAMATH_GPT_find_y_l381_38181

-- Conditions as definitions in Lean 4
def angle_AXB : ℝ := 180
def angle_AX : ℝ := 70
def angle_BX : ℝ := 40
def angle_CY : ℝ := 130

-- The Lean statement for the proof problem
theorem find_y (angle_AXB_eq : angle_AXB = 180)
               (angle_AX_eq : angle_AX = 70)
               (angle_BX_eq : angle_BX = 40)
               (angle_CY_eq : angle_CY = 130) : 
               ∃ y : ℝ, y = 60 :=
by
  sorry -- The actual proof goes here.

end NUMINAMATH_GPT_find_y_l381_38181


namespace NUMINAMATH_GPT_total_volume_of_pyramids_l381_38136

theorem total_volume_of_pyramids :
  let base := 40
  let height_base := 20
  let height_pyramid := 30
  let area_base := (1 / 2) * base * height_base
  let volume_pyramid := (1 / 3) * area_base * height_pyramid
  3 * volume_pyramid = 12000 :=
by 
  sorry

end NUMINAMATH_GPT_total_volume_of_pyramids_l381_38136


namespace NUMINAMATH_GPT_solve_inequality_l381_38126

theorem solve_inequality :
  ∀ x : ℝ, (3 * x^2 - 4 * x - 7 < 0) ↔ (-1 < x ∧ x < 7 / 3) :=
by
  sorry

end NUMINAMATH_GPT_solve_inequality_l381_38126


namespace NUMINAMATH_GPT_slope_of_tangent_at_4_l381_38121

def f (x : ℝ) : ℝ := x^3 - 7 * x^2 + 1

theorem slope_of_tangent_at_4 : (deriv f 4) = -8 := by
  sorry

end NUMINAMATH_GPT_slope_of_tangent_at_4_l381_38121


namespace NUMINAMATH_GPT_find_years_l381_38120

variable (p m x : ℕ)

def two_years_ago := p - 2 = 2 * (m - 2)
def four_years_ago := p - 4 = 3 * (m - 4)
def ratio_in_x_years (x : ℕ) := (p + x) * 2 = (m + x) * 3

theorem find_years (h1 : two_years_ago p m) (h2 : four_years_ago p m) : ratio_in_x_years p m 2 :=
by
  sorry

end NUMINAMATH_GPT_find_years_l381_38120


namespace NUMINAMATH_GPT_find_remainder_l381_38139

-- Definition of N based on given conditions
def N : ℕ := 44 * 432

-- Definition of next multiple of 432
def next_multiple_of_432 : ℕ := N + 432

-- Statement to prove the remainder when next_multiple_of_432 is divided by 39 is 12
theorem find_remainder : next_multiple_of_432 % 39 = 12 := 
by sorry

end NUMINAMATH_GPT_find_remainder_l381_38139


namespace NUMINAMATH_GPT_fifteenth_battery_replacement_month_l381_38150

theorem fifteenth_battery_replacement_month :
  (98 % 12) + 1 = 4 :=
by
  sorry

end NUMINAMATH_GPT_fifteenth_battery_replacement_month_l381_38150


namespace NUMINAMATH_GPT_pentagonal_faces_count_l381_38198

theorem pentagonal_faces_count (x y : ℕ) (h : (5 * x + 6 * y) % 6 = 0) (h1 : ∃ v e f, v - e + f = 2 ∧ f = x + y ∧ e = (5 * x + 6 * y) / 2 ∧ v = (5 * x + 6 * y) / 3 ∧ (5 * x + 6 * y) / 3 * 3 = 5 * x + 6 * y) : 
  x = 12 :=
sorry

end NUMINAMATH_GPT_pentagonal_faces_count_l381_38198


namespace NUMINAMATH_GPT_count_odd_numbers_distinct_digits_l381_38161

theorem count_odd_numbers_distinct_digits : 
  ∃ n : ℕ, (∀ x : ℕ, 200 ≤ x ∧ x ≤ 999 ∧ x % 2 = 1 ∧ (∀ d ∈ [digit1, digit2, digit3], d ≤ 7) ∧ (digit1 ≠ digit2 ∧ digit2 ≠ digit3 ∧ digit1 ≠ digit3) → True) ∧
  n = 120 :=
sorry

end NUMINAMATH_GPT_count_odd_numbers_distinct_digits_l381_38161


namespace NUMINAMATH_GPT_fraction_zero_implies_value_l381_38162

theorem fraction_zero_implies_value (x : ℝ) (h : (|x| - 2) / (x + 2) = 0) (h_non_zero : x + 2 ≠ 0) : x = 2 :=
sorry

end NUMINAMATH_GPT_fraction_zero_implies_value_l381_38162


namespace NUMINAMATH_GPT_stickers_on_first_page_l381_38165

theorem stickers_on_first_page :
  ∀ (a b c d e : ℕ), 
    (b = 16) →
    (c = 24) →
    (d = 32) →
    (e = 40) →
    (b - a = 8) →
    (c - b = 8) →
    (d - c = 8) →
    (e - d = 8) →
    a = 8 :=
by
  intros a b c d e hb hc hd he h1 h2 h3 h4
  -- Proof would go here
  sorry

end NUMINAMATH_GPT_stickers_on_first_page_l381_38165


namespace NUMINAMATH_GPT_problem1_problem3_l381_38171

-- Define the function f(x)
def f (x : ℚ) : ℚ := (1 - x) / (1 + x)

-- Problem 1: Prove f(1/x) = -f(x), given x ≠ -1, x ≠ 0
theorem problem1 (x : ℚ) (hx1 : x ≠ -1) (hx2 : x ≠ 0) : f (1 / x) = -f x :=
by sorry

-- Problem 2: Comment on graph transformations for f(x)
-- This is a conceptual question about graph translation and is not directly translatable to a Lean theorem.

-- Problem 3: Find the minimum value of M - m such that m ≤ f(x) ≤ M for x ∈ ℤ
theorem problem3 : ∃ (M m : ℤ), (∀ x : ℤ, m ≤ f x ∧ f x ≤ M) ∧ (M - m = 4) :=
by sorry

end NUMINAMATH_GPT_problem1_problem3_l381_38171


namespace NUMINAMATH_GPT_sum_distances_eq_6sqrt2_l381_38159

-- Define the curves C1 and C2 in Cartesian coordinates
def curve_C1 := { p : ℝ × ℝ | p.1 + p.2 = 3 }
def curve_C2 := { p : ℝ × ℝ | p.2^2 = 2 * p.1 }

-- Defining the point P in ℝ²
def point_P : ℝ × ℝ := (1, 2)

-- Find the sum of distances |PA| + |PB|
theorem sum_distances_eq_6sqrt2 : 
  ∃ A B : ℝ × ℝ, A ∈ curve_C1 ∧ A ∈ curve_C2 ∧ 
                B ∈ curve_C1 ∧ B ∈ curve_C2 ∧ 
                (dist point_P A) + (dist point_P B) = 6 * Real.sqrt 2 := 
sorry

end NUMINAMATH_GPT_sum_distances_eq_6sqrt2_l381_38159


namespace NUMINAMATH_GPT_sum_of_remainders_mod_53_l381_38145

theorem sum_of_remainders_mod_53 (x y z : ℕ) (h1 : x % 53 = 31) (h2 : y % 53 = 17) (h3 : z % 53 = 8) : 
  (x + y + z) % 53 = 3 :=
by {
  sorry
}

end NUMINAMATH_GPT_sum_of_remainders_mod_53_l381_38145


namespace NUMINAMATH_GPT_min_dot_product_l381_38160

noncomputable def vec_a (m : ℝ) : ℝ × ℝ := (1 + 2^m, 1 - 2^m)
noncomputable def vec_b (m : ℝ) : ℝ × ℝ := (4^m - 3, 4^m + 5)

def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

theorem min_dot_product : ∃ m : ℝ, dot_product (vec_a m) (vec_b m) = -6 := by
  sorry

end NUMINAMATH_GPT_min_dot_product_l381_38160


namespace NUMINAMATH_GPT_extreme_values_range_of_a_l381_38133

noncomputable def f (x : ℝ) := x^2 * Real.exp x
noncomputable def y (x : ℝ) (a : ℝ) := f x - a * x

theorem extreme_values :
  ∃ x_max x_min,
    (x_max = -2 ∧ f x_max = 4 / Real.exp 2) ∧
    (x_min = 0 ∧ f x_min = 0) := sorry

theorem range_of_a (a : ℝ) :
  (∃ x₁ x₂, x₁ ≠ 0 ∧ x₂ ≠ 0 ∧ y x₁ a = 0 ∧ y x₂ a = 0) ↔
  -1 / Real.exp 1 < a ∧ a < 0 := sorry

end NUMINAMATH_GPT_extreme_values_range_of_a_l381_38133


namespace NUMINAMATH_GPT_meena_work_days_l381_38128

theorem meena_work_days (M : ℝ) : 1/5 + 1/M = 3/10 → M = 10 :=
by
  sorry

end NUMINAMATH_GPT_meena_work_days_l381_38128


namespace NUMINAMATH_GPT_atomic_weight_of_calcium_l381_38192

theorem atomic_weight_of_calcium (Ca I : ℝ) (h1 : 294 = Ca + 2 * I) (h2 : I = 126.9) : Ca = 40.2 :=
by
  sorry

end NUMINAMATH_GPT_atomic_weight_of_calcium_l381_38192


namespace NUMINAMATH_GPT_jose_cupcakes_l381_38164

theorem jose_cupcakes (lemons_needed : ℕ) (tablespoons_per_lemon : ℕ) (tablespoons_per_dozen : ℕ) (target_lemons : ℕ) : 
  (lemons_needed = 12) → 
  (tablespoons_per_lemon = 4) → 
  (target_lemons = 9) → 
  ((target_lemons * tablespoons_per_lemon / lemons_needed) = 3) :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_jose_cupcakes_l381_38164


namespace NUMINAMATH_GPT_vinnie_makes_more_l381_38190

-- Define the conditions
def paul_tips : ℕ := 14
def vinnie_tips : ℕ := 30

-- Define the theorem to prove
theorem vinnie_makes_more :
  vinnie_tips - paul_tips = 16 := by
  sorry

end NUMINAMATH_GPT_vinnie_makes_more_l381_38190


namespace NUMINAMATH_GPT_solve_eq_l381_38182

noncomputable def fx (x : ℝ) : ℝ :=
  ((x - 1) * (x - 2) * (x - 3) * (x - 4) * (x - 3) * (x - 2) * (x - 1) * (x - 5)) /
  ((x - 2) * (x - 4) * (x - 2) * (x - 5))

theorem solve_eq (x : ℝ) (h : x ≠ 2 ∧ x ≠ 4 ∧ x ≠ 5) :
  fx x = 1 ↔ x = 2 + Real.sqrt 2 ∨ x = 2 - Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_solve_eq_l381_38182


namespace NUMINAMATH_GPT_find_missing_number_l381_38157

theorem find_missing_number (x y : ℝ) 
  (h1 : (x + 50 + 78 + 104 + y) / 5 = 62)
  (h2 : (48 + 62 + 98 + 124 + x) / 5 = 76.4) : 
  y = 28 :=
by
  sorry

end NUMINAMATH_GPT_find_missing_number_l381_38157


namespace NUMINAMATH_GPT_product_of_b_l381_38178

noncomputable def b_product : ℤ :=
  let y1 := 3
  let y2 := 8
  let x1 := 2
  let l := y2 - y1 -- Side length of the square
  let b₁ := x1 - l -- One possible value of b
  let b₂ := x1 + l -- Another possible value of b
  b₁ * b₂ -- Product of possible values of b

theorem product_of_b :
  b_product = -21 := by
  sorry

end NUMINAMATH_GPT_product_of_b_l381_38178


namespace NUMINAMATH_GPT_find_a_l381_38148

-- Conditions: x = 5 is a solution to the equation 2x - a = -5
-- We need to prove that a = 15 under these conditions

theorem find_a (x a : ℤ) (h1 : x = 5) (h2 : 2 * x - a = -5) : a = 15 :=
by
  -- We are required to prove the statement, so we skip the proof part here
  sorry

end NUMINAMATH_GPT_find_a_l381_38148


namespace NUMINAMATH_GPT_number_of_frames_bought_l381_38132

/- 
   Define the problem conditions:
   1. Each photograph frame costs 3 dollars.
   2. Sally paid with a 20 dollar bill.
   3. Sally got 11 dollars in change.
-/ 

def frame_cost : Int := 3
def initial_payment : Int := 20
def change_received : Int := 11

/- 
   Prove that the number of photograph frames Sally bought is 3.
-/

theorem number_of_frames_bought : (initial_payment - change_received) / frame_cost = 3 := 
by
  sorry

end NUMINAMATH_GPT_number_of_frames_bought_l381_38132


namespace NUMINAMATH_GPT_interest_rate_supposed_to_be_invested_l381_38123

variable (P T : ℕ) (additional_interest interest_rate_15 interest_rate_R : ℚ)

def simple_interest (principal: ℚ) (time: ℚ) (rate: ℚ) : ℚ := (principal * time * rate) / 100

theorem interest_rate_supposed_to_be_invested :
  P = 15000 → T = 2 → additional_interest = 900 → interest_rate_15 = 15 →
  simple_interest P T interest_rate_15 = simple_interest P T interest_rate_R + additional_interest →
  interest_rate_R = 12 := by
  intros hP hT h_add h15 h_interest
  simp [simple_interest] at *
  sorry

end NUMINAMATH_GPT_interest_rate_supposed_to_be_invested_l381_38123


namespace NUMINAMATH_GPT_infinite_points_inside_circle_l381_38142

theorem infinite_points_inside_circle:
  ∀ c : ℝ, c = 3 → ∀ x y : ℚ, 0 < x ∧ 0 < y  ∧ x^2 + y^2 < 9 → ∃ a b : ℚ, 0 < a ∧ 0 < b ∧ a^2 + b^2 < 9 :=
sorry

end NUMINAMATH_GPT_infinite_points_inside_circle_l381_38142


namespace NUMINAMATH_GPT_kennedy_is_larger_l381_38151

-- Definitions based on given problem conditions
def KennedyHouse : ℕ := 10000
def BenedictHouse : ℕ := 2350
def FourTimesBenedictHouse : ℕ := 4 * BenedictHouse

-- Goal defined as a theorem to be proved
theorem kennedy_is_larger : KennedyHouse - FourTimesBenedictHouse = 600 :=
by 
  -- these are the conditions translated into Lean format
  let K := KennedyHouse
  let B := BenedictHouse
  let FourB := 4 * B
  let Goal := K - FourB
  -- prove the goal
  sorry

end NUMINAMATH_GPT_kennedy_is_larger_l381_38151


namespace NUMINAMATH_GPT_evaluate_expression_l381_38149

theorem evaluate_expression : 3000^3 - 2999 * 3000^2 - 2999^2 * 3000 + 2999^3 = 5999 := by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l381_38149


namespace NUMINAMATH_GPT_no_solution_in_nat_for_xx_plus_2yy_eq_zz_l381_38175

theorem no_solution_in_nat_for_xx_plus_2yy_eq_zz :
  ¬∃ (x y z : ℕ), x^x + 2 * y^y = z^z := by
  sorry

end NUMINAMATH_GPT_no_solution_in_nat_for_xx_plus_2yy_eq_zz_l381_38175


namespace NUMINAMATH_GPT_solve_x_from_equation_l381_38129

theorem solve_x_from_equation :
  ∀ (x : ℝ), x ≠ 0 ∧ (3 * x)^5 = (9 * x)^4 → x = 27 :=
by
  intro x
  rintro ⟨hx, h⟩
  sorry

end NUMINAMATH_GPT_solve_x_from_equation_l381_38129


namespace NUMINAMATH_GPT_total_oranges_picked_l381_38109

/-- Michaela needs 20 oranges to get full --/
def oranges_michaela_needs : ℕ := 20

/-- Cassandra needs twice as many oranges as Michaela to get full --/
def oranges_cassandra_needs : ℕ := 2 * oranges_michaela_needs

/-- After both have eaten until they are full, 30 oranges remain --/
def oranges_remaining : ℕ := 30

/-- The total number of oranges eaten by both Michaela and Cassandra --/
def oranges_eaten : ℕ := oranges_michaela_needs + oranges_cassandra_needs

/-- Prove that the total number of oranges picked from the farm is 90 --/
theorem total_oranges_picked : oranges_eaten + oranges_remaining = 90 := by
  sorry

end NUMINAMATH_GPT_total_oranges_picked_l381_38109


namespace NUMINAMATH_GPT_norris_money_left_l381_38176

-- Defining the conditions
def sept_savings : ℕ := 29
def oct_savings : ℕ := 25
def nov_savings : ℕ := 31
def dec_savings : ℕ := 35
def jan_savings : ℕ := 40

def initial_savings : ℕ := sept_savings + oct_savings + nov_savings + dec_savings + jan_savings
def interest_rate : ℝ := 0.02

def total_interest : ℝ :=
  sept_savings * interest_rate + 
  (sept_savings + oct_savings) * interest_rate + 
  (sept_savings + oct_savings + nov_savings) * interest_rate +
  (sept_savings + oct_savings + nov_savings + dec_savings) * interest_rate

def total_savings_with_interest : ℝ := initial_savings + total_interest
def hugo_owes_norris : ℕ := 20 - 10

-- The final statement to prove Norris' total amount of money
theorem norris_money_left : total_savings_with_interest + hugo_owes_norris = 175.76 := by
  sorry

end NUMINAMATH_GPT_norris_money_left_l381_38176


namespace NUMINAMATH_GPT_positive_integer_solution_l381_38103

/-- Given that x, y, and t are all equal to 1, and x + y + z + t = 10, we need to prove that z = 7. -/
theorem positive_integer_solution {x y z t : ℕ} (hx : x = 1) (hy : y = 1) (ht : t = 1) (h : x + y + z + t = 10) : z = 7 :=
by {
  -- We would provide the proof here, but for now, we use sorry
  sorry
}

end NUMINAMATH_GPT_positive_integer_solution_l381_38103


namespace NUMINAMATH_GPT_expression_evaluates_to_47_l381_38167

theorem expression_evaluates_to_47 : 
  (3 * 4 * 5) * ((1 / 3) + (1 / 4) + (1 / 5)) = 47 := 
by 
  sorry

end NUMINAMATH_GPT_expression_evaluates_to_47_l381_38167


namespace NUMINAMATH_GPT_fourth_competitor_jump_l381_38140

theorem fourth_competitor_jump :
  let first_jump := 22
  let second_jump := first_jump + 1
  let third_jump := second_jump - 2
  let fourth_jump := third_jump + 3
  fourth_jump = 24 := by
  sorry

end NUMINAMATH_GPT_fourth_competitor_jump_l381_38140


namespace NUMINAMATH_GPT_cannot_finish_third_l381_38173

variable (P Q R S T U : ℕ)
variable (beats : ℕ → ℕ → Prop)
variable (finishes_after : ℕ → ℕ → Prop)
variable (finishes_before : ℕ → ℕ → Prop)

noncomputable def race_conditions (P Q R S T U : ℕ) (beats finishes_after finishes_before : ℕ → ℕ → Prop) : Prop :=
  beats P Q ∧
  beats P R ∧
  beats Q S ∧
  finishes_after T P ∧
  finishes_before T Q ∧
  finishes_after U R ∧
  beats U T

theorem cannot_finish_third (P Q R S T U : ℕ) (beats finishes_after finishes_before : ℕ → ℕ → Prop) :
  race_conditions P Q R S T U beats finishes_after finishes_before →
  ¬ (finishes_before P T ∧ finishes_before T S ∧ finishes_after P R ∧ finishes_after P S) ∧ ¬ (finishes_before S T ∧ finishes_before T P) :=
sorry

end NUMINAMATH_GPT_cannot_finish_third_l381_38173


namespace NUMINAMATH_GPT_cylindrical_plane_l381_38130

open Set

-- Define a cylindrical coordinate point (r, θ, z)
structure CylindricalCoord where
  r : ℝ
  theta : ℝ
  z : ℝ

-- Condition 1: In cylindrical coordinates, z is the height
def height_in_cylindrical := λ coords : CylindricalCoord => coords.z 

-- Condition 2: z is constant c
variable (c : ℝ)

-- The theorem to be proven
theorem cylindrical_plane (c : ℝ) :
  {p : CylindricalCoord | p.z = c} = {q : CylindricalCoord | q.z = c} :=
by
  sorry

end NUMINAMATH_GPT_cylindrical_plane_l381_38130


namespace NUMINAMATH_GPT_square_of_cube_of_smallest_prime_l381_38186

def smallest_prime : Nat := 2

theorem square_of_cube_of_smallest_prime :
  ((smallest_prime ^ 3) ^ 2) = 64 := by
  sorry

end NUMINAMATH_GPT_square_of_cube_of_smallest_prime_l381_38186


namespace NUMINAMATH_GPT_lily_pads_cover_entire_lake_l381_38112

/-- 
If a patch of lily pads doubles in size every day and takes 57 days to cover half the lake,
then it will take 58 days to cover the entire lake.
-/
theorem lily_pads_cover_entire_lake (days_to_half : ℕ) (h : days_to_half = 57) : (days_to_half + 1 = 58) := by
  sorry

end NUMINAMATH_GPT_lily_pads_cover_entire_lake_l381_38112


namespace NUMINAMATH_GPT_room_dimension_l381_38118

theorem room_dimension {a : ℝ} (h1 : a > 0) 
  (h2 : 4 = 2^2) 
  (h3 : 14 = 2 * (7)) 
  (h4 : 2 * a = 14) :
  (a + 2 * a - 2 = 19) :=
sorry

end NUMINAMATH_GPT_room_dimension_l381_38118


namespace NUMINAMATH_GPT_gcd_a_b_is_one_l381_38199

-- Definitions
def a : ℤ := 100^2 + 221^2 + 320^2
def b : ℤ := 101^2 + 220^2 + 321^2

-- Theorem statement
theorem gcd_a_b_is_one : Int.gcd a b = 1 := by
  sorry

end NUMINAMATH_GPT_gcd_a_b_is_one_l381_38199


namespace NUMINAMATH_GPT_max_g_of_15_l381_38195

noncomputable def g (x : ℝ) : ℝ := x^3  -- Assume the polynomial g(x) = x^3 based on the maximum value found.

theorem max_g_of_15 (g : ℝ → ℝ) (h_coeff : ∀ x, 0 ≤ g x)
  (h3 : g 3 = 3) (h27 : g 27 = 1701) : g 15 = 3375 :=
by
  -- According to the problem's constraint and identified solution,
  -- here is the statement asserting that the maximum value of g(15) is 3375
  sorry

end NUMINAMATH_GPT_max_g_of_15_l381_38195


namespace NUMINAMATH_GPT_min_value_expr_l381_38113

theorem min_value_expr (a b : ℝ) (h_a_pos : 0 < a) (h_b_pos : 0 < b) (h_sum : a + b = 1) : 
  (b / (3 * a)) + (3 / b) ≥ 5 := 
sorry

end NUMINAMATH_GPT_min_value_expr_l381_38113


namespace NUMINAMATH_GPT_buratino_correct_l381_38119

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ (∀ m : ℕ, m ∣ n → m = 1 ∨ m = n)

def valid_nine_digit_number (n : ℕ) : Prop :=
  n >= 10^8 ∧ n < 10^9 ∧ (∀ i j : ℕ, i < 9 ∧ j < 9 ∧ i ≠ j → ((n / 10^i) % 10 ≠ (n / 10^j) % 10)) ∧
  (∀ i : ℕ, i < 9 → (n / 10^i) % 10 ≠ 7)

def can_form_prime (n : ℕ) : Prop :=
  ∃ m : ℕ, valid_nine_digit_number n ∧ (m < 1000 ∧ is_prime m ∧
   (∃ erase_indices : List ℕ, erase_indices.length = 6 ∧ 
    ∀ i : ℕ, i ∈ erase_indices → i < 9 ∧ 
    (n % 10^(9 - i)) / 10^(3 - i) = m))

theorem buratino_correct : 
  ∀ n : ℕ, valid_nine_digit_number n → ¬ can_form_prime n :=
by
  sorry

end NUMINAMATH_GPT_buratino_correct_l381_38119
