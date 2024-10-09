import Mathlib

namespace not_satisfiable_conditions_l2301_230133

theorem not_satisfiable_conditions (x y : ℕ) (h1 : 0 ≤ x ∧ x ≤ 9) (h2 : 0 ≤ y ∧ y ≤ 9) 
    (h3 : 10 * x + y % 80 = 0) (h4 : x + y = 2) : false := 
by 
  -- The proof is omitted because we are only asked for the statement.
  sorry

end not_satisfiable_conditions_l2301_230133


namespace middle_tree_less_half_tallest_tree_l2301_230194

theorem middle_tree_less_half_tallest_tree (T M S : ℝ)
  (hT : T = 108)
  (hS : S = 1/4 * M)
  (hS_12 : S = 12) :
  (1/2 * T) - M = 6 := 
sorry

end middle_tree_less_half_tallest_tree_l2301_230194


namespace integer_points_inequality_l2301_230118

theorem integer_points_inequality
  (a b : ℕ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : a * b + a - b - 5 = 0)
  (M := max ((a : ℤ)^2 + (b : ℤ)^2)) :
  (3 * x^2 + 2 * y^2 <= M) → ∃ (n : ℕ), n = 51 :=
by sorry

end integer_points_inequality_l2301_230118


namespace continuous_function_triples_l2301_230121

theorem continuous_function_triples (f g h : ℝ → ℝ) (h₁ : Continuous f) (h₂ : Continuous g) (h₃ : Continuous h)
  (h₄ : ∀ x y : ℝ, f (x + y) = g x + h y) :
  ∃ (c a b : ℝ), (∀ x : ℝ, f x = c * x + a + b) ∧ (∀ x : ℝ, g x = c * x + a) ∧ (∀ x : ℝ, h x = c * x + b) :=
sorry

end continuous_function_triples_l2301_230121


namespace x_squared_plus_y_squared_l2301_230122

theorem x_squared_plus_y_squared (x y : ℝ) (h1 : (x + y)^2 = 49) (h2 : x * y = 10) : x^2 + y^2 = 29 := 
by
  sorry

end x_squared_plus_y_squared_l2301_230122


namespace number_of_integers_divisible_by_18_or_21_but_not_both_l2301_230142

theorem number_of_integers_divisible_by_18_or_21_but_not_both :
  let num_less_2019_div_by_18 := 112
  let num_less_2019_div_by_21 := 96
  let num_less_2019_div_by_both := 16
  num_less_2019_div_by_18 + num_less_2019_div_by_21 - 2 * num_less_2019_div_by_both = 176 :=
by
  sorry

end number_of_integers_divisible_by_18_or_21_but_not_both_l2301_230142


namespace number_of_male_animals_l2301_230136

def total_original_animals : ℕ := 100 + 29 + 9
def animals_bought_by_brian : ℕ := total_original_animals / 2
def animals_after_brian : ℕ := total_original_animals - animals_bought_by_brian
def animals_after_jeremy : ℕ := animals_after_brian + 37

theorem number_of_male_animals : animals_after_jeremy / 2 = 53 :=
by
  sorry

end number_of_male_animals_l2301_230136


namespace equivalent_expression_l2301_230143

theorem equivalent_expression :
  (2 + 5) * (2^2 + 5^2) * (2^4 + 5^4) * (2^8 + 5^8) * (2^16 + 5^16) * (2^32 + 5^32) * (2^64 + 5^64) =
  5^128 - 2^128 := by
  sorry

end equivalent_expression_l2301_230143


namespace cost_price_of_cricket_bat_l2301_230186

variable (CP_A CP_B SP_C : ℝ)

-- Conditions
def condition1 : CP_B = 1.20 * CP_A := sorry
def condition2 : SP_C = 1.25 * CP_B := sorry
def condition3 : SP_C = 234 := sorry

-- The statement to prove
theorem cost_price_of_cricket_bat : CP_A = 156 := sorry

end cost_price_of_cricket_bat_l2301_230186


namespace second_competitor_distance_difference_l2301_230183

theorem second_competitor_distance_difference (jump1 jump2 jump3 jump4 : ℕ) : 
  jump1 = 22 → 
  jump4 = 24 → 
  jump3 = jump2 - 2 → 
  jump4 = jump3 + 3 → 
  jump2 - jump1 = 1 :=
by
  sorry

end second_competitor_distance_difference_l2301_230183


namespace crossing_time_proof_l2301_230188

/-
  Problem:
  Given:
  1. length_train: 600 (length of the train in meters)
  2. time_signal_post: 40 (time taken to cross the signal post in seconds)
  3. time_bridge_minutes: 20 (time taken to cross the bridge in minutes)

  Prove:
  t_cross_bridge: the time it takes to cross the bridge and the full length of the train is 1240 seconds
-/

def length_train : ℕ := 600
def time_signal_post : ℕ := 40
def time_bridge_minutes : ℕ := 20

-- Converting time to cross the bridge from minutes to seconds
def time_bridge_seconds : ℕ := time_bridge_minutes * 60

-- Finding the speed
def speed_train : ℕ := length_train / time_signal_post

-- Finding the length of the bridge
def length_bridge : ℕ := speed_train * time_bridge_seconds

-- Finding the total distance covered
def total_distance : ℕ := length_train + length_bridge

-- Given distance and speed, find the time to cross
def time_to_cross : ℕ := total_distance / speed_train

theorem crossing_time_proof : time_to_cross = 1240 := by
  sorry

end crossing_time_proof_l2301_230188


namespace arithmetic_sequence_sum_l2301_230187

def sum_of_arithmetic_sequence (a d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a + (n - 1) * d) / 2

theorem arithmetic_sequence_sum (a₁ d : ℕ)
  (h₁ : a₁ + (a₁ + 6 * d) + (a₁ + 13 * d) + (a₁ + 17 * d) = 120) :
  sum_of_arithmetic_sequence a₁ d 19 = 570 :=
by
  sorry

end arithmetic_sequence_sum_l2301_230187


namespace tetrahedron_edges_vertices_product_l2301_230126

theorem tetrahedron_edges_vertices_product :
  let vertices := 4
  let edges := 6
  edges * vertices = 24 :=
by
  let vertices := 4
  let edges := 6
  sorry

end tetrahedron_edges_vertices_product_l2301_230126


namespace phoneExpences_l2301_230115

structure PhonePlan where
  fixed_fee : ℝ
  free_minutes : ℕ
  excess_rate : ℝ -- rate per minute

def JanuaryUsage : ℕ := 15 * 60 + 17 -- 15 hours 17 minutes in minutes
def FebruaryUsage : ℕ := 9 * 60 + 55 -- 9 hours 55 minutes in minutes

def computeBill (plan : PhonePlan) (usage : ℕ) : ℝ :=
  let excess_minutes := (usage - plan.free_minutes).max 0
  plan.fixed_fee + (excess_minutes * plan.excess_rate)

theorem phoneExpences (plan : PhonePlan) :
  plan = { fixed_fee := 18.00, free_minutes := 600, excess_rate := 0.03 } →
  computeBill plan JanuaryUsage + computeBill plan FebruaryUsage = 45.51 := by
  sorry

end phoneExpences_l2301_230115


namespace smallest_natural_number_exists_l2301_230197

theorem smallest_natural_number_exists (n : ℕ) : (∃ n, ∃ a b c : ℕ, n = 15 ∧ 1998 = a * (5 ^ 4) + b * (3 ^ 4) + c * (1 ^ 4) ∧ a + b + c = 15) :=
sorry

end smallest_natural_number_exists_l2301_230197


namespace arithmetic_sequence_sum_l2301_230130

theorem arithmetic_sequence_sum :
  let a₁ := 3
  let d := 4
  let n := 30
  let aₙ := a₁ + (n - 1) * d
  let Sₙ := (n / 2) * (a₁ + aₙ)
  Sₙ = 1830 :=
by
  intros
  let a₁ := 3
  let d := 4
  let n := 30
  let aₙ := a₁ + (n - 1) * d
  let Sₙ := (n / 2) * (a₁ + aₙ)
  sorry

end arithmetic_sequence_sum_l2301_230130


namespace verify_tin_amount_l2301_230193

def ratio_to_fraction (part1 part2 : ℕ) : ℚ :=
  part2 / (part1 + part2 : ℕ)

def tin_amount_in_alloy (total_weight : ℚ) (ratio : ℚ) : ℚ :=
  total_weight * ratio

def alloy_mixture_tin_weight_is_correct
    (weight_A weight_B : ℚ)
    (ratio_A_lead ratio_A_tin : ℕ)
    (ratio_B_tin ratio_B_copper : ℕ) : Prop :=
  let tin_ratio_A := ratio_to_fraction ratio_A_lead ratio_A_tin
  let tin_ratio_B := ratio_to_fraction ratio_B_tin ratio_B_copper
  let tin_weight_A := tin_amount_in_alloy weight_A tin_ratio_A
  let tin_weight_B := tin_amount_in_alloy weight_B tin_ratio_B
  tin_weight_A + tin_weight_B = 146.57

theorem verify_tin_amount :
    alloy_mixture_tin_weight_is_correct 130 160 2 3 3 4 :=
by
  sorry

end verify_tin_amount_l2301_230193


namespace find_number_l2301_230129

theorem find_number (N : ℝ) 
    (h : 0.20 * ((0.05)^3 * 0.35 * (0.70 * N)) = 182.7) : 
    N = 20880000 :=
by
  -- proof to be filled
  sorry

end find_number_l2301_230129


namespace final_ranking_l2301_230177

-- Define data types for participants and their initial positions
inductive Participant
| X
| Y
| Z

open Participant

-- Define the initial conditions and number of position changes
def initial_positions : List Participant := [X, Y, Z]

def position_changes : Participant → Nat
| X => 5
| Y => 0  -- Not given explicitly but derived from the conditions.
| Z => 6

-- Final condition stating Y finishes before X
def Y_before_X : Prop := True

-- The theorem stating the final ranking
theorem final_ranking :
  Y_before_X →
  (initial_positions = [X, Y, Z]) →
  (position_changes X = 5) →
  (position_changes Z = 6) →
  (position_changes Y = 0) →
  [Y, X, Z] = [Y, X, Z] :=
by
  intros
  exact rfl

end final_ranking_l2301_230177


namespace paper_clips_in_two_cases_l2301_230135

-- Defining the problem statement in Lean 4
theorem paper_clips_in_two_cases (c b : ℕ) :
  2 * (c * b * 300) = 2 * c * b * 300 :=
by
  sorry

end paper_clips_in_two_cases_l2301_230135


namespace find_value_of_2a10_minus_a12_l2301_230101

-- Define the arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- Define the given conditions
def condition (a : ℕ → ℝ) : Prop :=
  is_arithmetic_sequence a ∧ (a 4 + a 6 + a 8 + a 10 + a 12 = 120)

-- State the theorem
theorem find_value_of_2a10_minus_a12 (a : ℕ → ℝ) (h : condition a) : 2 * a 10 - a 12 = 24 :=
by sorry

end find_value_of_2a10_minus_a12_l2301_230101


namespace arithmetic_sequence_common_difference_l2301_230169

-- Define the conditions
variables {S_3 a_1 a_3 : ℕ}
variables (d : ℕ)
axiom h1 : S_3 = 6
axiom h2 : a_3 = 4
axiom h3 : S_3 = 3 * (a_1 + a_3) / 2

-- Prove that the common difference d is 2
theorem arithmetic_sequence_common_difference :
  d = (a_3 - a_1) / 2 → d = 2 :=
by
  sorry -- Proof to be completed

end arithmetic_sequence_common_difference_l2301_230169


namespace triangle_side_difference_l2301_230104

theorem triangle_side_difference (x : ℕ) : 3 < x ∧ x < 17 → (∃ a b : ℕ, 3 < a ∧ a < 17 ∧ 3 < b ∧ b < 17 ∧ a - b = 12) :=
by
  sorry

end triangle_side_difference_l2301_230104


namespace sticks_left_is_correct_l2301_230132

-- Define the initial conditions
def initial_popsicle_sticks : ℕ := 170
def popsicle_sticks_per_group : ℕ := 15
def number_of_groups : ℕ := 10

-- Define the total number of popsicle sticks given out to the groups
def total_sticks_given : ℕ := popsicle_sticks_per_group * number_of_groups

-- Define the number of popsicle sticks left
def sticks_left : ℕ := initial_popsicle_sticks - total_sticks_given

-- Prove that the number of sticks left is 20
theorem sticks_left_is_correct : sticks_left = 20 :=
by
  sorry

end sticks_left_is_correct_l2301_230132


namespace gcd_lcm_240_l2301_230176

theorem gcd_lcm_240 (a b : ℕ) (h : Nat.gcd a b * Nat.lcm a b = 240) : 
  ∃ n, ∃ gcds : Finset ℕ, (gcds.card = n) ∧ (Nat.gcd a b ∈ gcds) :=
by
  sorry

end gcd_lcm_240_l2301_230176


namespace find_b_squared_l2301_230109

theorem find_b_squared
    (b : ℝ)
    (c_ellipse c_hyperbola a_ellipse a2_hyperbola b2_hyperbola : ℝ)
    (h1: a_ellipse^2 = 25)
    (h2 : b2_hyperbola = 9 / 4)
    (h3 : a2_hyperbola = 4)
    (h4 : c_hyperbola = Real.sqrt (a2_hyperbola + b2_hyperbola))
    (h5 : c_ellipse = c_hyperbola)
    (h6 : b^2 = a_ellipse^2 - c_ellipse^2)
: b^2 = 75 / 4 :=
sorry

end find_b_squared_l2301_230109


namespace maximize_profit_l2301_230147

noncomputable def g (x : ℝ) : ℝ :=
if h : (0 < x ∧ x ≤ 10) then 13.5 - (1 / 30) * x^2
else if h : (x > 10) then 168 / x - 2000 / (3 * x^2)
else 0 -- default case included for totality

noncomputable def y (x : ℝ) : ℝ :=
if h : (0 < x ∧ x ≤ 10) then 8.1 * x - (1 / 30) * x^3 - 20
else if h : (x > 10) then 148 - 2 * (1000 / (3 * x) + 2.7 * x)
else 0 -- default case included for totality

theorem maximize_profit (x : ℝ) : 0 < x → y 9 = 28.6 :=
by sorry

end maximize_profit_l2301_230147


namespace system_of_equations_has_solution_l2301_230178

theorem system_of_equations_has_solution (m : ℝ) : 
  ∃ (x y : ℝ), y = 3 * m * x + 4 ∧ y = (3 * m - 1) * x + 3 :=
by
  sorry

end system_of_equations_has_solution_l2301_230178


namespace douglas_votes_in_county_X_l2301_230149

theorem douglas_votes_in_county_X (V : ℝ) :
  (0.64 * (2 * V + V) - 0.4000000000000002 * V) / (2 * V) * 100 = 76 := by
sorry

end douglas_votes_in_county_X_l2301_230149


namespace sequence_length_l2301_230112

theorem sequence_length (a : ℕ) (h : a = 10800) (h1 : ∀ n, (n ≠ 0 → ∃ m, n = 2 * m ∧ m ≠ 0) ∧ 2 ∣ n)
  : ∃ k : ℕ, k = 5 := 
sorry

end sequence_length_l2301_230112


namespace slope_of_asymptotes_is_one_l2301_230124

-- Given definitions and axioms
variables (a b c : ℝ)
  (h1 : a > 0)
  (h2 : b > 0)
  (A1 : ℝ × ℝ := (-a, 0))
  (A2 : ℝ × ℝ := (a, 0))
  (B : ℝ × ℝ := (c, b^2 / a))
  (C : ℝ × ℝ := (c, -b^2 / a))
  (Perp : (b^2 / a) / (c + a) * -(b^2 / a) / (c - a) = -1)

-- Proof goal
theorem slope_of_asymptotes_is_one : a = b → (∀ m : ℝ, m = (b / a) ∨ m = -(b / a)) ↔ ∀ m : ℝ, m = 1 ∨ m = -1 :=
by
  sorry

end slope_of_asymptotes_is_one_l2301_230124


namespace incorrect_statements_l2301_230161

-- Define basic properties for lines and their equations.
def point_slope_form (y y1 x x1 k : ℝ) : Prop := (y - y1) = k * (x - x1)
def intercept_form (x y a b : ℝ) : Prop := x / a + y / b = 1
def distance_to_origin_on_y_axis (k b : ℝ) : ℝ := abs b
def slope_intercept_form (y m x c : ℝ) : Prop := y = m * x + c

-- The conditions specified in the problem.
variables (A B C D : Prop)
  (hA : A ↔ ∀ (y y1 x x1 k : ℝ), ¬point_slope_form y y1 x x1 k)
  (hB : B ↔ ∀ (x y a b : ℝ), intercept_form x y a b)
  (hC : C ↔ ∀ (k b : ℝ), distance_to_origin_on_y_axis k b = abs b)
  (hD : D ↔ ∀ (y m x c : ℝ), slope_intercept_form y m x c)

theorem incorrect_statements : ¬ B ∧ ¬ C ∧ ¬ D :=
by
  -- Intermediate steps would be to show each statement B, C, and D are false.
  sorry

end incorrect_statements_l2301_230161


namespace joan_trip_time_l2301_230180

-- Definitions of given conditions as parameters
def distance : ℕ := 480
def speed : ℕ := 60
def lunch_break_minutes : ℕ := 30
def bathroom_break_minutes : ℕ := 15
def number_of_bathroom_breaks : ℕ := 2

-- Conversion factors
def minutes_to_hours (m : ℕ) : ℚ := m / 60

-- Calculation of total time taken
def total_time : ℚ := 
  (distance / speed) + 
  (minutes_to_hours lunch_break_minutes) + 
  (number_of_bathroom_breaks * minutes_to_hours bathroom_break_minutes)

-- Statement of the problem
theorem joan_trip_time : total_time = 9 := 
  by 
    sorry

end joan_trip_time_l2301_230180


namespace messages_tuesday_l2301_230148

theorem messages_tuesday (T : ℕ) (h1 : 300 + T + (T + 300) + 2 * (T + 300) = 2000) : 
  T = 200 := by
  sorry

end messages_tuesday_l2301_230148


namespace number_of_neutrons_l2301_230127

def mass_number (element : Type) : ℕ := 61
def atomic_number (element : Type) : ℕ := 27

theorem number_of_neutrons (element : Type) : mass_number element - atomic_number element = 34 :=
by
  -- Place the proof here
  sorry

end number_of_neutrons_l2301_230127


namespace arithmetic_arrangement_result_l2301_230107

theorem arithmetic_arrangement_result :
    (1 / 8) * (1 / 9) * (1 / 28) = 1 / 2016 ∨ ((1 / 8) - (1 / 9)) * (1 / 28) = 1 / 2016 :=
by {
    sorry
}

end arithmetic_arrangement_result_l2301_230107


namespace alice_favorite_number_l2301_230190

-- Define the conditions for Alice's favorite number
def sum_of_digits (n : ℕ) : ℕ :=
  (n / 100) + ((n % 100) / 10) + (n % 10)

-- Define the problem statement
theorem alice_favorite_number :
  ∃ n : ℕ, 100 ≤ n ∧ n ≤ 200 ∧
           n % 13 = 0 ∧
           n % 3 ≠ 0 ∧
           sum_of_digits n % 4 = 0 ∧
           n = 130 :=
by
  sorry

end alice_favorite_number_l2301_230190


namespace digit_problem_l2301_230116

variable {x y : ℕ}

theorem digit_problem (h1 : 10 * x + y - (10 * y + x) = 36) (h2 : x * 2 = y) :
  (x + y) - (x - y) = 16 :=
by sorry

end digit_problem_l2301_230116


namespace eval_expr_l2301_230106
open Real

theorem eval_expr : 4 * ((3.6 * 0.48 * 2.50) / (0.12 * 0.09 * 0.5)) = 8000 := by
  sorry

end eval_expr_l2301_230106


namespace smallest_class_number_l2301_230195

theorem smallest_class_number (sum_classes : ℕ) (n_classes interval number_of_classes : ℕ) 
                              (h_sum : sum_classes = 87) (h_n_classes : n_classes = 30) 
                              (h_interval : interval = 5) (h_number_of_classes : number_of_classes = 6) : 
                              ∃ x, x + (interval + x) + (2 * interval + x) + (3 * interval + x) 
                              + (4 * interval + x) + (5 * interval + x) = sum_classes ∧ x = 2 :=
by {
  use 2,
  sorry
}

end smallest_class_number_l2301_230195


namespace proof_problem_l2301_230171

-- Define the proportional relationship
def proportional_relationship (y x : ℝ) (k : ℝ) : Prop :=
  y - 1 = k * (x + 2)

-- Define the function y = 2x + 5
def function_y_x (y x : ℝ) : Prop :=
  y = 2 * x + 5

-- The theorem for part (1) and (2)
theorem proof_problem (x y a : ℝ) (h1 : proportional_relationship 7 1 2) (h2 : proportional_relationship y x 2) :
  function_y_x y x ∧ function_y_x (-2) a → a = -7 / 2 :=
by
  sorry

end proof_problem_l2301_230171


namespace zongzi_profit_l2301_230138

def initial_cost : ℕ := 10
def initial_price : ℕ := 16
def initial_bags_sold : ℕ := 200
def additional_sales_per_yuan (x : ℕ) : ℕ := 80 * x
def profit_per_bag (x : ℕ) : ℕ := initial_price - x - initial_cost
def number_of_bags_sold (x : ℕ) : ℕ := initial_bags_sold + additional_sales_per_yuan x
def total_profit (profit_per_bag : ℕ) (number_of_bags_sold : ℕ) : ℕ := profit_per_bag * number_of_bags_sold

theorem zongzi_profit (x : ℕ) : 
  total_profit (profit_per_bag x) (number_of_bags_sold x) = 1440 := 
sorry

end zongzi_profit_l2301_230138


namespace value_of_x_l2301_230189

noncomputable def sum_integers_30_to_50 : ℕ :=
  (50 - 30 + 1) * (30 + 50) / 2

def even_count_30_to_50 : ℕ :=
  11

theorem value_of_x 
  (x := sum_integers_30_to_50)
  (y := even_count_30_to_50)
  (h : x + y = 851) : x = 840 :=
sorry

end value_of_x_l2301_230189


namespace rainbow_nerds_total_l2301_230156

theorem rainbow_nerds_total
  (purple yellow green red blue : ℕ)
  (h1 : purple = 10)
  (h2 : yellow = purple + 4)
  (h3 : green = yellow - 2)
  (h4 : red = 3 * green)
  (h5 : blue = red / 2) :
  (purple + yellow + green + red + blue = 90) :=
by
  sorry

end rainbow_nerds_total_l2301_230156


namespace curve_is_circle_l2301_230144

theorem curve_is_circle (r : ℝ) (θ : ℝ) (h : r = 3) : 
  ∃ (c : ℝ) (p : ℝ × ℝ), c = 3 ∧ p = (3 * Real.cos θ, 3 * Real.sin θ) := 
sorry

end curve_is_circle_l2301_230144


namespace matrix_cubic_l2301_230105

noncomputable def matrix_entries (x y z : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![x, y, z], ![y, z, x], ![z, x, y]]

theorem matrix_cubic (x y z : ℝ) (N : Matrix (Fin 3) (Fin 3) ℝ)
    (hN : N = matrix_entries x y z)
    (hn : N ^ 2 = 2 • (1 : Matrix (Fin 3) (Fin 3) ℝ))
    (hxyz : x * y * z = -2) :
  x^3 + y^3 + z^3 = -6 + 2 * Real.sqrt 2 ∨ x^3 + y^3 + z^3 = -6 - 2 * Real.sqrt 2 :=
by
  sorry

end matrix_cubic_l2301_230105


namespace coordinates_of_OC_l2301_230131

-- Define the given vectors
def OP : ℝ × ℝ := (2, 1)
def OA : ℝ × ℝ := (1, 7)
def OB : ℝ × ℝ := (5, 1)

-- Define the dot product for ℝ × ℝ
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- Define OC as a point on line OP, parameterized by t
def OC (t : ℝ) : ℝ × ℝ := (2 * t, t)

-- Define CA and CB
def CA (t : ℝ) : ℝ × ℝ := (OA.1 - (OC t).1, OA.2 - (OC t).2)
def CB (t : ℝ) : ℝ × ℝ := (OB.1 - (OC t).1, OB.2 - (OC t).2)

-- Prove that minimization of dot_product (CA t) (CB t) occurs at OC = (4, 2)
noncomputable def find_coordinates_at_min_dot_product : Prop :=
  ∃ (t : ℝ), t = 2 ∧ OC t = (4, 2)

-- The theorem statement
theorem coordinates_of_OC : find_coordinates_at_min_dot_product :=
sorry

end coordinates_of_OC_l2301_230131


namespace unique_exponential_solution_l2301_230154

theorem unique_exponential_solution (a x : ℝ) (ha_pos : 0 < a) (ha_ne_one : a ≠ 1) (hx_pos : 0 < x) :
  ∃! y : ℝ, a^y = x :=
by
  sorry

end unique_exponential_solution_l2301_230154


namespace PolyCoeffInequality_l2301_230141

open Real

variable (p q : ℝ[X])
variable (a : ℝ)
variable (n : ℕ)
variable (h k : ℝ)
variable (deg_p : p.degree = n)
variable (deg_q : q.degree = n - 1)
variable (hp : ∀ i, i ≤ n → |p.coeff i| ≤ h)
variable (hq : ∀ i, i < n → |q.coeff i| ≤ k)
variable (hpq : p = (X + C a) * q)

theorem PolyCoeffInequality : k ≤ h^n := by
  sorry

end PolyCoeffInequality_l2301_230141


namespace students_in_school_l2301_230196

variable (S : ℝ)
variable (W : ℝ)
variable (L : ℝ)

theorem students_in_school {S W L : ℝ} 
  (h1 : W = 0.55 * 0.25 * S)
  (h2 : L = 0.45 * 0.25 * S)
  (h3 : W = L + 50) : 
  S = 2000 := 
sorry

end students_in_school_l2301_230196


namespace max_value_of_expression_l2301_230150

theorem max_value_of_expression (x y z : ℝ) (h_nonneg : x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0) (h_sum : x + y + z = 3) :
  (xy / (x + y + 1) + xz / (x + z + 1) + yz / (y + z + 1)) ≤ 1 :=
sorry

end max_value_of_expression_l2301_230150


namespace georgie_enter_and_exit_ways_l2301_230157

-- Define the number of windows
def num_windows := 8

-- Define the magical barrier window
def barrier_window := 8

-- Define a function to count the number of ways Georgie can enter and exit the house
def count_ways_to_enter_and_exit : Nat :=
  let entry_choices := num_windows
  let exit_choices_from_normal := 6
  let exit_choices_from_barrier := 7
  let ways_from_normal := (entry_choices - 1) * exit_choices_from_normal  -- entering through windows 1 to 7
  let ways_from_barrier := 1 * exit_choices_from_barrier  -- entering through window 8
  ways_from_normal + ways_from_barrier

-- Prove the correct number of ways is 49
theorem georgie_enter_and_exit_ways : count_ways_to_enter_and_exit = 49 :=
by
  -- The calculation details are skipped with 'sorry'
  sorry

end georgie_enter_and_exit_ways_l2301_230157


namespace sum_of_ages_l2301_230165

theorem sum_of_ages (rachel_age leah_age : ℕ) 
  (h1 : rachel_age = leah_age + 4) 
  (h2 : rachel_age = 19) : rachel_age + leah_age = 34 :=
by
  -- Proof steps are omitted since we only need the statement
  sorry

end sum_of_ages_l2301_230165


namespace lateral_surface_area_of_square_pyramid_l2301_230182

-- Definitions based on the conditions in a)
def baseEdgeLength : ℝ := 4
def slantHeight : ℝ := 3

-- Lean 4 statement for the proof problem
theorem lateral_surface_area_of_square_pyramid :
  let height := Real.sqrt (slantHeight^2 - (baseEdgeLength / 2)^2)
  let lateralArea := (1 / 2) * 4 * (baseEdgeLength * height)
  lateralArea = 8 * Real.sqrt 5 :=
by
  sorry

end lateral_surface_area_of_square_pyramid_l2301_230182


namespace emily_coloring_books_l2301_230175

variable (initial_books : ℕ) (given_away : ℕ) (total_books : ℕ) (bought_books : ℕ)

theorem emily_coloring_books :
  initial_books = 7 →
  given_away = 2 →
  total_books = 19 →
  initial_books - given_away + bought_books = total_books →
  bought_books = 14 :=
by
  intros h1 h2 h3 h4
  sorry

end emily_coloring_books_l2301_230175


namespace common_number_in_sequence_l2301_230139

theorem common_number_in_sequence 
  (a b c d e f g h i j : ℕ) 
  (h1 : (a + b + c + d + e) / 5 = 4) 
  (h2 : (f + g + h + i + j) / 5 = 9)
  (h3 : (a + b + c + d + e + f + g + h + i + j) / 10 = 7)
  (h4 : e = f) :
  e = 5 :=
by
  sorry

end common_number_in_sequence_l2301_230139


namespace distance_from_neg2_eq4_l2301_230117

theorem distance_from_neg2_eq4 (x : ℤ) : |x + 2| = 4 ↔ x = 2 ∨ x = -6 :=
by
  sorry

end distance_from_neg2_eq4_l2301_230117


namespace find_MN_l2301_230191

theorem find_MN (d D : ℝ) (h_d_lt_D : d < D) :
  ∃ MN : ℝ, MN = (d * D) / (D - d) :=
by
  sorry

end find_MN_l2301_230191


namespace determine_m_l2301_230164

noncomputable def has_equal_real_roots (m : ℝ) : Prop :=
  m ≠ 0 ∧ (m^2 - 8 * m = 0)

theorem determine_m (m : ℝ) (h : has_equal_real_roots m) : m = 8 :=
  sorry

end determine_m_l2301_230164


namespace cone_heights_l2301_230185

theorem cone_heights (H x r1 r2 : ℝ) (H_frustum : H - x = 18)
  (A_lower : 400 * Real.pi = Real.pi * r1^2)
  (A_upper : 100 * Real.pi = Real.pi * r2^2)
  (ratio_radii : r2 / r1 = 1 / 2)
  (ratio_heights : x / H = 1 / 2) :
  x = 18 ∧ H = 36 :=
by
  sorry

end cone_heights_l2301_230185


namespace find_C_and_D_l2301_230137

variables (C D : ℝ)

theorem find_C_and_D (h : 4 * C + 2 * D + 5 = 30) : C = 5.25 ∧ D = 2 :=
by
  sorry

end find_C_and_D_l2301_230137


namespace Jenny_walked_distance_l2301_230134

-- Given: Jenny ran 0.6 mile.
-- Given: Jenny ran 0.2 miles farther than she walked.
-- Prove: Jenny walked 0.4 miles.

variable (r w : ℝ)

theorem Jenny_walked_distance
  (h1 : r = 0.6) 
  (h2 : r = w + 0.2) : 
  w = 0.4 :=
sorry

end Jenny_walked_distance_l2301_230134


namespace t_plus_reciprocal_l2301_230128

theorem t_plus_reciprocal (t : ℝ) (h : t^2 - 3 * t + 1 = 0) (ht : t ≠ 0) : t + 1/t = 3 :=
by sorry

end t_plus_reciprocal_l2301_230128


namespace min_ab_12_min_rec_expression_2_l2301_230153

noncomputable def condition1 (a b : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ (1 / a + 3 / b = 1)

theorem min_ab_12 {a b : ℝ} (h : condition1 a b) : 
  a * b = 12 :=
sorry

theorem min_rec_expression_2 {a b : ℝ} (h : condition1 a b) :
  (1 / (a - 1)) + (3 / (b - 3)) = 2 :=
sorry

end min_ab_12_min_rec_expression_2_l2301_230153


namespace total_apples_l2301_230181

theorem total_apples (baskets apples_per_basket : ℕ) (h1 : baskets = 37) (h2 : apples_per_basket = 17) : baskets * apples_per_basket = 629 := by
  sorry

end total_apples_l2301_230181


namespace problem_solution_l2301_230119

noncomputable def equilateral_triangle_area_to_perimeter_square_ratio (s : ℝ) (h : s = 10) : ℝ :=
  let altitude := s * Real.sqrt 3 / 2
  let area := 1 / 2 * s * altitude
  let perimeter := 3 * s
  let perimeter_squared := perimeter^2
  area / perimeter_squared

theorem problem_solution :
  equilateral_triangle_area_to_perimeter_square_ratio 10 rfl = Real.sqrt 3 / 36 :=
sorry

end problem_solution_l2301_230119


namespace square_can_be_divided_into_40_smaller_squares_l2301_230113

theorem square_can_be_divided_into_40_smaller_squares 
: ∃ (n : ℕ), n * n = 40 := 
sorry

end square_can_be_divided_into_40_smaller_squares_l2301_230113


namespace recreation_percentage_l2301_230163

variable (W : ℝ) 

def recreation_last_week (W : ℝ) : ℝ := 0.10 * W
def wages_this_week (W : ℝ) : ℝ := 0.90 * W
def recreation_this_week (W : ℝ) : ℝ := 0.40 * (wages_this_week W)

theorem recreation_percentage : 
  (recreation_this_week W) / (recreation_last_week W) * 100 = 360 :=
by sorry

end recreation_percentage_l2301_230163


namespace determine_a_from_equation_l2301_230158

theorem determine_a_from_equation (a : ℝ) (x : ℝ) (h1 : x = 1) (h2 : a * x + 3 * x = 2) : a = -1 := by
  sorry

end determine_a_from_equation_l2301_230158


namespace sum_of_ages_l2301_230184

-- Definitions based on given conditions
def J : ℕ := 19
def age_difference (B J : ℕ) : Prop := B - J = 32

-- Theorem stating the problem
theorem sum_of_ages (B : ℕ) (H : age_difference B J) : B + J = 70 :=
sorry

end sum_of_ages_l2301_230184


namespace determine_m_to_satisfy_conditions_l2301_230146

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (m^2 - m - 5) * x^(m - 1)

theorem determine_m_to_satisfy_conditions : 
  ∃ (m : ℝ), (m = 3) ∧ ∀ (x : ℝ), (0 < x → (m^2 - m - 5 > 0) ∧ (m - 1 > 0)) :=
by
  sorry

end determine_m_to_satisfy_conditions_l2301_230146


namespace net_effect_sale_value_net_effect_sale_value_percentage_increase_l2301_230103

def sale_value (P Q : ℝ) : ℝ := P * Q

theorem net_effect_sale_value (P Q : ℝ) :
  sale_value (0.8 * P) (1.8 * Q) = 1.44 * sale_value P Q :=
by
  sorry

theorem net_effect_sale_value_percentage_increase (P Q : ℝ) :
  (sale_value (0.8 * P) (1.8 * Q) - sale_value P Q) / sale_value P Q = 0.44 :=
by
  sorry

end net_effect_sale_value_net_effect_sale_value_percentage_increase_l2301_230103


namespace conjugate_system_solution_l2301_230111

theorem conjugate_system_solution (a b : ℝ) :
  (∀ x y : ℝ,
    (x + (2-a) * y = b + 1) ∧ ((2*a-7) * x + y = -5 - b)
    ↔ x + (2*a-7) * y = -5 - b ∧ (x + (2-a) * y = b + 1))
  ↔ a = 3 ∧ b = -3 := by
  sorry

end conjugate_system_solution_l2301_230111


namespace B_share_is_2400_l2301_230162

noncomputable def calculate_B_share (total_profit : ℝ) (x : ℝ) : ℝ :=
  let A_investment_months := 3 * x * 12
  let B_investment_months := x * 6
  let C_investment_months := (3/2) * x * 9
  let D_investment_months := (3/2) * x * 8
  let total_investment_months := A_investment_months + B_investment_months + C_investment_months + D_investment_months
  (B_investment_months / total_investment_months) * total_profit

theorem B_share_is_2400 :
  calculate_B_share 27000 1 = 2400 :=
sorry

end B_share_is_2400_l2301_230162


namespace maximum_value_of_f_l2301_230160

noncomputable def f (x : ℝ) : ℝ := ((x - 3) * (12 - x)) / x

theorem maximum_value_of_f :
  ∀ x : ℝ, 3 < x ∧ x < 12 → f x ≤ 3 :=
by
  sorry

end maximum_value_of_f_l2301_230160


namespace income_calculation_l2301_230199

theorem income_calculation (x : ℕ) (h1 : ∃ x : ℕ, income = 8*x ∧ expenditure = 7*x)
  (h2 : savings = 5000)
  (h3 : income = expenditure + savings) : income = 40000 :=
by {
  sorry
}

end income_calculation_l2301_230199


namespace rita_remaining_money_l2301_230125

-- Defining the conditions
def num_dresses := 5
def price_dress := 20
def num_pants := 3
def price_pant := 12
def num_jackets := 4
def price_jacket := 30
def transport_cost := 5
def initial_amount := 400

-- Calculating the total cost
def total_cost : ℕ :=
  (num_dresses * price_dress) + 
  (num_pants * price_pant) + 
  (num_jackets * price_jacket) + 
  transport_cost

-- Stating the proof problem 
theorem rita_remaining_money : initial_amount - total_cost = 139 := by
  sorry

end rita_remaining_money_l2301_230125


namespace find_angle_l2301_230179

theorem find_angle (θ : Real) (h1 : 0 ≤ θ ∧ θ ≤ π) (h2 : Real.sin θ = (Real.sqrt 2) / 2) :
  θ = Real.pi / 4 ∨ θ = 3 * Real.pi / 4 :=
by
  sorry

end find_angle_l2301_230179


namespace seq_value_at_2018_l2301_230152

noncomputable def f (x : ℝ) : ℝ := sorry
def seq (a : ℕ → ℝ) : Prop :=
  a 1 = f 0 ∧ ∀ (n : ℕ), n > 0 → f (a (n + 1)) = 1 / f (-2 - a n)

theorem seq_value_at_2018 (a : ℕ → ℝ) (h_seq : seq a) : a 2018 = 4035 := 
by sorry

end seq_value_at_2018_l2301_230152


namespace find_x_when_y_is_10_l2301_230170

-- Definitions of inverse proportionality and initial conditions
def inversely_proportional (x y : ℝ) (k : ℝ) : Prop := x * y = k

-- Given constants
def k : ℝ := 160
def x_initial : ℝ := 40
def y_initial : ℝ := 4

-- Theorem statement to prove the value of x when y = 10
theorem find_x_when_y_is_10 (h : inversely_proportional x_initial y_initial k) : 
  ∃ (x : ℝ), inversely_proportional x 10 k :=
sorry

end find_x_when_y_is_10_l2301_230170


namespace sum_smallest_numbers_eq_six_l2301_230173

theorem sum_smallest_numbers_eq_six :
  let smallest_natural := 0
  let smallest_prime := 2
  let smallest_composite := 4
  smallest_natural + smallest_prime + smallest_composite = 6 := by
  sorry

end sum_smallest_numbers_eq_six_l2301_230173


namespace find_number_l2301_230108

theorem find_number (x : ℝ) (h₁ : |x| + 1/x = 0) (h₂ : x ≠ 0) : x = -1 :=
sorry

end find_number_l2301_230108


namespace polynomial_divisible_l2301_230155

theorem polynomial_divisible (p q : ℤ) (h_p : p = -26) (h_q : q = 25) :
  ∀ x : ℤ, (x^4 + p*x^2 + q) % (x^2 - 6*x + 5) = 0 :=
by
  sorry

end polynomial_divisible_l2301_230155


namespace tan_A_in_triangle_ABC_l2301_230198

theorem tan_A_in_triangle_ABC (a b c : ℝ) (A B C : ℝ) (ha : 0 < A) (ha_90 : A < π / 2) 
(hb : b = 3 * a * Real.sin B) : Real.tan A = Real.sqrt 2 / 4 :=
sorry

end tan_A_in_triangle_ABC_l2301_230198


namespace chord_length_l2301_230123

-- Define radii of the circles
def r1 : ℝ := 5
def r2 : ℝ := 12
def r3 : ℝ := r1 + r2

-- Define the centers of the circles
variable (O1 O2 O3 : ℝ)

-- Define the points of tangency and foot of the perpendicular
def T1 : ℝ := O1 + r1
def T2 : ℝ := O2 + r2
def T : ℝ := O3 - r3

-- Given the conditions
theorem chord_length (m n p : ℤ) : 
  (∃ (C1 C2 C3 : ℝ) (tangent1 tangent2 : ℝ),
    C1 = r1 ∧ C2 = r2 ∧ C3 = r3 ∧
    -- Externally tangent: distance between centers of C1 and C2 is r1 + r2
    dist O1 O2 = r1 + r2 ∧
    -- Internally tangent: both C1 and C2 are tangent to C3
    dist O1 O3 = r3 - r1 ∧
    dist O2 O3 = r3 - r2 ∧
    -- The chord in C3 is a common external tangent to C1 and C2
    tangent1 = O3 + ((O1 * O2) - (O1 * O3)) / r1 ∧
    tangent2 = O3 + ((O2 * O1) - (O2 * O3)) / r2 ∧
    m = 10 ∧ n = 546 ∧ p = 7 ∧
    m + n + p = 563)
  := sorry

end chord_length_l2301_230123


namespace game_winning_strategy_l2301_230167

theorem game_winning_strategy (n : ℕ) (h : n ≥ 3) :
  (∃ k : ℕ, n = 3 * k + 2) → (∃ k : ℕ, n = 3 * k + 2 ∨ ∀ k : ℕ, n ≠ 3 * k + 2) :=
by
  sorry

end game_winning_strategy_l2301_230167


namespace product_of_four_consecutive_integers_is_product_of_two_consecutive_even_numbers_l2301_230100

theorem product_of_four_consecutive_integers_is_product_of_two_consecutive_even_numbers (n : ℤ) :
  let T := (n - 1) * n * (n + 1) * (n + 2)
  let M := n * (n + 1)
  T = (M - 2) * M :=
by
  -- proof here
  sorry

end product_of_four_consecutive_integers_is_product_of_two_consecutive_even_numbers_l2301_230100


namespace midpoint_coord_sum_l2301_230110

theorem midpoint_coord_sum (x1 y1 x2 y2 : ℝ) (hx1 : x1 = 10) (hy1 : y1 = -2) (hx2 : x2 = -4) (hy2 : y2 = 8)
: (x1 + x2) / 2 + (y1 + y2) / 2 = 6 :=
by
  rw [hx1, hx2, hy1, hy2]
  /-
  Have (10 + (-4)) / 2 + (-2 + 8) / 2 = (6 / 2) + (6 / 2)
  Prove that (6 / 2) + (6 / 2) = 6
  -/
  sorry

end midpoint_coord_sum_l2301_230110


namespace find_g_neg1_l2301_230151

-- Define the function f and its property of being an odd function
def odd_function (f : ℝ → ℝ) : Prop :=
∀ x : ℝ, f (-x) = -f x

-- Define the function g in terms of f
def g (f : ℝ → ℝ) (x : ℝ) : ℝ := f x + 2

-- Given conditions
variables {f : ℝ → ℝ}
variable  (h_odd : odd_function f)
variable  (h_g1 : g f 1 = 1)

-- The statement we want to prove
theorem find_g_neg1 : g f (-1) = 3 :=
sorry

end find_g_neg1_l2301_230151


namespace gary_current_weekly_eggs_l2301_230145

noncomputable def egg_laying_rates : List ℕ := [6, 5, 7, 4]

def total_eggs_per_day (rates : List ℕ) : ℕ :=
  rates.foldl (· + ·) 0

def total_eggs_per_week (eggs_per_day : ℕ) : ℕ :=
  eggs_per_day * 7

theorem gary_current_weekly_eggs : 
  total_eggs_per_week (total_eggs_per_day egg_laying_rates) = 154 :=
by
  sorry

end gary_current_weekly_eggs_l2301_230145


namespace find_special_N_l2301_230102

theorem find_special_N : ∃ N : ℕ, 
  (Nat.digits 10 N).length = 1112 ∧
  (Nat.digits 10 N).sum % 2000 = 0 ∧
  (Nat.digits 10 (N + 1)).sum % 2000 = 0 ∧
  (Nat.digits 10 N).contains 1 ∧
  (N = 9 * 10^1111 + 1 * 10^221 + 9 * (10^220 - 1) / 9 + 10^890 - 1) :=
sorry

end find_special_N_l2301_230102


namespace fraction_broke_off_l2301_230172

variable (p p_1 p_2 : ℝ)
variable (k : ℝ)

-- Conditions
def initial_mass : Prop := p_1 + p_2 = p
def value_relation : Prop := p_1^2 + p_2^2 = 0.68 * p^2

-- Goal
theorem fraction_broke_off (h1 : initial_mass p p_1 p_2)
                           (h2 : value_relation p p_1 p_2) :
  (p_2 / p) = 1 / 5 :=
sorry

end fraction_broke_off_l2301_230172


namespace cube_probability_l2301_230159

theorem cube_probability :
  let m := 1
  let n := 504
  ∀ (faces : Finset (Fin 6)) (nums : Finset (Fin 9)), 
    faces.card = 6 → nums.card = 9 →
    (∀ f ∈ faces, ∃ n ∈ nums, true) →
    m + n = 505 :=
by
  sorry

end cube_probability_l2301_230159


namespace count_even_digits_in_base_5_of_567_l2301_230174

def is_even (n : ℕ) : Bool := n % 2 = 0

def base_5_representation (n : ℕ) : List ℕ :=
  if h : n > 0 then
    let rec loop (n : ℕ) (acc : List ℕ) : List ℕ :=
      if n = 0 then acc else loop (n / 5) ((n % 5) :: acc)
    loop n []
  else [0]

def count_even_digits_in_base_5 (n : ℕ) : ℕ :=
  (base_5_representation n).filter is_even |>.length

theorem count_even_digits_in_base_5_of_567 :
  count_even_digits_in_base_5 567 = 2 := by
  sorry

end count_even_digits_in_base_5_of_567_l2301_230174


namespace unique_four_digit_numbers_l2301_230140

theorem unique_four_digit_numbers (digits : Finset ℕ) (odd_digits : Finset ℕ) :
  digits = {2, 3, 4, 5, 6} → 
  odd_digits = {3, 5} → 
  ∃ (n : ℕ), n = 14 :=
by
  sorry

end unique_four_digit_numbers_l2301_230140


namespace marys_mother_paid_correct_total_l2301_230168

def mary_and_friends_payment_per_person : ℕ := 1 -- $1 each
def number_of_people : ℕ := 3 -- Mary and two friends

def total_chicken_cost : ℕ := mary_and_friends_payment_per_person * number_of_people -- Total cost of the chicken

def beef_cost_per_pound : ℕ := 4 -- $4 per pound
def total_beef_pounds : ℕ := 3 -- 3 pounds of beef
def total_beef_cost : ℕ := beef_cost_per_pound * total_beef_pounds -- Total cost of the beef

def oil_cost : ℕ := 1 -- $1 for 1 liter of oil

def total_grocery_cost : ℕ := total_chicken_cost + total_beef_cost + oil_cost -- Total grocery cost

theorem marys_mother_paid_correct_total : total_grocery_cost = 16 := by
  -- Here you would normally provide the proof steps which we're skipping per instructions.
  sorry

end marys_mother_paid_correct_total_l2301_230168


namespace difference_in_nickels_is_correct_l2301_230166

variable (q : ℤ)

def charles_quarters : ℤ := 7 * q + 2
def richard_quarters : ℤ := 3 * q + 8

theorem difference_in_nickels_is_correct :
  5 * (charles_quarters - richard_quarters) = 20 * q - 30 :=
by
  sorry

end difference_in_nickels_is_correct_l2301_230166


namespace balloon_permutations_l2301_230114

theorem balloon_permutations : 
  let total_letters := 7
  let repetitions_L := 2
  let repetitions_O := 2
  (Nat.factorial total_letters) / ((Nat.factorial repetitions_L) * (Nat.factorial repetitions_O)) = 1260 := 
by
  sorry

end balloon_permutations_l2301_230114


namespace correct_exp_operation_l2301_230192

theorem correct_exp_operation (a b : ℝ) : (-a^3 * b) ^ 2 = a^6 * b^2 :=
  sorry

end correct_exp_operation_l2301_230192


namespace find_birth_rate_l2301_230120

noncomputable def average_birth_rate (B : ℕ) : Prop :=
  let death_rate := 3
  let net_increase_per_2_seconds := B - death_rate
  let seconds_per_hour := 3600
  let hours_per_day := 24
  let seconds_per_day := seconds_per_hour * hours_per_day
  let net_increase_times := seconds_per_day / 2
  let total_net_increase := net_increase_times * net_increase_per_2_seconds
  total_net_increase = 172800

theorem find_birth_rate (B : ℕ) (h : average_birth_rate B) : B = 7 :=
  sorry

end find_birth_rate_l2301_230120
