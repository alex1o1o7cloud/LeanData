import Mathlib

namespace cupcakes_initial_count_l1172_117226

theorem cupcakes_initial_count (x : ℕ) (h1 : x - 5 + 10 = 24) : x = 19 :=
by sorry

end cupcakes_initial_count_l1172_117226


namespace cost_per_pound_of_sausages_l1172_117210

/-- Jake buys 2-pound packages of sausages. He buys 3 packages. He pays $24. 
To find the cost per pound of sausages. --/
theorem cost_per_pound_of_sausages 
  (pkg_weight : ℕ) 
  (num_pkg : ℕ) 
  (total_cost : ℕ) 
  (cost_per_pound : ℕ) 
  (h_pkg_weight : pkg_weight = 2) 
  (h_num_pkg : num_pkg = 3) 
  (h_total_cost : total_cost = 24) 
  (h_total_weight : num_pkg * pkg_weight = 6) :
  total_cost / (num_pkg * pkg_weight) = cost_per_pound :=
sorry

end cost_per_pound_of_sausages_l1172_117210


namespace apps_left_on_phone_l1172_117251

-- Definitions for the given conditions
def initial_apps : ℕ := 15
def added_apps : ℕ := 71
def deleted_apps : ℕ := added_apps + 1

-- Proof statement
theorem apps_left_on_phone : initial_apps + added_apps - deleted_apps = 14 := by
  sorry

end apps_left_on_phone_l1172_117251


namespace shiela_used_seven_colors_l1172_117208

theorem shiela_used_seven_colors (total_blocks : ℕ) (blocks_per_color : ℕ)
  (h1 : total_blocks = 49) (h2 : blocks_per_color = 7) : (total_blocks / blocks_per_color) = 7 :=
by
  sorry

end shiela_used_seven_colors_l1172_117208


namespace solve_xy_l1172_117290

variable (x y : ℝ)

-- Given conditions
def condition1 : Prop := y = (2 / 3) * x
def condition2 : Prop := 0.4 * x = (1 / 3) * y + 110

-- Statement we want to prove
theorem solve_xy (h1 : condition1 x y) (h2 : condition2 x y) : x = 618.75 ∧ y = 412.5 :=
  by sorry

end solve_xy_l1172_117290


namespace equal_number_of_experienced_fishermen_and_children_l1172_117232

theorem equal_number_of_experienced_fishermen_and_children 
  (n : ℕ)
  (total_fish : ℕ)
  (children_catch : ℕ)
  (fishermen_catch : ℕ)
  (h1 : total_fish = n^2 + 5 * n + 22)
  (h2 : fishermen_catch - 10 = children_catch)
  (h3 : total_fish = n * children_catch + 11 * fishermen_catch)
  (h4 : fishermen_catch > children_catch)
  : n = 11 := 
sorry

end equal_number_of_experienced_fishermen_and_children_l1172_117232


namespace apartments_per_floor_l1172_117257

theorem apartments_per_floor (floors apartments_per: ℕ) (total_people : ℕ) (each_apartment_houses : ℕ)
    (h1 : floors = 25)
    (h2 : each_apartment_houses = 2)
    (h3 : total_people = 200)
    (h4 : floors * apartments_per * each_apartment_houses = total_people) :
    apartments_per = 4 := 
sorry

end apartments_per_floor_l1172_117257


namespace at_least_two_pairs_in_one_drawer_l1172_117278

theorem at_least_two_pairs_in_one_drawer (n : ℕ) (hn : n > 0) : 
  ∃ i j : ℕ, i ≠ j ∧ i < n ∧ j < n :=
by {
  sorry
}

end at_least_two_pairs_in_one_drawer_l1172_117278


namespace exists_sum_coprime_seventeen_not_sum_coprime_l1172_117209

/-- 
 For any integer \( n \) where \( n > 17 \), there exist integers \( a \) and \( b \) 
 such that \( n = a + b \), \( a > 1 \), \( b > 1 \), and \( \gcd(a, b) = 1 \).
 Additionally, the integer 17 does not have this property.
-/
theorem exists_sum_coprime (n : ℤ) (h : n > 17) : 
  ∃ (a b : ℤ), n = a + b ∧ a > 1 ∧ b > 1 ∧ Int.gcd a b = 1 :=
sorry

/-- 
 The integer 17 cannot be expressed as the sum of two integers greater than 1 
 that are coprime.
-/
theorem seventeen_not_sum_coprime : 
  ¬ ∃ (a b : ℤ), 17 = a + b ∧ a > 1 ∧ b > 1 ∧ Int.gcd a b = 1 :=
sorry

end exists_sum_coprime_seventeen_not_sum_coprime_l1172_117209


namespace circle_radius_l1172_117217

theorem circle_radius (M N : ℝ) (hM : M = Real.pi * r ^ 2) (hN : N = 2 * Real.pi * r) (h : M / N = 15) : r = 30 := by
  sorry

end circle_radius_l1172_117217


namespace find_t_l1172_117224

theorem find_t (t : ℝ) (a_n : ℕ → ℝ) (S_n : ℕ → ℝ) :
  (∀ n, a_n n = a_n 1 * (a_n 2 / a_n 1)^(n-1)) → -- Geometric sequence condition
  (∀ n, S_n n = 2017 * 2016^n - 2018 * t) →     -- Given sum formula
  t = 2017 / 2018 :=
by
  sorry

end find_t_l1172_117224


namespace ratio_of_a_to_b_l1172_117279

theorem ratio_of_a_to_b 
  (b c d a : ℚ)
  (h1 : b / c = 13 / 9)
  (h2 : c / d = 5 / 13)
  (h3 : a / d = 1 / 7.2) :
  a / b = 1 / 4 := 
by sorry

end ratio_of_a_to_b_l1172_117279


namespace roots_exist_for_all_K_l1172_117255

theorem roots_exist_for_all_K (K : ℝ) : ∃ x : ℝ, x = K^3 * (x - 1) * (x - 3) :=
by
  -- Applied conditions and approach
  sorry

end roots_exist_for_all_K_l1172_117255


namespace dilution_plate_count_lower_than_actual_l1172_117267

theorem dilution_plate_count_lower_than_actual
  (bacteria_count : ℕ)
  (colony_count : ℕ)
  (dilution_factor : ℕ)
  (plate_count : ℕ)
  (count_error_margin : ℕ)
  (method_estimation_error : ℕ)
  (H1 : method_estimation_error > 0)
  (H2 : colony_count = bacteria_count / dilution_factor - method_estimation_error)
  : colony_count < bacteria_count :=
by
  sorry

end dilution_plate_count_lower_than_actual_l1172_117267


namespace perpendicular_lines_unique_a_l1172_117237

open Real

theorem perpendicular_lines_unique_a (a : ℝ) 
  (l1 : ∀ x y : ℝ, (a - 1) * x + y - 1 = 0) 
  (l2 : ∀ x y : ℝ, 3 * x + a * y + 2 = 0) 
  (perpendicular : True) : 
  a = 3 / 4 := 
sorry

end perpendicular_lines_unique_a_l1172_117237


namespace quadratic_is_binomial_square_l1172_117246

theorem quadratic_is_binomial_square 
  (a : ℤ) : 
  (∃ b : ℤ, 9 * (x: ℤ)^2 - 24 * x + a = (3 * x + b)^2) ↔ a = 16 := 
by 
  sorry

end quadratic_is_binomial_square_l1172_117246


namespace time_for_a_alone_l1172_117289

theorem time_for_a_alone
  (b_work_time : ℕ := 20)
  (c_work_time : ℕ := 45)
  (together_work_time : ℕ := 72 / 10) :
  ∃ (a_work_time : ℕ), a_work_time = 15 :=
by
  sorry

end time_for_a_alone_l1172_117289


namespace percent_eighth_graders_combined_l1172_117288

theorem percent_eighth_graders_combined (p_students : ℕ) (m_students : ℕ)
  (p_grade8_percent : ℚ) (m_grade8_percent : ℚ) :
  p_students = 160 → m_students = 250 →
  p_grade8_percent = 18 / 100 → m_grade8_percent = 22 / 100 →
  100 * (p_grade8_percent * p_students + m_grade8_percent * m_students) / (p_students + m_students) = 20 := 
by
  intros h1 h2 h3 h4
  sorry

end percent_eighth_graders_combined_l1172_117288


namespace geometric_sequence_sum_2018_l1172_117269

noncomputable def geometric_sum (n : ℕ) (a1 q : ℝ) : ℝ :=
  if q = 1 then n * a1 else a1 * (1 - q^n) / (1 - q)

theorem geometric_sequence_sum_2018 :
  ∀ (a : ℕ → ℝ) (S : ℕ → ℝ),
    (∀ n, S n = geometric_sum n (a 1) 2) →
    a 1 = 1 / 2 →
    (a 1 * 2^2)^2 = 8 * a 1 * 2^3 - 16 →
    S 2018 = 2^2017 - 1 / 2 :=
by sorry

end geometric_sequence_sum_2018_l1172_117269


namespace sequence_formula_l1172_117296

theorem sequence_formula (S : ℕ → ℕ) (a : ℕ → ℕ) 
  (hS : ∀ n, S n = n^2 + n + 1) :
  (a 1 = 3) ∧ (∀ n, n ≥ 2 → a n = 2 * n) :=
by
  sorry

end sequence_formula_l1172_117296


namespace bus_weight_conversion_l1172_117264

noncomputable def round_to_nearest (x : ℚ) : ℤ := Int.floor (x + 0.5)

theorem bus_weight_conversion (kg_to_pound : ℚ) (bus_weight_kg : ℚ) 
  (h : kg_to_pound = 0.4536) (h_bus : bus_weight_kg = 350) : 
  round_to_nearest (bus_weight_kg / kg_to_pound) = 772 := by
  sorry

end bus_weight_conversion_l1172_117264


namespace triangle_angle_contradiction_l1172_117275

theorem triangle_angle_contradiction (α β γ : ℝ) (h : α + β + γ = 180) :
  (α > 60 ∧ β > 60 ∧ γ > 60) -> false :=
by
  sorry

end triangle_angle_contradiction_l1172_117275


namespace peter_total_food_l1172_117227

-- Given conditions
variables (chicken hamburgers hot_dogs sides total_food : ℕ)
  (h1 : chicken = 16)
  (h2 : hamburgers = chicken / 2)
  (h3 : hot_dogs = hamburgers + 2)
  (h4 : sides = hot_dogs / 2)
  (total_food : ℕ := chicken + hamburgers + hot_dogs + sides)

theorem peter_total_food (h1 : chicken = 16)
                          (h2 : hamburgers = chicken / 2)
                          (h3 : hot_dogs = hamburgers + 2)
                          (h4 : sides = hot_dogs / 2) :
                          total_food = 39 :=
by sorry

end peter_total_food_l1172_117227


namespace adjacent_complementary_is_complementary_l1172_117286

/-- Two angles are complementary if their sum is 90 degrees. -/
def complementary (α β : ℝ) : Prop :=
  α + β = 90

/-- Two angles are adjacent complementary if they are complementary and adjacent. -/
def adjacent_complementary (α β : ℝ) : Prop :=
  complementary α β ∧ α > 0 ∧ β > 0

/-- Prove that adjacent complementary angles are complementary. -/
theorem adjacent_complementary_is_complementary (α β : ℝ) : adjacent_complementary α β → complementary α β :=
by
  sorry

end adjacent_complementary_is_complementary_l1172_117286


namespace find_circle_equation_l1172_117291

noncomputable def circle_equation (D E F : ℝ) : Prop :=
  ∀ (x y : ℝ), (x, y) = (-1, 3) ∨ (x, y) = (0, 0) ∨ (x, y) = (0, 2) →
  x^2 + y^2 + D * x + E * y + F = 0

theorem find_circle_equation :
  ∃ D E F : ℝ, circle_equation D E F ∧
               (∀ x y, x^2 + y^2 + D * x + E * y + F = x^2 + y^2 + 4 * x - 2 * y) :=
sorry

end find_circle_equation_l1172_117291


namespace least_possible_b_l1172_117297

def is_prime (n : Nat) : Prop := Nat.Prime n

theorem least_possible_b (a b : Nat) (h1 : is_prime a) (h2 : is_prime b) (h3 : a + 2 * b = 180) (h4 : a > b) : b = 19 :=
by 
  sorry

end least_possible_b_l1172_117297


namespace strawberries_remaining_l1172_117249

theorem strawberries_remaining (initial : ℝ) (eaten_yesterday : ℝ) (eaten_today : ℝ) :
  initial = 1.6 ∧ eaten_yesterday = 0.8 ∧ eaten_today = 0.3 → initial - eaten_yesterday - eaten_today = 0.5 :=
by
  sorry

end strawberries_remaining_l1172_117249


namespace sufficient_and_necessary_condition_l1172_117216

noncomputable def f (x : ℝ) : ℝ := x^3 + Real.log (x + Real.sqrt (x^2 + 1)) / Real.log 2

theorem sufficient_and_necessary_condition {a b : ℝ} (h : a + b ≥ 0) : f a + f b ≥ 0 :=
sorry

end sufficient_and_necessary_condition_l1172_117216


namespace max_value_expression_l1172_117214

theorem max_value_expression (a b c : ℝ) 
  (ha : 300 ≤ a ∧ a ≤ 500) 
  (hb : 500 ≤ b ∧ b ≤ 1500) 
  (hc : c = 100) : 
  (∃ M, M = 8 ∧ ∀ x, x = (b + c) / (a - c) → x ≤ M) := 
sorry

end max_value_expression_l1172_117214


namespace james_hives_l1172_117204

-- Define all conditions
def hive_honey : ℕ := 20  -- Each hive produces 20 liters of honey
def jar_capacity : ℕ := 1/2  -- Each jar holds 0.5 liters
def jars_needed : ℕ := 100  -- James needs 100 jars for half the honey

-- Translate to Lean statement
theorem james_hives (hive_honey jar_capacity jars_needed : ℕ) :
  (hive_honey = 20) → 
  (jar_capacity = 1 / 2) →
  (jars_needed = 100) →
  (∀ hives : ℕ, (hives * hive_honey = 200) → hives = 5) :=
by
  intros Hhoney Hjar Hjars
  intros hives Hprod
  sorry

end james_hives_l1172_117204


namespace xy_sum_greater_two_l1172_117238

theorem xy_sum_greater_two (x y : ℝ) (h1 : x^3 > y^2) (h2 : y^3 > x^2) : x + y > 2 := 
by 
  sorry

end xy_sum_greater_two_l1172_117238


namespace polynomial_transformation_l1172_117276

-- Given the conditions of the polynomial function g and the provided transformation
-- We aim to prove the equivalence in a mathematically formal way using Lean

theorem polynomial_transformation (g : ℝ → ℝ) (h : ∀ x : ℝ, g (x^2 + 2) = x^4 + 5 * x^2 + 1) :
  ∀ x : ℝ, g (x^2 - 2) = x^4 - 3 * x^2 - 3 :=
by
  intro x
  sorry

end polynomial_transformation_l1172_117276


namespace speed_of_stream_l1172_117239

-- Definitions based on given conditions
def speed_still_water := 24 -- km/hr
def distance_downstream := 140 -- km
def time_downstream := 5 -- hours

-- Proof problem statement
theorem speed_of_stream (v : ℕ) :
  24 + v = distance_downstream / time_downstream → v = 4 :=
by
  sorry

end speed_of_stream_l1172_117239


namespace part1_part2_l1172_117228

def setA := {x : ℝ | -3 < x ∧ x < 4}
def setB (a : ℝ) := {x : ℝ | x^2 - 4 * a * x + 3 * a^2 = 0}

theorem part1 (a : ℝ) : (setA ∩ setB a = ∅) ↔ (a ≤ -3 ∨ a ≥ 4) :=
sorry

theorem part2 (a : ℝ) : (setA ∪ setB a = setA) ↔ (-1 < a ∧ a < 4/3) :=
sorry

end part1_part2_l1172_117228


namespace total_balls_estimate_l1172_117262

theorem total_balls_estimate 
  (total_balls : ℕ) 
  (red_balls : ℕ) 
  (frequency : ℚ)
  (h_red_balls : red_balls = 12)
  (h_frequency : frequency = 0.6) 
  (h_fraction : (red_balls : ℚ) / total_balls = frequency): 
  total_balls = 20 := 
by 
  sorry

end total_balls_estimate_l1172_117262


namespace coordinates_P_correct_l1172_117212

noncomputable def coordinates_of_P : ℝ × ℝ :=
  let x_distance_to_y_axis : ℝ := 5
  let y_distance_to_x_axis : ℝ := 4
  -- x-coordinate must be negative, y-coordinate must be positive
  let x_coord : ℝ := -x_distance_to_y_axis
  let y_coord : ℝ := y_distance_to_x_axis
  (x_coord, y_coord)

theorem coordinates_P_correct:
  coordinates_of_P = (-5, 4) :=
by
  sorry

end coordinates_P_correct_l1172_117212


namespace complex_division_l1172_117236

theorem complex_division (i : ℂ) (h : i ^ 2 = -1) : (3 - 4 * i) / i = -4 - 3 * i :=
by
  sorry

end complex_division_l1172_117236


namespace evaluate_expression_l1172_117242

theorem evaluate_expression :
  (3 ^ 1002 + 7 ^ 1003) ^ 2 - (3 ^ 1002 - 7 ^ 1003) ^ 2 = 56 * 10 ^ 1003 :=
by
  sorry

end evaluate_expression_l1172_117242


namespace only_integer_triplet_solution_l1172_117206

theorem only_integer_triplet_solution 
  (a b c : ℤ) : 5 * a^2 + 9 * b^2 = 13 * c^2 → a = 0 ∧ b = 0 ∧ c = 0 :=
by 
  intro h
  sorry

end only_integer_triplet_solution_l1172_117206


namespace find_number_l1172_117229

theorem find_number (n : ℤ) (h : 7 * n = 3 * n + 12) : n = 3 :=
sorry

end find_number_l1172_117229


namespace gcd_eq_gcd_of_eq_add_mul_l1172_117231

theorem gcd_eq_gcd_of_eq_add_mul (a b q r : Int) (h_q : b > 0) (h_r : 0 ≤ r) (h_ar : a = b * q + r) : Int.gcd a b = Int.gcd b r :=
by
  -- Conditions: constraints and assertion
  exact sorry

end gcd_eq_gcd_of_eq_add_mul_l1172_117231


namespace exponent_multiplication_l1172_117273

-- Define the variables and exponentiation property
variable (a : ℝ)

-- State the theorem
theorem exponent_multiplication : a^3 * a^2 = a^5 := by
  sorry

end exponent_multiplication_l1172_117273


namespace rhombus_area_l1172_117200

theorem rhombus_area (side diagonal₁ : ℝ) (h_side : side = 20) (h_diagonal₁ : diagonal₁ = 16) : 
  ∃ (diagonal₂ : ℝ), (2 * diagonal₂ * diagonal₂ + 8 * 8 = side * side) ∧ 
  (1 / 2 * diagonal₁ * diagonal₂ = 64 * Real.sqrt 21) := by
  sorry

end rhombus_area_l1172_117200


namespace Liza_initial_balance_l1172_117230

theorem Liza_initial_balance
  (W: Nat)   -- Liza's initial balance on Tuesday
  (rent: Nat := 450)
  (deposit: Nat := 1500)
  (electricity: Nat := 117)
  (internet: Nat := 100)
  (phone: Nat := 70)
  (final_balance: Nat := 1563) 
  (balance_eq: W - rent + deposit - electricity - internet - phone = final_balance) 
  : W = 800 :=
sorry

end Liza_initial_balance_l1172_117230


namespace exists_finite_group_with_normal_subgroup_GT_Aut_l1172_117292

noncomputable def finite_group_G (n : ℕ) : Type := sorry -- Specific construction details omitted
noncomputable def normal_subgroup_H (n : ℕ) : Type := sorry -- Specific construction details omitted

def Aut_G (n : ℕ) : ℕ := sorry -- Number of automorphisms of G
def Aut_H (n : ℕ) : ℕ := sorry -- Number of automorphisms of H

theorem exists_finite_group_with_normal_subgroup_GT_Aut (n : ℕ) :
  ∃ G H, finite_group_G n = G ∧ normal_subgroup_H n = H ∧ Aut_H n > Aut_G n := sorry

end exists_finite_group_with_normal_subgroup_GT_Aut_l1172_117292


namespace minimal_difference_big_small_sum_l1172_117294

theorem minimal_difference_big_small_sum :
  ∀ (N : ℕ), N > 0 → ∃ (S : ℕ), 
  S = (N * (N - 1) * (2 * N + 5)) / 6 :=
  by 
    sorry

end minimal_difference_big_small_sum_l1172_117294


namespace min_squared_sum_l1172_117285

theorem min_squared_sum (x y z : ℝ) (h1 : 0 ≤ x) (h2 : 0 ≤ y) (h3 : 0 ≤ z) (h4 : x + y + z = 3) : 
  x^2 + y^2 + z^2 ≥ 9 := 
sorry

end min_squared_sum_l1172_117285


namespace cos_15_degree_l1172_117282

theorem cos_15_degree : 
  let d15 := 15 * Real.pi / 180
  let d45 := 45 * Real.pi / 180
  let d30 := 30 * Real.pi / 180
  Real.cos d15 = (Real.sqrt 6 + Real.sqrt 2) / 4 := by
  sorry

end cos_15_degree_l1172_117282


namespace symmetric_line_equation_l1172_117256

theorem symmetric_line_equation {l : ℝ} (h1 : ∀ x y : ℝ, x + y - 1 = 0 → (-x) - y + 1 = l) : l = 0 :=
by
  sorry

end symmetric_line_equation_l1172_117256


namespace product_of_numbers_l1172_117299

theorem product_of_numbers (x y : ℝ) (h1 : x - y = 11) (h2 : x^2 + y^2 = 185) : x * y = 26 :=
sorry

end product_of_numbers_l1172_117299


namespace player_A_wins_after_4_rounds_geometric_sequence_differences_find_P_2_l1172_117220

-- Define probabilities of shots
def shooting_probability_A : ℝ := 0.5
def shooting_probability_B : ℝ := 0.6

-- Define initial points for questions
def initial_points_question_1 : ℝ := 0
def initial_points_question_2 : ℝ := 2

-- Given initial probabilities
def P_0 : ℝ := 0
def P_4 : ℝ := 1

-- Probability that player A wins after exactly 4 rounds
def probability_A_wins_after_4_rounds : ℝ :=
  let P_A := shooting_probability_A * (1 - shooting_probability_B)
  let P_B := shooting_probability_B * (1 - shooting_probability_A)
  let P_C := 1 - P_A - P_B
  P_A * P_C^2 * P_A + P_A * P_B * P_A^2

-- Define the probabilities P(i) for i=0..4
def P (i : ℕ) : ℝ := sorry -- Placeholder for the function

-- Define the proof problem
theorem player_A_wins_after_4_rounds : probability_A_wins_after_4_rounds = 0.0348 :=
sorry

theorem geometric_sequence_differences :
  ∀ i : ℕ, i < 4 → (P (i + 1) - P i) / (P (i + 2) - P (i + 1)) = 2/3 :=
sorry

theorem find_P_2 : P 2 = 4/13 :=
sorry

end player_A_wins_after_4_rounds_geometric_sequence_differences_find_P_2_l1172_117220


namespace initial_trucks_l1172_117205

def trucks_given_to_Jeff : ℕ := 13
def trucks_left_with_Sarah : ℕ := 38

theorem initial_trucks (initial_trucks_count : ℕ) :
  initial_trucks_count = trucks_given_to_Jeff + trucks_left_with_Sarah → initial_trucks_count = 51 :=
by
  sorry

end initial_trucks_l1172_117205


namespace range_of_a_l1172_117274

noncomputable def setM (a : ℝ) : Set ℝ := {x | x * (x - a - 1) < 0}
def setN : Set ℝ := {x | -1 ≤ x ∧ x ≤ 3}

theorem range_of_a (a : ℝ) : setM a ∪ setN = setN ↔ (-2 ≤ a ∧ a ≤ 2) :=
sorry

end range_of_a_l1172_117274


namespace badges_total_l1172_117235

theorem badges_total :
  let hermione_badges := 14
  let luna_badges := 17
  let celestia_badges := 52
  hermione_badges + luna_badges + celestia_badges = 83 :=
by
  let hermione_badges := 14
  let luna_badges := 17
  let celestia_badges := 52
  sorry

end badges_total_l1172_117235


namespace find_x_value_l1172_117261

theorem find_x_value (x : ℝ) :
  |x - 25| + |x - 21| = |3 * x - 75| → x = 71 / 3 :=
by
  sorry

end find_x_value_l1172_117261


namespace bus_speed_l1172_117281

theorem bus_speed (d t : ℕ) (h1 : d = 201) (h2 : t = 3) : d / t = 67 :=
by sorry

end bus_speed_l1172_117281


namespace Emir_needs_more_money_l1172_117240

theorem Emir_needs_more_money
  (cost_dictionary : ℝ)
  (cost_dinosaur_book : ℝ)
  (cost_cookbook : ℝ)
  (cost_science_kit : ℝ)
  (cost_colored_pencils : ℝ)
  (saved_amount : ℝ)
  (total_cost : ℝ := cost_dictionary + cost_dinosaur_book + cost_cookbook + cost_science_kit + cost_colored_pencils)
  (more_money_needed : ℝ := total_cost - saved_amount) :
  cost_dictionary = 5.50 →
  cost_dinosaur_book = 11.25 →
  cost_cookbook = 5.75 →
  cost_science_kit = 8.40 →
  cost_colored_pencils = 3.60 →
  saved_amount = 24.50 →
  more_money_needed = 10.00 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end Emir_needs_more_money_l1172_117240


namespace segment_length_BD_eq_CB_l1172_117283

theorem segment_length_BD_eq_CB {AC CB BD x : ℝ}
  (h1 : AC = 4 * CB)
  (h2 : BD = CB)
  (h3 : CB = x) :
  BD = CB := 
by
  -- Proof omitted
  sorry

end segment_length_BD_eq_CB_l1172_117283


namespace probability_three_hearts_l1172_117244

noncomputable def probability_of_three_hearts : ℚ :=
  (13/52) * (12/51) * (11/50)

theorem probability_three_hearts :
  probability_of_three_hearts = 26/2025 :=
by
  sorry

end probability_three_hearts_l1172_117244


namespace contestant_wins_probability_l1172_117215

section RadioProgramQuiz
  -- Defining the conditions
  def number_of_questions : ℕ := 4
  def number_of_choices_per_question : ℕ := 3
  def probability_of_correct_answer : ℚ := 1 / 3
  
  -- Defining the target probability
  def winning_probability : ℚ := 1 / 9

  -- The theorem
  theorem contestant_wins_probability :
    (let p := probability_of_correct_answer
     let p_correct_all := p^4
     let p_correct_three :=
       4 * (p^3 * (1 - p))
     p_correct_all + p_correct_three = winning_probability) :=
    sorry
end RadioProgramQuiz

end contestant_wins_probability_l1172_117215


namespace sum_abc_l1172_117295

theorem sum_abc (a b c: ℝ) 
  (h1 : ∃ x: ℝ, x^2 + a * x + 1 = 0 ∧ x^2 + b * x + c = 0)
  (h2 : ∃ x: ℝ, x^2 + x + a = 0 ∧ x^2 + c * x + b = 0) :
  a + b + c = -3 := 
sorry

end sum_abc_l1172_117295


namespace sum_possible_values_for_k_l1172_117253

theorem sum_possible_values_for_k :
  ∃ (k_vals : Finset ℕ), (∀ j k : ℕ, 0 < j → 0 < k → (1 / j + 1 / k = 1 / 4) → k ∈ k_vals) ∧ 
    k_vals.sum id = 51 :=
by 
  sorry

end sum_possible_values_for_k_l1172_117253


namespace greatest_three_digit_multiple_of_17_is_986_l1172_117225

theorem greatest_three_digit_multiple_of_17_is_986:
  ∃ n, 100 ≤ n ∧ n ≤ 999 ∧ 17 ∣ n ∧ (∀ m, 100 ≤ m ∧ m ≤ 999 ∧ 17 ∣ m → m ≤ 986) :=
sorry

end greatest_three_digit_multiple_of_17_is_986_l1172_117225


namespace sufficient_condition_transitive_l1172_117252

theorem sufficient_condition_transitive
  (C B A : Prop) (h1 : (C → B)) (h2 : (B → A)) : (C → A) :=
  sorry

end sufficient_condition_transitive_l1172_117252


namespace correct_meiosis_sequence_l1172_117218

-- Define the events as types
inductive Event : Type
| Replication : Event
| Synapsis : Event
| Separation : Event
| Division : Event

-- Define options as lists of events
def option_A := [Event.Replication, Event.Synapsis, Event.Separation, Event.Division]
def option_B := [Event.Synapsis, Event.Replication, Event.Separation, Event.Division]
def option_C := [Event.Synapsis, Event.Replication, Event.Division, Event.Separation]
def option_D := [Event.Replication, Event.Separation, Event.Synapsis, Event.Division]

-- Define the theorem to be proved
theorem correct_meiosis_sequence : option_A = [Event.Replication, Event.Synapsis, Event.Separation, Event.Division] :=
by
  sorry

end correct_meiosis_sequence_l1172_117218


namespace binary_operation_correct_l1172_117265

-- Define the binary numbers involved
def bin1 := 0b110110 -- 110110_2
def bin2 := 0b101010 -- 101010_2
def bin3 := 0b100    -- 100_2

-- Define the operation in binary
def result := 0b111001101100 -- 111001101100_2

-- Lean statement to verify the operation result
theorem binary_operation_correct : (bin1 * bin2) / bin3 = result :=
by sorry

end binary_operation_correct_l1172_117265


namespace minimum_value_of_expression_l1172_117207

theorem minimum_value_of_expression (x y z w : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hw : 0 < w) (h : 5 * w = 3 * x ∧ 5 * w = 4 * y ∧ 5 * w = 7 * z) : x - y + z - w = 11 :=
sorry

end minimum_value_of_expression_l1172_117207


namespace initial_cards_collected_l1172_117219

  -- Ralph collects some cards.
  variable (initial_cards: ℕ)

  -- Ralph's father gives Ralph 8 more cards.
  variable (added_cards: ℕ := 8)

  -- Now Ralph has 12 cards.
  variable (total_cards: ℕ := 12)

  -- Proof statement: Prove that the initial number of cards Ralph collected plus 8 equals 12.
  theorem initial_cards_collected: initial_cards + added_cards = total_cards := by
    sorry
  
end initial_cards_collected_l1172_117219


namespace find_K_values_l1172_117250

-- Define summation of first K natural numbers
def sum_natural_numbers (K : ℕ) : ℕ :=
  K * (K + 1) / 2

-- Define the main problem conditions
theorem find_K_values (K N : ℕ) (hN_positive : N > 0) (hN_bound : N < 150) (h_sum_eq : sum_natural_numbers K = 3 * N^2) :
  K = 2 ∨ K = 12 ∨ K = 61 :=
  sorry

end find_K_values_l1172_117250


namespace find_non_equivalent_fraction_l1172_117243

-- Define the fractions mentioned in the problem
def sevenSixths := 7 / 6
def optionA := 14 / 12
def optionB := 1 + 1 / 6
def optionC := 1 + 5 / 30
def optionD := 1 + 2 / 6
def optionE := 1 + 14 / 42

-- The main problem statement
theorem find_non_equivalent_fraction :
  optionD ≠ sevenSixths := by
  -- We put a 'sorry' here because we are not required to provide the proof
  sorry

end find_non_equivalent_fraction_l1172_117243


namespace expression_evaluation_l1172_117258

theorem expression_evaluation (m n : ℤ) (h1 : m = 2) (h2 : n = -1 ^ 2023) :
  (2 * m + n) * (2 * m - n) - (2 * m - n) ^ 2 + 2 * n * (m + n) = -12 := by
  sorry

end expression_evaluation_l1172_117258


namespace XiaoMingAgeWhenFathersAgeIsFiveTimes_l1172_117221

-- Define the conditions
def XiaoMingAgeCurrent : ℕ := 12
def FatherAgeCurrent : ℕ := 40

-- Prove the question given the conditions
theorem XiaoMingAgeWhenFathersAgeIsFiveTimes : 
  ∃ (x : ℕ), (FatherAgeCurrent - x) = 5 * x - XiaoMingAgeCurrent ∧ x = 7 := 
by
  use 7
  sorry

end XiaoMingAgeWhenFathersAgeIsFiveTimes_l1172_117221


namespace stamps_max_l1172_117234

theorem stamps_max (price_per_stamp : ℕ) (total_cents : ℕ) (h1 : price_per_stamp = 25) (h2 : total_cents = 5000) : 
  ∃ n : ℕ, (n * price_per_stamp ≤ total_cents) ∧ (∀ m : ℕ, (m > n) → (m * price_per_stamp > total_cents)) ∧ n = 200 := 
by
  sorry

end stamps_max_l1172_117234


namespace intersection_complement_l1172_117259

-- Definitions of the sets
def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 3}
def B : Set ℕ := {2, 5}

-- Complement of B with respect to U
def comp_B : Set ℕ := U \ B

-- Statement to be proven
theorem intersection_complement : A ∩ comp_B = {1, 3} :=
by 
  sorry

end intersection_complement_l1172_117259


namespace peter_speed_l1172_117245

theorem peter_speed (p : ℝ) (v_juan : ℝ) (d : ℝ) (t : ℝ) 
  (h1 : v_juan = p + 3) 
  (h2 : d = t * p + t * v_juan) 
  (h3 : t = 1.5) 
  (h4 : d = 19.5) : 
  p = 5 :=
by
  sorry

end peter_speed_l1172_117245


namespace thirty_six_hundredths_is_decimal_l1172_117298

namespace thirty_six_hundredths

-- Define the fraction representation of thirty-six hundredths
def fraction_thirty_six_hundredths : ℚ := 36 / 100

-- The problem is to prove that this fraction is equal to 0.36 in decimal form
theorem thirty_six_hundredths_is_decimal : fraction_thirty_six_hundredths = 0.36 := 
sorry

end thirty_six_hundredths

end thirty_six_hundredths_is_decimal_l1172_117298


namespace stratified_sampling_vision_test_l1172_117201

theorem stratified_sampling_vision_test 
  (n_total : ℕ) (n_HS : ℕ) (n_selected : ℕ)
  (h1 : n_total = 165)
  (h2 : n_HS = 66)
  (h3 : n_selected = 15) :
  (n_HS * n_selected / n_total) = 6 := 
by 
  sorry

end stratified_sampling_vision_test_l1172_117201


namespace gcd_of_consecutive_digit_sums_is_1111_l1172_117260

theorem gcd_of_consecutive_digit_sums_is_1111 (p q r s : ℕ) (hc : q = p+1 ∧ r = p+2 ∧ s = p+3) :
  ∃ d, d = 1111 ∧ ∀ n : ℕ, n = (1000 * p + 100 * q + 10 * r + s) + (1000 * s + 100 * r + 10 * q + p) → d ∣ n := by
  use 1111
  sorry

end gcd_of_consecutive_digit_sums_is_1111_l1172_117260


namespace union_set_solution_l1172_117248

theorem union_set_solution (M N : Set ℝ) 
    (hM : M = { x | 0 ≤ x ∧ x ≤ 3 }) 
    (hN : N = { x | x < 1 }) : 
    M ∪ N = { x | x ≤ 3 } := 
by 
    sorry

end union_set_solution_l1172_117248


namespace cost_of_goods_l1172_117241

-- Define variables and conditions
variables (x y z : ℝ)

-- Assume the given conditions
axiom h1 : x + 2 * y + 3 * z = 136
axiom h2 : 3 * x + 2 * y + z = 240

-- Statement to prove
theorem cost_of_goods : x + y + z = 94 := 
sorry

end cost_of_goods_l1172_117241


namespace find_numbers_l1172_117263

theorem find_numbers :
  ∃ a d : ℝ, 
    ((a - d) + a + (a + d) = 12) ∧ 
    ((a - d) * a * (a + d) = 48) ∧
    (a = 4) ∧ 
    (d = -2) ∧ 
    (a - d = 6) ∧ 
    (a + d = 2) :=
by
  sorry

end find_numbers_l1172_117263


namespace find_k_l1172_117270

theorem find_k (k : ℝ) (h1 : k > 0) (h2 : ∀ x ∈ Set.Icc (2 : ℝ) 4, y = k / x → y ≥ 5) : k = 20 :=
sorry

end find_k_l1172_117270


namespace initial_boys_count_l1172_117284

theorem initial_boys_count (b : ℕ) (h1 : b + 10 - 4 - 3 = 17) : b = 14 :=
by
  sorry

end initial_boys_count_l1172_117284


namespace coefficient_x8_expansion_l1172_117254

-- Define the problem statement in Lean
theorem coefficient_x8_expansion : 
  (Nat.choose 7 4) * (1 : ℤ)^3 * (-2 : ℤ)^4 = 560 :=
by
  sorry

end coefficient_x8_expansion_l1172_117254


namespace service_charge_percentage_is_correct_l1172_117277

-- Define the conditions
def orderAmount : ℝ := 450
def totalAmountPaid : ℝ := 468
def serviceCharge : ℝ := totalAmountPaid - orderAmount

-- Define the target percentage
def expectedServiceChargePercentage : ℝ := 4.0

-- Proof statement: the service charge percentage is expectedServiceChargePercentage
theorem service_charge_percentage_is_correct : 
  (serviceCharge / orderAmount) * 100 = expectedServiceChargePercentage :=
by
  sorry

end service_charge_percentage_is_correct_l1172_117277


namespace probability_of_X_conditioned_l1172_117293

variables (P_X P_Y P_XY : ℝ)

-- Conditions
def probability_of_Y : Prop := P_Y = 2/5
def probability_of_XY : Prop := P_XY = 0.05714285714285714
def independent_selection : Prop := P_XY = P_X * P_Y

-- Theorem statement
theorem probability_of_X_conditioned (P_X P_Y P_XY : ℝ) 
  (h1 : probability_of_Y P_Y) 
  (h2 : probability_of_XY P_XY) 
  (h3 : independent_selection P_X P_Y P_XY) :
  P_X = 0.14285714285714285 := 
sorry

end probability_of_X_conditioned_l1172_117293


namespace radius_of_circle_B_l1172_117272

theorem radius_of_circle_B (r_A r_D : ℝ) (r_B : ℝ) (hA : r_A = 2) (hD : r_D = 4) 
  (congruent_BC : r_B = r_B) (tangent_condition : true) -- placeholder conditions
  (center_pass : true) -- placeholder conditions
  : r_B = (4 / 3) * (Real.sqrt 7 - 1) :=
sorry

end radius_of_circle_B_l1172_117272


namespace three_bodies_with_triangle_front_view_l1172_117213

def has_triangle_front_view (b : Type) : Prop :=
  -- Placeholder definition for example purposes
  sorry

theorem three_bodies_with_triangle_front_view :
  ∃ (body1 body2 body3 : Type),
  has_triangle_front_view body1 ∧
  has_triangle_front_view body2 ∧
  has_triangle_front_view body3 :=
sorry

end three_bodies_with_triangle_front_view_l1172_117213


namespace time_to_empty_tank_by_leakage_l1172_117203

theorem time_to_empty_tank_by_leakage (R_t R_l : ℝ) (h1 : R_t = 1 / 12) (h2 : R_t - R_l = 1 / 18) :
  (1 / R_l) = 36 :=
by
  sorry

end time_to_empty_tank_by_leakage_l1172_117203


namespace integer_pair_condition_l1172_117247

theorem integer_pair_condition (m n : ℤ) (h : (m^2 + m * n + n^2 : ℚ) / (m + 2 * n) = 13 / 3) : m + 2 * n = 9 :=
sorry

end integer_pair_condition_l1172_117247


namespace sibling_discount_is_correct_l1172_117268

-- Defining the given conditions
def tuition_per_person : ℕ := 45
def total_cost_with_discount : ℕ := 75

-- Defining the calculation of sibling discount
def sibling_discount : ℕ :=
  let original_cost := 2 * tuition_per_person
  let discount := original_cost - total_cost_with_discount
  discount

-- Statement to prove
theorem sibling_discount_is_correct : sibling_discount = 15 :=
by
  unfold sibling_discount
  simp
  sorry

end sibling_discount_is_correct_l1172_117268


namespace election_winner_votes_difference_l1172_117280

theorem election_winner_votes_difference (V : ℝ) (h1 : 0.62 * V = 1054) : 0.24 * V = 408 :=
by
  sorry

end election_winner_votes_difference_l1172_117280


namespace simplify_fraction_l1172_117287

theorem simplify_fraction :
  (1 / (2 + Real.sqrt 3)) * (1 / (2 - Real.sqrt 3)) = 1 := 
by
  sorry

end simplify_fraction_l1172_117287


namespace selling_price_before_brokerage_l1172_117222

theorem selling_price_before_brokerage (cash_realized : ℝ) (brokerage_rate : ℝ) (final_cash : ℝ) : 
  final_cash = 104.25 → brokerage_rate = 1 / 400 → cash_realized = 104.51 :=
by
  intro h1 h2
  sorry

end selling_price_before_brokerage_l1172_117222


namespace common_chord_eqn_l1172_117271

theorem common_chord_eqn (x y : ℝ) :
  (x^2 + y^2 + 2 * x - 6 * y + 1 = 0) ∧
  (x^2 + y^2 - 4 * x + 2 * y - 11 = 0) →
  3 * x - 4 * y + 6 = 0 :=
by
  intro h
  sorry

end common_chord_eqn_l1172_117271


namespace space_between_trees_l1172_117266

theorem space_between_trees (n_trees : ℕ) (tree_space : ℕ) (total_length : ℕ) (spaces_between_trees : ℕ) (result_space : ℕ) 
  (h1 : n_trees = 8)
  (h2 : tree_space = 1)
  (h3 : total_length = 148)
  (h4 : spaces_between_trees = n_trees - 1)
  (h5 : result_space = (total_length - n_trees * tree_space) / spaces_between_trees) : 
  result_space = 20 := 
by sorry

end space_between_trees_l1172_117266


namespace nate_search_time_l1172_117202

def sectionG_rows : ℕ := 15
def sectionG_cars_per_row : ℕ := 10
def sectionH_rows : ℕ := 20
def sectionH_cars_per_row : ℕ := 9
def cars_per_minute : ℕ := 11

theorem nate_search_time :
  (sectionG_rows * sectionG_cars_per_row + sectionH_rows * sectionH_cars_per_row) / cars_per_minute = 30 :=
  by
    sorry

end nate_search_time_l1172_117202


namespace average_speed_for_trip_l1172_117211

theorem average_speed_for_trip 
  (Speed1 Speed2 : ℝ) 
  (AverageSpeed : ℝ) 
  (h1 : Speed1 = 110) 
  (h2 : Speed2 = 72) 
  (h3 : AverageSpeed = (2 * Speed1 * Speed2) / (Speed1 + Speed2)) :
  AverageSpeed = 87 := 
by
  -- solution steps would go here
  sorry

end average_speed_for_trip_l1172_117211


namespace complex_seventh_root_identity_l1172_117223

open Complex

theorem complex_seventh_root_identity (r : ℂ) (h1 : r^7 = 1) (h2 : r ≠ 1) :
  (r - 1) * (r^2 - 1) * (r^3 - 1) * (r^4 - 1) * (r^5 - 1) * (r^6 - 1) = 11 :=
by
  sorry

end complex_seventh_root_identity_l1172_117223


namespace new_unemployment_rate_is_66_percent_l1172_117233

theorem new_unemployment_rate_is_66_percent
  (initial_unemployment_rate : ℝ)
  (initial_employment_rate : ℝ)
  (u_increases_by_10_percent : initial_unemployment_rate * 1.1 = new_unemployment_rate)
  (e_decreases_by_15_percent : initial_employment_rate * 0.85 = new_employment_rate)
  (sum_is_100_percent : initial_unemployment_rate + initial_employment_rate = 100) :
  new_unemployment_rate = 66 :=
by
  sorry

end new_unemployment_rate_is_66_percent_l1172_117233
