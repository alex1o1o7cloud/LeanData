import Mathlib

namespace josanna_minimum_test_score_l64_6409

theorem josanna_minimum_test_score 
  (scores : List ℕ) (target_increase : ℕ) (new_score : ℕ)
  (h_scores : scores = [92, 78, 84, 76, 88]) 
  (h_target_increase : target_increase = 5):
  (List.sum scores + new_score) / (List.length scores + 1) ≥ (List.sum scores / List.length scores + target_increase) →
  new_score = 114 :=
by
  sorry

end josanna_minimum_test_score_l64_6409


namespace quadratic_roots_eqn_l64_6417

theorem quadratic_roots_eqn (b c : ℝ) (x1 x2 : ℝ) (h1 : x1 = -2) (h2 : x2 = 3) (h3 : b = -(x1 + x2)) (h4 : c = x1 * x2) : 
    (x^2 + b * x + c = 0) ↔ (x^2 - x - 6 = 0) :=
by
  sorry

end quadratic_roots_eqn_l64_6417


namespace geometric_sequence_product_l64_6473

theorem geometric_sequence_product
  (a : ℕ → ℝ)
  (h_geo : ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r)
  (h_pos : ∀ n : ℕ, 0 < a n)
  (h_roots : (∃ a₁ a₁₉ : ℝ, (a₁ + a₁₉ = 10) ∧ (a₁ * a₁₉ = 16) ∧ a 1 = a₁ ∧ a 19 = a₁₉)) :
  a 8 * a 12 = 16 := 
sorry

end geometric_sequence_product_l64_6473


namespace matches_for_ladder_l64_6427

theorem matches_for_ladder (n : ℕ) (h : n = 25) : 
  (6 + 6 * (n - 1) = 150) :=
by
  sorry

end matches_for_ladder_l64_6427


namespace find_D_l64_6432

theorem find_D (A B C D : ℕ) (h₁ : A + A = 6) (h₂ : B - A = 4) (h₃ : C + B = 9) (h₄ : D - C = 7) : D = 9 :=
sorry

end find_D_l64_6432


namespace geometric_seq_an_minus_2_l64_6412

-- Definitions of conditions based on given problem
def seq_a : ℕ → ℝ := sorry -- The sequence {a_n}
def sum_s : ℕ → ℝ := sorry -- The sum of the first n terms {s_n}

axiom cond1 (n : ℕ) (hn : n > 0) : seq_a (n + 1) ≠ seq_a n
axiom cond2 (n : ℕ) (hn : n > 0) : sum_s n + seq_a n = 2 * n

-- Theorem statement
theorem geometric_seq_an_minus_2 (n : ℕ) (hn : n > 0) : 
  ∃ r : ℝ, ∀ k : ℕ, seq_a (k + 1) - 2 = r * (seq_a k - 2) := 
sorry

end geometric_seq_an_minus_2_l64_6412


namespace max_legs_lengths_l64_6465

theorem max_legs_lengths (a x y : ℝ) (h₁ : x^2 + y^2 = a^2) (h₂ : 3 * x + 4 * y ≤ 5 * a) :
  3 * x + 4 * y = 5 * a → x = (3 * a / 5) ∧ y = (4 * a / 5) :=
by
  sorry

end max_legs_lengths_l64_6465


namespace number_of_saturday_sales_l64_6404

def caricatures_sold_on_saturday (total_earnings weekend_earnings price_per_drawing sunday_sales : ℕ) : ℕ :=
  (total_earnings - (sunday_sales * price_per_drawing)) / price_per_drawing

theorem number_of_saturday_sales : caricatures_sold_on_saturday 800 800 20 16 = 24 := 
by 
  sorry

end number_of_saturday_sales_l64_6404


namespace find_integer_value_of_a_l64_6474

-- Define the conditions for the equation and roots
def equation_has_two_distinct_negative_integer_roots (a : ℤ) : Prop :=
  ∃ x1 x2 : ℤ, x1 ≠ x2 ∧ x1 < 0 ∧ x2 < 0 ∧ (a^2 - 1) * x1^2 - 2 * (5 * a + 1) * x1 + 24 = 0 ∧ (a^2 - 1) * x2^2 - 2 * (5 * a + 1) * x2 + 24 = 0 ∧
  x1 = 6 / (a - 1) ∧ x2 = 4 / (a + 1)

-- Prove that the only integer value of a that satisfies these conditions is -2
theorem find_integer_value_of_a : 
  ∃ (a : ℤ), equation_has_two_distinct_negative_integer_roots a ∧ a = -2 := 
sorry

end find_integer_value_of_a_l64_6474


namespace total_carrots_l64_6456

/-- 
  If Pleasant Goat and Beautiful Goat each receive 6 carrots, and the other goats each receive 3 carrots, there will be 6 carrots left over.
  If Pleasant Goat and Beautiful Goat each receive 7 carrots, and the other goats each receive 5 carrots, there will be a shortage of 14 carrots.
  Prove the total number of carrots (n) is 45. 
--/
theorem total_carrots (X n : ℕ) 
  (h1 : n = 3 * X + 18) 
  (h2 : n = 5 * X) : 
  n = 45 := 
by
  sorry

end total_carrots_l64_6456


namespace smallest_n_inequality_l64_6448

theorem smallest_n_inequality :
  ∃ n : ℕ, (∀ x y z w : ℝ, (x^2 + y^2 + z^2 + w^2)^2 ≤ n * (x^4 + y^4 + z^4 + w^4)) ∧
  (∀ m : ℕ, (∀ x y z w : ℝ, (x^2 + y^2 + z^2 + w^2)^2 ≤ m * (x^4 + y^4 + z^4 + w^4)) → n ≤ m) :=
sorry

end smallest_n_inequality_l64_6448


namespace camels_horses_oxen_elephants_l64_6401

theorem camels_horses_oxen_elephants :
  ∀ (C H O E : ℝ),
  10 * C = 24 * H →
  H = 4 * O →
  6 * O = 4 * E →
  10 * E = 170000 →
  C = 4184.615384615385 →
  (4 * O) / H = 1 :=
by
  intros C H O E h1 h2 h3 h4 h5
  sorry

end camels_horses_oxen_elephants_l64_6401


namespace Peter_bought_4_notebooks_l64_6446

theorem Peter_bought_4_notebooks :
  (let green_notebooks := 2
   let black_notebook := 1
   let pink_notebook := 1
   green_notebooks + black_notebook + pink_notebook = 4) :=
by sorry

end Peter_bought_4_notebooks_l64_6446


namespace avg_age_women_is_52_l64_6410

-- Definitions
def avg_age_men (A : ℚ) := 9 * A
def total_increase := 36
def combined_age_replaced := 36 + 32
def combined_age_women := combined_age_replaced + total_increase
def avg_age_women (W : ℚ) := W / 2

-- Theorem statement
theorem avg_age_women_is_52 (A : ℚ) : avg_age_women combined_age_women = 52 :=
by
  sorry

end avg_age_women_is_52_l64_6410


namespace evaluate_expression_l64_6460

theorem evaluate_expression : 
  ( (2^12)^2 - (2^10)^2 ) / ( (2^11)^2 - (2^9)^2 ) = 4 :=
by
  sorry

end evaluate_expression_l64_6460


namespace coordinates_of_B_l64_6462

-- Define the initial coordinates of point A
def A : ℝ × ℝ := (1, -2)

-- Define the transformation to get point B from A
def B : ℝ × ℝ := (A.1 - 2, A.2 + 3)

theorem coordinates_of_B : B = (-1, 1) :=
by
  sorry

end coordinates_of_B_l64_6462


namespace grid_spiral_infinite_divisible_by_68_grid_spiral_unique_center_sums_l64_6447

theorem grid_spiral_infinite_divisible_by_68 (n : ℕ) :
  ∃ (k : ℕ), ∃ (m : ℕ), ∃ (t : ℕ), 
  let A := t + 0;
  let B := t + 4;
  let C := t + 12;
  let D := t + 8;
  (k = n * 68 ∧ (n ≥ 1)) ∧ 
  (m = A + B + C + D) ∧ (m % 68 = 0) := by
  sorry

theorem grid_spiral_unique_center_sums (n : ℕ) :
  ∀ (i j : ℕ), 
  let Si := n * 68 + i;
  let Sj := n * 68 + j;
  ¬ (Si = Sj) := by
  sorry

end grid_spiral_infinite_divisible_by_68_grid_spiral_unique_center_sums_l64_6447


namespace find_base_and_digit_sum_l64_6418

theorem find_base_and_digit_sum (n d : ℕ) (h1 : 4 * n^2 + 5 * n + d = 392) (h2 : 4 * n^2 + 5 * n + 7 = 740 + 7 * d) : n + d = 12 :=
by
  sorry

end find_base_and_digit_sum_l64_6418


namespace calories_in_250g_of_lemonade_l64_6436

structure Lemonade :=
(lemon_juice_grams : ℕ)
(sugar_grams : ℕ)
(water_grams : ℕ)
(lemon_juice_calories_per_100g : ℕ)
(sugar_calories_per_100g : ℕ)
(water_calories_per_100g : ℕ)

def calorie_count (l : Lemonade) : ℕ :=
(l.lemon_juice_grams * l.lemon_juice_calories_per_100g / 100) +
(l.sugar_grams * l.sugar_calories_per_100g / 100) +
(l.water_grams * l.water_calories_per_100g / 100)

def total_weight (l : Lemonade) : ℕ :=
l.lemon_juice_grams + l.sugar_grams + l.water_grams

def caloric_density (l : Lemonade) : ℚ :=
calorie_count l / total_weight l

theorem calories_in_250g_of_lemonade :
  ∀ (l : Lemonade), 
  l = { lemon_juice_grams := 200, sugar_grams := 300, water_grams := 500,
        lemon_juice_calories_per_100g := 40,
        sugar_calories_per_100g := 390,
        water_calories_per_100g := 0 } →
  (caloric_density l * 250 = 312.5) :=
sorry

end calories_in_250g_of_lemonade_l64_6436


namespace number_of_tests_initially_l64_6483

-- Given conditions
variables (n S : ℕ)
variables (h1 : S / n = 70)
variables (h2 : S = 70 * n)
variables (h3 : (S - 55) / (n - 1) = 75)

-- Prove the number of tests initially, n, is 4.
theorem number_of_tests_initially (n : ℕ) (S : ℕ)
  (h1 : S / n = 70) (h2 : S = 70 * n) (h3 : (S - 55) / (n - 1) = 75) :
  n = 4 :=
sorry

end number_of_tests_initially_l64_6483


namespace polynomial_expansion_l64_6415

variable (x : ℝ)

theorem polynomial_expansion : 
  (-2*x - 1) * (3*x - 2) = -6*x^2 + x + 2 :=
by
  sorry

end polynomial_expansion_l64_6415


namespace farmer_total_land_l64_6495

noncomputable def total_land_owned_by_farmer (cleared_land_with_tomato : ℝ) (cleared_percentage : ℝ) (grape_percentage : ℝ) (potato_percentage : ℝ) : ℝ :=
  let cleared_land := cleared_percentage
  let total_clearance_with_tomato := cleared_land_with_tomato
  let unused_cleared_percentage := 1 - grape_percentage - potato_percentage
  let total_cleared_land := total_clearance_with_tomato / unused_cleared_percentage
  total_cleared_land / cleared_land

theorem farmer_total_land (cleared_land_with_tomato : ℝ) (cleared_percentage : ℝ) (grape_percentage : ℝ) (potato_percentage : ℝ) :
  (cleared_land_with_tomato = 450) →
  (cleared_percentage = 0.90) →
  (grape_percentage = 0.10) →
  (potato_percentage = 0.80) →
  total_land_owned_by_farmer cleared_land_with_tomato 90 10 80 = 1666.6667 :=
by
  intro h1 h2 h3 h4
  sorry

end farmer_total_land_l64_6495


namespace calc_radical_power_l64_6458

theorem calc_radical_power : (Real.sqrt (Real.sqrt (Real.sqrt (Real.sqrt 16))) ^ 12) = 4096 := sorry

end calc_radical_power_l64_6458


namespace outlet_pipe_emptying_time_l64_6451

noncomputable def fill_rate_pipe1 : ℝ := 1 / 18
noncomputable def fill_rate_pipe2 : ℝ := 1 / 30
noncomputable def empty_rate_outlet_pipe (x : ℝ) : ℝ := 1 / x
noncomputable def combined_rate (x : ℝ) : ℝ := fill_rate_pipe1 + fill_rate_pipe2 - empty_rate_outlet_pipe x
noncomputable def total_fill_time : ℝ := 0.06666666666666665

theorem outlet_pipe_emptying_time : ∃ x : ℝ, combined_rate x = 1 / total_fill_time ∧ x = 45 :=
by
  sorry

end outlet_pipe_emptying_time_l64_6451


namespace mutually_exclusive_not_opposite_l64_6452

namespace event_theory

-- Definition to represent the student group
structure Group where
  boys : ℕ
  girls : ℕ

def student_group : Group := {boys := 3, girls := 2}

-- Definition of events
inductive Event
| AtLeastOneBoyAndOneGirl
| ExactlyOneBoyExactlyTwoBoys
| AtLeastOneBoyAllGirls
| AtMostOneBoyAllGirls

open Event

-- Conditions provided in the problem
def condition (grp : Group) : Prop :=
  grp.boys = 3 ∧ grp.girls = 2

-- The main statement to prove in Lean
theorem mutually_exclusive_not_opposite :
  condition student_group →
  ∃ e₁ e₂ : Event, e₁ = ExactlyOneBoyExactlyTwoBoys ∧ e₂ = ExactlyOneBoyExactlyTwoBoys ∧ (
    (e₁ ≠ e₂) ∧ (¬ (e₁ = e₂ ∧ e₁ = ExactlyOneBoyExactlyTwoBoys))
  ) :=
by
  sorry

end event_theory

end mutually_exclusive_not_opposite_l64_6452


namespace simplify_expression_l64_6472

theorem simplify_expression : (3 + 3 + 5) / 2 - 1 / 2 = 5 := by
  sorry

end simplify_expression_l64_6472


namespace domain_of_composite_function_l64_6422

theorem domain_of_composite_function (f : ℝ → ℝ) :
  (∀ x, -2 ≤ x ∧ x ≤ 2 → -1 ≤ x + 1) →
  (∀ x, -1 ≤ 2*x - 1 ∧ 2*x - 1 ≤ 3 → -1 ≤ 2*x + 1 ∧ 2*x + 1 ≤ 3 → 0 ≤ x ∧ x ≤ 1) :=
by
  sorry

end domain_of_composite_function_l64_6422


namespace max_marks_l64_6443

-- Define the conditions
def passing_marks (M : ℕ) : ℕ := 40 * M / 100

def Ravish_got_marks : ℕ := 40
def marks_failed_by : ℕ := 40

-- Lean statement to prove
theorem max_marks (M : ℕ) (h : passing_marks M = Ravish_got_marks + marks_failed_by) : M = 200 :=
by
  sorry

end max_marks_l64_6443


namespace min_x_4y_is_minimum_l64_6478

noncomputable def min_value_x_4y (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : (1 / x) + (1 / (2 * y)) = 2) : ℝ :=
  x + 4 * y

theorem min_x_4y_is_minimum : ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ (1 / x + 1 / (2 * y) = 2) ∧ (x + 4 * y = (3 / 2) + Real.sqrt 2) :=
sorry

end min_x_4y_is_minimum_l64_6478


namespace avg_difference_in_circumferences_l64_6459

-- Define the conditions
def inner_circle_diameter : ℝ := 30
def min_track_width : ℝ := 10
def max_track_width : ℝ := 15

-- Define the average difference in the circumferences of the two circles
theorem avg_difference_in_circumferences :
  let avg_width := (min_track_width + max_track_width) / 2
  let outer_circle_diameter := inner_circle_diameter + 2 * avg_width
  let inner_circle_circumference := Real.pi * inner_circle_diameter
  let outer_circle_circumference := Real.pi * outer_circle_diameter
  outer_circle_circumference - inner_circle_circumference = 25 * Real.pi :=
by
  sorry

end avg_difference_in_circumferences_l64_6459


namespace common_ratio_l64_6485

-- Problem Statement Definitions
variable (a1 q : ℝ)

-- Given Conditions
def a3 := a1 * q^2
def S3 := a1 * (1 + q + q^2)

-- Proof Statement
theorem common_ratio (h1 : a3 = 3/2) (h2 : S3 = 9/2) : q = 1 ∨ q = -1/2 := by
  sorry

end common_ratio_l64_6485


namespace area_of_triangle_formed_by_lines_l64_6450

theorem area_of_triangle_formed_by_lines (x y : ℝ) (h1 : y = x) (h2 : x = -5) :
  let base := 5
  let height := 5
  let area := (1 / 2 : ℝ) * base * height
  area = 12.5 := 
by
  sorry

end area_of_triangle_formed_by_lines_l64_6450


namespace g_at_pi_over_4_l64_6416

noncomputable def f (x : ℝ) : ℝ := (Real.sqrt 2) / 2 * Real.sin (2 * x) + (Real.sqrt 6) / 2 * Real.cos (2 * x)

noncomputable def g (x : ℝ) : ℝ := f (x - Real.pi / 4)

theorem g_at_pi_over_4 : g (Real.pi / 4) = (Real.sqrt 6) / 2 := by
  sorry

end g_at_pi_over_4_l64_6416


namespace linear_function_points_relation_l64_6419

theorem linear_function_points_relation (x1 x2 : ℝ) (y1 y2 : ℝ) 
  (h1 : y1 = 5 * x1 - 3) 
  (h2 : y2 = 5 * x2 - 3) 
  (h3 : x1 < x2) : 
  y1 < y2 :=
sorry

end linear_function_points_relation_l64_6419


namespace solve_system_eq_pos_reals_l64_6484

theorem solve_system_eq_pos_reals (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (h1 : x^2 + y^2 + x * y = 7)
  (h2 : x^2 + z^2 + x * z = 13)
  (h3 : y^2 + z^2 + y * z = 19) :
  x = 1 ∧ y = 2 ∧ z = 3 :=
sorry

end solve_system_eq_pos_reals_l64_6484


namespace area_of_frame_l64_6491

def width : ℚ := 81 / 4
def depth : ℚ := 148 / 9
def area (w d : ℚ) : ℚ := w * d

theorem area_of_frame : area width depth = 333 := by
  sorry

end area_of_frame_l64_6491


namespace find_z_plus_one_over_y_l64_6411

theorem find_z_plus_one_over_y 
  (x y z : ℝ) 
  (h1 : 0 < x)
  (h2 : 0 < y)
  (h3 : 0 < z)
  (h4 : x * y * z = 1)
  (h5 : x + 1/z = 4)
  (h6 : y + 1/x = 20) :
  z + 1/y = 26 / 79 :=
by
  sorry

end find_z_plus_one_over_y_l64_6411


namespace tan_alpha_eq_two_l64_6499

theorem tan_alpha_eq_two (α : ℝ) (h1 : α ∈ Set.Ioc 0 (Real.pi / 2))
    (h2 : Real.sin ((Real.pi / 4) - α) * Real.sin ((Real.pi / 4) + α) = -3 / 10) :
    Real.tan α = 2 := by
  sorry

end tan_alpha_eq_two_l64_6499


namespace ant_population_percentage_l64_6431

theorem ant_population_percentage (R : ℝ) 
  (h1 : 0.45 * R = 46.75) 
  (h2 : R * 0.55 = 46.75) : 
  R = 0.85 := 
by 
  sorry

end ant_population_percentage_l64_6431


namespace max_value_of_z_l64_6423

theorem max_value_of_z
  (x y : ℝ)
  (h1 : y ≥ x)
  (h2 : x + y ≤ 1)
  (h3 : y ≥ -1) :
  ∃ x y, (y ≥ x) ∧ (x + y ≤ 1) ∧ (y ≥ -1) ∧ (2 * x - y = 1 / 2) := by 
  sorry

end max_value_of_z_l64_6423


namespace magnitude_BC_range_l64_6476

theorem magnitude_BC_range (AB AC : EuclideanSpace ℝ (Fin 2)) 
  (h₁ : ‖AB‖ = 18) (h₂ : ‖AC‖ = 5) : 
  13 ≤ ‖AC - AB‖ ∧ ‖AC - AB‖ ≤ 23 := 
  sorry

end magnitude_BC_range_l64_6476


namespace table_coverage_percentage_l64_6464

def A := 204  -- Total area of the runners
def T := 175  -- Area of the table
def A2 := 24  -- Area covered by exactly two layers of runner
def A3 := 20  -- Area covered by exactly three layers of runner

theorem table_coverage_percentage : 
  (A - 2 * A2 - 3 * A3 + A2 + A3) / T * 100 = 80 := 
by
  sorry

end table_coverage_percentage_l64_6464


namespace discarded_number_l64_6487

theorem discarded_number (S x : ℕ) (h1 : S / 50 = 50) (h2 : (S - x - 55) / 48 = 50) : x = 45 :=
by
  sorry

end discarded_number_l64_6487


namespace cost_of_each_ruler_l64_6454
-- Import the necessary library

-- Define the conditions and statement
theorem cost_of_each_ruler (students : ℕ) (rulers_each : ℕ) (cost_per_ruler : ℕ) (total_cost : ℕ) 
  (cond1 : students = 42)
  (cond2 : students / 2 < 42 / 2)
  (cond3 : cost_per_ruler > rulers_each)
  (cond4 : students * rulers_each * cost_per_ruler = 2310) : 
  cost_per_ruler = 11 :=
sorry

end cost_of_each_ruler_l64_6454


namespace mom_t_shirts_total_l64_6444

-- Definitions based on the conditions provided in the problem
def packages : ℕ := 71
def t_shirts_per_package : ℕ := 6

-- The statement to prove that the total number of white t-shirts is 426
theorem mom_t_shirts_total : packages * t_shirts_per_package = 426 := by sorry

end mom_t_shirts_total_l64_6444


namespace bret_spends_77_dollars_l64_6405

def num_people : ℕ := 4
def main_meal_cost : ℝ := 12.0
def num_appetizers : ℕ := 2
def appetizer_cost : ℝ := 6.0
def tip_rate : ℝ := 0.20
def rush_order_fee : ℝ := 5.0

def total_cost (num_people : ℕ) (main_meal_cost : ℝ) (num_appetizers : ℕ) (appetizer_cost : ℝ) (tip_rate : ℝ) (rush_order_fee : ℝ) : ℝ :=
  let main_meal_total := num_people * main_meal_cost
  let appetizer_total := num_appetizers * appetizer_cost
  let subtotal := main_meal_total + appetizer_total
  let tip := tip_rate * subtotal
  subtotal + tip + rush_order_fee

theorem bret_spends_77_dollars :
  total_cost num_people main_meal_cost num_appetizers appetizer_cost tip_rate rush_order_fee = 77.0 :=
by
  sorry

end bret_spends_77_dollars_l64_6405


namespace car_average_speed_l64_6421

theorem car_average_speed 
  (d1 d2 d3 d5 d6 d7 d8 : ℝ) 
  (t_total : ℝ) 
  (avg_speed : ℝ)
  (h1 : d1 = 90)
  (h2 : d2 = 50)
  (h3 : d3 = 70)
  (h5 : d5 = 80)
  (h6 : d6 = 60)
  (h7 : d7 = -40)
  (h8 : d8 = -55)
  (h_t_total : t_total = 8)
  (h_avg_speed : avg_speed = (d1 + d2 + d3 + d5 + d6 + d7 + d8) / t_total) :
  avg_speed = 31.875 := 
by sorry

end car_average_speed_l64_6421


namespace second_number_is_22_l64_6466

theorem second_number_is_22 
    (A B : ℤ)
    (h1 : A - B = 88) 
    (h2 : A = 110) :
    B = 22 :=
by
  sorry

end second_number_is_22_l64_6466


namespace sum_of_roots_l64_6481

theorem sum_of_roots : 
  (∃ x1 x2 : ℚ, (3 * x1 + 4) * (2 * x1 - 12) = 0 ∧ (3 * x2 + 4) * (2 * x2 - 12) = 0 ∧ x1 ≠ x2 ∧ x1 + x2 = 14 / 3) :=
sorry

end sum_of_roots_l64_6481


namespace objective_function_range_l64_6470

noncomputable def feasible_region (A B C : ℝ × ℝ) := 
  let (x, y) := A
  let (x1, y1) := B 
  let (x2, y2) := C 
  {p : ℝ × ℝ | True} -- The exact feasible region description is not specified

theorem objective_function_range
  (A B C: ℝ × ℝ)
  (a b : ℝ)
  (x y : ℝ)
  (hA : A = (x, y))
  (hB : B = (1, 1))
  (hC : C = (5, 2))
  (h1 : a + b = 3)
  (h2 : 5 * a + 2 * b = 12) :
  let z := a * x + b * y
  3 ≤ z ∧ z ≤ 12 :=
by
  sorry

end objective_function_range_l64_6470


namespace arithmetic_expression_eval_l64_6407

theorem arithmetic_expression_eval : 3 + (12 / 3 - 1) ^ 2 = 12 := by
  sorry

end arithmetic_expression_eval_l64_6407


namespace find_sum_of_m1_m2_l64_6420

-- Define the quadratic equation and the conditions
def quadratic (m : ℂ) (x : ℂ) : ℂ := m * x^2 - (3 * m - 2) * x + 7

-- Define the roots a and b
def are_roots (m a b : ℂ) : Prop := quadratic m a = 0 ∧ quadratic m b = 0

-- The condition given in the problem
def root_condition (a b : ℂ) : Prop := a / b + b / a = 3 / 2

-- Main theorem to be proved
theorem find_sum_of_m1_m2 (m1 m2 a1 a2 b1 b2 : ℂ) 
  (h1 : are_roots m1 a1 b1) 
  (h2 : are_roots m2 a2 b2) 
  (hc1 : root_condition a1 b1) 
  (hc2 : root_condition a2 b2) : 
  m1 + m2 = 73 / 18 :=
by sorry

end find_sum_of_m1_m2_l64_6420


namespace distance_between_centers_l64_6445

noncomputable def distance_centers_inc_exc (PQ PR QR: ℝ) (hPQ: PQ = 17) (hPR: PR = 15) (hQR: QR = 8) : ℝ :=
  let s := (PQ + PR + QR) / 2
  let area := Real.sqrt (s * (s - PQ) * (s - PR) * (s - QR))
  let r := area / s
  let r' := area / (s - QR)
  let PU := s - PQ
  let PV := s
  let PI := Real.sqrt ((PU)^2 + (r)^2)
  let PE := Real.sqrt ((PV)^2 + (r')^2)
  PE - PI

theorem distance_between_centers (PQ PR QR : ℝ) (hPQ: PQ = 17) (hPR: PR = 15) (hQR: QR = 8) :
  distance_centers_inc_exc PQ PR QR hPQ hPR hQR = 5 * Real.sqrt 17 - 3 * Real.sqrt 2 :=
by sorry

end distance_between_centers_l64_6445


namespace time_to_plough_together_l64_6441

def work_rate_r := 1 / 15
def work_rate_s := 1 / 20
def combined_work_rate := work_rate_r + work_rate_s
def total_field := 1
def T := total_field / combined_work_rate

theorem time_to_plough_together : T = 60 / 7 :=
by
  -- Here you would provide the proof steps if it were required
  -- Since the proof steps are not needed, we indicate the end with sorry
  sorry

end time_to_plough_together_l64_6441


namespace parameter_conditions_l64_6475

theorem parameter_conditions (p x y : ℝ) :
  (x - p)^2 = 16 * (y - 3 + p) →
  y^2 + ((x - 3) / (|x| - 3))^2 = 1 →
  |x| ≠ 3 →
  p > 3 ∧ 
  ((p ≤ 4 ∨ p ≥ 12) ∧ (p < 19 ∨ 19 < p)) :=
sorry

end parameter_conditions_l64_6475


namespace smallest_value_l64_6489

noncomputable def smallest_possible_value (a b : ℝ) : ℝ := 2 * a + b

theorem smallest_value (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a^2 ≥ 3 * b) (h4 : b^2 ≥ (8 / 9) * a) :
  smallest_possible_value a b = 5.602 :=
sorry

end smallest_value_l64_6489


namespace factorize_difference_of_squares_l64_6482

theorem factorize_difference_of_squares (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) := 
by 
  sorry

end factorize_difference_of_squares_l64_6482


namespace winningTicketProbability_l64_6435

-- Given conditions
def sharpBallProbability : ℚ := 1 / 30
def prizeBallsProbability : ℚ := 1 / (Nat.descFactorial 50 6)

-- The target probability that we are supposed to prove
def targetWinningProbability : ℚ := 1 / 476721000

-- Main theorem stating the required probability calculation
theorem winningTicketProbability :
  sharpBallProbability * prizeBallsProbability = targetWinningProbability :=
  sorry

end winningTicketProbability_l64_6435


namespace vector_addition_example_l64_6438

theorem vector_addition_example :
  let a := (1, 2)
  let b := (-2, 1)
  a.1 + 2 * b.1 = -3 ∧ a.2 + 2 * b.2 = 4 :=
by
  sorry

end vector_addition_example_l64_6438


namespace adding_2_to_odd_integer_can_be_prime_l64_6428

def is_odd (n : ℤ) : Prop := n % 2 ≠ 0
def is_prime (n : ℤ) : Prop := n > 1 ∧ ∀ m : ℤ, m ∣ n → m = 1 ∨ m = n

theorem adding_2_to_odd_integer_can_be_prime :
  ∃ n : ℤ, is_odd n ∧ is_prime (n + 2) :=
by
  sorry

end adding_2_to_odd_integer_can_be_prime_l64_6428


namespace horizontal_asymptote_value_l64_6457

theorem horizontal_asymptote_value :
  (∃ y : ℝ, ∀ x : ℝ, (y = (18 * x^5 + 6 * x^3 + 3 * x^2 + 5 * x + 4) / (6 * x^5 + 4 * x^3 + 5 * x^2 + 2 * x + 1)) → y = 3) :=
by
  sorry

end horizontal_asymptote_value_l64_6457


namespace factorize_expression_l64_6461

theorem factorize_expression (a b : ℝ) :
  ab^(3 : ℕ) - 4 * ab = ab * (b + 2) * (b - 2) :=
by
  -- proof to be provided
  sorry

end factorize_expression_l64_6461


namespace area_error_percent_l64_6479

theorem area_error_percent (L W : ℝ) (L_pos : 0 < L) (W_pos : 0 < W) :
  let A := L * W
  let A_measured := (1.05 * L) * (0.96 * W)
  let error_percent := ((A_measured - A) / A) * 100
  error_percent = 0.8 :=
by
  let A := L * W
  let A_measured := (1.05 * L) * (0.96 * W)
  let error := A_measured - A
  let error_percent := (error / A) * 100
  sorry

end area_error_percent_l64_6479


namespace sum_q_p_values_is_neg42_l64_6442

def p (x : Int) : Int := 2 * Int.natAbs x - 1

def q (x : Int) : Int := -(Int.natAbs x) - 1

def values : List Int := [-4, -3, -2, -1, 0, 1, 2, 3, 4]

def q_p_sum : Int :=
  let q_p_values := values.map (λ x => q (p x))
  q_p_values.sum

theorem sum_q_p_values_is_neg42 : q_p_sum = -42 :=
  by
    sorry

end sum_q_p_values_is_neg42_l64_6442


namespace audrey_older_than_heracles_l64_6486

variable (A H : ℕ)
variable (hH : H = 10)
variable (hFutureAge : A + 3 = 2 * H)

theorem audrey_older_than_heracles : A - H = 7 :=
by
  have h1 : H = 10 := by assumption
  have h2 : A + 3 = 2 * H := by assumption
  -- Proof is omitted
  sorry

end audrey_older_than_heracles_l64_6486


namespace triangle_area_l64_6425

theorem triangle_area (a b c : ℝ) (h1 : a = 6) (h2 : b = 8) (h3 : c = 10) :
  a^2 + b^2 = c^2 ∧ 0.5 * a * b = 24 :=
by {
  sorry
}

end triangle_area_l64_6425


namespace sum_of_star_angles_l64_6439

theorem sum_of_star_angles :
  let n := 12
  let angle_per_arc := 360 / n
  let arcs_per_tip := 3
  let internal_angle_per_tip := 360 - arcs_per_tip * angle_per_arc
  let sum_of_angles := n * (360 - internal_angle_per_tip)
  sum_of_angles = 1080 :=
by
  sorry

end sum_of_star_angles_l64_6439


namespace exponentiation_evaluation_l64_6433

theorem exponentiation_evaluation :
  (8^3 / 8^2) * 2^10 = 8192 := by
  sorry

end exponentiation_evaluation_l64_6433


namespace cost_to_fix_car_l64_6477

variable {S A : ℝ}

theorem cost_to_fix_car (h1 : A = 3 * S + 50) (h2 : S + A = 450) : A = 350 := 
by
  sorry

end cost_to_fix_car_l64_6477


namespace monotonic_intervals_value_of_a_inequality_a_minus_one_l64_6493

noncomputable def f (a x : ℝ) : ℝ := a * x + Real.log x

theorem monotonic_intervals (a : ℝ) :
  (∀ x, 0 < x → 0 ≤ a → 0 < (a * x + 1) / x) ∧
  (∀ x, 0 < x → a < 0 → (0 < x ∧ x < -1/a → 0 < (a * x + 1) / x) ∧
    (-1/a < x → 0 > (a * x + 1) / x)) :=
sorry

theorem value_of_a (a : ℝ) (h_a : a < 0) (h_max : (∀ x, x ∈ Set.Icc 0 e → f a x ≤ -2) ∧ (∃ x, x ∈ Set.Icc 0 e ∧ f a x = -2)) :
  a = -Real.exp 1 := 
sorry

theorem inequality_a_minus_one (a : ℝ) (h_a : a = -1) :
  (∀ x, 0 < x → x * |f a x| > Real.log x + 1/2 * x) :=
sorry

end monotonic_intervals_value_of_a_inequality_a_minus_one_l64_6493


namespace marble_draw_l64_6403

/-- A container holds 30 red marbles, 25 green marbles, 23 yellow marbles,
15 blue marbles, 10 white marbles, and 7 black marbles. Prove that the
minimum number of marbles that must be drawn from the container without
replacement to ensure that at least 10 marbles of a single color are drawn
is 53. -/
theorem marble_draw (R G Y B W Bl : ℕ) (hR : R = 30) (hG : G = 25)
                               (hY : Y = 23) (hB : B = 15) (hW : W = 10)
                               (hBl : Bl = 7) : 
  ∃ (n : ℕ), n = 53 ∧ (∀ (x : ℕ), x ≠ n → 
  (x ≤ R → x ≤ G → x ≤ Y → x ≤ B → x ≤ W → x ≤ Bl → x < 10)) := 
by
  sorry

end marble_draw_l64_6403


namespace complex_division_l64_6488

theorem complex_division (i : ℂ) (h : i^2 = -1) : (2 + i) / (1 - 2 * i) = i := 
by
  sorry

end complex_division_l64_6488


namespace hide_and_seek_friends_l64_6413

open Classical

variables (A B V G D : Prop)

/-- Conditions -/
axiom cond1 : A → (B ∧ ¬V)
axiom cond2 : B → (G ∨ D)
axiom cond3 : ¬V → (¬B ∧ ¬D)
axiom cond4 : ¬A → (B ∧ ¬G)

/-- Proof that Alex played hide and seek with Boris, Vasya, and Denis -/
theorem hide_and_seek_friends : B ∧ V ∧ D := by
  sorry

end hide_and_seek_friends_l64_6413


namespace abs_neg_ten_l64_6453

theorem abs_neg_ten : abs (-10) = 10 := 
by {
  sorry
}

end abs_neg_ten_l64_6453


namespace compare_a_b_c_l64_6426

noncomputable
def a : ℝ := Real.exp 0.1 - 1

def b : ℝ := 0.1

noncomputable
def c : ℝ := Real.log 1.1

theorem compare_a_b_c : a > b ∧ b > c := by
  sorry

end compare_a_b_c_l64_6426


namespace fraction_of_ABCD_is_shaded_l64_6402

noncomputable def squareIsDividedIntoTriangles : Type := sorry
noncomputable def areTrianglesIdentical (s : squareIsDividedIntoTriangles) : Prop := sorry
noncomputable def isFractionShadedCorrect : Prop := 
  ∃ (s : squareIsDividedIntoTriangles), 
  areTrianglesIdentical s ∧ 
  (7 / 16 : ℚ) = 7 / 16

theorem fraction_of_ABCD_is_shaded (s : squareIsDividedIntoTriangles) :
  areTrianglesIdentical s → (7 / 16 : ℚ) = 7 / 16 :=
sorry

end fraction_of_ABCD_is_shaded_l64_6402


namespace train_crossing_time_l64_6480

-- Define the conditions
def length_of_train : ℕ := 200  -- in meters
def speed_of_train_kmph : ℕ := 90  -- in km per hour
def length_of_tunnel : ℕ := 2500  -- in meters

-- Conversion of speed from kmph to m/s
def speed_of_train_mps : ℕ := speed_of_train_kmph * 1000 / 3600

-- Define the total distance to be covered (train length + tunnel length)
def total_distance : ℕ := length_of_train + length_of_tunnel

-- Define the expected time to cross the tunnel (in seconds)
def expected_time : ℕ := 108

-- The theorem statement to prove
theorem train_crossing_time : (total_distance / speed_of_train_mps) = expected_time := 
by
  sorry

end train_crossing_time_l64_6480


namespace maximal_number_of_coins_l64_6406

noncomputable def largest_number_of_coins (n k : ℕ) : Prop :=
n < 100 ∧ n = 12 * k + 3

theorem maximal_number_of_coins (n k : ℕ) : largest_number_of_coins n k → n = 99 :=
by
  sorry

end maximal_number_of_coins_l64_6406


namespace bottle_caps_total_l64_6430

theorem bottle_caps_total (groups : ℕ) (bottle_caps_per_group : ℕ) (h1 : groups = 7) (h2 : bottle_caps_per_group = 5) : (groups * bottle_caps_per_group = 35) :=
by
  sorry

end bottle_caps_total_l64_6430


namespace distance_internal_tangent_l64_6497

noncomputable def radius_O := 5
noncomputable def distance_external := 9

theorem distance_internal_tangent (radius_O radius_dist_external : ℝ) 
  (h1 : radius_O = 5) (h2: radius_dist_external = 9) : 
  ∃ r : ℝ, r = 4 ∧ abs (r - radius_O) = 1 := by
  sorry

end distance_internal_tangent_l64_6497


namespace max_candy_one_student_l64_6471

theorem max_candy_one_student (n : ℕ) (mu : ℕ) (at_least_two : ℕ → Prop) :
  n = 35 → mu = 6 →
  (∀ x, at_least_two x → x ≥ 2) →
  ∃ max_candy : ℕ, (∀ x, at_least_two x → x ≤ max_candy) ∧ max_candy = 142 :=
by
sorry

end max_candy_one_student_l64_6471


namespace problem_am_gm_inequality_l64_6492

theorem problem_am_gm_inequality
  (a b c : ℝ)
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_sum_sq : a^2 + b^2 + c^2 = 3) : 
  (1 / (1 + a * b)) + (1 / (1 + b * c)) + (1 / (1 + c * a)) ≥ 3 / 2 :=
by
  sorry

end problem_am_gm_inequality_l64_6492


namespace moles_of_Cl2_combined_l64_6400

theorem moles_of_Cl2_combined (nCH4 : ℕ) (nCl2 : ℕ) (nHCl : ℕ) 
  (h1 : nCH4 = 3) 
  (h2 : nHCl = nCl2) 
  (h3 : nHCl ≤ nCH4) : 
  nCl2 = 3 :=
by
  sorry

end moles_of_Cl2_combined_l64_6400


namespace pipe_B_fills_6_times_faster_l64_6463

theorem pipe_B_fills_6_times_faster :
  let R_A := 1 / 32
  let combined_rate := 7 / 32
  let R_B := combined_rate - R_A
  (R_B / R_A = 6) :=
by
  let R_A := 1 / 32
  let combined_rate := 7 / 32
  let R_B := combined_rate - R_A
  sorry

end pipe_B_fills_6_times_faster_l64_6463


namespace emily_subtracts_99_from_50_squared_l64_6469

theorem emily_subtracts_99_from_50_squared :
  (50 - 1) ^ 2 = 50 ^ 2 - 99 := by
  sorry

end emily_subtracts_99_from_50_squared_l64_6469


namespace find_time_ball_hits_ground_l64_6496

theorem find_time_ball_hits_ground :
  ∃ t : ℝ, (-16 * t^2 + 40 * t + 30 = 0) ∧ (t = (5 + 5 * Real.sqrt 22) / 4) := 
by
  sorry

end find_time_ball_hits_ground_l64_6496


namespace value_of_m_l64_6468

theorem value_of_m (x m : ℝ) (h : x ≠ 3) (H : (x / (x - 3) = 2 - m / (3 - x))) : m = 3 :=
sorry

end value_of_m_l64_6468


namespace distance_is_660_km_l64_6429

def distance_between_cities (x y : ℝ) : ℝ :=
  3.3 * (x + y)

def train_A_dep_earlier (x y : ℝ) : Prop :=
  3.4 * (x + y) = 3.3 * (x + y) + 14

def train_B_dep_earlier (x y : ℝ) : Prop :=
  3.6 * (x + y) = 3.3 * (x + y) + 9

theorem distance_is_660_km (x y : ℝ) (hx : train_A_dep_earlier x y) (hy : train_B_dep_earlier x y) :
    distance_between_cities x y = 660 :=
sorry

end distance_is_660_km_l64_6429


namespace angles_cosine_condition_l64_6467

theorem angles_cosine_condition {A B : ℝ} (hA : 0 < A ∧ A < π) (hB : 0 < B ∧ B < π) :
  (A > B) ↔ (Real.cos A < Real.cos B) :=
by
sorry

end angles_cosine_condition_l64_6467


namespace units_digit_33_exp_l64_6494

def units_digit_of_power_cyclic (base exponent : ℕ) (cycle : List ℕ) : ℕ :=
  cycle.get! (exponent % cycle.length)

theorem units_digit_33_exp (n : ℕ) (h1 : 33 = 1 + 4 * 8) (h2 : 44 = 4 * 11) :
  units_digit_of_power_cyclic 33 (33 * 44 ^ 44) [3, 9, 7, 1] = 3 :=
by
  sorry

end units_digit_33_exp_l64_6494


namespace sum_of_50th_row_l64_6455

-- Define triangular numbers
def T (n : ℕ) : ℕ := n * (n + 1) / 2

-- Define the sum of numbers in the nth row
def f (n : ℕ) : ℕ :=
  if n = 0 then 0
  else if n = 1 then 1 -- T_1 is 1 for the base case
  else 2 * f (n - 1) + n * (n + 1)

-- Prove the sum of the 50th row
theorem sum_of_50th_row : f 50 = 2^50 - 2550 := 
  sorry

end sum_of_50th_row_l64_6455


namespace power_comparison_l64_6437

theorem power_comparison (A B : ℝ) (h1 : A = 1997 ^ (1998 ^ 1999)) (h2 : B = 1999 ^ (1998 ^ 1997)) (h3 : 1997 < 1999) :
  A > B :=
by
  sorry

end power_comparison_l64_6437


namespace black_grid_probability_l64_6498

theorem black_grid_probability : 
  (let n := 4
   let unit_squares := n * n
   let pairs := unit_squares / 2
   let probability_each_pair := (1:ℝ) / 4
   let total_probability := probability_each_pair ^ pairs
   total_probability = (1:ℝ) / 65536) :=
by
  let n := 4
  let unit_squares := n * n
  let pairs := unit_squares / 2
  let probability_each_pair := (1:ℝ) / 4
  let total_probability := probability_each_pair ^ pairs
  sorry

end black_grid_probability_l64_6498


namespace fraction_of_single_female_students_l64_6424

variables (total_students : ℕ) (male_students married_students married_male_students female_students single_female_students : ℕ)

-- Given conditions
def condition1 : male_students = (7 * total_students) / 10 := sorry
def condition2 : married_students = (3 * total_students) / 10 := sorry
def condition3 : married_male_students = male_students / 7 := sorry

-- Derived conditions
def condition4 : female_students = total_students - male_students := sorry
def condition5 : married_female_students = married_students - married_male_students := sorry
def condition6 : single_female_students = female_students - married_female_students := sorry

-- The proof goal
theorem fraction_of_single_female_students 
  (h1 : male_students = (7 * total_students) / 10)
  (h2 : married_students = (3 * total_students) / 10)
  (h3 : married_male_students = male_students / 7)
  (h4 : female_students = total_students - male_students)
  (h5 : married_female_students = married_students - married_male_students)
  (h6 : single_female_students = female_students - married_female_students) :
  (single_female_students : ℚ) / (female_students : ℚ) = 1 / 3 :=
sorry

end fraction_of_single_female_students_l64_6424


namespace largest_fraction_l64_6449

theorem largest_fraction (A B C D E : ℚ)
    (hA: A = 5 / 11)
    (hB: B = 7 / 16)
    (hC: C = 23 / 50)
    (hD: D = 99 / 200)
    (hE: E = 202 / 403) : 
    E > A ∧ E > B ∧ E > C ∧ E > D :=
by
  sorry

end largest_fraction_l64_6449


namespace arithmetic_sequence_diff_l64_6490

-- Define the arithmetic sequence properties
def is_arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

-- Given conditions
variable (a : ℕ → ℤ)
variable (h1 : is_arithmetic_sequence a 2)

-- Prove that a_5 - a_2 = 6
theorem arithmetic_sequence_diff : a 5 - a 2 = 6 :=
by sorry

end arithmetic_sequence_diff_l64_6490


namespace parallel_lines_perpendicular_lines_l64_6434

-- Definitions of the lines
def l1 (a x y : ℝ) := x + a * y - 2 * a - 2 = 0
def l2 (a x y : ℝ) := a * x + y - 1 - a = 0

-- Statement for parallel lines
theorem parallel_lines (a : ℝ) : (∀ x y, l1 a x y → l2 a x y → x = 0 ∨ x = 1) → a = 1 :=
by 
  -- proof outline
  sorry

-- Statement for perpendicular lines
theorem perpendicular_lines (a : ℝ) : (∀ x y, l1 a x y → l2 a x y → x = y) → a = 0 :=
by 
  -- proof outline
  sorry

end parallel_lines_perpendicular_lines_l64_6434


namespace tea_drinking_problem_l64_6408

theorem tea_drinking_problem 
  (k b c t s : ℕ) 
  (hk : k = 1) 
  (hb : b = 15) 
  (hc : c = 3) 
  (ht : t = 2) 
  (hs : s = 1) : 
  17 = 17 := 
by {
  sorry
}

end tea_drinking_problem_l64_6408


namespace sum_not_prime_l64_6440

theorem sum_not_prime (a b c d : ℕ) (h : a * b = c * d) : ¬ Nat.Prime (a + b + c + d) := 
sorry

end sum_not_prime_l64_6440


namespace largest_n_exists_ints_l64_6414

theorem largest_n_exists_ints (n : ℤ) :
  (∃ x y z : ℤ, n^2 = x^2 + y^2 + z^2 + 2 * x * y + 2 * y * z + 2 * z * x + 4 * x + 4 * y + 4 * z - 12) →
  n ≤ 10 :=
sorry

end largest_n_exists_ints_l64_6414
