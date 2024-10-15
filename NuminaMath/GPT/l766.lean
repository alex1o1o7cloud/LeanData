import Mathlib

namespace NUMINAMATH_GPT_calum_spend_per_disco_ball_l766_76689

def calum_budget := 330
def food_cost_per_box := 25
def number_of_food_boxes := 10
def number_of_disco_balls := 4

theorem calum_spend_per_disco_ball : (calum_budget - food_cost_per_box * number_of_food_boxes) / number_of_disco_balls = 20 :=
by
  sorry

end NUMINAMATH_GPT_calum_spend_per_disco_ball_l766_76689


namespace NUMINAMATH_GPT_total_number_of_animals_l766_76691

-- Definitions for the number of each type of animal
def cats : ℕ := 645
def dogs : ℕ := 567
def rabbits : ℕ := 316
def reptiles : ℕ := 120

-- The statement to prove
theorem total_number_of_animals :
  cats + dogs + rabbits + reptiles = 1648 := by
  sorry

end NUMINAMATH_GPT_total_number_of_animals_l766_76691


namespace NUMINAMATH_GPT_find_n_l766_76603

def P_X_eq_2 (n : ℕ) : Prop :=
  (3 * n) / ((n + 3) * (n + 2)) = (7 : ℚ) / 30

theorem find_n (n : ℕ) (h : P_X_eq_2 n) : n = 7 :=
by sorry

end NUMINAMATH_GPT_find_n_l766_76603


namespace NUMINAMATH_GPT_ones_digit_of_prime_in_arithmetic_sequence_is_one_l766_76618

theorem ones_digit_of_prime_in_arithmetic_sequence_is_one 
  (p q r s : ℕ) 
  (hp : Prime p) 
  (hq : Prime q) 
  (hr : Prime r) 
  (hs : Prime s) 
  (h₁ : p > 10) 
  (h₂ : q = p + 10) 
  (h₃ : r = q + 10) 
  (h₄ : s = r + 10) 
  (h₅ : s > r) 
  (h₆ : r > q) 
  (h₇ : q > p) : 
  p % 10 = 1 :=
sorry

end NUMINAMATH_GPT_ones_digit_of_prime_in_arithmetic_sequence_is_one_l766_76618


namespace NUMINAMATH_GPT_sin_alpha_third_quadrant_l766_76643

theorem sin_alpha_third_quadrant 
  (α : ℝ) 
  (hcos : Real.cos α = -3 / 5) 
  (hquad : Real.pi < α ∧ α < 3 * Real.pi / 2) : 
  Real.sin α = -4 / 5 := 
sorry

end NUMINAMATH_GPT_sin_alpha_third_quadrant_l766_76643


namespace NUMINAMATH_GPT_find_radius_k_l766_76646

/-- Mathematical conditions for the given geometry problem -/
structure problem_conditions where
  radius_F : ℝ := 15
  radius_G : ℝ := 4
  radius_H : ℝ := 3
  radius_I : ℝ := 3
  radius_J : ℝ := 1

/-- Proof problem statement defining the required theorem -/
theorem find_radius_k (conditions : problem_conditions) :
  let r := (137:ℝ) / 8
  20 * r = (342.5 : ℝ) :=
by
  sorry

end NUMINAMATH_GPT_find_radius_k_l766_76646


namespace NUMINAMATH_GPT_infinite_geometric_sum_example_l766_76654

noncomputable def infinite_geometric_sum (a₁ q : ℝ) : ℝ :=
a₁ / (1 - q)

theorem infinite_geometric_sum_example :
  infinite_geometric_sum 18 (-1/2) = 12 := by
  sorry

end NUMINAMATH_GPT_infinite_geometric_sum_example_l766_76654


namespace NUMINAMATH_GPT_total_votes_l766_76635

theorem total_votes (V : ℝ) (h1 : 0.70 * V = V - 240) (h2 : 0.30 * V = 240) : V = 800 :=
by
  sorry

end NUMINAMATH_GPT_total_votes_l766_76635


namespace NUMINAMATH_GPT_solve_quadratic_equation_l766_76682

theorem solve_quadratic_equation (x : ℝ) : 4 * (2 * x + 1) ^ 2 = 9 * (x - 3) ^ 2 ↔ x = -11 ∨ x = 1 := 
by sorry

end NUMINAMATH_GPT_solve_quadratic_equation_l766_76682


namespace NUMINAMATH_GPT_A_ge_B_l766_76612

def A (a b : ℝ) : ℝ := a^3 + 3 * a^2 * b^2 + 2 * b^2 + 3 * b
def B (a b : ℝ) : ℝ := a^3 - a^2 * b^2 + b^2 + 3 * b

theorem A_ge_B (a b : ℝ) : A a b ≥ B a b := by
  sorry

end NUMINAMATH_GPT_A_ge_B_l766_76612


namespace NUMINAMATH_GPT_find_m_values_l766_76628

theorem find_m_values {m : ℝ} :
  (∀ x : ℝ, mx^2 + (m+2) * x + (1 / 2) * m + 1 = 0 → x = 0) 
  ↔ (m = 0 ∨ m = 2 ∨ m = -2) :=
by sorry

end NUMINAMATH_GPT_find_m_values_l766_76628


namespace NUMINAMATH_GPT_minimum_candies_to_identify_coins_l766_76601

-- Set up the problem: define the relevant elements.
inductive Coin : Type
| C1 : Coin
| C2 : Coin
| C3 : Coin
| C4 : Coin
| C5 : Coin

def values : List ℕ := [1, 2, 5, 10, 20]

-- Statement of the problem in Lean 4, no means to identify which is which except through purchases and change from vending machine.
theorem minimum_candies_to_identify_coins : ∃ n : ℕ, n = 4 :=
by
  -- Skipping the proof
  sorry

end NUMINAMATH_GPT_minimum_candies_to_identify_coins_l766_76601


namespace NUMINAMATH_GPT_teacher_proctor_arrangements_l766_76667

theorem teacher_proctor_arrangements {f m : ℕ} (hf : f = 2) (hm : m = 5) :
  (∃ moving_teachers : ℕ, moving_teachers = 1 ∧ (f - moving_teachers) + m = 7 
   ∧ (f - moving_teachers).choose 2 = 21)
  ∧ 2 * 21 = 42 :=
by
    sorry

end NUMINAMATH_GPT_teacher_proctor_arrangements_l766_76667


namespace NUMINAMATH_GPT_total_people_on_hike_l766_76663

def cars : Nat := 3
def people_per_car : Nat := 4
def taxis : Nat := 6
def people_per_taxi : Nat := 6
def vans : Nat := 2
def people_per_van : Nat := 5

theorem total_people_on_hike :
  cars * people_per_car + taxis * people_per_taxi + vans * people_per_van = 58 := by
  sorry

end NUMINAMATH_GPT_total_people_on_hike_l766_76663


namespace NUMINAMATH_GPT_shaded_triangle_area_l766_76679

/--
The large equilateral triangle shown consists of 36 smaller equilateral triangles.
Each of the smaller equilateral triangles has an area of 10 cm². 
The area of the shaded triangle is K cm².
Prove that K = 110 cm².
-/
theorem shaded_triangle_area 
  (n : ℕ) (area_small : ℕ) (area_total : ℕ) (K : ℕ)
  (H1 : n = 36)
  (H2 : area_small = 10)
  (H3 : area_total = n * area_small)
  (H4 : K = 110)
: K = 110 :=
by
  -- Adding 'sorry' indicating missing proof steps.
  sorry

end NUMINAMATH_GPT_shaded_triangle_area_l766_76679


namespace NUMINAMATH_GPT_routes_from_M_to_N_l766_76622

structure Paths where
  -- Specify the paths between nodes
  C_to_N : ℕ
  D_to_N : ℕ
  A_to_C : ℕ
  A_to_D : ℕ
  B_to_N : ℕ
  B_to_A : ℕ
  B_to_C : ℕ
  M_to_B : ℕ
  M_to_A : ℕ

theorem routes_from_M_to_N (p : Paths) : 
  p.C_to_N = 1 → 
  p.D_to_N = 1 →
  p.A_to_C = 1 →
  p.A_to_D = 1 →
  p.B_to_N = 1 →
  p.B_to_A = 1 →
  p.B_to_C = 1 →
  p.M_to_B = 1 →
  p.M_to_A = 1 →
  (p.M_to_B * (p.B_to_N + (p.B_to_A * (p.A_to_C + p.A_to_D)) + p.B_to_C)) + 
  (p.M_to_A * (p.A_to_C + p.A_to_D)) = 6 
:= by
  sorry

end NUMINAMATH_GPT_routes_from_M_to_N_l766_76622


namespace NUMINAMATH_GPT_sum_of_x_and_y_l766_76633

theorem sum_of_x_and_y (x y : ℝ) (h : x^2 + y^2 = 8*x - 10*y + 5) : x + y = -1 := by
  sorry

end NUMINAMATH_GPT_sum_of_x_and_y_l766_76633


namespace NUMINAMATH_GPT_not_buy_either_l766_76611

-- Definitions
variables (n T C B : ℕ)
variables (h_n : n = 15)
variables (h_T : T = 9)
variables (h_C : C = 7)
variables (h_B : B = 3)

-- Theorem statement
theorem not_buy_either (n T C B : ℕ) (h_n : n = 15) (h_T : T = 9) (h_C : C = 7) (h_B : B = 3) :
  n - (T - B) - (C - B) - B = 2 :=
sorry

end NUMINAMATH_GPT_not_buy_either_l766_76611


namespace NUMINAMATH_GPT_arithmetic_sequence_value_l766_76664

theorem arithmetic_sequence_value (a : ℕ) (h : 2 * a = 12) : a = 6 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_value_l766_76664


namespace NUMINAMATH_GPT_simplify_and_evaluate_expression_l766_76693

theorem simplify_and_evaluate_expression (x : ℝ) (h : x = Real.sqrt 2 - 1) :
  (1 - (x + 1) / x) / ((x^2 - 1) / (x^2 - x)) = -Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_GPT_simplify_and_evaluate_expression_l766_76693


namespace NUMINAMATH_GPT_quadratic_distinct_real_roots_l766_76642

theorem quadratic_distinct_real_roots (m : ℝ) :
  (∃ a b c : ℝ, a = 1 ∧ b = m ∧ c = 9 ∧ a * c * 4 < b^2) ↔ (m < -6 ∨ m > 6) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_distinct_real_roots_l766_76642


namespace NUMINAMATH_GPT_total_trip_length_is570_l766_76670

theorem total_trip_length_is570 (v D : ℝ) (h1 : (2:ℝ) + (2/3) + (6 * (D - 2 * v) / (5 * v)) = 2.75)
(h2 : (2:ℝ) + (50 / v) + (2/3) + (6 * (D - 2 * v - 50) / (5 * v)) = 2.33) :
D = 570 :=
sorry

end NUMINAMATH_GPT_total_trip_length_is570_l766_76670


namespace NUMINAMATH_GPT_stratified_sample_sum_l766_76690

theorem stratified_sample_sum :
  let grains := 40
  let veg_oils := 10
  let animal_foods := 30
  let fruits_veggies := 20
  let total_varieties := grains + veg_oils + animal_foods + fruits_veggies
  let sample_size := 20
  let veg_oils_proportion := (veg_oils:ℚ) / total_varieties
  let fruits_veggies_proportion := (fruits_veggies:ℚ) / total_varieties
  let veg_oils_sample := sample_size * veg_oils_proportion
  let fruits_veggies_sample := sample_size * fruits_veggies_proportion
  veg_oils_sample + fruits_veggies_sample = 6 := sorry

end NUMINAMATH_GPT_stratified_sample_sum_l766_76690


namespace NUMINAMATH_GPT_sum_of_coefficients_l766_76615

-- Defining the given conditions
def vertex : ℝ × ℝ := (5, -4)
def point : ℝ × ℝ := (3, -2)

-- Defining the problem to prove the sum of the coefficients
theorem sum_of_coefficients (a b c : ℝ)
  (h_eq : ∀ y, 5 = a * ((-4) + y)^2 + c)
  (h_pt : 3 = a * ((-4) + (-2))^2 + b * (-2) + c) :
  a + b + c = -15 / 2 :=
sorry

end NUMINAMATH_GPT_sum_of_coefficients_l766_76615


namespace NUMINAMATH_GPT_circle_and_tangent_lines_l766_76651

-- Define the problem conditions
def passes_through (a b r : ℝ) : Prop :=
  (a - (-2))^2 + (b - 2)^2 = r^2 ∧
  (a - (-5))^2 + (b - 5)^2 = r^2

def lies_on_line (a b : ℝ) : Prop :=
  a + b + 3 = 0

-- Define the standard equation of the circle
def is_circle_eq (a b r : ℝ) : Prop := ∀ x y : ℝ, 
  (x - a)^2 + (y - b)^2 = r^2 ↔ (x + 5)^2 + (y - 2)^2 = 9

-- Define the tangent lines
def is_tangent_lines (x y k : ℝ) : Prop :=
  (k = (20 / 21) ∨ x = -2) → (20 * x - 21 * y + 229 = 0 ∨ x = -2)

-- The theorem statement in Lean 4
theorem circle_and_tangent_lines (a b r : ℝ) (x y k : ℝ) :
  passes_through a b r →
  lies_on_line a b →
  is_circle_eq a b r →
  is_tangent_lines x y k :=
by {
  sorry
}

end NUMINAMATH_GPT_circle_and_tangent_lines_l766_76651


namespace NUMINAMATH_GPT_tammy_avg_speed_l766_76672

theorem tammy_avg_speed 
  (v t : ℝ) 
  (h1 : t + (t - 2) = 14)
  (h2 : v * t + (v + 0.5) * (t - 2) = 52) : 
  v + 0.5 = 4 :=
by
  sorry

end NUMINAMATH_GPT_tammy_avg_speed_l766_76672


namespace NUMINAMATH_GPT_largest_rectangle_area_l766_76648

noncomputable def max_rectangle_area_with_perimeter (p : ℕ) : ℕ := sorry

theorem largest_rectangle_area (p : ℕ) (h : p = 60) : max_rectangle_area_with_perimeter p = 225 :=
sorry

end NUMINAMATH_GPT_largest_rectangle_area_l766_76648


namespace NUMINAMATH_GPT_smallest_possible_value_l766_76640

theorem smallest_possible_value 
  (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a + b + c) * (1 / (a + b) + 1 / (a + c) + 1 / (b + c)) ≥ 9 / 2 :=
sorry

end NUMINAMATH_GPT_smallest_possible_value_l766_76640


namespace NUMINAMATH_GPT_probability_xi_l766_76620

noncomputable def xi_distribution (k : ℕ) : ℚ :=
  if h : k > 0 then 1 / (2 : ℚ)^k else 0

theorem probability_xi (h : ∀ k : ℕ, k > 0 → xi_distribution k = 1 / (2 : ℚ)^k) :
  (xi_distribution 3 + xi_distribution 4) = 3 / 16 :=
by
  sorry

end NUMINAMATH_GPT_probability_xi_l766_76620


namespace NUMINAMATH_GPT_polygon_sides_l766_76677

theorem polygon_sides (x : ℕ) (h1 : 180 * (x - 2) = 3 * 360) : x = 8 := by
  sorry

end NUMINAMATH_GPT_polygon_sides_l766_76677


namespace NUMINAMATH_GPT_min_value_of_quadratic_expression_l766_76653

variable (x y z : ℝ)

theorem min_value_of_quadratic_expression 
  (h1 : 2 * x + 2 * y + z + 8 = 0) : 
  (x - 1)^2 + (y + 2)^2 + (z - 3)^2 = 9 :=
sorry

end NUMINAMATH_GPT_min_value_of_quadratic_expression_l766_76653


namespace NUMINAMATH_GPT_no_adjacent_performers_probability_l766_76681

-- A definition to model the probability of non-adjacent performers in a circle of 6 people.
def probability_no_adjacent_performers : ℚ :=
  -- Given conditions: fair coin tosses by six people, modeling permutations
  -- and specific valid configurations derived from the problem.
  9 / 32

-- Proving the final probability calculation is correct
theorem no_adjacent_performers_probability :
  probability_no_adjacent_performers = 9 / 32 :=
by
  -- Using sorry to indicate the proof needs to be filled in, acknowledging the correct answer.
  sorry

end NUMINAMATH_GPT_no_adjacent_performers_probability_l766_76681


namespace NUMINAMATH_GPT_find_z_plus_inverse_y_l766_76699

theorem find_z_plus_inverse_y
  (x y z : ℝ)
  (h1 : x * y * z = 1)
  (h2 : x + 1/z = 10)
  (h3 : y + 1/x = 5) :
  z + 1/y = 17 / 49 :=
by
  sorry

end NUMINAMATH_GPT_find_z_plus_inverse_y_l766_76699


namespace NUMINAMATH_GPT_xiaoming_comprehensive_score_l766_76617

theorem xiaoming_comprehensive_score :
  ∀ (a b c d : ℝ),
  a = 92 → b = 90 → c = 88 → d = 95 →
  (0.4 * a + 0.3 * b + 0.2 * c + 0.1 * d) = 90.9 :=
by
  intros a b c d ha hb hc hd
  simp [ha, hb, hc, hd]
  norm_num
  done

end NUMINAMATH_GPT_xiaoming_comprehensive_score_l766_76617


namespace NUMINAMATH_GPT_rectangular_field_area_l766_76631

noncomputable def length : ℝ := 1.2
noncomputable def width : ℝ := (3/4) * length

theorem rectangular_field_area : (length * width = 1.08) :=
by 
  -- The proof steps would go here
  sorry

end NUMINAMATH_GPT_rectangular_field_area_l766_76631


namespace NUMINAMATH_GPT_smallest_sum_of_integers_on_square_vertices_l766_76647

theorem smallest_sum_of_integers_on_square_vertices :
  ∃ (a b c d : ℕ), 
  0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧ 
  (a % b = 0 ∨ b % a = 0) ∧ (c % a = 0 ∨ a % c = 0) ∧ 
  (d % b = 0 ∨ b % d = 0) ∧ (d % c = 0 ∨ c % d = 0) ∧ 
  a % c ≠ 0 ∧ a % d ≠ 0 ∧ b % c ≠ 0 ∧ b % d ≠ 0 ∧ 
  (a + b + c + d = 35) := sorry

end NUMINAMATH_GPT_smallest_sum_of_integers_on_square_vertices_l766_76647


namespace NUMINAMATH_GPT_solve_inequality_l766_76626

noncomputable def inequality_statement (x : ℝ) : Prop :=
  2 / (x - 2) - 5 / (x - 3) + 5 / (x - 4) - 2 / (x - 5) < 1 / 20

theorem solve_inequality (x : ℝ) :
  (x ≠ 2 ∧ x ≠ 3 ∧ x ≠ 4 ∧ x ≠ 5) →
  (inequality_statement x ↔ (x < -3 ∨ (-2 < x ∧ x < 2) ∨ (3 < x ∧ x < 4) ∨ (5 < x ∧ x < 7) ∨ 8 < x)) :=
by sorry

end NUMINAMATH_GPT_solve_inequality_l766_76626


namespace NUMINAMATH_GPT_mary_final_books_l766_76644

def mary_initial_books := 5
def mary_first_return := 3
def mary_first_checkout := 5
def mary_second_return := 2
def mary_second_checkout := 7

theorem mary_final_books :
  (mary_initial_books - mary_first_return + mary_first_checkout - mary_second_return + mary_second_checkout) = 12 := 
by 
  sorry

end NUMINAMATH_GPT_mary_final_books_l766_76644


namespace NUMINAMATH_GPT_solution_l766_76683

-- Define the linear equations and their solutions
def system_of_equations (x y : ℕ) :=
  3 * x + y = 500 ∧ x + 2 * y = 250

-- Define the budget constraint
def budget_constraint (m : ℕ) :=
  150 * m + 50 * (25 - m) ≤ 2700

-- Define the purchasing plans and costs
def purchasing_plans (m n : ℕ) :=
  (m = 12 ∧ n = 13 ∧ 150 * m + 50 * n = 2450) ∨ 
  (m = 13 ∧ n = 12 ∧ 150 * m + 50 * n = 2550) ∨ 
  (m = 14 ∧ n = 11 ∧ 150 * m + 50 * n = 2650)

-- Define the Lean statement
theorem solution :
  (∃ x y, system_of_equations x y ∧ x = 150 ∧ y = 50) ∧
  (∃ m, budget_constraint m ∧ m ≤ 14) ∧
  (∃ m n, 12 ≤ m ∧ m ≤ 14 ∧ m + n = 25 ∧ purchasing_plans m n ∧ 150 * m + 50 * n = 2450) :=
sorry

end NUMINAMATH_GPT_solution_l766_76683


namespace NUMINAMATH_GPT_parabola_one_intersection_l766_76680

theorem parabola_one_intersection (k : ℝ) :
  (∀ x : ℝ, x^2 - x + k = 0 → x = 0) → k = 1 / 4 :=
sorry

end NUMINAMATH_GPT_parabola_one_intersection_l766_76680


namespace NUMINAMATH_GPT_max_value_f_on_interval_l766_76606

noncomputable def f (x : ℝ) : ℝ := Real.exp x - x

theorem max_value_f_on_interval : 
  ∃ x ∈ Set.Icc (-1 : ℝ) 1, (∀ y ∈ Set.Icc (-1 : ℝ) 1, f y ≤ f x) ∧ f x = Real.exp 1 - 1 :=
sorry

end NUMINAMATH_GPT_max_value_f_on_interval_l766_76606


namespace NUMINAMATH_GPT_students_not_take_test_l766_76676

theorem students_not_take_test
  (total_students : ℕ)
  (q1_correct : ℕ)
  (q2_correct : ℕ)
  (both_correct : ℕ)
  (h_total : total_students = 29)
  (h_q1 : q1_correct = 19)
  (h_q2 : q2_correct = 24)
  (h_both : both_correct = 19)
  : (total_students - (q1_correct + q2_correct - both_correct) = 5) :=
by
  sorry

end NUMINAMATH_GPT_students_not_take_test_l766_76676


namespace NUMINAMATH_GPT_divisible_by_91_l766_76632

def a (n : ℕ) : ℕ :=
  match n with
  | 0 => 202020
  | _ => -- Define the sequence here, ensuring it constructs the number properly with inserted '2's
    sorry -- this might be a more complex function to define

theorem divisible_by_91 (n : ℕ) : 91 ∣ a n :=
  sorry

end NUMINAMATH_GPT_divisible_by_91_l766_76632


namespace NUMINAMATH_GPT_lighter_shopping_bag_weight_l766_76685

theorem lighter_shopping_bag_weight :
  ∀ (G : ℕ), (G + 7 = 10) → (G = 3) := by
  intros G h
  sorry

end NUMINAMATH_GPT_lighter_shopping_bag_weight_l766_76685


namespace NUMINAMATH_GPT_problem_set_equiv_l766_76616

def positive_nats (x : ℕ) : Prop := x > 0

def problem_set : Set ℕ := {x | positive_nats x ∧ x - 3 < 2}

theorem problem_set_equiv : problem_set = {1, 2, 3, 4} :=
by 
  sorry

end NUMINAMATH_GPT_problem_set_equiv_l766_76616


namespace NUMINAMATH_GPT_seconds_in_8_point_5_minutes_l766_76605

def minutesToSeconds (minutes : ℝ) : ℝ := minutes * 60

theorem seconds_in_8_point_5_minutes : minutesToSeconds 8.5 = 510 := 
by
  sorry

end NUMINAMATH_GPT_seconds_in_8_point_5_minutes_l766_76605


namespace NUMINAMATH_GPT_tim_grew_cantaloupes_l766_76641

theorem tim_grew_cantaloupes (fred_cantaloupes : ℕ) (total_cantaloupes : ℕ) (h1 : fred_cantaloupes = 38) (h2 : total_cantaloupes = 82) :
  ∃ tim_cantaloupes : ℕ, tim_cantaloupes = total_cantaloupes - fred_cantaloupes ∧ tim_cantaloupes = 44 :=
by
  sorry

end NUMINAMATH_GPT_tim_grew_cantaloupes_l766_76641


namespace NUMINAMATH_GPT_arithmetic_mean_of_fractions_l766_76660

theorem arithmetic_mean_of_fractions :
  let a := (3 : ℚ) / 8
  let b := (5 : ℚ) / 9
  (a + b) / 2 = 67 / 144 := 
by 
  sorry

end NUMINAMATH_GPT_arithmetic_mean_of_fractions_l766_76660


namespace NUMINAMATH_GPT_geometric_sequence_cannot_determine_a3_l766_76669

/--
Suppose we have a geometric sequence {a_n} such that 
the product of the first five terms a_1 * a_2 * a_3 * a_4 * a_5 = 32.
We aim to show that the value of a_3 cannot be determined with the given information.
-/
theorem geometric_sequence_cannot_determine_a3 (a : ℕ → ℝ) (r : ℝ) (h : a 0 * a 1 * a 2 * a 3 * a 4 = 32) : 
  ¬ ∃ x : ℝ, a 2 = x :=
sorry

end NUMINAMATH_GPT_geometric_sequence_cannot_determine_a3_l766_76669


namespace NUMINAMATH_GPT_remainder_sum_div_11_l766_76613

theorem remainder_sum_div_11 :
  ((100001 + 100002 + 100003 + 100004 + 100005 + 100006 + 100007 + 100008 + 100009 + 100010) % 11) = 10 :=
by
  sorry

end NUMINAMATH_GPT_remainder_sum_div_11_l766_76613


namespace NUMINAMATH_GPT_max_value_x_plus_2y_l766_76674

theorem max_value_x_plus_2y (x y : ℝ) (h : |x| + |y| ≤ 1) : x + 2 * y ≤ 2 :=
sorry

end NUMINAMATH_GPT_max_value_x_plus_2y_l766_76674


namespace NUMINAMATH_GPT_find_total_students_l766_76627

variables (x X : ℕ)
variables (x_percent_students : ℕ) (total_students : ℕ)
variables (boys_fraction : ℝ)

-- Provided Conditions
axiom a1 : x_percent_students = 120
axiom a2 : boys_fraction = 0.30
axiom a3 : total_students = X

-- The theorem we need to prove
theorem find_total_students (a1 : 120 = x_percent_students) 
                            (a2 : boys_fraction = 0.30) 
                            (a3 : total_students = X) : 
  120 = (x / 100) * (boys_fraction * total_students) :=
sorry

end NUMINAMATH_GPT_find_total_students_l766_76627


namespace NUMINAMATH_GPT_convert_base_10_to_base_5_l766_76649

theorem convert_base_10_to_base_5 :
  (256 : ℕ) = 2 * 5^3 + 0 * 5^2 + 1 * 5^1 + 1 * 5^0 :=
by
  sorry

end NUMINAMATH_GPT_convert_base_10_to_base_5_l766_76649


namespace NUMINAMATH_GPT_measure_angle_BCQ_l766_76652

/-- Given:
  - Segment AB has a length of 12 units.
  - Segment AC is 9 units long.
  - Segment AC : CB = 3 : 1.
  - A semi-circle is constructed with diameter AB.
  - Another smaller semi-circle is constructed with diameter CB.
  - A line segment CQ divides the combined area of the two semi-circles into two equal areas.

  Prove: The degree measure of angle BCQ is 11.25°.
-/ 
theorem measure_angle_BCQ (AB AC CB : ℝ) (hAB : AB = 12) (hAC : AC = 9) (hRatio : AC / CB = 3) :
  ∃ θ : ℝ, θ = 11.25 :=
by
  sorry

end NUMINAMATH_GPT_measure_angle_BCQ_l766_76652


namespace NUMINAMATH_GPT_only_polynomial_is_identity_l766_76688

-- Define the number composed only of digits 1
def Ones (k : ℕ) : ℕ := (10^k - 1) / 9

theorem only_polynomial_is_identity (P : ℕ → ℕ) :
  (∀ k : ℕ, P (Ones k) = Ones k) → (∀ x : ℕ, P x = x) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_only_polynomial_is_identity_l766_76688


namespace NUMINAMATH_GPT_p_minus_q_l766_76638

-- Define the given equation as a predicate.
def eqn (x : ℝ) : Prop := (3*x - 9) / (x*x + 3*x - 18) = x + 3

-- Define the values p and q as distinct solutions.
def p_and_q (p q : ℝ) : Prop := eqn p ∧ eqn q ∧ p ≠ q ∧ p > q

theorem p_minus_q {p q : ℝ} (h : p_and_q p q) : p - q = 2 := sorry

end NUMINAMATH_GPT_p_minus_q_l766_76638


namespace NUMINAMATH_GPT_total_plant_count_l766_76684

-- Definitions for conditions.
def total_rows : ℕ := 96
def columns_per_row : ℕ := 24
def divided_rows : ℕ := total_rows / 3
def undivided_rows : ℕ := total_rows - divided_rows
def beans_in_undivided_row : ℕ := columns_per_row
def corn_in_divided_row : ℕ := columns_per_row / 2
def tomatoes_in_divided_row : ℕ := columns_per_row / 2

-- Total number of plants calculation.
def total_bean_plants : ℕ := undivided_rows * beans_in_undivided_row
def total_corn_plants : ℕ := divided_rows * corn_in_divided_row
def total_tomato_plants : ℕ := divided_rows * tomatoes_in_divided_row

def total_plants : ℕ := total_bean_plants + total_corn_plants + total_tomato_plants

-- Proof statement.
theorem total_plant_count : total_plants = 2304 :=
by
  sorry

end NUMINAMATH_GPT_total_plant_count_l766_76684


namespace NUMINAMATH_GPT_cost_of_pencil_and_pens_l766_76658

variable (p q : ℝ)

def equation1 := 3 * p + 4 * q = 3.20
def equation2 := 2 * p + 3 * q = 2.50

theorem cost_of_pencil_and_pens (h1 : equation1 p q) (h2 : equation2 p q) : p + 2 * q = 1.80 := 
by 
  sorry

end NUMINAMATH_GPT_cost_of_pencil_and_pens_l766_76658


namespace NUMINAMATH_GPT_quadratic_roots_bounds_l766_76673

theorem quadratic_roots_bounds (a b c : ℤ) (p1 p2 : ℝ) (h_a_pos : a > 0) 
  (h_int_coeff : ∀ x : ℤ, x = a ∨ x = b ∨ x = c) 
  (h_distinct_roots : p1 ≠ p2) 
  (h_roots : a * p1^2 + b * p1 + c = 0 ∧ a * p2^2 + b * p2 + c = 0) 
  (h_roots_bounds : 0 < p1 ∧ p1 < 1 ∧ 0 < p2 ∧ p2 < 1) : 
     a ≥ 5 := 
sorry

end NUMINAMATH_GPT_quadratic_roots_bounds_l766_76673


namespace NUMINAMATH_GPT_monotonic_range_of_a_l766_76639

theorem monotonic_range_of_a (a : ℝ) :
  (a ≥ 9 ∨ a ≤ 3) → 
  ∀ x y : ℝ, (1 ≤ x ∧ x ≤ 4) → (1 ≤ y ∧ y ≤ 4) → x ≤ y → 
  (x^2 + (1-a)*x + 3) ≤ (y^2 + (1-a)*y + 3) :=
by
  intro ha x y hx hy hxy
  sorry

end NUMINAMATH_GPT_monotonic_range_of_a_l766_76639


namespace NUMINAMATH_GPT_solve_for_x_l766_76609

theorem solve_for_x (x y z : ℝ) (h1 : x * y = 8 - 3 * x - 2 * y) 
                                  (h2 : y * z = 8 - 2 * y - 3 * z) 
                                  (h3 : x * z = 35 - 5 * x - 3 * z) : 
  x = 8 :=
sorry

end NUMINAMATH_GPT_solve_for_x_l766_76609


namespace NUMINAMATH_GPT_proof_problem_l766_76662

-- definitions of the given conditions
variable (a b c : ℝ)
variables (h₁ : 6 < a) (h₂ : a < 10) 
variable (h₃ : (a / 2) ≤ b ∧ b ≤ 2 * a) 
variable (h₄ : c = a + b)

-- statement to be proved
theorem proof_problem (h₁ : 6 < a) (h₂ : a < 10) (h₃ : (a / 2) ≤ b ∧ b ≤ 2 * a) (h₄ : c = a + b) : 9 < c ∧ c < 30 := 
sorry

end NUMINAMATH_GPT_proof_problem_l766_76662


namespace NUMINAMATH_GPT_first_character_more_lines_than_second_l766_76661

theorem first_character_more_lines_than_second :
  let x := 2
  let second_character_lines := 3 * x + 6
  20 - second_character_lines = 8 := by
  sorry

end NUMINAMATH_GPT_first_character_more_lines_than_second_l766_76661


namespace NUMINAMATH_GPT_ants_movement_impossible_l766_76636

theorem ants_movement_impossible (initial_positions final_positions : Fin 3 → ℝ × ℝ) :
  initial_positions 0 = (0,0) ∧ initial_positions 1 = (0,1) ∧ initial_positions 2 = (1,0) →
  final_positions 0 = (-1,0) ∧ final_positions 1 = (0,1) ∧ final_positions 2 = (1,0) →
  (∀ t : ℕ, ∃ m : Fin 3, 
    ∀ i : Fin 3, (i ≠ m → ∃ k l : ℝ, 
      (initial_positions i).2 - l * (initial_positions i).1 = 0 ∧ 
      ∀ (p : ℕ → ℝ × ℝ), p 0 = initial_positions i ∧ p t = final_positions i → 
      (p 0).1 + k * (p 0).2 = 0)) →
  false :=
by 
  sorry

end NUMINAMATH_GPT_ants_movement_impossible_l766_76636


namespace NUMINAMATH_GPT_minimum_value_of_f_l766_76619

noncomputable def f (x : ℝ) : ℝ := x^2 / (x - 5)

theorem minimum_value_of_f : ∃ (x : ℝ), x > 5 ∧ f x = 20 :=
by
  use 10
  sorry

end NUMINAMATH_GPT_minimum_value_of_f_l766_76619


namespace NUMINAMATH_GPT_angle_2016_in_third_quadrant_l766_76655

def quadrant (θ : ℤ) : ℤ :=
  let angle := θ % 360
  if 0 ≤ angle ∧ angle < 90 then 1
  else if 90 ≤ angle ∧ angle < 180 then 2
  else if 180 ≤ angle ∧ angle < 270 then 3
  else 4

theorem angle_2016_in_third_quadrant : 
  quadrant 2016 = 3 := 
by
  sorry

end NUMINAMATH_GPT_angle_2016_in_third_quadrant_l766_76655


namespace NUMINAMATH_GPT_quadratic_function_l766_76686

theorem quadratic_function :
  ∃ a : ℝ, ∃ f : ℝ → ℝ, (∀ x : ℝ, f x = a * (x - 1) * (x - 5)) ∧ f 3 = 10 ∧ 
  f = fun x => -2.5 * x^2 + 15 * x - 12.5 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_function_l766_76686


namespace NUMINAMATH_GPT_candy_left_l766_76600

theorem candy_left (total_candy : ℕ) (eaten_per_person : ℕ) (number_of_people : ℕ)
  (h_total_candy : total_candy = 68)
  (h_eaten_per_person : eaten_per_person = 4)
  (h_number_of_people : number_of_people = 2) :
  total_candy - (eaten_per_person * number_of_people) = 60 :=
by
  sorry

end NUMINAMATH_GPT_candy_left_l766_76600


namespace NUMINAMATH_GPT_charles_nickels_l766_76623

theorem charles_nickels :
  ∀ (num_pennies num_cents penny_value nickel_value n : ℕ),
  num_pennies = 6 →
  num_cents = 21 →
  penny_value = 1 →
  nickel_value = 5 →
  (num_cents - num_pennies * penny_value) / nickel_value = n →
  n = 3 :=
by
  intros num_pennies num_cents penny_value nickel_value n hnum_pennies hnum_cents hpenny_value hnickel_value hn
  sorry

end NUMINAMATH_GPT_charles_nickels_l766_76623


namespace NUMINAMATH_GPT_evaluate_f_at_t_plus_one_l766_76668

-- Define the function f
def f (x : ℝ) : ℝ := 3 * x - 1

-- Define the proposition to be proved
theorem evaluate_f_at_t_plus_one (t : ℝ) : f (t + 1) = 3 * t + 2 := by
  sorry

end NUMINAMATH_GPT_evaluate_f_at_t_plus_one_l766_76668


namespace NUMINAMATH_GPT_stuffed_animal_ratio_l766_76694

theorem stuffed_animal_ratio
  (K : ℕ)
  (h1 : 34 + K + (K + 5) = 175) :
  K / 34 = 2 :=
by sorry

end NUMINAMATH_GPT_stuffed_animal_ratio_l766_76694


namespace NUMINAMATH_GPT_water_left_after_operations_l766_76696

theorem water_left_after_operations :
  let initial_water := (3 : ℚ)
  let water_used := (4 / 3 : ℚ)
  let extra_water := (1 / 2 : ℚ)
  initial_water - water_used + extra_water = (13 / 6 : ℚ) := 
by
  -- Skips the proof, as the focus is on the problem statement
  sorry

end NUMINAMATH_GPT_water_left_after_operations_l766_76696


namespace NUMINAMATH_GPT_donut_selection_l766_76624

-- Lean statement for the proof problem
theorem donut_selection (n k : ℕ) (h1 : n = 5) (h2 : k = 4) : (n + k - 1).choose (k - 1) = 56 :=
by
  rw [h1, h2]
  sorry

end NUMINAMATH_GPT_donut_selection_l766_76624


namespace NUMINAMATH_GPT_smallest_value_l766_76697

theorem smallest_value (a b c : ℕ) (pos_a : a > 0) (pos_b : b > 0) (pos_c : c > 0)
    (h1 : a = 2 * b) (h2 : b = 2 * c) (h3 : 4 * c = a) :
    (Int.floor ((a + b : ℚ) / c) + Int.floor ((b + c : ℚ) / a) + Int.floor ((c + a : ℚ) / b)) = 8 := 
sorry

end NUMINAMATH_GPT_smallest_value_l766_76697


namespace NUMINAMATH_GPT_value_of_m_if_f_is_power_function_l766_76657

theorem value_of_m_if_f_is_power_function (m : ℤ) :
  (2 * m + 3 = 1) → m = -1 :=
by
  sorry

end NUMINAMATH_GPT_value_of_m_if_f_is_power_function_l766_76657


namespace NUMINAMATH_GPT_ginger_total_water_l766_76698

def hours_worked : Nat := 8
def cups_per_bottle : Nat := 2
def bottles_drank_per_hour : Nat := 1
def bottles_for_plants : Nat := 5

theorem ginger_total_water : 
  (hours_worked * cups_per_bottle * bottles_drank_per_hour) + (bottles_for_plants * cups_per_bottle) = 26 :=
by
  sorry

end NUMINAMATH_GPT_ginger_total_water_l766_76698


namespace NUMINAMATH_GPT_find_difference_of_squares_l766_76665

variable (x y : ℝ)
variable (h1 : (x + y) ^ 2 = 81)
variable (h2 : x * y = 18)

theorem find_difference_of_squares : (x - y) ^ 2 = 9 := by
  sorry

end NUMINAMATH_GPT_find_difference_of_squares_l766_76665


namespace NUMINAMATH_GPT_lines_intersect_at_3_6_l766_76607

theorem lines_intersect_at_3_6 (c d : ℝ) 
  (h1 : 3 = 2 * 6 + c) 
  (h2 : 6 = 2 * 3 + d) : 
  c + d = -9 := by 
  sorry

end NUMINAMATH_GPT_lines_intersect_at_3_6_l766_76607


namespace NUMINAMATH_GPT_part1_part2_l766_76666

namespace Problem

def A : Set ℝ := {x | -2 ≤ x ∧ x ≤ 5}
def B (m : ℝ) : Set ℝ := {x | m + 1 ≤ x ∧ x ≤ 2 * m - 1}

theorem part1 (m : ℝ) : (B m ⊆ A) → (m ≤ 3) :=
by
  intro h
  sorry

theorem part2 (m : ℝ) : (A ∩ B m = ∅) → (m < 2 ∨ 4 < m) :=
by
  intro h
  sorry

end Problem

end NUMINAMATH_GPT_part1_part2_l766_76666


namespace NUMINAMATH_GPT_water_fee_relationship_xiao_qiangs_water_usage_l766_76614

variable (x y : ℝ)
variable (H1 : x > 10)
variable (H2 : y = 3 * x - 8)

theorem water_fee_relationship : y = 3 * x - 8 := 
  by 
    exact H2

theorem xiao_qiangs_water_usage : y = 67 → x = 25 :=
  by
    intro H
    have H_eq : 67 = 3 * x - 8 := by 
      rw [←H2, H]
    linarith

end NUMINAMATH_GPT_water_fee_relationship_xiao_qiangs_water_usage_l766_76614


namespace NUMINAMATH_GPT_P_at_10_l766_76630

-- Define the main properties of the polynomial
variable (P : ℤ → ℤ)
axiom quadratic (a b c : ℤ) : (∀ n : ℤ, P n = a * n^2 + b * n + c) 

-- Conditions for the polynomial
axiom int_coefficients : ∃ (a b c : ℤ), ∀ n : ℤ, P n = a * n^2 + b * n + c
axiom relatively_prime (n : ℤ) (hn : 0 < n) : Int.gcd (P n) n = 1 ∧ Int.gcd (P (P n)) n = 1
axiom P_at_3 : P 3 = 89

-- The main theorem to prove
theorem P_at_10 : P 10 = 859 := by sorry

end NUMINAMATH_GPT_P_at_10_l766_76630


namespace NUMINAMATH_GPT_baseball_card_decrease_l766_76621

theorem baseball_card_decrease (V : ℝ) (hV : V > 0) (x : ℝ) :
  (1 - x / 100) * (1 - 0.30) = 1 - 0.44 -> x = 20 :=
by {
  -- proof omitted 
  sorry
}

end NUMINAMATH_GPT_baseball_card_decrease_l766_76621


namespace NUMINAMATH_GPT_age_of_b_l766_76625

variable (a b c d : ℕ)
variable (h1 : a = b + 2)
variable (h2 : b = 2 * c)
variable (h3 : d = b / 2)
variable (h4 : a + b + c + d = 44)

theorem age_of_b : b = 14 :=
by 
  sorry

end NUMINAMATH_GPT_age_of_b_l766_76625


namespace NUMINAMATH_GPT_find_explicit_formula_l766_76650

variable (f : ℝ → ℝ)

theorem find_explicit_formula 
  (h : ∀ x : ℝ, f (x - 1) = 2 * x^2 - 8 * x + 11) :
  ∀ x : ℝ, f x = 2 * x^2 - 4 * x + 5 :=
by
  sorry

end NUMINAMATH_GPT_find_explicit_formula_l766_76650


namespace NUMINAMATH_GPT_find_pencils_l766_76675

theorem find_pencils :
  ∃ (n : ℕ), 10 ≤ n ∧ n < 100 ∧ (6 ∣ n) ∧ (9 ∣ n) ∧ n % 7 = 1 ∧ n = 36 :=
by
  sorry

end NUMINAMATH_GPT_find_pencils_l766_76675


namespace NUMINAMATH_GPT_problem1_problem2_l766_76629

theorem problem1 (x : ℝ) : (4 * x ^ 2 + 12 * x - 7 ≤ 0) ∧ (a = 0) ∧ (x < -3 ∨ x > 3) → (-7/2 ≤ x ∧ x < -3) := by
  sorry

theorem problem2 (a : ℝ) : (∀ x : ℝ, 4 * x ^ 2 + 12 * x - 7 ≤ 0 → a - 3 ≤ x ∧ x ≤ a + 3) → (-5/2 ≤ a ∧ a ≤ -1/2) := by
  sorry

end NUMINAMATH_GPT_problem1_problem2_l766_76629


namespace NUMINAMATH_GPT_birthday_guests_l766_76656

theorem birthday_guests (total_guests : ℕ) (women men children guests_left men_left children_left : ℕ)
  (h_total : total_guests = 60)
  (h_women : women = total_guests / 2)
  (h_men : men = 15)
  (h_children : children = total_guests - (women + men))
  (h_men_left : men_left = men / 3)
  (h_children_left : children_left = 5)
  (h_guests_left : guests_left = men_left + children_left) :
  (total_guests - guests_left) = 50 :=
by sorry

end NUMINAMATH_GPT_birthday_guests_l766_76656


namespace NUMINAMATH_GPT_marissa_sunflower_height_l766_76678

def height_sister_in_inches : ℚ := 4 * 12 + 3
def height_difference_in_inches : ℚ := 21
def inches_to_cm (inches : ℚ) : ℚ := inches * 2.54
def cm_to_m (cm : ℚ) : ℚ := cm / 100

theorem marissa_sunflower_height :
  cm_to_m (inches_to_cm (height_sister_in_inches + height_difference_in_inches)) = 1.8288 :=
by sorry

end NUMINAMATH_GPT_marissa_sunflower_height_l766_76678


namespace NUMINAMATH_GPT_ratio_of_population_l766_76645

theorem ratio_of_population (Z : ℕ) :
  let Y := 2 * Z
  let X := 3 * Y
  let W := X + Y
  X / (Z + W) = 2 / 3 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_population_l766_76645


namespace NUMINAMATH_GPT_max_perimeter_of_polygons_l766_76671

theorem max_perimeter_of_polygons 
  (t s : ℕ) 
  (hts : t + s = 7) 
  (hsum_angles : 60 * t + 90 * s = 360) 
  (max_squares : s ≤ 4) 
  (side_length : ℕ := 2) 
  (tri_perimeter : ℕ := 3 * side_length) 
  (square_perimeter : ℕ := 4 * side_length) :
  2 * (t * tri_perimeter + s * square_perimeter) = 68 := 
sorry

end NUMINAMATH_GPT_max_perimeter_of_polygons_l766_76671


namespace NUMINAMATH_GPT_circle_radius_formula_correct_l766_76604

noncomputable def touch_circles_radius (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  let Δ := (s - a) * (s - b) * (s - c)
  let numerator := c * Real.sqrt ((s - a) * (s - b) * (s - c))
  let denominator := c * Real.sqrt s + 2 * Real.sqrt ((s - a) * (s - b) * (s - c))
  numerator / denominator

theorem circle_radius_formula_correct (a b c : ℝ) : 
  let s := (a + b + c) / 2
  let Δ := (s - a) * (s - b) * (s - c)
  ∀ (r : ℝ), (r = touch_circles_radius a b c) :=
sorry

end NUMINAMATH_GPT_circle_radius_formula_correct_l766_76604


namespace NUMINAMATH_GPT_emily_collected_8484_eggs_l766_76634

def number_of_baskets : ℕ := 303
def eggs_per_basket : ℕ := 28
def total_eggs : ℕ := number_of_baskets * eggs_per_basket

theorem emily_collected_8484_eggs : total_eggs = 8484 :=
by
  sorry

end NUMINAMATH_GPT_emily_collected_8484_eggs_l766_76634


namespace NUMINAMATH_GPT_highest_digit_a_divisible_by_eight_l766_76659

theorem highest_digit_a_divisible_by_eight :
  ∃ a : ℕ, a ≤ 9 ∧ 8 ∣ (100 * a + 16) ∧ ∀ b : ℕ, b > a → b ≤ 9 → ¬ (8 ∣ (100 * b + 16)) := by
  sorry

end NUMINAMATH_GPT_highest_digit_a_divisible_by_eight_l766_76659


namespace NUMINAMATH_GPT_range_of_a_l766_76692

theorem range_of_a (a : ℝ) (f : ℝ → ℝ) :
  (∀ x : ℝ, f x = x^3 - a) → (∀ x : ℝ, f 0 ≤ 0) → (0 ≤ a) :=
by
  intro h1 h2
  suffices h : -a ≤ 0 by
    simpa using h
  have : f 0 = -a
  simp [h1]
  sorry -- Proof steps are omitted

end NUMINAMATH_GPT_range_of_a_l766_76692


namespace NUMINAMATH_GPT_jason_initial_quarters_l766_76602

theorem jason_initial_quarters (q_d q_n q_i : ℕ) (h1 : q_d = 25) (h2 : q_n = 74) :
  q_i = q_n - q_d → q_i = 49 :=
by
  sorry

end NUMINAMATH_GPT_jason_initial_quarters_l766_76602


namespace NUMINAMATH_GPT_polynomial_problem_l766_76687

noncomputable def F (x : ℝ) : ℝ := sorry

theorem polynomial_problem
  (F : ℝ → ℝ)
  (h1 : F 4 = 22)
  (h2 : ∀ x : ℝ, (F (2 * x) / F (x + 2) = 4 - (16 * x + 8) / (x^2 + x + 1))) :
  F 8 = 1078 / 9 := sorry

end NUMINAMATH_GPT_polynomial_problem_l766_76687


namespace NUMINAMATH_GPT_smallest_number_of_students_l766_76637

-- Define the structure of the problem
def unique_row_configurations (n : ℕ) : Prop :=
  (∀ k : ℕ, k ∣ n → k < 10) → ∃ divs : Finset ℕ, divs.card = 9 ∧ ∀ d ∈ divs, d ∣ n ∧ (∀ d' ∈ divs, d ≠ d') 

-- The main statement to be proven in Lean 4
theorem smallest_number_of_students : ∃ n : ℕ, unique_row_configurations n ∧ n = 36 :=
by
  sorry

end NUMINAMATH_GPT_smallest_number_of_students_l766_76637


namespace NUMINAMATH_GPT_problem_solution_l766_76608

theorem problem_solution (x : ℝ) (h : ∃ (A B : Set ℝ), A = {0, 1, 2, 4, 5} ∧ B = {x-2, x, x+2} ∧ A ∩ B = {0, 2}) : x = 0 :=
sorry

end NUMINAMATH_GPT_problem_solution_l766_76608


namespace NUMINAMATH_GPT_laura_mowing_time_correct_l766_76610

noncomputable def laura_mowing_time : ℝ := 
  let combined_time := 1.71428571429
  let sammy_time := 3
  let combined_rate := 1 / combined_time
  let sammy_rate := 1 / sammy_time
  let laura_rate := combined_rate - sammy_rate
  1 / laura_rate

theorem laura_mowing_time_correct : laura_mowing_time = 4.2 := 
  by
    sorry

end NUMINAMATH_GPT_laura_mowing_time_correct_l766_76610


namespace NUMINAMATH_GPT_b_gives_c_start_l766_76695

variable (Va Vb Vc : ℝ)

-- Conditions given in the problem
def condition1 : Prop := Va / Vb = 1000 / 930
def condition2 : Prop := Va / Vc = 1000 / 800
def race_distance : ℝ := 1000

-- Proposition to prove
theorem b_gives_c_start (h1 : condition1 Va Vb) (h2 : condition2 Va Vc) :
  ∃ x : ℝ, (1000 - x) / 1000 = (930 / 800) :=
sorry

end NUMINAMATH_GPT_b_gives_c_start_l766_76695
