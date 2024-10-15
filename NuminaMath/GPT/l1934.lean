import Mathlib

namespace NUMINAMATH_GPT_handshake_count_l1934_193429

-- Definitions based on conditions
def groupA_size : ℕ := 25
def groupB_size : ℕ := 15

-- Total number of handshakes is calculated as product of their sizes
def total_handshakes : ℕ := groupA_size * groupB_size

-- The theorem we need to prove
theorem handshake_count : total_handshakes = 375 :=
by
  -- skipped proof
  sorry

end NUMINAMATH_GPT_handshake_count_l1934_193429


namespace NUMINAMATH_GPT_sequence_is_geometric_not_arithmetic_l1934_193498

def is_arithmetic_sequence (a b c : ℕ) : Prop :=
  b - a = c - b

def is_geometric_sequence (a b c : ℕ) : Prop :=
  b / a = c / b

theorem sequence_is_geometric_not_arithmetic :
  ∀ (a₁ a₂ an : ℕ), a₁ = 3 ∧ a₂ = 9 ∧ an = 729 →
    ¬ is_arithmetic_sequence a₁ a₂ an ∧ is_geometric_sequence a₁ a₂ an :=
by
  intros a₁ a₂ an h
  sorry

end NUMINAMATH_GPT_sequence_is_geometric_not_arithmetic_l1934_193498


namespace NUMINAMATH_GPT_number_of_smaller_cubes_l1934_193495

theorem number_of_smaller_cubes (edge : ℕ) (N : ℕ) (h_edge : edge = 5)
  (h_divisors : ∃ (a b c : ℕ), a + b + c = N ∧ a * 1^3 + b * 2^3 + c * 3^3 = edge^3 ∧ a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0) :
  N = 22 :=
by
  sorry

end NUMINAMATH_GPT_number_of_smaller_cubes_l1934_193495


namespace NUMINAMATH_GPT_least_n_condition_l1934_193483

-- Define the conditions and the question in Lean 4
def jackson_position (n : ℕ) : ℕ := sorry  -- Defining the position of Jackson after n steps

def expected_value (n : ℕ) : ℝ := sorry  -- Defining the expected value E_n

theorem least_n_condition : ∃ n : ℕ, (1 / expected_value n > 2017) ∧ (∀ m < n, 1 / expected_value m ≤ 2017) ∧ n = 13446 :=
by {
  -- Jackson starts at position 1
  -- The conditions described in the problem will be formulated here
  -- We need to show that the least n such that 1 / E_n > 2017 is 13446
  sorry
}

end NUMINAMATH_GPT_least_n_condition_l1934_193483


namespace NUMINAMATH_GPT_solve_for_y_l1934_193466

theorem solve_for_y (x y : ℤ) (h1 : x - y = 16) (h2 : x + y = 10) : y = -3 :=
sorry

end NUMINAMATH_GPT_solve_for_y_l1934_193466


namespace NUMINAMATH_GPT_solution_set_M_inequality_ab_l1934_193455

def f (x : ℝ) : ℝ := |x - 1| + |x + 3|

theorem solution_set_M :
  {x | -3 ≤ x ∧ x ≤ 1} = { x : ℝ | f x ≤ 4 } :=
sorry

theorem inequality_ab
  (a b : ℝ) (h1 : -3 ≤ a ∧ a ≤ 1) (h2 : -3 ≤ b ∧ b ≤ 1) :
  (a^2 + 2 * a - 3) * (b^2 + 2 * b - 3) ≥ 0 :=
sorry

end NUMINAMATH_GPT_solution_set_M_inequality_ab_l1934_193455


namespace NUMINAMATH_GPT_largest_club_size_is_four_l1934_193474

variable {Player : Type} -- Assume Player is a type

-- Definition of the lesson-taking relation
variable (takes_lessons_from : Player → Player → Prop)

-- Club conditions
def club_conditions (A B C : Player) : Prop :=
  (takes_lessons_from A B ∧ ¬takes_lessons_from B C ∧ ¬takes_lessons_from C A) ∨ 
  (¬takes_lessons_from A B ∧ takes_lessons_from B C ∧ ¬takes_lessons_from C A) ∨ 
  (¬takes_lessons_from A B ∧ ¬takes_lessons_from B C ∧ takes_lessons_from C A)

theorem largest_club_size_is_four :
  ∀ (club : Finset Player),
  (∀ (A B C : Player), A ≠ B → B ≠ C → C ≠ A → A ∈ club → B ∈ club → C ∈ club → club_conditions takes_lessons_from A B C) →
  club.card ≤ 4 :=
sorry

end NUMINAMATH_GPT_largest_club_size_is_four_l1934_193474


namespace NUMINAMATH_GPT_servings_per_day_l1934_193473

-- Conditions
def week_servings := 21
def days_per_week := 7

-- Question and Answer
theorem servings_per_day : week_servings / days_per_week = 3 := 
by
  sorry

end NUMINAMATH_GPT_servings_per_day_l1934_193473


namespace NUMINAMATH_GPT_final_price_is_correct_l1934_193415

-- Define the conditions as constants
def price_smartphone : ℝ := 300
def price_pc : ℝ := price_smartphone + 500
def price_tablet : ℝ := price_smartphone + price_pc
def total_price : ℝ := price_smartphone + price_pc + price_tablet
def discount : ℝ := 0.10 * total_price
def price_after_discount : ℝ := total_price - discount
def sales_tax : ℝ := 0.05 * price_after_discount
def final_price : ℝ := price_after_discount + sales_tax

-- Theorem statement asserting the final price value
theorem final_price_is_correct : final_price = 2079 := by sorry

end NUMINAMATH_GPT_final_price_is_correct_l1934_193415


namespace NUMINAMATH_GPT_rectangle_ratio_l1934_193442

theorem rectangle_ratio (s y x : ℝ) (hs : s > 0) (hy : y > 0) (hx : x > 0)
  (h1 : s + 2 * y = 3 * s)
  (h2 : x + y = 3 * s)
  (h3 : y = s)
  (h4 : x = 2 * s) :
  x / y = 2 := by
  sorry

end NUMINAMATH_GPT_rectangle_ratio_l1934_193442


namespace NUMINAMATH_GPT_min_value_y_l1934_193496

theorem min_value_y (x : ℝ) (hx : x > 3) : 
  ∃ y, (∀ x > 3, y = min_value) ∧ min_value = 5 :=
by 
  sorry

end NUMINAMATH_GPT_min_value_y_l1934_193496


namespace NUMINAMATH_GPT_smallest_nat_div_7_and_11_l1934_193420

theorem smallest_nat_div_7_and_11 (n : ℕ) (h1 : n > 1) (h2 : n % 7 = 1) (h3 : n % 11 = 1) : n = 78 :=
by
  sorry

end NUMINAMATH_GPT_smallest_nat_div_7_and_11_l1934_193420


namespace NUMINAMATH_GPT_inequality_sum_of_reciprocals_l1934_193400

variable {a b c : ℝ}

theorem inequality_sum_of_reciprocals
  (hpos_a : 0 < a) (hpos_b : 0 < b) (hpos_c : 0 < c)
  (hsum : a + b + c = 3) :
  (1 / (2 * a^2 + b^2 + c^2) + 1 / (2 * b^2 + c^2 + a^2) + 1 / (2 * c^2 + a^2 + b^2)) ≤ 3/4 :=
sorry

end NUMINAMATH_GPT_inequality_sum_of_reciprocals_l1934_193400


namespace NUMINAMATH_GPT_solve_system_of_equations_l1934_193410

-- Definition of the system of equations as conditions
def eq1 (x y : ℤ) : Prop := 3 * x + y = 2
def eq2 (x y : ℤ) : Prop := 2 * x - 3 * y = 27

-- The theorem claiming the solution set is { (3, -7) }
theorem solve_system_of_equations :
  ∀ x y : ℤ, eq1 x y ∧ eq2 x y ↔ (x, y) = (3, -7) :=
by
  sorry

end NUMINAMATH_GPT_solve_system_of_equations_l1934_193410


namespace NUMINAMATH_GPT_average_salary_of_employees_l1934_193401

theorem average_salary_of_employees
  (A : ℝ)  -- Define the average monthly salary A of 18 employees
  (h1 : 18*A + 5800 = 19*(A + 200))  -- Condition given in the problem
  : A = 2000 :=  -- The conclusion we need to prove
by
  sorry

end NUMINAMATH_GPT_average_salary_of_employees_l1934_193401


namespace NUMINAMATH_GPT_find_x_l1934_193414

-- Definitions for the problem
def a (x : ℝ) : ℝ × ℝ := (1, x)
def b : ℝ × ℝ := (-2, 1)

-- Theorem statement
theorem find_x (x : ℝ) (h : ∃ k : ℝ, a x = k • b) : x = -1/2 := by
  sorry

end NUMINAMATH_GPT_find_x_l1934_193414


namespace NUMINAMATH_GPT_union_M_N_l1934_193491

def M := {x : ℝ | -2 < x ∧ x < -1}
def N := {x : ℝ | (1 / 2 : ℝ)^x ≤ 4}

theorem union_M_N :
  M ∪ N = {x : ℝ | x ≥ -2} :=
sorry

end NUMINAMATH_GPT_union_M_N_l1934_193491


namespace NUMINAMATH_GPT_find_min_n_l1934_193447

variable (a : Nat → Int)
variable (S : Nat → Int)
variable (d : Nat)
variable (n : Nat)

-- Definitions based on given conditions
def arithmetic_sequence (a : Nat → Int) (d : Nat) : Prop :=
  ∀ n, a (n + 1) = a n + d

def a1_eq_neg3 (a : Nat → Int) : Prop :=
  a 1 = -3

def condition (a : Nat → Int) (d : Nat) : Prop :=
  11 * a 5 = 5 * a 8

-- Correct answer condition
def minimized_sum_condition (a : Nat → Int) (S : Nat → Int) (d : Nat) (n : Nat) : Prop :=
  S n ≤ S (n + 1)

theorem find_min_n (a : Nat → Int) (S : Nat → Int) (d : Nat) :
  arithmetic_sequence a d ->
  a1_eq_neg3 a ->
  condition a 2 ->
  minimized_sum_condition a S 2 2 :=
by
  sorry

end NUMINAMATH_GPT_find_min_n_l1934_193447


namespace NUMINAMATH_GPT_determine_number_of_students_l1934_193452

theorem determine_number_of_students 
  (n : ℕ) 
  (h1 : n < 600) 
  (h2 : n % 25 = 24) 
  (h3 : n % 19 = 15) : 
  n = 399 :=
by
  -- The proof will be provided here.
  sorry

end NUMINAMATH_GPT_determine_number_of_students_l1934_193452


namespace NUMINAMATH_GPT_no_all_nine_odd_l1934_193433

theorem no_all_nine_odd
  (a1 a2 a3 a4 a5 b1 b2 b3 b4 : ℤ)
  (h1 : a1 % 2 = 1) (h2 : a2 % 2 = 1) (h3 : a3 % 2 = 1)
  (h4 : a4 % 2 = 1) (h5 : a5 % 2 = 1) (h6 : b1 % 2 = 1)
  (h7 : b2 % 2 = 1) (h8 : b3 % 2 = 1) (h9 : b4 % 2 = 1)
  (sum_eq : a1 + a2 + a3 + a4 + a5 = b1 + b2 + b3 + b4) : 
  false :=
sorry

end NUMINAMATH_GPT_no_all_nine_odd_l1934_193433


namespace NUMINAMATH_GPT_time_for_train_to_pass_pole_l1934_193461

-- Definitions based on conditions
def train_length_meters : ℕ := 160
def train_speed_kmph : ℕ := 72

-- The calculated speed in m/s
def train_speed_mps : ℕ := train_speed_kmph * 1000 / 3600

-- The calculation of time taken to pass the pole
def time_to_pass_pole : ℕ := train_length_meters / train_speed_mps

-- The theorem statement
theorem time_for_train_to_pass_pole : time_to_pass_pole = 8 := sorry

end NUMINAMATH_GPT_time_for_train_to_pass_pole_l1934_193461


namespace NUMINAMATH_GPT_compare_neg_fractions_l1934_193453

theorem compare_neg_fractions : - (1 : ℝ) / 3 < - (1 : ℝ) / 4 :=
  sorry

end NUMINAMATH_GPT_compare_neg_fractions_l1934_193453


namespace NUMINAMATH_GPT_problem_l1934_193426

noncomputable def a : ℝ := Real.log 8 / Real.log 3
noncomputable def b : ℝ := Real.log 25 / Real.log 4
noncomputable def c : ℝ := Real.log 24 / Real.log 4

theorem problem : a < c ∧ c < b :=
by
  sorry

end NUMINAMATH_GPT_problem_l1934_193426


namespace NUMINAMATH_GPT_worker_net_salary_change_l1934_193422

theorem worker_net_salary_change (S : ℝ) :
  let final_salary := S * 1.15 * 0.90 * 1.20 * 0.95
  let net_change := final_salary - S
  net_change = 0.0355 * S := by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_worker_net_salary_change_l1934_193422


namespace NUMINAMATH_GPT_least_sum_of_factors_l1934_193436

theorem least_sum_of_factors (a b : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : a * b = 2400) : a + b = 98 :=
sorry

end NUMINAMATH_GPT_least_sum_of_factors_l1934_193436


namespace NUMINAMATH_GPT_moss_flower_pollen_scientific_notation_l1934_193419

theorem moss_flower_pollen_scientific_notation (d : ℝ) (h : d = 0.0000084) : ∃ n : ℤ, d = 8.4 * 10^n ∧ n = -6 :=
by
  use -6
  rw [h]
  simp
  sorry

end NUMINAMATH_GPT_moss_flower_pollen_scientific_notation_l1934_193419


namespace NUMINAMATH_GPT_age_of_other_man_replaced_l1934_193444

-- Define the conditions
variables (A : ℝ) (x : ℝ)
variable (average_age_women : ℝ := 50)
variable (num_men : ℕ := 10)
variable (increase_age : ℝ := 6)
variable (one_man_age : ℝ := 22)

-- State the theorem to be proved
theorem age_of_other_man_replaced :
  2 * average_age_women - (one_man_age + x) = 10 * (A + increase_age) - 10 * A →
  x = 18 :=
by
  sorry

end NUMINAMATH_GPT_age_of_other_man_replaced_l1934_193444


namespace NUMINAMATH_GPT_train_stoppage_time_l1934_193482

-- Definitions from conditions
def speed_without_stoppages := 60 -- kmph
def speed_with_stoppages := 36 -- kmph

-- Main statement to prove
theorem train_stoppage_time : (60 - 36) / 60 * 60 = 24 := by
  sorry

end NUMINAMATH_GPT_train_stoppage_time_l1934_193482


namespace NUMINAMATH_GPT_total_surface_area_of_cube_l1934_193488

theorem total_surface_area_of_cube (edge_sum : ℕ) (h_edge_sum : edge_sum = 180) :
  ∃ (S : ℕ), S = 1350 := 
by
  sorry

end NUMINAMATH_GPT_total_surface_area_of_cube_l1934_193488


namespace NUMINAMATH_GPT_total_circle_area_within_triangle_l1934_193490

-- Define the sides of the triangle
def triangle_sides : Prop := ∃ (a b c : ℝ), a = 3 ∧ b = 4 ∧ c = 5

-- Define the radii and center of the circles at each vertex of the triangle
def circle_centers_and_radii : Prop := ∃ (r : ℝ) (A B C : ℝ × ℝ), r = 1

-- The formal statement that we need to prove:
theorem total_circle_area_within_triangle :
  triangle_sides ∧ circle_centers_and_radii → 
  (total_area_of_circles_within_triangle = π / 2) := sorry

end NUMINAMATH_GPT_total_circle_area_within_triangle_l1934_193490


namespace NUMINAMATH_GPT_sphere_surface_area_l1934_193445

-- Let A, B, C, D be distinct points on the same sphere
variables (A B C D : ℝ)

-- Defining edges AB, AC, AD and their lengths
variables (AB AC AD : ℝ)
variable (is_perpendicular : AB * AC = 0 ∧ AB * AD = 0 ∧ AC * AD = 0)

-- Setting specific edge lengths
variables (AB_length : AB = 1) (AC_length : AC = 2) (AD_length : AD = 3)

-- The proof problem: Prove that the surface area of the sphere is 14π
theorem sphere_surface_area : 4 * Real.pi * ((1 + 4 + 9) / 4) = 14 * Real.pi :=
by
  sorry

end NUMINAMATH_GPT_sphere_surface_area_l1934_193445


namespace NUMINAMATH_GPT_dressing_q_vinegar_percentage_l1934_193424

/-- 
Given:
1. P is 30% vinegar and 70% oil.
2. Q is V% vinegar and the rest is oil.
3. The new dressing is produced from 10% of P and 90% of Q and is 12% vinegar.
Prove:
The percentage of vinegar in dressing Q is 10%.
-/
theorem dressing_q_vinegar_percentage (V : ℝ) (h : 0.10 * 0.30 + 0.90 * V = 0.12) : V = 0.10 :=
by 
    sorry

end NUMINAMATH_GPT_dressing_q_vinegar_percentage_l1934_193424


namespace NUMINAMATH_GPT_solve_first_equation_solve_second_equation_l1934_193484

-- Statement for the first equation
theorem solve_first_equation : ∀ x : ℝ, x^2 - 3*x - 4 = 0 ↔ x = 4 ∨ x = -1 := by
  sorry

-- Statement for the second equation
theorem solve_second_equation : ∀ x : ℝ, x * (x - 2) = 1 ↔ x = 1 + Real.sqrt 2 ∨ x = 1 - Real.sqrt 2 := by
  sorry

end NUMINAMATH_GPT_solve_first_equation_solve_second_equation_l1934_193484


namespace NUMINAMATH_GPT_minimum_value_of_2x_3y_l1934_193416

noncomputable def minimum_value (x y : ℝ) (h₁ : 0 < x) (h₂ : 0 < y) (h₃ : (2/x) + (3/y) = 1) : ℝ :=
  2*x + 3*y

theorem minimum_value_of_2x_3y {x y : ℝ} (hx : 0 < x) (hy : 0 < y) (hxy : (2/x) + (3/y) = 1) : minimum_value x y hx hy hxy = 25 :=
sorry

end NUMINAMATH_GPT_minimum_value_of_2x_3y_l1934_193416


namespace NUMINAMATH_GPT_consecutive_numbers_average_l1934_193497

theorem consecutive_numbers_average (a b c d e f g : ℕ)
  (h1 : (a + b + c + d + e + f + g) / 7 = 9)
  (h2 : 2 * a = g) : 
  7 = 7 :=
by sorry

end NUMINAMATH_GPT_consecutive_numbers_average_l1934_193497


namespace NUMINAMATH_GPT_points_per_other_player_l1934_193451

-- Define the conditions as variables
variables (total_points : ℕ) (faye_points : ℕ) (total_players : ℕ)

-- Assume the given conditions
def conditions : Prop :=
  total_points = 68 ∧ faye_points = 28 ∧ total_players = 5

-- Define the proof problem: Prove that the points scored by each of the other players is 10
theorem points_per_other_player :
  conditions total_points faye_points total_players →
  (total_points - faye_points) / (total_players - 1) = 10 :=
by
  sorry

end NUMINAMATH_GPT_points_per_other_player_l1934_193451


namespace NUMINAMATH_GPT_problem_may_not_be_equal_l1934_193460

-- Define the four pairs of expressions
def expr_A (a b : ℕ) := (a + b) = (b + a)
def expr_B (a : ℕ) := (3 * a) = (a + a + a)
def expr_C (a b : ℕ) := (3 * (a + b)) ≠ (3 * a + b)
def expr_D (a : ℕ) := (a ^ 3) = (a * a * a)

-- State the theorem stating that the expression in condition C may not be equal
theorem problem_may_not_be_equal (a b : ℕ) : (3 * (a + b)) ≠ (3 * a + b) :=
by
  sorry

end NUMINAMATH_GPT_problem_may_not_be_equal_l1934_193460


namespace NUMINAMATH_GPT_min_value_2_l1934_193440

noncomputable def min_value (a b : ℝ) : ℝ :=
  1 / a + 1 / (b + 1)

theorem min_value_2 {a b : ℝ} (h1 : a > 0) (h2 : b > -1) (h3 : a + b = 1) : min_value a b = 2 :=
by
  sorry

end NUMINAMATH_GPT_min_value_2_l1934_193440


namespace NUMINAMATH_GPT_length_of_rectangle_l1934_193403

theorem length_of_rectangle (P L B : ℕ) (h₁ : P = 800) (h₂ : B = 300) (h₃ : P = 2 * (L + B)) : L = 100 := by
  sorry

end NUMINAMATH_GPT_length_of_rectangle_l1934_193403


namespace NUMINAMATH_GPT_muffins_sold_in_afternoon_l1934_193472

variable (total_muffins : ℕ)
variable (morning_muffins : ℕ)
variable (remaining_muffins : ℕ)

theorem muffins_sold_in_afternoon 
  (h1 : total_muffins = 20) 
  (h2 : morning_muffins = 12) 
  (h3 : remaining_muffins = 4) : 
  (total_muffins - remaining_muffins - morning_muffins) = 4 := 
by
  sorry

end NUMINAMATH_GPT_muffins_sold_in_afternoon_l1934_193472


namespace NUMINAMATH_GPT_simplify_polynomial_l1934_193464

variable (p : ℝ)

theorem simplify_polynomial :
  (7 * p ^ 5 - 4 * p ^ 3 + 8 * p ^ 2 - 5 * p + 3) + (- p ^ 5 + 3 * p ^ 3 - 7 * p ^ 2 + 6 * p + 2) =
  6 * p ^ 5 - p ^ 3 + p ^ 2 + p + 5 :=
by
  sorry

end NUMINAMATH_GPT_simplify_polynomial_l1934_193464


namespace NUMINAMATH_GPT_problem_water_percentage_l1934_193471

noncomputable def percentage_water_in_mixture 
  (volA volB volC volD : ℕ) 
  (pctA pctB pctC pctD : ℝ) : ℝ :=
  let total_volume := volA + volB + volC + volD
  let total_solution := volA * pctA + volB * pctB + volC * pctC + volD * pctD
  let total_water := total_volume - total_solution
  (total_water / total_volume) * 100

theorem problem_water_percentage :
  percentage_water_in_mixture 100 90 60 50 0.25 0.3 0.4 0.2 = 71.33 :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_problem_water_percentage_l1934_193471


namespace NUMINAMATH_GPT_velocity_at_1_eq_5_l1934_193418

def S (t : ℝ) : ℝ := 2 * t^2 + t

theorem velocity_at_1_eq_5 : (deriv S 1) = 5 :=
by sorry

end NUMINAMATH_GPT_velocity_at_1_eq_5_l1934_193418


namespace NUMINAMATH_GPT_solve_for_x_l1934_193465

theorem solve_for_x (x : ℝ) : 7 * (4 * x + 3) - 5 = -3 * (2 - 5 * x) ↔ x = -22 / 13 := 
by 
  sorry

end NUMINAMATH_GPT_solve_for_x_l1934_193465


namespace NUMINAMATH_GPT_plastering_cost_l1934_193439

variable (l w d : ℝ) (c : ℝ)

theorem plastering_cost :
  l = 60 → w = 25 → d = 10 → c = 0.90 →
    let A_bottom := l * w;
    let A_long_walls := 2 * (l * d);
    let A_short_walls := 2 * (w * d);
    let A_total := A_bottom + A_long_walls + A_short_walls;
    let C_total := A_total * c;
    C_total = 2880 :=
by sorry

end NUMINAMATH_GPT_plastering_cost_l1934_193439


namespace NUMINAMATH_GPT_value_of_half_plus_five_l1934_193480

theorem value_of_half_plus_five (n : ℕ) (h₁ : n = 20) : (n / 2) + 5 = 15 := 
by {
  sorry
}

end NUMINAMATH_GPT_value_of_half_plus_five_l1934_193480


namespace NUMINAMATH_GPT_find_b_from_root_l1934_193421

theorem find_b_from_root (b : ℝ) :
  (Polynomial.eval (-10) (Polynomial.C 1 * X^2 + Polynomial.C b * X + Polynomial.C (-30)) = 0) →
  b = 7 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_find_b_from_root_l1934_193421


namespace NUMINAMATH_GPT_simplify_expression_l1934_193450

theorem simplify_expression : (4 + 3) + (8 - 3 - 1) = 11 := by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1934_193450


namespace NUMINAMATH_GPT_power_congruence_l1934_193435

theorem power_congruence (a b n : ℕ) (h : a ≡ b [MOD n]) : a^n ≡ b^n [MOD n^2] :=
sorry

end NUMINAMATH_GPT_power_congruence_l1934_193435


namespace NUMINAMATH_GPT_dimes_count_l1934_193479

def num_dimes (total_in_cents : ℤ) (value_quarter value_dime value_nickel : ℤ) (num_each : ℤ) : Prop :=
  total_in_cents = num_each * (value_quarter + value_dime + value_nickel)

theorem dimes_count (num_each : ℤ) :
  num_dimes 440 25 10 5 num_each → num_each = 11 :=
by sorry

end NUMINAMATH_GPT_dimes_count_l1934_193479


namespace NUMINAMATH_GPT_range_log_div_pow3_div3_l1934_193475

noncomputable def log_div (x y : ℝ) : ℝ := Real.log (x / y)
noncomputable def log_div_pow3 (x y : ℝ) : ℝ := Real.log (x^3 / y^(1/2))
noncomputable def log_div_pow3_div3 (x y : ℝ) : ℝ := Real.log (x^3 / (3 * y))

theorem range_log_div_pow3_div3 
  (x y : ℝ) 
  (h1 : 1 ≤ log_div x y ∧ log_div x y ≤ 2)
  (h2 : 2 ≤ log_div_pow3 x y ∧ log_div_pow3 x y ≤ 3) 
  : Real.log (x^3 / (3 * y)) ∈ Set.Icc (26/15 : ℝ) 3 :=
sorry

end NUMINAMATH_GPT_range_log_div_pow3_div3_l1934_193475


namespace NUMINAMATH_GPT_pythagorean_triple_third_number_l1934_193481

theorem pythagorean_triple_third_number (x : ℕ) (h1 : x^2 + 8^2 = 17^2) : x = 15 :=
sorry

end NUMINAMATH_GPT_pythagorean_triple_third_number_l1934_193481


namespace NUMINAMATH_GPT_cars_produced_in_europe_l1934_193494

theorem cars_produced_in_europe (total_cars : ℕ) (cars_in_north_america : ℕ) (cars_in_europe : ℕ) :
  total_cars = 6755 → cars_in_north_america = 3884 → cars_in_europe = total_cars - cars_in_north_america → cars_in_europe = 2871 :=
by
  -- necessary calculations and logical steps
  sorry

end NUMINAMATH_GPT_cars_produced_in_europe_l1934_193494


namespace NUMINAMATH_GPT_floor_equation_solution_l1934_193486

theorem floor_equation_solution (a b : ℕ) (a_pos : 0 < a) (b_pos : 0 < b) :
  (⌊ (a^2 : ℝ) / b ⌋ + ⌊ (b^2 : ℝ) / a ⌋ = ⌊ (a^2 + b^2 : ℝ) / (a * b) ⌋ + a * b) ↔
    (∃ n : ℕ, a = n ∧ b = n^2 + 1) ∨ (∃ n : ℕ, a = n^2 + 1 ∧ b = n) :=
sorry

end NUMINAMATH_GPT_floor_equation_solution_l1934_193486


namespace NUMINAMATH_GPT_slope_of_line_dividing_rectangle_l1934_193406

theorem slope_of_line_dividing_rectangle (h_vertices : 
  ∃ (A B C D : ℝ × ℝ), A = (1, 0) ∧ B = (9, 0) ∧ C = (1, 2) ∧ D = (9, 2) ∧ 
  (∃ line : ℝ × ℝ, line = (0, 0) ∧ line = (5, 1))) : 
  ∃ m : ℝ, m = 1 / 5 :=
sorry

end NUMINAMATH_GPT_slope_of_line_dividing_rectangle_l1934_193406


namespace NUMINAMATH_GPT_distance_y_axis_l1934_193437

def point_M (m : ℝ) : ℝ × ℝ := (2 - m, 1 + 2 * m)

theorem distance_y_axis :
  ∀ m : ℝ, abs (2 - m) = 2 → (point_M m = (2, 1)) ∨ (point_M m = (-2, 9)) :=
by
  sorry

end NUMINAMATH_GPT_distance_y_axis_l1934_193437


namespace NUMINAMATH_GPT_evenFunctionExists_l1934_193489

-- Definitions based on conditions
def isEvenFunction (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = f (-x)

def passesThroughPoints (f : ℝ → ℝ) (points : List (ℝ × ℝ)) : Prop :=
  ∀ p ∈ points, f p.1 = p.2

-- Example function
def exampleEvenFunction (x : ℝ) : ℝ := x^2 * (x - 3) * (x + 1)

-- Points to pass through
def givenPoints : List (ℝ × ℝ) := [(-1, 0), (0.5, 2.5), (3, 0)]

-- Theorem to be proven
theorem evenFunctionExists : 
  isEvenFunction exampleEvenFunction ∧ passesThroughPoints exampleEvenFunction givenPoints :=
by
  sorry

end NUMINAMATH_GPT_evenFunctionExists_l1934_193489


namespace NUMINAMATH_GPT_inequalities_l1934_193449

variable {a b c : ℝ}

theorem inequalities (ha : a < 0) (hab : a < b) (hbc : b < c) :
  a^2 * b < b^2 * c ∧ a^2 * c < b^2 * c ∧ a^2 * b < a^2 * c :=
by
  sorry

end NUMINAMATH_GPT_inequalities_l1934_193449


namespace NUMINAMATH_GPT_cost_per_bag_l1934_193462

theorem cost_per_bag
  (friends : ℕ)
  (payment_per_friend : ℕ)
  (total_bags : ℕ)
  (total_cost : ℕ)
  (h1 : friends = 3)
  (h2 : payment_per_friend = 5)
  (h3 : total_bags = 5)
  (h4 : total_cost = friends * payment_per_friend) :
  total_cost / total_bags = 3 :=
by {
  sorry
}

end NUMINAMATH_GPT_cost_per_bag_l1934_193462


namespace NUMINAMATH_GPT_k_is_perfect_square_l1934_193459

theorem k_is_perfect_square (m n : ℕ) (h_pos_m : 0 < m) (h_pos_n : 0 < n) (k : ℕ)
  (h_k : k = (m + n)^2 / (4 * m * (m - n)^2 + 4)) 
  (h_int_k : k * (4 * m * (m - n)^2 + 4) = (m + n)^2) :
  ∃ x : ℕ, k = x^2 := 
sorry

end NUMINAMATH_GPT_k_is_perfect_square_l1934_193459


namespace NUMINAMATH_GPT_train_cars_count_l1934_193434

theorem train_cars_count
  (cars_in_first_15_seconds : ℕ)
  (time_for_first_5_cars : ℕ)
  (total_time_to_pass : ℕ)
  (h_cars_in_first_15_seconds : cars_in_first_15_seconds = 5)
  (h_time_for_first_5_cars : time_for_first_5_cars = 15)
  (h_total_time_to_pass : total_time_to_pass = 210) :
  (total_time_to_pass / time_for_first_5_cars) * cars_in_first_15_seconds = 70 := 
by 
  sorry

end NUMINAMATH_GPT_train_cars_count_l1934_193434


namespace NUMINAMATH_GPT_mila_social_media_time_week_l1934_193412

theorem mila_social_media_time_week
  (hours_per_day_on_phone : ℕ)
  (half_on_social_media : ℕ)
  (days_in_week : ℕ)
  (h1 : hours_per_day_on_phone = 6)
  (h2 : half_on_social_media = hours_per_day_on_phone / 2)
  (h3 : days_in_week = 7) : 
  half_on_social_media * days_in_week = 21 := 
by
  rw [h2, h3]
  norm_num
  exact h1.symm ▸ rfl

end NUMINAMATH_GPT_mila_social_media_time_week_l1934_193412


namespace NUMINAMATH_GPT_smallest_c_minus_a_l1934_193430

theorem smallest_c_minus_a (a b c : ℕ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) 
  (h_prod : a * b * c = 362880) (h_ineq : a < b ∧ b < c) : 
  c - a = 109 :=
sorry

end NUMINAMATH_GPT_smallest_c_minus_a_l1934_193430


namespace NUMINAMATH_GPT_paying_students_pay_7_l1934_193428

/-- At a school, 40% of the students receive a free lunch. 
These lunches are paid for by making sure the price paid by the 
paying students is enough to cover everyone's meal. 
It costs $210 to feed 50 students. 
Prove that each paying student pays $7. -/
theorem paying_students_pay_7 (total_students : ℕ) 
  (free_lunch_percentage : ℤ)
  (cost_per_50_students : ℕ) : 
  free_lunch_percentage = 40 ∧ cost_per_50_students = 210 →
  ∃ (paying_students_pay : ℕ), paying_students_pay = 7 :=
by
  -- Let the proof steps and conditions be set up as follows
  -- (this part is not required, hence using sorry)
  sorry

end NUMINAMATH_GPT_paying_students_pay_7_l1934_193428


namespace NUMINAMATH_GPT_find_speed_of_goods_train_l1934_193485

variable (v : ℕ) -- Speed of the goods train in km/h

theorem find_speed_of_goods_train
  (h1 : 0 < v) 
  (h2 : 6 * v + 4 * 90 = 10 * v) :
  v = 36 :=
by
  sorry

end NUMINAMATH_GPT_find_speed_of_goods_train_l1934_193485


namespace NUMINAMATH_GPT_lassis_from_mangoes_l1934_193469

theorem lassis_from_mangoes (m l m' : ℕ) (h : m' = 18) (hlm : l / m = 8 / 3) : l / m' = 48 / 18 :=
by
  sorry

end NUMINAMATH_GPT_lassis_from_mangoes_l1934_193469


namespace NUMINAMATH_GPT_divisible_by_six_l1934_193413

theorem divisible_by_six (m : ℕ) : 6 ∣ (m^3 + 11 * m) := 
sorry

end NUMINAMATH_GPT_divisible_by_six_l1934_193413


namespace NUMINAMATH_GPT_cube_properties_l1934_193417

theorem cube_properties (x : ℝ) (h1 : 6 * (2 * (8 * x)^(1/3))^2 = x) : x = 13824 :=
sorry

end NUMINAMATH_GPT_cube_properties_l1934_193417


namespace NUMINAMATH_GPT_jason_needs_87_guppies_per_day_l1934_193476

def guppies_needed_per_day (moray_eel_guppies : Nat)
  (betta_fish_number : Nat) (betta_fish_guppies : Nat)
  (angelfish_number : Nat) (angelfish_guppies : Nat)
  (lionfish_number : Nat) (lionfish_guppies : Nat) : Nat :=
  moray_eel_guppies +
  betta_fish_number * betta_fish_guppies +
  angelfish_number * angelfish_guppies +
  lionfish_number * lionfish_guppies

theorem jason_needs_87_guppies_per_day :
  guppies_needed_per_day 20 5 7 3 4 2 10 = 87 := by
  sorry

end NUMINAMATH_GPT_jason_needs_87_guppies_per_day_l1934_193476


namespace NUMINAMATH_GPT_annual_interest_rate_l1934_193477

variable (P : ℝ) (t : ℝ)
variable (h1 : t = 25)
variable (h2 : ∀ r : ℝ, P * 2 = P * (1 + r * t))

theorem annual_interest_rate : ∃ r : ℝ, P * 2 = P * (1 + r * t) ∧ r = 0.04 := by
  sorry

end NUMINAMATH_GPT_annual_interest_rate_l1934_193477


namespace NUMINAMATH_GPT_range_of_a_l1934_193402

noncomputable def f (a : ℝ) (x : ℝ) := 2 * a * x^2 + 4 * (a - 3) * x + 5

theorem range_of_a (a : ℝ) :
  (∀ x < 3, f a x ≤ f a (3 : ℝ)) ↔ 0 ≤ a ∧ a ≤ 3 / 4 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l1934_193402


namespace NUMINAMATH_GPT_fraction_inequality_l1934_193487

variables (a b m : ℝ)

theorem fraction_inequality (h1 : a > b) (h2 : m > 0) : (b + m) / (a + m) > b / a :=
sorry

end NUMINAMATH_GPT_fraction_inequality_l1934_193487


namespace NUMINAMATH_GPT_distance_from_point_to_focus_l1934_193431

theorem distance_from_point_to_focus (P : ℝ × ℝ) (hP : P.2^2 = 8 * P.1) (hX : P.1 = 8) :
  dist P (2, 0) = 10 :=
sorry

end NUMINAMATH_GPT_distance_from_point_to_focus_l1934_193431


namespace NUMINAMATH_GPT_simplify_expression_l1934_193408

-- Define the algebraic expressions
def expr1 (x : ℝ) := (3 * x - 4) * (x + 9)
def expr2 (x : ℝ) := (x + 6) * (3 * x + 2)
def combined_expr (x : ℝ) := expr1 x + expr2 x
def result_expr (x : ℝ) := 6 * x^2 + 43 * x - 24

-- Theorem stating the equivalence
theorem simplify_expression (x : ℝ) : combined_expr x = result_expr x := 
by 
  sorry

end NUMINAMATH_GPT_simplify_expression_l1934_193408


namespace NUMINAMATH_GPT_solve_s_l1934_193441

theorem solve_s (s : ℝ) (h_pos : 0 < s) (h_eq : s^3 = 256) : s = 4 :=
sorry

end NUMINAMATH_GPT_solve_s_l1934_193441


namespace NUMINAMATH_GPT_part1_m_n_part2_k_l1934_193468

-- Definitions of vectors a, b, and c
def veca : ℝ × ℝ := (3, 2)
def vecb : ℝ × ℝ := (-1, 2)
def vecc : ℝ × ℝ := (4, 1)

-- Part (1)
theorem part1_m_n : 
  ∃ (m n : ℝ), (-m + 4 * n = 3) ∧ (2 * m + n = 2) :=
sorry

-- Part (2)
theorem part2_k : 
  ∃ (k : ℝ), (3 + 4 * k) * 2 - (-5) * (2 + k) = 0 :=
sorry

end NUMINAMATH_GPT_part1_m_n_part2_k_l1934_193468


namespace NUMINAMATH_GPT_sum_congruent_mod_9_l1934_193427

theorem sum_congruent_mod_9 : 
  (2 + 33 + 444 + 5555 + 66666 + 777777 + 8888888 + 99999999) % 9 = 6 := 
by 
  -- Proof steps here
  sorry

end NUMINAMATH_GPT_sum_congruent_mod_9_l1934_193427


namespace NUMINAMATH_GPT_daniel_pages_to_read_l1934_193432

-- Definitions from conditions
def total_pages : ℕ := 980
def daniel_read_time_per_page : ℕ := 50
def emma_read_time_per_page : ℕ := 40

-- The theorem that states the solution
theorem daniel_pages_to_read (d : ℕ) :
  d = 436 ↔ daniel_read_time_per_page * d = emma_read_time_per_page * (total_pages - d) :=
by sorry

end NUMINAMATH_GPT_daniel_pages_to_read_l1934_193432


namespace NUMINAMATH_GPT_binary_to_decimal_is_1023_l1934_193499

-- Define the binary number 1111111111 in terms of its decimal representation
def binary_to_decimal : ℕ :=
  (1 * 2^9 + 1 * 2^8 + 1 * 2^7 + 1 * 2^6 + 1 * 2^5 + 1 * 2^4 + 1 * 2^3 + 1 * 2^2 + 1 * 2^1 + 1 * 2^0)

-- The theorem statement
theorem binary_to_decimal_is_1023 : binary_to_decimal = 1023 :=
by
  sorry

end NUMINAMATH_GPT_binary_to_decimal_is_1023_l1934_193499


namespace NUMINAMATH_GPT_prob_white_given_popped_l1934_193458

-- Definitions for given conditions:
def P_white : ℚ := 1 / 2
def P_yellow : ℚ := 1 / 4
def P_blue : ℚ := 1 / 4

def P_popped_given_white : ℚ := 1 / 3
def P_popped_given_yellow : ℚ := 3 / 4
def P_popped_given_blue : ℚ := 2 / 3

-- Calculations derived from conditions:
def P_white_popped : ℚ := P_white * P_popped_given_white
def P_yellow_popped : ℚ := P_yellow * P_popped_given_yellow
def P_blue_popped : ℚ := P_blue * P_popped_given_blue

def P_popped : ℚ := P_white_popped + P_yellow_popped + P_blue_popped

-- Main theorem to be proved:
theorem prob_white_given_popped : (P_white_popped / P_popped) = 2 / 11 :=
by sorry

end NUMINAMATH_GPT_prob_white_given_popped_l1934_193458


namespace NUMINAMATH_GPT_inequality_correct_l1934_193446

variable (m n c : ℝ)

theorem inequality_correct (h : m > n) : m + c > n + c := 
by sorry

end NUMINAMATH_GPT_inequality_correct_l1934_193446


namespace NUMINAMATH_GPT_total_cost_is_correct_l1934_193470

def bus_ride_cost : ℝ := 1.75
def train_ride_cost : ℝ := bus_ride_cost + 6.35
def total_cost : ℝ := bus_ride_cost + train_ride_cost

theorem total_cost_is_correct : total_cost = 9.85 :=
by
  -- proof here
  sorry

end NUMINAMATH_GPT_total_cost_is_correct_l1934_193470


namespace NUMINAMATH_GPT_spaghetti_cost_l1934_193492

theorem spaghetti_cost (hamburger_cost french_fry_cost soda_cost spaghetti_cost split_payment friends : ℝ) 
(hamburger_count : ℕ) (french_fry_count : ℕ) (soda_count : ℕ) (friend_count : ℕ)
(h_split_payment : split_payment * friend_count = 25)
(h_hamburger_cost : hamburger_cost = 3 * hamburger_count)
(h_french_fry_cost : french_fry_cost = 1.20 * french_fry_count)
(h_soda_cost : soda_cost = 0.5 * soda_count)
(h_total_order_cost : hamburger_cost + french_fry_cost + soda_cost + spaghetti_cost = split_payment * friend_count) :
spaghetti_cost = 2.70 :=
by {
  sorry
}

end NUMINAMATH_GPT_spaghetti_cost_l1934_193492


namespace NUMINAMATH_GPT_remaining_pages_after_a_week_l1934_193478

-- Define the conditions
def total_pages : Nat := 381
def pages_read_initial : Nat := 149
def pages_per_day : Nat := 20
def days : Nat := 7

-- Define the final statement to prove
theorem remaining_pages_after_a_week :
  let pages_left_initial := total_pages - pages_read_initial
  let pages_read_week := pages_per_day * days
  let pages_remaining := pages_left_initial - pages_read_week
  pages_remaining = 92 := by
  sorry

end NUMINAMATH_GPT_remaining_pages_after_a_week_l1934_193478


namespace NUMINAMATH_GPT_interest_calculation_l1934_193443

variables (P R SI : ℝ) (T : ℕ)

-- Given conditions
def principal := (P = 8)
def rate := (R = 0.05)
def simple_interest := (SI = 4.8)

-- Goal
def time_calculated := (T = 12)

-- Lean statement combining the conditions
theorem interest_calculation : principal P → rate R → simple_interest SI → T = 12 :=
by
  intros hP hR hSI
  sorry

end NUMINAMATH_GPT_interest_calculation_l1934_193443


namespace NUMINAMATH_GPT_range_of_a_for_critical_point_l1934_193409

noncomputable def f (x : ℝ) : ℝ := (1 / 2) * x^2 - 9 * Real.log x

theorem range_of_a_for_critical_point :
  ∀ a : ℝ, (∃ x ∈ Set.Icc (a - 1) (a + 1), deriv f x = 0) ↔ 2 < a ∧ a < 4 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_for_critical_point_l1934_193409


namespace NUMINAMATH_GPT_min_a_4_l1934_193493

theorem min_a_4 (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 9 * x + y = x * y) : 
  4 * x + y ≥ 25 :=
sorry

end NUMINAMATH_GPT_min_a_4_l1934_193493


namespace NUMINAMATH_GPT_square_side_length_in_right_triangle_l1934_193454

theorem square_side_length_in_right_triangle
  (AC BC : ℝ)
  (h1 : AC = 3)
  (h2 : BC = 7)
  (right_triangle : ∃ A B C : ℝ × ℝ, A = (3, 0) ∧ B = (0, 7) ∧ C = (0, 0) ∧ (A.1 - C.1)^2 + (A.2 - C.2)^2 = AC^2 ∧ (B.1 - C.1)^2 + (B.2 - C.2)^2 = BC^2 ∧ (A.1 - B.1)^2 + (A.2 - B.2)^2 = AC^2 + BC^2) :
  ∃ s : ℝ, s = 2.1 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_square_side_length_in_right_triangle_l1934_193454


namespace NUMINAMATH_GPT_solve_for_j_l1934_193405

variable (j : ℝ)
variable (h1 : j > 0)
variable (v1 : ℝ × ℝ × ℝ := (3, 4, 5))
variable (v2 : ℝ × ℝ × ℝ := (2, j, 3))
variable (v3 : ℝ × ℝ × ℝ := (2, 3, j))

theorem solve_for_j :
  |(3 * (j * j - 3 * 3) - 2 * (4 * j - 5 * 3) + 2 * (4 * 3 - 5 * j))| = 36 →
  j = (9 + Real.sqrt 585) / 6 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_j_l1934_193405


namespace NUMINAMATH_GPT_planks_needed_for_surface_l1934_193448

theorem planks_needed_for_surface
  (total_tables : ℕ := 5)
  (total_planks : ℕ := 45)
  (planks_per_leg : ℕ := 4) :
  ∃ S : ℕ, total_tables * (planks_per_leg + S) = total_planks ∧ S = 5 :=
by
  use 5
  sorry

end NUMINAMATH_GPT_planks_needed_for_surface_l1934_193448


namespace NUMINAMATH_GPT_neg_existence_of_ge_impl_universal_lt_l1934_193457

theorem neg_existence_of_ge_impl_universal_lt : (¬ ∃ x : ℕ, x^2 ≥ x) ↔ ∀ x : ℕ, x^2 < x := 
sorry

end NUMINAMATH_GPT_neg_existence_of_ge_impl_universal_lt_l1934_193457


namespace NUMINAMATH_GPT_travel_time_between_resorts_l1934_193467

theorem travel_time_between_resorts
  (num_cars : ℕ)
  (car_interval : ℕ)
  (opposing_encounter_time : ℕ)
  (travel_time : ℕ) :
  num_cars = 80 →
  car_interval = 15 →
  (opposing_encounter_time * 2 * car_interval / travel_time) = num_cars →
  travel_time = 20 :=
by
  sorry

end NUMINAMATH_GPT_travel_time_between_resorts_l1934_193467


namespace NUMINAMATH_GPT_jackson_sandwiches_l1934_193423

noncomputable def total_sandwiches (weeks : ℕ) (miss_wed : ℕ) (miss_fri : ℕ) : ℕ :=
  let total_wednesdays := weeks - miss_wed
  let total_fridays := weeks - miss_fri
  total_wednesdays + total_fridays

theorem jackson_sandwiches : total_sandwiches 36 1 2 = 69 := by
  sorry

end NUMINAMATH_GPT_jackson_sandwiches_l1934_193423


namespace NUMINAMATH_GPT_fish_caught_together_l1934_193463

theorem fish_caught_together (Blaines_fish Keiths_fish : ℕ) 
  (h1 : Blaines_fish = 5) 
  (h2 : Keiths_fish = 2 * Blaines_fish) : 
  Blaines_fish + Keiths_fish = 15 := 
by 
  sorry

end NUMINAMATH_GPT_fish_caught_together_l1934_193463


namespace NUMINAMATH_GPT_num_parallelograms_4x6_grid_l1934_193411

noncomputable def numberOfParallelograms (m n : ℕ) : ℕ :=
  let numberOfRectangles := (Nat.choose (m + 1) 2) * (Nat.choose (n + 1) 2)
  let numberOfSquares := (m * n) + ((m - 1) * (n - 1)) + ((m - 2) * (n - 2)) + ((m - 3) * (n - 3))
  let numberOfRectanglesWithUnequalSides := numberOfRectangles - numberOfSquares
  2 * numberOfRectanglesWithUnequalSides

theorem num_parallelograms_4x6_grid : numberOfParallelograms 4 6 = 320 := by
  sorry

end NUMINAMATH_GPT_num_parallelograms_4x6_grid_l1934_193411


namespace NUMINAMATH_GPT_ab_equals_one_l1934_193425

theorem ab_equals_one {a b : ℝ} (h : a ≠ b) (hf : |Real.log a| = |Real.log b|) : a * b = 1 :=
  sorry

end NUMINAMATH_GPT_ab_equals_one_l1934_193425


namespace NUMINAMATH_GPT_minimum_slope_tangent_point_coordinates_l1934_193404

theorem minimum_slope_tangent_point_coordinates :
  ∃ a : ℝ, a > 0 ∧ (∀ x : ℝ, (2 * x + a / x ≥ 4) ∧ (2 * x + a / x = 4 ↔ x = 1)) → 
  (1, 1) = (1, 1) := by
sorry

end NUMINAMATH_GPT_minimum_slope_tangent_point_coordinates_l1934_193404


namespace NUMINAMATH_GPT_min_students_in_class_l1934_193438

noncomputable def min_possible_students (b g : ℕ) : Prop :=
  (3 * b) / 4 = 2 * (2 * g) / 3 ∧ b = (16 * g) / 9

theorem min_students_in_class : ∃ (b g : ℕ), min_possible_students b g ∧ b + g = 25 :=
by
  sorry

end NUMINAMATH_GPT_min_students_in_class_l1934_193438


namespace NUMINAMATH_GPT_PetyaColorsAll64Cells_l1934_193456

-- Assuming a type for representing cell coordinates
structure Cell where
  row : ℕ
  col : ℕ

def isColored (c : Cell) : Prop := true  -- All cells are colored
def LShapedFigures : Set (Set Cell) := sorry  -- Define what constitutes an L-shaped figure

theorem PetyaColorsAll64Cells :
  (∀ tilesVector ∈ LShapedFigures, ¬∀ cell ∈ tilesVector, isColored cell) → (∀ c : Cell, c.row < 8 ∧ c.col < 8 ∧ isColored c) := sorry

end NUMINAMATH_GPT_PetyaColorsAll64Cells_l1934_193456


namespace NUMINAMATH_GPT_problem_l1934_193407

theorem problem (x y z : ℕ) (h1 : xy + z = 56) (h2 : yz + x = 56) (h3 : zx + y = 56) : x + y + z = 21 :=
sorry

end NUMINAMATH_GPT_problem_l1934_193407
