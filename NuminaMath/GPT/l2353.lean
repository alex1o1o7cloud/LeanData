import Mathlib

namespace NUMINAMATH_GPT_sum_of_interior_angles_of_pentagon_l2353_235378

theorem sum_of_interior_angles_of_pentagon : (5 - 2) * 180 = 540 := 
by
  sorry

end NUMINAMATH_GPT_sum_of_interior_angles_of_pentagon_l2353_235378


namespace NUMINAMATH_GPT_find_a4_l2353_235331

variable {α : Type*} [Field α] [Inhabited α]

-- Definitions of the geometric sequence conditions
def geometric_sequence_condition1 (a₁ q : α) : Prop :=
  a₁ * (1 + q) = -1

def geometric_sequence_condition2 (a₁ q : α) : Prop :=
  a₁ * (1 - q^2) = -3

-- Definition of the geometric sequence
def geometric_sequence (a₁ q : α) (n : ℕ) : α :=
  a₁ * q^n

-- The theorem to be proven
theorem find_a4 (a₁ q : α) (h₁ : geometric_sequence_condition1 a₁ q) (h₂ : geometric_sequence_condition2 a₁ q) :
  geometric_sequence a₁ q 3 = -8 :=
  sorry

end NUMINAMATH_GPT_find_a4_l2353_235331


namespace NUMINAMATH_GPT_problem_Z_value_l2353_235364

def Z (a b : ℕ) : ℕ := 3 * (a - b) ^ 2

theorem problem_Z_value : Z 5 3 = 12 := by
  sorry

end NUMINAMATH_GPT_problem_Z_value_l2353_235364


namespace NUMINAMATH_GPT_one_fourth_of_six_point_three_as_fraction_l2353_235346

noncomputable def one_fourth_of_six_point_three_is_simplified : ℚ :=
  6.3 / 4

theorem one_fourth_of_six_point_three_as_fraction :
  one_fourth_of_six_point_three_is_simplified = 63 / 40 :=
by
  sorry

end NUMINAMATH_GPT_one_fourth_of_six_point_three_as_fraction_l2353_235346


namespace NUMINAMATH_GPT_total_surface_area_of_cylinder_l2353_235391

noncomputable def rectangle_length : ℝ := 4 * Real.pi
noncomputable def rectangle_width : ℝ := 2

noncomputable def cylinder_radius (length : ℝ) : ℝ := length / (2 * Real.pi)
noncomputable def cylinder_height (width : ℝ) : ℝ := width

noncomputable def cylinder_surface_area (radius height : ℝ) : ℝ :=
  2 * Real.pi * radius^2 + 2 * Real.pi * radius * height

theorem total_surface_area_of_cylinder :
  cylinder_surface_area (cylinder_radius rectangle_length) (cylinder_height rectangle_width) = 16 * Real.pi :=
by
  sorry

end NUMINAMATH_GPT_total_surface_area_of_cylinder_l2353_235391


namespace NUMINAMATH_GPT_find_constant_term_l2353_235386

theorem find_constant_term (c : ℤ) (y : ℤ) (h1 : y = 2) (h2 : 5 * y^2 - 8 * y + c = 59) : c = 55 :=
by
  sorry

end NUMINAMATH_GPT_find_constant_term_l2353_235386


namespace NUMINAMATH_GPT_solve_quadratic_l2353_235335

theorem solve_quadratic (x : ℝ) (h : (9 / x^2) - (6 / x) + 1 = 0) : 2 / x = 2 / 3 :=
by
  sorry

end NUMINAMATH_GPT_solve_quadratic_l2353_235335


namespace NUMINAMATH_GPT_kevin_feeds_each_toad_3_worms_l2353_235371

theorem kevin_feeds_each_toad_3_worms
  (num_toads : ℕ) (minutes_per_worm : ℕ) (hours_to_minutes : ℕ) (total_minutes : ℕ)
  (H1 : num_toads = 8)
  (H2 : minutes_per_worm = 15)
  (H3 : hours_to_minutes = 60)
  (H4 : total_minutes = 6 * hours_to_minutes)
  :
  total_minutes / minutes_per_worm / num_toads = 3 :=
sorry

end NUMINAMATH_GPT_kevin_feeds_each_toad_3_worms_l2353_235371


namespace NUMINAMATH_GPT_range_of_a_l2353_235328

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  a * Real.log (x + 1) - x^2

theorem range_of_a (a : ℝ) (p q : ℝ) (h₀ : p ≠ q) (h₁ : -1 < p ∧ p < 0) (h₂ : -1 < q ∧ q < 0) :
  (∀ p q : ℝ, -1 < p ∧ p < 0 → -1 < q ∧ q < 0 → p ≠ q → (f a (p + 1) - f a (q + 1)) / (p - q) > 1) ↔ (6 ≤ a) :=
by
  -- proof is omitted
  sorry

end NUMINAMATH_GPT_range_of_a_l2353_235328


namespace NUMINAMATH_GPT_line_relation_with_plane_l2353_235338

variables {P : Type} [Infinite P] [MetricSpace P]

variables (a b : Line P) (α : Plane P)

-- Conditions
axiom intersecting_lines : ∃ p : P, p ∈ a ∧ p ∈ b
axiom line_parallel_plane : ∀ p : P, p ∈ a → p ∈ α

-- Theorem statement for the proof problem
theorem line_relation_with_plane : (∀ p : P, p ∈ b → p ∈ α) ∨ (∃ q : P, q ∈ α ∧ q ∈ b) :=
sorry

end NUMINAMATH_GPT_line_relation_with_plane_l2353_235338


namespace NUMINAMATH_GPT_intersection_A_compl_B_subset_E_B_l2353_235368

namespace MathProof

-- Definitions
def A := {x : ℝ | (x + 3) * (x - 6) ≥ 0}
def B := {x : ℝ | (x + 2) / (x - 14) < 0}
def compl_R_B := {x : ℝ | x ≤ -2 ∨ x ≥ 14}
def E (a : ℝ) := {x : ℝ | 2 * a < x ∧ x < a + 1}

-- Theorem for intersection of A and complement of B
theorem intersection_A_compl_B : A ∩ compl_R_B = {x : ℝ | x ≤ -3 ∨ x ≥ 14} :=
by
  sorry

-- Theorem for subset relationship to determine range of a
theorem subset_E_B (a : ℝ) : (E a ⊆ B) → a ≥ -1 :=
by
  sorry

end MathProof

end NUMINAMATH_GPT_intersection_A_compl_B_subset_E_B_l2353_235368


namespace NUMINAMATH_GPT_cj_more_stamps_than_twice_kj_l2353_235399

variable (C K A : ℕ) (x : ℕ)

theorem cj_more_stamps_than_twice_kj :
  (C = 2 * K + x) →
  (K = A / 2) →
  (C + K + A = 930) →
  (A = 370) →
  (x = 25) →
  (C - 2 * K = 5) :=
by
  intros h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_cj_more_stamps_than_twice_kj_l2353_235399


namespace NUMINAMATH_GPT_num_prime_factors_30_fact_l2353_235355

noncomputable def factorial (n : ℕ) : ℕ :=
  if h : n = 0 then 1 else n * factorial (n - 1)

def is_prime (n : ℕ) : Bool :=
  if h : n ≤ 1 then false else
    let divisors := List.range (n - 2) |>.map (· + 2)
    !divisors.any (· ∣ n)

def primes_upto (n : ℕ) : List ℕ :=
  List.range (n - 1) |>.map (· + 1) |>.filter is_prime

def count_primes_factorial_upto (n : ℕ) : ℕ :=
  (primes_upto n).length

theorem num_prime_factors_30_fact : count_primes_factorial_upto 30 = 10 := sorry

end NUMINAMATH_GPT_num_prime_factors_30_fact_l2353_235355


namespace NUMINAMATH_GPT_benny_spent_on_baseball_gear_l2353_235315

theorem benny_spent_on_baseball_gear (initial_amount left_over spent : ℕ) 
  (h_initial : initial_amount = 67) 
  (h_left : left_over = 33) 
  (h_spent : spent = initial_amount - left_over) : 
  spent = 34 :=
by
  rw [h_initial, h_left] at h_spent
  exact h_spent

end NUMINAMATH_GPT_benny_spent_on_baseball_gear_l2353_235315


namespace NUMINAMATH_GPT_perfect_square_l2353_235396

theorem perfect_square (a b : ℝ) : a^2 + 2 * a * b + b^2 = (a + b)^2 := by
  sorry

end NUMINAMATH_GPT_perfect_square_l2353_235396


namespace NUMINAMATH_GPT_proof_2d_minus_r_l2353_235320

theorem proof_2d_minus_r (d r: ℕ) (h1 : 1059 % d = r)
  (h2 : 1482 % d = r) (h3 : 2340 % d = r) (hd : d > 1) : 2 * d - r = 6 := 
by 
  sorry

end NUMINAMATH_GPT_proof_2d_minus_r_l2353_235320


namespace NUMINAMATH_GPT_problem_1_problem_2_problem_3_l2353_235365

section MathProblems

variable (a b c m n x y : ℝ)
-- Problem 1
theorem problem_1 :
  (-6 * a^2 * b^5 * c) / (-2 * a * b^2)^2 = (3/2) * b * c := sorry

-- Problem 2
theorem problem_2 :
  (-3 * m - 2 * n) * (3 * m + 2 * n) = -9 * m^2 - 12 * m * n - 4 * n^2 := sorry

-- Problem 3
theorem problem_3 :
  ((x - 2 * y)^2 - (x - 2 * y) * (x + 2 * y)) / (2 * y) = -2 * x + 4 * y := sorry

end MathProblems

end NUMINAMATH_GPT_problem_1_problem_2_problem_3_l2353_235365


namespace NUMINAMATH_GPT_find_a_plus_b_l2353_235356
-- Definition of the problem variables and conditions
variables (a b : ℝ)
def condition1 : Prop := a - b = 3
def condition2 : Prop := a^2 - b^2 = -12

-- Goal: Prove that a + b = -4 given the conditions
theorem find_a_plus_b (h1 : condition1 a b) (h2 : condition2 a b) : a + b = -4 :=
  sorry

end NUMINAMATH_GPT_find_a_plus_b_l2353_235356


namespace NUMINAMATH_GPT_no_even_sum_of_four_consecutive_in_circle_l2353_235313

theorem no_even_sum_of_four_consecutive_in_circle (n : ℕ) (h1 : n = 2018) :
  ¬ ∃ (f : ℕ → ℕ), (∀ i, 1 ≤ f i ∧ f i ≤ n) ∧ (∀ i, i < n → (f (i % n) + f ((i + 1) % n) + f ((i + 2) % n) + f ((i + 3) % n)) % 2 = 1) :=
by { sorry }

end NUMINAMATH_GPT_no_even_sum_of_four_consecutive_in_circle_l2353_235313


namespace NUMINAMATH_GPT_find_integers_l2353_235321

theorem find_integers (n : ℕ) (h1 : n < 10^100)
  (h2 : n ∣ 2^n) (h3 : n - 1 ∣ 2^n - 1) (h4 : n - 2 ∣ 2^n - 2) :
  n = 2^2 ∨ n = 2^4 ∨ n = 2^16 ∨ n = 2^256 := by
  sorry

end NUMINAMATH_GPT_find_integers_l2353_235321


namespace NUMINAMATH_GPT_circle_passing_through_points_l2353_235351

noncomputable def parabola (x: ℝ) (a b: ℝ) : ℝ :=
  x^2 + a * x + b

theorem circle_passing_through_points (a b α β k: ℝ) :
  parabola 0 a b = b ∧ parabola α a b = 0 ∧ parabola β a b = 0 ∧
  ((0 - (α + β) / 2)^2 + (1 - k)^2 = ((α + β) / 2)^2 + (k - b)^2) →
  b = 1 :=
by
  sorry

end NUMINAMATH_GPT_circle_passing_through_points_l2353_235351


namespace NUMINAMATH_GPT_sqrt_meaningful_l2353_235318

theorem sqrt_meaningful (x : ℝ) : (x - 2 ≥ 0) ↔ (x ≥ 2) := by
  sorry

end NUMINAMATH_GPT_sqrt_meaningful_l2353_235318


namespace NUMINAMATH_GPT_top_angle_degrees_l2353_235350

def isosceles_triangle_with_angle_ratio (x : ℝ) (a b c : ℝ) : Prop :=
  a = x ∧ b = 4 * x ∧ a + b + c = 180 ∧ (a = b ∨ a = c ∨ b = c)

theorem top_angle_degrees (x : ℝ) (a b c : ℝ) :
  isosceles_triangle_with_angle_ratio x a b c → c = 20 ∨ c = 120 :=
by
  sorry

end NUMINAMATH_GPT_top_angle_degrees_l2353_235350


namespace NUMINAMATH_GPT_painted_cells_solutions_l2353_235314

def painted_cells (k l : ℕ) : ℕ := (2 * k + 1) * (2 * l + 1) - 74

theorem painted_cells_solutions : ∃ k l : ℕ, k * l = 74 ∧ (painted_cells k l = 373 ∨ painted_cells k l = 301) :=
by
  sorry

end NUMINAMATH_GPT_painted_cells_solutions_l2353_235314


namespace NUMINAMATH_GPT_tangent_line_at_pi_one_l2353_235381

noncomputable def function (x : ℝ) : ℝ := Real.exp x * Real.sin x + 1
noncomputable def tangent_line (x : ℝ) (y : ℝ) : ℝ := x * Real.exp Real.pi + y - 1 - Real.pi * Real.exp Real.pi

theorem tangent_line_at_pi_one :
  tangent_line x y = 0 ↔ y = function x → x = Real.pi ∧ y = 1 :=
by
  sorry

end NUMINAMATH_GPT_tangent_line_at_pi_one_l2353_235381


namespace NUMINAMATH_GPT_slices_eaten_l2353_235354

theorem slices_eaten (slices_cheese : ℕ) (slices_pepperoni : ℕ) (slices_left_per_person : ℕ) (phil_andre_slices_left : ℕ) :
  (slices_cheese + slices_pepperoni = 22) →
  (slices_left_per_person = 2) →
  (phil_andre_slices_left = 2 + 2) →
  (slices_cheese + slices_pepperoni - phil_andre_slices_left = 18) :=
by
  intros
  sorry

end NUMINAMATH_GPT_slices_eaten_l2353_235354


namespace NUMINAMATH_GPT_probability_equals_two_thirds_l2353_235389

-- Definitions for total arrangements and favorable arrangements
def total_arrangements : ℕ := Nat.choose 6 2
def favorable_arrangements : ℕ := Nat.choose 5 2

-- Probability that 2 zeros are not adjacent
def probability_not_adjacent : ℚ := favorable_arrangements / total_arrangements

theorem probability_equals_two_thirds : probability_not_adjacent = 2 / 3 := 
by 
  let total_arrangements := 15
  let favorable_arrangements := 10
  have h1 : probability_not_adjacent = (10 : ℚ) / (15 : ℚ) := rfl
  have h2 : (10 : ℚ) / (15 : ℚ) = 2 / 3 := by norm_num
  exact Eq.trans h1 h2 

end NUMINAMATH_GPT_probability_equals_two_thirds_l2353_235389


namespace NUMINAMATH_GPT_min_cost_to_package_fine_arts_collection_l2353_235300

theorem min_cost_to_package_fine_arts_collection :
  let box_length := 20
  let box_width := 20
  let box_height := 12
  let cost_per_box := 0.50
  let required_volume := 1920000
  let volume_of_one_box := box_length * box_width * box_height
  let number_of_boxes := required_volume / volume_of_one_box
  let total_cost := number_of_boxes * cost_per_box
  total_cost = 200 := 
by
  sorry

end NUMINAMATH_GPT_min_cost_to_package_fine_arts_collection_l2353_235300


namespace NUMINAMATH_GPT_max_angle_line_plane_l2353_235327

theorem max_angle_line_plane (θ : ℝ) (h_angle : θ = 72) :
  ∃ φ : ℝ, φ = 90 ∧ (72 ≤ φ ∧ φ ≤ 90) :=
by sorry

end NUMINAMATH_GPT_max_angle_line_plane_l2353_235327


namespace NUMINAMATH_GPT_find_s_l2353_235387

theorem find_s (k s : ℝ) (h1 : 5 = k * 2^s) (h2 : 45 = k * 8^s) : s = (Real.log 9) / (2 * Real.log 2) :=
by
  sorry

end NUMINAMATH_GPT_find_s_l2353_235387


namespace NUMINAMATH_GPT_minimum_value_a_2b_3c_l2353_235326

theorem minimum_value_a_2b_3c (a b c : ℝ)
  (h : ∀ x y : ℝ, x + 2*y - 3 ≤ a*x + b*y + c ∧ a*x + b*y + c ≤ x + 2*y + 3) :
  (a + 2*b - 3*c) = -4 :=
sorry

end NUMINAMATH_GPT_minimum_value_a_2b_3c_l2353_235326


namespace NUMINAMATH_GPT_multiply_polynomials_l2353_235369

open Polynomial

variable {R : Type*} [CommRing R]

theorem multiply_polynomials (x : R) :
  (x^4 + 6*x^2 + 9) * (x^2 - 3) = x^4 + 6*x^2 :=
  sorry

end NUMINAMATH_GPT_multiply_polynomials_l2353_235369


namespace NUMINAMATH_GPT_no_five_integer_solutions_divisibility_condition_l2353_235357

variables (k : ℤ) 

-- Definition of equation
def equation (x y : ℤ) : Prop :=
  y^2 - k = x^3

-- Variables to capture the integer solutions
variables (x1 x2 x3 x4 x5 y1 : ℤ)

-- Prove that there do not exist five solutions satisfying the given forms
theorem no_five_integer_solutions :
  ¬(equation k x1 y1 ∧ 
    equation k x2 (y1 - 1) ∧ 
    equation k x3 (y1 - 2) ∧ 
    equation k x4 (y1 - 3) ∧ 
    equation k x5 (y1 - 4)) :=
sorry

-- Prove divisibility condition for the first four solutions
theorem divisibility_condition :
  (equation k x1 y1 ∧ 
   equation k x2 (y1 - 1) ∧ 
   equation k x3 (y1 - 2) ∧ 
   equation k x4 (y1 - 3)) → 
  63 ∣ (k - 17) :=
sorry

end NUMINAMATH_GPT_no_five_integer_solutions_divisibility_condition_l2353_235357


namespace NUMINAMATH_GPT_train_ride_time_in_hours_l2353_235339

-- Definition of conditions
def lukes_total_trip_time_hours : ℕ := 8
def bus_ride_minutes : ℕ := 75
def walk_to_train_center_minutes : ℕ := 15
def wait_time_minutes : ℕ := 2 * walk_to_train_center_minutes

-- Convert total trip time to minutes
def lukes_total_trip_time_minutes : ℕ := lukes_total_trip_time_hours * 60

-- Calculate the total time spent on bus, walking, and waiting
def bus_walk_wait_time_minutes : ℕ :=
  bus_ride_minutes + walk_to_train_center_minutes + wait_time_minutes

-- Calculate the train ride time in minutes
def train_ride_time_minutes : ℕ :=
  lukes_total_trip_time_minutes - bus_walk_wait_time_minutes

-- Prove the train ride time in hours
theorem train_ride_time_in_hours : train_ride_time_minutes / 60 = 6 :=
by
  sorry

end NUMINAMATH_GPT_train_ride_time_in_hours_l2353_235339


namespace NUMINAMATH_GPT_min_value_of_function_l2353_235392

theorem min_value_of_function : 
  ∃ x > 2, ∀ y > 2, (y + 1 / (y - 2)) ≥ 4 ∧ (x + 1 / (x - 2)) = 4 := 
by sorry

end NUMINAMATH_GPT_min_value_of_function_l2353_235392


namespace NUMINAMATH_GPT_salary_january_l2353_235340

theorem salary_january
  (J F M A May : ℝ)  -- declare the salaries as real numbers
  (h1 : (J + F + M + A) / 4 = 8000)  -- condition 1
  (h2 : (F + M + A + May) / 4 = 9500)  -- condition 2
  (h3 : May = 6500) :  -- condition 3
  J = 500 := 
by
  sorry

end NUMINAMATH_GPT_salary_january_l2353_235340


namespace NUMINAMATH_GPT_johns_gym_time_l2353_235359

noncomputable def time_spent_at_gym (day : String) : ℝ :=
  match day with
  | "Monday" => 1 + 0.5
  | "Tuesday" => 40/60 + 20/60 + 15/60
  | "Thursday" => 40/60 + 20/60 + 15/60
  | "Saturday" => 1.5 + 0.75
  | "Sunday" => 10/60 + 50/60 + 10/60
  | _ => 0

noncomputable def total_hours_per_week : ℝ :=
  time_spent_at_gym "Monday" 
  + 2 * time_spent_at_gym "Tuesday" 
  + time_spent_at_gym "Saturday" 
  + time_spent_at_gym "Sunday"

theorem johns_gym_time : total_hours_per_week = 7.4167 := by
  sorry

end NUMINAMATH_GPT_johns_gym_time_l2353_235359


namespace NUMINAMATH_GPT_problem_1_problem_2_l2353_235358

noncomputable def f (x : ℝ) := Real.sin x + (x - 1) / Real.exp x

theorem problem_1 (x : ℝ) (h₀ : x ∈ Set.Icc (-Real.pi) (Real.pi / 2)) :
  MonotoneOn f (Set.Icc (-Real.pi) (Real.pi / 2)) :=
sorry

theorem problem_2 (k : ℝ) :
  ∀ x ∈ Set.Icc (-Real.pi) 0, ((f x - Real.sin x) * Real.exp x - Real.cos x) ≤ k * Real.sin x → 
  k ∈ Set.Iic (1 + Real.pi / 2) :=
sorry

end NUMINAMATH_GPT_problem_1_problem_2_l2353_235358


namespace NUMINAMATH_GPT_annual_income_earned_by_both_investments_l2353_235312

noncomputable def interest (principal: ℝ) (rate: ℝ) (time: ℝ) : ℝ :=
  principal * rate * time

theorem annual_income_earned_by_both_investments :
  let total_amount := 8000
  let first_investment := 3000
  let first_interest_rate := 0.085
  let second_interest_rate := 0.064
  let second_investment := total_amount - first_investment
  interest first_investment first_interest_rate 1 + interest second_investment second_interest_rate 1 = 575 :=
by
  sorry

end NUMINAMATH_GPT_annual_income_earned_by_both_investments_l2353_235312


namespace NUMINAMATH_GPT_complement_intersection_eq_l2353_235379

variable (U P Q : Set ℕ)
variable (hU : U = {1, 2, 3})
variable (hP : P = {1, 2})
variable (hQ : Q = {2, 3})

theorem complement_intersection_eq : 
  (U \ (P ∩ Q)) = {1, 3} := by
  sorry

end NUMINAMATH_GPT_complement_intersection_eq_l2353_235379


namespace NUMINAMATH_GPT_common_chord_and_length_l2353_235333

-- Define the two circles
def circle1 (x y : ℝ) := x^2 + y^2 + 2*x - 4*y - 5 = 0
def circle2 (x y : ℝ) := x^2 + y^2 + 2*x - 1 = 0

-- The theorem statement with the conditions and expected solutions
theorem common_chord_and_length :
  (∀ x y : ℝ, circle1 x y ∧ circle2 x y → y = -1)
  ∧
  (∃ A B : (ℝ × ℝ), (circle1 A.1 A.2 ∧ circle2 A.1 A.2) ∧ 
                    (circle1 B.1 B.2 ∧ circle2 B.1 B.2) ∧ 
                    (|A.1 - B.1|^2 + |A.2 - B.2|^2 = 4)) :=
by
  sorry

end NUMINAMATH_GPT_common_chord_and_length_l2353_235333


namespace NUMINAMATH_GPT_find_numbers_l2353_235361

-- Define the conditions
def condition_1 (L S : ℕ) : Prop := L - S = 8327
def condition_2 (L S : ℕ) : Prop := ∃ q r, L = q * S + r ∧ q = 21 ∧ r = 125

-- Define the math proof problem
theorem find_numbers (S L : ℕ) (h1 : condition_1 L S) (h2 : condition_2 L S) : S = 410 ∧ L = 8735 :=
by
  sorry

end NUMINAMATH_GPT_find_numbers_l2353_235361


namespace NUMINAMATH_GPT_sum_m_n_l2353_235390

open Real

noncomputable def f (x : ℝ) : ℝ := |log x / log 2|

theorem sum_m_n (m n : ℝ) (hm_pos : 0 < m) (hn_pos : 0 < n) (h_mn : m < n) 
  (h_f_eq : f m = f n) (h_max_f : ∀ x : ℝ, m^2 ≤ x ∧ x ≤ n → f x ≤ 2) :
  m + n = 5 / 2 :=
sorry

end NUMINAMATH_GPT_sum_m_n_l2353_235390


namespace NUMINAMATH_GPT_circle_tangent_to_parabola_directrix_l2353_235360

theorem circle_tangent_to_parabola_directrix (m : ℝ) :
  (∀ x y : ℝ, x^2 + y^2 + m * x - 1/4 = 0 → y^2 = 4 * x → x = -1) → m = 3/4 :=
by
  sorry

end NUMINAMATH_GPT_circle_tangent_to_parabola_directrix_l2353_235360


namespace NUMINAMATH_GPT_john_unanswered_problems_is_9_l2353_235375

variables (x y z : ℕ)

theorem john_unanswered_problems_is_9 (h1 : 5 * x + 2 * z = 93)
                                      (h2 : 4 * x - y = 54)
                                      (h3 : x + y + z = 30) : 
  z = 9 :=
by 
  sorry

end NUMINAMATH_GPT_john_unanswered_problems_is_9_l2353_235375


namespace NUMINAMATH_GPT_number_of_girls_l2353_235349

open Rat

theorem number_of_girls 
  (G B : ℕ) 
  (h1 : G / B = 5 / 8)
  (h2 : G + B = 300) 
  : G = 116 := 
by
  sorry

end NUMINAMATH_GPT_number_of_girls_l2353_235349


namespace NUMINAMATH_GPT_max_sum_integers_differ_by_60_l2353_235353

theorem max_sum_integers_differ_by_60 (b : ℕ) (c : ℕ) (h_diff : 0 < b) (h_sqrt : (Nat.sqrt b : ℝ) + (Nat.sqrt (b + 60) : ℝ) = (Nat.sqrt c : ℝ)) (h_not_square : ¬ ∃ (k : ℕ), k * k = c) :
  ∃ (b : ℕ), b + (b + 60) = 156 := 
sorry

end NUMINAMATH_GPT_max_sum_integers_differ_by_60_l2353_235353


namespace NUMINAMATH_GPT_relationship_between_exponents_l2353_235332

theorem relationship_between_exponents 
  (p r : ℝ) (u v s t m n : ℝ)
  (h1 : p^u = r^s)
  (h2 : r^v = p^t)
  (h3 : m = r^s)
  (h4 : n = r^v)
  (h5 : m^2 = n^3) :
  (s / u = v / t) ∧ (2 * s = 3 * v) :=
  by
  sorry

end NUMINAMATH_GPT_relationship_between_exponents_l2353_235332


namespace NUMINAMATH_GPT_avg_three_numbers_l2353_235306

theorem avg_three_numbers (A B C : ℝ) 
  (h1 : A + B = 53)
  (h2 : B + C = 69)
  (h3 : A + C = 58) : 
  (A + B + C) / 3 = 30 := 
by
  sorry

end NUMINAMATH_GPT_avg_three_numbers_l2353_235306


namespace NUMINAMATH_GPT_eulers_formula_l2353_235344

structure PlanarGraph :=
(vertices : ℕ)
(edges : ℕ)
(faces : ℕ)
(connected : Prop)

theorem eulers_formula (G: PlanarGraph) (H_conn: G.connected) : G.vertices - G.edges + G.faces = 2 :=
sorry

end NUMINAMATH_GPT_eulers_formula_l2353_235344


namespace NUMINAMATH_GPT_simplify_expression_l2353_235383

theorem simplify_expression (w : ℝ) :
  2 * w^2 + 3 - 4 * w^2 + 2 * w - 6 * w + 4 = -2 * w^2 - 4 * w + 7 :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l2353_235383


namespace NUMINAMATH_GPT_no_distinct_positive_integers_l2353_235304

noncomputable def P (x : ℕ) : ℕ := x^2000 - x^1000 + 1

theorem no_distinct_positive_integers (a : Fin 2001 → ℕ) (h_distinct : Function.Injective a) :
  ¬ (∀ i j, i ≠ j → a i * a j ∣ P (a i) * P (a j)) :=
sorry

end NUMINAMATH_GPT_no_distinct_positive_integers_l2353_235304


namespace NUMINAMATH_GPT_smallest_positive_solution_eq_sqrt_29_l2353_235388

theorem smallest_positive_solution_eq_sqrt_29 :
  ∃ x : ℝ, 0 < x ∧ x^4 - 58 * x^2 + 841 = 0 ∧ x = Real.sqrt 29 :=
by
  sorry

end NUMINAMATH_GPT_smallest_positive_solution_eq_sqrt_29_l2353_235388


namespace NUMINAMATH_GPT_pet_store_cages_l2353_235316

theorem pet_store_cages (initial_puppies sold_puppies puppies_per_cage remaining_puppies num_cages : ℕ)
  (h1 : initial_puppies = 102) 
  (h2 : sold_puppies = 21) 
  (h3 : puppies_per_cage = 9) 
  (h4 : remaining_puppies = initial_puppies - sold_puppies)
  (h5 : num_cages = remaining_puppies / puppies_per_cage) : 
  num_cages = 9 := 
by
  sorry

end NUMINAMATH_GPT_pet_store_cages_l2353_235316


namespace NUMINAMATH_GPT_Greg_gold_amount_l2353_235385

noncomputable def gold_amounts (G K : ℕ) : Prop :=
  G = K / 4 ∧ G + K = 100

theorem Greg_gold_amount (G K : ℕ) (h : gold_amounts G K) : G = 20 := 
by
  sorry

end NUMINAMATH_GPT_Greg_gold_amount_l2353_235385


namespace NUMINAMATH_GPT_wendy_full_face_time_l2353_235317

-- Define the constants based on the conditions
def num_products := 5
def wait_time := 5
def makeup_time := 30

-- Calculate the total time to put on "full face"
def total_time (products : ℕ) (wait_time : ℕ) (makeup_time : ℕ) : ℕ :=
  (products - 1) * wait_time + makeup_time

-- The theorem stating that Wendy's full face routine takes 50 minutes
theorem wendy_full_face_time : total_time num_products wait_time makeup_time = 50 :=
by {
  -- the proof would be provided here, for now we use sorry
  sorry
}

end NUMINAMATH_GPT_wendy_full_face_time_l2353_235317


namespace NUMINAMATH_GPT_solve_for_k_l2353_235366

-- Definition and conditions
def ellipse_eq (k : ℝ) : Prop := ∀ x y, k * x^2 + 5 * y^2 = 5

-- Problem: Prove k = 1 given the above definitions
theorem solve_for_k (k : ℝ) :
  (exists (x y : ℝ), ellipse_eq k ∧ x = 2 ∧ y = 0) -> k = 1 :=
sorry

end NUMINAMATH_GPT_solve_for_k_l2353_235366


namespace NUMINAMATH_GPT_winning_candidate_votes_l2353_235302

-- Define the conditions as hypotheses in Lean.
def two_candidates (candidates : ℕ) : Prop := candidates = 2
def winner_received_62_percent (V : ℝ) (votes_winner : ℝ) : Prop := votes_winner = 0.62 * V
def winning_margin (V : ℝ) : Prop := 0.24 * V = 384

-- The main theorem to prove: the winner candidate received 992 votes.
theorem winning_candidate_votes (V votes_winner : ℝ) (candidates : ℕ) 
  (h1 : two_candidates candidates) 
  (h2 : winner_received_62_percent V votes_winner)
  (h3 : winning_margin V) : 
  votes_winner = 992 :=
by
  sorry

end NUMINAMATH_GPT_winning_candidate_votes_l2353_235302


namespace NUMINAMATH_GPT_difference_is_20_l2353_235322

def x : ℕ := 10

def a : ℕ := 3 * x

def b : ℕ := 20 - x

theorem difference_is_20 : a - b = 20 := 
by 
  sorry

end NUMINAMATH_GPT_difference_is_20_l2353_235322


namespace NUMINAMATH_GPT_min_value_of_one_over_a_plus_one_over_b_l2353_235373

theorem min_value_of_one_over_a_plus_one_over_b (a b : ℝ) (h₀ : a > 0) (h₁ : b > 0) (h₂ : a + b = 12) :
  (∃ c : ℝ, (c = 1/ a + 1 / b) ∧ c = 1 / 3) :=
sorry

end NUMINAMATH_GPT_min_value_of_one_over_a_plus_one_over_b_l2353_235373


namespace NUMINAMATH_GPT_no_odd_total_given_ratio_l2353_235342

theorem no_odd_total_given_ratio (T : ℕ) (hT1 : 50 < T) (hT2 : T < 150) (hT3 : T % 2 = 1) : 
  ∀ (B : ℕ), T ≠ 8 * B + B / 4 :=
sorry

end NUMINAMATH_GPT_no_odd_total_given_ratio_l2353_235342


namespace NUMINAMATH_GPT_fish_minimum_catch_l2353_235397

theorem fish_minimum_catch (a1 a2 a3 a4 a5 : ℕ) (h_sum : a1 + a2 + a3 + a4 + a5 = 100)
  (h_non_increasing : a1 ≥ a2 ∧ a2 ≥ a3 ∧ a3 ≥ a4 ∧ a4 ≥ a5) : 
  a1 + a3 + a5 ≥ 50 :=
sorry

end NUMINAMATH_GPT_fish_minimum_catch_l2353_235397


namespace NUMINAMATH_GPT_find_c_in_triangle_l2353_235323

theorem find_c_in_triangle
  (A : Real) (a b S : Real) (c : Real)
  (hA : A = 60) 
  (ha : a = 6 * Real.sqrt 3)
  (hb : b = 12)
  (hS : S = 18 * Real.sqrt 3) :
  c = 6 := by
  sorry

end NUMINAMATH_GPT_find_c_in_triangle_l2353_235323


namespace NUMINAMATH_GPT_largest_n_for_divisibility_l2353_235352

theorem largest_n_for_divisibility (n : ℕ) (h : (n + 20) ∣ (n^3 + 1000)) : n ≤ 180 := 
sorry

example : ∃ n : ℕ, (n + 20) ∣ (n^3 + 1000) ∧ n = 180 :=
by
  use 180
  sorry

end NUMINAMATH_GPT_largest_n_for_divisibility_l2353_235352


namespace NUMINAMATH_GPT_min_value_l2353_235348

theorem min_value (a b c x y z : ℝ) (h1 : a + b + c = 1) (h2 : x + y + z = 1) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) :
  ∃ val : ℝ, val = -1 / 4 ∧ ∀ a b c x y z : ℝ, 0 ≤ a → 0 ≤ b → 0 ≤ c → 0 ≤ x → 0 ≤ y → 0 ≤ z → a + b + c = 1 → x + y + z = 1 → (a - x^2) * (b - y^2) * (c - z^2) ≥ val :=
sorry

end NUMINAMATH_GPT_min_value_l2353_235348


namespace NUMINAMATH_GPT_eighty_percent_of_number_l2353_235334

theorem eighty_percent_of_number (x : ℝ) (h : 0.20 * x = 60) : 0.80 * x = 240 := 
by sorry

end NUMINAMATH_GPT_eighty_percent_of_number_l2353_235334


namespace NUMINAMATH_GPT_average_runs_in_second_set_l2353_235395

theorem average_runs_in_second_set
  (avg_first_set : ℕ → ℕ → ℕ)
  (avg_all_matches : ℕ → ℕ → ℕ)
  (avg1 : ℕ := avg_first_set 20 30)
  (avg2 : ℕ := avg_all_matches 30 25) :
  ∃ (A : ℕ), A = 15 := by
  sorry

end NUMINAMATH_GPT_average_runs_in_second_set_l2353_235395


namespace NUMINAMATH_GPT_medieval_society_hierarchy_l2353_235362

-- Given conditions
def members := 12
def king_choices := members
def remaining_after_king := members - 1
def duke_choices : ℕ := remaining_after_king * (remaining_after_king - 1) * (remaining_after_king - 2)
def knight_choices : ℕ := Nat.choose (remaining_after_king - 2) 2 * Nat.choose (remaining_after_king - 4) 2 * Nat.choose (remaining_after_king - 6) 2

-- The number of ways to establish the hierarchy can be stated as:
def total_ways : ℕ := king_choices * duke_choices * knight_choices

-- Our main theorem
theorem medieval_society_hierarchy : total_ways = 907200 := by
  -- Proof would go here, we skip it with sorry
  sorry

end NUMINAMATH_GPT_medieval_society_hierarchy_l2353_235362


namespace NUMINAMATH_GPT_dagger_computation_l2353_235336

def dagger (m n p q : ℕ) (hn : n ≠ 0) (hm : m ≠ 0) : ℚ :=
  (m^2 * p * (q / n)) + ((p : ℚ) / m)

theorem dagger_computation :
  dagger 5 9 6 2 (by norm_num) (by norm_num) = 518 / 15 :=
sorry

end NUMINAMATH_GPT_dagger_computation_l2353_235336


namespace NUMINAMATH_GPT_largest_value_expression_l2353_235301

theorem largest_value_expression (a b c : ℝ) (ha : a ∈ ({1, 2, 4} : Set ℝ)) (hb : b ∈ ({1, 2, 4} : Set ℝ)) (hc : c ∈ ({1, 2, 4} : Set ℝ)) (habc_diff : a ≠ b ∧ b ≠ c ∧ a ≠ c) :
  (a / 2) / (b / c) ≤ 4 :=
sorry

end NUMINAMATH_GPT_largest_value_expression_l2353_235301


namespace NUMINAMATH_GPT_regular_polygon_sides_l2353_235347

theorem regular_polygon_sides (n : ℕ) (h1 : 2 ≤ n) (h2 : (n - 2) * 180 / n = 120) : n = 6 :=
by
  sorry

end NUMINAMATH_GPT_regular_polygon_sides_l2353_235347


namespace NUMINAMATH_GPT_cookie_distribution_l2353_235374

def trays := 4
def cookies_per_tray := 24
def total_cookies := trays * cookies_per_tray
def packs := 8
def cookies_per_pack := total_cookies / packs

theorem cookie_distribution : cookies_per_pack = 12 := by
  sorry

end NUMINAMATH_GPT_cookie_distribution_l2353_235374


namespace NUMINAMATH_GPT_game_show_possible_guesses_l2353_235308

theorem game_show_possible_guesses : 
  (∃ A B C : ℕ, 
    A + B + C = 8 ∧ 
    A > 0 ∧ B > 0 ∧ C > 0 ∧ 
    (A = 1 ∨ A = 4) ∧
    (B = 1 ∨ B = 4) ∧
    (C = 1 ∨ C = 4) ) →
  (number_of_possible_guesses : ℕ) = 210 :=
sorry

end NUMINAMATH_GPT_game_show_possible_guesses_l2353_235308


namespace NUMINAMATH_GPT_proof_ineq_l2353_235311

noncomputable def P (f g : ℤ → ℤ) (m n k : ℕ) :=
  (∀ x y : ℤ, -1000 ≤ x ∧ x ≤ 1000 ∧ -1000 ≤ y ∧ y ≤ 1000 ∧ f x = g y → m = m + 1) ∧
  (∀ x y : ℤ, -1000 ≤ x ∧ x ≤ 1000 ∧ -1000 ≤ y ∧ y ≤ 1000 ∧ f x = f y → n = n + 1) ∧
  (∀ x y : ℤ, -1000 ≤ x ∧ x ≤ 1000 ∧ -1000 ≤ y ∧ y ≤ 1000 ∧ g x = g y → k = k + 1)

theorem proof_ineq (f g : ℤ → ℤ) (m n k : ℕ) (h : P f g m n k) : 
  2 * m ≤ n + k :=
  sorry

end NUMINAMATH_GPT_proof_ineq_l2353_235311


namespace NUMINAMATH_GPT_lcm_28_72_l2353_235382

theorem lcm_28_72 : Nat.lcm 28 72 = 504 := by
  sorry

end NUMINAMATH_GPT_lcm_28_72_l2353_235382


namespace NUMINAMATH_GPT_ned_initial_lives_l2353_235309

-- Define the initial number of lives Ned had
def initial_lives (start_lives current_lives lost_lives : ℕ) : ℕ :=
  current_lives + lost_lives

-- Define the conditions
def current_lives := 70
def lost_lives := 13

-- State the theorem
theorem ned_initial_lives : initial_lives current_lives current_lives lost_lives = 83 := by
  sorry

end NUMINAMATH_GPT_ned_initial_lives_l2353_235309


namespace NUMINAMATH_GPT_sqrt_fourth_root_l2353_235337

theorem sqrt_fourth_root (h : Real.sqrt (Real.sqrt (0.00000081)) = 0.1732) : Real.sqrt (Real.sqrt (0.00000081)) = 0.2 :=
by
  sorry

end NUMINAMATH_GPT_sqrt_fourth_root_l2353_235337


namespace NUMINAMATH_GPT_extra_cost_from_online_purchase_l2353_235329

-- Define the in-store price
def inStorePrice : ℝ := 150.00

-- Define the online payment and processing fee
def onlinePayment : ℝ := 35.00
def processingFee : ℝ := 12.00

-- Calculate the total online cost
def totalOnlineCost : ℝ := (4 * onlinePayment) + processingFee

-- Calculate the difference in cents
def differenceInCents : ℝ := (totalOnlineCost - inStorePrice) * 100

-- The proof statement
theorem extra_cost_from_online_purchase : differenceInCents = 200 :=
by
  -- Proof steps go here
  sorry

end NUMINAMATH_GPT_extra_cost_from_online_purchase_l2353_235329


namespace NUMINAMATH_GPT_arithmetic_sequence_product_l2353_235372

theorem arithmetic_sequence_product (b : ℕ → ℤ) (h1 : ∀ n, b (n + 1) = b n + d) 
  (h2 : b 5 * b 6 = 35) : b 4 * b 7 = 27 :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_product_l2353_235372


namespace NUMINAMATH_GPT_inverse_function_fixed_point_l2353_235341

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^(x-2) - 1

theorem inverse_function_fixed_point
  (a : ℝ) (h1 : 0 < a) (h2 : a ≠ 1) :
  ∃ g : ℝ → ℝ, (∀ y : ℝ, g (f a y) = y) ∧ g 0 = 2 :=
sorry

end NUMINAMATH_GPT_inverse_function_fixed_point_l2353_235341


namespace NUMINAMATH_GPT_fraction_equiv_l2353_235384

theorem fraction_equiv (x y : ℚ) (h : (5/6) * 192 = (x/y) * 192 + 100) : x/y = 5/16 :=
sorry

end NUMINAMATH_GPT_fraction_equiv_l2353_235384


namespace NUMINAMATH_GPT_max_digits_product_l2353_235307

def digitsProduct (A B : ℕ) : ℕ := A * B

theorem max_digits_product 
  (A B : ℕ) 
  (h1 : A + B + 5 ≡ 0 [MOD 9]) 
  (h2 : 0 ≤ A ∧ A ≤ 9) 
  (h3 : 0 ≤ B ∧ B ≤ 9) 
  : digitsProduct A B = 42 := 
sorry

end NUMINAMATH_GPT_max_digits_product_l2353_235307


namespace NUMINAMATH_GPT_circle_equation_l2353_235330

-- Definitions based on the conditions
def center_on_x_axis (a b r : ℝ) := b = 0
def tangent_at_point (a b r : ℝ) := (b - 1) / a = -1/2

-- Proof statement
theorem circle_equation (a b r : ℝ) (h1: center_on_x_axis a b r) (h2: tangent_at_point a b r) :
    ∃ (a b r : ℝ), (x - a)^2 + y^2 = r^2 ∧ a = 2 ∧ b = 0 ∧ r^2 = 5 :=
by 
  sorry

end NUMINAMATH_GPT_circle_equation_l2353_235330


namespace NUMINAMATH_GPT_find_bc_div_a_l2353_235370

noncomputable def f (x : ℝ) : ℝ := 3 * Real.sin x + 2 * Real.cos x + 1

variable (a b c : ℝ)

def satisfied (x : ℝ) : Prop := a * f x + b * f (x - c) = 1

theorem find_bc_div_a (ha : ∀ x, satisfied a b c x) : (b * Real.cos c / a) = -1 := 
by sorry

end NUMINAMATH_GPT_find_bc_div_a_l2353_235370


namespace NUMINAMATH_GPT_geometric_sequence_fifth_term_l2353_235398

theorem geometric_sequence_fifth_term
  (a : ℕ) (r : ℕ)
  (h₁ : a = 3)
  (h₂ : a * r^3 = 243) :
  a * r^4 = 243 :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_fifth_term_l2353_235398


namespace NUMINAMATH_GPT_bernardo_larger_probability_l2353_235319

-- Mathematical definitions
def bernardo_set : Finset ℕ := {1,2,3,4,5,6,7,8,10}
def silvia_set : Finset ℕ := {1,2,3,4,5,6}

-- Probability calculation function (you need to define the detailed implementation)
noncomputable def probability_bernardo_gt_silvia : ℚ := sorry

-- The proof statement
theorem bernardo_larger_probability : 
  probability_bernardo_gt_silvia = 13 / 20 :=
sorry

end NUMINAMATH_GPT_bernardo_larger_probability_l2353_235319


namespace NUMINAMATH_GPT_base_6_to_base_10_exact_value_l2353_235394

def base_6_to_base_10 (n : ℕ) : ℕ :=
  1 * 6^2 + 5 * 6^1 + 4 * 6^0

theorem base_6_to_base_10_exact_value : base_6_to_base_10 154 = 70 := by
  rfl

end NUMINAMATH_GPT_base_6_to_base_10_exact_value_l2353_235394


namespace NUMINAMATH_GPT_difference_between_possible_x_values_l2353_235377

theorem difference_between_possible_x_values :
  ∀ (x : ℝ), (x + 3) ^ 2 / (2 * x + 15) = 3 → (x = 6 ∨ x = -6) →
  (abs (6 - (-6)) = 12) :=
by
  intro x h1 h2
  sorry

end NUMINAMATH_GPT_difference_between_possible_x_values_l2353_235377


namespace NUMINAMATH_GPT_diff_not_equal_l2353_235345

variable (A B : Set ℕ)

def diff (A B : Set ℕ) : Set ℕ := {x | x ∈ A ∧ x ∉ B}

theorem diff_not_equal (A B : Set ℕ) :
  A ≠ ∅ ∧ B ≠ ∅ → (diff A B ≠ diff B A) :=
by
  sorry

end NUMINAMATH_GPT_diff_not_equal_l2353_235345


namespace NUMINAMATH_GPT_shoe_size_percentage_difference_l2353_235324

theorem shoe_size_percentage_difference :
  ∀ (size8_len size15_len size17_len : ℝ)
  (h1 : size8_len = size15_len - (7 * (1 / 5)))
  (h2 : size17_len = size15_len + (2 * (1 / 5)))
  (h3 : size15_len = 10.4),
  ((size17_len - size8_len) / size8_len) * 100 = 20 := by
  intros size8_len size15_len size17_len h1 h2 h3
  sorry

end NUMINAMATH_GPT_shoe_size_percentage_difference_l2353_235324


namespace NUMINAMATH_GPT_steiner_ellipse_equation_l2353_235367

theorem steiner_ellipse_equation
  (α β γ : ℝ) 
  (h : α + β + γ = 1) :
  β * γ + α * γ + α * β = 0 := 
sorry

end NUMINAMATH_GPT_steiner_ellipse_equation_l2353_235367


namespace NUMINAMATH_GPT_minutes_before_4_angle_same_as_4_l2353_235310

def hour_hand_angle_at_4 := 120
def minute_hand_angle_at_4 := 0
def minute_hand_angle_per_minute := 6
def hour_hand_angle_per_minute := 0.5

theorem minutes_before_4_angle_same_as_4 :
  ∃ m : ℚ, abs (hour_hand_angle_at_4 - 5.5 * m) = hour_hand_angle_at_4 ∧ 
           (60 - m) = 21 + 9 / 11 := by
  sorry

end NUMINAMATH_GPT_minutes_before_4_angle_same_as_4_l2353_235310


namespace NUMINAMATH_GPT_total_number_of_notes_l2353_235303

-- The total amount of money in Rs.
def total_amount : ℕ := 400

-- The number of each type of note is equal.
variable (n : ℕ)

-- The total value equation given the number of each type of note.
def total_value : ℕ := n * 1 + n * 5 + n * 10

-- Prove that if the total value equals 400, the total number of notes is 75.
theorem total_number_of_notes : total_value n = total_amount → 3 * n = 75 :=
by
  sorry

end NUMINAMATH_GPT_total_number_of_notes_l2353_235303


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_l2353_235305

theorem arithmetic_sequence_sum (a : ℕ → ℝ) (d : ℝ)
  (h_seq : ∀ n, a (n + 1) = a n + d)
  (h_a7 : a 7 = 12) :
  a 3 + a 11 = 24 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_l2353_235305


namespace NUMINAMATH_GPT_max_age_l2353_235325

-- Definitions of the conditions
def born_same_day (max_birth luka_turn4 : ℕ) : Prop := max_birth = luka_turn4
def age_difference (luka_age aubrey_age : ℕ) : Prop := luka_age = aubrey_age + 2
def aubrey_age_on_birthday : ℕ := 8

-- Prove that Max's age is 6 years when Aubrey is 8 years old
theorem max_age (luka_birth aubrey_birth max_birth : ℕ) 
                (h1 : born_same_day max_birth luka_birth) 
                (h2 : age_difference luka_birth aubrey_birth) : 
                (aubrey_birth + 4 - luka_birth) = 6 :=
by
  sorry

end NUMINAMATH_GPT_max_age_l2353_235325


namespace NUMINAMATH_GPT_find_k_l2353_235393

theorem find_k (x k : ℝ) (h1 : (x^2 - k) * (x + k) = x^3 + k * (x^2 - x - 2)) (h2 : k ≠ 0) :
  k = 2 :=
sorry

end NUMINAMATH_GPT_find_k_l2353_235393


namespace NUMINAMATH_GPT_second_term_is_4_l2353_235376

-- Define the arithmetic sequence conditions
variables (a d : ℝ) -- first term a, common difference d

-- The condition given in the problem
def sum_first_and_third_term (a d : ℝ) : Prop :=
  a + (a + 2 * d) = 8

-- What we need to prove
theorem second_term_is_4 (a d : ℝ) (h : sum_first_and_third_term a d) : a + d = 4 :=
sorry

end NUMINAMATH_GPT_second_term_is_4_l2353_235376


namespace NUMINAMATH_GPT_intersection_A_B_l2353_235363

def A := {y : ℝ | ∃ x : ℝ, y = 2^x}
def B := {y : ℝ | ∃ x : ℝ, y = -x^2 + 2}
def Intersection := {y : ℝ | 0 < y ∧ y ≤ 2}

theorem intersection_A_B :
  (A ∩ B) = Intersection :=
by
  sorry

end NUMINAMATH_GPT_intersection_A_B_l2353_235363


namespace NUMINAMATH_GPT_max_obtuse_angles_in_quadrilateral_l2353_235380

theorem max_obtuse_angles_in_quadrilateral (a b c d : ℝ) 
  (h₁ : a + b + c + d = 360)
  (h₂ : 90 < a)
  (h₃ : 90 < b)
  (h₄ : 90 < c) :
  90 > d :=
sorry

end NUMINAMATH_GPT_max_obtuse_angles_in_quadrilateral_l2353_235380


namespace NUMINAMATH_GPT_melissa_driving_time_l2353_235343

theorem melissa_driving_time
  (trips_per_month: ℕ)
  (months_per_year: ℕ)
  (total_hours_per_year: ℕ)
  (total_trips: ℕ)
  (hours_per_trip: ℕ) :
  trips_per_month = 2 ∧
  months_per_year = 12 ∧
  total_hours_per_year = 72 ∧
  total_trips = (trips_per_month * months_per_year) ∧
  hours_per_trip = (total_hours_per_year / total_trips) →
  hours_per_trip = 3 :=
by
  intro h
  obtain ⟨h1, h2, h3, h4, h5⟩ := h
  sorry

end NUMINAMATH_GPT_melissa_driving_time_l2353_235343
