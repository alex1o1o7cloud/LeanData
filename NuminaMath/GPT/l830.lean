import Mathlib

namespace side_length_of_square_perimeter_of_square_l830_83048

theorem side_length_of_square {d s: ℝ} (h: d = 2 * Real.sqrt 2): s = 2 :=
by
  sorry

theorem perimeter_of_square {s P: ℝ} (h: s = 2): P = 8 :=
by
  sorry

end side_length_of_square_perimeter_of_square_l830_83048


namespace cost_of_one_book_l830_83065

theorem cost_of_one_book (m : ℕ) (H1: 1100 < 900 + 9 * m ∧ 900 + 9 * m < 1200)
                                (H2: 1500 < 1300 + 13 * m ∧ 1300 + 13 * m < 1600) : 
                                m = 23 :=
by {
  sorry
}

end cost_of_one_book_l830_83065


namespace k_minus_2_divisible_by_3_l830_83013

theorem k_minus_2_divisible_by_3
  (k : ℕ)
  (a : ℕ → ℤ)
  (h_a0_pos : 0 < k)
  (h_seq : ∀ n ≥ 1, a n = (a (n - 1) + n^k) / n) :
  (k - 2) % 3 = 0 :=
sorry

end k_minus_2_divisible_by_3_l830_83013


namespace inverse_function_evaluation_l830_83098

def g (x : ℕ) : ℕ :=
  if x = 1 then 4
  else if x = 2 then 5
  else if x = 3 then 2
  else if x = 4 then 3
  else if x = 5 then 1
  else 0  -- default case, though it shouldn't be used given the conditions

noncomputable def g_inv (y : ℕ) : ℕ :=
  if y = 4 then 1
  else if y = 5 then 2
  else if y = 2 then 3
  else if y = 3 then 4
  else if y = 1 then 5
  else 0  -- default case, though it shouldn't be used given the conditions

theorem inverse_function_evaluation : g_inv (g_inv (g_inv 4)) = 2 := by
  sorry

end inverse_function_evaluation_l830_83098


namespace number_of_elderly_employees_in_sample_l830_83008

variables (total_employees young_employees sample_young_employees elderly_employees : ℕ)
variables (sample_total : ℕ)

def conditions (total_employees young_employees sample_young_employees elderly_employees : ℕ) :=
  total_employees = 430 ∧
  young_employees = 160 ∧
  sample_young_employees = 32 ∧
  (∃ M, M = 2 * elderly_employees ∧ elderly_employees + M + young_employees = total_employees)

theorem number_of_elderly_employees_in_sample
  (total_employees young_employees sample_young_employees elderly_employees : ℕ)
  (sample_total : ℕ) :
  conditions total_employees young_employees sample_young_employees elderly_employees →
  sample_total = 430 * 32 / 160 →
  sample_total = 90 * 32 / 430 :=
by
  sorry

end number_of_elderly_employees_in_sample_l830_83008


namespace linear_term_coefficient_is_neg_two_l830_83019

-- Define the quadratic equation
def quadratic_eq (a b c x : ℝ) : Prop :=
  a * x^2 + b * x + c = 0

-- Define the specific quadratic equation
def specific_quadratic_eq (x : ℝ) : Prop :=
  quadratic_eq 1 (-2) (-1) x

-- The statement to prove the coefficient of the linear term
theorem linear_term_coefficient_is_neg_two : ∀ x : ℝ, specific_quadratic_eq x → ∀ a b c : ℝ, quadratic_eq a b c x → b = -2 :=
by
  intros x h_eq a b c h_quadratic_eq
  -- Proof is omitted
  sorry

end linear_term_coefficient_is_neg_two_l830_83019


namespace isosceles_triangle_perimeter_l830_83062

-- Define the conditions
def isosceles_triangle (a b c : ℝ) : Prop :=
  (a = b ∨ b = c ∨ c = a) ∧ (a + b > c) ∧ (b + c > a) ∧ (c + a > b)

-- Define the side lengths
def side1 := 2
def side2 := 2
def base := 5

-- Define the perimeter
def perimeter (a b c : ℝ) := a + b + c

-- State the theorem
theorem isosceles_triangle_perimeter : isosceles_triangle side1 side2 base → perimeter side1 side2 base = 9 :=
  by sorry

end isosceles_triangle_perimeter_l830_83062


namespace find_y_l830_83020

-- Define the sequence from 1 to 50
def seq_sum : ℕ := (50 * 51) / 2

-- Define y and the average condition
def average_condition (y : ℚ) : Prop :=
  (seq_sum + y) / 51 = 51 * y

-- Theorem statement
theorem find_y (y : ℚ) (h : average_condition y) : y = 51 / 104 :=
by
  sorry

end find_y_l830_83020


namespace sum_term_S2018_l830_83016

def arithmetic_sequence (a : ℕ → ℤ) := 
  ∃ (d : ℤ), ∀ n, a (n + 1) = a n + d

def sum_first_n_terms (a : ℕ → ℤ) (S : ℕ → ℤ) := 
  S 0 = 0 ∧ ∀ n, S (n + 1) = S n + a (n + 1)

theorem sum_term_S2018 :
  ∃ (a S : ℕ → ℤ),
    arithmetic_sequence a ∧ 
    sum_first_n_terms a S ∧ 
    a 1 = -2018 ∧ 
    ((S 2015) / 2015 - (S 2013) / 2013 = 2) ∧ 
    S 2018 = -2018 
:= by
  sorry

end sum_term_S2018_l830_83016


namespace sum_of_coeffs_l830_83066

theorem sum_of_coeffs (x y : ℤ) : (x - 3 * y) ^ 20 = 2 ^ 20 := by
  sorry

end sum_of_coeffs_l830_83066


namespace total_respondents_l830_83089

theorem total_respondents (x_preference resp_y : ℕ) (h1 : x_preference = 360) (h2 : 9 * resp_y = x_preference) : 
  resp_y + x_preference = 400 :=
by 
  sorry

end total_respondents_l830_83089


namespace transformed_coords_of_point_l830_83024

noncomputable def polar_to_rectangular_coordinates (r θ : ℝ) : ℝ × ℝ :=
  (r * Real.cos θ, r * Real.sin θ)

noncomputable def transformed_coordinates (r θ : ℝ) : ℝ × ℝ :=
  let new_r := r ^ 3
  let new_θ := (3 * Real.pi / 2) * θ
  polar_to_rectangular_coordinates new_r new_θ

theorem transformed_coords_of_point (r θ : ℝ)
  (h_r : r = Real.sqrt (8^2 + 6^2))
  (h_cosθ : Real.cos θ = 8 / 10)
  (h_sinθ : Real.sin θ = 6 / 10)
  (coords_match : polar_to_rectangular_coordinates r θ = (8, 6)) :
  transformed_coordinates r θ = (-600, -800) :=
by
  -- The proof goes here
  sorry

end transformed_coords_of_point_l830_83024


namespace rectangle_proof_right_triangle_proof_l830_83004

-- Definition of rectangle condition
def rectangle_condition (a b : ℕ) : Prop :=
  a * b = 2 * (a + b)

-- Definition of right triangle condition
def right_triangle_condition (a b : ℕ) : Prop :=
  a + b + Int.natAbs (Int.sqrt (a^2 + b^2)) = a * b / 2 ∧
  (∃ c : ℕ, c = Int.natAbs (Int.sqrt (a^2 + b^2)))

-- Recangle proof
theorem rectangle_proof : ∃! p : ℕ × ℕ, rectangle_condition p.1 p.2 := sorry

-- Right triangle proof
theorem right_triangle_proof : ∃! t : ℕ × ℕ, right_triangle_condition t.1 t.2 := sorry

end rectangle_proof_right_triangle_proof_l830_83004


namespace min_value_of_squares_l830_83074

variable (a b t : ℝ)

theorem min_value_of_squares (ht : 0 < t) (habt : a + b = t) : 
  a^2 + b^2 ≥ t^2 / 2 := 
by
  sorry

end min_value_of_squares_l830_83074


namespace average_bowling_score_l830_83092

-- Definitions of the scores
def g : ℕ := 120
def m : ℕ := 113
def b : ℕ := 85

-- Theorem statement: The average score is 106
theorem average_bowling_score : (g + m + b) / 3 = 106 := by
  sorry

end average_bowling_score_l830_83092


namespace heap_holds_20_sheets_l830_83081

theorem heap_holds_20_sheets :
  ∀ (num_bundles num_bunches num_heaps sheets_per_bundle sheets_per_bunch total_sheets : ℕ),
    num_bundles = 3 →
    num_bunches = 2 →
    num_heaps = 5 →
    sheets_per_bundle = 2 →
    sheets_per_bunch = 4 →
    total_sheets = 114 →
    (total_sheets - (num_bundles * sheets_per_bundle + num_bunches * sheets_per_bunch)) / num_heaps = 20 := 
by
  intros num_bundles num_bunches num_heaps sheets_per_bundle sheets_per_bunch total_sheets 
  intros h1 h2 h3 h4 h5 h6 
  sorry

end heap_holds_20_sheets_l830_83081


namespace paving_rate_l830_83036

theorem paving_rate
  (length : ℝ) (width : ℝ) (total_cost : ℝ)
  (h_length : length = 5.5)
  (h_width : width = 3.75)
  (h_total_cost : total_cost = 16500) :
  total_cost / (length * width) = 800 := by
  sorry

end paving_rate_l830_83036


namespace min_nSn_l830_83009

theorem min_nSn 
  (a : ℕ → ℝ) 
  (S : ℕ → ℝ)
  (m : ℕ)
  (h1 : m ≥ 2)
  (h2 : S (m-1) = -2) 
  (h3 : S m = 0) 
  (h4 : S (m+1) = 3) : 
  ∃ n : ℕ, n * S n = -9 :=
by {
  sorry
}

end min_nSn_l830_83009


namespace polygon_at_least_9_sides_l830_83056

theorem polygon_at_least_9_sides (n : ℕ) (h1 : ∀ i, 1 ≤ i ∧ i ≤ n → (∃ θ, θ < 45 ∧ (∀ j, 1 ≤ j ∧ j ≤ n → θ = 360 / n))):
  9 ≤ n :=
sorry

end polygon_at_least_9_sides_l830_83056


namespace gwen_spent_money_l830_83099

theorem gwen_spent_money (initial : ℕ) (remaining : ℕ) (spent : ℕ) 
  (h_initial : initial = 7) 
  (h_remaining : remaining = 5) 
  (h_spent : spent = initial - remaining) : 
  spent = 2 := 
sorry

end gwen_spent_money_l830_83099


namespace multiply_polynomials_l830_83055

theorem multiply_polynomials (x y : ℝ) : 
  (3 * x^2 - 4 * y^3) * (9 * x^4 + 12 * x^2 * y^3 + 16 * y^6) = 27 * x^6 - 64 * y^9 :=
by 
  sorry

end multiply_polynomials_l830_83055


namespace car_speed_is_80_l830_83070

theorem car_speed_is_80 
  (d : ℝ) (t_delay : ℝ) (v_train_factor : ℝ)
  (t_car t_train : ℝ) (v : ℝ) :
  ((d = 75) ∧ (t_delay = 12.5 / 60) ∧ (v_train_factor = 1.5) ∧ 
   (d = v * t_car) ∧ (d = v_train_factor * v * (t_car - t_delay))) →
  v = 80 := 
sorry

end car_speed_is_80_l830_83070


namespace walking_time_l830_83042

theorem walking_time (distance_walking_rate : ℕ) 
                     (distance : ℕ)
                     (rest_distance : ℕ) 
                     (rest_time : ℕ) 
                     (total_walking_time : ℕ) : 
  distance_walking_rate = 10 → 
  rest_distance = 10 → 
  rest_time = 7 → 
  distance = 50 → 
  total_walking_time = 328 → 
  total_walking_time = (distance / distance_walking_rate) * 60 + ((distance / rest_distance) - 1) * rest_time :=
by
  sorry

end walking_time_l830_83042


namespace unique_real_solution_l830_83005

theorem unique_real_solution :
  ∃! x : ℝ, -((x + 2) ^ 2) ≥ 0 :=
sorry

end unique_real_solution_l830_83005


namespace project_completion_time_saving_l830_83085

/-- A theorem stating that if a project with initial and additional workforce configuration,
the project will be completed 10 days ahead of schedule. -/
theorem project_completion_time_saving
  (total_days : ℕ := 100)
  (initial_people : ℕ := 10)
  (initial_days : ℕ := 30)
  (initial_fraction : ℚ := 1 / 5)
  (additional_people : ℕ := 10)
  : (total_days - ((initial_days + (1 / (initial_people + additional_people * initial_fraction)) * (total_days * initial_fraction) / initial_fraction)) = 10) :=
sorry

end project_completion_time_saving_l830_83085


namespace part_a_not_divisible_by_29_part_b_divisible_by_11_l830_83059
open Nat

-- Part (a): Checking divisibility of 5641713 by 29
def is_divisible_by_29 (n : ℕ) : Prop :=
  n % 29 = 0

theorem part_a_not_divisible_by_29 : ¬is_divisible_by_29 5641713 :=
  by sorry

-- Part (b): Checking divisibility of 1379235 by 11
def is_divisible_by_11 (n : ℕ) : Prop :=
  n % 11 = 0

theorem part_b_divisible_by_11 : is_divisible_by_11 1379235 :=
  by sorry

end part_a_not_divisible_by_29_part_b_divisible_by_11_l830_83059


namespace oblique_projection_correctness_l830_83083

structure ProjectionConditions where
  intuitive_diagram_of_triangle_is_triangle : Prop
  intuitive_diagram_of_parallelogram_is_parallelogram : Prop

theorem oblique_projection_correctness (c : ProjectionConditions)
  (h1 : c.intuitive_diagram_of_triangle_is_triangle)
  (h2 : c.intuitive_diagram_of_parallelogram_is_parallelogram) :
  c.intuitive_diagram_of_triangle_is_triangle ∧ c.intuitive_diagram_of_parallelogram_is_parallelogram :=
by
  sorry

end oblique_projection_correctness_l830_83083


namespace find_x_plus_y_l830_83015

theorem find_x_plus_y (x y : ℝ) (h1 : x + Real.sin y = 2008) (h2 : x + 2008 * Real.cos y = 2007) (hy : 0 ≤ y ∧ y ≤ Real.pi / 2) :
  x + y = 2007 + Real.pi / 2 := 
by
  sorry

end find_x_plus_y_l830_83015


namespace position_2023_l830_83027

def initial_position := "ABCD"

def rotate_180 (pos : String) : String :=
  match pos with
  | "ABCD" => "CDAB"
  | "CDAB" => "ABCD"
  | "DCBA" => "BADC"
  | "BADC" => "DCBA"
  | _ => pos

def reflect_horizontal (pos : String) : String :=
  match pos with
  | "ABCD" => "ABCD"
  | "CDAB" => "DCBA"
  | "DCBA" => "CDAB"
  | "BADC" => "BADC"
  | _ => pos

def transformation (n : ℕ) : String :=
  let cnt := n % 4
  if cnt = 1 then rotate_180 initial_position
  else if cnt = 2 then rotate_180 (rotate_180 initial_position)
  else if cnt = 3 then rotate_180 (reflect_horizontal (rotate_180 initial_position))
  else reflect_horizontal initial_position

theorem position_2023 : transformation 2023 = "DCBA" := by
  sorry

end position_2023_l830_83027


namespace find_x_plus_y_of_parallel_vectors_l830_83053

theorem find_x_plus_y_of_parallel_vectors 
  (x y : ℝ) 
  (a b : ℝ × ℝ × ℝ)
  (ha : a = (x, 2, -2)) 
  (hb : b = (2, y, 4)) 
  (h_parallel : ∃ k : ℝ, a = k • b) 
  : x + y = -5 := 
by 
  sorry

end find_x_plus_y_of_parallel_vectors_l830_83053


namespace problem1_solution_problem2_solution_l830_83069

-- Problem 1: f(x-2) = 3x - 5 implies f(x) = 3x + 1
def problem1 (x : ℝ) (f : ℝ → ℝ) : Prop := 
  ∀ x : ℝ, f (x - 2) = 3 * x - 5 → f x = 3 * x + 1

-- Problem 2: Quadratic function satisfying specific conditions
def is_quadratic (f : ℝ → ℝ) : Prop := 
  ∃ a b c : ℝ, ∀ x : ℝ, f x = a*x^2 + b*x + c

def problem2 (f : ℝ → ℝ) : Prop :=
  is_quadratic f ∧
  (f 0 = 4) ∧
  (∀ x : ℝ, f (3 - x) = f x) ∧
  (∀ x : ℝ, f x ≥ 7/4) →
  (∀ x : ℝ, f x = x^2 - 3*x + 4)

-- Statements to be proved
theorem problem1_solution : ∀ f : ℝ → ℝ, problem1 x f := sorry
theorem problem2_solution : ∀ f : ℝ → ℝ, problem2 f := sorry

end problem1_solution_problem2_solution_l830_83069


namespace combined_sum_correct_l830_83052

-- Define the sum of integers in a range
def sum_of_integers (a b : Int) : Int := (b - a + 1) * (a + b) / 2

-- Define the sum of squares of integers in a range
def sum_of_squares (a b : Int) : Int :=
  let sum_sq (n : Int) : Int := n * (n + 1) * (2 * n + 1) / 6
  sum_sq b - sum_sq (a - 1)

-- Define the combined sum function
def combined_sum (a b c d : Int) : Int :=
  sum_of_integers a b + sum_of_squares c d

-- Theorem statement: Prove the combined sum of integers from -50 to 40 and squares of integers from 10 to 40 is 21220
theorem combined_sum_correct :
  combined_sum (-50) 40 10 40 = 21220 :=
by
  -- leaving the proof as a sorry
  sorry

end combined_sum_correct_l830_83052


namespace maximize_product_l830_83072

variable (x y : ℝ)
variable (h_xy_pos : x > 0 ∧ y > 0)
variable (h_sum : x + y = 35)

theorem maximize_product : x^5 * y^2 ≤ (25: ℝ)^5 * (10: ℝ)^2 :=
by
  -- Here we need to prove that the product x^5 y^2 is maximized at (x, y) = (25, 10)
  sorry

end maximize_product_l830_83072


namespace least_positive_period_of_f_maximum_value_of_f_monotonically_increasing_intervals_of_f_l830_83079

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi / 3) + 2

theorem least_positive_period_of_f :
  ∃ T > 0, ∀ x, f (x + T) = f x :=
sorry

theorem maximum_value_of_f :
  ∃ x, f x = 3 :=
sorry

theorem monotonically_increasing_intervals_of_f :
  ∀ k : ℤ, ∃ a b : ℝ, a = -Real.pi / 12 + k * Real.pi ∧ b = 5 * Real.pi / 12 + k * Real.pi ∧ ∀ x, a < x ∧ x < b → ∀ x', a ≤ x' ∧ x' ≤ x → f x' < f x :=
sorry

end least_positive_period_of_f_maximum_value_of_f_monotonically_increasing_intervals_of_f_l830_83079


namespace total_savings_at_end_of_year_l830_83012

-- Defining constants for daily savings and the number of days in a year
def daily_savings : ℕ := 24
def days_in_year : ℕ := 365

-- Stating the theorem
theorem total_savings_at_end_of_year : daily_savings * days_in_year = 8760 :=
by
  sorry

end total_savings_at_end_of_year_l830_83012


namespace part1_part2_l830_83071

theorem part1 (m : ℂ) : (m * (m + 2)).re = 0 ∧ (m^2 + m - 2).im ≠ 0 → m = 0 := by
  sorry

theorem part2 (m : ℝ) : (m * (m + 2) > 0 ∧ m^2 + m - 2 < 0) → 0 < m ∧ m < 1 := by
  sorry

end part1_part2_l830_83071


namespace derivative_at_neg_one_l830_83017

-- Define the function f
def f (x : ℝ) : ℝ := x ^ 6

-- Define the derivative f'
def f' (x : ℝ) : ℝ := 6 * x ^ 5

-- The statement we want to prove
theorem derivative_at_neg_one : f' (-1) = -6 := sorry

end derivative_at_neg_one_l830_83017


namespace correct_option_l830_83001

def M : Set ℝ := { x | x^2 - 4 = 0 }

theorem correct_option : -2 ∈ M :=
by
  -- Definitions and conditions from the problem
  -- Set M is defined as the set of all x such that x^2 - 4 = 0
  have hM : M = { x | x^2 - 4 = 0 } := rfl
  -- Goal is to show that -2 belongs to the set M
  sorry

end correct_option_l830_83001


namespace complex_sum_l830_83032

-- Define the given condition as a hypothesis
variables {z : ℂ} (h : z^2 + z + 1 = 0)

-- Define the statement to prove
theorem complex_sum (h : z^2 + z + 1 = 0) : z^96 + z^97 + z^98 + z^99 + z^100 + z^101 = 0 :=
sorry

end complex_sum_l830_83032


namespace min_distance_MN_l830_83006

open Real

noncomputable def f (x : ℝ) := exp x - (1 / 2) * x^2
noncomputable def g (x : ℝ) := x - 1

theorem min_distance_MN (x1 x2 : ℝ) (h1 : x1 ≥ 0) (h2 : x2 > 0) (h3 : f x1 = g x2) :
  abs (x2 - x1) = 2 :=
by
  sorry

end min_distance_MN_l830_83006


namespace PartI_PartII_l830_83049

-- Define the function f
def f (x : ℝ) : ℝ := |2 * x - 1|

-- Problem statement for (Ⅰ)
theorem PartI (x : ℝ) : (f x < |x| + 1) ↔ (0 < x ∧ x < 2) :=
by sorry

-- Define conditions for PartII
variables (x y : ℝ)
def condition1 : Prop := |x - y - 1| ≤ 1 / 3
def condition2 : Prop := |2 * y + 1| ≤ 1 / 6

-- Problem statement for (Ⅱ)
theorem PartII (h1 : condition1 x y) (h2 : condition2 y) : f x < 1 :=
by sorry

end PartI_PartII_l830_83049


namespace cloth_gain_representation_l830_83044

theorem cloth_gain_representation (C S : ℝ) (h1 : S = 1.20 * C) (h2 : ∃ gain, gain = 60 * S - 60 * C) :
  ∃ meters : ℝ, meters = (60 * S - 60 * C) / S ∧ meters = 12 :=
by
  sorry

end cloth_gain_representation_l830_83044


namespace dalmatians_with_right_ear_spots_l830_83018

def TotalDalmatians := 101
def LeftOnlySpots := 29
def RightOnlySpots := 17
def NoEarSpots := 22

theorem dalmatians_with_right_ear_spots : 
  (TotalDalmatians - LeftOnlySpots - NoEarSpots) = 50 :=
by
  -- Proof goes here, but for now, we use sorry
  sorry

end dalmatians_with_right_ear_spots_l830_83018


namespace factor_expression_l830_83026

theorem factor_expression (a b c : ℝ) : 
  ( (a^2 - b^2)^4 + (b^2 - c^2)^4 + (c^2 - a^2)^4 ) / 
  ( (a - b)^4 + (b - c)^4 + (c - a)^4 ) = 1 := 
by sorry

end factor_expression_l830_83026


namespace age_of_teacher_l830_83088

theorem age_of_teacher
    (n_students : ℕ)
    (avg_age_students : ℕ)
    (new_avg_age : ℕ)
    (n_total : ℕ)
    (H1 : n_students = 22)
    (H2 : avg_age_students = 21)
    (H3 : new_avg_age = avg_age_students + 1)
    (H4 : n_total = n_students + 1) :
    ((new_avg_age * n_total) - (avg_age_students * n_students) = 44) :=
by
    sorry

end age_of_teacher_l830_83088


namespace total_accidents_l830_83082

noncomputable def A (k x : ℕ) : ℕ := 96 + k * x

theorem total_accidents :
  let k_morning := 1
  let k_evening := 3
  let x_morning := 2000
  let x_evening := 1000
  A k_morning x_morning + A k_evening x_evening = 5192 := by
  sorry

end total_accidents_l830_83082


namespace farmer_rent_l830_83035

-- Definitions based on given conditions
def rent_per_acre_per_month : ℕ := 60
def length_of_plot : ℕ := 360
def width_of_plot : ℕ := 1210
def square_feet_per_acre : ℕ := 43560

-- Problem statement: 
-- Prove that the monthly rent to rent the rectangular plot is $600.
theorem farmer_rent : 
  (length_of_plot * width_of_plot) / square_feet_per_acre * rent_per_acre_per_month = 600 :=
by
  sorry

end farmer_rent_l830_83035


namespace each_girl_gets_2_dollars_l830_83050

theorem each_girl_gets_2_dollars :
  let debt := 40
  let lulu_savings := 6
  let nora_savings := 5 * lulu_savings
  let tamara_savings := nora_savings / 3
  let total_savings := tamara_savings + nora_savings + lulu_savings
  total_savings - debt = 6 → (total_savings - debt) / 3 = 2 :=
by
  sorry

end each_girl_gets_2_dollars_l830_83050


namespace heights_inequality_l830_83076

theorem heights_inequality (a b c h_a h_b h_c p R : ℝ) (h₁ : a ≤ b) (h₂ : b ≤ c) :
  h_a + h_b + h_c ≤ (3 * b * (a^2 + a * c + c^2)) / (4 * p * R) :=
by
  sorry

end heights_inequality_l830_83076


namespace john_finishes_ahead_l830_83058

noncomputable def InitialDistanceBehind : ℝ := 12
noncomputable def JohnSpeed : ℝ := 4.2
noncomputable def SteveSpeed : ℝ := 3.7
noncomputable def PushTime : ℝ := 28

theorem john_finishes_ahead :
  (JohnSpeed * PushTime - InitialDistanceBehind) - (SteveSpeed * PushTime) = 2 := by
  sorry

end john_finishes_ahead_l830_83058


namespace men_work_in_80_days_l830_83037

theorem men_work_in_80_days (x : ℕ) (work_eq_20men_56days : x * 80 = 20 * 56) : x = 14 :=
by 
  sorry

end men_work_in_80_days_l830_83037


namespace total_jokes_proof_l830_83043

-- Definitions of the conditions
def jokes_jessy_last_saturday : Nat := 11
def jokes_alan_last_saturday : Nat := 7
def jokes_jessy_next_saturday : Nat := 2 * jokes_jessy_last_saturday
def jokes_alan_next_saturday : Nat := 2 * jokes_alan_last_saturday

-- Sum of jokes over two Saturdays
def total_jokes : Nat := (jokes_jessy_last_saturday + jokes_alan_last_saturday) + (jokes_jessy_next_saturday + jokes_alan_next_saturday)

-- The proof problem
theorem total_jokes_proof : total_jokes = 54 := 
by
  sorry

end total_jokes_proof_l830_83043


namespace right_triangle_angles_ratio_l830_83025

theorem right_triangle_angles_ratio (α β : ℝ) (h1 : α + β = 90) (h2 : α / β = 3) :
  α = 67.5 ∧ β = 22.5 :=
sorry

end right_triangle_angles_ratio_l830_83025


namespace wall_print_costs_are_15_l830_83084

-- Define the cost of curtains, installation, total cost, and number of wall prints.
variable (cost_curtain : ℕ := 30)
variable (num_curtains : ℕ := 2)
variable (cost_installation : ℕ := 50)
variable (num_wall_prints : ℕ := 9)
variable (total_cost : ℕ := 245)

-- Define the total cost of curtains
def total_cost_curtains : ℕ := num_curtains * cost_curtain

-- Define the total fixed costs
def total_fixed_costs : ℕ := total_cost_curtains + cost_installation

-- Define the total cost of wall prints
def total_cost_wall_prints : ℕ := total_cost - total_fixed_costs

-- Define the cost per wall print
def cost_per_wall_print : ℕ := total_cost_wall_prints / num_wall_prints

-- Prove the cost per wall print is $15.00
theorem wall_print_costs_are_15 : cost_per_wall_print = 15 := by
  -- This is a placeholder for the proof
  sorry

end wall_print_costs_are_15_l830_83084


namespace purple_tile_cost_correct_l830_83067

-- Definitions of given conditions
def turquoise_cost_per_tile : ℕ := 13
def wall1_area : ℕ := 5 * 8
def wall2_area : ℕ := 7 * 8
def total_area : ℕ := wall1_area + wall2_area
def tiles_per_square_foot : ℕ := 4
def total_tiles_needed : ℕ := total_area * tiles_per_square_foot
def turquoise_total_cost : ℕ := total_tiles_needed * turquoise_cost_per_tile
def savings : ℕ := 768
def purple_total_cost : ℕ := turquoise_total_cost - savings
def purple_cost_per_tile : ℕ := 11

-- Theorem stating the problem
theorem purple_tile_cost_correct :
  purple_total_cost / total_tiles_needed = purple_cost_per_tile :=
sorry

end purple_tile_cost_correct_l830_83067


namespace geometric_sequence_S20_l830_83045

-- Define the conditions and target statement
theorem geometric_sequence_S20
  (a : ℕ → ℝ) -- defining the sequence as a function from natural numbers to real numbers
  (q : ℝ) -- common ratio
  (h_pos : ∀ n, a n > 0) -- all terms are positive
  (h_geo : ∀ n, a (n + 1) = q * a n) -- geometric sequence property
  (S : ℕ → ℝ) -- sum function
  (h_S : ∀ n, S n = (a 1 * (1 - q ^ n)) / (1 - q)) -- sum formula for a geometric progression
  (h_S5 : S 5 = 3) -- given S_5 = 3
  (h_S15 : S 15 = 21) -- given S_15 = 21
  : S 20 = 45 := sorry

end geometric_sequence_S20_l830_83045


namespace find_value_of_y_l830_83077

theorem find_value_of_y (x y : ℤ) (h1 : x^2 = y - 3) (h2 : x = 7) : y = 52 :=
by
  sorry

end find_value_of_y_l830_83077


namespace factor_expression_l830_83041

theorem factor_expression (x : ℤ) : 84 * x^7 - 270 * x^13 = 6 * x^7 * (14 - 45 * x^6) := 
by sorry

end factor_expression_l830_83041


namespace geometric_sequence_fifth_term_l830_83038

variable {a : ℕ → ℝ} (h1 : a 1 = 1) (h4 : a 4 = 8)

theorem geometric_sequence_fifth_term (h_geom : ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q) :
  a 5 = 16 :=
sorry

end geometric_sequence_fifth_term_l830_83038


namespace remainder_when_divided_by_20_l830_83080

theorem remainder_when_divided_by_20 
  (n r : ℤ) 
  (k : ℤ)
  (h1 : n % 20 = r) 
  (h2 : 2 * n % 10 = 2)
  (h3 : 0 ≤ r ∧ r < 20)
  : r = 1 := 
sorry

end remainder_when_divided_by_20_l830_83080


namespace find_t_l830_83097

variable (a b c : ℝ × ℝ)
variable (t : ℝ)

-- Definitions based on given conditions
def vec_a : ℝ × ℝ := (3, 1)
def vec_b : ℝ × ℝ := (1, 3)
def vec_c (t : ℝ) : ℝ × ℝ := (t, 2)

-- Dot product definition
def dot_product (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

-- Condition that (vec_a - vec_c) is perpendicular to vec_b
def perpendicular_condition (t : ℝ) : Prop :=
  dot_product (vec_a - vec_c t) vec_b = 0

-- Proof statement
theorem find_t : ∃ t : ℝ, perpendicular_condition t ∧ t = 0 := 
by
  sorry

end find_t_l830_83097


namespace initial_investments_l830_83010

theorem initial_investments (x y : ℝ) : 
  -- Conditions
  5000 = y + (5000 - y) ∧
  (y * (1 + x / 100) = 2100) ∧
  ((5000 - y) * (1 + (x + 1) / 100) = 3180) →
  -- Conclusion
  y = 2000 ∧ (5000 - y) = 3000 := 
by 
  sorry

end initial_investments_l830_83010


namespace find_quotient_l830_83011

-- Definitions for the variables and conditions
variables (D d q r : ℕ)

-- Conditions
axiom eq1 : D = q * d + r
axiom eq2 : D + 65 = q * (d + 5) + r

-- Theorem statement
theorem find_quotient : q = 13 :=
by
  sorry

end find_quotient_l830_83011


namespace intersection_A_B_l830_83000
-- Lean 4 code statement

def set_A : Set ℝ := {x | |x - 1| > 2}
def set_B : Set ℝ := {x | x * (x - 5) < 0}
def set_intersection : Set ℝ := {x | 3 < x ∧ x < 5}

theorem intersection_A_B :
  (set_A ∩ set_B) = set_intersection := by
  sorry

end intersection_A_B_l830_83000


namespace initial_speed_is_correct_l830_83021

def initial_speed (v : ℝ) : Prop :=
  let D_total : ℝ := 70 * 5
  let D_2 : ℝ := 85 * 2
  let D_1 := v * 3
  D_total = D_1 + D_2

theorem initial_speed_is_correct :
  ∃ v : ℝ, initial_speed v ∧ v = 60 :=
by
  sorry

end initial_speed_is_correct_l830_83021


namespace roots_of_equation_l830_83095

theorem roots_of_equation :
  ∀ x : ℝ, (x^4 + x^2 - 20 = 0) ↔ (x = 2 ∨ x = -2) :=
by
  -- This will be the proof.
  -- We are claiming that x is a root of the polynomial if and only if x = 2 or x = -2.
  sorry

end roots_of_equation_l830_83095


namespace determine_y_l830_83063

theorem determine_y (x y : ℝ) (h₁ : x^2 = y - 7) (h₂ : x = 7) : y = 56 :=
sorry

end determine_y_l830_83063


namespace fractional_units_l830_83096

-- Define the mixed number and the smallest composite number
def mixed_number := 3 + 2/7
def smallest_composite := 4

-- To_struct fractional units of 3 2/7
theorem fractional_units (u : ℚ) (n : ℕ) (m : ℕ):
  u = 1/7 ∧ n = 23 ∧ m = 5 :=
by
  have h1 : u = 1 / 7 := sorry
  have h2 : mixed_number = 23 * u := sorry
  have h3 : smallest_composite - mixed_number = 5 * u := sorry
  have h4 : n = 23 := sorry
  have h5 : m = 5 := sorry
  exact ⟨h1, h4, h5⟩

end fractional_units_l830_83096


namespace intersection_nonempty_implies_t_lt_1_l830_83078

def M (x : ℝ) := x ≤ 1
def P (t : ℝ) (x : ℝ) := x > t

theorem intersection_nonempty_implies_t_lt_1 {t : ℝ} (h : ∃ x, M x ∧ P t x) : t < 1 :=
by
  sorry

end intersection_nonempty_implies_t_lt_1_l830_83078


namespace fraction_is_one_twelve_l830_83031

variables (A E : ℝ) (f : ℝ)

-- Given conditions
def condition1 : E = 200 := sorry
def condition2 : A - E = f * (A + E) := sorry
def condition3 : A * 1.10 = E * 1.20 + 20 := sorry

-- Proving the fraction f is 1/12
theorem fraction_is_one_twelve : E = 200 → (A - E = f * (A + E)) → (A * 1.10 = E * 1.20 + 20) → 
f = 1 / 12 :=
by
  intros hE hDiff hIncrease
  sorry

end fraction_is_one_twelve_l830_83031


namespace domain_log_sin_sqrt_l830_83002

theorem domain_log_sin_sqrt (x : ℝ) : 
  (2 < x ∧ x < (5 * Real.pi) / 3) ↔ 
  (∃ k : ℤ, (Real.pi / 3) + (4 * k * Real.pi) < x ∧ x < (5 * Real.pi / 3) + (4 * k * Real.pi) ∧ 2 < x) :=
by
  sorry

end domain_log_sin_sqrt_l830_83002


namespace rectangle_width_decrease_percent_l830_83046

theorem rectangle_width_decrease_percent (L W : ℝ) (h : L * W = L * W) :
  let L_new := 1.3 * L
  let W_new := W / 1.3 
  let percent_decrease := (1 - (W_new / W)) * 100
  percent_decrease = 23.08 :=
sorry

end rectangle_width_decrease_percent_l830_83046


namespace inequality_square_l830_83087

theorem inequality_square (a b : ℝ) (h1 : a > b) (h2 : b > 0) : a^2 > b^2 :=
sorry

end inequality_square_l830_83087


namespace m_range_decrease_y_l830_83086

theorem m_range_decrease_y {m : ℝ} : (∀ x1 x2 : ℝ, x1 < x2 → (2 * m + 2) * x1 + 5 > (2 * m + 2) * x2 + 5) ↔ m < -1 :=
by
  sorry

end m_range_decrease_y_l830_83086


namespace derivative_of_f_at_pi_over_2_l830_83094

noncomputable def f (x : ℝ) : ℝ := 5 * Real.cos x

theorem derivative_of_f_at_pi_over_2 :
  deriv f (Real.pi / 2) = -5 :=
sorry

end derivative_of_f_at_pi_over_2_l830_83094


namespace matrix_A_to_power_4_l830_83057

def matrix_A : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![2, -1], ![1, 1]]

def matrix_pow4 : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![0, -9], ![9, -9]]

theorem matrix_A_to_power_4 :
  matrix_A ^ 4 = matrix_pow4 :=
by
  sorry

end matrix_A_to_power_4_l830_83057


namespace Nell_initial_cards_l830_83022

theorem Nell_initial_cards (given_away : ℕ) (now_has : ℕ) : 
  given_away = 276 → now_has = 252 → (now_has + given_away) = 528 :=
by
  intros h_given_away h_now_has
  sorry

end Nell_initial_cards_l830_83022


namespace Timi_has_five_ears_l830_83034

theorem Timi_has_five_ears (seeing_ears_Imi seeing_ears_Dimi seeing_ears_Timi : ℕ)
  (H1 : seeing_ears_Imi = 8)
  (H2 : seeing_ears_Dimi = 7)
  (H3 : seeing_ears_Timi = 5)
  (total_ears : ℕ := (seeing_ears_Imi + seeing_ears_Dimi + seeing_ears_Timi) / 2) :
  total_ears - seeing_ears_Timi = 5 :=
by
  sorry -- Proof not required.

end Timi_has_five_ears_l830_83034


namespace negation_of_existential_prop_l830_83028

theorem negation_of_existential_prop :
  (¬ ∃ (x₀ : ℝ), x₀^2 + x₀ + 1 < 0) ↔ (∀ (x : ℝ), x^2 + x + 1 ≥ 0) :=
by
  sorry

end negation_of_existential_prop_l830_83028


namespace trigonometric_identity_l830_83040

open Real

theorem trigonometric_identity :
  sin (72 * pi / 180) * cos (12 * pi / 180) - cos (72 * pi / 180) * sin (12 * pi / 180) = sqrt 3 / 2 :=
by
  sorry

end trigonometric_identity_l830_83040


namespace min_cos_y_plus_sin_x_l830_83003

theorem min_cos_y_plus_sin_x
  (x y : ℝ)
  (h1 : Real.sin y + Real.cos x = Real.sin (3 * x))
  (h2 : Real.sin (2 * y) - Real.sin (2 * x) = Real.cos (4 * x) - Real.cos (2 * x)) :
  ∃ (v : ℝ), v = -1 - Real.sqrt (2 + Real.sqrt 2) / 2 :=
sorry

end min_cos_y_plus_sin_x_l830_83003


namespace shopping_money_l830_83093

theorem shopping_money (X : ℝ) (h : 0.70 * X = 840) : X = 1200 :=
sorry

end shopping_money_l830_83093


namespace jose_peanuts_l830_83073

def kenya_peanuts : Nat := 133
def difference_peanuts : Nat := 48

theorem jose_peanuts : (kenya_peanuts - difference_peanuts) = 85 := by
  sorry

end jose_peanuts_l830_83073


namespace money_out_of_pocket_l830_83090

theorem money_out_of_pocket
  (old_system_cost : ℝ)
  (trade_in_percent : ℝ)
  (new_system_cost : ℝ)
  (discount_percent : ℝ)
  (trade_in_value : ℝ)
  (discount_value : ℝ)
  (discounted_price : ℝ)
  (money_out_of_pocket : ℝ) :
  old_system_cost = 250 →
  trade_in_percent = 80 / 100 →
  new_system_cost = 600 →
  discount_percent = 25 / 100 →
  trade_in_value = old_system_cost * trade_in_percent →
  discount_value = new_system_cost * discount_percent →
  discounted_price = new_system_cost - discount_value →
  money_out_of_pocket = discounted_price - trade_in_value →
  money_out_of_pocket = 250 := by
  intros
  sorry

end money_out_of_pocket_l830_83090


namespace quadratic_roots_in_intervals_l830_83033

theorem quadratic_roots_in_intervals (a b c : ℝ) (h₁ : a < b) (h₂ : b < c) :
  ∃ x₁ x₂ : ℝ, (a < x₁ ∧ x₁ < b) ∧ (b < x₂ ∧ x₂ < c) ∧
  3 * x₁^2 - 2 * (a + b + c) * x₁ + (a * b + b * c + c * a) = 0 ∧
  3 * x₂^2 - 2 * (a + b + c) * x₂ + (a * b + b * c + c * a) = 0 :=
by
  sorry

end quadratic_roots_in_intervals_l830_83033


namespace traffic_accident_emergency_number_l830_83060

theorem traffic_accident_emergency_number (A B C D : ℕ) (h1 : A = 122) (h2 : B = 110) (h3 : C = 120) (h4 : D = 114) : 
  A = 122 := 
by
  exact h1

end traffic_accident_emergency_number_l830_83060


namespace intersection_of_A_B_l830_83007

variable (A : Set ℝ) (B : Set ℝ)

theorem intersection_of_A_B (hA : A = {-1, 0, 1, 2, 3}) (hB : B = {x : ℝ | 0 < x ∧ x < 3}) :
  A ∩ B = {1, 2} :=
  sorry

end intersection_of_A_B_l830_83007


namespace find_angle_B_l830_83039

-- Define the necessary trigonometric identities and dependencies
open Real

-- Declare the conditions under which we are working
theorem find_angle_B : 
  ∀ {a b A B : ℝ}, 
    a = 1 → 
    b = sqrt 3 → 
    A = π / 6 → 
    (B = π / 3 ∨ B = 2 * π / 3) := 
  by 
    intros a b A B ha hb hA
    sorry

end find_angle_B_l830_83039


namespace race_distance_l830_83064

theorem race_distance {a b c : ℝ} (h1 : b = 0.9 * a) (h2 : c = 0.95 * b) :
  let andrei_distance := 1000
  let boris_distance := andrei_distance - 100
  let valentin_distance := boris_distance - 50
  let valentin_actual_distance := (c / a) * andrei_distance
  andrei_distance - valentin_actual_distance = 145 :=
by
  sorry

end race_distance_l830_83064


namespace total_sleep_per_week_l830_83023

namespace TotalSleep

def hours_sleep_wd (days: Nat) : Nat := 6 * days
def hours_sleep_wknd (days: Nat) : Nat := 10 * days

theorem total_sleep_per_week : 
  hours_sleep_wd 5 + hours_sleep_wknd 2 = 50 := by
  sorry

end TotalSleep

end total_sleep_per_week_l830_83023


namespace negation_of_exists_statement_l830_83051

theorem negation_of_exists_statement :
  (¬ ∃ x : ℝ, x^2 + 1 < 0) ↔ (∀ x : ℝ, x^2 + 1 ≥ 0) :=
by
  sorry

end negation_of_exists_statement_l830_83051


namespace emails_in_afternoon_l830_83047

theorem emails_in_afternoon (A : ℕ) 
  (morning_emails : A + 3 = 10) : A = 7 :=
by {
    sorry
}

end emails_in_afternoon_l830_83047


namespace numbers_left_on_blackboard_l830_83061

theorem numbers_left_on_blackboard (n11 n12 n13 n14 n15 : ℕ)
    (h_n11 : n11 = 11) (h_n12 : n12 = 12) (h_n13 : n13 = 13) (h_n14 : n14 = 14) (h_n15 : n15 = 15)
    (total_numbers : n11 + n12 + n13 + n14 + n15 = 65) :
  ∃ (remaining1 remaining2 : ℕ), remaining1 = 12 ∧ remaining2 = 14 := 
sorry

end numbers_left_on_blackboard_l830_83061


namespace tiles_per_row_24_l830_83054

noncomputable def num_tiles_per_row (area : ℝ) (tile_size : ℝ) : ℝ :=
  let side_length_ft := Real.sqrt area
  let side_length_in := side_length_ft * 12
  side_length_in / tile_size

theorem tiles_per_row_24 :
  num_tiles_per_row 324 9 = 24 :=
by
  sorry

end tiles_per_row_24_l830_83054


namespace length_of_real_axis_l830_83091

noncomputable def hyperbola_1 : Prop :=
  ∃ (x y: ℝ), (x^2 / 16) - (y^2 / 4) = 1

noncomputable def hyperbola_2 (a b: ℝ) (ha : 0 < a) (hb : 0 < b) : Prop :=
  ∃ (x y: ℝ), (x^2 / a^2) - (y^2 / b^2) = 1

noncomputable def same_eccentricity (a b: ℝ) : Prop :=
  (1 + b^2 / a^2) = (1 + 1 / 4 / 16)

noncomputable def area_of_triangle (a b: ℝ) : Prop :=
  (a * b) = 32

theorem length_of_real_axis (a b: ℝ) (ha : 0 < a) (hb : 0 < b) :
  hyperbola_1 ∧ hyperbola_2 a b ha hb ∧ same_eccentricity a b ∧ area_of_triangle a b →
  2 * a = 16 :=
by
  sorry

end length_of_real_axis_l830_83091


namespace t_50_mod_7_l830_83014

theorem t_50_mod_7 (T : ℕ → ℕ) (h₁ : T 1 = 9) (h₂ : ∀ n > 1, T n = 9 ^ (T (n - 1))) :
  T 50 % 7 = 4 :=
sorry

end t_50_mod_7_l830_83014


namespace round_robin_chess_l830_83068

/-- 
In a round-robin chess tournament, two boys and several girls participated. 
The boys together scored 8 points, while all the girls scored an equal number of points.
We are to prove that the number of girls could have participated in the tournament is 7 or 14,
given that a win is 1 point, a draw is 0.5 points, and a loss is 0 points.
-/
theorem round_robin_chess (n : ℕ) (x : ℚ) (h : 2 * n * x + 16 = n ^ 2 + 3 * n + 2) : n = 7 ∨ n = 14 :=
sorry

end round_robin_chess_l830_83068


namespace largest_AC_value_l830_83029

theorem largest_AC_value : ∃ (a b c d : ℕ), 
  a < 20 ∧ b < 20 ∧ c < 20 ∧ d < 20 ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ 
  (∃ (AC BD : ℝ), AC * BD = a * c + b * d ∧
  AC ^ 2 + BD ^ 2 = a ^ 2 + b ^ 2 + c ^ 2 + d ^ 2 ∧
  AC = Real.sqrt 458) :=
sorry

end largest_AC_value_l830_83029


namespace kishore_savings_l830_83030

-- Define the monthly expenses and condition
def expenses : Real :=
  5000 + 1500 + 4500 + 2500 + 2000 + 6100

-- Define the monthly salary and savings conditions
def salary (S : Real) : Prop :=
  expenses + 0.1 * S = S

-- Define the savings amount
def savings (S : Real) : Real :=
  0.1 * S

-- The theorem to prove
theorem kishore_savings : ∃ S : Real, salary S ∧ savings S = 2733.33 :=
by
  sorry

end kishore_savings_l830_83030


namespace area_of_triangle_pqr_l830_83075

noncomputable def area_of_triangle (P Q R : ℝ) : ℝ :=
  let PQ := P + Q
  let PR := P + R
  let QR := Q + R
  if PQ^2 = PR^2 + QR^2 then
    1 / 2 * PR * QR
  else
    0

theorem area_of_triangle_pqr : 
  area_of_triangle 3 2 1 = 6 :=
by
  simp [area_of_triangle]
  sorry

end area_of_triangle_pqr_l830_83075
