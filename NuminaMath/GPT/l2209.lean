import Mathlib

namespace NUMINAMATH_GPT_common_difference_l2209_220985

-- Define the arithmetic sequence with general term
def arithmetic_seq (a₁ d : ℕ) (n : ℕ) : ℕ := a₁ + (n - 1) * d

theorem common_difference (a₁ a₅ a₄ d : ℕ) 
  (h₁ : a₁ + a₅ = 10)
  (h₂ : a₄ = 7)
  (h₅ : a₅ = a₁ + 4 * d)
  (h₄ : a₄ = a₁ + 3 * d) :
  d = 2 :=
by
  sorry

end NUMINAMATH_GPT_common_difference_l2209_220985


namespace NUMINAMATH_GPT_range_of_x_l2209_220967

noncomputable def f : ℝ → ℝ := sorry  -- f is an even function and decreasing on [0, +∞)

theorem range_of_x (x : ℝ) (h_even : ∀ x, f x = f (-x)) 
  (h_decreasing : ∀ x y, 0 ≤ x → x ≤ y → f x ≥ f y) 
  (h_condition : f (Real.log x) > f 1) : 
  1 / 10 < x ∧ x < 10 := 
sorry

end NUMINAMATH_GPT_range_of_x_l2209_220967


namespace NUMINAMATH_GPT_intersection_eq_neg1_l2209_220905

open Set

noncomputable def setA : Set Int := {x : Int | x^2 - 1 ≤ 0}
def setB : Set Int := {x : Int | x^2 - x - 2 = 0}

theorem intersection_eq_neg1 : setA ∩ setB = {-1} := by
  sorry

end NUMINAMATH_GPT_intersection_eq_neg1_l2209_220905


namespace NUMINAMATH_GPT_problem_solution_l2209_220944

noncomputable def alpha : ℝ := (3 + Real.sqrt 13) / 2
noncomputable def beta  : ℝ := (3 - Real.sqrt 13) / 2

theorem problem_solution : 7 * alpha ^ 4 + 10 * beta ^ 3 = 1093 :=
by
  -- Prove roots relation
  have hr1 : alpha * alpha - 3 * alpha - 1 = 0 := by sorry
  have hr2 : beta * beta - 3 * beta - 1 = 0 := by sorry
  -- Proceed to prove the required expression
  sorry

end NUMINAMATH_GPT_problem_solution_l2209_220944


namespace NUMINAMATH_GPT_min_val_of_3x_add_4y_l2209_220968

theorem min_val_of_3x_add_4y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + 3 * y = 5 * x * y) : 
  (3 * x + 4 * y ≥ 5) ∧ (3 * x + 4 * y = 5 → x + 4 * y = 3) := 
by
  sorry

end NUMINAMATH_GPT_min_val_of_3x_add_4y_l2209_220968


namespace NUMINAMATH_GPT_simplify_and_rationalize_l2209_220939

theorem simplify_and_rationalize : 
  (1 / (2 + (1 / (Real.sqrt 5 + 2)))) = (Real.sqrt 5 / 5) :=
by
  sorry

end NUMINAMATH_GPT_simplify_and_rationalize_l2209_220939


namespace NUMINAMATH_GPT_parabola_points_l2209_220999

theorem parabola_points :
  {p : ℝ × ℝ | p.2 = p.1^2 - 1 ∧ p.2 = 3} = {(-2, 3), (2, 3)} :=
by
  sorry

end NUMINAMATH_GPT_parabola_points_l2209_220999


namespace NUMINAMATH_GPT_sum_of_squares_is_perfect_square_l2209_220987

theorem sum_of_squares_is_perfect_square (n p k : ℤ) : 
  (∃ m : ℤ, n^2 + p^2 + k^2 = m^2) ↔ (n * k = (p / 2)^2) :=
by
  sorry

end NUMINAMATH_GPT_sum_of_squares_is_perfect_square_l2209_220987


namespace NUMINAMATH_GPT_ellipse_condition_l2209_220906

variables (m n : ℝ)

-- Definition of the curve
def curve_eqn (x y : ℝ) := m * x^2 + n * y^2 = 1

-- Define the condition for being an ellipse
def is_ellipse (m n : ℝ) : Prop :=
  m > 0 ∧ n > 0 ∧ m ≠ n

def mn_positive (m n : ℝ) : Prop := m * n > 0

-- Prove that mn > 0 is a necessary but not sufficient condition
theorem ellipse_condition (m n : ℝ) : mn_positive m n → is_ellipse m n → False := sorry

end NUMINAMATH_GPT_ellipse_condition_l2209_220906


namespace NUMINAMATH_GPT_simplify_complex_fraction_l2209_220927

-- Define the complex numbers involved
def numerator := 3 + 4 * Complex.I
def denominator := 5 - 2 * Complex.I

-- Define what we need to prove: the simplified form
theorem simplify_complex_fraction : 
    (numerator / denominator : Complex) = (7 / 29) + (26 / 29) * Complex.I := 
by
  -- Proof is omitted here
  sorry

end NUMINAMATH_GPT_simplify_complex_fraction_l2209_220927


namespace NUMINAMATH_GPT_grooming_time_correct_l2209_220921

def time_to_groom_poodle : ℕ := 30
def time_to_groom_terrier : ℕ := time_to_groom_poodle / 2
def num_poodles : ℕ := 3
def num_terriers : ℕ := 8

def total_grooming_time : ℕ := 
  (num_poodles * time_to_groom_poodle) + (num_terriers * time_to_groom_terrier)

theorem grooming_time_correct : 
  total_grooming_time = 210 := by
  sorry

end NUMINAMATH_GPT_grooming_time_correct_l2209_220921


namespace NUMINAMATH_GPT_geometric_sequence_a_formula_l2209_220989

noncomputable def a (n : ℕ) : ℤ :=
  if n = 1 then 1
  else if n = 2 then 2
  else n - 2

noncomputable def b (n : ℕ) : ℤ :=
  a (n + 1) - a n

theorem geometric_sequence (n : ℕ) (h : n ≥ 2) : 
  b n = (-1) * b (n - 1) := 
  sorry

theorem a_formula (n : ℕ) : 
  a n = (-1) ^ (n - 1) := 
  sorry

end NUMINAMATH_GPT_geometric_sequence_a_formula_l2209_220989


namespace NUMINAMATH_GPT_fence_perimeter_l2209_220990

-- Definitions for the given conditions
def num_posts : ℕ := 36
def gap_between_posts : ℕ := 6

-- Defining the proof problem to show the perimeter equals 192 feet
theorem fence_perimeter (n : ℕ) (g : ℕ) (sq_field : ℕ → ℕ → Prop) : n = 36 ∧ g = 6 ∧ sq_field n g → 4 * ((n / 4 - 1) * g) = 192 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_fence_perimeter_l2209_220990


namespace NUMINAMATH_GPT_solve_for_x_l2209_220907

def star (a b : ℤ) := a * b + 3 * b - a

theorem solve_for_x : ∃ x : ℤ, star 4 x = 46 := by
  sorry

end NUMINAMATH_GPT_solve_for_x_l2209_220907


namespace NUMINAMATH_GPT_xy_equals_one_l2209_220925

-- Define the mathematical theorem
theorem xy_equals_one (x y : ℝ) (h : x + y = 1 / x + 1 / y) (h₂ : x + y ≠ 0) : x * y = 1 := 
by
  sorry

end NUMINAMATH_GPT_xy_equals_one_l2209_220925


namespace NUMINAMATH_GPT_twelfth_term_arithmetic_sequence_l2209_220916

theorem twelfth_term_arithmetic_sequence (a d : ℤ) (h1 : a + 2 * d = 13) (h2 : a + 6 * d = 25) : a + 11 * d = 40 := 
sorry

end NUMINAMATH_GPT_twelfth_term_arithmetic_sequence_l2209_220916


namespace NUMINAMATH_GPT_prime_factors_2310_l2209_220930

-- Define the number in question
def number : ℕ := 2310

-- The main theorem statement
theorem prime_factors_2310 : ∃ s : Finset ℕ, (∀ p ∈ s, Nat.Prime p) ∧ 
Finset.prod s id = number ∧ Finset.card s = 5 := by
  sorry

end NUMINAMATH_GPT_prime_factors_2310_l2209_220930


namespace NUMINAMATH_GPT_john_has_leftover_correct_l2209_220988

-- Define the initial conditions
def initial_gallons : ℚ := 5
def given_away : ℚ := 18 / 7

-- Define the target result after subtraction
def remaining_gallons : ℚ := 17 / 7

-- The theorem statement
theorem john_has_leftover_correct :
  initial_gallons - given_away = remaining_gallons :=
by
  sorry

end NUMINAMATH_GPT_john_has_leftover_correct_l2209_220988


namespace NUMINAMATH_GPT_original_number_j_l2209_220920

noncomputable def solution (n : ℚ) : ℚ := (3 * (n + 3) - 5) / 3

theorem original_number_j { n : ℚ } (h : solution n = 10) : n = 26 / 3 :=
by
  sorry

end NUMINAMATH_GPT_original_number_j_l2209_220920


namespace NUMINAMATH_GPT_pebble_difference_l2209_220981

-- Definitions and conditions
variables (x : ℚ) -- we use rational numbers for exact division
def Candy := 2 * x
def Lance := 5 * x
def Sandy := 4 * x
def condition1 := Lance = Candy + 10

-- Theorem statement
theorem pebble_difference (h : condition1) : Lance + Sandy - Candy = 30 :=
sorry

end NUMINAMATH_GPT_pebble_difference_l2209_220981


namespace NUMINAMATH_GPT_tape_recorder_cost_l2209_220910

theorem tape_recorder_cost (x y : ℕ) (h1 : 170 ≤ x * y) (h2 : x * y ≤ 195)
  (h3 : (y - 2) * (x + 1) = x * y) : x * y = 180 :=
by
  sorry

end NUMINAMATH_GPT_tape_recorder_cost_l2209_220910


namespace NUMINAMATH_GPT_problem1_problem2_l2209_220933

-- Problem (1)
theorem problem1 (a : ℚ) (h : a = -1/2) : 
  a * (a - 4) - (a + 6) * (a - 2) = 16 := by
  sorry

-- Problem (2)
theorem problem2 (x y : ℚ) (hx : x = 8) (hy : y = -8) :
  (x + 2 * y) * (x - 2 * y) - (2 * x - y) * (-2 * x - y) = 0 := by
  sorry

end NUMINAMATH_GPT_problem1_problem2_l2209_220933


namespace NUMINAMATH_GPT_area_of_smaller_circle_l2209_220941

theorem area_of_smaller_circle (r R : ℝ) (PA AB : ℝ) 
  (h1 : R = 2 * r) (h2 : PA = 4) (h3 : AB = 4) :
  π * r^2 = 2 * π :=
by
  sorry

end NUMINAMATH_GPT_area_of_smaller_circle_l2209_220941


namespace NUMINAMATH_GPT_butter_mixture_price_l2209_220992

theorem butter_mixture_price :
  let cost1 := 48 * 150
  let cost2 := 36 * 125
  let cost3 := 24 * 100
  let revenue1 := cost1 + cost1 * (20 / 100)
  let revenue2 := cost2 + cost2 * (30 / 100)
  let revenue3 := cost3 + cost3 * (50 / 100)
  let total_weight := 48 + 36 + 24
  (revenue1 + revenue2 + revenue3) / total_weight = 167.5 :=
by
  sorry

end NUMINAMATH_GPT_butter_mixture_price_l2209_220992


namespace NUMINAMATH_GPT_minimum_distinct_values_is_145_l2209_220952

-- Define the conditions
def n_series : ℕ := 2023
def unique_mode_occurrence : ℕ := 15

-- Define the minimum number of distinct values satisfying the conditions
def min_distinct_values (n : ℕ) (mode_count : ℕ) : ℕ :=
  if mode_count < n then 
    (n - mode_count + 13) / 14 + 1
  else
    1

-- The theorem restating the problem to be solved
theorem minimum_distinct_values_is_145 : 
  min_distinct_values n_series unique_mode_occurrence = 145 :=
by
  sorry

end NUMINAMATH_GPT_minimum_distinct_values_is_145_l2209_220952


namespace NUMINAMATH_GPT_buicks_count_l2209_220924

-- Definitions
def total_cars := 301
def ford_eqn (chevys : ℕ) := 3 + 2 * chevys
def buicks_eqn (chevys : ℕ) := 12 + 8 * chevys

-- Statement
theorem buicks_count (chevys : ℕ) (fords : ℕ) (buicks : ℕ) :
  total_cars = chevys + fords + buicks ∧
  fords = ford_eqn chevys ∧
  buicks = buicks_eqn chevys →
  buicks = 220 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_buicks_count_l2209_220924


namespace NUMINAMATH_GPT_min_colors_required_l2209_220951

-- Defining the color type
def Color := ℕ

-- Defining a 6x6 grid
def Grid := Fin 6 → Fin 6 → Color

-- Defining the conditions of the problem for a valid coloring
def is_valid_coloring (c : Grid) : Prop :=
  (∀ i j k, i ≠ j → c i k ≠ c j k) ∧ -- each row has all cells with different colors
  (∀ i j k, i ≠ j → c k i ≠ c k j) ∧ -- each column has all cells with different colors
  (∀ i j, i ≠ j → c i (i+j) ≠ c j (i+j)) ∧ -- each 45° diagonal has all different colors
  (∀ i j, i ≠ j → (i-j ≥ 0 → c (i-j) i ≠ c (i-j) j) ∧ (j-i ≥ 0 → c i (j-i) ≠ c j (j-i))) -- each 135° diagonal has all different colors

-- The formal statement of the math problem
theorem min_colors_required : ∃ (n : ℕ), (∀ c : Grid, is_valid_coloring c → n ≥ 7) :=
sorry

end NUMINAMATH_GPT_min_colors_required_l2209_220951


namespace NUMINAMATH_GPT_correct_number_of_paths_l2209_220956

-- Define the number of paths for each segment.
def paths_A_to_B : ℕ := 2
def paths_B_to_D : ℕ := 2
def paths_D_to_C : ℕ := 2
def direct_path_A_to_C : ℕ := 1

-- Define the function to calculate the total paths from A to C.
def total_paths_A_to_C : ℕ :=
  (paths_A_to_B * paths_B_to_D * paths_D_to_C) + direct_path_A_to_C

-- Prove that the total number of paths from A to C is 9.
theorem correct_number_of_paths : total_paths_A_to_C = 9 := by
  -- This is where the proof would go, but it is not required for this task.
  sorry

end NUMINAMATH_GPT_correct_number_of_paths_l2209_220956


namespace NUMINAMATH_GPT_whole_numbers_between_sqrts_l2209_220931

theorem whole_numbers_between_sqrts :
  let lower_bound := Real.sqrt 50
  let upper_bound := Real.sqrt 200
  let start := Nat.ceil lower_bound
  let end_ := Nat.floor upper_bound
  ∃ n, n = end_ - start + 1 ∧ n = 7 := by
  sorry

end NUMINAMATH_GPT_whole_numbers_between_sqrts_l2209_220931


namespace NUMINAMATH_GPT_minimum_length_of_segment_PQ_l2209_220947

theorem minimum_length_of_segment_PQ:
  (∀ x y : ℝ, (x^2 + y^2 - 2 * x + 4 * y + 1 = 0) → 
              (xy >= 2) → 
              (x - y >= 0) → 
              (y <= 1) → 
              ℝ) :=
sorry

end NUMINAMATH_GPT_minimum_length_of_segment_PQ_l2209_220947


namespace NUMINAMATH_GPT_time_after_1456_minutes_l2209_220914

noncomputable def hours_in_minutes := 1456 / 60
noncomputable def minutes_remainder := 1456 % 60

def current_time : Nat := 6 * 60  -- 6:00 a.m. in minutes
def added_time : Nat := current_time + 1456

def six_sixteen_am : Nat := (6 * 60) + 16  -- 6:16 a.m. in minutes the next day

theorem time_after_1456_minutes : added_time % (24 * 60) = six_sixteen_am :=
by
  sorry

end NUMINAMATH_GPT_time_after_1456_minutes_l2209_220914


namespace NUMINAMATH_GPT_chi_squared_test_expected_value_correct_l2209_220976
open ProbabilityTheory

section Part1

def n : ℕ := 400
def a : ℕ := 60
def b : ℕ := 20
def c : ℕ := 180
def d : ℕ := 140
def alpha : ℝ := 0.005
def chi_critical : ℝ := 7.879

noncomputable def chi_squared : ℝ :=
  (n * (a * d - b * c) ^ 2) / ((a + b) * (c + d) * (a + c) * (b + d))

theorem chi_squared_test : chi_squared > chi_critical :=
  sorry

end Part1

section Part2

def reward_med : ℝ := 6  -- 60,000 yuan in 10,000 yuan unit
def reward_small : ℝ := 2  -- 20,000 yuan in 10,000 yuan unit
def total_support : ℕ := 12
def total_rewards : ℕ := 9

noncomputable def dist_table : List (ℝ × ℝ) :=
  [(180, 1 / 220),
   (220, 27 / 220),
   (260, 27 / 55),
   (300, 21 / 55)]

noncomputable def expected_value : ℝ :=
  dist_table.foldr (fun (xi : ℝ × ℝ) acc => acc + xi.1 * xi.2) 0

theorem expected_value_correct : expected_value = 270 :=
  sorry

end Part2

end NUMINAMATH_GPT_chi_squared_test_expected_value_correct_l2209_220976


namespace NUMINAMATH_GPT_rectangle_area_l2209_220948

theorem rectangle_area (A1 A2 : ℝ) (h1 : A1 = 40) (h2 : A2 = 10) :
    ∃ n : ℕ, n = 240 ∧ ∃ R : ℝ, R = 2 * Real.sqrt (40 / Real.pi) + 2 * Real.sqrt (10 / Real.pi) ∧ 
               (4 * Real.sqrt (10) / Real.sqrt (Real.pi)) * (6 * Real.sqrt (10) / Real.sqrt (Real.pi)) = n / Real.pi :=
by
  sorry

end NUMINAMATH_GPT_rectangle_area_l2209_220948


namespace NUMINAMATH_GPT_division_quotient_l2209_220942

theorem division_quotient (dividend divisor remainder quotient : Nat) 
  (h_dividend : dividend = 109)
  (h_divisor : divisor = 12)
  (h_remainder : remainder = 1)
  (h_division_equation : dividend = divisor * quotient + remainder)
  : quotient = 9 := 
by
  sorry

end NUMINAMATH_GPT_division_quotient_l2209_220942


namespace NUMINAMATH_GPT_negate_exactly_one_even_l2209_220937

def is_even (n : ℕ) : Prop := n % 2 = 0
def is_odd (n : ℕ) : Prop := ¬ is_even n

theorem negate_exactly_one_even (a b c : ℕ) :
  ¬ ((is_even a ∧ is_odd b ∧ is_odd c) ∨ (is_odd a ∧ is_even b ∧ is_odd c) ∨ (is_odd a ∧ is_odd b ∧ is_even c)) ↔ 
  ((is_odd a ∧ is_odd b ∧ is_odd c) ∨ (is_even a ∧ is_even b) ∨ (is_even a ∧ is_even c) ∨ (is_even b ∧ is_even c)) :=
sorry

end NUMINAMATH_GPT_negate_exactly_one_even_l2209_220937


namespace NUMINAMATH_GPT_isosceles_triangle_largest_angle_l2209_220997

/-- 
  Given an isosceles triangle where one of the angles is 20% smaller than a right angle,
  prove that the measure of one of the two largest angles is 54 degrees.
-/
theorem isosceles_triangle_largest_angle 
  (A B C : ℝ) 
  (triangle_ABC : A + B + C = 180)
  (isosceles_triangle : A = B ∨ A = C ∨ B = C)
  (smaller_angle : A = 0.80 * 90) :
  A = 54 ∨ B = 54 ∨ C = 54 :=
sorry

end NUMINAMATH_GPT_isosceles_triangle_largest_angle_l2209_220997


namespace NUMINAMATH_GPT_new_students_joined_l2209_220961

-- Define conditions
def initial_students : ℕ := 160
def end_year_students : ℕ := 120
def fraction_transferred_out : ℚ := 1 / 3
def total_students_at_start := end_year_students * 3 / 2

-- Theorem statement
theorem new_students_joined : (total_students_at_start - initial_students = 20) :=
by
  -- Placeholder for proof
  sorry

end NUMINAMATH_GPT_new_students_joined_l2209_220961


namespace NUMINAMATH_GPT_range_of_a_l2209_220957

theorem range_of_a (a : ℝ) (h1 : a > 0) (h2 : ∃ x : ℝ, |x - 4| + |x + 3| < a) : a > 7 :=
sorry

end NUMINAMATH_GPT_range_of_a_l2209_220957


namespace NUMINAMATH_GPT_root_expression_value_l2209_220926

theorem root_expression_value (m : ℝ) (h : 2 * m^2 + 3 * m - 1 = 0) : 4 * m^2 + 6 * m - 2019 = -2017 :=
by
  sorry

end NUMINAMATH_GPT_root_expression_value_l2209_220926


namespace NUMINAMATH_GPT_no_integers_six_digit_cyclic_permutation_l2209_220909

theorem no_integers_six_digit_cyclic_permutation (n : ℕ) (a b c d e f : ℕ) (h : 10 ≤ a ∧ a < 10) :
  ¬(n = 5 ∨ n = 6 ∨ n = 8 ∧
    n * (a * 10^5 + b * 10^4 + c * 10^3 + d * 10^2 + e * 10 + f) =
    b * 10^5 + c * 10^4 + d * 10^3 + e * 10^2 + f * 10 + a) :=
by sorry

end NUMINAMATH_GPT_no_integers_six_digit_cyclic_permutation_l2209_220909


namespace NUMINAMATH_GPT_correct_factorization_l2209_220928

-- Define the expressions involved in the options
def option_A (x a b : ℝ) : Prop := x * (a - b) = a * x - b * x
def option_B (x y : ℝ) : Prop := x^2 - 1 + y^2 = (x - 1) * (x + 1) + y^2
def option_C (x : ℝ) : Prop := x^2 - 1 = (x + 1) * (x - 1)
def option_D (x a b c : ℝ) : Prop := a * x + b * x + c = x * (a + b) + c

-- Theorem stating that option C represents true factorization
theorem correct_factorization (x : ℝ) : option_C x := by
  sorry

end NUMINAMATH_GPT_correct_factorization_l2209_220928


namespace NUMINAMATH_GPT_copy_pages_cost_l2209_220911

theorem copy_pages_cost :
  (7 : ℕ) * (n : ℕ) = 3500 * 4 / 7 → n = 2000 :=
by
  sorry

end NUMINAMATH_GPT_copy_pages_cost_l2209_220911


namespace NUMINAMATH_GPT_greatest_of_consecutive_integers_l2209_220923

theorem greatest_of_consecutive_integers (x y z : ℤ) (h1: y = x + 1) (h2: z = x + 2) (h3: x + y + z = 21) : z = 8 :=
by
  sorry

end NUMINAMATH_GPT_greatest_of_consecutive_integers_l2209_220923


namespace NUMINAMATH_GPT_terminal_side_half_angle_l2209_220958

theorem terminal_side_half_angle {k : ℤ} {α : ℝ} 
  (h : 2 * k * π < α ∧ α < 2 * k * π + π / 2) : 
  (k * π < α / 2 ∧ α / 2 < k * π + π / 4) ∨ (k * π + π <= α / 2 ∧ α / 2 < (k + 1) * π + π / 4) :=
sorry

end NUMINAMATH_GPT_terminal_side_half_angle_l2209_220958


namespace NUMINAMATH_GPT_expected_winnings_correct_l2209_220994

def probability_1 := (1:ℚ) / 4
def probability_2 := (1:ℚ) / 4
def probability_3 := (1:ℚ) / 6
def probability_4 := (1:ℚ) / 6
def probability_5 := (1:ℚ) / 8
def probability_6 := (1:ℚ) / 8

noncomputable def expected_winnings : ℚ :=
  (probability_1 + probability_3 + probability_5) * 2 +
  (probability_2 + probability_4) * 4 +
  probability_6 * (-6 + 4)

theorem expected_winnings_correct : expected_winnings = 1.67 := by
  sorry

end NUMINAMATH_GPT_expected_winnings_correct_l2209_220994


namespace NUMINAMATH_GPT_solve_exponent_problem_l2209_220940

theorem solve_exponent_problem
  (h : (1 / 8) * (2 ^ 36) = 8 ^ x) : x = 11 :=
by
  sorry

end NUMINAMATH_GPT_solve_exponent_problem_l2209_220940


namespace NUMINAMATH_GPT_consecutive_composites_l2209_220938

theorem consecutive_composites 
  (a t d r : ℕ) (h_a_comp : ∃ p q, p > 1 ∧ q > 1 ∧ a = p * q)
  (h_t_comp : ∃ p q, p > 1 ∧ q > 1 ∧ t = p * q)
  (h_d_comp : ∃ p q, p > 1 ∧ q > 1 ∧ d = p * q)
  (h_r_pos : r > 0) :
  ∃ n : ℕ, n > 0 ∧ ∀ k : ℕ, k < r → ∃ m : ℕ, m > 1 ∧ m ∣ (a * t^(n + k) + d) :=
  sorry

end NUMINAMATH_GPT_consecutive_composites_l2209_220938


namespace NUMINAMATH_GPT_geometric_sequence_product_identity_l2209_220979

theorem geometric_sequence_product_identity 
  {a : ℕ → ℝ} (is_geometric_sequence : ∃ r, ∀ n, a (n+1) = a n * r)
  (h : a 3 * a 4 * a 6 * a 7 = 81):
  a 1 * a 9 = 9 :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_product_identity_l2209_220979


namespace NUMINAMATH_GPT_train_crossing_time_l2209_220950

/-- Time for a train of length 1500 meters traveling at 108 km/h to cross an electric pole is 50 seconds -/
theorem train_crossing_time (length : ℕ) (speed_kmph : ℕ) 
    (h₁ : length = 1500) (h₂ : speed_kmph = 108) : 
    (length / ((speed_kmph * 1000) / 3600) = 50) :=
by
  sorry

end NUMINAMATH_GPT_train_crossing_time_l2209_220950


namespace NUMINAMATH_GPT_compute_expression_l2209_220998

theorem compute_expression :
  20 * (150 / 3 + 40 / 5 + 16 / 25 + 2) = 1212.8 :=
by
  -- skipping the proof steps
  sorry

end NUMINAMATH_GPT_compute_expression_l2209_220998


namespace NUMINAMATH_GPT_jovana_shells_l2209_220963

theorem jovana_shells (initial_shells : ℕ) (added_shells : ℕ) (total_shells : ℕ) 
  (h_initial : initial_shells = 5) (h_added : added_shells = 12) :
  total_shells = 17 :=
by
  sorry

end NUMINAMATH_GPT_jovana_shells_l2209_220963


namespace NUMINAMATH_GPT_reynald_volleyballs_l2209_220954

def total_balls : ℕ := 145
def soccer_balls : ℕ := 20
def basketballs : ℕ := soccer_balls + 5
def tennis_balls : ℕ := 2 * soccer_balls
def baseballs : ℕ := soccer_balls + 10
def volleyballs : ℕ := total_balls - (soccer_balls + basketballs + tennis_balls + baseballs)

theorem reynald_volleyballs : volleyballs = 30 :=
by
  sorry

end NUMINAMATH_GPT_reynald_volleyballs_l2209_220954


namespace NUMINAMATH_GPT_determine_parallel_planes_l2209_220917

-- Definition of planes and lines with parallelism
structure Plane :=
  (points : Set (ℝ × ℝ × ℝ))

structure Line :=
  (point1 point2 : ℝ × ℝ × ℝ)
  (in_plane : Plane)

def parallel_planes (α β : Plane) : Prop :=
  ∀ (l1 : Line) (l2 : Line), l1.in_plane = α → l2.in_plane = β → (l1 = l2)

def parallel_lines (l1 l2 : Line) : Prop :=
  ∀ p1 p2, l1.point1 = p1 → l1.point2 = p2 → l2.point1 = p1 → l2.point2 = p2


theorem determine_parallel_planes (α β γ : Plane)
  (h1 : parallel_planes γ α)
  (h2 : parallel_planes γ β)
  (l1 l2 : Line)
  (l1_in_alpha : l1.in_plane = α)
  (l2_in_alpha : l2.in_plane = α)
  (parallel_l1_l2 : ¬ (l1 = l2) → parallel_lines l1 l2)
  (l1_parallel_beta : ∀ l, l.in_plane = β → parallel_lines l l1)
  (l2_parallel_beta : ∀ l, l.in_plane = β → parallel_lines l l2) :
  parallel_planes α β := 
sorry

end NUMINAMATH_GPT_determine_parallel_planes_l2209_220917


namespace NUMINAMATH_GPT_correct_op_l2209_220980

-- Declare variables and conditions
variables {a b : ℝ} {m n : ℤ}
variable (ha : a > 0)
variable (hb : b ≠ 0)

-- Define and state the theorem
theorem correct_op (ha : a > 0) (hb : b ≠ 0) : (b / a)^m = a^(-m) * b^m :=
sorry  -- Proof omitted

end NUMINAMATH_GPT_correct_op_l2209_220980


namespace NUMINAMATH_GPT_minimum_value_of_m_l2209_220955

-- Define a function to determine if a number is a perfect square
def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, k * k = n

-- Define a function to determine if a number is a perfect cube
def is_perfect_cube (n : ℕ) : Prop := ∃ k : ℕ, k * k * k = n

-- Our main theorem statement
theorem minimum_value_of_m :
  ∃ m : ℕ, (600 < m ∧ m ≤ 800) ∧
           is_perfect_square (3 * m) ∧
           is_perfect_cube (5 * m) :=
sorry

end NUMINAMATH_GPT_minimum_value_of_m_l2209_220955


namespace NUMINAMATH_GPT_max_value_x_sq_y_l2209_220922

theorem max_value_x_sq_y (x y : ℝ) (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : x + 3 * y = 9) :
  x^2 * y ≤ 36 :=
sorry

end NUMINAMATH_GPT_max_value_x_sq_y_l2209_220922


namespace NUMINAMATH_GPT_ball_hits_ground_approx_time_l2209_220966

-- Conditions
def height (t : ℝ) : ℝ := -6.1 * t^2 + 4.5 * t + 10

-- Main statement to be proved
theorem ball_hits_ground_approx_time :
  ∃ t : ℝ, (height t = 0) ∧ (abs (t - 1.70) < 0.01) :=
sorry

end NUMINAMATH_GPT_ball_hits_ground_approx_time_l2209_220966


namespace NUMINAMATH_GPT_probability_multiple_of_7_condition_l2209_220936

theorem probability_multiple_of_7_condition :
  ∃ (a b : ℕ), 1 ≤ a ∧ a ≤ 100 ∧ 1 ≤ b ∧ b ≤ 100 ∧ a ≠ b ∧ (ab + a + b + 1) % 7 = 0 → 
  (1295 / 4950 = 259 / 990) :=
sorry

end NUMINAMATH_GPT_probability_multiple_of_7_condition_l2209_220936


namespace NUMINAMATH_GPT_anna_not_lose_l2209_220900

theorem anna_not_lose :
  ∀ (cards : Fin 9 → ℕ),
    ∃ (A B C D : ℕ),
      (A + B ≥ C + D) :=
by
  sorry

end NUMINAMATH_GPT_anna_not_lose_l2209_220900


namespace NUMINAMATH_GPT_percentage_decrease_in_area_l2209_220912

variable (L B : ℝ)

def original_area (L B : ℝ) : ℝ := L * B
def new_length (L : ℝ) : ℝ := 0.70 * L
def new_breadth (B : ℝ) : ℝ := 0.85 * B
def new_area (L B : ℝ) : ℝ := new_length L * new_breadth B

theorem percentage_decrease_in_area (L B : ℝ) (hL : L > 0) (hB : B > 0) :
  ((original_area L B - new_area L B) / original_area L B) * 100 = 40.5 :=
by
  sorry

end NUMINAMATH_GPT_percentage_decrease_in_area_l2209_220912


namespace NUMINAMATH_GPT_part1_eq_of_line_l_part2_eq_of_line_l1_l2209_220902

def intersection_point (m n : ℝ × ℝ × ℝ) : ℝ × ℝ := sorry

def line_through_point_eq_dists (P A B : ℝ × ℝ) : ℝ × ℝ × ℝ := sorry
def line_area_triangle (P : ℝ × ℝ) (triangle_area : ℝ) : ℝ × ℝ × ℝ := sorry

-- Conditions defined:
def m : ℝ × ℝ × ℝ := (2, -1, -3)
def n : ℝ × ℝ × ℝ := (1, 1, -3)
def P : ℝ × ℝ := intersection_point m n
def A : ℝ × ℝ := (1, 3)
def B : ℝ × ℝ := (3, 2)
def triangle_area : ℝ := 4

-- Questions translated into Lean 4 statements:
theorem part1_eq_of_line_l : ∃ l : ℝ × ℝ × ℝ, 
  (l = line_through_point_eq_dists P A B) := sorry

theorem part2_eq_of_line_l1 : ∃ l1 : ℝ × ℝ × ℝ,
  (l1 = line_area_triangle P triangle_area) := sorry

end NUMINAMATH_GPT_part1_eq_of_line_l_part2_eq_of_line_l1_l2209_220902


namespace NUMINAMATH_GPT_juan_original_number_l2209_220996

theorem juan_original_number (n : ℤ) 
  (h : ((2 * (n + 3) - 2) / 2) = 8) : 
  n = 6 := 
sorry

end NUMINAMATH_GPT_juan_original_number_l2209_220996


namespace NUMINAMATH_GPT_total_cost_l2209_220913

theorem total_cost (cost_sandwich cost_soda cost_cookie : ℕ)
    (num_sandwich num_soda num_cookie : ℕ) 
    (h1 : cost_sandwich = 4) 
    (h2 : cost_soda = 3) 
    (h3 : cost_cookie = 2) 
    (h4 : num_sandwich = 4) 
    (h5 : num_soda = 6) 
    (h6 : num_cookie = 7):
    cost_sandwich * num_sandwich + cost_soda * num_soda + cost_cookie * num_cookie = 48 :=
by
  sorry

end NUMINAMATH_GPT_total_cost_l2209_220913


namespace NUMINAMATH_GPT_smallest_constant_N_l2209_220965

-- Given that a, b, c are sides of a triangle and in arithmetic progression, prove that
-- (a^2 + b^2 + c^2) / (ab + bc + ca) ≥ 1.

theorem smallest_constant_N
  (a b c : ℝ)
  (habc : a + b > c ∧ a + c > b ∧ b + c > a) -- Triangle inequality
  (hap : ∃ d : ℝ, b = a + d ∧ c = a + 2 * d) -- Arithmetic progression
  : (a^2 + b^2 + c^2) / (a * b + b * c + c * a) ≥ 1 := 
sorry

end NUMINAMATH_GPT_smallest_constant_N_l2209_220965


namespace NUMINAMATH_GPT_range_of_a_l2209_220984

noncomputable def f (a x : ℝ) := (1 / 3) * x^3 - x^2 - 3 * x - a

theorem range_of_a (a : ℝ) : 
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ z ≠ x ∧ f a x = 0 ∧ f a y = 0 ∧ f a z = 0) ↔ (-9 < a ∧ a < 5 / 3) :=
by apply sorry

end NUMINAMATH_GPT_range_of_a_l2209_220984


namespace NUMINAMATH_GPT_red_markers_count_l2209_220993

-- Define the given conditions
def blue_markers : ℕ := 1028
def total_markers : ℕ := 3343

-- Define the red_makers calculation based on the conditions
def red_markers (total_markers blue_markers : ℕ) : ℕ := total_markers - blue_markers

-- Prove that the number of red markers is 2315 given the conditions
theorem red_markers_count : red_markers total_markers blue_markers = 2315 := by
  -- We can skip the proof for this demonstration
  sorry

end NUMINAMATH_GPT_red_markers_count_l2209_220993


namespace NUMINAMATH_GPT_polynomial_evaluation_l2209_220901

theorem polynomial_evaluation (x : ℝ) (h1 : x^2 - 4 * x - 12 = 0) (h2 : 0 < x) : x^3 - 4 * x^2 - 12 * x + 16 = 16 := 
by
  sorry

end NUMINAMATH_GPT_polynomial_evaluation_l2209_220901


namespace NUMINAMATH_GPT_shortest_chord_eqn_of_circle_l2209_220971

theorem shortest_chord_eqn_of_circle 
    (k x y : ℝ)
    (C_eq : x^2 + y^2 - 2*x - 24 = 0)
    (line_l : y = k * (x - 2) - 1) :
  y = x - 3 :=
by
  sorry

end NUMINAMATH_GPT_shortest_chord_eqn_of_circle_l2209_220971


namespace NUMINAMATH_GPT_total_time_correct_l2209_220903

-- Definitions for the conditions
def dean_time : ℕ := 9
def micah_time : ℕ := (2 * dean_time) / 3
def jake_time : ℕ := micah_time + micah_time / 3

-- Proof statement for the total time
theorem total_time_correct : micah_time + dean_time + jake_time = 23 := by
  sorry

end NUMINAMATH_GPT_total_time_correct_l2209_220903


namespace NUMINAMATH_GPT_seonyeong_class_size_l2209_220982

theorem seonyeong_class_size :
  (12 * 4 + 3) - 12 = 39 :=
by
  sorry

end NUMINAMATH_GPT_seonyeong_class_size_l2209_220982


namespace NUMINAMATH_GPT_regular_polygons_constructible_l2209_220929

-- Define a right triangle where the smaller leg is half the length of the hypotenuse
structure RightTriangle30_60_90 :=
(smaller_leg hypotenuse : ℝ)
(ratio : smaller_leg = hypotenuse / 2)

-- Define the constructibility of polygons
def canConstructPolygon (n: ℕ) : Prop :=
n = 3 ∨ n = 4 ∨ n = 6 ∨ n = 12

theorem regular_polygons_constructible (T : RightTriangle30_60_90) :
  ∀ n : ℕ, canConstructPolygon n :=
by
  intro n
  sorry

end NUMINAMATH_GPT_regular_polygons_constructible_l2209_220929


namespace NUMINAMATH_GPT_cos_double_angle_tan_sum_angles_l2209_220953

variable (α β : ℝ)
variable (α_acute : 0 < α ∧ α < π / 2)
variable (β_acute : 0 < β ∧ β < π / 2)
variable (tan_alpha : Real.tan α = 4 / 3)
variable (sin_alpha_minus_beta : Real.sin (α - β) = - (Real.sqrt 5) / 5)

/- Prove that cos 2α = -7/25 given the conditions -/
theorem cos_double_angle :
  Real.cos (2 * α) = -7 / 25 :=
by
  sorry

/- Prove that tan (α + β) = -41/38 given the conditions -/
theorem tan_sum_angles :
  Real.tan (α + β) = -41 / 38 :=
by
  sorry

end NUMINAMATH_GPT_cos_double_angle_tan_sum_angles_l2209_220953


namespace NUMINAMATH_GPT_simplify_fraction_l2209_220960

-- Define the numerator and denominator
def numerator := 5^4 + 5^2
def denominator := 5^3 - 5

-- Define the simplified fraction
def simplified_fraction := 65 / 12

-- The proof problem statement
theorem simplify_fraction : (numerator / denominator) = simplified_fraction := 
by 
   -- Proof will go here
   sorry

end NUMINAMATH_GPT_simplify_fraction_l2209_220960


namespace NUMINAMATH_GPT_total_items_l2209_220975

theorem total_items (B M C : ℕ) 
  (h1 : B = 58) 
  (h2 : B = M + 18) 
  (h3 : B = C - 27) : 
  B + M + C = 183 :=
by 
  sorry

end NUMINAMATH_GPT_total_items_l2209_220975


namespace NUMINAMATH_GPT_no_six_digit_numbers_exists_l2209_220962

theorem no_six_digit_numbers_exists :
  ¬(∃ (N : Fin 6 → Fin 720), ∀ (a b c : Fin 6), a ≠ b → a ≠ c → b ≠ c →
  (∃ (i : Fin 6), N i == 720)) := sorry

end NUMINAMATH_GPT_no_six_digit_numbers_exists_l2209_220962


namespace NUMINAMATH_GPT_additional_amount_per_10_cents_l2209_220918

-- Definitions of the given conditions
def expected_earnings_per_share : ℝ := 0.80
def dividend_ratio : ℝ := 0.5
def actual_earnings_per_share : ℝ := 1.10
def shares_owned : ℕ := 600
def total_dividend_paid : ℝ := 312

-- Proof statement
theorem additional_amount_per_10_cents (additional_amount : ℝ) :
  (total_dividend_paid - (shares_owned * (expected_earnings_per_share * dividend_ratio))) / shares_owned / 
  ((actual_earnings_per_share - expected_earnings_per_share) / 0.10) = additional_amount :=
sorry

end NUMINAMATH_GPT_additional_amount_per_10_cents_l2209_220918


namespace NUMINAMATH_GPT_delta_solution_l2209_220949

theorem delta_solution : ∃ Δ : ℤ, 4 * (-3) = Δ - 1 ∧ Δ = -11 :=
by
  -- Using the condition 4(-3) = Δ - 1, 
  -- we need to prove that Δ = -11
  sorry

end NUMINAMATH_GPT_delta_solution_l2209_220949


namespace NUMINAMATH_GPT_unique_function_l2209_220915

def functional_equation (f : ℤ → ℤ) : Prop :=
  ∀ x y : ℤ, f (-f x - f y) = 1 - x - y

theorem unique_function :
  ∀ f : ℤ → ℤ, (functional_equation f) → (∀ x : ℤ, f x = x - 1) :=
by
  intros f h
  sorry

end NUMINAMATH_GPT_unique_function_l2209_220915


namespace NUMINAMATH_GPT_parametric_eq_C1_rectangular_eq_C2_max_dist_PQ_l2209_220935

-- Curve C1 given by x^2 / 9 + y^2 = 1, prove its parametric form
theorem parametric_eq_C1 (α : ℝ) : 
  (∃ (x y : ℝ), x = 3 * Real.cos α ∧ y = Real.sin α ∧ (x ^ 2 / 9 + y ^ 2 = 1)) := 
sorry

-- Curve C2 given by ρ^2 - 8ρ sin θ + 15 = 0, prove its rectangular form
theorem rectangular_eq_C2 (ρ θ : ℝ) : 
  (∃ (x y : ℝ), x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ ∧ 
    (ρ ^ 2 - 8 * ρ * Real.sin θ + 15 = 0) ↔ (x ^ 2 + y ^ 2 - 8 * y + 15 = 0)) := 
sorry

-- Prove the maximum value of |PQ|
theorem max_dist_PQ : 
  (∃ (P Q : ℝ × ℝ), 
    (P = (3 * Real.cos α, Real.sin α)) ∧ 
    (Q = (0, 4)) ∧ 
    (∀ α : ℝ, Real.sqrt ((3 * Real.cos α) ^ 2 + (Real.sin α - 4) ^ 2) ≤ 8)) := 
sorry

end NUMINAMATH_GPT_parametric_eq_C1_rectangular_eq_C2_max_dist_PQ_l2209_220935


namespace NUMINAMATH_GPT_rachel_problems_solved_each_minute_l2209_220972

-- Definitions and conditions
def problems_solved_each_minute (x : ℕ) : Prop :=
  let problems_before_bed := 12 * x
  let problems_at_lunch := 16
  let total_problems := problems_before_bed + problems_at_lunch
  total_problems = 76

-- Theorem to be proved
theorem rachel_problems_solved_each_minute : ∃ x : ℕ, problems_solved_each_minute x ∧ x = 5 :=
by
  sorry

end NUMINAMATH_GPT_rachel_problems_solved_each_minute_l2209_220972


namespace NUMINAMATH_GPT_range_of_a_l2209_220964

noncomputable def f (a x : ℝ) : ℝ := x * Real.exp x + (1 / 2) * a * x^2 + a * x

theorem range_of_a (a : ℝ) : 
    (∀ x : ℝ, 2 * Real.exp (f a x) + Real.exp 1 + 2 ≥ 0) ↔ (0 ≤ a ∧ a ≤ 1) := 
sorry

end NUMINAMATH_GPT_range_of_a_l2209_220964


namespace NUMINAMATH_GPT_two_digits_same_in_three_digit_numbers_l2209_220932

theorem two_digits_same_in_three_digit_numbers (h1 : (100 : ℕ) ≤ n) (h2 : n < 600) : 
  ∃ n, n = 140 := sorry

end NUMINAMATH_GPT_two_digits_same_in_three_digit_numbers_l2209_220932


namespace NUMINAMATH_GPT_certain_percentage_l2209_220970

theorem certain_percentage (P : ℝ) : 
  0.15 * P * 0.50 * 4000 = 90 → P = 0.3 :=
by
  sorry

end NUMINAMATH_GPT_certain_percentage_l2209_220970


namespace NUMINAMATH_GPT_like_terms_exponents_l2209_220934

theorem like_terms_exponents {m n : ℕ} (h1 : 4 * a * b^n = 4 * (a^1) * (b^n)) (h2 : -2 * a^m * b^4 = -2 * (a^m) * (b^4)) :
  (m = 1 ∧ n = 4) :=
by sorry

end NUMINAMATH_GPT_like_terms_exponents_l2209_220934


namespace NUMINAMATH_GPT_arithmetic_sequence_part_a_arithmetic_sequence_part_b_l2209_220983

theorem arithmetic_sequence_part_a (e u k : ℕ) (n : ℕ) 
  (h1 : e = 1) 
  (h2 : u = 1000) 
  (h3 : k = 343) 
  (h4 : n = 100) : ¬ (∃ d m, e + (m - 1) * d = k ∧ u = e + (n - 1) * d ∧ 1 < m ∧ m < n) :=
by sorry

theorem arithmetic_sequence_part_b (e u k : ℝ) (n : ℕ) 
  (h1 : e = 81 * Real.sqrt 2 - 64 * Real.sqrt 3) 
  (h2 : u = 54 * Real.sqrt 2 - 28 * Real.sqrt 3)
  (h3 : k = 69 * Real.sqrt 2 - 48 * Real.sqrt 3)
  (h4 : n = 100) : (∃ d m, e + (m - 1) * d = k ∧ u = e + (n - 1) * d ∧ 1 < m ∧ m < n) :=
by sorry

end NUMINAMATH_GPT_arithmetic_sequence_part_a_arithmetic_sequence_part_b_l2209_220983


namespace NUMINAMATH_GPT_find_correct_value_l2209_220973

theorem find_correct_value (k : ℕ) (h1 : 173 * 240 = 41520) (h2 : 41520 / 48 = 865) : k * 48 = 173 * 240 → k = 865 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_find_correct_value_l2209_220973


namespace NUMINAMATH_GPT_max_lift_times_l2209_220991

theorem max_lift_times (n : ℕ) :
  (2 * 30 * 10) = (2 * 25 * n) → n = 12 :=
by
  sorry

end NUMINAMATH_GPT_max_lift_times_l2209_220991


namespace NUMINAMATH_GPT_xiao_ming_cube_division_l2209_220978

theorem xiao_ming_cube_division (large_edge small_cubes : ℕ)
  (large_edge_eq : large_edge = 4)
  (small_cubes_eq : small_cubes = 29)
  (total_volume : large_edge ^ 3 = 64) :
  ∃ (small_edge_1_cube : ℕ), small_edge_1_cube = 24 ∧ small_cubes = 29 ∧ 
  small_edge_1_cube + (small_cubes - small_edge_1_cube) * 8 = 64 := 
by
  -- We only need to assert the existence here as per the instruction.
  sorry

end NUMINAMATH_GPT_xiao_ming_cube_division_l2209_220978


namespace NUMINAMATH_GPT_inequality_of_abc_l2209_220969

theorem inequality_of_abc (a b c : ℝ) (hpos_a : 0 < a) (hpos_b : 0 < b) (hpos_c : 0 < c) (h_abc : a * b * c = 1) :
  1 / (a^3 * (b + c)) + 1 / (b^3 * (c + a)) + 1 / (c^3 * (a + b)) ≥ 3 / 2 :=
by {
  sorry
}

end NUMINAMATH_GPT_inequality_of_abc_l2209_220969


namespace NUMINAMATH_GPT_incorrect_major_premise_l2209_220977

-- Define a structure for Line and Plane
structure Line : Type :=
  (name : String)

structure Plane : Type :=
  (name : String)

-- Define relationships: parallel and contains
def parallel (l1 l2 : Line) : Prop := sorry
def line_in_plane (l : Line) (p : Plane) : Prop := sorry
def parallel_line_plane (l : Line) (p : Plane) : Prop := sorry

-- Conditions
variables 
  (a b : Line) 
  (α : Plane)
  (H1 : line_in_plane a α) 
  (H2 : parallel_line_plane b α)

-- Major premise to disprove
def major_premise (l : Line) (p : Plane) : Prop :=
  ∀ (l_in : Line), line_in_plane l_in p → parallel l l_in

-- State the problem
theorem incorrect_major_premise : ¬major_premise b α :=
sorry

end NUMINAMATH_GPT_incorrect_major_premise_l2209_220977


namespace NUMINAMATH_GPT_new_person_weight_l2209_220946

theorem new_person_weight (avg_increase : ℝ) (num_persons : ℕ) (old_weight : ℝ) : 
    avg_increase = 2.5 ∧ num_persons = 8 ∧ old_weight = 65 → 
    (old_weight + num_persons * avg_increase = 85) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_new_person_weight_l2209_220946


namespace NUMINAMATH_GPT_graph_of_equation_is_two_lines_l2209_220943

theorem graph_of_equation_is_two_lines :
  ∀ x y : ℝ, x^2 - 16*y^2 - 8*x + 16 = 0 ↔ (x = 4 + 4*y ∨ x = 4 - 4*y) :=
by
  sorry

end NUMINAMATH_GPT_graph_of_equation_is_two_lines_l2209_220943


namespace NUMINAMATH_GPT_fraction_calculation_l2209_220908

theorem fraction_calculation : (4 / 9 + 1 / 9) / (5 / 8 - 1 / 8) = 10 / 9 := by
  sorry

end NUMINAMATH_GPT_fraction_calculation_l2209_220908


namespace NUMINAMATH_GPT_weight_of_person_replaced_l2209_220919

theorem weight_of_person_replaced (W_new : ℝ) (h1 : W_new = 74) (h2 : (W_new - W_old) = 9) : W_old = 65 := 
by
  sorry

end NUMINAMATH_GPT_weight_of_person_replaced_l2209_220919


namespace NUMINAMATH_GPT_max_value_of_angle_B_l2209_220986

theorem max_value_of_angle_B (a b c : ℝ) (A B C : ℝ) 
  (h1: a + c = 2 * b)
  (h2: a^2 + b^2 - 2*a*b <= c^2 - 2*b*c - 2*a*c)
  (h3: A + B + C = π)
  (h4: 0 < A ∧ A < π) :  
  B ≤ π / 3 :=
sorry

end NUMINAMATH_GPT_max_value_of_angle_B_l2209_220986


namespace NUMINAMATH_GPT_max_a_plus_b_l2209_220945

theorem max_a_plus_b (a b : ℝ) (h1 : 4 * a + 3 * b ≤ 10) (h2 : 3 * a + 6 * b ≤ 12) : a + b ≤ 14 / 5 := 
sorry

end NUMINAMATH_GPT_max_a_plus_b_l2209_220945


namespace NUMINAMATH_GPT_coin_difference_l2209_220995

theorem coin_difference : 
  ∀ (c : ℕ), c = 50 → 
  (∃ (n m : ℕ), 
    (n ≥ m) ∧ 
    (∃ (a b d e : ℕ), n = a + b + d + e ∧ 5 * a + 10 * b + 20 * d + 25 * e = c) ∧
    (∃ (p q r s : ℕ), m = p + q + r + s ∧ 5 * p + 10 * q + 20 * r + 25 * s = c) ∧ 
    (n - m = 8)) :=
by
  sorry

end NUMINAMATH_GPT_coin_difference_l2209_220995


namespace NUMINAMATH_GPT_fraction_female_to_male_fraction_male_to_total_l2209_220904

-- Define the number of male and female students
def num_male_students : ℕ := 30
def num_female_students : ℕ := 24
def total_students : ℕ := num_male_students + num_female_students

-- Prove the fraction of female students to male students
theorem fraction_female_to_male :
  (num_female_students : ℚ) / num_male_students = 4 / 5 :=
by sorry

-- Prove the fraction of male students to total students
theorem fraction_male_to_total :
  (num_male_students : ℚ) / total_students = 5 / 9 :=
by sorry

end NUMINAMATH_GPT_fraction_female_to_male_fraction_male_to_total_l2209_220904


namespace NUMINAMATH_GPT_find_y_perpendicular_l2209_220959

theorem find_y_perpendicular (y : ℝ) (A B : ℝ × ℝ) (a : ℝ × ℝ)
  (hA : A = (-1, 2))
  (hB : B = (2, y))
  (ha : a = (2, 1))
  (h_perp : (B.1 - A.1) * a.1 + (B.2 - A.2) * a.2 = 0) :
  y = -4 :=
sorry

end NUMINAMATH_GPT_find_y_perpendicular_l2209_220959


namespace NUMINAMATH_GPT_garage_sale_records_l2209_220974

/--
Roberta started off with 8 vinyl records. Her friends gave her 12
records for her birthday and she bought some more at a garage
sale. It takes her 2 days to listen to 1 record. It will take her
100 days to listen to her record collection. Prove that she bought
30 records at the garage sale.
-/
theorem garage_sale_records :
  let initial_records := 8
  let gift_records := 12
  let days_per_record := 2
  let total_listening_days := 100
  let total_records := total_listening_days / days_per_record
  let records_before_sale := initial_records + gift_records
  let records_bought := total_records - records_before_sale
  records_bought = 30 := 
by
  -- Variable assumptions
  let initial_records := 8
  let gift_records := 12
  let days_per_record := 2
  let total_listening_days := 100

  -- Definitions
  let total_records := total_listening_days / days_per_record
  let records_before_sale := initial_records + gift_records
  let records_bought := total_records - records_before_sale

  -- Conclusion to prove
  show records_bought = 30
  sorry

end NUMINAMATH_GPT_garage_sale_records_l2209_220974
