import Mathlib

namespace NUMINAMATH_CALUDE_minimum_point_of_transformed_graph_l1071_107116

-- Define the function representing the transformed graph
def f (x : ℝ) : ℝ := 2 * abs (x + 3) - 7

-- State the theorem
theorem minimum_point_of_transformed_graph :
  ∃ (x : ℝ), f x = f (-3) ∧ ∀ (y : ℝ), f y ≥ f (-3) ∧ f (-3) = -7 :=
sorry

end NUMINAMATH_CALUDE_minimum_point_of_transformed_graph_l1071_107116


namespace NUMINAMATH_CALUDE_cubic_root_form_l1071_107143

theorem cubic_root_form : ∃ (x : ℝ), 
  8 * x^3 - 3 * x^2 - 3 * x - 1 = 0 ∧ 
  x = (Real.rpow 81 (1/3 : ℝ) + Real.rpow 9 (1/3 : ℝ) + 1) / 8 := by
  sorry

end NUMINAMATH_CALUDE_cubic_root_form_l1071_107143


namespace NUMINAMATH_CALUDE_product_sum_theorem_l1071_107149

theorem product_sum_theorem (a b c : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 267) 
  (h2 : a + b + c = 23) : 
  a*b + b*c + a*c = 131 := by
sorry

end NUMINAMATH_CALUDE_product_sum_theorem_l1071_107149


namespace NUMINAMATH_CALUDE_black_squares_10th_row_l1071_107189

def stair_step_squares (n : ℕ) : ℕ :=
  if n = 0 then 0
  else if n = 1 then 1
  else stair_step_squares (n - 1) + 2^(n - 1)

def black_squares (n : ℕ) : ℕ :=
  (stair_step_squares n - 1) / 2

theorem black_squares_10th_row :
  black_squares 10 = 511 := by
  sorry

end NUMINAMATH_CALUDE_black_squares_10th_row_l1071_107189


namespace NUMINAMATH_CALUDE_mary_chopped_chairs_l1071_107118

/-- Represents the number of sticks of wood produced by different furniture types -/
structure FurnitureWood where
  chair : ℕ
  table : ℕ
  stool : ℕ

/-- Represents the furniture Mary chopped up -/
structure ChoppedFurniture where
  chairs : ℕ
  tables : ℕ
  stools : ℕ

/-- Calculates the total number of sticks from chopped furniture -/
def totalSticks (fw : FurnitureWood) (cf : ChoppedFurniture) : ℕ :=
  fw.chair * cf.chairs + fw.table * cf.tables + fw.stool * cf.stools

theorem mary_chopped_chairs :
  ∀ (fw : FurnitureWood) (cf : ChoppedFurniture) (burn_rate hours_warm : ℕ),
    fw.chair = 6 →
    fw.table = 9 →
    fw.stool = 2 →
    burn_rate = 5 →
    hours_warm = 34 →
    cf.tables = 6 →
    cf.stools = 4 →
    totalSticks fw cf = burn_rate * hours_warm →
    cf.chairs = 18 := by
  sorry

end NUMINAMATH_CALUDE_mary_chopped_chairs_l1071_107118


namespace NUMINAMATH_CALUDE_solve_for_Q_l1071_107140

theorem solve_for_Q : ∃ Q : ℝ, (Q ^ 4).sqrt = 32 * (64 ^ (1/6)) → Q = 8 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_Q_l1071_107140


namespace NUMINAMATH_CALUDE_function_difference_theorem_l1071_107112

theorem function_difference_theorem (f g : ℝ → ℝ) (k : ℝ) : 
  (∀ x, f x = 3 * x^2 - 1/x + 5) →
  (∀ x, g x = 2 * x^2 - k) →
  f 3 - g 3 = 6 →
  k = -23/3 := by sorry

end NUMINAMATH_CALUDE_function_difference_theorem_l1071_107112


namespace NUMINAMATH_CALUDE_jackson_pbj_sandwiches_l1071_107182

/-- Calculates the number of peanut butter and jelly sandwiches Jackson eats during the school year -/
def pbj_sandwiches (weeks : ℕ) (wed_holidays : ℕ) (fri_holidays : ℕ) (ham_cheese_interval : ℕ) (wed_absences : ℕ) (fri_absences : ℕ) : ℕ :=
  let total_wed := weeks
  let total_fri := weeks
  let remaining_wed := total_wed - wed_holidays - wed_absences
  let remaining_fri := total_fri - fri_holidays - fri_absences
  let ham_cheese_weeks := weeks / ham_cheese_interval
  let ham_cheese_wed := ham_cheese_weeks
  let ham_cheese_fri := ham_cheese_weeks * 2
  let pbj_wed := remaining_wed - ham_cheese_wed
  let pbj_fri := remaining_fri - ham_cheese_fri
  pbj_wed + pbj_fri

/-- Theorem stating that Jackson eats 37 peanut butter and jelly sandwiches during the school year -/
theorem jackson_pbj_sandwiches :
  pbj_sandwiches 36 2 3 4 1 2 = 37 := by
  sorry

end NUMINAMATH_CALUDE_jackson_pbj_sandwiches_l1071_107182


namespace NUMINAMATH_CALUDE_apple_box_weight_l1071_107178

theorem apple_box_weight (total_weight : ℝ) (pies : ℕ) (apples_per_pie : ℝ) : 
  total_weight / 2 = pies * apples_per_pie →
  pies = 15 →
  apples_per_pie = 4 →
  total_weight = 120 := by
sorry

end NUMINAMATH_CALUDE_apple_box_weight_l1071_107178


namespace NUMINAMATH_CALUDE_sqrt_sum_greater_than_sqrt_of_sum_l1071_107164

theorem sqrt_sum_greater_than_sqrt_of_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  Real.sqrt a + Real.sqrt b > Real.sqrt (a + b) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_greater_than_sqrt_of_sum_l1071_107164


namespace NUMINAMATH_CALUDE_principal_calculation_l1071_107104

theorem principal_calculation (P r : ℝ) : 
  P * r * 2 = 10200 →
  P * ((1 + r)^2 - 1) = 11730 →
  P = 17000 := by
sorry

end NUMINAMATH_CALUDE_principal_calculation_l1071_107104


namespace NUMINAMATH_CALUDE_rotten_oranges_percentage_l1071_107132

/-- Proves that the percentage of rotten oranges is 15% given the conditions -/
theorem rotten_oranges_percentage
  (total_oranges : ℕ)
  (total_bananas : ℕ)
  (rotten_bananas_percentage : ℝ)
  (good_fruits_percentage : ℝ)
  (h1 : total_oranges = 600)
  (h2 : total_bananas = 400)
  (h3 : rotten_bananas_percentage = 6)
  (h4 : good_fruits_percentage = 88.6)
  : (100 - (good_fruits_percentage * (total_oranges + total_bananas) / total_oranges -
     rotten_bananas_percentage * total_bananas / total_oranges)) = 15 :=
by sorry

end NUMINAMATH_CALUDE_rotten_oranges_percentage_l1071_107132


namespace NUMINAMATH_CALUDE_brenda_weighs_220_l1071_107110

def mel_weight : ℕ := 70

def brenda_weight (m : ℕ) : ℕ := 3 * m + 10

theorem brenda_weighs_220 : brenda_weight mel_weight = 220 := by
  sorry

end NUMINAMATH_CALUDE_brenda_weighs_220_l1071_107110


namespace NUMINAMATH_CALUDE_quadratic_expression_values_l1071_107138

theorem quadratic_expression_values (a c : ℝ) : 
  (∀ x : ℝ, a * x^2 + x + c = 10 → x = 1) →
  (∀ x : ℝ, a * x^2 + x + c = 8 → x = -1) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_expression_values_l1071_107138


namespace NUMINAMATH_CALUDE_mo_drink_difference_l1071_107193

theorem mo_drink_difference (n : ℕ) : 
  (n ≥ 0) →  -- n is non-negative
  (2 * n + 5 * 4 = 26) →  -- total cups constraint
  (5 * 4 - 2 * n = 14) :=  -- difference between tea and hot chocolate
by
  sorry

end NUMINAMATH_CALUDE_mo_drink_difference_l1071_107193


namespace NUMINAMATH_CALUDE_male_average_tickets_l1071_107123

/-- Proves that the average number of tickets sold by male members is 58,
    given the overall average, female average, and male-to-female ratio. -/
theorem male_average_tickets (total_members : ℕ) (male_members : ℕ) (female_members : ℕ) :
  male_members > 0 →
  female_members = 2 * male_members →
  (male_members * q + female_members * 70) / total_members = 66 →
  total_members = male_members + female_members →
  q = 58 :=
by sorry

end NUMINAMATH_CALUDE_male_average_tickets_l1071_107123


namespace NUMINAMATH_CALUDE_frog_jumped_farther_l1071_107103

/-- The distance jumped by the grasshopper in inches -/
def grasshopper_jump : ℕ := 9

/-- The distance jumped by the frog in inches -/
def frog_jump : ℕ := 12

/-- The difference between the frog's jump and the grasshopper's jump -/
def jump_difference : ℕ := frog_jump - grasshopper_jump

/-- Theorem stating that the frog jumped 3 inches farther than the grasshopper -/
theorem frog_jumped_farther : jump_difference = 3 := by
  sorry

end NUMINAMATH_CALUDE_frog_jumped_farther_l1071_107103


namespace NUMINAMATH_CALUDE_no_distinct_complex_numbers_satisfying_equations_l1071_107122

theorem no_distinct_complex_numbers_satisfying_equations :
  ∀ (a b c d : ℂ),
  (a^3 - b*c*d = b^3 - c*d*a) ∧
  (b^3 - c*d*a = c^3 - d*a*b) ∧
  (c^3 - d*a*b = d^3 - a*b*c) →
  ¬(a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) :=
by sorry

end NUMINAMATH_CALUDE_no_distinct_complex_numbers_satisfying_equations_l1071_107122


namespace NUMINAMATH_CALUDE_secret_code_is_819_l1071_107168

/-- Represents a three-digit code -/
structure Code :=
  (d1 d2 d3 : Nat)
  (h1 : d1 < 10)
  (h2 : d2 < 10)
  (h3 : d3 < 10)

/-- Checks if a given digit is in the correct position in the code -/
def correctPosition (c : Code) (pos : Nat) (digit : Nat) : Prop :=
  match pos with
  | 1 => c.d1 = digit
  | 2 => c.d2 = digit
  | 3 => c.d3 = digit
  | _ => False

/-- Checks if a given digit is in the code but in the wrong position -/
def correctButWrongPosition (c : Code) (pos : Nat) (digit : Nat) : Prop :=
  (c.d1 = digit ∨ c.d2 = digit ∨ c.d3 = digit) ∧ ¬correctPosition c pos digit

/-- Represents the clues given in the problem -/
def clues (c : Code) : Prop :=
  (∃ d, (d = 0 ∨ d = 7 ∨ d = 9) ∧ (correctPosition c 1 d ∨ correctPosition c 2 d ∨ correctPosition c 3 d)) ∧
  (c.d1 ≠ 0 ∧ c.d2 ≠ 3 ∧ c.d3 ≠ 2) ∧
  (∃ d1 d2, (d1 = 1 ∨ d1 = 0 ∨ d1 = 8) ∧ (d2 = 1 ∨ d2 = 0 ∨ d2 = 8) ∧ d1 ≠ d2 ∧
    correctButWrongPosition c 1 d1 ∧ correctButWrongPosition c 2 d2) ∧
  (∃ d, (d = 9 ∨ d = 2 ∨ d = 6) ∧ correctButWrongPosition c 1 d) ∧
  (∃ d, (d = 6 ∨ d = 7 ∨ d = 8) ∧ correctButWrongPosition c 2 d)

theorem secret_code_is_819 : ∀ c : Code, clues c → c.d1 = 8 ∧ c.d2 = 1 ∧ c.d3 = 9 := by
  sorry

end NUMINAMATH_CALUDE_secret_code_is_819_l1071_107168


namespace NUMINAMATH_CALUDE_initial_workers_correct_l1071_107180

/-- The number of workers initially working on the job -/
def initial_workers : ℕ := 6

/-- The number of days to finish the job initially -/
def initial_days : ℕ := 8

/-- The number of days worked before new workers join -/
def days_before_join : ℕ := 3

/-- The number of new workers that join -/
def new_workers : ℕ := 4

/-- The number of additional days needed to finish the job after new workers join -/
def additional_days : ℕ := 3

/-- Theorem stating that the initial number of workers is correct -/
theorem initial_workers_correct : 
  initial_workers * initial_days = 
  initial_workers * days_before_join + 
  (initial_workers + new_workers) * additional_days := by
  sorry

#check initial_workers_correct

end NUMINAMATH_CALUDE_initial_workers_correct_l1071_107180


namespace NUMINAMATH_CALUDE_binomial_expansion_102_l1071_107167

theorem binomial_expansion_102 : 
  102^4 - 4 * 102^3 + 6 * 102^2 - 4 * 102 + 1 = 104060401 := by
  sorry

end NUMINAMATH_CALUDE_binomial_expansion_102_l1071_107167


namespace NUMINAMATH_CALUDE_train_speed_problem_l1071_107124

theorem train_speed_problem (length1 length2 speed1 time : ℝ) 
  (h1 : length1 = 145)
  (h2 : length2 = 165)
  (h3 : speed1 = 60)
  (h4 : time = 8)
  (h5 : speed1 > 0) :
  let total_length := length1 + length2
  let relative_speed := total_length / time
  let speed2 := relative_speed - speed1
  speed2 = 79.5 := by
sorry

end NUMINAMATH_CALUDE_train_speed_problem_l1071_107124


namespace NUMINAMATH_CALUDE_complex_product_l1071_107187

theorem complex_product (z₁ z₂ : ℂ) (h1 : Complex.abs z₁ = 1) (h2 : Complex.abs z₂ = 1)
  (h3 : z₁ + z₂ = Complex.mk (-7/5) (1/5)) :
  z₁ * z₂ = Complex.mk (24/25) (-7/25) := by
  sorry

end NUMINAMATH_CALUDE_complex_product_l1071_107187


namespace NUMINAMATH_CALUDE_count_unique_polygonal_chains_l1071_107134

/-- The number of unique closed 2n-segment polygonal chains on an n x n grid -/
def uniquePolygonalChains (n : ℕ) : ℕ :=
  (n.factorial * (n - 1).factorial) / 2

/-- Theorem stating the number of unique closed 2n-segment polygonal chains
    that can be drawn on an n x n grid, passing through all horizontal and
    vertical lines exactly once -/
theorem count_unique_polygonal_chains (n : ℕ) (h : n > 0) :
  uniquePolygonalChains n = (n.factorial * (n - 1).factorial) / 2 := by
  sorry

end NUMINAMATH_CALUDE_count_unique_polygonal_chains_l1071_107134


namespace NUMINAMATH_CALUDE_percentage_of_360_l1071_107151

theorem percentage_of_360 : (42 : ℝ) / 100 * 360 = 151.2 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_360_l1071_107151


namespace NUMINAMATH_CALUDE_sum_representations_is_fibonacci_l1071_107184

/-- Represents the number of ways to sum to n using 1s and 2s -/
def sumRepresentations (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | 1 => 1
  | n+2 => sumRepresentations (n+1) + sumRepresentations n

/-- The n-th Fibonacci number -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n+2 => fib (n+1) + fib n

theorem sum_representations_is_fibonacci (n : ℕ) :
  sumRepresentations n = fib (n+1) := by
  sorry

end NUMINAMATH_CALUDE_sum_representations_is_fibonacci_l1071_107184


namespace NUMINAMATH_CALUDE_quadratic_always_positive_l1071_107131

theorem quadratic_always_positive (k : ℝ) :
  (∀ x : ℝ, 0 < x ∧ x < 1 → x^2 - 2*k*x + 2*k - 1 > 0) ↔ k ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_always_positive_l1071_107131


namespace NUMINAMATH_CALUDE_flower_garden_area_proof_l1071_107128

/-- The area of a circular flower garden -/
def flower_garden_area (radius : ℝ) (pi : ℝ) : ℝ :=
  pi * radius ^ 2

/-- Proof that the area of a circular flower garden with radius 0.6 meters is 1.08 square meters, given that π is assumed to be 3 -/
theorem flower_garden_area_proof :
  let radius : ℝ := 0.6
  let pi : ℝ := 3
  flower_garden_area radius pi = 1.08 := by
  sorry

end NUMINAMATH_CALUDE_flower_garden_area_proof_l1071_107128


namespace NUMINAMATH_CALUDE_cubic_equation_solution_l1071_107144

theorem cubic_equation_solution (y : ℝ) (h : y ≠ 0) :
  (3 * y)^5 = (9 * y)^4 ↔ y = 27 := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_solution_l1071_107144


namespace NUMINAMATH_CALUDE_function_always_positive_l1071_107172

/-- A function satisfying the given differential inequality is always positive -/
theorem function_always_positive
  (f : ℝ → ℝ)
  (hf : Differentiable ℝ f)
  (hf' : Differentiable ℝ (deriv f))
  (h : ∀ x, x * (deriv^[2] f x) + 2 * f x > x^2) :
  ∀ x, f x > 0 := by sorry

end NUMINAMATH_CALUDE_function_always_positive_l1071_107172


namespace NUMINAMATH_CALUDE_largest_n_proof_l1071_107179

/-- The largest value of n for which 6x^2 + nx + 72 can be factored with integer coefficients -/
def largest_n : ℕ := 433

/-- Checks if a pair of integers (A, B) is a valid factorization for 6x^2 + nx + 72 -/
def is_valid_factorization (n A B : ℤ) : Prop :=
  A * B = 72 ∧ n = 6 * B + A

/-- The theorem stating that 433 is the largest value of n for which 
    6x^2 + nx + 72 can be factored as (6x + A)(x + B) with A and B integers -/
theorem largest_n_proof :
  (∀ n : ℤ, (∃ A B : ℤ, is_valid_factorization n A B) → n ≤ largest_n) ∧
  (∃ A B : ℤ, is_valid_factorization largest_n A B) :=
sorry

end NUMINAMATH_CALUDE_largest_n_proof_l1071_107179


namespace NUMINAMATH_CALUDE_probability_correct_l1071_107199

-- Define the number of red and blue marbles
def red_marbles : ℕ := 15
def blue_marbles : ℕ := 9

-- Define the total number of marbles
def total_marbles : ℕ := red_marbles + blue_marbles

-- Define the number of marbles to be selected
def selected_marbles : ℕ := 4

-- Define the probability of selecting 2 red and 2 blue marbles
def probability_two_red_two_blue : ℚ := 4 / 27

-- Theorem statement
theorem probability_correct : 
  (Nat.choose red_marbles 2 * Nat.choose blue_marbles 2) / Nat.choose total_marbles selected_marbles = probability_two_red_two_blue := by
  sorry

end NUMINAMATH_CALUDE_probability_correct_l1071_107199


namespace NUMINAMATH_CALUDE_fibonacci_sum_equals_two_l1071_107145

-- Define the Fibonacci sequence
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

-- Define the sum of the series
noncomputable def fibonacciSum : ℝ := ∑' n, (fib n : ℝ) / 2^n

-- Theorem statement
theorem fibonacci_sum_equals_two : fibonacciSum = 2 := by
  sorry

end NUMINAMATH_CALUDE_fibonacci_sum_equals_two_l1071_107145


namespace NUMINAMATH_CALUDE_smallest_k_is_2011_l1071_107165

def is_valid_sequence (s : ℕ → ℕ) : Prop :=
  (∀ n, s n < s (n + 1)) ∧
  (∀ n, (1005 ∣ s n) ∨ (1006 ∣ s n)) ∧
  (∀ n, ¬(97 ∣ s n)) ∧
  (∀ n, s (n + 1) - s n ≤ 2011)

theorem smallest_k_is_2011 :
  ∀ k : ℕ, (∃ s : ℕ → ℕ, is_valid_sequence s ∧ ∀ n, s (n + 1) - s n ≤ k) → k ≥ 2011 :=
sorry

end NUMINAMATH_CALUDE_smallest_k_is_2011_l1071_107165


namespace NUMINAMATH_CALUDE_teacher_age_l1071_107181

theorem teacher_age (n : ℕ) (initial_avg : ℚ) (student_age : ℕ) (final_avg : ℚ) 
  (h1 : n = 30)
  (h2 : initial_avg = 10)
  (h3 : student_age = 11)
  (h4 : final_avg = 11) :
  (n : ℚ) * initial_avg - student_age + (n - 1 : ℚ) * final_avg - ((n - 1 : ℚ) * initial_avg - student_age) = 30 := by
  sorry

end NUMINAMATH_CALUDE_teacher_age_l1071_107181


namespace NUMINAMATH_CALUDE_john_spending_l1071_107111

def supermarket_spending (x : ℝ) (total : ℝ) : Prop :=
  let fruits_veg := x / 100 * total
  let meat := (1 / 3) * total
  let bakery := (1 / 6) * total
  let candy := 6
  fruits_veg + meat + bakery + candy = total ∧
  candy = 0.1 * total ∧
  x = 40 ∧
  fruits_veg = 24 ∧
  total = 60

theorem john_spending :
  ∃ (x : ℝ) (total : ℝ), supermarket_spending x total :=
sorry

end NUMINAMATH_CALUDE_john_spending_l1071_107111


namespace NUMINAMATH_CALUDE_jelly_bean_problem_l1071_107126

theorem jelly_bean_problem (b c : ℕ) : 
  b = 3 * c →                  -- Initial condition: 3 times as many blueberry as cherry
  b - 15 = 4 * (c - 15) →      -- Condition after eating 15 of each
  b = 135 :=                   -- Conclusion: original number of blueberry jelly beans
by sorry

end NUMINAMATH_CALUDE_jelly_bean_problem_l1071_107126


namespace NUMINAMATH_CALUDE_total_animals_l1071_107171

/-- Given a field with cows, sheep, and goats, calculate the total number of animals -/
theorem total_animals (cows sheep goats : ℕ) 
  (h_cows : cows = 40)
  (h_sheep : sheep = 56)
  (h_goats : goats = 104) :
  cows + sheep + goats = 200 := by
  sorry

end NUMINAMATH_CALUDE_total_animals_l1071_107171


namespace NUMINAMATH_CALUDE_angle_supplement_in_parallel_lines_l1071_107129

-- Define the structure for our parallel lines and transversal system
structure ParallelLinesSystem where
  -- The smallest angle created by the transversal with line m
  smallest_angle : ℝ
  -- The angle between the transversal and line n on the same side
  other_angle : ℝ

-- Define our theorem
theorem angle_supplement_in_parallel_lines 
  (system : ParallelLinesSystem) 
  (h1 : system.smallest_angle = 40)
  (h2 : system.other_angle = 70) :
  180 - system.other_angle = 110 :=
by
  sorry

#check angle_supplement_in_parallel_lines

end NUMINAMATH_CALUDE_angle_supplement_in_parallel_lines_l1071_107129


namespace NUMINAMATH_CALUDE_system_solutions_l1071_107191

-- Define the system of equations
def system (x y a b : ℝ) : Prop :=
  x / (x - a) + y / (y - b) = 2 ∧ a * x + b * y = 2 * a * b

-- Theorem statement
theorem system_solutions (a b : ℝ) :
  (∀ x y : ℝ, system x y a b → x = 2 * a * b / (a + b) ∧ y = 2 * a * b / (a + b)) ∨
  (a = b ∧ ∀ x y : ℝ, system x y a b → x + y = 2 * a) ∨
  (a = -b ∧ ¬∃ x y : ℝ, system x y a b) :=
by sorry

end NUMINAMATH_CALUDE_system_solutions_l1071_107191


namespace NUMINAMATH_CALUDE_triangle_is_equilateral_l1071_107197

-- Define a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the properties of the triangle
def TriangleProperties (t : Triangle) : Prop :=
  (2 * t.a = t.b + t.c) ∧ 
  ((Real.sin t.A)^2 = Real.sin t.B * Real.sin t.C)

-- Define what it means for a triangle to be equilateral
def IsEquilateral (t : Triangle) : Prop :=
  t.a = t.b ∧ t.b = t.c

-- The theorem to be proved
theorem triangle_is_equilateral (t : Triangle) 
  (h : TriangleProperties t) : IsEquilateral t := by
  sorry

end NUMINAMATH_CALUDE_triangle_is_equilateral_l1071_107197


namespace NUMINAMATH_CALUDE_inequality_proof_l1071_107100

theorem inequality_proof (x y : ℝ) (h : x^12 + y^12 ≤ 2) : x^2 + y^2 + x^2*y^2 ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1071_107100


namespace NUMINAMATH_CALUDE_xy_value_l1071_107121

theorem xy_value (x y : ℝ) 
  (h1 : (8:ℝ)^x / 2^(x+y) = 16)
  (h2 : (16:ℝ)^(x+y) / 4^(5*y) = 1024) : 
  x * y = 7/8 := by
sorry

end NUMINAMATH_CALUDE_xy_value_l1071_107121


namespace NUMINAMATH_CALUDE_bus_speed_calculation_prove_bus_speed_l1071_107146

/-- The speed of buses traveling along a country road -/
def bus_speed : ℝ := 46

/-- The speed of the cyclist -/
def cyclist_speed : ℝ := 16

/-- The number of buses counted approaching from the front -/
def buses_front : ℕ := 31

/-- The number of buses counted from behind -/
def buses_behind : ℕ := 15

theorem bus_speed_calculation :
  bus_speed * (buses_front : ℝ) / (bus_speed + cyclist_speed) = 
  bus_speed * (buses_behind : ℝ) / (bus_speed - cyclist_speed) :=
by sorry

/-- The main theorem proving the speed of the buses -/
theorem prove_bus_speed : 
  ∃ (speed : ℝ), speed > 0 ∧ 
  speed * (buses_front : ℝ) / (speed + cyclist_speed) = 
  speed * (buses_behind : ℝ) / (speed - cyclist_speed) ∧
  speed = bus_speed :=
by sorry

end NUMINAMATH_CALUDE_bus_speed_calculation_prove_bus_speed_l1071_107146


namespace NUMINAMATH_CALUDE_overlapping_tape_length_l1071_107115

/-- 
Given three tapes of equal length attached with equal overlapping parts,
this theorem proves the length of one overlapping portion.
-/
theorem overlapping_tape_length 
  (tape_length : ℝ) 
  (attached_length : ℝ) 
  (h1 : tape_length = 217) 
  (h2 : attached_length = 627) : 
  (3 * tape_length - attached_length) / 2 = 12 := by
  sorry

#check overlapping_tape_length

end NUMINAMATH_CALUDE_overlapping_tape_length_l1071_107115


namespace NUMINAMATH_CALUDE_square_circles_intersection_area_l1071_107125

/-- The area of intersection between a square and four circles --/
theorem square_circles_intersection_area :
  let square_side : ℝ := 8
  let circle_radius : ℝ := 3
  let square_area : ℝ := square_side ^ 2
  let circle_sector_area : ℝ := π * circle_radius ^ 2 / 4
  let total_sector_area : ℝ := 4 * circle_sector_area
  let triangle_area : ℝ := (square_side / 2 - circle_radius) ^ 2 / 2
  let total_triangle_area : ℝ := 4 * triangle_area
  let shaded_area : ℝ := square_area - (total_sector_area + total_triangle_area)
  shaded_area = 64 - 9 * π - 18 :=
by sorry

end NUMINAMATH_CALUDE_square_circles_intersection_area_l1071_107125


namespace NUMINAMATH_CALUDE_at_least_one_geq_two_l1071_107196

theorem at_least_one_geq_two (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a + 1/b ≥ 2) ∨ (b + 1/c ≥ 2) ∨ (c + 1/a ≥ 2) := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_geq_two_l1071_107196


namespace NUMINAMATH_CALUDE_arithmetic_sequence_properties_l1071_107117

/-- An arithmetic sequence with specific terms -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  third_term : a 3 = 9
  ninth_term : a 9 = 3

/-- The general term of the arithmetic sequence -/
def generalTerm (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  15 - 3/2 * (n - 1)

/-- The term from which the sequence becomes negative -/
def negativeStartTerm (seq : ArithmeticSequence) : ℕ := 13

theorem arithmetic_sequence_properties (seq : ArithmeticSequence) :
  (∀ n, seq.a n = generalTerm seq n) ∧
  (∀ n, n ≥ negativeStartTerm seq → seq.a n < 0) ∧
  (∀ n, n < negativeStartTerm seq → seq.a n ≥ 0) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_properties_l1071_107117


namespace NUMINAMATH_CALUDE_solve_for_p_l1071_107166

theorem solve_for_p (n m p : ℚ) 
  (h1 : (3 : ℚ) / 4 = n / 48)
  (h2 : (3 : ℚ) / 4 = (m + n) / 96)
  (h3 : (3 : ℚ) / 4 = (p - m) / 160) : 
  p = 156 := by sorry

end NUMINAMATH_CALUDE_solve_for_p_l1071_107166


namespace NUMINAMATH_CALUDE_inequality_proof_l1071_107113

theorem inequality_proof (x y z t : ℝ) 
  (hx : 0 < x ∧ x < 1) 
  (hy : 0 < y ∧ y < 1) 
  (hz : 0 < z ∧ z < 1) 
  (ht : 0 < t ∧ t < 1) : 
  Real.sqrt (x^2 + (1-t)^2) + Real.sqrt (y^2 + (1-x)^2) + 
  Real.sqrt (z^2 + (1-y)^2) + Real.sqrt (t^2 + (1-z)^2) < 4 :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l1071_107113


namespace NUMINAMATH_CALUDE_sandy_net_spent_l1071_107183

def shorts_cost : ℚ := 13.99
def shirt_cost : ℚ := 12.14
def jacket_return : ℚ := 7.43

theorem sandy_net_spent (shorts_cost shirt_cost jacket_return : ℚ) :
  shorts_cost = 13.99 →
  shirt_cost = 12.14 →
  jacket_return = 7.43 →
  shorts_cost + shirt_cost - jacket_return = 18.70 :=
by sorry

end NUMINAMATH_CALUDE_sandy_net_spent_l1071_107183


namespace NUMINAMATH_CALUDE_exists_valid_grid_l1071_107101

-- Define a 5x5 grid of integers (0 or 1)
def Grid := Fin 5 → Fin 5 → Fin 2

-- Function to check if a number is divisible by 3
def divisible_by_three (n : ℕ) : Prop := ∃ k, n = 3 * k

-- Function to sum a 2x2 subgrid
def sum_subgrid (g : Grid) (i j : Fin 4) : ℕ :=
  (g i j).val + (g i (j + 1)).val + (g (i + 1) j).val + (g (i + 1) (j + 1)).val

-- Theorem statement
theorem exists_valid_grid : ∃ (g : Grid),
  (∀ i j : Fin 4, divisible_by_three (sum_subgrid g i j)) ∧
  (∃ i j : Fin 5, g i j = 0) ∧
  (∃ i j : Fin 5, g i j = 1) :=
sorry

end NUMINAMATH_CALUDE_exists_valid_grid_l1071_107101


namespace NUMINAMATH_CALUDE_point_in_fourth_quadrant_l1071_107114

/-- Given the equation (x-2)^2 + √(y+1) = 0, prove that the point (x,y) lies in the fourth quadrant -/
theorem point_in_fourth_quadrant (x y : ℝ) (h : (x - 2)^2 + Real.sqrt (y + 1) = 0) :
  x > 0 ∧ y < 0 := by
  sorry

end NUMINAMATH_CALUDE_point_in_fourth_quadrant_l1071_107114


namespace NUMINAMATH_CALUDE_simplify_expression_l1071_107157

theorem simplify_expression : 
  ((5^2010)^2 - (5^2008)^2) / ((5^2009)^2 - (5^2007)^2) = 25 := by
sorry

end NUMINAMATH_CALUDE_simplify_expression_l1071_107157


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l1071_107176

theorem quadratic_inequality_solution_set (a : ℝ) (h : a < -2) :
  {x : ℝ | a * x^2 + (a - 2) * x - 2 ≥ 0} = {x : ℝ | -1 ≤ x ∧ x ≤ 2/a} := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l1071_107176


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_l1071_107173

theorem quadratic_roots_sum (a b : ℝ) : 
  (∀ x, a * x^2 + b * x - 2 = 0 ↔ x = -2 ∨ x = -1/4) → 
  a + b = -13 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_l1071_107173


namespace NUMINAMATH_CALUDE_problem_statement_l1071_107158

theorem problem_statement (a b c : ℝ) (h1 : a - b = 3) (h2 : a - c = 1) :
  (c - b)^2 - 2*(c - b) + 2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1071_107158


namespace NUMINAMATH_CALUDE_sufficient_but_not_necessary_l1071_107170

theorem sufficient_but_not_necessary (a b : ℝ) : 
  (((0 ≤ a) ∧ (a ≤ 1) ∧ (0 ≤ b) ∧ (b ≤ 1)) → (0 ≤ a * b) ∧ (a * b ≤ 1)) ∧ 
  (∃ (a b : ℝ), ((0 ≤ a * b) ∧ (a * b ≤ 1)) ∧ ¬((0 ≤ a) ∧ (a ≤ 1) ∧ (0 ≤ b) ∧ (b ≤ 1))) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_but_not_necessary_l1071_107170


namespace NUMINAMATH_CALUDE_inequality_solution_l1071_107169

theorem inequality_solution (x : ℝ) : 
  1 / (x - 2) + 4 / (x + 5) ≥ 1 ↔ x ∈ Set.Icc (-1) (7/2) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l1071_107169


namespace NUMINAMATH_CALUDE_tom_spent_seven_tickets_on_hat_l1071_107198

/-- The number of tickets Tom spent on the hat -/
def tickets_spent_on_hat (whack_a_mole_tickets : ℕ) (skee_ball_tickets : ℕ) (tickets_left : ℕ) : ℕ :=
  whack_a_mole_tickets + skee_ball_tickets - tickets_left

/-- Theorem stating that Tom spent 7 tickets on the hat -/
theorem tom_spent_seven_tickets_on_hat :
  tickets_spent_on_hat 32 25 50 = 7 := by
  sorry

end NUMINAMATH_CALUDE_tom_spent_seven_tickets_on_hat_l1071_107198


namespace NUMINAMATH_CALUDE_polynomial_equality_l1071_107153

theorem polynomial_equality (a b c : ℝ) :
  (∀ x : ℝ, 4 * x^2 - 3 * x + 1 = a * (x - 1)^2 + b * (x - 1) + c) →
  4 * a + 2 * b + c = 28 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_equality_l1071_107153


namespace NUMINAMATH_CALUDE_anika_age_l1071_107133

/-- Given the ages of Ben, Clara, and Anika, prove that Anika is 15 years old. -/
theorem anika_age (ben_age clara_age anika_age : ℕ) 
  (h1 : clara_age = ben_age + 5)
  (h2 : anika_age = clara_age - 10)
  (h3 : ben_age = 20) : 
  anika_age = 15 := by
  sorry

end NUMINAMATH_CALUDE_anika_age_l1071_107133


namespace NUMINAMATH_CALUDE_average_weight_of_boys_l1071_107150

theorem average_weight_of_boys (group1_count : ℕ) (group1_avg : ℚ) 
  (group2_count : ℕ) (group2_avg : ℚ) : 
  group1_count = 16 → 
  group1_avg = 50.25 → 
  group2_count = 8 → 
  group2_avg = 45.15 → 
  let total_count := group1_count + group2_count
  let total_weight := group1_count * group1_avg + group2_count * group2_avg
  total_weight / total_count = 48.55 := by
sorry

end NUMINAMATH_CALUDE_average_weight_of_boys_l1071_107150


namespace NUMINAMATH_CALUDE_system_A_is_valid_other_systems_not_valid_l1071_107127

/-- A linear equation in two variables is of the form ax + by = c, where a, b, and c are constants. -/
structure LinearEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A system of two linear equations. -/
structure LinearSystem where
  eq1 : LinearEquation
  eq2 : LinearEquation

/-- The system of equations from option A. -/
def systemA : LinearSystem := {
  eq1 := { a := 1, b := 0, c := 2 }
  eq2 := { a := 0, b := 1, c := 7 }
}

/-- Predicate to check if a given system is a valid system of two linear equations. -/
def isValidLinearSystem (s : LinearSystem) : Prop :=
  -- Additional conditions can be added here if needed
  True

theorem system_A_is_valid : isValidLinearSystem systemA := by
  sorry

/-- The other systems (B, C, D) are not valid systems of two linear equations. -/
theorem other_systems_not_valid :
  ∃ (systemB systemC systemD : LinearSystem),
    ¬ isValidLinearSystem systemB ∧
    ¬ isValidLinearSystem systemC ∧
    ¬ isValidLinearSystem systemD := by
  sorry

end NUMINAMATH_CALUDE_system_A_is_valid_other_systems_not_valid_l1071_107127


namespace NUMINAMATH_CALUDE_confetti_area_difference_l1071_107106

/-- The difference between the area of a square with side length 8 cm and 
    the area of a rectangle with sides 10 cm and 5 cm is 14 cm². -/
theorem confetti_area_difference : 
  let square_side : ℝ := 8
  let rect_length : ℝ := 10
  let rect_width : ℝ := 5
  let square_area := square_side ^ 2
  let rect_area := rect_length * rect_width
  square_area - rect_area = 14 := by sorry

end NUMINAMATH_CALUDE_confetti_area_difference_l1071_107106


namespace NUMINAMATH_CALUDE_lcm_of_ratio_numbers_l1071_107156

theorem lcm_of_ratio_numbers (a b : ℕ) (h1 : a = 21) (h2 : 4 * a = 3 * b) : 
  Nat.lcm a b = 84 := by sorry

end NUMINAMATH_CALUDE_lcm_of_ratio_numbers_l1071_107156


namespace NUMINAMATH_CALUDE_intersection_line_passes_through_intersection_points_l1071_107102

/-- The equation of the first circle -/
def circle1 (x y : ℝ) : Prop := x^2 + y^2 + 4*x - 4*y - 12 = 0

/-- The equation of the second circle -/
def circle2 (x y : ℝ) : Prop := x^2 + y^2 + 2*x + 4*y - 4 = 0

/-- The equation of the line passing through the intersection points -/
def intersection_line (x y : ℝ) : Prop := x - 4*y - 4 = 0

/-- Theorem stating that the intersection_line passes through the intersection points of circle1 and circle2 -/
theorem intersection_line_passes_through_intersection_points :
  ∀ x y : ℝ, circle1 x y ∧ circle2 x y → intersection_line x y :=
by sorry

end NUMINAMATH_CALUDE_intersection_line_passes_through_intersection_points_l1071_107102


namespace NUMINAMATH_CALUDE_max_dominoes_8x9_board_l1071_107174

/-- Represents a checkerboard -/
structure Checkerboard :=
  (rows : ℕ)
  (cols : ℕ)

/-- Represents a domino -/
structure Domino :=
  (length : ℕ)
  (width : ℕ)

/-- The maximum number of non-overlapping dominoes on a checkerboard -/
def max_dominoes (board : Checkerboard) (domino : Domino) (initial_placed : ℕ) : ℕ :=
  sorry

theorem max_dominoes_8x9_board :
  let board : Checkerboard := ⟨8, 9⟩
  let domino : Domino := ⟨2, 1⟩
  let initial_placed : ℕ := 6
  max_dominoes board domino initial_placed = 34 := by
  sorry

end NUMINAMATH_CALUDE_max_dominoes_8x9_board_l1071_107174


namespace NUMINAMATH_CALUDE_axis_of_symmetry_l1071_107175

-- Define the parabola
def parabola (x : ℝ) : ℝ := (2 - x) * x

-- Theorem statement
theorem axis_of_symmetry :
  ∃ (a : ℝ), a = 1 ∧ ∀ (x : ℝ), parabola (a + x) = parabola (a - x) :=
by
  sorry

end NUMINAMATH_CALUDE_axis_of_symmetry_l1071_107175


namespace NUMINAMATH_CALUDE_mary_has_seven_balloons_l1071_107119

-- Define the number of balloons for each person
def fred_balloons : ℕ := 5
def sam_balloons : ℕ := 6
def total_balloons : ℕ := 18

-- Define Mary's balloons as the difference between total and the sum of Fred's and Sam's
def mary_balloons : ℕ := total_balloons - (fred_balloons + sam_balloons)

-- Theorem to prove
theorem mary_has_seven_balloons : mary_balloons = 7 := by
  sorry

end NUMINAMATH_CALUDE_mary_has_seven_balloons_l1071_107119


namespace NUMINAMATH_CALUDE_max_sum_arithmetic_sequence_max_sum_value_max_sum_at_12_max_sum_is_144_l1071_107155

/-- The maximum sum of the first n terms of an arithmetic sequence with a_1 = 23 and d = -2 -/
theorem max_sum_arithmetic_sequence : ℕ → ℝ :=
  fun n => -n^2 + 24*n

/-- The maximum value of the sum of the first n terms is 144 -/
theorem max_sum_value : ∃ (n : ℕ), ∀ (m : ℕ), max_sum_arithmetic_sequence n ≥ max_sum_arithmetic_sequence m :=
by
  sorry

/-- The value of n that maximizes the sum is 12 -/
theorem max_sum_at_12 : ∃ (n : ℕ), n = 12 ∧ ∀ (m : ℕ), max_sum_arithmetic_sequence n ≥ max_sum_arithmetic_sequence m :=
by
  sorry

/-- The maximum sum value is 144 -/
theorem max_sum_is_144 : ∃ (n : ℕ), max_sum_arithmetic_sequence n = 144 ∧ ∀ (m : ℕ), max_sum_arithmetic_sequence n ≥ max_sum_arithmetic_sequence m :=
by
  sorry

end NUMINAMATH_CALUDE_max_sum_arithmetic_sequence_max_sum_value_max_sum_at_12_max_sum_is_144_l1071_107155


namespace NUMINAMATH_CALUDE_pizza_fraction_l1071_107109

theorem pizza_fraction (initial_parts : ℕ) (cuts_per_part : ℕ) (pieces_eaten : ℕ) : 
  initial_parts = 12 →
  cuts_per_part = 2 →
  pieces_eaten = 3 →
  (pieces_eaten : ℚ) / (initial_parts * cuts_per_part : ℚ) = 1 / 8 := by
sorry

end NUMINAMATH_CALUDE_pizza_fraction_l1071_107109


namespace NUMINAMATH_CALUDE_circle_area_sum_l1071_107141

/-- The sum of the areas of an infinite sequence of circles, where the first circle
    has a radius of 3 inches and each subsequent circle's radius is 2/3 of the previous one,
    is equal to 81π/5. -/
theorem circle_area_sum : 
  let r : ℕ → ℝ := λ n => 3 * (2/3)^(n-1)
  let area : ℕ → ℝ := λ n => π * (r n)^2
  (∑' n, area n) = 81*π/5 :=
sorry

end NUMINAMATH_CALUDE_circle_area_sum_l1071_107141


namespace NUMINAMATH_CALUDE_completing_square_result_l1071_107186

/-- Represents the completing the square method applied to a quadratic equation -/
def completing_square (a b c : ℝ) : ℝ × ℝ := sorry

theorem completing_square_result :
  let (p, q) := completing_square 1 4 3
  p = 2 ∧ q = 1 := by sorry

end NUMINAMATH_CALUDE_completing_square_result_l1071_107186


namespace NUMINAMATH_CALUDE_remaining_area_is_27_l1071_107148

/-- Represents the square grid --/
def Grid := Fin 6 → Fin 6 → Bool

/-- The area of a single cell in square centimeters --/
def cellArea : ℝ := 1

/-- The total area of the square in square centimeters --/
def totalArea : ℝ := 36

/-- The area of the dark grey triangles in square centimeters --/
def darkGreyArea : ℝ := 3

/-- The area of the light grey triangles in square centimeters --/
def lightGreyArea : ℝ := 6

/-- The total area of removed triangles in square centimeters --/
def removedArea : ℝ := darkGreyArea + lightGreyArea

/-- Theorem: The area of the remaining shape after cutting out triangles is 27 square cm --/
theorem remaining_area_is_27 : totalArea - removedArea = 27 := by
  sorry

end NUMINAMATH_CALUDE_remaining_area_is_27_l1071_107148


namespace NUMINAMATH_CALUDE_arithmetic_progression_prime_divisibility_l1071_107188

theorem arithmetic_progression_prime_divisibility
  (p : ℕ) (a : ℕ → ℕ) (d : ℕ) 
  (h_prime : Prime p)
  (h_seq : ∀ i ∈ Finset.range p, Prime (a i))
  (h_arith : ∀ i ∈ Finset.range (p - 1), a (i + 1) = a i + d)
  (h_incr : ∀ i ∈ Finset.range (p - 1), a i < a (i + 1))
  (h_greater : p < a 0) :
  p ∣ d := by
sorry

end NUMINAMATH_CALUDE_arithmetic_progression_prime_divisibility_l1071_107188


namespace NUMINAMATH_CALUDE_martha_crayon_count_l1071_107161

def final_crayon_count (initial : ℕ) (first_purchase : ℕ) (contest_win : ℕ) (second_purchase : ℕ) : ℕ :=
  (initial / 2) + first_purchase + contest_win + second_purchase

theorem martha_crayon_count :
  final_crayon_count 18 20 15 25 = 69 := by
  sorry

end NUMINAMATH_CALUDE_martha_crayon_count_l1071_107161


namespace NUMINAMATH_CALUDE_unique_m_value_l1071_107136

def U (m : ℝ) : Set ℝ := {4, m^2 + 2*m - 3, 19}
def A : Set ℝ := {5}

theorem unique_m_value :
  ∃! m : ℝ, (U m \ A = {|4*m - 3|, 4}) ∧ (m^2 + 2*m - 3 = 5) :=
by sorry

end NUMINAMATH_CALUDE_unique_m_value_l1071_107136


namespace NUMINAMATH_CALUDE_fencing_cost_proof_l1071_107192

/-- Calculates the total cost of fencing a rectangular plot -/
def total_fencing_cost (length : ℝ) (breadth : ℝ) (cost_per_meter : ℝ) : ℝ :=
  2 * (length + breadth) * cost_per_meter

/-- Proves that the total cost of fencing the given rectangular plot is 5300 currency units -/
theorem fencing_cost_proof :
  let length : ℝ := 66
  let breadth : ℝ := 34
  let cost_per_meter : ℝ := 26.50
  total_fencing_cost length breadth cost_per_meter = 5300 := by
  sorry

end NUMINAMATH_CALUDE_fencing_cost_proof_l1071_107192


namespace NUMINAMATH_CALUDE_rabbit_calories_l1071_107195

/-- Brandon's hunting scenario -/
structure HuntingScenario where
  squirrels_per_hour : ℕ := 6
  rabbits_per_hour : ℕ := 2
  calories_per_squirrel : ℕ := 300
  calorie_difference : ℕ := 200

/-- Calculates the calories per rabbit in Brandon's hunting scenario -/
def calories_per_rabbit (scenario : HuntingScenario) : ℕ :=
  (scenario.squirrels_per_hour * scenario.calories_per_squirrel - scenario.calorie_difference) / scenario.rabbits_per_hour

/-- Theorem stating that each rabbit has 800 calories in Brandon's scenario -/
theorem rabbit_calories (scenario : HuntingScenario) :
  calories_per_rabbit scenario = 800 := by
  sorry

end NUMINAMATH_CALUDE_rabbit_calories_l1071_107195


namespace NUMINAMATH_CALUDE_quadratic_roots_and_isosceles_triangle_l1071_107147

-- Define the quadratic equation
def quadratic (k : ℝ) (x : ℝ) : ℝ := x^2 - (2*k + 1)*x + k^2 + k

-- Define the discriminant of the quadratic equation
def discriminant (k : ℝ) : ℝ := (2*k + 1)^2 - 4*(k^2 + k)

-- Define a function to check if three sides form an isosceles triangle
def is_isosceles (a b c : ℝ) : Prop := (a = b ∧ a ≠ c) ∨ (a = c ∧ a ≠ b) ∨ (b = c ∧ b ≠ a)

-- Theorem statement
theorem quadratic_roots_and_isosceles_triangle (k : ℝ) :
  (∀ k, discriminant k > 0) ∧
  (∃ x y, x ≠ y ∧ quadratic k x = 0 ∧ quadratic k y = 0 ∧ is_isosceles x y 4) ↔
  (k = 3 ∨ k = 4) :=
sorry

end NUMINAMATH_CALUDE_quadratic_roots_and_isosceles_triangle_l1071_107147


namespace NUMINAMATH_CALUDE_smallest_m_for_meaningful_sqrt_l1071_107105

theorem smallest_m_for_meaningful_sqrt (m : ℤ) : 
  (∀ k : ℤ, k < m → ¬(2*k + 1 ≥ 0)) → (2*m + 1 ≥ 0) → m = 0 :=
by sorry

end NUMINAMATH_CALUDE_smallest_m_for_meaningful_sqrt_l1071_107105


namespace NUMINAMATH_CALUDE_last_two_digits_of_seven_power_l1071_107154

theorem last_two_digits_of_seven_power : 7^30105 ≡ 7 [ZMOD 100] := by
  sorry

end NUMINAMATH_CALUDE_last_two_digits_of_seven_power_l1071_107154


namespace NUMINAMATH_CALUDE_percent_change_condition_l1071_107139

theorem percent_change_condition (p q N : ℝ) 
  (hp : p > 0) (hq : q > 0) (hN : N > 0) (hq_bound : q < 50) :
  N * (1 + 3 * p / 100) * (1 - 2 * q / 100) > N ↔ p > 100 * q / (147 - 3 * q) :=
sorry

end NUMINAMATH_CALUDE_percent_change_condition_l1071_107139


namespace NUMINAMATH_CALUDE_simpsons_formula_volume_l1071_107177

/-- Simpson's formula for volume calculation -/
theorem simpsons_formula_volume
  (S : ℝ → ℝ) -- Cross-sectional area function
  (x₀ x₁ : ℝ) -- Start and end coordinates
  (h : ℝ) -- Height of the figure
  (hpos : 0 < h) -- Height is positive
  (hdiff : h = x₁ - x₀) -- Height definition
  (hquad : ∃ (a b c : ℝ), ∀ x, S x = a * x^2 + b * x + c) -- S is a quadratic polynomial
  :
  (∫ (x : ℝ) in x₀..x₁, S x) = 
    (h / 6) * (S x₀ + 4 * S ((x₀ + x₁) / 2) + S x₁) :=
by sorry

end NUMINAMATH_CALUDE_simpsons_formula_volume_l1071_107177


namespace NUMINAMATH_CALUDE_num_sequences_l1071_107163

/-- The number of distinct elements in the set -/
def num_elements : ℕ := 5

/-- The length of the sequences to be formed -/
def sequence_length : ℕ := 4

/-- The minimum number of times each element appears -/
def min_appearances : ℕ := 3

/-- Theorem stating the number of possible sequences -/
theorem num_sequences (h : min_appearances ≥ sequence_length) :
  num_elements ^ sequence_length = 625 := by sorry

end NUMINAMATH_CALUDE_num_sequences_l1071_107163


namespace NUMINAMATH_CALUDE_base7_25_to_binary_l1071_107162

def base7ToDecimal (n : ℕ) : ℕ :=
  (n / 10) * 7 + (n % 10)

def decimalToBinary (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
  let rec toBinary (m : ℕ) (acc : List ℕ) : List ℕ :=
    if m = 0 then acc else toBinary (m / 2) ((m % 2) :: acc)
  toBinary n []

theorem base7_25_to_binary :
  decimalToBinary (base7ToDecimal 25) = [1, 0, 0, 1, 1] := by
  sorry

end NUMINAMATH_CALUDE_base7_25_to_binary_l1071_107162


namespace NUMINAMATH_CALUDE_correct_calculation_l1071_107159

theorem correct_calculation (x : ℤ) (h : x - 954 = 468) : x + 954 = 2376 :=
by sorry

end NUMINAMATH_CALUDE_correct_calculation_l1071_107159


namespace NUMINAMATH_CALUDE_inequality_and_equality_condition_l1071_107142

theorem inequality_and_equality_condition (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) : 
  a^2 + b^2 + c^2 + (1/a + 1/b + 1/c)^2 ≥ 6 * Real.sqrt 3 ∧ 
  (a^2 + b^2 + c^2 + (1/a + 1/b + 1/c)^2 = 6 * Real.sqrt 3 ↔ 
   a = b ∧ b = c ∧ c = Real.rpow 3 (1/4)) :=
sorry

end NUMINAMATH_CALUDE_inequality_and_equality_condition_l1071_107142


namespace NUMINAMATH_CALUDE_simplify_sqrt_expression_l1071_107135

theorem simplify_sqrt_expression :
  (Real.sqrt 8 + Real.sqrt 3) * Real.sqrt 6 - 4 * Real.sqrt (1/2) = 4 * Real.sqrt 3 + Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_simplify_sqrt_expression_l1071_107135


namespace NUMINAMATH_CALUDE_random_selection_probability_l1071_107108

theorem random_selection_probability (m : ℝ) : 
  m > -1 →
  (1 - (-1)) / (m - (-1)) = 2/5 →
  m = 4 := by
sorry

end NUMINAMATH_CALUDE_random_selection_probability_l1071_107108


namespace NUMINAMATH_CALUDE_percentage_problem_l1071_107190

theorem percentage_problem :
  ∃ x : ℝ, 0.0425 * x = 2.125 ∧ x = 50 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l1071_107190


namespace NUMINAMATH_CALUDE_expression_equality_l1071_107194

theorem expression_equality (x y : ℝ) (h1 : x = 2 * y) (h2 : y ≠ 0) :
  (x - y) * (2 * x + y) = 5 * y^2 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l1071_107194


namespace NUMINAMATH_CALUDE_three_k_values_with_integer_roots_l1071_107137

/-- A quadratic equation with coefficient k has only integer roots -/
def has_only_integer_roots (k : ℝ) : Prop :=
  ∃ r s : ℤ, ∀ x : ℝ, x^2 + k*x + 4*k = 0 ↔ (x = r ∨ x = s)

/-- The set of real numbers k for which the quadratic equation has only integer roots -/
def integer_root_k_values : Set ℝ :=
  {k : ℝ | has_only_integer_roots k}

/-- There are exactly three values of k for which the quadratic equation has only integer roots -/
theorem three_k_values_with_integer_roots :
  ∃ k₁ k₂ k₃ : ℝ, k₁ ≠ k₂ ∧ k₁ ≠ k₃ ∧ k₂ ≠ k₃ ∧
  integer_root_k_values = {k₁, k₂, k₃} :=
sorry

end NUMINAMATH_CALUDE_three_k_values_with_integer_roots_l1071_107137


namespace NUMINAMATH_CALUDE_sheila_work_hours_l1071_107160

/-- Represents Sheila's work schedule and earnings --/
structure WorkSchedule where
  monday_wednesday_friday_hours : ℕ
  tuesday_thursday_hours : ℕ
  weekly_earnings : ℕ
  hourly_rate : ℕ

/-- Theorem stating the number of hours Sheila works on Monday, Wednesday, and Friday --/
theorem sheila_work_hours (schedule : WorkSchedule) : 
  schedule.monday_wednesday_friday_hours = 24 :=
by
  have h1 : schedule.tuesday_thursday_hours = 6 * 2 := by sorry
  have h2 : schedule.weekly_earnings = 468 := by sorry
  have h3 : schedule.hourly_rate = 13 := by sorry
  sorry

end NUMINAMATH_CALUDE_sheila_work_hours_l1071_107160


namespace NUMINAMATH_CALUDE_stamp_collection_problem_l1071_107185

theorem stamp_collection_problem :
  ∃! n : ℕ, n < 3000 ∧
    n % 2 = 1 ∧
    n % 3 = 2 ∧
    n % 4 = 3 ∧
    n % 5 = 4 ∧
    n % 6 = 5 ∧
    n % 7 = 6 ∧
    n % 8 = 7 ∧
    n % 9 = 8 ∧
    n % 10 = 9 ∧
    n = 2519 :=
by sorry

end NUMINAMATH_CALUDE_stamp_collection_problem_l1071_107185


namespace NUMINAMATH_CALUDE_constant_product_l1071_107120

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 - y^2/4 = 1

-- Define the asymptotes
def asymptote (x y : ℝ) : Prop := y = 2*x ∨ y = -2*x

-- Define a point on the right branch of the hyperbola
def right_branch (x y : ℝ) : Prop := hyperbola x y ∧ x > 0

-- Define a line passing through a point
def line_through (x₀ y₀ x y : ℝ) : Prop := ∃ (m b : ℝ), y - y₀ = m * (x - x₀) ∧ y = m*x + b

-- Define the midpoint condition
def is_midpoint (x₀ y₀ x₁ y₁ x₂ y₂ : ℝ) : Prop := x₀ = (x₁ + x₂)/2 ∧ y₀ = (y₁ + y₂)/2

-- Main theorem
theorem constant_product (x₀ y₀ x_A y_A x_B y_B : ℝ) :
  right_branch x₀ y₀ →
  asymptote x_A y_A ∧ asymptote x_B y_B →
  line_through x₀ y₀ x_A y_A ∧ line_through x₀ y₀ x_B y_B →
  is_midpoint x₀ y₀ x_A y_A x_B y_B →
  (x_A^2 + y_A^2) * (x_B^2 + y_B^2) = 25 :=
sorry

end NUMINAMATH_CALUDE_constant_product_l1071_107120


namespace NUMINAMATH_CALUDE_books_in_boxes_l1071_107152

theorem books_in_boxes (total_books : ℕ) (books_per_box : ℕ) (num_boxes : ℕ) : 
  total_books = 24 → books_per_box = 3 → num_boxes * books_per_box = total_books → num_boxes = 8 := by
  sorry

end NUMINAMATH_CALUDE_books_in_boxes_l1071_107152


namespace NUMINAMATH_CALUDE_complement_of_union_l1071_107130

def U : Set ℕ := {x | x > 0 ∧ x < 9}
def M : Set ℕ := {1, 3, 5, 7}
def N : Set ℕ := {5, 6, 7}

theorem complement_of_union : 
  (U \ (M ∪ N)) = {2, 4, 8} := by sorry

end NUMINAMATH_CALUDE_complement_of_union_l1071_107130


namespace NUMINAMATH_CALUDE_convex_ngon_angle_theorem_l1071_107107

theorem convex_ngon_angle_theorem (n : ℕ) : 
  n ≥ 3 →  -- n-gon must have at least 3 sides
  (∃ (x : ℝ), x > 0 ∧ x < 150 ∧ 150 * (n - 1) + x = 180 * (n - 2)) →
  (n = 8 ∨ n = 9 ∨ n = 10 ∨ n = 11) :=
by sorry

end NUMINAMATH_CALUDE_convex_ngon_angle_theorem_l1071_107107
