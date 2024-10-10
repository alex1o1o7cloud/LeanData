import Mathlib

namespace equation_solution_l3114_311429

theorem equation_solution (a b : ℝ) (h : a * 3^2 - b * 3 = 6) : 2023 - 6 * a + 2 * b = 2019 := by
  sorry

end equation_solution_l3114_311429


namespace jennys_number_l3114_311405

theorem jennys_number (y : ℝ) : 10 * (y / 2 - 6) = 70 → y = 26 := by
  sorry

end jennys_number_l3114_311405


namespace polynomial_property_l3114_311468

-- Define the polynomial Q(x)
def Q (x d e f : ℝ) : ℝ := 3 * x^3 + d * x^2 + e * x + f

-- Define the conditions
theorem polynomial_property (d e f : ℝ) :
  -- The y-intercept is 12
  Q 0 d e f = 12 →
  -- The product of zeros is equal to -f/3
  (∃ α β γ : ℝ, Q α d e f = 0 ∧ Q β d e f = 0 ∧ Q γ d e f = 0 ∧ α * β * γ = -f / 3) →
  -- The mean of zeros is equal to the product of zeros
  (∃ α β γ : ℝ, Q α d e f = 0 ∧ Q β d e f = 0 ∧ Q γ d e f = 0 ∧ (α + β + γ) / 3 = -f / 3) →
  -- The sum of coefficients is equal to the product of zeros
  3 + d + e + f = -f / 3 →
  -- Then e = -55
  e = -55 :=
by
  sorry

end polynomial_property_l3114_311468


namespace quadratic_function_proof_l3114_311445

-- Define the quadratic function
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- State the theorem
theorem quadratic_function_proof (a b c : ℝ) :
  (∀ x, f a b c x = 0 ↔ x = -2 ∨ x = 1) →  -- Roots condition
  (∃ m, ∀ x, f a b c x ≤ m) →              -- Maximum value condition
  (∀ x, f a b c x = -x^2 - x + 2) :=
by sorry

end quadratic_function_proof_l3114_311445


namespace unique_solution_condition_l3114_311448

def A (a : ℝ) : Set ℝ := {x : ℝ | a * x^2 - 4 * x + 2 = 0}

theorem unique_solution_condition (a : ℝ) : 
  (∃! x, x ∈ A a) ↔ a = 0 ∨ a = 2 := by sorry

end unique_solution_condition_l3114_311448


namespace tyrone_gives_seven_point_five_verify_final_ratio_l3114_311489

/-- The number of marbles Tyrone gives to Eric to end up with three times as many marbles as Eric, given their initial marble counts. -/
def marbles_given (tyrone_initial : ℚ) (eric_initial : ℚ) : ℚ :=
  let x : ℚ := (tyrone_initial + eric_initial) / 4 - eric_initial
  x

/-- Theorem stating that given the initial conditions, Tyrone gives 7.5 marbles to Eric. -/
theorem tyrone_gives_seven_point_five :
  marbles_given 120 30 = 7.5 := by
  sorry

/-- Verification that after giving marbles, Tyrone has three times as many as Eric. -/
theorem verify_final_ratio 
  (tyrone_initial eric_initial : ℚ) 
  (h : tyrone_initial = 120 ∧ eric_initial = 30) :
  let x := marbles_given tyrone_initial eric_initial
  (tyrone_initial - x) = 3 * (eric_initial + x) := by
  sorry

end tyrone_gives_seven_point_five_verify_final_ratio_l3114_311489


namespace fourth_grade_students_l3114_311475

/-- The number of students in fourth grade at the end of the year -/
def final_students (initial : ℕ) (left : ℕ) (new : ℕ) : ℕ :=
  initial - left + new

/-- Theorem: Given the initial conditions, the final number of students is 11 -/
theorem fourth_grade_students : final_students 8 5 8 = 11 := by
  sorry

end fourth_grade_students_l3114_311475


namespace handshake_count_l3114_311486

theorem handshake_count (n : ℕ) (h : n = 25) : 
  (n * (n - 1) / 2) * 3 = 900 := by
  sorry

end handshake_count_l3114_311486


namespace min_value_M_l3114_311480

theorem min_value_M (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  let M := (((a / (b + c)) ^ (1/4) : ℝ) + ((b / (c + a)) ^ (1/4) : ℝ) + ((c / (b + a)) ^ (1/4) : ℝ) +
            ((b + c) / a) ^ (1/2) + ((a + c) / b) ^ (1/2) + ((a + b) / c) ^ (1/2))
  M ≥ 3 * Real.sqrt 2 + (3 / 2) * (8 ^ (1/4) : ℝ) :=
by sorry

end min_value_M_l3114_311480


namespace locus_is_circle_l3114_311434

def locus_of_z (z₀ : ℂ) (z : ℂ) : Prop :=
  z₀ ≠ 0 ∧ z ≠ 0 ∧ ∃ z₁ : ℂ, Complex.abs (z₁ - z₀) = Complex.abs z₁ ∧ z₁ * z = -1

theorem locus_is_circle (z₀ : ℂ) (z : ℂ) :
  locus_of_z z₀ z → Complex.abs (z + 1 / z₀) = 1 / Complex.abs z₀ :=
by sorry

end locus_is_circle_l3114_311434


namespace tan_a_values_l3114_311473

theorem tan_a_values (a : ℝ) (h : Real.sin (2 * a) = 2 - 2 * Real.cos (2 * a)) :
  Real.tan a = 0 ∨ Real.tan a = 1/2 := by
sorry

end tan_a_values_l3114_311473


namespace car_speed_problem_l3114_311471

/-- Given a car traveling for 2 hours with a speed of 40 km/h in the second hour
    and an average speed of 90 km/h over the entire journey,
    prove that the speed in the first hour must be 140 km/h. -/
theorem car_speed_problem (speed_second_hour : ℝ) (average_speed : ℝ) (speed_first_hour : ℝ) :
  speed_second_hour = 40 →
  average_speed = 90 →
  average_speed = (speed_first_hour + speed_second_hour) / 2 →
  speed_first_hour = 140 := by
  sorry

end car_speed_problem_l3114_311471


namespace sqrt_sum_eq_sqrt_1968_l3114_311483

theorem sqrt_sum_eq_sqrt_1968 : 
  ∃! (s : Finset (ℕ × ℕ)), 
    (∀ (x y : ℕ), (x, y) ∈ s ↔ Real.sqrt x + Real.sqrt y = Real.sqrt 1968) ∧ 
    s.card = 5 := by
  sorry

end sqrt_sum_eq_sqrt_1968_l3114_311483


namespace diophantine_equation_equivalence_l3114_311458

theorem diophantine_equation_equivalence (n k : ℕ) (h : n > k) :
  (∃ (x y z : ℕ+), x^n + y^n = z^k) ↔
  (∃ (x y z : ℕ+), x^n + y^n = z^(n-k)) :=
sorry

end diophantine_equation_equivalence_l3114_311458


namespace polynomial_divisibility_implies_coefficients_l3114_311441

theorem polynomial_divisibility_implies_coefficients
  (p q : ℤ)
  (h : ∀ x : ℝ, (x + 2) * (x - 1) ∣ (x^5 - x^4 + x^3 - p*x^2 + q*x + 4)) :
  p = -7 ∧ q = -12 := by
  sorry

end polynomial_divisibility_implies_coefficients_l3114_311441


namespace exists_prime_with_greater_remainder_l3114_311488

theorem exists_prime_with_greater_remainder
  (a b : ℕ+) (h : a < b) :
  ∃ p : ℕ, Nat.Prime p ∧ a % p > b % p :=
sorry

end exists_prime_with_greater_remainder_l3114_311488


namespace arithmetic_sequence_sum_l3114_311465

/-- Given an arithmetic sequence {a_n} with sum of first n terms S_n,
    if S_n = 2 and S_{3n} = 18, then S_{4n} = 26 -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) (S : ℕ → ℝ) (n : ℕ) :
  (∀ k, S k = (k * (2 * (a 1) + (k - 1) * (a 2 - a 1))) / 2) →
  S n = 2 →
  S (3 * n) = 18 →
  S (4 * n) = 26 := by
sorry

end arithmetic_sequence_sum_l3114_311465


namespace unicorns_games_played_l3114_311469

theorem unicorns_games_played (initial_games : ℕ) (initial_wins : ℕ) : 
  initial_wins = (initial_games * 45 / 100) →
  (initial_wins + 6) = ((initial_games + 8) * 1 / 2) →
  initial_games + 8 = 48 := by
sorry

end unicorns_games_played_l3114_311469


namespace even_sum_probability_l3114_311436

theorem even_sum_probability (p_even_1 p_odd_1 p_even_2 p_odd_2 : ℝ) :
  p_even_1 = 1/2 →
  p_odd_1 = 1/2 →
  p_even_2 = 1/5 →
  p_odd_2 = 4/5 →
  p_even_1 + p_odd_1 = 1 →
  p_even_2 + p_odd_2 = 1 →
  p_even_1 * p_even_2 + p_odd_1 * p_odd_2 = 1/2 :=
by sorry

end even_sum_probability_l3114_311436


namespace a_perpendicular_to_a_plus_b_l3114_311437

-- Define the vectors a and b
def a : Fin 2 → ℝ := ![2, -1]
def b : Fin 2 → ℝ := ![1, 7]

-- Theorem statement
theorem a_perpendicular_to_a_plus_b :
  (a 0 * (a 0 + b 0) + a 1 * (a 1 + b 1) = 0) := by
  sorry

end a_perpendicular_to_a_plus_b_l3114_311437


namespace joys_remaining_tape_l3114_311499

/-- Calculates the remaining tape after wrapping a rectangular field once. -/
def remaining_tape (total_tape : ℝ) (width : ℝ) (length : ℝ) : ℝ :=
  total_tape - (2 * (width + length))

/-- Theorem stating the remaining tape for Joy's specific problem. -/
theorem joys_remaining_tape :
  remaining_tape 250 20 60 = 90 := by
  sorry

end joys_remaining_tape_l3114_311499


namespace hexagon_star_perimeter_constant_l3114_311497

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a hexagon -/
structure Hexagon where
  vertices : Fin 6 → Point

/-- Checks if a hexagon is equilateral -/
def isEquilateral (h : Hexagon) : Prop := sorry

/-- Calculates the perimeter of a hexagon -/
def perimeter (h : Hexagon) : ℝ := sorry

/-- Calculates the perimeter of the star formed by extending the sides of the hexagon -/
def starPerimeter (h : Hexagon) : ℝ := sorry

theorem hexagon_star_perimeter_constant 
  (h : Hexagon) 
  (equilateral : isEquilateral h) 
  (unit_perimeter : perimeter h = 1) :
  ∀ (h' : Hexagon), 
    isEquilateral h' → 
    perimeter h' = 1 → 
    starPerimeter h = starPerimeter h' :=
sorry

end hexagon_star_perimeter_constant_l3114_311497


namespace three_color_right_triangle_l3114_311424

/-- A color type representing red, blue, and green --/
inductive Color
  | Red
  | Blue
  | Green

/-- A function that assigns a color to each point with integer coordinates --/
def coloring : ℤ × ℤ → Color := sorry

/-- Predicate to check if three points form a right-angled triangle --/
def is_right_triangle (p1 p2 p3 : ℤ × ℤ) : Prop := sorry

theorem three_color_right_triangle 
  (h1 : ∀ p : ℤ × ℤ, coloring p = Color.Red ∨ coloring p = Color.Blue ∨ coloring p = Color.Green)
  (h2 : ∃ p : ℤ × ℤ, coloring p = Color.Red)
  (h3 : ∃ p : ℤ × ℤ, coloring p = Color.Blue)
  (h4 : ∃ p : ℤ × ℤ, coloring p = Color.Green)
  (h5 : coloring (0, 0) = Color.Red)
  (h6 : coloring (0, 1) = Color.Blue) :
  ∃ p1 p2 p3 : ℤ × ℤ, 
    coloring p1 ≠ coloring p2 ∧ 
    coloring p2 ≠ coloring p3 ∧ 
    coloring p3 ≠ coloring p1 ∧ 
    is_right_triangle p1 p2 p3 :=
sorry

end three_color_right_triangle_l3114_311424


namespace quadratic_increasing_implies_a_range_l3114_311432

/-- A function f is increasing on an interval [a, +∞) if for all x, y in [a, +∞) with x < y, f(x) < f(y) -/
def IncreasingOnInterval (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y → f x < f y

/-- The quadratic function f(x) = x^2 + (a-1)x + a -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + (a-1)*x + a

theorem quadratic_increasing_implies_a_range (a : ℝ) :
  IncreasingOnInterval (f a) 2 → a ∈ Set.Ici (-3) :=
sorry

end quadratic_increasing_implies_a_range_l3114_311432


namespace fifth_root_of_unity_l3114_311450

theorem fifth_root_of_unity (p q r s m : ℂ) (hp : p ≠ 0) (hq : q ≠ 0) (hr : r ≠ 0) (hs : s ≠ 0)
  (h1 : p * m^4 + q * m^3 + r * m^2 + s * m + 1 = 0)
  (h2 : q * m^4 + r * m^3 + s * m^2 + m + p = 0) :
  m^5 = 1 := by sorry

end fifth_root_of_unity_l3114_311450


namespace excluded_angle_measure_l3114_311439

/-- Given a polygon where the sum of all but one interior angle is 1680°,
    prove that the measure of the excluded interior angle is 120°. -/
theorem excluded_angle_measure (n : ℕ) (h : n ≥ 3) :
  let sum_interior := (n - 2) * 180
  let sum_except_one := 1680
  sum_interior - sum_except_one = 120 := by
  sorry

end excluded_angle_measure_l3114_311439


namespace intersection_when_a_is_one_range_of_a_when_union_is_reals_l3114_311427

-- Define sets A and B
def A (a : ℝ) : Set ℝ := {x | |x - a| < 4}
def B : Set ℝ := {x | x^2 - 4*x - 5 > 0}

-- Part 1
theorem intersection_when_a_is_one :
  A 1 ∩ B = {x | -3 < x ∧ x < -1} := by sorry

-- Part 2
theorem range_of_a_when_union_is_reals :
  (∃ a, A a ∪ B = Set.univ) → ∃ a, 1 < a ∧ a < 3 := by sorry

end intersection_when_a_is_one_range_of_a_when_union_is_reals_l3114_311427


namespace cookfire_logs_after_three_hours_l3114_311406

/-- Calculates the number of logs left in a cookfire after a given number of hours. -/
def logs_left (initial_logs : ℕ) (burn_rate : ℕ) (add_rate : ℕ) (hours : ℕ) : ℕ :=
  initial_logs + hours * add_rate - hours * burn_rate

/-- Proves that after 3 hours, the cookfire will have 3 logs left. -/
theorem cookfire_logs_after_three_hours :
  logs_left 6 3 2 3 = 3 := by
  sorry

#eval logs_left 6 3 2 3

end cookfire_logs_after_three_hours_l3114_311406


namespace cakes_left_l3114_311410

def cakes_per_day : ℕ := 20
def baking_days : ℕ := 9
def total_cakes : ℕ := cakes_per_day * baking_days
def sold_cakes : ℕ := total_cakes / 2
def remaining_cakes : ℕ := total_cakes - sold_cakes

theorem cakes_left : remaining_cakes = 90 := by
  sorry

end cakes_left_l3114_311410


namespace cans_collection_problem_l3114_311463

theorem cans_collection_problem (solomon_cans juwan_cans levi_cans : ℕ) : 
  solomon_cans = 66 →
  solomon_cans = 3 * juwan_cans →
  levi_cans = juwan_cans / 2 →
  solomon_cans + juwan_cans + levi_cans = 99 :=
by sorry

end cans_collection_problem_l3114_311463


namespace existence_implies_upper_bound_l3114_311428

theorem existence_implies_upper_bound (a : ℝ) :
  (∃ x : ℝ, x ∈ Set.Icc 1 2 ∧ x^2 + 2*x - a ≥ 0) → a ≤ 8 := by
  sorry

end existence_implies_upper_bound_l3114_311428


namespace rope_length_problem_l3114_311418

theorem rope_length_problem (shorter_piece longer_piece total_length : ℝ) :
  shorter_piece / longer_piece = 3 / 4 →
  longer_piece = 20 →
  total_length = shorter_piece + longer_piece →
  total_length = 35 := by
sorry

end rope_length_problem_l3114_311418


namespace smallest_common_multiple_of_6_and_15_l3114_311440

theorem smallest_common_multiple_of_6_and_15 (a : ℕ) :
  (a > 0 ∧ 6 ∣ a ∧ 15 ∣ a) → a ≥ 30 :=
by sorry

end smallest_common_multiple_of_6_and_15_l3114_311440


namespace smallest_n_with_conditions_n_3584_satisfies_conditions_smallest_n_is_3584_l3114_311414

/-- A function to check if a number contains the digit 9 -/
def contains_nine (n : ℕ) : Prop :=
  ∃ k : ℕ, n / (10^k) % 10 = 9

/-- A function to check if a fraction terminates -/
def is_terminating (n : ℕ) : Prop :=
  ∃ a b : ℕ, n = 2^a * 5^b

theorem smallest_n_with_conditions :
  ∀ n : ℕ, n > 0 →
    (is_terminating n ∧ contains_nine n ∧ n % 7 = 0) →
    n ≥ 3584 :=
sorry

theorem n_3584_satisfies_conditions :
  is_terminating 3584 ∧ contains_nine 3584 ∧ 3584 % 7 = 0 :=
sorry

theorem smallest_n_is_3584 :
  ∀ n : ℕ, n > 0 →
    (is_terminating n ∧ contains_nine n ∧ n % 7 = 0) →
    n = 3584 :=
sorry

end smallest_n_with_conditions_n_3584_satisfies_conditions_smallest_n_is_3584_l3114_311414


namespace smallest_N_value_l3114_311464

/-- Represents a point in the rectangular array. -/
structure Point where
  row : Fin 5
  col : ℕ

/-- The first numbering system (left to right, top to bottom). -/
def firstNumber (N : ℕ) (p : Point) : ℕ :=
  N * p.row.val + p.col

/-- The second numbering system (top to bottom, left to right). -/
def secondNumber (p : Point) : ℕ :=
  5 * (p.col - 1) + p.row.val + 1

/-- The main theorem stating the smallest possible value of N. -/
theorem smallest_N_value :
  ∃ (N : ℕ) (p₁ p₂ p₃ p₄ p₅ : Point),
    p₁.row = 0 ∧ p₂.row = 1 ∧ p₃.row = 2 ∧ p₄.row = 3 ∧ p₅.row = 4 ∧
    (∀ p : Point, p.col < N) ∧
    firstNumber N p₁ = secondNumber p₂ ∧
    firstNumber N p₂ = secondNumber p₁ ∧
    firstNumber N p₃ = secondNumber p₄ ∧
    firstNumber N p₄ = secondNumber p₅ ∧
    firstNumber N p₅ = secondNumber p₃ ∧
    (∀ N' : ℕ, N' < N →
      ¬∃ (q₁ q₂ q₃ q₄ q₅ : Point),
        q₁.row = 0 ∧ q₂.row = 1 ∧ q₃.row = 2 ∧ q₄.row = 3 ∧ q₅.row = 4 ∧
        (∀ q : Point, q.col < N') ∧
        firstNumber N' q₁ = secondNumber q₂ ∧
        firstNumber N' q₂ = secondNumber q₁ ∧
        firstNumber N' q₃ = secondNumber q₄ ∧
        firstNumber N' q₄ = secondNumber q₅ ∧
        firstNumber N' q₅ = secondNumber q₃) ∧
    N = 149 := by
  sorry

end smallest_N_value_l3114_311464


namespace simplify_expression_l3114_311419

theorem simplify_expression : 
  ((5^1010)^2 - (5^1008)^2) / ((5^1009)^2 - (5^1007)^2) = 25 := by
  sorry

end simplify_expression_l3114_311419


namespace equation_solutions_l3114_311461

theorem equation_solutions : 
  {x : ℝ | (1 / ((x - 1) * (x - 2)) + 1 / ((x - 2) * (x - 3)) + 1 / ((x - 3) * (x - 4)) = 1 / 6)} = {7, -2} := by
  sorry

end equation_solutions_l3114_311461


namespace factorization_equality_l3114_311413

theorem factorization_equality (a b : ℝ) : a * b^2 - 2*a*b + a = a * (b - 1)^2 := by
  sorry

end factorization_equality_l3114_311413


namespace system_solution_system_solution_zero_system_solution_one_l3114_311433

theorem system_solution :
  ∀ x y z : ℝ,
  (2 * y + x - x^2 - y^2 = 0 ∧
   z - x + y - y * (x + z) = 0 ∧
   -2 * y + z - y^2 - z^2 = 0) →
  ((x = 0 ∧ y = 0 ∧ z = 0) ∨ (x = 1 ∧ y = 0 ∧ z = 1)) :=
by
  sorry

-- Alternatively, we can split it into two theorems for each solution

theorem system_solution_zero :
  2 * 0 + 0 - 0^2 - 0^2 = 0 ∧
  0 - 0 + 0 - 0 * (0 + 0) = 0 ∧
  -2 * 0 + 0 - 0^2 - 0^2 = 0 :=
by
  sorry

theorem system_solution_one :
  2 * 0 + 1 - 1^2 - 0^2 = 0 ∧
  1 - 1 + 0 - 0 * (1 + 1) = 0 ∧
  -2 * 0 + 1 - 0^2 - 1^2 = 0 :=
by
  sorry

end system_solution_system_solution_zero_system_solution_one_l3114_311433


namespace intersection_of_lines_l3114_311442

/-- The intersection point of two lines in 2D space -/
structure IntersectionPoint where
  x : ℚ
  y : ℚ

/-- Represents a line in 2D space of the form y = mx + b -/
structure Line where
  m : ℚ
  b : ℚ

/-- Checks if a point lies on a given line -/
def pointOnLine (p : IntersectionPoint) (l : Line) : Prop :=
  p.y = l.m * p.x + l.b

/-- The theorem stating the intersection point of two specific lines -/
theorem intersection_of_lines :
  let line1 : Line := { m := 3, b := -1 }
  let line2 : Line := { m := -6, b := -4 }
  let point : IntersectionPoint := { x := -1/3, y := -2 }
  (pointOnLine point line1) ∧ (pointOnLine point line2) ∧
  (∀ p : IntersectionPoint, (pointOnLine p line1) ∧ (pointOnLine p line2) → p = point) :=
by sorry

end intersection_of_lines_l3114_311442


namespace car_rental_cost_l3114_311470

/-- Calculates the total cost of renting a car for three days with varying rates and mileage. -/
theorem car_rental_cost (base_rate_day1 base_rate_day2 base_rate_day3 : ℚ)
                        (per_mile_day1 per_mile_day2 per_mile_day3 : ℚ)
                        (miles_day1 miles_day2 miles_day3 : ℚ) :
  base_rate_day1 = 150 →
  base_rate_day2 = 100 →
  base_rate_day3 = 75 →
  per_mile_day1 = 0.5 →
  per_mile_day2 = 0.4 →
  per_mile_day3 = 0.3 →
  miles_day1 = 620 →
  miles_day2 = 744 →
  miles_day3 = 510 →
  base_rate_day1 + miles_day1 * per_mile_day1 +
  base_rate_day2 + miles_day2 * per_mile_day2 +
  base_rate_day3 + miles_day3 * per_mile_day3 = 1085.6 :=
by sorry

end car_rental_cost_l3114_311470


namespace textbook_weight_difference_l3114_311459

theorem textbook_weight_difference :
  let chemistry_weight : ℝ := 7.12
  let geometry_weight : ℝ := 0.62
  let history_weight : ℝ := 4.25
  let literature_weight : ℝ := 3.8
  let chem_geo_combined : ℝ := chemistry_weight + geometry_weight
  let hist_lit_combined : ℝ := history_weight + literature_weight
  chem_geo_combined - hist_lit_combined = -0.31 :=
by
  sorry

end textbook_weight_difference_l3114_311459


namespace stock_value_return_l3114_311417

theorem stock_value_return (initial_value : ℝ) (h : initial_value > 0) :
  let first_year_value := initial_value * 1.4
  let second_year_decrease := 2 / 7
  first_year_value * (1 - second_year_decrease) = initial_value :=
by sorry

end stock_value_return_l3114_311417


namespace professors_age_l3114_311412

def guesses : List Nat := [34, 37, 39, 41, 43, 46, 48, 51, 53, 56]

def is_prime (n : Nat) : Prop := Nat.Prime n

theorem professors_age :
  ∃ (age : Nat),
    age ∈ guesses ∧
    is_prime age ∧
    (guesses.filter (· < age)).length ≥ guesses.length / 2 ∧
    (guesses.filter (fun x => x = age - 1 ∨ x = age + 1)).length = 2 ∧
    age = 47 :=
  sorry

end professors_age_l3114_311412


namespace smallest_w_l3114_311422

def is_factor (a b : ℕ) : Prop := ∃ k : ℕ, b = a * k

theorem smallest_w (w : ℕ) : 
  w > 0 → 
  is_factor (2^5) (936 * w) → 
  is_factor (3^3) (936 * w) → 
  is_factor (11^2) (936 * w) → 
  ∀ v : ℕ, v > 0 → 
    is_factor (2^5) (936 * v) → 
    is_factor (3^3) (936 * v) → 
    is_factor (11^2) (936 * v) → 
    w ≤ v →
  w = 4356 :=
sorry

end smallest_w_l3114_311422


namespace integer_puzzle_l3114_311435

theorem integer_puzzle (x y : ℕ+) (h1 : x + y = 60) (h2 : x - y = 16) :
  x^2 - y^2 = 960 ∧ x * y = 836 := by
  sorry

end integer_puzzle_l3114_311435


namespace operations_to_equality_l3114_311444

def num_operations (a b : ℕ) (subtract_a add_b : ℕ) : ℕ :=
  (a - b) / (subtract_a + add_b)

theorem operations_to_equality : num_operations 365 24 19 12 = 11 := by
  sorry

end operations_to_equality_l3114_311444


namespace function_equivalence_and_coefficient_sum_l3114_311498

noncomputable def f (x : ℝ) : ℝ := (x^3 + 3*x^2 - 4*x - 12) / (x + 3)

def g (x : ℝ) : ℝ := x^2 - 4

def A : ℝ := 1
def B : ℝ := 0
def C : ℝ := -4
def D : ℝ := -3

theorem function_equivalence_and_coefficient_sum :
  (∀ x : ℝ, x ≠ D → f x = g x) ∧
  A + B + C + D = -6 := by sorry

end function_equivalence_and_coefficient_sum_l3114_311498


namespace sandwich_problem_l3114_311454

theorem sandwich_problem (billy_sandwiches katelyn_sandwiches chloe_sandwiches : ℕ) : 
  billy_sandwiches = 49 →
  chloe_sandwiches = katelyn_sandwiches / 4 →
  billy_sandwiches + katelyn_sandwiches + chloe_sandwiches = 169 →
  katelyn_sandwiches > billy_sandwiches →
  katelyn_sandwiches - billy_sandwiches = 47 := by
sorry


end sandwich_problem_l3114_311454


namespace square_sum_equality_l3114_311430

theorem square_sum_equality (x y : ℝ) 
  (h1 : x + 3 * y = 3) 
  (h2 : x * y = -3) : 
  x^2 + 9 * y^2 = 27 := by
  sorry

end square_sum_equality_l3114_311430


namespace chocolate_bars_count_l3114_311477

/-- Represents the number of chocolate bars in the gigantic box -/
def chocolate_bars_in_gigantic_box : ℕ :=
  let small_box_bars : ℕ := 45
  let medium_box_small_boxes : ℕ := 10
  let large_box_medium_boxes : ℕ := 25
  let gigantic_box_large_boxes : ℕ := 50
  gigantic_box_large_boxes * large_box_medium_boxes * medium_box_small_boxes * small_box_bars

/-- Theorem stating that the number of chocolate bars in the gigantic box is 562,500 -/
theorem chocolate_bars_count : chocolate_bars_in_gigantic_box = 562500 := by
  sorry

end chocolate_bars_count_l3114_311477


namespace total_jumps_eq_308_l3114_311460

/-- The total number of times Joonyoung and Namyoung jumped rope --/
def total_jumps (joonyoung_freq : ℕ) (joonyoung_months : ℕ) (namyoung_freq : ℕ) (namyoung_months : ℕ) : ℕ :=
  joonyoung_freq * joonyoung_months + namyoung_freq * namyoung_months

/-- Theorem stating that the total jumps for Joonyoung and Namyoung is 308 --/
theorem total_jumps_eq_308 :
  total_jumps 56 3 35 4 = 308 := by
  sorry

end total_jumps_eq_308_l3114_311460


namespace jihoons_class_size_l3114_311401

theorem jihoons_class_size :
  ∃! n : ℕ, 35 < n ∧ n < 70 ∧ n % 6 = 3 ∧ n % 8 = 1 ∧ n = 57 := by
  sorry

end jihoons_class_size_l3114_311401


namespace quadratic_inequality_range_l3114_311494

theorem quadratic_inequality_range (a : ℝ) : 
  (∃ x : ℝ, x^2 + a*x + 1 < 0) → (a < -2 ∨ a > 2) := by
  sorry

end quadratic_inequality_range_l3114_311494


namespace cashback_strategies_reduce_losses_l3114_311411

/-- Represents a bank's cashback program -/
structure CashbackProgram where
  name : String
  maxCashbackPercentage : Float
  monthlyCapExists : Bool
  variableRate : Bool
  nonMonetaryRewards : Bool

/-- Represents a customer's behavior -/
structure CustomerBehavior where
  financialLiteracy : Float
  prefersHighCashbackCategories : Bool

/-- Calculates the profitability of a cashback program -/
def calculateProfitability (program : CashbackProgram) (customer : CustomerBehavior) : Float :=
  sorry

/-- Theorem: Implementing certain cashback strategies can reduce potential losses for banks -/
theorem cashback_strategies_reduce_losses 
  (program : CashbackProgram) 
  (customer : CustomerBehavior) :
  (program.monthlyCapExists ∨ program.variableRate ∨ program.nonMonetaryRewards) →
  (customer.financialLiteracy > 0.8 ∧ customer.prefersHighCashbackCategories) →
  calculateProfitability program customer > 0 := by
  sorry

#check cashback_strategies_reduce_losses

end cashback_strategies_reduce_losses_l3114_311411


namespace small_cubes_in_large_cube_l3114_311402

/-- Converts decimeters to centimeters -/
def dm_to_cm (dm : ℕ) : ℕ := dm * 10

/-- Calculates the number of small cubes that fit in a large cube -/
def num_small_cubes (large_side_dm : ℕ) (small_side_cm : ℕ) : ℕ :=
  let large_side_cm := dm_to_cm large_side_dm
  let num_cubes_per_edge := large_side_cm / small_side_cm
  num_cubes_per_edge ^ 3

theorem small_cubes_in_large_cube :
  num_small_cubes 8 4 = 8000 := by
  sorry

end small_cubes_in_large_cube_l3114_311402


namespace no_positive_integer_solutions_l3114_311423

theorem no_positive_integer_solutions : 
  ¬ ∃ (x y : ℕ+), x^2 + y^2 = x^3 + 2*y := by
  sorry

end no_positive_integer_solutions_l3114_311423


namespace camp_wonka_marshmallows_l3114_311474

theorem camp_wonka_marshmallows (total_campers : ℕ) 
  (boys_fraction : ℚ) (girls_fraction : ℚ) 
  (boys_toasting_percentage : ℚ) (girls_toasting_percentage : ℚ) :
  total_campers = 96 →
  boys_fraction = 2/3 →
  girls_fraction = 1/3 →
  boys_toasting_percentage = 1/2 →
  girls_toasting_percentage = 3/4 →
  (boys_fraction * total_campers * boys_toasting_percentage + 
   girls_fraction * total_campers * girls_toasting_percentage : ℚ) = 56 := by
  sorry

end camp_wonka_marshmallows_l3114_311474


namespace percentage_relationship_l3114_311400

theorem percentage_relationship (j p t m n x y : ℕ) (r : ℚ) : 
  j > 0 ∧ p > 0 ∧ t > 0 ∧ m > 0 ∧ n > 0 ∧ x > 0 ∧ y > 0 →
  j = (3 / 4 : ℚ) * p →
  j = (4 / 5 : ℚ) * t →
  t = p - (r / 100) * p →
  m = (11 / 10 : ℚ) * p →
  n = (7 / 10 : ℚ) * m →
  j + p + t = m * n →
  x = (23 / 20 : ℚ) * j →
  y = (4 / 5 : ℚ) * n →
  x * y = (j + p + t)^2 →
  r = (25 / 4 : ℚ) := by
sorry

end percentage_relationship_l3114_311400


namespace limits_involving_x_and_n_l3114_311478

open Real

/-- For x > 0, prove the limits of two expressions involving n and x as n approaches infinity. -/
theorem limits_involving_x_and_n (x : ℝ) (h : x > 0) :
  (∀ ε > 0, ∃ N, ∀ n ≥ N, |n * log (1 + x / n) - x| < ε) ∧
  (∀ ε > 0, ∃ N, ∀ n ≥ N, |(1 + x / n)^n - exp x| < ε) :=
sorry

end limits_involving_x_and_n_l3114_311478


namespace polynomial_roots_comparison_l3114_311416

theorem polynomial_roots_comparison (a₁ a₂ a₃ b₁ b₂ b₃ : ℝ) 
  (h_order_a : a₁ ≤ a₂ ∧ a₂ ≤ a₃)
  (h_order_b : b₁ ≤ b₂ ∧ b₂ ≤ b₃)
  (h_sum : a₁ + a₂ + a₃ = b₁ + b₂ + b₃)
  (h_prod : a₁*a₂ + a₂*a₃ + a₁*a₃ = b₁*b₂ + b₂*b₃ + b₁*b₃)
  (h_first : a₁ ≤ b₁) :
  a₃ ≤ b₃ := by sorry

end polynomial_roots_comparison_l3114_311416


namespace tangent_slope_implies_a_l3114_311443

/-- Given a function f(x) = ax^2 + 3x - 2, prove that if the slope of the tangent line
    at the point (2, f(2)) is 7, then a = 1. -/
theorem tangent_slope_implies_a (a : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ a * x^2 + 3 * x - 2
  (deriv f 2 = 7) → a = 1 := by
sorry

end tangent_slope_implies_a_l3114_311443


namespace occur_permutations_correct_l3114_311495

/-- The number of unique permutations of the letters in "OCCUR" -/
def occurrPermutations : ℕ := 60

/-- The total number of letters in "OCCUR" -/
def totalLetters : ℕ := 5

/-- The number of times the letter "C" appears in "OCCUR" -/
def cCount : ℕ := 2

/-- Theorem stating that the number of unique permutations of "OCCUR" is correct -/
theorem occur_permutations_correct :
  occurrPermutations = (Nat.factorial totalLetters) / (Nat.factorial cCount) :=
by sorry

end occur_permutations_correct_l3114_311495


namespace fourth_term_of_sequence_l3114_311438

def geometric_sequence (a : ℝ) (r : ℝ) (n : ℕ) : ℝ := a * r^(n - 1)

theorem fourth_term_of_sequence (x : ℝ) :
  let a₁ := x
  let a₂ := 3 * x + 6
  let a₃ := 7 * x + 21
  (∃ r : ℝ, r ≠ 0 ∧ 
    geometric_sequence a₁ r 2 = a₂ ∧ 
    geometric_sequence a₁ r 3 = a₃) →
  geometric_sequence a₁ ((3 * x + 6) / x) 4 = 220.5 :=
by sorry

end fourth_term_of_sequence_l3114_311438


namespace road_repaving_today_distance_l3114_311490

/-- Represents the repaving progress of a road construction project -/
structure RoadRepaving where
  totalRepaved : ℕ
  repavedBefore : ℕ

/-- Calculates the distance repaved today given the total repaved and repaved before -/
def distanceRepavedToday (r : RoadRepaving) : ℕ :=
  r.totalRepaved - r.repavedBefore

/-- Theorem: For the given road repaving project, the distance repaved today is 805 inches -/
theorem road_repaving_today_distance 
  (r : RoadRepaving) 
  (h1 : r.totalRepaved = 4938) 
  (h2 : r.repavedBefore = 4133) : 
  distanceRepavedToday r = 805 := by
  sorry

#eval distanceRepavedToday { totalRepaved := 4938, repavedBefore := 4133 }

end road_repaving_today_distance_l3114_311490


namespace function_identity_l3114_311446

theorem function_identity (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, f (x^2 + f y) = y + (f x)^2) : 
  ∀ x : ℝ, f x = x := by
sorry

end function_identity_l3114_311446


namespace square_difference_plus_two_cubed_l3114_311467

theorem square_difference_plus_two_cubed : (7^2 - 3^2 + 2)^3 = 74088 := by
  sorry

end square_difference_plus_two_cubed_l3114_311467


namespace right_triangle_angle_bisectors_l3114_311493

/-- Given a right triangle with legs 3 and 4, prove the lengths of its angle bisectors. -/
theorem right_triangle_angle_bisectors :
  let a : ℝ := 3
  let b : ℝ := 4
  let c : ℝ := (a^2 + b^2).sqrt
  let d : ℝ := (a * b * ((a + b + c) / (a + b - c))).sqrt
  let l_b : ℝ := (b * a * (1 - ((b - a)^2 / b^2))).sqrt
  let l_a : ℝ := (a * b * (1 - ((a - b)^2 / a^2))).sqrt
  (d = 12 * Real.sqrt 2 / 7) ∧
  (l_b = 3 * Real.sqrt 5 / 2) ∧
  (l_a = 4 * Real.sqrt 10 / 3) := by
sorry


end right_triangle_angle_bisectors_l3114_311493


namespace systematic_sample_theorem_l3114_311455

/-- Represents a systematic sample of students -/
structure SystematicSample where
  total_students : Nat
  sample_size : Nat
  common_difference : Nat
  start : Nat

/-- Checks if a number is in the systematic sample -/
def in_sample (s : SystematicSample) (n : Nat) : Prop :=
  ∃ k : Nat, k < s.sample_size ∧ n = (s.start + k * s.common_difference) % s.total_students

/-- The main theorem to prove -/
theorem systematic_sample_theorem :
  ∃ (s : SystematicSample),
    s.total_students = 52 ∧
    s.sample_size = 4 ∧
    in_sample s 6 ∧
    in_sample s 32 ∧
    in_sample s 45 ∧
    in_sample s 19 :=
  sorry

end systematic_sample_theorem_l3114_311455


namespace meeting_arrangement_count_l3114_311482

/-- Represents the number of schools -/
def num_schools : ℕ := 3

/-- Represents the number of members per school -/
def members_per_school : ℕ := 6

/-- Represents the total number of members -/
def total_members : ℕ := num_schools * members_per_school

/-- Represents the number of representatives sent by the host school -/
def host_representatives : ℕ := 2

/-- Represents the number of representatives sent by each non-host school -/
def non_host_representatives : ℕ := 1

/-- The number of ways to arrange the meeting -/
def meeting_arrangements : ℕ := 1620

/-- Theorem stating the number of ways to arrange the meeting -/
theorem meeting_arrangement_count :
  num_schools * (Nat.choose members_per_school host_representatives *
    (Nat.choose members_per_school non_host_representatives)^(num_schools - 1)) = meeting_arrangements :=
by sorry

end meeting_arrangement_count_l3114_311482


namespace ladder_problem_l3114_311447

theorem ladder_problem (ladder_length height base : ℝ) 
  (h1 : ladder_length = 13)
  (h2 : height = 12)
  (h3 : ladder_length^2 = height^2 + base^2) : 
  base = 5 := by sorry

end ladder_problem_l3114_311447


namespace problem_statement_l3114_311472

/-- Given real numbers x, y, and z satisfying certain conditions, 
    prove that a specific expression equals 13.5 -/
theorem problem_statement (x y z : ℝ) 
  (h1 : x*z/(x+y) + y*x/(y+z) + z*y/(z+x) = -9)
  (h2 : y*z/(x+y) + z*x/(y+z) + x*y/(z+x) = 15) :
  y/(x+y) + z/(y+z) + x/(z+x) = 13.5 := by
  sorry

end problem_statement_l3114_311472


namespace greatest_base_seven_digit_sum_proof_l3114_311407

/-- The greatest possible sum of the digits in the base-seven representation of a positive integer less than 2019 -/
def greatest_base_seven_digit_sum : ℕ := 22

/-- A function that converts a natural number to its base-seven representation -/
def to_base_seven (n : ℕ) : List ℕ := sorry

/-- A function that calculates the sum of digits in a list -/
def digit_sum (digits : List ℕ) : ℕ := sorry

theorem greatest_base_seven_digit_sum_proof :
  ∀ n : ℕ, 0 < n → n < 2019 →
  digit_sum (to_base_seven n) ≤ greatest_base_seven_digit_sum ∧
  ∃ m : ℕ, 0 < m ∧ m < 2019 ∧ digit_sum (to_base_seven m) = greatest_base_seven_digit_sum :=
sorry

end greatest_base_seven_digit_sum_proof_l3114_311407


namespace perpendicular_vectors_l3114_311420

/-- Given vectors a and b, find k such that k*a - 2*b is perpendicular to a -/
theorem perpendicular_vectors (a b : ℝ × ℝ) (k : ℝ) : 
  a = (1, 1) → b = (2, -3) → 
  (k * a.1 - 2 * b.1, k * a.2 - 2 * b.2) • a = 0 → 
  k = -1 := by sorry

end perpendicular_vectors_l3114_311420


namespace lcm_210_396_l3114_311481

theorem lcm_210_396 : Nat.lcm 210 396 = 13860 := by
  sorry

end lcm_210_396_l3114_311481


namespace remainder_theorem_l3114_311456

theorem remainder_theorem (P D Q R D' Q'' R'' : ℕ) 
  (h1 : P = Q * D + R)
  (h2 : Q^2 = D' * Q'' + R'') :
  P % (D * D') = R := by
  sorry

end remainder_theorem_l3114_311456


namespace prime_power_sum_l3114_311466

theorem prime_power_sum (p : ℕ) (x y z : ℕ) 
  (hp : Prime p) 
  (hxyz : x > 0 ∧ y > 0 ∧ z > 0) 
  (heq : x^p + y^p = p^z) : 
  z = 2 := by
sorry

end prime_power_sum_l3114_311466


namespace transformed_graph_point_l3114_311415

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the theorem
theorem transformed_graph_point (h : f 4 = 7) : 
  ∃ (x y : ℝ), 2 * y = 3 * f (4 * x) + 5 ∧ x + y = 14 := by
sorry

end transformed_graph_point_l3114_311415


namespace isosceles_triangle_dimensions_l3114_311452

/-- An isosceles triangle with equal sides of length x and base of length y,
    where a median on one of the equal sides divides the perimeter into parts of 6 and 12 -/
structure IsoscelesTriangle where
  x : ℝ  -- Length of equal sides
  y : ℝ  -- Length of base
  perimeter_division : x + x/2 = 6 ∧ y/2 + y = 12

theorem isosceles_triangle_dimensions (t : IsoscelesTriangle) : t.x = 8 ∧ t.y = 2 := by
  sorry

end isosceles_triangle_dimensions_l3114_311452


namespace last_digit_of_large_prime_l3114_311487

theorem last_digit_of_large_prime (n : ℕ) (h : n = 859433) :
  (2^n - 1) % 10 = 1 := by
  sorry

end last_digit_of_large_prime_l3114_311487


namespace total_legs_in_collection_l3114_311484

theorem total_legs_in_collection (num_ants num_spiders : ℕ) 
  (ant_legs spider_legs : ℕ) (h1 : num_ants = 12) (h2 : num_spiders = 8) 
  (h3 : ant_legs = 6) (h4 : spider_legs = 8) : 
  num_ants * ant_legs + num_spiders * spider_legs = 136 := by
  sorry

end total_legs_in_collection_l3114_311484


namespace power_of_two_l3114_311449

theorem power_of_two : (1 : ℕ) * 2^6 = 64 := by sorry

end power_of_two_l3114_311449


namespace percent_within_one_std_dev_l3114_311404

-- Define a symmetric distribution
structure SymmetricDistribution where
  mean : ℝ
  std_dev : ℝ
  is_symmetric : Bool
  percent_less_than_mean_plus_std_dev : ℝ

-- Theorem statement
theorem percent_within_one_std_dev 
  (dist : SymmetricDistribution) 
  (h1 : dist.is_symmetric = true) 
  (h2 : dist.percent_less_than_mean_plus_std_dev = 84) : 
  ∃ (p : ℝ), p = 68 ∧ 
  p = dist.percent_less_than_mean_plus_std_dev - (100 - dist.percent_less_than_mean_plus_std_dev) := by
  sorry

end percent_within_one_std_dev_l3114_311404


namespace incorrect_inequality_l3114_311451

theorem incorrect_inequality (a b : ℝ) (h : (1 / a) < (1 / b) ∧ (1 / b) < 0) :
  ¬(abs a + abs b > abs (a + b)) := by
  sorry

end incorrect_inequality_l3114_311451


namespace intersection_M_N_l3114_311421

def M : Set ℝ := {x | (x + 3) * (x - 2) < 0}
def N : Set ℝ := {0, 1, 2}

theorem intersection_M_N : M ∩ N = {0, 1} := by sorry

end intersection_M_N_l3114_311421


namespace smallest_number_after_removal_largest_number_after_removal_l3114_311491

-- Define the original number as a string
def originalNumber : String := "123456789101112...5657585960"

-- Define the number of digits to remove
def digitsToRemove : Nat := 100

-- Define the function to remove digits and get the smallest number
def smallestAfterRemoval (s : String) (n : Nat) : Nat :=
  sorry

-- Define the function to remove digits and get the largest number
def largestAfterRemoval (s : String) (n : Nat) : Nat :=
  sorry

-- Theorem for the smallest number
theorem smallest_number_after_removal :
  smallestAfterRemoval originalNumber digitsToRemove = 123450 :=
by sorry

-- Theorem for the largest number
theorem largest_number_after_removal :
  largestAfterRemoval originalNumber digitsToRemove = 56758596049 :=
by sorry

end smallest_number_after_removal_largest_number_after_removal_l3114_311491


namespace percentage_problem_l3114_311485

theorem percentage_problem (a b : ℝ) (ha : a > 0) (hb : b > 0) (h1 : 8 = (6 / 100) * a) 
  (h2 : ∃ x, (x / 100) * b = 6) (h3 : b / a = 9 / 2) : 
  ∃ x, (x / 100) * b = 6 ∧ x = 1 := by
  sorry

end percentage_problem_l3114_311485


namespace hypotenuse_product_square_l3114_311431

-- Define the triangles and their properties
def right_triangle (a b c : ℝ) : Prop := a^2 + b^2 = c^2

theorem hypotenuse_product_square (x y h₁ h₂ : ℝ) :
  right_triangle x (2*y) h₁ →  -- T1
  right_triangle x y h₂ →      -- T2
  x * (2*y) / 2 = 8 →          -- Area of T1
  x * y / 2 = 4 →              -- Area of T2
  (h₁ * h₂)^2 = 160 := by
sorry

end hypotenuse_product_square_l3114_311431


namespace expression_value_l3114_311457

theorem expression_value (x y : ℝ) (h : y = 2 - x) : 4 * x + 4 * y - 3 = 5 := by
  sorry

end expression_value_l3114_311457


namespace expression_evaluation_l3114_311496

theorem expression_evaluation (a : ℝ) (h : a = Real.sqrt 2 + 1) :
  (1 + 1 / a) / ((a^2 - 1) / a) = Real.sqrt 2 / 2 := by
  sorry

end expression_evaluation_l3114_311496


namespace parabola_properties_l3114_311403

def f (x : ℝ) := -(x + 1)^2 + 3

theorem parabola_properties :
  (∀ x y : ℝ, f x ≤ f y → (x - (-1))^2 ≥ (y - (-1))^2) ∧ 
  (∀ x y : ℝ, x + y = -2 → f x = f y) ∧
  (f (-1) = 3 ∧ ∀ x : ℝ, f x ≤ f (-1)) ∧
  (∀ x y : ℝ, x > 1 ∧ y > 1 ∧ x > y → f x < f y) :=
by sorry

end parabola_properties_l3114_311403


namespace max_type_c_tubes_l3114_311462

/-- Represents the number of test tubes of each type -/
structure TestTubes where
  a : ℕ  -- Type A (10% solution)
  b : ℕ  -- Type B (20% solution)
  c : ℕ  -- Type C (90% solution)

/-- The problem constraints -/
def validSolution (t : TestTubes) : Prop :=
  -- Total number of test tubes is 1000
  t.a + t.b + t.c = 1000 ∧
  -- The resulting solution is 20.17%
  10 * t.a + 20 * t.b + 90 * t.c = 2017 * (t.a + t.b + t.c) ∧
  -- Two consecutive pourings cannot use test tubes of the same type
  t.a > 0 ∧ t.b > 0

/-- The theorem statement -/
theorem max_type_c_tubes :
  ∃ (t : TestTubes), validSolution t ∧
    (∀ (t' : TestTubes), validSolution t' → t'.c ≤ t.c) ∧
    t.c = 73 := by
  sorry

end max_type_c_tubes_l3114_311462


namespace sum_of_reciprocals_squared_l3114_311426

noncomputable def a : ℝ := Real.sqrt 2 + Real.sqrt 3 + Real.sqrt 7
noncomputable def b : ℝ := -Real.sqrt 2 + Real.sqrt 3 + Real.sqrt 7
noncomputable def c : ℝ := Real.sqrt 2 - Real.sqrt 3 + Real.sqrt 7
noncomputable def d : ℝ := -Real.sqrt 2 - Real.sqrt 3 + Real.sqrt 7

theorem sum_of_reciprocals_squared :
  (1/a + 1/b + 1/c + 1/d)^2 = 7/49 := by
  sorry

end sum_of_reciprocals_squared_l3114_311426


namespace lcm_count_theorem_l3114_311476

theorem lcm_count_theorem : 
  ∃ (S : Finset ℕ), 
    S.card = 19 ∧ 
    (∀ k : ℕ, k > 0 → (Nat.lcm (Nat.lcm (9^9) (12^12)) k = 18^18 ↔ k ∈ S)) := by
  sorry

end lcm_count_theorem_l3114_311476


namespace correct_bonus_distribution_l3114_311453

/-- Represents the bonus distribution problem for a corporation --/
def BonusDistribution (total : ℕ) (a b c d e f : ℕ) : Prop :=
  -- Total bonus is $25,000
  total = 25000 ∧
  -- A receives twice the amount of B
  a = 2 * b ∧
  -- B and C receive the same amount
  b = c ∧
  -- D receives $1,500 less than A
  d = a - 1500 ∧
  -- E receives $2,000 more than C
  e = c + 2000 ∧
  -- F receives half of the total amount received by A and D combined
  f = (a + d) / 2 ∧
  -- The sum of all amounts equals the total bonus
  a + b + c + d + e + f = total

/-- Theorem stating the correct distribution of the bonus --/
theorem correct_bonus_distribution :
  BonusDistribution 25000 4950 2475 2475 3450 4475 4200 := by
  sorry

#check correct_bonus_distribution

end correct_bonus_distribution_l3114_311453


namespace albert_earnings_increase_l3114_311479

theorem albert_earnings_increase (E : ℝ) (P : ℝ) 
  (h1 : E * (1 + P) = 693)
  (h2 : E * 1.20 = 660) :
  P = 0.26 := by
  sorry

end albert_earnings_increase_l3114_311479


namespace train_or_plane_prob_not_ship_prob_prob_half_combinations_prob_sum_one_l3114_311492

-- Define the probabilities of each transportation mode
def train_prob : ℝ := 0.3
def ship_prob : ℝ := 0.2
def car_prob : ℝ := 0.1
def plane_prob : ℝ := 0.4

-- Define the sum of all probabilities
def total_prob : ℝ := train_prob + ship_prob + car_prob + plane_prob

-- Theorem for the probability of taking either a train or a plane
theorem train_or_plane_prob : train_prob + plane_prob = 0.7 := by sorry

-- Theorem for the probability of not taking a ship
theorem not_ship_prob : 1 - ship_prob = 0.8 := by sorry

-- Theorem for the combinations with probability 0.5
theorem prob_half_combinations :
  (train_prob + ship_prob = 0.5 ∧ car_prob + plane_prob = 0.5) := by sorry

-- Ensure that the probabilities sum to 1
theorem prob_sum_one : total_prob = 1 := by sorry

end train_or_plane_prob_not_ship_prob_prob_half_combinations_prob_sum_one_l3114_311492


namespace least_four_digit_special_number_l3114_311408

/-- A function that checks if a number has all different digits -/
def has_different_digits (n : ℕ) : Prop := sorry

/-- A function that checks if a number is divisible by all of its digits -/
def divisible_by_digits (n : ℕ) : Prop := sorry

theorem least_four_digit_special_number :
  ∀ n : ℕ,
  1000 ≤ n →
  n < 10000 →
  has_different_digits n →
  divisible_by_digits n →
  n % 5 = 0 →
  1425 ≤ n :=
sorry

end least_four_digit_special_number_l3114_311408


namespace unique_solution_l3114_311425

theorem unique_solution (x y : ℝ) : 
  x * (x + y)^2 = 9 ∧ x * (y^3 - x^3) = 7 → x = 1 ∧ y = 2 := by
  sorry

end unique_solution_l3114_311425


namespace system_solution_unique_l3114_311409

def system_solution (a₁ a₂ a₃ a₄ : ℝ) (x₁ x₂ x₃ x₄ : ℝ) : Prop :=
  a₁ < a₂ ∧ a₂ < a₃ ∧ a₃ < a₄ ∧
  (|a₁ - a₂| * x₂ + |a₁ - a₃| * x₃ + |a₁ - a₄| * x₄ = 1) ∧
  (|a₂ - a₁| * x₁ + |a₂ - a₃| * x₃ + |a₂ - a₄| * x₄ = 1) ∧
  (|a₃ - a₁| * x₁ + |a₃ - a₂| * x₂ + |a₃ - a₄| * x₄ = 1) ∧
  (|a₄ - a₁| * x₁ + |a₄ - a₂| * x₂ + |a₄ - a₃| * x₃ = 1)

theorem system_solution_unique (a₁ a₂ a₃ a₄ : ℝ) :
  ∃! (x₁ x₂ x₃ x₄ : ℝ), system_solution a₁ a₂ a₃ a₄ x₁ x₂ x₃ x₄ ∧
    x₁ = 1 / (a₄ - a₁) ∧ x₂ = 0 ∧ x₃ = 0 ∧ x₄ = 1 / (a₄ - a₁) :=
by
  sorry

end system_solution_unique_l3114_311409
