import Mathlib

namespace NUMINAMATH_CALUDE_partitions_divisible_by_2_pow_n_l2474_247435

/-- Represents a valid partition of a 1 × n strip -/
def StripPartition (n : ℕ) : Type := Unit

/-- The number of valid partitions for a 1 × n strip -/
def num_partitions (n : ℕ) : ℕ := sorry

/-- The main theorem: the number of valid partitions is divisible by 2^n -/
theorem partitions_divisible_by_2_pow_n (n : ℕ) (h : n > 0) : 
  ∃ k : ℕ, num_partitions n = 2^n * k :=
sorry

end NUMINAMATH_CALUDE_partitions_divisible_by_2_pow_n_l2474_247435


namespace NUMINAMATH_CALUDE_octagon_edge_length_l2474_247407

/-- The length of one edge of a regular octagon made from the same thread as a regular pentagon with one edge of 16 cm -/
theorem octagon_edge_length (pentagon_edge : ℝ) (thread_length : ℝ) : 
  pentagon_edge = 16 → thread_length = 5 * pentagon_edge → thread_length / 8 = 10 := by
  sorry

#check octagon_edge_length

end NUMINAMATH_CALUDE_octagon_edge_length_l2474_247407


namespace NUMINAMATH_CALUDE_tangent_line_x_intercept_l2474_247401

-- Define the function f
def f (x : ℝ) : ℝ := x^3 + 4*x + 5

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 3*x^2 + 4

-- Theorem statement
theorem tangent_line_x_intercept :
  let tangent_slope : ℝ := f' 1
  let tangent_point : ℝ × ℝ := (1, f 1)
  let x_intercept : ℝ := tangent_point.1 - tangent_point.2 / tangent_slope
  x_intercept = -3/7 := by sorry

end NUMINAMATH_CALUDE_tangent_line_x_intercept_l2474_247401


namespace NUMINAMATH_CALUDE_triangle_inequality_theorem_l2474_247403

theorem triangle_inequality_theorem (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b > c ∧ b + c > a ∧ c + a > b) :
  a^2 * b * (a - b) + b^2 * c * (b - c) + c^2 * a * (c - a) ≥ 0 ∧
  (a^2 * b * (a - b) + b^2 * c * (b - c) + c^2 * a * (c - a) = 0 ↔ a = b ∧ b = c) :=
by sorry

end NUMINAMATH_CALUDE_triangle_inequality_theorem_l2474_247403


namespace NUMINAMATH_CALUDE_product_not_ending_1999_l2474_247462

theorem product_not_ending_1999 (a b c d e : ℕ) : 
  a + b + c + d + e = 200 → 
  ∃ k : ℕ, a * b * c * d * e = 1000 * k ∨ a * b * c * d * e = 1000 * k + 1 ∨ 
          a * b * c * d * e = 1000 * k + 2 ∨ a * b * c * d * e = 1000 * k + 3 ∨ 
          a * b * c * d * e = 1000 * k + 4 ∨ a * b * c * d * e = 1000 * k + 5 ∨ 
          a * b * c * d * e = 1000 * k + 6 ∨ a * b * c * d * e = 1000 * k + 7 ∨ 
          a * b * c * d * e = 1000 * k + 8 ∨ a * b * c * d * e = 1000 * k + 9 := by
  sorry

end NUMINAMATH_CALUDE_product_not_ending_1999_l2474_247462


namespace NUMINAMATH_CALUDE_angle_equivalence_l2474_247413

-- Define α in degrees
def α : ℝ := 2010

-- Theorem statement
theorem angle_equivalence (α : ℝ) : 
  -- Part 1: Rewrite α in the form θ + 2kπ
  (α * π / 180 = 7 * π / 6 + 10 * π) ∧
  -- Part 2: Find equivalent angles in [-5π, 0)
  (∀ β : ℝ, -5 * π ≤ β ∧ β < 0 ∧ 
    (∃ k : ℤ, β = 7 * π / 6 + 2 * k * π) ↔ 
    (β = -29 * π / 6 ∨ β = -17 * π / 6 ∨ β = -5 * π / 6)) :=
by sorry

end NUMINAMATH_CALUDE_angle_equivalence_l2474_247413


namespace NUMINAMATH_CALUDE_decimal_111_to_base5_l2474_247437

/-- Converts a natural number to its base-5 representation as a list of digits -/
def toBase5 (n : ℕ) : List ℕ :=
  if n < 5 then [n]
  else (n % 5) :: toBase5 (n / 5)

/-- Theorem: The base-5 representation of 111 (decimal) is [2, 3, 3] -/
theorem decimal_111_to_base5 :
  toBase5 111 = [2, 3, 3] := by
  sorry

#eval toBase5 111  -- This will output [2, 3, 3]

end NUMINAMATH_CALUDE_decimal_111_to_base5_l2474_247437


namespace NUMINAMATH_CALUDE_consecutive_integers_sqrt_11_l2474_247446

theorem consecutive_integers_sqrt_11 (a b : ℤ) : 
  (b = a + 1) →  -- a and b are consecutive integers
  (a < Real.sqrt 11) → 
  (Real.sqrt 11 < b) → 
  a + b = 7 := by
sorry

end NUMINAMATH_CALUDE_consecutive_integers_sqrt_11_l2474_247446


namespace NUMINAMATH_CALUDE_function_properties_l2474_247408

open Real

noncomputable def f (k : ℝ) (x : ℝ) : ℝ := (log x) / x - k / x

theorem function_properties (k : ℝ) :
  (∀ x ≥ 1, x^2 * f k x + 1 / (x + 1) ≥ 0) →
  (∀ x ≥ 1, k ≥ 1/2 * x^2 + (exp 2 - 2) * x - exp x - 7) →
  (∀ x > 0, deriv (f k) x = (1 - log x + k) / x^2) →
  (deriv (f k) 1 = 10) →
  (∃ x_max > 0, ∀ x > 0, f k x ≤ f k x_max ∧ f k x_max = 1 / (exp 10)) ∧
  (exp 2 - 9 ≤ k ∧ k ≤ 1/2) :=
sorry

end NUMINAMATH_CALUDE_function_properties_l2474_247408


namespace NUMINAMATH_CALUDE_arithmetic_sequence_perfect_square_sum_l2474_247495

def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, n = k^2

def arithmetic_sequence (a d : ℕ) (n : ℕ) : ℕ := a + (n - 1) * d

def sum_arithmetic_sequence (a d : ℕ) (n : ℕ) : ℕ := n * (2 * a + (n - 1) * d) / 2

theorem arithmetic_sequence_perfect_square_sum (a d : ℕ) :
  (∀ n : ℕ, is_perfect_square (sum_arithmetic_sequence a d n)) ↔
  (∃ b : ℕ, a = b^2 ∧ d = 2 * b^2) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_perfect_square_sum_l2474_247495


namespace NUMINAMATH_CALUDE_first_question_percentage_l2474_247475

theorem first_question_percentage
  (second_correct : ℝ)
  (neither_correct : ℝ)
  (both_correct : ℝ)
  (h1 : second_correct = 55)
  (h2 : neither_correct = 20)
  (h3 : both_correct = 40)
  : ℝ :=
by
  -- The percentage answering the first question correctly is 65%
  sorry

#check first_question_percentage

end NUMINAMATH_CALUDE_first_question_percentage_l2474_247475


namespace NUMINAMATH_CALUDE_element_value_l2474_247419

theorem element_value (a : Nat) : 
  a ∈ ({0, 1, 2, 3} : Set Nat) → 
  a ∉ ({0, 1, 2} : Set Nat) → 
  a = 3 := by
  sorry

end NUMINAMATH_CALUDE_element_value_l2474_247419


namespace NUMINAMATH_CALUDE_coordinates_of_C_l2474_247420

-- Define the points
def A : ℝ × ℝ := (2, 8)
def M : ℝ × ℝ := (4, 11)
def L : ℝ × ℝ := (6, 6)

-- Define the properties
def is_midpoint (M A B : ℝ × ℝ) : Prop :=
  M.1 = (A.1 + B.1) / 2 ∧ M.2 = (A.2 + B.2) / 2

def on_angle_bisector (L B C : ℝ × ℝ) : Prop :=
  (L.1 - B.1) * (C.2 - B.2) = (L.2 - B.2) * (C.1 - B.1)

-- Theorem statement
theorem coordinates_of_C (B C : ℝ × ℝ) 
  (h1 : is_midpoint M A B)
  (h2 : on_angle_bisector L B C) :
  C = (14, 2) := by sorry

end NUMINAMATH_CALUDE_coordinates_of_C_l2474_247420


namespace NUMINAMATH_CALUDE_y2_less_than_y1_l2474_247453

def f (x : ℝ) := -4 * x - 3

theorem y2_less_than_y1 (y₁ y₂ : ℝ) 
  (h1 : f (-2) = y₁) 
  (h2 : f 5 = y₂) : 
  y₂ < y₁ := by
sorry

end NUMINAMATH_CALUDE_y2_less_than_y1_l2474_247453


namespace NUMINAMATH_CALUDE_polygon_labeling_exists_l2474_247499

/-- A labeling of a polygon is a function that assigns a unique label to each vertex and midpoint -/
def Labeling (n : ℕ) := Fin (4*n+2) → Fin (4*n+2)

/-- The sum of labels for a side is the sum of the labels of its two vertices and midpoint -/
def sideSum (f : Labeling n) (i : Fin (2*n+1)) : ℕ :=
  f i + f (i+1) + f (i+2*n+1)

theorem polygon_labeling_exists (n : ℕ) :
  ∃ (f : Labeling n), Function.Injective f ∧
    ∀ (i j : Fin (2*n+1)), sideSum f i = sideSum f j :=
sorry

end NUMINAMATH_CALUDE_polygon_labeling_exists_l2474_247499


namespace NUMINAMATH_CALUDE_sum_of_squares_l2474_247487

theorem sum_of_squares (a b c x y z : ℝ) 
  (h1 : x/a + y/b + z/c = 5)
  (h2 : a/x + b/y + c/z = 3) :
  x^2/a^2 + y^2/b^2 + z^2/c^2 = 19 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_l2474_247487


namespace NUMINAMATH_CALUDE_plums_for_oranges_l2474_247414

-- Define the cost of fruits as real numbers
variables (orange pear plum : ℝ)

-- Define the conditions
def condition1 : Prop := 5 * orange = 3 * pear
def condition2 : Prop := 4 * pear = 6 * plum

-- Theorem to prove
theorem plums_for_oranges 
  (h1 : condition1 orange pear) 
  (h2 : condition2 pear plum) : 
  20 * orange = 18 * plum :=
sorry

end NUMINAMATH_CALUDE_plums_for_oranges_l2474_247414


namespace NUMINAMATH_CALUDE_braiding_time_proof_l2474_247481

/-- Calculates the time in minutes required to braid dancers' hair -/
def braiding_time (num_dancers : ℕ) (braids_per_dancer : ℕ) (seconds_per_braid : ℕ) : ℚ :=
  (num_dancers * braids_per_dancer * seconds_per_braid : ℚ) / 60

/-- Proves that given 15 dancers, 10 braids per dancer, and 45 seconds per braid,
    the total time required to braid all dancers' hair is 112.5 minutes -/
theorem braiding_time_proof :
  braiding_time 15 10 45 = 112.5 := by
  sorry

end NUMINAMATH_CALUDE_braiding_time_proof_l2474_247481


namespace NUMINAMATH_CALUDE_trig_expression_equals_one_l2474_247411

theorem trig_expression_equals_one :
  let numerator := Real.sin (18 * π / 180) * Real.cos (12 * π / 180) + 
                   Real.cos (162 * π / 180) * Real.cos (102 * π / 180)
  let denominator := Real.sin (22 * π / 180) * Real.cos (8 * π / 180) + 
                     Real.cos (158 * π / 180) * Real.cos (98 * π / 180)
  numerator / denominator = 1 := by
sorry

end NUMINAMATH_CALUDE_trig_expression_equals_one_l2474_247411


namespace NUMINAMATH_CALUDE_zeroes_elimination_theorem_l2474_247402

/-- A step in the digit replacement process. -/
structure Step where
  digits_removed : ℕ := 2

/-- The initial state of the blackboard. -/
structure Blackboard where
  zeroes : ℕ
  ones : ℕ

/-- The final state after all steps are completed. -/
structure FinalState where
  steps : ℕ
  remaining_ones : ℕ

/-- The theorem to be proved. -/
theorem zeroes_elimination_theorem (initial : Blackboard) (final : FinalState) :
  initial.zeroes = 150 ∧
  final.steps = 76 ∧
  final.remaining_ones = initial.ones - 2 →
  initial.ones = 78 :=
by sorry

end NUMINAMATH_CALUDE_zeroes_elimination_theorem_l2474_247402


namespace NUMINAMATH_CALUDE_find_x_value_l2474_247484

theorem find_x_value (x : ℝ) : 
  (max 1 (max 2 (max 3 x)) = 1 + 2 + 3 + x) → x = -3 := by
  sorry

end NUMINAMATH_CALUDE_find_x_value_l2474_247484


namespace NUMINAMATH_CALUDE_trig_value_comparison_l2474_247422

theorem trig_value_comparison :
  let a : ℝ := Real.tan (-7 * π / 6)
  let b : ℝ := Real.cos (23 * π / 4)
  let c : ℝ := Real.sin (-33 * π / 4)
  b > a ∧ a > c := by sorry

end NUMINAMATH_CALUDE_trig_value_comparison_l2474_247422


namespace NUMINAMATH_CALUDE_intersection_equality_necessary_not_sufficient_l2474_247452

theorem intersection_equality_necessary_not_sufficient :
  (∀ (M N P : Set α), M = N → M ∩ P = N ∩ P) ∧
  (∃ (M N P : Set α), M ∩ P = N ∩ P ∧ M ≠ N) :=
by sorry

end NUMINAMATH_CALUDE_intersection_equality_necessary_not_sufficient_l2474_247452


namespace NUMINAMATH_CALUDE_abs_neg_three_times_two_l2474_247460

theorem abs_neg_three_times_two : |(-3)| * 2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_abs_neg_three_times_two_l2474_247460


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2474_247409

/-- An arithmetic sequence. -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  d : ℝ
  h : ∀ n, a (n + 1) = a n + d

/-- The problem statement. -/
theorem arithmetic_sequence_problem (seq : ArithmeticSequence) 
  (h1 : seq.a 1 * seq.a 5 = 9)
  (h2 : seq.a 2 = 3) :
  seq.a 4 = 3 ∨ seq.a 4 = 7 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2474_247409


namespace NUMINAMATH_CALUDE_constant_value_proof_l2474_247497

theorem constant_value_proof (b c : ℝ) :
  (∀ x : ℝ, (x + 3) * (x + b) = x^2 + c*x + 12) →
  c = 7 := by
sorry

end NUMINAMATH_CALUDE_constant_value_proof_l2474_247497


namespace NUMINAMATH_CALUDE_distinct_numbers_probability_l2474_247448

def num_sides : ℕ := 5
def num_dice : ℕ := 5

theorem distinct_numbers_probability : 
  (Nat.factorial num_sides : ℚ) / (num_sides ^ num_dice : ℚ) = 24 / 625 := by
  sorry

end NUMINAMATH_CALUDE_distinct_numbers_probability_l2474_247448


namespace NUMINAMATH_CALUDE_roberto_skipping_rate_l2474_247441

/-- Roberto's skipping rate problem -/
theorem roberto_skipping_rate 
  (valerie_rate : ℕ) 
  (total_skips : ℕ) 
  (duration : ℕ) 
  (h1 : valerie_rate = 80)
  (h2 : total_skips = 2250)
  (h3 : duration = 15) :
  ∃ (roberto_hourly_rate : ℕ), 
    roberto_hourly_rate = 4200 ∧ 
    roberto_hourly_rate * duration = (total_skips - valerie_rate * duration) * 4 :=
sorry

end NUMINAMATH_CALUDE_roberto_skipping_rate_l2474_247441


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2474_247458

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The statement to prove -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmetic_sequence a →
  a 3 ^ 2 - 6 * a 3 + 5 = 0 →
  a 15 ^ 2 - 6 * a 15 + 5 = 0 →
  a 7 + a 8 + a 9 + a 10 + a 11 = 15 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2474_247458


namespace NUMINAMATH_CALUDE_smallest_n_for_perfect_square_l2474_247496

theorem smallest_n_for_perfect_square : ∃ (n : ℕ), 
  (n = 12) ∧ 
  (∃ (k : ℕ), (2^n + 2^8 + 2^11 : ℕ) = k^2) ∧ 
  (∀ (m : ℕ), m < n → ¬∃ (k : ℕ), (2^m + 2^8 + 2^11 : ℕ) = k^2) :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_for_perfect_square_l2474_247496


namespace NUMINAMATH_CALUDE_cube_order_l2474_247455

theorem cube_order (a b : ℝ) (h : a > b) : a^3 > b^3 := by
  sorry

end NUMINAMATH_CALUDE_cube_order_l2474_247455


namespace NUMINAMATH_CALUDE_max_value_theorem_l2474_247429

theorem max_value_theorem (x y : ℝ) (h : x^2 + 4*y^2 ≤ 4) :
  ∃ (M : ℝ), M = 12 ∧ ∀ (a b : ℝ), a^2 + 4*b^2 ≤ 4 → |a+2*b-4| + |3-a-b| ≤ M :=
sorry

end NUMINAMATH_CALUDE_max_value_theorem_l2474_247429


namespace NUMINAMATH_CALUDE_count_solution_pairs_l2474_247431

/-- The number of ordered pairs (a, b) of complex numbers satisfying the given equations -/
def solution_count : ℕ := 24

/-- The predicate defining the condition for a pair of complex numbers -/
def satisfies_equations (a b : ℂ) : Prop :=
  a^4 * b^6 = 1 ∧ a^8 * b^3 = 1

theorem count_solution_pairs :
  (∃! (s : Finset (ℂ × ℂ)), s.card = solution_count ∧ 
   ∀ p ∈ s, satisfies_equations p.1 p.2 ∧
   ∀ a b, satisfies_equations a b → (a, b) ∈ s) :=
sorry

end NUMINAMATH_CALUDE_count_solution_pairs_l2474_247431


namespace NUMINAMATH_CALUDE_open_box_volume_l2474_247449

/-- The volume of an open box constructed from a rectangular sheet -/
def box_volume (x : ℝ) : ℝ :=
  (16 - 2*x) * (12 - 2*x) * x

/-- Theorem stating the volume of the open box -/
theorem open_box_volume (x : ℝ) : 
  box_volume x = 4*x^3 - 56*x^2 + 192*x :=
by sorry

end NUMINAMATH_CALUDE_open_box_volume_l2474_247449


namespace NUMINAMATH_CALUDE_corner_cut_rectangle_l2474_247428

/-- Given a rectangle ABCD with dimensions AB = 18 m and AD = 12 m,
    and identical right-angled isosceles triangles cut off from the corners,
    leaving a smaller rectangle PQRS. The total area cut off is 180 m². -/
theorem corner_cut_rectangle (AB AD : ℝ) (area_cut : ℝ) (PR : ℝ) : AB = 18 → AD = 12 → area_cut = 180 → PR = 18 - 6 * Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_corner_cut_rectangle_l2474_247428


namespace NUMINAMATH_CALUDE_quadrilateral_diagonal_bounds_l2474_247433

/-- Given four points A, B, C, D in a plane, with distances AB = 2, BC = 7, CD = 5, and DA = 12,
    the minimum possible length of AC is 7 and the maximum possible length of AC is 9. -/
theorem quadrilateral_diagonal_bounds (A B C D : EuclideanSpace ℝ (Fin 2)) 
  (h1 : dist A B = 2)
  (h2 : dist B C = 7)
  (h3 : dist C D = 5)
  (h4 : dist D A = 12) :
  (∃ (m M : ℝ), m = 7 ∧ M = 9 ∧ 
    (∀ (AC : ℝ), AC = dist A C → m ≤ AC ∧ AC ≤ M)) := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_diagonal_bounds_l2474_247433


namespace NUMINAMATH_CALUDE_farmer_turkeys_l2474_247480

theorem farmer_turkeys (total_cost : ℝ) (kept_turkeys : ℕ) (sale_revenue : ℝ) (profit_per_bird : ℝ) :
  total_cost = 60 ∧
  kept_turkeys = 15 ∧
  sale_revenue = 54 ∧
  profit_per_bird = 0.1 →
  ∃ n : ℕ,
    n * (total_cost / n) = total_cost ∧
    ((total_cost / n) + profit_per_bird) * (n - kept_turkeys) = sale_revenue ∧
    n = 75 :=
by sorry

end NUMINAMATH_CALUDE_farmer_turkeys_l2474_247480


namespace NUMINAMATH_CALUDE_regular_pentagon_perimeter_l2474_247476

/-- A pentagon with all sides of equal length -/
structure RegularPentagon where
  side_length : ℝ

/-- The sum of all side lengths of a regular pentagon -/
def total_perimeter (p : RegularPentagon) : ℝ := 5 * p.side_length

/-- Theorem: If one side of a regular pentagon is 15 cm long, 
    then the sum of all side lengths is 75 cm -/
theorem regular_pentagon_perimeter : 
  ∀ (p : RegularPentagon), p.side_length = 15 → total_perimeter p = 75 := by
  sorry

#check regular_pentagon_perimeter

end NUMINAMATH_CALUDE_regular_pentagon_perimeter_l2474_247476


namespace NUMINAMATH_CALUDE_sum_of_fractions_inequality_l2474_247400

theorem sum_of_fractions_inequality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  x / (y + z) + y / (z + x) + z / (x + y) ≥ (3 : ℝ) / 2 := by sorry

end NUMINAMATH_CALUDE_sum_of_fractions_inequality_l2474_247400


namespace NUMINAMATH_CALUDE_jed_speeding_fine_l2474_247425

/-- Calculates the speeding fine in Zeoland -/
def speeding_fine (speed_limit : ℕ) (actual_speed : ℕ) (fine_per_mph : ℕ) : ℕ :=
  if actual_speed > speed_limit
  then (actual_speed - speed_limit) * fine_per_mph
  else 0

/-- Proves that Jed's speeding fine is $256 -/
theorem jed_speeding_fine :
  let speed_limit : ℕ := 50
  let actual_speed : ℕ := 66
  let fine_per_mph : ℕ := 16
  speeding_fine speed_limit actual_speed fine_per_mph = 256 :=
by
  sorry

end NUMINAMATH_CALUDE_jed_speeding_fine_l2474_247425


namespace NUMINAMATH_CALUDE_quadratic_problem_l2474_247472

-- Define the quadratic function y₁
def y₁ (x b c : ℝ) : ℝ := x^2 + b*x + c

-- Define the quadratic function y₂
def y₂ (x m : ℝ) : ℝ := 2*x^2 + x + m

theorem quadratic_problem :
  ∀ (b c m : ℝ),
  (y₁ 0 b c = 4) →                        -- y₁ passes through (0,4)
  (∀ x, y₁ (1 + x) b c = y₁ (1 - x) b c) →  -- symmetry axis x = 1
  (b^2 - c = 0) →                         -- condition b² - c = 0
  (∃ x₀, b - 3 ≤ x₀ ∧ x₀ ≤ b ∧ 
    (∀ x, b - 3 ≤ x ∧ x ≤ b → y₁ x₀ b c ≤ y₁ x b c) ∧
    y₁ x₀ b c = 21) →                     -- minimum value 21 when b-3 ≤ x ≤ b
  (∀ x, 0 ≤ x ∧ x ≤ 1 → y₂ x m ≥ y₁ x b c) →  -- y₂ ≥ y₁ for 0 ≤ x ≤ 1
  ((∀ x, y₁ x b c = x^2 - 2*x + 4) ∧      -- Part 1 result
   (b = -Real.sqrt 7 ∨ b = 4) ∧           -- Part 2 result
   (m = 4))                               -- Part 3 result
  := by sorry

end NUMINAMATH_CALUDE_quadratic_problem_l2474_247472


namespace NUMINAMATH_CALUDE_distance_on_number_line_distance_negative_five_negative_one_l2474_247486

theorem distance_on_number_line : ∀ (a b : ℝ), abs (a - b) = abs (b - a) :=
by sorry

theorem distance_negative_five_negative_one : abs (-5 - (-1)) = 4 :=
by sorry

end NUMINAMATH_CALUDE_distance_on_number_line_distance_negative_five_negative_one_l2474_247486


namespace NUMINAMATH_CALUDE_unique_right_triangle_l2474_247478

-- Define a structure for a right-angled triangle with integer sides
structure RightTriangle where
  a : ℕ+
  b : ℕ+
  c : ℕ+
  right_angle : a * a + b * b = c * c
  perimeter_80 : a + b + c = 80

-- Theorem statement
theorem unique_right_triangle : 
  ∃! (t : RightTriangle), t.a = 30 ∧ t.b = 16 ∧ t.c = 34 :=
by sorry

end NUMINAMATH_CALUDE_unique_right_triangle_l2474_247478


namespace NUMINAMATH_CALUDE_todd_ate_cupcakes_l2474_247451

theorem todd_ate_cupcakes (initial_cupcakes : ℕ) (packages : ℕ) (cupcakes_per_package : ℕ) 
  (h1 : initial_cupcakes = 20)
  (h2 : packages = 3)
  (h3 : cupcakes_per_package = 3) :
  initial_cupcakes - (packages * cupcakes_per_package) = 11 :=
by sorry

end NUMINAMATH_CALUDE_todd_ate_cupcakes_l2474_247451


namespace NUMINAMATH_CALUDE_workshop_sampling_theorem_l2474_247440

-- Define the total number of workers in each group
def total_workers_A : ℕ := 10
def total_workers_B : ℕ := 10

-- Define the number of female workers in each group
def female_workers_A : ℕ := 4
def female_workers_B : ℕ := 6

-- Define the total number of workers selected for assessment
def total_selected : ℕ := 4

-- Define the number of workers selected from each group
def selected_from_A : ℕ := 2
def selected_from_B : ℕ := 2

-- Define the probability of selecting exactly 1 female worker from Group A
def prob_one_female_A : ℚ := (Nat.choose female_workers_A 1 * Nat.choose (total_workers_A - female_workers_A) (selected_from_A - 1)) / Nat.choose total_workers_A selected_from_A

-- Define the probability of selecting exactly 2 male workers from both groups
def prob_two_males : ℚ :=
  (Nat.choose (total_workers_A - female_workers_A) 0 * Nat.choose female_workers_A 2 *
   Nat.choose (total_workers_B - female_workers_B) 2 * Nat.choose female_workers_B 0 +
   Nat.choose (total_workers_A - female_workers_A) 1 * Nat.choose female_workers_A 1 *
   Nat.choose (total_workers_B - female_workers_B) 1 * Nat.choose female_workers_B 1 +
   Nat.choose (total_workers_A - female_workers_A) 2 * Nat.choose female_workers_A 0 *
   Nat.choose (total_workers_B - female_workers_B) 0 * Nat.choose female_workers_B 2) /
  (Nat.choose total_workers_A selected_from_A * Nat.choose total_workers_B selected_from_B)

theorem workshop_sampling_theorem :
  (selected_from_A + selected_from_B = total_selected) ∧
  (prob_one_female_A = (Nat.choose female_workers_A 1 * Nat.choose (total_workers_A - female_workers_A) (selected_from_A - 1)) / Nat.choose total_workers_A selected_from_A) ∧
  (prob_two_males = (Nat.choose (total_workers_A - female_workers_A) 0 * Nat.choose female_workers_A 2 *
                     Nat.choose (total_workers_B - female_workers_B) 2 * Nat.choose female_workers_B 0 +
                     Nat.choose (total_workers_A - female_workers_A) 1 * Nat.choose female_workers_A 1 *
                     Nat.choose (total_workers_B - female_workers_B) 1 * Nat.choose female_workers_B 1 +
                     Nat.choose (total_workers_A - female_workers_A) 2 * Nat.choose female_workers_A 0 *
                     Nat.choose (total_workers_B - female_workers_B) 0 * Nat.choose female_workers_B 2) /
                    (Nat.choose total_workers_A selected_from_A * Nat.choose total_workers_B selected_from_B)) := by
  sorry


end NUMINAMATH_CALUDE_workshop_sampling_theorem_l2474_247440


namespace NUMINAMATH_CALUDE_circle_area_theorem_l2474_247445

/-- Line passing through two points -/
structure Line where
  point1 : ℝ × ℝ
  point2 : ℝ × ℝ

/-- Circle with center at origin -/
structure Circle where
  radius : ℝ

/-- The perpendicular line to a given line passing through a point -/
def perpendicularLine (l : Line) (p : ℝ × ℝ) : Line :=
  sorry

/-- The length of the chord formed by the intersection of a line and a circle -/
def chordLength (l : Line) (c : Circle) : ℝ :=
  sorry

/-- The area of a circle -/
def circleArea (c : Circle) : ℝ :=
  sorry

theorem circle_area_theorem (l : Line) (c : Circle) :
  l.point1 = (2, 1) →
  l.point2 = (1, -1) →
  let m := perpendicularLine l (2, 1)
  chordLength m c = 6 * Real.sqrt 5 / 5 →
  circleArea c = 5 * Real.pi :=
sorry

end NUMINAMATH_CALUDE_circle_area_theorem_l2474_247445


namespace NUMINAMATH_CALUDE_probability_sum_six_is_three_sixteenths_l2474_247468

/-- A uniform tetrahedral die with faces numbered 1, 2, 3, 4 -/
def TetrahedralDie : Finset ℕ := {1, 2, 3, 4}

/-- The sample space of throwing the die twice -/
def SampleSpace : Finset (ℕ × ℕ) := TetrahedralDie.product TetrahedralDie

/-- The event where the sum of two throws equals 6 -/
def SumSixEvent : Finset (ℕ × ℕ) := SampleSpace.filter (fun p => p.1 + p.2 = 6)

/-- The probability of the sum being 6 when throwing the die twice -/
def probability_sum_six : ℚ := (SumSixEvent.card : ℚ) / (SampleSpace.card : ℚ)

theorem probability_sum_six_is_three_sixteenths : 
  probability_sum_six = 3 / 16 := by
  sorry

end NUMINAMATH_CALUDE_probability_sum_six_is_three_sixteenths_l2474_247468


namespace NUMINAMATH_CALUDE_right_handed_players_count_l2474_247442

theorem right_handed_players_count (total_players throwers : ℕ) : 
  total_players = 150 →
  throwers = 60 →
  (total_players - throwers) % 2 = 0 →
  105 = throwers + (total_players - throwers) / 2 := by
  sorry

end NUMINAMATH_CALUDE_right_handed_players_count_l2474_247442


namespace NUMINAMATH_CALUDE_ones_digit_of_largest_power_of_3_dividing_27_factorial_l2474_247432

/-- The largest power of 3 that divides n! -/
def largestPowerOf3DividingFactorial (n : ℕ) : ℕ :=
  (n / 3) + (n / 9) + (n / 27)

/-- The ones digit of 3^n -/
def onesDigitOf3ToPower (n : ℕ) : ℕ :=
  match n % 4 with
  | 0 => 1
  | 1 => 3
  | 2 => 9
  | 3 => 7
  | _ => 0 -- This case is impossible, but Lean requires it

theorem ones_digit_of_largest_power_of_3_dividing_27_factorial :
  onesDigitOf3ToPower (largestPowerOf3DividingFactorial 27) = 3 := by
  sorry

#eval onesDigitOf3ToPower (largestPowerOf3DividingFactorial 27)

end NUMINAMATH_CALUDE_ones_digit_of_largest_power_of_3_dividing_27_factorial_l2474_247432


namespace NUMINAMATH_CALUDE_ratio_equality_l2474_247417

theorem ratio_equality (p q r u v w : ℝ) 
  (h_pos : p > 0 ∧ q > 0 ∧ r > 0 ∧ u > 0 ∧ v > 0 ∧ w > 0)
  (h_pqr : p^2 + q^2 + r^2 = 49)
  (h_uvw : u^2 + v^2 + w^2 = 64)
  (h_sum : p*u + q*v + r*w = 56) :
  (p + q + r) / (u + v + w) = 7/8 := by
sorry

end NUMINAMATH_CALUDE_ratio_equality_l2474_247417


namespace NUMINAMATH_CALUDE_cubic_rational_roots_l2474_247454

/-- A cubic polynomial with rational coefficients -/
structure CubicPolynomial where
  a : ℚ
  b : ℚ
  c : ℚ

/-- The roots of a cubic polynomial -/
def roots (p : CubicPolynomial) : Set ℚ :=
  {x : ℚ | x^3 + p.a * x^2 + p.b * x + p.c = 0}

/-- The theorem stating the only possible sets of rational roots for a cubic polynomial -/
theorem cubic_rational_roots (p : CubicPolynomial) :
  roots p = {0, 1, -2} ∨ roots p = {1, -1, -1} := by
  sorry


end NUMINAMATH_CALUDE_cubic_rational_roots_l2474_247454


namespace NUMINAMATH_CALUDE_eulers_formula_l2474_247443

/-- A planar graph structure -/
structure PlanarGraph where
  V : Type*  -- Set of vertices
  E : Type*  -- Set of edges
  F : Type*  -- Set of faces
  n : ℕ      -- Number of vertices
  m : ℕ      -- Number of edges
  ℓ : ℕ      -- Number of faces
  is_connected : Prop  -- Property that the graph is connected

/-- Euler's formula for planar graphs -/
theorem eulers_formula (G : PlanarGraph) : G.n - G.m + G.ℓ = 2 := by
  sorry

end NUMINAMATH_CALUDE_eulers_formula_l2474_247443


namespace NUMINAMATH_CALUDE_soccer_preference_and_goals_l2474_247483

/-- Chi-square test statistic for 2x2 contingency table -/
def chi_square (a b c d : ℕ) : ℚ :=
  (200 * (a * d - b * c)^2 : ℚ) / ((a + b) * (c + d) * (a + c) * (b + d))

/-- Critical value for chi-square test at α = 0.001 -/
def critical_value : ℚ := 10828 / 1000

/-- Probability of scoring a goal for male students -/
def p_male : ℚ := 2 / 3

/-- Probability of scoring a goal for female student -/
def p_female : ℚ := 1 / 2

/-- Expected value of goals scored by 2 male and 1 female student -/
def expected_goals : ℚ := 11 / 6

theorem soccer_preference_and_goals (a b c d : ℕ) 
  (h1 : a + b = 100) (h2 : c + d = 100) (h3 : a + c = 90) (h4 : b + d = 110) :
  chi_square a b c d > critical_value ∧ 
  2 * p_male + p_female = expected_goals :=
sorry

end NUMINAMATH_CALUDE_soccer_preference_and_goals_l2474_247483


namespace NUMINAMATH_CALUDE_g_sum_equals_two_l2474_247427

-- Define the function g
def g (a b c : ℝ) (x : ℝ) : ℝ := a * x^6 + b * x^4 - c * x^2 + 5

-- State the theorem
theorem g_sum_equals_two (a b c : ℝ) :
  g a b c 11 = 1 → g a b c 11 + g a b c (-11) = 2 := by
sorry

end NUMINAMATH_CALUDE_g_sum_equals_two_l2474_247427


namespace NUMINAMATH_CALUDE_quadrilateral_area_l2474_247469

/-- The area of a quadrilateral with vertices at (4,0), (0,5), (3,4), and (10,10) is 22.5 square units. -/
theorem quadrilateral_area : 
  let vertices : List (ℝ × ℝ) := [(4,0), (0,5), (3,4), (10,10)]
  ∃ (area : ℝ), area = 22.5 ∧ 
  area = (1/2) * abs (
    (4 * 5 + 0 * 4 + 3 * 10 + 10 * 0) - 
    (0 * 0 + 5 * 3 + 4 * 10 + 10 * 4)
  ) := by sorry

end NUMINAMATH_CALUDE_quadrilateral_area_l2474_247469


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l2474_247404

theorem quadratic_inequality_solution (a : ℝ) (x₁ x₂ : ℝ) : 
  a < 0 →
  (∀ x, x^2 - a*x - 6*a^2 > 0 ↔ x < x₁ ∨ x > x₂) →
  x₂ - x₁ = 5 * Real.sqrt 2 →
  a = -Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l2474_247404


namespace NUMINAMATH_CALUDE_circular_garden_area_increase_l2474_247464

theorem circular_garden_area_increase : 
  let r₁ : ℝ := 6  -- radius of larger garden
  let r₂ : ℝ := 4  -- radius of smaller garden
  let area₁ := π * r₁^2  -- area of larger garden
  let area₂ := π * r₂^2  -- area of smaller garden
  (area₁ - area₂) / area₂ * 100 = 125
  := by sorry

end NUMINAMATH_CALUDE_circular_garden_area_increase_l2474_247464


namespace NUMINAMATH_CALUDE_product_equals_fraction_fraction_is_simplified_l2474_247463

/-- The repeating decimal 0.256̄ as a rational number -/
def repeating_decimal : ℚ := 256 / 999

/-- The product of 0.256̄ and 12 -/
def product : ℚ := repeating_decimal * 12

/-- Theorem stating that the product of 0.256̄ and 12 is equal to 1024/333 -/
theorem product_equals_fraction : product = 1024 / 333 := by
  sorry

/-- Theorem stating that 1024/333 is in its simplest form -/
theorem fraction_is_simplified : Int.gcd 1024 333 = 1 := by
  sorry

end NUMINAMATH_CALUDE_product_equals_fraction_fraction_is_simplified_l2474_247463


namespace NUMINAMATH_CALUDE_quadratic_polynomial_solutions_l2474_247493

-- Define a quadratic polynomial
def QuadraticPolynomial (α : Type*) [Field α] := α → α

-- Define the property of having exactly three solutions for (f(x))^3 - 4f(x) = 0
def HasThreeSolutionsCubicMinusFour (f : QuadraticPolynomial ℝ) : Prop :=
  ∃ (x₁ x₂ x₃ : ℝ), (∀ x : ℝ, f x ^ 3 - 4 * f x = 0 ↔ x = x₁ ∨ x = x₂ ∨ x = x₃) ∧
  x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃

-- Define the property of having exactly two solutions for (f(x))^2 = 1
def HasTwoSolutionsSquaredEqualsOne (f : QuadraticPolynomial ℝ) : Prop :=
  ∃ (y₁ y₂ : ℝ), (∀ y : ℝ, f y ^ 2 = 1 ↔ y = y₁ ∨ y = y₂) ∧ y₁ ≠ y₂

-- State the theorem
theorem quadratic_polynomial_solutions 
  (f : QuadraticPolynomial ℝ) 
  (h : HasThreeSolutionsCubicMinusFour f) : 
  HasTwoSolutionsSquaredEqualsOne f :=
sorry

end NUMINAMATH_CALUDE_quadratic_polynomial_solutions_l2474_247493


namespace NUMINAMATH_CALUDE_profit_percentage_calculation_l2474_247479

theorem profit_percentage_calculation
  (purchase_price : ℕ)
  (repair_cost : ℕ)
  (transportation_charges : ℕ)
  (selling_price : ℕ)
  (h1 : purchase_price = 12000)
  (h2 : repair_cost = 5000)
  (h3 : transportation_charges = 1000)
  (h4 : selling_price = 27000) :
  (selling_price - (purchase_price + repair_cost + transportation_charges)) * 100 /
  (purchase_price + repair_cost + transportation_charges) = 50 :=
by
  sorry

#check profit_percentage_calculation

end NUMINAMATH_CALUDE_profit_percentage_calculation_l2474_247479


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2474_247473

theorem inequality_solution_set :
  {x : ℝ | (x + 1) * (2 - x) < 0} = {x : ℝ | x < -1 ∨ x > 2} := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2474_247473


namespace NUMINAMATH_CALUDE_monotonicity_indeterminate_l2474_247410

-- Define the concept of an increasing function on an open interval
def IncreasingOn (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a < x ∧ x < y ∧ y < b → f x < f y

-- Define the theorem
theorem monotonicity_indeterminate
  (f : ℝ → ℝ) (a b c : ℝ) 
  (hab : a < b) (hbc : b < c)
  (h1 : IncreasingOn f a b)
  (h2 : IncreasingOn f b c) :
  ¬ (IncreasingOn f a c ∨ (∀ x y, a < x ∧ x < y ∧ y < c → f x > f y)) :=
sorry

end NUMINAMATH_CALUDE_monotonicity_indeterminate_l2474_247410


namespace NUMINAMATH_CALUDE_nonzero_terms_count_l2474_247450

-- Define the polynomials
def p (x : ℝ) : ℝ := 2*x + 3
def q (x : ℝ) : ℝ := x^2 + 4*x + 5
def r (x : ℝ) : ℝ := x^3 - x^2 + 2*x + 1

-- Define the expanded expression
def expanded_expr (x : ℝ) : ℝ := p x * q x - 4 * r x

-- Theorem statement
theorem nonzero_terms_count :
  ∃ (a b c d : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧
  ∀ x, expanded_expr x = a*x^3 + b*x^2 + c*x + d :=
by sorry

end NUMINAMATH_CALUDE_nonzero_terms_count_l2474_247450


namespace NUMINAMATH_CALUDE_sin_cos_extrema_l2474_247490

theorem sin_cos_extrema (x y : ℝ) (h : Real.sin x + Real.sin y = 1/3) :
  (∃ z w : ℝ, Real.sin z + Real.sin w = 1/3 ∧ 
    Real.sin w + (Real.cos z)^2 = 19/12) ∧
  (∀ a b : ℝ, Real.sin a + Real.sin b = 1/3 → 
    Real.sin b + (Real.cos a)^2 ≤ 19/12) ∧
  (∃ u v : ℝ, Real.sin u + Real.sin v = 1/3 ∧ 
    Real.sin v + (Real.cos u)^2 = -2/3) ∧
  (∀ c d : ℝ, Real.sin c + Real.sin d = 1/3 → 
    Real.sin d + (Real.cos c)^2 ≥ -2/3) :=
by sorry

end NUMINAMATH_CALUDE_sin_cos_extrema_l2474_247490


namespace NUMINAMATH_CALUDE_ladybug_count_l2474_247415

/-- The number of ladybugs with spots -/
def ladybugs_with_spots : ℕ := 12170

/-- The number of ladybugs without spots -/
def ladybugs_without_spots : ℕ := 54912

/-- The total number of ladybugs -/
def total_ladybugs : ℕ := ladybugs_with_spots + ladybugs_without_spots

theorem ladybug_count : total_ladybugs = 67082 := by
  sorry

end NUMINAMATH_CALUDE_ladybug_count_l2474_247415


namespace NUMINAMATH_CALUDE_stratified_sample_theorem_l2474_247430

/-- Represents a stratified sampling scenario -/
structure StratifiedSample where
  totalPopulation : ℕ
  totalSampleSize : ℕ
  stratumSize : ℕ
  stratumSampleSize : ℕ

/-- Checks if the stratified sample is proportional -/
def isProportional (s : StratifiedSample) : Prop :=
  s.stratumSampleSize * s.totalPopulation = s.totalSampleSize * s.stratumSize

theorem stratified_sample_theorem (s : StratifiedSample) 
  (h1 : s.totalPopulation = 2048)
  (h2 : s.totalSampleSize = 128)
  (h3 : s.stratumSize = 256)
  (h4 : isProportional s) :
  s.stratumSampleSize = 16 := by
  sorry

#check stratified_sample_theorem

end NUMINAMATH_CALUDE_stratified_sample_theorem_l2474_247430


namespace NUMINAMATH_CALUDE_complement_of_A_in_U_l2474_247444

def U : Set Int := {-2, -1, 1, 3, 5}
def A : Set Int := {-1, 3}

theorem complement_of_A_in_U : 
  {x ∈ U | x ∉ A} = {-2, 1, 5} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_in_U_l2474_247444


namespace NUMINAMATH_CALUDE_tangent_ellipse_hyperbola_l2474_247439

-- Define the ellipse equation
def ellipse (x y : ℝ) : Prop := x^2 + 9*y^2 = 9

-- Define the hyperbola equation
def hyperbola (x y m : ℝ) : Prop := x^2 - m*(y+1)^2 = 1

-- Define the tangency condition
def are_tangent (m : ℝ) : Prop :=
  ∃ x y : ℝ, ellipse x y ∧ hyperbola x y m ∧
  ∀ x' y' : ℝ, ellipse x' y' ∧ hyperbola x' y' m → (x', y') = (x, y)

-- State the theorem
theorem tangent_ellipse_hyperbola :
  ∀ m : ℝ, are_tangent m → m = 72 :=
by sorry

end NUMINAMATH_CALUDE_tangent_ellipse_hyperbola_l2474_247439


namespace NUMINAMATH_CALUDE_empty_quadratic_set_implies_m_greater_than_one_l2474_247491

theorem empty_quadratic_set_implies_m_greater_than_one (m : ℝ) : 
  (∀ x : ℝ, x^2 - 2*x + m ≠ 0) → m > 1 := by
  sorry

end NUMINAMATH_CALUDE_empty_quadratic_set_implies_m_greater_than_one_l2474_247491


namespace NUMINAMATH_CALUDE_zebra_stripes_l2474_247424

theorem zebra_stripes (w n b : ℕ) : 
  w + n = b + 1 →  -- Total black stripes = white stripes + 1
  b = w + 7 →      -- White stripes = wide black stripes + 7
  n = 8            -- Number of narrow black stripes is 8
:= by sorry

end NUMINAMATH_CALUDE_zebra_stripes_l2474_247424


namespace NUMINAMATH_CALUDE_symmetrical_line_intersection_l2474_247405

/-- Given points A and B, and a circle, prove that if the line symmetrical to AB about y=a intersects the circle, then a is in the range [1/3, 3/2]. -/
theorem symmetrical_line_intersection (a : ℝ) : 
  let A : ℝ × ℝ := (-2, 3)
  let B : ℝ × ℝ := (0, a)
  let circle (x y : ℝ) := (x + 3)^2 + (y + 2)^2 = 1
  let symmetrical_line (x y : ℝ) := (3 - a) * x - 2 * y + 2 * a = 0
  (∃ x y, circle x y ∧ symmetrical_line x y) → a ∈ Set.Icc (1/3) (3/2) :=
by sorry

end NUMINAMATH_CALUDE_symmetrical_line_intersection_l2474_247405


namespace NUMINAMATH_CALUDE_eliminate_denominators_l2474_247456

theorem eliminate_denominators (x : ℝ) :
  (1 / 2 * (x + 1) = 1 - 1 / 3 * x) →
  (3 * (x + 1) = 6 - 2 * x) :=
by sorry

end NUMINAMATH_CALUDE_eliminate_denominators_l2474_247456


namespace NUMINAMATH_CALUDE_no_even_three_digit_sum_31_l2474_247447

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def digit_sum (n : ℕ) : ℕ :=
  (n / 100) + ((n / 10) % 10) + (n % 10)

theorem no_even_three_digit_sum_31 :
  ¬ ∃ n : ℕ, is_three_digit n ∧ digit_sum n = 31 ∧ Even n :=
sorry

end NUMINAMATH_CALUDE_no_even_three_digit_sum_31_l2474_247447


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l2474_247482

def A : Set ℕ := {1, 2}
def B : Set ℕ := {2, 4, 6}

theorem union_of_A_and_B : A ∪ B = {1, 2, 4, 6} := by
  sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l2474_247482


namespace NUMINAMATH_CALUDE_males_not_listening_l2474_247477

/-- Radio station survey data -/
structure SurveyData where
  total_listeners : ℕ
  total_non_listeners : ℕ
  male_listeners : ℕ
  female_non_listeners : ℕ

/-- Theorem: Number of males who do not listen to the station -/
theorem males_not_listening (data : SurveyData)
  (h1 : data.total_listeners = 200)
  (h2 : data.total_non_listeners = 180)
  (h3 : data.male_listeners = 75)
  (h4 : data.female_non_listeners = 120) :
  data.total_listeners + data.total_non_listeners - data.male_listeners - data.female_non_listeners = 185 := by
  sorry

end NUMINAMATH_CALUDE_males_not_listening_l2474_247477


namespace NUMINAMATH_CALUDE_shondas_kids_l2474_247423

theorem shondas_kids (friends : ℕ) (other_adults : ℕ) (baskets : ℕ) (eggs_per_basket : ℕ) (eggs_per_person : ℕ) :
  friends = 10 →
  other_adults = 7 →
  baskets = 15 →
  eggs_per_basket = 12 →
  eggs_per_person = 9 →
  ∃ (shondas_kids : ℕ), shondas_kids = 2 ∧
    (shondas_kids + friends + other_adults + 1) * eggs_per_person = baskets * eggs_per_basket :=
by sorry

end NUMINAMATH_CALUDE_shondas_kids_l2474_247423


namespace NUMINAMATH_CALUDE_fraction_equivalence_l2474_247494

theorem fraction_equivalence (x y z : ℝ) (h1 : 2*x - z ≠ 0) (h2 : z ≠ 0) :
  (2*x + y) / (2*x - z) = y / (-z) ↔ y = -z :=
by sorry

end NUMINAMATH_CALUDE_fraction_equivalence_l2474_247494


namespace NUMINAMATH_CALUDE_complement_of_M_in_U_l2474_247466

def U : Set Nat := {1, 2, 3, 4, 5, 6}
def M : Set Nat := {1, 2, 4}

theorem complement_of_M_in_U : 
  (U \ M) = {3, 5, 6} := by sorry

end NUMINAMATH_CALUDE_complement_of_M_in_U_l2474_247466


namespace NUMINAMATH_CALUDE_sin_sixty_degrees_l2474_247471

theorem sin_sixty_degrees : Real.sin (π / 3) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_sixty_degrees_l2474_247471


namespace NUMINAMATH_CALUDE_perpendicular_vectors_l2474_247488

def a : ℝ × ℝ := (1, 2)
def b (m : ℝ) : ℝ × ℝ := (2*m - 1, -1)

theorem perpendicular_vectors (m : ℝ) : 
  (a.1 * (b m).1 + a.2 * (b m).2 = 0) → m = 3/2 :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_l2474_247488


namespace NUMINAMATH_CALUDE_age_ratio_proof_l2474_247492

/-- Given three people A, B, and C with the following conditions:
    - A is two years older than B
    - The total of the ages of A, B, and C is 22
    - B is 8 years old
    Prove that the ratio of B's age to C's age is 2:1 -/
theorem age_ratio_proof (a b c : ℕ) : 
  a = b + 2 →
  a + b + c = 22 →
  b = 8 →
  b / c = 2 := by
sorry

end NUMINAMATH_CALUDE_age_ratio_proof_l2474_247492


namespace NUMINAMATH_CALUDE_equation_solution_l2474_247467

theorem equation_solution :
  ∃ (x : ℝ), x > 0 ∧ 6 * Real.sqrt (4 + x) + 6 * Real.sqrt (4 - x) = 8 * Real.sqrt 5 ∧ x = Real.sqrt (1280 / 81) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2474_247467


namespace NUMINAMATH_CALUDE_committee_seating_arrangements_l2474_247474

/-- The number of Democrats on the committee -/
def num_democrats : ℕ := 6

/-- The number of Republicans on the committee -/
def num_republicans : ℕ := 4

/-- The total number of politicians on the committee -/
def total_politicians : ℕ := num_democrats + num_republicans

/-- Represents that all politicians are distinguishable -/
axiom politicians_distinguishable : True

/-- Represents the constraint that no two Republicans can sit next to each other -/
axiom no_adjacent_republicans : True

/-- The number of ways to arrange the politicians around a circular table -/
def arrangement_count : ℕ := 43200

/-- Theorem stating that the number of valid arrangements is 43,200 -/
theorem committee_seating_arrangements :
  arrangement_count = 43200 :=
sorry

end NUMINAMATH_CALUDE_committee_seating_arrangements_l2474_247474


namespace NUMINAMATH_CALUDE_john_average_increase_l2474_247412

def john_scores : List ℝ := [90, 85, 92, 95]

theorem john_average_increase :
  let initial_average := (john_scores.take 3).sum / 3
  let new_average := john_scores.sum / 4
  new_average - initial_average = 1.5 := by sorry

end NUMINAMATH_CALUDE_john_average_increase_l2474_247412


namespace NUMINAMATH_CALUDE_intersection_P_Q_l2474_247489

-- Define the sets P and Q
def P : Set ℝ := {x | x > 0}
def Q : Set ℝ := {x | -1 < x ∧ x < 2}

-- State the theorem
theorem intersection_P_Q : P ∩ Q = {x : ℝ | 0 < x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_intersection_P_Q_l2474_247489


namespace NUMINAMATH_CALUDE_james_fish_tanks_l2474_247406

def fish_tank_problem (num_tanks : ℕ) (fish_in_first_tank : ℕ) (total_fish : ℕ) : Prop :=
  ∃ (num_double_tanks : ℕ),
    num_tanks = 1 + num_double_tanks ∧
    fish_in_first_tank = 20 ∧
    total_fish = fish_in_first_tank + num_double_tanks * (2 * fish_in_first_tank) ∧
    total_fish = 100

theorem james_fish_tanks :
  ∃ (num_tanks : ℕ), fish_tank_problem num_tanks 20 100 ∧ num_tanks = 3 :=
sorry

end NUMINAMATH_CALUDE_james_fish_tanks_l2474_247406


namespace NUMINAMATH_CALUDE_percentage_difference_l2474_247470

theorem percentage_difference (n : ℝ) (h : n = 160) : 0.5 * n - 0.35 * n = 24 := by
  sorry

end NUMINAMATH_CALUDE_percentage_difference_l2474_247470


namespace NUMINAMATH_CALUDE_work_completion_theorem_l2474_247457

/-- Given a work that could be finished in 12 days, and was actually finished in 9 days
    after 10 more men joined, prove that the original number of men employed was 30. -/
theorem work_completion_theorem (original_days : ℕ) (actual_days : ℕ) (additional_men : ℕ) :
  original_days = 12 →
  actual_days = 9 →
  additional_men = 10 →
  ∃ (original_men : ℕ), original_men * original_days = (original_men + additional_men) * actual_days ∧ original_men = 30 :=
by
  sorry

end NUMINAMATH_CALUDE_work_completion_theorem_l2474_247457


namespace NUMINAMATH_CALUDE_sum_geq_sqrt_products_l2474_247465

theorem sum_geq_sqrt_products (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a + b + c ≥ Real.sqrt (a * b) + Real.sqrt (b * c) + Real.sqrt (c * a) := by
  sorry

end NUMINAMATH_CALUDE_sum_geq_sqrt_products_l2474_247465


namespace NUMINAMATH_CALUDE_largest_integer_negative_quadratic_seven_satisfies_inequality_eight_does_not_satisfy_inequality_l2474_247485

theorem largest_integer_negative_quadratic :
  ∀ n : ℤ, n^2 - 11*n + 24 < 0 → n ≤ 7 :=
by
  sorry

theorem seven_satisfies_inequality :
  (7 : ℤ)^2 - 11*7 + 24 < 0 :=
by
  sorry

theorem eight_does_not_satisfy_inequality :
  (8 : ℤ)^2 - 11*8 + 24 ≥ 0 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_integer_negative_quadratic_seven_satisfies_inequality_eight_does_not_satisfy_inequality_l2474_247485


namespace NUMINAMATH_CALUDE_polynomial_factor_sum_l2474_247434

theorem polynomial_factor_sum (d M N K : ℝ) :
  (∃ a b : ℝ, (X^2 + 3*X + 1) * (X^2 + a*X + b) = X^4 - d*X^3 + M*X^2 + N*X + K) →
  M + N + K = 5*K - 4*d - 11 :=
by sorry

end NUMINAMATH_CALUDE_polynomial_factor_sum_l2474_247434


namespace NUMINAMATH_CALUDE_point_translation_l2474_247416

/-- Given a point A with coordinates (1, -1), prove that moving it up by 2 units
    and then left by 3 units results in a point B with coordinates (-2, 1). -/
theorem point_translation (A B : ℝ × ℝ) :
  A = (1, -1) →
  B.1 = A.1 - 3 →
  B.2 = A.2 + 2 →
  B = (-2, 1) := by
sorry

end NUMINAMATH_CALUDE_point_translation_l2474_247416


namespace NUMINAMATH_CALUDE_largest_winning_start_is_correct_l2474_247421

/-- The largest starting integer that guarantees a win for Bernardo in the number game. -/
def largest_winning_start : ℕ := 40

/-- Checks if a given number is a valid starting number for Bernardo to win. -/
def is_valid_start (m : ℕ) : Prop :=
  m ≥ 1 ∧ m ≤ 500 ∧
  3 * m < 1500 ∧
  3 * m + 30 < 1500 ∧
  9 * m + 90 < 1500 ∧
  9 * m + 120 < 1500 ∧
  27 * m + 360 < 1500 ∧
  27 * m + 390 < 1500

theorem largest_winning_start_is_correct :
  is_valid_start largest_winning_start ∧
  ∀ n : ℕ, n > largest_winning_start → ¬ is_valid_start n :=
by sorry

end NUMINAMATH_CALUDE_largest_winning_start_is_correct_l2474_247421


namespace NUMINAMATH_CALUDE_sequence_sum_theorem_l2474_247438

theorem sequence_sum_theorem (a : ℕ+ → ℚ) (S : ℕ+ → ℚ) 
  (h1 : a 1 = 1)
  (h2 : ∀ n : ℕ+, S n = n^2 * a n) :
  (∀ n : ℕ+, S n = 2 * n / (n + 1)) ∧
  (∀ n : ℕ+, a n = 2 / (n * (n + 1))) := by
sorry

end NUMINAMATH_CALUDE_sequence_sum_theorem_l2474_247438


namespace NUMINAMATH_CALUDE_M_intersect_N_l2474_247426

def M : Set ℝ := {-1, 0, 1, 2}
def N : Set ℝ := {x : ℝ | -1 < x ∧ x < 1}

theorem M_intersect_N : M ∩ N = {0} := by
  sorry

end NUMINAMATH_CALUDE_M_intersect_N_l2474_247426


namespace NUMINAMATH_CALUDE_rotation_90_degrees_l2474_247498

-- Define the original line l
def line_l (x y : ℝ) : Prop := x - y + 1 = 0

-- Define the point A
def point_A : ℝ × ℝ := (2, 3)

-- Define the rotated line l₁
def line_l₁ (x y : ℝ) : Prop := x + y - 5 = 0

-- State the theorem
theorem rotation_90_degrees :
  ∀ (x y : ℝ),
  (∃ (x₀ y₀ : ℝ), line_l x₀ y₀ ∧ 
    (x - 2) = -(y₀ - 3) ∧ 
    (y - 3) = (x₀ - 2)) →
  line_l₁ x y := by sorry

end NUMINAMATH_CALUDE_rotation_90_degrees_l2474_247498


namespace NUMINAMATH_CALUDE_short_answer_time_l2474_247436

/-- Represents the time in minutes for various writing assignments --/
structure WritingTimes where
  essay : ℕ        -- Time for one essay in minutes
  paragraph : ℕ    -- Time for one paragraph in minutes
  shortAnswer : ℕ  -- Time for one short-answer question in minutes

/-- Represents the number of each type of assignment --/
structure AssignmentCounts where
  essays : ℕ
  paragraphs : ℕ
  shortAnswers : ℕ

/-- Calculates the total time in minutes for all assignments --/
def totalTime (times : WritingTimes) (counts : AssignmentCounts) : ℕ :=
  times.essay * counts.essays +
  times.paragraph * counts.paragraphs +
  times.shortAnswer * counts.shortAnswers

/-- The main theorem to prove --/
theorem short_answer_time 
  (times : WritingTimes) 
  (counts : AssignmentCounts) 
  (h1 : times.essay = 60)           -- Each essay takes 1 hour (60 minutes)
  (h2 : times.paragraph = 15)       -- Each paragraph takes 15 minutes
  (h3 : counts.essays = 2)          -- Karen assigns 2 essays
  (h4 : counts.paragraphs = 5)      -- Karen assigns 5 paragraphs
  (h5 : counts.shortAnswers = 15)   -- Karen assigns 15 short-answer questions
  (h6 : totalTime times counts = 240) -- Total homework time is 4 hours (240 minutes)
  : times.shortAnswer = 3 :=
by
  sorry


end NUMINAMATH_CALUDE_short_answer_time_l2474_247436


namespace NUMINAMATH_CALUDE_distance_from_circle_center_to_line_l2474_247461

/-- The distance from the center of a circle to a line --/
theorem distance_from_circle_center_to_line :
  let circle : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1^2 + p.2^2 - 4*p.1 = 0}
  let center : ℝ × ℝ := (2, 0)
  let line : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 - p.2 = 0}
  center ∈ circle →
  (∀ p ∈ circle, (p.1 - 2)^2 + p.2^2 = 4) →
  (∀ p ∈ line, p.1 = p.2) →
  ∃ d : ℝ, d = Real.sqrt 2 ∧ ∀ p ∈ line, (center.1 - p.1)^2 + (center.2 - p.2)^2 = d^2 :=
by sorry

end NUMINAMATH_CALUDE_distance_from_circle_center_to_line_l2474_247461


namespace NUMINAMATH_CALUDE_max_value_theorem_l2474_247418

theorem max_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : x^2 - 3*x*y + 5*y^2 = 9) : 
  x^2 + 3*x*y + 5*y^2 ≤ (315 + 297 * Real.sqrt 5) / 55 :=
by sorry

end NUMINAMATH_CALUDE_max_value_theorem_l2474_247418


namespace NUMINAMATH_CALUDE_f_min_at_three_l2474_247459

/-- The quadratic function f(x) = x^2 - 6x + 8 -/
def f (x : ℝ) : ℝ := x^2 - 6*x + 8

/-- Theorem stating that f(x) attains its minimum value when x = 3 -/
theorem f_min_at_three : 
  ∀ x : ℝ, f x ≥ f 3 := by sorry

end NUMINAMATH_CALUDE_f_min_at_three_l2474_247459
