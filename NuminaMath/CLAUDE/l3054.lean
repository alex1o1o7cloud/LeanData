import Mathlib

namespace NUMINAMATH_CALUDE_geometry_book_pages_l3054_305451

theorem geometry_book_pages :
  let new_edition : ℕ := 450
  let old_edition : ℕ := 340
  let deluxe_edition : ℕ := new_edition + old_edition + 125
  (2 * old_edition - 230 = new_edition) ∧
  (deluxe_edition ≥ old_edition + (old_edition / 10)) →
  old_edition = 340 :=
by sorry

end NUMINAMATH_CALUDE_geometry_book_pages_l3054_305451


namespace NUMINAMATH_CALUDE_sixth_term_equals_23_l3054_305446

/-- Given a sequence with general term a(n) = 4n - 1, prove that a(6) = 23 -/
theorem sixth_term_equals_23 (a : ℕ → ℤ) (h : ∀ n, a n = 4 * n - 1) : a 6 = 23 := by
  sorry

end NUMINAMATH_CALUDE_sixth_term_equals_23_l3054_305446


namespace NUMINAMATH_CALUDE_factorial_120_121_is_perfect_square_l3054_305444

/-- Definition of factorial -/
def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

/-- Definition of perfect square -/
def is_perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, n = k * k

/-- Theorem: 120! · 121! is a perfect square -/
theorem factorial_120_121_is_perfect_square :
  is_perfect_square (factorial 120 * factorial 121) := by
  sorry

end NUMINAMATH_CALUDE_factorial_120_121_is_perfect_square_l3054_305444


namespace NUMINAMATH_CALUDE_geometric_sequence_general_term_l3054_305488

/-- A geometric sequence {a_n} satisfying the given conditions has the specified general term. -/
theorem geometric_sequence_general_term (a : ℕ → ℝ) :
  (∀ n, a (n + 1) = a n * (a 2 / a 1)) →  -- Geometric sequence condition
  a 1 + a 3 = 10 →                        -- First given condition
  a 2 + a 4 = 5 →                         -- Second given condition
  ∀ n, a n = 8 * (1/2)^(n - 1) :=         -- Conclusion: general term
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_general_term_l3054_305488


namespace NUMINAMATH_CALUDE_integral_of_f_minus_x_l3054_305428

/-- Given a function f: ℝ → ℝ such that f'(x) = 2x + 1 for all x ∈ ℝ,
    prove that the definite integral of f(-x) from -1 to 3 equals 14/3. -/
theorem integral_of_f_minus_x (f : ℝ → ℝ) (h : ∀ x, deriv f x = 2 * x + 1) :
  ∫ x in (-1)..(3), f (-x) = 14/3 := by
  sorry

end NUMINAMATH_CALUDE_integral_of_f_minus_x_l3054_305428


namespace NUMINAMATH_CALUDE_hulk_jump_exceeds_500_l3054_305492

def hulk_jump (n : ℕ) : ℝ := 2 * (2 ^ (n - 1))

theorem hulk_jump_exceeds_500 :
  (∀ k < 9, hulk_jump k ≤ 500) ∧ hulk_jump 9 > 500 := by
  sorry

end NUMINAMATH_CALUDE_hulk_jump_exceeds_500_l3054_305492


namespace NUMINAMATH_CALUDE_triangle_abc_solutions_l3054_305433

theorem triangle_abc_solutions (b c : ℝ) (angle_B : ℝ) :
  b = 3 → c = 3 * Real.sqrt 3 → angle_B = π / 6 →
  ∃ (a angle_A angle_C : ℝ),
    ((angle_A = π / 2 ∧ angle_C = π / 3 ∧ a = Real.sqrt 21) ∨
     (angle_A = π / 6 ∧ angle_C = 2 * π / 3 ∧ a = 3)) ∧
    angle_A + angle_B + angle_C = π ∧
    a / (Real.sin angle_A) = b / (Real.sin angle_B) ∧
    b / (Real.sin angle_B) = c / (Real.sin angle_C) :=
by sorry

end NUMINAMATH_CALUDE_triangle_abc_solutions_l3054_305433


namespace NUMINAMATH_CALUDE_distribute_over_sum_diff_l3054_305472

theorem distribute_over_sum_diff (a b c : ℝ) : a * (a + b - c) = a^2 + a*b - a*c := by
  sorry

end NUMINAMATH_CALUDE_distribute_over_sum_diff_l3054_305472


namespace NUMINAMATH_CALUDE_complex_modulus_squared_l3054_305412

theorem complex_modulus_squared (w : ℂ) (h : w + 3 * Complex.abs w = -1 + 12 * Complex.I) :
  Complex.abs w ^ 2 = 2545 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_squared_l3054_305412


namespace NUMINAMATH_CALUDE_acute_triangle_inequality_l3054_305417

theorem acute_triangle_inequality (A B C : Real) (hA : 0 < A) (hB : 0 < B) (hC : 0 < C)
  (hABC : A + B + C = π) (hAcute : A < π/2 ∧ B < π/2 ∧ C < π/2) :
  (Real.sin A + Real.sin B + Real.sin C) * (1 / Real.sin A + 1 / Real.sin B + 1 / Real.sin C)
  ≤ π * (1 / A + 1 / B + 1 / C) := by
  sorry

end NUMINAMATH_CALUDE_acute_triangle_inequality_l3054_305417


namespace NUMINAMATH_CALUDE_range_of_a_l3054_305485

-- Define the function f(x) = |x+3| - |x-1|
def f (x : ℝ) : ℝ := |x + 3| - |x - 1|

-- Define the property that the solution set is non-empty
def has_solution (a : ℝ) : Prop :=
  ∃ x : ℝ, f x ≤ a^2 - 5*a

-- Theorem statement
theorem range_of_a (a : ℝ) :
  has_solution a → (a ≥ 4 ∨ a ≤ 1) :=
by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l3054_305485


namespace NUMINAMATH_CALUDE_angle_sum_in_square_configuration_l3054_305464

/-- Given a configuration of 13 identical squares with marked points, this theorem proves
    that the sum of specific angles equals 405 degrees. -/
theorem angle_sum_in_square_configuration :
  ∀ (FPB FPD APC APE AQG QCF RQF CQD : ℝ),
  RQF + CQD = 45 →
  FPB + FPD + APE = 180 →
  AQG + QCF + APC = 180 →
  (FPB + FPD + APC + APE) + (AQG + QCF + RQF + CQD) = 405 := by
  sorry

end NUMINAMATH_CALUDE_angle_sum_in_square_configuration_l3054_305464


namespace NUMINAMATH_CALUDE_sum_difference_theorem_l3054_305489

def sara_sum (n : ℕ) : ℕ := n * (n + 1) / 2

def round_to_nearest_five (x : ℕ) : ℕ :=
  5 * ((x + 2) / 5)

def mike_sum (n : ℕ) : ℕ :=
  List.sum (List.map round_to_nearest_five (List.range n))

theorem sum_difference_theorem :
  sara_sum 120 - mike_sum 120 = 6900 := by
  sorry

end NUMINAMATH_CALUDE_sum_difference_theorem_l3054_305489


namespace NUMINAMATH_CALUDE_typeC_migration_time_l3054_305453

/-- Represents the lakes in the migration sequence -/
inductive Lake : Type
| Jim : Lake
| Disney : Lake
| London : Lake
| Everest : Lake

/-- Defines the distance between two lakes -/
def distance (a b : Lake) : ℝ :=
  match a, b with
  | Lake.Jim, Lake.Disney => 42
  | Lake.Disney, Lake.London => 57
  | Lake.London, Lake.Everest => 65
  | Lake.Everest, Lake.Jim => 70
  | _, _ => 0  -- For all other combinations

/-- Calculates the total distance of one complete sequence -/
def totalDistance : ℝ :=
  distance Lake.Jim Lake.Disney +
  distance Lake.Disney Lake.London +
  distance Lake.London Lake.Everest +
  distance Lake.Everest Lake.Jim

/-- The average speed of Type C birds in miles per hour -/
def typeCSpeed : ℝ := 12

/-- Theorem: Type C birds take 39 hours to complete two full sequences -/
theorem typeC_migration_time :
  2 * (totalDistance / typeCSpeed) = 39 := by sorry


end NUMINAMATH_CALUDE_typeC_migration_time_l3054_305453


namespace NUMINAMATH_CALUDE_average_weight_increase_l3054_305429

/-- Proves that replacing a person weighing 58 kg with a person weighing 106 kg
    in a group of 12 people increases the average weight by 4 kg -/
theorem average_weight_increase (initial_average : ℝ) : 
  let initial_total_weight := 12 * initial_average
  let new_total_weight := initial_total_weight - 58 + 106
  let new_average := new_total_weight / 12
  new_average - initial_average = 4 := by
  sorry

end NUMINAMATH_CALUDE_average_weight_increase_l3054_305429


namespace NUMINAMATH_CALUDE_pet_ownership_percentage_l3054_305424

theorem pet_ownership_percentage (total_students : ℕ) (both_pets : ℕ)
  (h1 : total_students = 500)
  (h2 : both_pets = 50) :
  (both_pets : ℚ) / total_students * 100 = 10 := by
  sorry

end NUMINAMATH_CALUDE_pet_ownership_percentage_l3054_305424


namespace NUMINAMATH_CALUDE_factorization_identities_l3054_305439

theorem factorization_identities (x y m n p : ℝ) : 
  (x^2 + 2*x + 1 - y^2 = (x + y + 1)*(x - y + 1)) ∧ 
  (m^2 - n^2 - 2*n*p - p^2 = (m + n + p)*(m - n - p)) := by
  sorry

end NUMINAMATH_CALUDE_factorization_identities_l3054_305439


namespace NUMINAMATH_CALUDE_sum_of_roots_eq_fourteen_l3054_305496

theorem sum_of_roots_eq_fourteen : ∀ x₁ x₂ : ℝ, 
  (x₁ - 7)^2 = 16 ∧ (x₂ - 7)^2 = 16 → x₁ + x₂ = 14 := by sorry

end NUMINAMATH_CALUDE_sum_of_roots_eq_fourteen_l3054_305496


namespace NUMINAMATH_CALUDE_center_is_four_l3054_305490

-- Define the grid as a 3x3 matrix of natural numbers
def Grid := Matrix (Fin 3) (Fin 3) Nat

-- Define a predicate for consecutive numbers being adjacent
def consecutive_adjacent (g : Grid) : Prop := sorry

-- Define a function to get the edge numbers (excluding corners)
def edge_numbers (g : Grid) : List Nat := sorry

-- Define the sum of edge numbers
def edge_sum (g : Grid) : Nat := (edge_numbers g).sum

-- Define a predicate for the grid containing all numbers from 1 to 9
def contains_one_to_nine (g : Grid) : Prop := sorry

-- Define a function to get the center number
def center_number (g : Grid) : Nat := g 1 1

-- Main theorem
theorem center_is_four (g : Grid) 
  (h1 : consecutive_adjacent g)
  (h2 : edge_sum g = 28)
  (h3 : contains_one_to_nine g)
  (h4 : Even (center_number g)) :
  center_number g = 4 := by sorry

end NUMINAMATH_CALUDE_center_is_four_l3054_305490


namespace NUMINAMATH_CALUDE_license_plate_theorem_l3054_305400

/-- The number of letters in the alphabet -/
def alphabet_size : ℕ := 26

/-- The number of letter positions in the license plate -/
def letter_positions : ℕ := 4

/-- The number of odd digits -/
def odd_digit_count : ℕ := 5

/-- The number of digit positions in the license plate -/
def digit_positions : ℕ := 2

/-- Calculate the number of license plate combinations -/
def license_plate_combinations : ℕ :=
  Nat.choose alphabet_size 2 * Nat.choose letter_positions 2 * odd_digit_count * (odd_digit_count - 1)

theorem license_plate_theorem : license_plate_combinations = 39000 := by
  sorry

end NUMINAMATH_CALUDE_license_plate_theorem_l3054_305400


namespace NUMINAMATH_CALUDE_birthday_stickers_l3054_305494

theorem birthday_stickers (initial_stickers final_stickers : ℕ) 
  (h1 : initial_stickers = 39)
  (h2 : final_stickers = 61) :
  final_stickers - initial_stickers = 22 := by
sorry

end NUMINAMATH_CALUDE_birthday_stickers_l3054_305494


namespace NUMINAMATH_CALUDE_taylors_pets_l3054_305438

theorem taylors_pets (taylor_pets : ℕ) (total_pets : ℕ) : 
  (3 * (2 * taylor_pets) + 2 * 2 + taylor_pets = total_pets) →
  (total_pets = 32) →
  (taylor_pets = 4) := by
sorry

end NUMINAMATH_CALUDE_taylors_pets_l3054_305438


namespace NUMINAMATH_CALUDE_repeating_base_k_representation_l3054_305462

/-- Given positive integers m and k, if the repeating base-k representation of 3/28 is 0.121212...₍ₖ₎, then k = 10 -/
theorem repeating_base_k_representation (m k : ℕ+) :
  (∃ (a : ℕ → ℕ), (∀ n, a n < k) ∧
    (∀ n, a (2*n) = 1 ∧ a (2*n+1) = 2) ∧
    (3 : ℚ) / 28 = ∑' n, (a n : ℚ) / k^(n+1)) →
  k = 10 := by sorry

end NUMINAMATH_CALUDE_repeating_base_k_representation_l3054_305462


namespace NUMINAMATH_CALUDE_lcm_of_numbers_in_ratio_l3054_305450

theorem lcm_of_numbers_in_ratio (a b : ℕ) (h_ratio : a * 5 = b * 4) (h_smaller : a = 36) : 
  Nat.lcm a b = 1620 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_numbers_in_ratio_l3054_305450


namespace NUMINAMATH_CALUDE_problem_statement_l3054_305467

-- Define the line x + y - 3 = 0
def line1 (x y : ℝ) : Prop := x + y - 3 = 0

-- Define the vector (1, -1)
def vector1 : ℝ × ℝ := (1, -1)

-- Define the lines x + 2y - 4 = 0 and 2x + 4y + 1 = 0
def line2 (x y : ℝ) : Prop := x + 2*y - 4 = 0
def line3 (x y : ℝ) : Prop := 2*x + 4*y + 1 = 0

-- Define the point (3, 4)
def point1 : ℝ × ℝ := (3, 4)

-- Define a function to check if a line has equal intercepts on both axes
def has_equal_intercepts (a b c : ℝ) : Prop :=
  ∃ (t : ℝ), a * t + b * t + c = 0 ∧ t ≠ 0

theorem problem_statement :
  -- 1. (1,-1) is a directional vector of the line x+y-3=0
  (∀ (t : ℝ), line1 (vector1.1 * t) (vector1.2 * t)) ∧
  -- 2. The distance between lines x+2y-4=0 and 2x+4y+1=0 is 9√5/10
  (let d := (9 * Real.sqrt 5) / 10;
   ∀ (x y : ℝ), line2 x y → ∀ (x' y' : ℝ), line3 x' y' →
   ((x - x')^2 + (y - y')^2).sqrt = d) ∧
  -- 3. There are exactly 2 lines passing through point (3,4) with equal intercepts on the two coordinate axes
  (∃! (a₁ b₁ c₁ a₂ b₂ c₂ : ℝ),
    a₁ * point1.1 + b₁ * point1.2 + c₁ = 0 ∧
    a₂ * point1.1 + b₂ * point1.2 + c₂ = 0 ∧
    has_equal_intercepts a₁ b₁ c₁ ∧
    has_equal_intercepts a₂ b₂ c₂ ∧
    (a₁, b₁, c₁) ≠ (a₂, b₂, c₂)) :=
sorry

end NUMINAMATH_CALUDE_problem_statement_l3054_305467


namespace NUMINAMATH_CALUDE_regular_rate_is_three_dollars_l3054_305415

/-- Represents a worker's pay structure and hours worked -/
structure PayStructure where
  regularRate : ℝ
  regularHours : ℝ
  overtimeHours : ℝ
  totalPay : ℝ

/-- Calculates the total pay based on the pay structure -/
def calculateTotalPay (p : PayStructure) : ℝ :=
  p.regularRate * p.regularHours + 2 * p.regularRate * p.overtimeHours

/-- Theorem: Given the specified pay structure, the regular rate is $3 per hour -/
theorem regular_rate_is_three_dollars (p : PayStructure) 
    (h1 : p.regularHours = 40)
    (h2 : p.overtimeHours = 10)
    (h3 : p.totalPay = 180)
    (h4 : calculateTotalPay p = p.totalPay) : 
    p.regularRate = 3 := by
  sorry

#check regular_rate_is_three_dollars

end NUMINAMATH_CALUDE_regular_rate_is_three_dollars_l3054_305415


namespace NUMINAMATH_CALUDE_base_13_conversion_l3054_305497

/-- Represents a digit in base 13 -/
inductive Base13Digit
| D0 | D1 | D2 | D3 | D4 | D5 | D6 | D7 | D8 | D9 | A | B | C

/-- Converts a natural number to its value in base 13 -/
def toBase13Value (d : Base13Digit) : ℕ :=
  match d with
  | Base13Digit.D0 => 0
  | Base13Digit.D1 => 1
  | Base13Digit.D2 => 2
  | Base13Digit.D3 => 3
  | Base13Digit.D4 => 4
  | Base13Digit.D5 => 5
  | Base13Digit.D6 => 6
  | Base13Digit.D7 => 7
  | Base13Digit.D8 => 8
  | Base13Digit.D9 => 9
  | Base13Digit.A => 10
  | Base13Digit.B => 11
  | Base13Digit.C => 12

/-- Represents a two-digit number in base 13 -/
structure Base13Number :=
  (msb : Base13Digit)
  (lsb : Base13Digit)

/-- Converts a Base13Number to its decimal (base 10) value -/
def toDecimal (n : Base13Number) : ℕ :=
  13 * (toBase13Value n.msb) + (toBase13Value n.lsb)

theorem base_13_conversion :
  toDecimal (Base13Number.mk Base13Digit.C Base13Digit.D0) = 156 := by
  sorry

end NUMINAMATH_CALUDE_base_13_conversion_l3054_305497


namespace NUMINAMATH_CALUDE_cost_function_cheaper_values_l3054_305457

def C (n : ℕ) : ℕ :=
  if 1 ≤ n ∧ n ≤ 30 then 15 * n
  else if 31 ≤ n ∧ n ≤ 65 then 13 * n
  else 11 * n

theorem cost_function_cheaper_values :
  (∃ (S : Finset ℕ), S.card = 6 ∧ 
    (∀ n, n ∈ S ↔ (C (n + 1) < C n ∧ n ≥ 1))) :=
by sorry

end NUMINAMATH_CALUDE_cost_function_cheaper_values_l3054_305457


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3054_305460

/-- An arithmetic sequence is a sequence where the difference between
    any two consecutive terms is constant. -/
def isArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Given an arithmetic sequence a_n where 
    2(a_1 + a_3 + a_5) + 3(a_8 + a_10) = 36, prove that a_6 = 3 -/
theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
  (h_arith : isArithmeticSequence a)
  (h_sum : 2 * (a 1 + a 3 + a 5) + 3 * (a 8 + a 10) = 36) :
  a 6 = 3 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3054_305460


namespace NUMINAMATH_CALUDE_three_quadrilaterals_with_circumcenter_l3054_305419

/-- A quadrilateral is a polygon with four sides and four vertices. -/
structure Quadrilateral :=
  (vertices : Fin 4 → ℝ × ℝ)

/-- A point is equidistant from all vertices of a quadrilateral. -/
def has_circumcenter (q : Quadrilateral) : Prop :=
  ∃ (c : ℝ × ℝ), ∀ (i : Fin 4), dist c (q.vertices i) = dist c (q.vertices 0)

/-- A kite is a quadrilateral with two pairs of adjacent congruent sides. -/
def is_kite (q : Quadrilateral) : Prop := sorry

/-- A quadrilateral has exactly two right angles. -/
def has_two_right_angles (q : Quadrilateral) : Prop := sorry

/-- A square is a quadrilateral with all sides equal and all angles right angles. -/
def is_square (q : Quadrilateral) : Prop := sorry

/-- A rhombus is a quadrilateral with all sides equal. -/
def is_rhombus (q : Quadrilateral) : Prop := sorry

/-- An equilateral trapezoid is a trapezoid with the non-parallel sides equal. -/
def is_equilateral_trapezoid (q : Quadrilateral) : Prop := sorry

/-- A quadrilateral can be inscribed in a circle. -/
def is_cyclic (q : Quadrilateral) : Prop := sorry

/-- The main theorem stating that exactly 3 types of the given quadrilaterals have a circumcenter. -/
theorem three_quadrilaterals_with_circumcenter : 
  ∃ (a b c : Quadrilateral),
    (is_kite a ∧ has_two_right_angles a ∧ has_circumcenter a) ∧
    (is_square b ∧ has_circumcenter b) ∧
    (is_equilateral_trapezoid c ∧ is_cyclic c ∧ has_circumcenter c) ∧
    (∀ (d : Quadrilateral), 
      (is_rhombus d ∧ ¬is_square d) → ¬has_circumcenter d) ∧
    (∀ (e : Quadrilateral),
      has_circumcenter e → 
      (e = a ∨ e = b ∨ e = c)) :=
sorry

end NUMINAMATH_CALUDE_three_quadrilaterals_with_circumcenter_l3054_305419


namespace NUMINAMATH_CALUDE_points_on_line_l3054_305404

/-- Given a line with equation 7x + 2y = 41, prove that the points A(5, 3) and B(-5, 38) lie on this line. -/
theorem points_on_line :
  let line : ℝ → ℝ → Prop := λ x y => 7 * x + 2 * y = 41
  line 5 3 ∧ line (-5) 38 := by
sorry

end NUMINAMATH_CALUDE_points_on_line_l3054_305404


namespace NUMINAMATH_CALUDE_pumps_emptying_time_l3054_305498

/-- Represents the time (in hours) it takes for pumps A, B, and C to empty a pool when working together. -/
def combined_emptying_time (rate_A rate_B rate_C : ℚ) : ℚ :=
  1 / (rate_A + rate_B + rate_C)

/-- Theorem stating that pumps A, B, and C with given rates will empty the pool in 24/13 hours when working together. -/
theorem pumps_emptying_time :
  let rate_A : ℚ := 1/4
  let rate_B : ℚ := 1/6
  let rate_C : ℚ := 1/8
  combined_emptying_time rate_A rate_B rate_C = 24/13 := by
  sorry

#eval (24 : ℚ) / 13 * 60 -- Converts the result to minutes

end NUMINAMATH_CALUDE_pumps_emptying_time_l3054_305498


namespace NUMINAMATH_CALUDE_f_sum_symmetric_l3054_305499

def f (x : ℝ) : ℝ := x^3 + 2*x

theorem f_sum_symmetric : f 5 + f (-5) = 0 := by
  sorry

end NUMINAMATH_CALUDE_f_sum_symmetric_l3054_305499


namespace NUMINAMATH_CALUDE_expected_value_of_marbles_l3054_305479

/-- The number of marbles in the bag -/
def n : ℕ := 7

/-- The number of marbles drawn -/
def k : ℕ := 3

/-- The set of marble numbers -/
def marbles : Finset ℕ := Finset.range n

/-- The sum of all marble numbers -/
def total_sum : ℕ := Finset.sum marbles id

/-- The number of ways to choose k marbles from n marbles -/
def combinations : ℕ := Nat.choose n k

/-- The expected value of the sum of k randomly drawn marbles -/
def expected_value : ℚ := (k : ℚ) * (total_sum : ℚ) / (n : ℚ)

theorem expected_value_of_marbles :
  expected_value = 12 := by sorry

end NUMINAMATH_CALUDE_expected_value_of_marbles_l3054_305479


namespace NUMINAMATH_CALUDE_shower_tiles_count_l3054_305405

/-- Represents a shower with three walls -/
structure Shower :=
  (width : Nat)  -- Number of tiles in width
  (height : Nat) -- Number of tiles in height

/-- Calculates the total number of tiles in a shower -/
def totalTiles (s : Shower) : Nat :=
  3 * s.width * s.height

/-- Theorem stating that a shower with 8 tiles in width and 20 in height has 480 tiles in total -/
theorem shower_tiles_count : 
  ∀ s : Shower, s.width = 8 → s.height = 20 → totalTiles s = 480 := by
  sorry

end NUMINAMATH_CALUDE_shower_tiles_count_l3054_305405


namespace NUMINAMATH_CALUDE_quadratic_factorization_l3054_305470

theorem quadratic_factorization (p q : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ = -2 ∧ x₂ = 3/2 ∧ 
   ∀ x : ℝ, 2*x^2 + p*x + q = 0 ↔ x = x₁ ∨ x = x₂) →
  ∀ x : ℝ, 2*x^2 + p*x + q = 0 ↔ (x + 2)*(2*x - 3) = 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l3054_305470


namespace NUMINAMATH_CALUDE_hyperbola_trisect_foci_eccentricity_l3054_305466

/-- A hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola (a b : ℝ) where
  pos_a : 0 < a
  pos_b : 0 < b

/-- The eccentricity of a hyperbola -/
def eccentricity (h : Hyperbola a b) : ℝ := sorry

/-- Theorem: If the vertices of a hyperbola trisect the line segment between its foci,
    then its eccentricity is 3 -/
theorem hyperbola_trisect_foci_eccentricity (a b : ℝ) (h : Hyperbola a b) 
    (trisect : ∃ (c : ℝ), c = 3 * a) : eccentricity h = 3 := by sorry

end NUMINAMATH_CALUDE_hyperbola_trisect_foci_eccentricity_l3054_305466


namespace NUMINAMATH_CALUDE_javiers_cats_l3054_305476

/-- Calculates the number of cats in Javier's household -/
def number_of_cats (adults children dogs total_legs : ℕ) : ℕ :=
  let human_legs := 2 * (adults + children)
  let dog_legs := 4 * dogs
  let remaining_legs := total_legs - human_legs - dog_legs
  remaining_legs / 4

/-- Theorem stating that the number of cats in Javier's household is 1 -/
theorem javiers_cats :
  number_of_cats 2 3 2 22 = 1 :=
by sorry

end NUMINAMATH_CALUDE_javiers_cats_l3054_305476


namespace NUMINAMATH_CALUDE_inequality_range_of_a_l3054_305480

theorem inequality_range_of_a (a : ℝ) : 
  (∀ x > 0, Real.log x + a * x + 1 - x * Real.exp (2 * x) ≤ 0) → 
  a ≤ 2 := by
sorry

end NUMINAMATH_CALUDE_inequality_range_of_a_l3054_305480


namespace NUMINAMATH_CALUDE_midpoint_triangle_area_ratio_l3054_305435

/-- Given a triangle with area T, the triangle formed by joining the midpoints of its sides has area M = T/4 -/
theorem midpoint_triangle_area_ratio (T : ℝ) (h : T > 0) : 
  ∃ M : ℝ, M = T / 4 ∧ M > 0 := by
sorry

end NUMINAMATH_CALUDE_midpoint_triangle_area_ratio_l3054_305435


namespace NUMINAMATH_CALUDE_total_savings_theorem_l3054_305421

/-- The amount of money saved per month in dollars -/
def monthly_savings : ℕ := 4000

/-- The number of months in a year -/
def months_in_year : ℕ := 12

/-- Theorem: If Abigail saves $4,000 every month for an entire year, 
    the total amount saved will be $48,000 -/
theorem total_savings_theorem : 
  monthly_savings * months_in_year = 48000 := by
  sorry

end NUMINAMATH_CALUDE_total_savings_theorem_l3054_305421


namespace NUMINAMATH_CALUDE_harry_beach_collection_l3054_305427

/-- The number of items Harry has left after his walk on the beach -/
def items_left (sea_stars seashells snails lost : ℕ) : ℕ :=
  sea_stars + seashells + snails - lost

/-- Theorem stating that Harry has 59 items left after his walk -/
theorem harry_beach_collection : items_left 34 21 29 25 = 59 := by
  sorry

end NUMINAMATH_CALUDE_harry_beach_collection_l3054_305427


namespace NUMINAMATH_CALUDE_prob_multiple_of_3_twice_in_four_rolls_l3054_305465

/-- The probability of rolling a multiple of 3 on a fair six-sided die -/
def prob_multiple_of_3 : ℚ := 1 / 3

/-- The number of times the die is rolled -/
def num_rolls : ℕ := 4

/-- The number of times we want to see a multiple of 3 -/
def target_occurrences : ℕ := 2

/-- The probability of rolling a multiple of 3 exactly twice in four rolls of a fair die -/
theorem prob_multiple_of_3_twice_in_four_rolls :
  Nat.choose num_rolls target_occurrences * prob_multiple_of_3 ^ target_occurrences * (1 - prob_multiple_of_3) ^ (num_rolls - target_occurrences) = 8 / 27 := by
  sorry

end NUMINAMATH_CALUDE_prob_multiple_of_3_twice_in_four_rolls_l3054_305465


namespace NUMINAMATH_CALUDE_polar_to_rectangular_conversion_l3054_305414

theorem polar_to_rectangular_conversion 
  (r φ x y : ℝ) 
  (h1 : r = 7 / (2 * Real.cos φ - 5 * Real.sin φ))
  (h2 : x = r * Real.cos φ)
  (h3 : y = r * Real.sin φ) :
  2 * x - 5 * y = 7 := by
sorry

end NUMINAMATH_CALUDE_polar_to_rectangular_conversion_l3054_305414


namespace NUMINAMATH_CALUDE_triangle_area_is_one_l3054_305475

/-- The area of a triangle bounded by the x-axis and two lines -/
def triangleArea (line1 : ℝ → ℝ → Prop) (line2 : ℝ → ℝ → Prop) : ℝ :=
  1 -- We define the area as 1 based on the problem statement

/-- The first line equation: y - 2x = 2 -/
def line1 (x y : ℝ) : Prop :=
  y - 2*x = 2

/-- The second line equation: 2y - x = 1 -/
def line2 (x y : ℝ) : Prop :=
  2*y - x = 1

/-- Theorem stating that the area of the triangle is 1 -/
theorem triangle_area_is_one :
  triangleArea line1 line2 = 1 := by
  sorry


end NUMINAMATH_CALUDE_triangle_area_is_one_l3054_305475


namespace NUMINAMATH_CALUDE_negation_of_forall_positive_negation_of_greater_than_zero_l3054_305486

theorem negation_of_forall_positive (p : ℝ → Prop) :
  (¬∀ x : ℝ, p x) ↔ (∃ x : ℝ, ¬(p x)) :=
by sorry

theorem negation_of_greater_than_zero :
  (¬∀ x : ℝ, 2 * x^2 - 1 > 0) ↔ (∃ x : ℝ, 2 * x^2 - 1 ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_forall_positive_negation_of_greater_than_zero_l3054_305486


namespace NUMINAMATH_CALUDE_sequence_sum_1993_l3054_305482

def sequence_sum (n : ℕ) : ℤ :=
  let group_sum := 5
  let num_groups := n / 5
  ↑num_groups * group_sum

theorem sequence_sum_1993 :
  sequence_sum 1993 = 1990 :=
by sorry

end NUMINAMATH_CALUDE_sequence_sum_1993_l3054_305482


namespace NUMINAMATH_CALUDE_no_solution_for_floor_sum_l3054_305406

theorem no_solution_for_floor_sum (x : ℝ) : 
  ⌊x⌋ + ⌊2*x⌋ + ⌊4*x⌋ + ⌊8*x⌋ + ⌊16*x⌋ + ⌊32*x⌋ ≠ 12345 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_for_floor_sum_l3054_305406


namespace NUMINAMATH_CALUDE_red_note_rows_l3054_305441

theorem red_note_rows (red_notes_per_row : ℕ) (blue_notes_per_red : ℕ) (additional_blue_notes : ℕ) (total_notes : ℕ) :
  red_notes_per_row = 6 →
  blue_notes_per_red = 2 →
  additional_blue_notes = 10 →
  total_notes = 100 →
  ∃ (rows : ℕ), rows * red_notes_per_row + rows * red_notes_per_row * blue_notes_per_red + additional_blue_notes = total_notes ∧ rows = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_red_note_rows_l3054_305441


namespace NUMINAMATH_CALUDE_outfit_combinations_l3054_305411

theorem outfit_combinations (s p h : ℕ) (hs : s = 5) (hp : p = 6) (hh : h = 3) :
  s * p * h = 90 := by sorry

end NUMINAMATH_CALUDE_outfit_combinations_l3054_305411


namespace NUMINAMATH_CALUDE_joan_has_six_balloons_l3054_305483

/-- The number of orange balloons Joan has now, given she initially had 8 and lost 2. -/
def joans_balloons : ℕ := 8 - 2

/-- Theorem stating that Joan has 6 orange balloons now. -/
theorem joan_has_six_balloons : joans_balloons = 6 := by
  sorry

end NUMINAMATH_CALUDE_joan_has_six_balloons_l3054_305483


namespace NUMINAMATH_CALUDE_sum_of_common_roots_l3054_305407

theorem sum_of_common_roots (a : ℝ) :
  (∃ x : ℝ, x^2 + (2*a - 5)*x + a^2 + 1 = 0 ∧ 
            x^3 + (2*a - 5)*x^2 + (a^2 + 1)*x + a^2 - 4 = 0) →
  (∃ x y : ℝ, x^2 - 9*x + 5 = 0 ∧ y^2 - 9*y + 5 = 0 ∧ x + y = 9) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_common_roots_l3054_305407


namespace NUMINAMATH_CALUDE_arithmetic_progression_x_value_l3054_305468

def is_arithmetic_progression (a b c : ℝ) : Prop :=
  b - a = c - b

theorem arithmetic_progression_x_value :
  ∀ x : ℝ, is_arithmetic_progression (2*x - 3) (3*x - 2) (5*x + 2) → x = -3 :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_progression_x_value_l3054_305468


namespace NUMINAMATH_CALUDE_z_coordinate_at_x_7_l3054_305416

/-- A line in 3D space passing through two points -/
structure Line3D where
  point1 : ℝ × ℝ × ℝ
  point2 : ℝ × ℝ × ℝ

/-- Get the z-coordinate of a point on the line given its x-coordinate -/
def get_z_coordinate (line : Line3D) (x : ℝ) : ℝ :=
  sorry

theorem z_coordinate_at_x_7 (line : Line3D) 
  (h1 : line.point1 = (1, 4, 3)) 
  (h2 : line.point2 = (4, 3, 0)) : 
  get_z_coordinate line 7 = -3 := by
  sorry

end NUMINAMATH_CALUDE_z_coordinate_at_x_7_l3054_305416


namespace NUMINAMATH_CALUDE_marys_stickers_l3054_305454

theorem marys_stickers (total_stickers : ℕ) (friends : ℕ) (other_students : ℕ) (stickers_per_other : ℕ) (leftover_stickers : ℕ) (total_students : ℕ) :
  total_stickers = 50 →
  friends = 5 →
  other_students = total_students - friends - 1 →
  stickers_per_other = 2 →
  leftover_stickers = 8 →
  total_students = 17 →
  (total_stickers - leftover_stickers - other_students * stickers_per_other) / friends = 4 := by
  sorry

#check marys_stickers

end NUMINAMATH_CALUDE_marys_stickers_l3054_305454


namespace NUMINAMATH_CALUDE_profit_equation_l3054_305422

/-- The profit function for a commodity -/
def profit (x : ℝ) : ℝ :=
  let cost_price : ℝ := 30
  let quantity_sold : ℝ := 200 - x
  (x - cost_price) * quantity_sold

theorem profit_equation (x : ℝ) : profit x = -x^2 + 230*x - 6000 := by
  sorry

end NUMINAMATH_CALUDE_profit_equation_l3054_305422


namespace NUMINAMATH_CALUDE_similarity_circle_theorem_l3054_305458

-- Define the types for figures, lines, and points
variable (F : Type) (L : Type) (P : Type)

-- Define the properties and relations
variable (similar : F → F → Prop)
variable (corresponding_line : F → L → Prop)
variable (intersect_at : L → L → L → P → Prop)
variable (lies_on_circumcircle : P → F → F → F → Prop)
variable (passes_through : L → P → Prop)

-- Theorem statement
theorem similarity_circle_theorem 
  (F₁ F₂ F₃ : F) 
  (l₁ l₂ l₃ : L) 
  (W : P) :
  similar F₁ F₂ ∧ similar F₂ F₃ ∧ similar F₁ F₃ →
  corresponding_line F₁ l₁ ∧ corresponding_line F₂ l₂ ∧ corresponding_line F₃ l₃ →
  intersect_at l₁ l₂ l₃ W →
  lies_on_circumcircle W F₁ F₂ F₃ ∧
  ∃ (J₁ J₂ J₃ : P),
    lies_on_circumcircle J₁ F₁ F₂ F₃ ∧
    lies_on_circumcircle J₂ F₁ F₂ F₃ ∧
    lies_on_circumcircle J₃ F₁ F₂ F₃ ∧
    passes_through l₁ J₁ ∧
    passes_through l₂ J₂ ∧
    passes_through l₃ J₃ :=
by sorry

end NUMINAMATH_CALUDE_similarity_circle_theorem_l3054_305458


namespace NUMINAMATH_CALUDE_exists_fraction_with_99th_digit_4_l3054_305436

/-- Represents a decimal expansion as a sequence of digits -/
def DecimalExpansion := ℕ → Fin 10

/-- Returns the nth digit after the decimal point in the decimal expansion of a rational number -/
noncomputable def nthDigitAfterDecimal (q : ℚ) (n : ℕ) : Fin 10 := sorry

/-- The decimal expansion of 3/11 -/
def threeElevenths : DecimalExpansion := 
  fun n => if n % 2 = 0 then 2 else 7

theorem exists_fraction_with_99th_digit_4 : 
  ∃ q : ℚ, nthDigitAfterDecimal (q + 3/11) 99 = 4 := by sorry

end NUMINAMATH_CALUDE_exists_fraction_with_99th_digit_4_l3054_305436


namespace NUMINAMATH_CALUDE_sin_function_value_l3054_305402

/-- Given that the terminal side of angle φ passes through point P(3, -4),
    and the distance between two adjacent symmetry axes of the graph of
    the function f(x) = sin(ωx + φ) (ω > 0) is equal to π/2,
    prove that f(π/4) = 3/5 -/
theorem sin_function_value (φ ω : ℝ) (h1 : ω > 0) 
    (h2 : (3 : ℝ) / Real.sqrt (3^2 + 4^2) = Real.cos φ)
    (h3 : (-4 : ℝ) / Real.sqrt (3^2 + 4^2) = Real.sin φ)
    (h4 : π / (2 * ω) = π / 2) :
  Real.sin (ω * (π / 4) + φ) = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_sin_function_value_l3054_305402


namespace NUMINAMATH_CALUDE_circle_tangent_to_parabola_directrix_l3054_305425

theorem circle_tangent_to_parabola_directrix (p : ℝ) : 
  p > 0 → 
  (∃ x y : ℝ, x^2 + y^2 - 6*x - 7 = 0 ∧ 
              y^2 = 2*p*x ∧ 
              x = -p) → 
  p = 2 := by
sorry

end NUMINAMATH_CALUDE_circle_tangent_to_parabola_directrix_l3054_305425


namespace NUMINAMATH_CALUDE_similar_triangles_problem_l3054_305423

/-- Triangle similarity -/
structure SimilarTriangles (G H I J K L : ℝ × ℝ) : Prop where
  similar : True  -- Placeholder for similarity condition

/-- Angle measure in degrees -/
def angle_measure (p q r : ℝ × ℝ) : ℝ := sorry

theorem similar_triangles_problem 
  (G H I J K L : ℝ × ℝ) 
  (sim : SimilarTriangles G H I J K L)
  (gh_length : dist G H = 8)
  (hi_length : dist H I = 16)
  (kl_length : dist K L = 24)
  (ghi_angle : angle_measure G H I = 30) :
  dist J K = 12 ∧ angle_measure J K L = 30 := by
  sorry

end NUMINAMATH_CALUDE_similar_triangles_problem_l3054_305423


namespace NUMINAMATH_CALUDE_factor_of_polynomial_l3054_305463

theorem factor_of_polynomial (x : ℝ) : 
  ∃ (q : ℝ → ℝ), (x^4 + 4 : ℝ) = (x^2 - 2*x + 2) * q x :=
sorry

end NUMINAMATH_CALUDE_factor_of_polynomial_l3054_305463


namespace NUMINAMATH_CALUDE_right_triangle_identification_l3054_305431

def is_right_triangle (a b c : ℕ) : Prop := a * a + b * b = c * c

theorem right_triangle_identification :
  (¬ is_right_triangle 2 3 4) ∧
  (is_right_triangle 3 4 5) ∧
  (¬ is_right_triangle 4 5 6) ∧
  (¬ is_right_triangle 5 6 7) :=
sorry

end NUMINAMATH_CALUDE_right_triangle_identification_l3054_305431


namespace NUMINAMATH_CALUDE_tank_filling_flow_rate_l3054_305477

/-- The flow rate of a pipe filling a tank, given specific conditions --/
theorem tank_filling_flow_rate : ℝ := by
  -- Define the tank capacity
  let tank_capacity : ℝ := 1000

  -- Define the initial water level (half-full)
  let initial_water : ℝ := tank_capacity / 2

  -- Define the drain rates
  let drain_rate_1 : ℝ := 1000 / 4  -- 1 kiloliter every 4 minutes
  let drain_rate_2 : ℝ := 1000 / 6  -- 1 kiloliter every 6 minutes

  -- Define the total drain rate
  let total_drain_rate : ℝ := drain_rate_1 + drain_rate_2

  -- Define the time to fill the tank completely
  let fill_time : ℝ := 6

  -- Define the volume of water added
  let water_added : ℝ := tank_capacity - initial_water

  -- Define the flow rate of the pipe
  let flow_rate : ℝ := (water_added / fill_time) + total_drain_rate

  -- Prove that the flow rate is 500 liters per minute
  have h : flow_rate = 500 := by sorry

  exact 500


end NUMINAMATH_CALUDE_tank_filling_flow_rate_l3054_305477


namespace NUMINAMATH_CALUDE_julie_bought_two_boxes_l3054_305471

/-- Represents the number of boxes of standard paper Julie bought -/
def boxes_bought : ℕ := 2

/-- Represents the number of packages per box -/
def packages_per_box : ℕ := 5

/-- Represents the number of sheets per package -/
def sheets_per_package : ℕ := 250

/-- Represents the number of sheets used per newspaper -/
def sheets_per_newspaper : ℕ := 25

/-- Represents the number of newspapers Julie can print -/
def newspapers_printed : ℕ := 100

/-- Theorem stating that Julie bought 2 boxes of standard paper -/
theorem julie_bought_two_boxes :
  boxes_bought * packages_per_box * sheets_per_package =
  newspapers_printed * sheets_per_newspaper := by
  sorry

end NUMINAMATH_CALUDE_julie_bought_two_boxes_l3054_305471


namespace NUMINAMATH_CALUDE_business_investment_problem_l3054_305440

/-- Proves that A's investment is 16000, given the conditions of the business problem -/
theorem business_investment_problem (b_investment c_investment : ℕ) 
  (b_profit : ℕ) (profit_difference : ℕ) :
  b_investment = 10000 →
  c_investment = 12000 →
  b_profit = 1400 →
  profit_difference = 560 →
  ∃ (a_investment : ℕ), 
    a_investment * b_profit = b_investment * (a_investment * b_profit / b_investment - c_investment * b_profit / b_investment + profit_difference) ∧ 
    a_investment = 16000 := by
  sorry

end NUMINAMATH_CALUDE_business_investment_problem_l3054_305440


namespace NUMINAMATH_CALUDE_checkerboard_corner_sum_l3054_305445

theorem checkerboard_corner_sum : 
  let n : ℕ := 8  -- size of the checkerboard
  let total_squares : ℕ := n * n
  let top_left : ℕ := 1
  let top_right : ℕ := n
  let bottom_left : ℕ := total_squares - n + 1
  let bottom_right : ℕ := total_squares
  top_left + top_right + bottom_left + bottom_right = 130 :=
by sorry

end NUMINAMATH_CALUDE_checkerboard_corner_sum_l3054_305445


namespace NUMINAMATH_CALUDE_triangle_trig_ratio_l3054_305430

theorem triangle_trig_ratio (A B C : ℝ) (h : 0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π) 
  (ratio : Real.sin A / Real.sin B = 2/3 ∧ Real.sin B / Real.sin C = 3/4) : 
  Real.cos C = -1/4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_trig_ratio_l3054_305430


namespace NUMINAMATH_CALUDE_paulines_garden_l3054_305426

/-- Represents the number of kinds of cucumbers in Pauline's garden -/
def cucumber_kinds : ℕ := sorry

/-- The total number of spaces in the garden -/
def total_spaces : ℕ := 10 * 15

/-- The number of tomatoes planted -/
def tomatoes : ℕ := 3 * 5

/-- The number of cucumbers planted -/
def cucumbers : ℕ := cucumber_kinds * 4

/-- The number of potatoes planted -/
def potatoes : ℕ := 30

/-- The number of additional vegetables that can be planted -/
def additional_vegetables : ℕ := 85

theorem paulines_garden :
  cucumber_kinds = 5 :=
by sorry

end NUMINAMATH_CALUDE_paulines_garden_l3054_305426


namespace NUMINAMATH_CALUDE_highway_scenario_solution_l3054_305413

/-- Represents the scenario of a person walking along a highway with buses passing by -/
structure HighwayScenario where
  personSpeed : ℝ
  busSpeed : ℝ
  busDepartureInterval : ℝ
  oncomingBusInterval : ℝ
  overtakingBusInterval : ℝ
  busDistance : ℝ

/-- Checks if the given scenario satisfies all conditions -/
def isValidScenario (s : HighwayScenario) : Prop :=
  s.personSpeed > 0 ∧
  s.busSpeed > s.personSpeed ∧
  s.oncomingBusInterval * (s.busSpeed + s.personSpeed) = s.busDistance ∧
  s.overtakingBusInterval * (s.busSpeed - s.personSpeed) = s.busDistance ∧
  s.busDepartureInterval = s.busDistance / s.busSpeed

/-- The main theorem stating the unique solution to the highway scenario -/
theorem highway_scenario_solution :
  ∃! s : HighwayScenario, isValidScenario s ∧
    s.oncomingBusInterval = 4 ∧
    s.overtakingBusInterval = 6 ∧
    s.busDistance = 1200 ∧
    s.personSpeed = 50 ∧
    s.busSpeed = 250 ∧
    s.busDepartureInterval = 4.8 := by
  sorry


end NUMINAMATH_CALUDE_highway_scenario_solution_l3054_305413


namespace NUMINAMATH_CALUDE_base_10_515_equals_base_6_2215_l3054_305459

/-- Converts a base-6 number to base-10 --/
def base6ToBase10 (a b c d : ℕ) : ℕ :=
  a * 6^3 + b * 6^2 + c * 6^1 + d * 6^0

/-- Theorem stating that 515 in base 10 is equal to 2215 in base 6 --/
theorem base_10_515_equals_base_6_2215 :
  515 = base6ToBase10 2 2 1 5 := by
  sorry

end NUMINAMATH_CALUDE_base_10_515_equals_base_6_2215_l3054_305459


namespace NUMINAMATH_CALUDE_units_digit_of_expression_l3054_305455

theorem units_digit_of_expression : 
  (20 * 21 * 22 * 23 * 24 * 25) / 1000 % 10 = 2 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_expression_l3054_305455


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l3054_305452

theorem partial_fraction_decomposition (M₁ M₂ : ℝ) : 
  (∀ x : ℝ, x ≠ 2 ∧ x ≠ 3 → (50 * x - 42) / (x^2 - 5*x + 6) = M₁ / (x - 2) + M₂ / (x - 3)) → 
  M₁ * M₂ = -6264 := by
sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l3054_305452


namespace NUMINAMATH_CALUDE_select_students_count_l3054_305434

/-- The number of ways to select 3 students from 5 boys and 3 girls, including both genders -/
def select_students : ℕ :=
  Nat.choose 3 1 * Nat.choose 5 2 + Nat.choose 3 2 * Nat.choose 5 1

/-- Theorem stating that the number of ways to select the students is 45 -/
theorem select_students_count : select_students = 45 := by
  sorry

#eval select_students

end NUMINAMATH_CALUDE_select_students_count_l3054_305434


namespace NUMINAMATH_CALUDE_prop_a_prop_b_prop_c_prop_d_l3054_305493

-- Proposition A
theorem prop_a (a b : ℝ) (h : b > a ∧ a > 0) : 1 / a > 1 / b := by sorry

-- Proposition B
theorem prop_b : ∃ a b c : ℝ, a > b ∧ a * c ≤ b * c := by sorry

-- Proposition C
theorem prop_c (a b c : ℝ) (h1 : a * c^2 > b * c^2) (h2 : c ≠ 0) : a > b := by sorry

-- Proposition D
theorem prop_d : 
  (∃ x : ℝ, x > -3 ∧ x^2 ≤ 9) ↔ ¬(∀ x : ℝ, x > -3 → x^2 > 9) := by sorry

end NUMINAMATH_CALUDE_prop_a_prop_b_prop_c_prop_d_l3054_305493


namespace NUMINAMATH_CALUDE_ellipse_triangle_area_l3054_305487

-- Define the ellipse
def is_on_ellipse (x y : ℝ) : Prop := x^2 / 9 + y^2 / 4 = 1

-- Define the foci
def F₁ : ℝ × ℝ := sorry
def F₂ : ℝ × ℝ := sorry

-- Define a point on the ellipse
def P : ℝ × ℝ := sorry

-- Assume P is on the ellipse
axiom P_on_ellipse : is_on_ellipse P.1 P.2

-- Define the distances from P to the foci
def PF₁ : ℝ := sorry
def PF₂ : ℝ := sorry

-- Assume the ratio of PF₁ to PF₂ is 2:1
axiom distance_ratio : PF₁ = 2 * PF₂

-- Define the area of the triangle
def triangle_area : ℝ := sorry

-- State the theorem
theorem ellipse_triangle_area : triangle_area = 4 := sorry

end NUMINAMATH_CALUDE_ellipse_triangle_area_l3054_305487


namespace NUMINAMATH_CALUDE_football_water_cooler_problem_l3054_305473

/-- Represents the number of skill position players who must wait for a refill --/
def skillPlayersWaiting (coolerCapacity : ℕ) (numLinemen : ℕ) (numSkillPlayers : ℕ) 
  (linemenConsumption : ℕ) (skillPlayerConsumption : ℕ) : ℕ :=
  let waterLeftForSkillPlayers := coolerCapacity - numLinemen * linemenConsumption
  let skillPlayersThatCanDrink := waterLeftForSkillPlayers / skillPlayerConsumption
  numSkillPlayers - skillPlayersThatCanDrink

theorem football_water_cooler_problem :
  skillPlayersWaiting 126 12 10 8 6 = 5 := by
  sorry

end NUMINAMATH_CALUDE_football_water_cooler_problem_l3054_305473


namespace NUMINAMATH_CALUDE_exists_fixed_point_l3054_305491

def S : Set ℕ := {n : ℕ | 1 ≤ n ∧ n ≤ 999}

def is_fixed_point (f : S → S) (a : S) : Prop := f a = a

def satisfies_condition (f : S → S) : Prop :=
  ∀ n : S, (f^[n + f n + 1] n = n) ∧ (f^[n * f n] n = n)

theorem exists_fixed_point (f : S → S) (h : satisfies_condition f) :
  ∃ a : S, is_fixed_point f a := by
  sorry

end NUMINAMATH_CALUDE_exists_fixed_point_l3054_305491


namespace NUMINAMATH_CALUDE_arithmetic_mean_root_mean_square_inequality_l3054_305418

theorem arithmetic_mean_root_mean_square_inequality 
  (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a + b + c) / 3 ≤ Real.sqrt ((a^2 + b^2 + c^2) / 3) ∧
  ((a + b + c) / 3 = Real.sqrt ((a^2 + b^2 + c^2) / 3) ↔ a = b ∧ b = c) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_mean_root_mean_square_inequality_l3054_305418


namespace NUMINAMATH_CALUDE_combined_swimming_distance_is_1890_l3054_305403

/-- Calculates the combined swimming distance for Jamir, Sarah, and Julien over a week. -/
def combinedSwimmingDistance (julienDailyDistance : ℕ) (daysInWeek : ℕ) : ℕ :=
  let sarahDailyDistance := 2 * julienDailyDistance
  let jamirDailyDistance := sarahDailyDistance + 20
  (julienDailyDistance + sarahDailyDistance + jamirDailyDistance) * daysInWeek

/-- Proves that the combined swimming distance for Jamir, Sarah, and Julien over a week is 1890 meters. -/
theorem combined_swimming_distance_is_1890 :
  combinedSwimmingDistance 50 7 = 1890 := by
  sorry

end NUMINAMATH_CALUDE_combined_swimming_distance_is_1890_l3054_305403


namespace NUMINAMATH_CALUDE_value_of_a_l3054_305456

-- Define the conversion rate from paise to rupees
def paise_to_rupees (paise : ℚ) : ℚ := paise / 100

-- Define the condition
def condition (a : ℚ) : Prop := 0.5 / 100 * a = paise_to_rupees 85

-- Theorem statement
theorem value_of_a (a : ℚ) : condition a → a = 170 := by
  sorry

end NUMINAMATH_CALUDE_value_of_a_l3054_305456


namespace NUMINAMATH_CALUDE_coin_selection_probability_l3054_305448

/-- Represents the placement of boxes in drawers -/
inductive BoxPlacement
  | AloneInDrawer
  | WithOneOther
  | Random

/-- Probability of selecting the coin-containing box given a placement -/
def probability (placement : BoxPlacement) : ℚ :=
  match placement with
  | BoxPlacement.AloneInDrawer => 1/2
  | BoxPlacement.WithOneOther => 1/4
  | BoxPlacement.Random => 1/3

theorem coin_selection_probability 
  (boxes : Nat) 
  (drawers : Nat) 
  (coin_box : Nat) 
  (h1 : boxes = 3) 
  (h2 : drawers = 2) 
  (h3 : coin_box = 1) 
  (h4 : ∀ d, d ≤ drawers → d > 0 → ∃ b, b ≤ boxes ∧ b > 0) :
  (probability BoxPlacement.AloneInDrawer = 1/2) ∧
  (probability BoxPlacement.WithOneOther = 1/4) ∧
  (probability BoxPlacement.Random = 1/3) := by
  sorry

end NUMINAMATH_CALUDE_coin_selection_probability_l3054_305448


namespace NUMINAMATH_CALUDE_point_not_on_transformed_plane_l3054_305481

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a plane in 3D space -/
structure Plane3D where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Applies a similarity transformation to a plane -/
def transformPlane (p : Plane3D) (k : ℝ) : Plane3D :=
  { a := p.a, b := p.b, c := p.c, d := k * p.d }

/-- Checks if a point lies on a plane -/
def pointOnPlane (point : Point3D) (plane : Plane3D) : Prop :=
  plane.a * point.x + plane.b * point.y + plane.c * point.z + plane.d = 0

/-- The main theorem -/
theorem point_not_on_transformed_plane :
  let A : Point3D := { x := 5, y := 0, z := -1 }
  let a : Plane3D := { a := 2, b := -1, c := 3, d := -1 }
  let k : ℝ := 3
  let a' : Plane3D := transformPlane a k
  ¬ pointOnPlane A a' := by
  sorry

end NUMINAMATH_CALUDE_point_not_on_transformed_plane_l3054_305481


namespace NUMINAMATH_CALUDE_max_self_intersections_specific_cases_max_self_intersections_formula_l3054_305447

/-- Maximum number of self-intersection points for a closed polygonal chain -/
def max_self_intersections (n : ℕ) : ℕ :=
  if n % 2 = 1 then
    n * (n - 3) / 2
  else
    n * (n - 4) / 2 + 1

/-- Theorem stating the maximum number of self-intersection points for specific cases -/
theorem max_self_intersections_specific_cases :
  (max_self_intersections 13 = 65) ∧ (max_self_intersections 1950 = 1898851) := by
  sorry

/-- Theorem for the general formula of maximum self-intersection points -/
theorem max_self_intersections_formula (n : ℕ) (h : n > 2) :
  max_self_intersections n = 
    if n % 2 = 1 then
      n * (n - 3) / 2
    else
      n * (n - 4) / 2 + 1 := by
  sorry

end NUMINAMATH_CALUDE_max_self_intersections_specific_cases_max_self_intersections_formula_l3054_305447


namespace NUMINAMATH_CALUDE_intersection_of_lines_l3054_305449

theorem intersection_of_lines :
  ∃! (x y : ℚ), 5 * x - 3 * y = 7 ∧ 4 * x + 2 * y = 18 ∧ x = 34 / 11 ∧ y = 31 / 11 := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_lines_l3054_305449


namespace NUMINAMATH_CALUDE_parabola_intersection_value_l3054_305437

theorem parabola_intersection_value (m : ℝ) : m^2 - m - 1 = 0 → m^2 - m + 2017 = 2018 := by
  sorry

end NUMINAMATH_CALUDE_parabola_intersection_value_l3054_305437


namespace NUMINAMATH_CALUDE_percentage_saved_approx_l3054_305410

/-- Represents the discount information for each day of the sale -/
structure DayDiscount where
  minQuantity : Nat
  discountedQuantity : Nat
  discountedPrice : Nat

/-- Calculates the savings for a given day's discount -/
def calculateSavings (discount : DayDiscount) : Nat :=
  discount.minQuantity - discount.discountedPrice

/-- Calculates the total savings and original price for all days -/
def calculateTotals (discounts : List DayDiscount) : (Nat × Nat) :=
  let savings := discounts.map calculateSavings |>.sum
  let originalPrice := discounts.map (fun d => d.minQuantity) |>.sum
  (savings, originalPrice)

/-- The discounts for each day of the five-day sale -/
def saleDays : List DayDiscount := [
  { minQuantity := 11, discountedQuantity := 12, discountedPrice := 4 },
  { minQuantity := 15, discountedQuantity := 15, discountedPrice := 5 },
  { minQuantity := 18, discountedQuantity := 18, discountedPrice := 6 },
  { minQuantity := 21, discountedQuantity := 25, discountedPrice := 8 },
  { minQuantity := 26, discountedQuantity := 30, discountedPrice := 10 }
]

/-- Theorem stating that the percentage saved is approximately 63.74% -/
theorem percentage_saved_approx (ε : ℝ) (h : ε > 0) :
  let (savings, originalPrice) := calculateTotals saleDays
  let percentageSaved := (savings : ℝ) / (originalPrice : ℝ) * 100
  |percentageSaved - 63.74| < ε :=
sorry

end NUMINAMATH_CALUDE_percentage_saved_approx_l3054_305410


namespace NUMINAMATH_CALUDE_children_off_bus_l3054_305484

theorem children_off_bus (initial : ℕ) (remaining : ℕ) (h1 : initial = 43) (h2 : remaining = 21) :
  initial - remaining = 22 := by
  sorry

end NUMINAMATH_CALUDE_children_off_bus_l3054_305484


namespace NUMINAMATH_CALUDE_baymax_testing_system_l3054_305443

theorem baymax_testing_system (x y : ℕ) : 
  (200 * y = x + 18 ∧ 180 * y = x - 42) ↔ 
  (∀ (z : ℕ), z = 200 → z * y = x + 18) ∧ 
  (∀ (w : ℕ), w = 180 → w * y + 42 = x) :=
sorry

end NUMINAMATH_CALUDE_baymax_testing_system_l3054_305443


namespace NUMINAMATH_CALUDE_sqrt_81_div_3_l3054_305495

theorem sqrt_81_div_3 : Real.sqrt 81 / 3 = 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_81_div_3_l3054_305495


namespace NUMINAMATH_CALUDE_apple_baskets_l3054_305461

/-- 
Given two baskets A and B with apples, prove that:
1. If the total amount of apples in both baskets is 75 kg
2. And after transferring 5 kg from A to B, A has 7 kg more than B
Then the original amounts in A and B were 46 kg and 29 kg, respectively
-/
theorem apple_baskets (a b : ℕ) : 
  a + b = 75 → 
  (a - 5) = (b + 5) + 7 → 
  (a = 46 ∧ b = 29) := by
sorry

end NUMINAMATH_CALUDE_apple_baskets_l3054_305461


namespace NUMINAMATH_CALUDE_new_average_production_l3054_305469

/-- Given a company's production data, prove that the new average daily production is 45 units. -/
theorem new_average_production (n : ℕ) (past_average : ℝ) (today_production : ℝ) :
  n = 9 →
  past_average = 40 →
  today_production = 90 →
  (n * past_average + today_production) / (n + 1) = 45 := by
  sorry

end NUMINAMATH_CALUDE_new_average_production_l3054_305469


namespace NUMINAMATH_CALUDE_square_between_prime_sums_l3054_305420

/-- Sum of the first n prime numbers -/
def S (n : ℕ) : ℕ := sorry

/-- The nth prime number -/
def nthPrime (n : ℕ) : ℕ := sorry

theorem square_between_prime_sums :
  ∀ n : ℕ, n > 0 → ∃ k : ℕ, S n < k^2 ∧ k^2 < S (n + 1) :=
sorry

end NUMINAMATH_CALUDE_square_between_prime_sums_l3054_305420


namespace NUMINAMATH_CALUDE_largest_x_value_l3054_305432

theorem largest_x_value : ∃ x : ℝ,
  (15 * x^2 - 30 * x + 9) / (4 * x - 3) + 6 * x = 7 * x - 2 ∧
  x = (19 + Real.sqrt 229) / 22 ∧
  ∀ y : ℝ, (15 * y^2 - 30 * y + 9) / (4 * y - 3) + 6 * y = 7 * y - 2 → y ≤ x :=
by sorry

end NUMINAMATH_CALUDE_largest_x_value_l3054_305432


namespace NUMINAMATH_CALUDE_paint_project_cost_l3054_305408

/-- Calculates the total cost of paint and primer for a house painting project. -/
def total_cost (rooms : ℕ) (primer_cost : ℚ) (primer_discount : ℚ) (paint_cost : ℚ) : ℚ :=
  let discounted_primer_cost := primer_cost * (1 - primer_discount)
  let total_primer_cost := rooms * discounted_primer_cost
  let total_paint_cost := rooms * paint_cost
  total_primer_cost + total_paint_cost

/-- Proves that the total cost for paint and primer is $245.00 under given conditions. -/
theorem paint_project_cost :
  total_cost 5 30 (1/5) 25 = 245 :=
by sorry

end NUMINAMATH_CALUDE_paint_project_cost_l3054_305408


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l3054_305478

theorem sum_of_coefficients : 
  let p (x : ℝ) := (4 * x^2 - 4 * x + 3)^4 * (4 + 3 * x - 3 * x^2)^2
  (p 1) = 1296 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l3054_305478


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l3054_305474

theorem quadratic_inequality_range (a : ℝ) :
  (∀ x : ℝ, x^2 + 2*a*x + a > 0) → 0 < a ∧ a < 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l3054_305474


namespace NUMINAMATH_CALUDE_intersection_A_B_l3054_305401

-- Define set A
def A : Set ℝ := {x | (x - 1) * (x - 4) < 0}

-- Define set B
def B : Set ℝ := {x | ∃ y, y = 2 - x^2}

-- Theorem statement
theorem intersection_A_B : A ∩ B = {x : ℝ | 1 < x ∧ x ≤ 2} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_l3054_305401


namespace NUMINAMATH_CALUDE_polynomial_coefficient_sum_l3054_305442

theorem polynomial_coefficient_sum (a b c d : ℤ) : 
  (∀ x : ℝ, (x^2 + a*x + b) * (x^2 + c*x + d) = x^4 + 2*x^3 - 3*x^2 + 7*x - 6) →
  a + b + c + d = 7 := by
sorry

end NUMINAMATH_CALUDE_polynomial_coefficient_sum_l3054_305442


namespace NUMINAMATH_CALUDE_triangle_y_value_l3054_305409

-- Define the triangle
structure AcuteTriangle where
  a : ℝ
  y : ℝ
  area_small : ℝ

-- Define the properties of the triangle
def triangle_properties (t : AcuteTriangle) : Prop :=
  t.a > 0 ∧ t.y > 0 ∧
  6 > 0 ∧ 4 > 0 ∧
  t.area_small = 12 ∧
  (6 * (6 + t.y) = t.y * (10 + t.a)) ∧
  (1/2 * 10 * (24 / t.y) = 12)

-- Theorem statement
theorem triangle_y_value (t : AcuteTriangle) 
  (h : triangle_properties t) : t.y = 10 := by
  sorry

end NUMINAMATH_CALUDE_triangle_y_value_l3054_305409
