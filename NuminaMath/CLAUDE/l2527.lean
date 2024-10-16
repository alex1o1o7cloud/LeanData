import Mathlib

namespace NUMINAMATH_CALUDE_naval_formation_arrangements_l2527_252787

/-- The number of ways to arrange 2 submarines one in front of the other -/
def submarine_arrangements : ℕ := 2

/-- The number of ways to arrange 6 ships in two groups of 3 -/
def ship_arrangements : ℕ := 720

/-- The number of invalid arrangements where all ships on one side are of the same type -/
def invalid_arrangements : ℕ := 2 * 2

/-- The total number of valid arrangements -/
def total_arrangements : ℕ := submarine_arrangements * (ship_arrangements - invalid_arrangements)

theorem naval_formation_arrangements : total_arrangements = 1296 := by
  sorry

end NUMINAMATH_CALUDE_naval_formation_arrangements_l2527_252787


namespace NUMINAMATH_CALUDE_probability_opposite_rooms_is_one_fifth_l2527_252767

/-- Represents a hotel with 6 rooms -/
structure Hotel :=
  (rooms : Fin 6 → ℕ)
  (opposite : Fin 3 → Fin 2 → Fin 6)
  (opposite_bijective : ∀ i, Function.Bijective (opposite i))

/-- Represents the random selection of room keys by 6 people -/
def RoomSelection := Fin 6 → Fin 6

/-- The probability of two specific people selecting opposite rooms -/
def probability_opposite_rooms (h : Hotel) : ℚ :=
  1 / 5

/-- Theorem stating that the probability of two specific people
    selecting opposite rooms is 1/5 -/
theorem probability_opposite_rooms_is_one_fifth (h : Hotel) :
  probability_opposite_rooms h = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_probability_opposite_rooms_is_one_fifth_l2527_252767


namespace NUMINAMATH_CALUDE_middle_number_proof_l2527_252778

theorem middle_number_proof (a b c : ℕ) (h1 : a < b) (h2 : b < c) 
  (h3 : a + b = 15) (h4 : a + c = 20) (h5 : b + c = 23) (h6 : c = 2 * a) : 
  b = 25 / 3 := by
  sorry

end NUMINAMATH_CALUDE_middle_number_proof_l2527_252778


namespace NUMINAMATH_CALUDE_factorization_of_2m_squared_minus_2_l2527_252764

theorem factorization_of_2m_squared_minus_2 (m : ℝ) : 2 * m^2 - 2 = 2 * (m + 1) * (m - 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_of_2m_squared_minus_2_l2527_252764


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_l2527_252701

theorem sum_of_reciprocals (a b c : ℝ) 
  (sum_condition : a + b + c = 6)
  (sum_squares_condition : a^2 + b^2 + c^2 = 10)
  (sum_cubes_condition : a^3 + b^3 + c^3 = 36) :
  1/a + 1/b + 1/c = 13/18 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_l2527_252701


namespace NUMINAMATH_CALUDE_june_design_white_tiles_l2527_252703

/-- Calculates the number of white tiles in June's design -/
theorem june_design_white_tiles :
  let total_tiles : ℕ := 20
  let yellow_tiles : ℕ := 3
  let blue_tiles : ℕ := yellow_tiles + 1
  let purple_tiles : ℕ := 6
  let colored_tiles : ℕ := yellow_tiles + blue_tiles + purple_tiles
  let white_tiles : ℕ := total_tiles - colored_tiles
  white_tiles = 7 := by
  sorry

end NUMINAMATH_CALUDE_june_design_white_tiles_l2527_252703


namespace NUMINAMATH_CALUDE_vanessa_deleted_files_l2527_252776

/-- Calculates the number of deleted files given the initial number of music files,
    initial number of video files, and the number of remaining files. -/
def deleted_files (initial_music : ℕ) (initial_video : ℕ) (remaining : ℕ) : ℕ :=
  initial_music + initial_video - remaining

/-- Theorem stating that Vanessa deleted 30 files from her flash drive. -/
theorem vanessa_deleted_files :
  deleted_files 16 48 34 = 30 := by
  sorry

end NUMINAMATH_CALUDE_vanessa_deleted_files_l2527_252776


namespace NUMINAMATH_CALUDE_square_area_with_four_circles_l2527_252723

theorem square_area_with_four_circles (r : ℝ) (h : r = 7) : 
  let side_length := 2 * (2 * r)
  (side_length ^ 2 : ℝ) = 784 := by
  sorry

end NUMINAMATH_CALUDE_square_area_with_four_circles_l2527_252723


namespace NUMINAMATH_CALUDE_ellipse_condition_l2527_252733

/-- The equation of the curve -/
def curve_equation (x y k : ℝ) : Prop :=
  9 * x^2 + y^2 - 18 * x - 2 * y = k

/-- Definition of a non-degenerate ellipse -/
def is_non_degenerate_ellipse (k : ℝ) : Prop :=
  ∃ a b c d e : ℝ, a > 0 ∧ b > 0 ∧ (a ≠ b ∨ c ≠ 0 ∨ d ≠ 0) ∧
    ∀ x y : ℝ, curve_equation x y k ↔ a * (x - c)^2 + b * (y - d)^2 = e

/-- The main theorem -/
theorem ellipse_condition (k : ℝ) :
  is_non_degenerate_ellipse k ↔ k > -10 :=
sorry

end NUMINAMATH_CALUDE_ellipse_condition_l2527_252733


namespace NUMINAMATH_CALUDE_negation_of_existence_proposition_l2527_252706

theorem negation_of_existence_proposition :
  (¬ ∃ c : ℝ, c > 0 ∧ ∃ x : ℝ, x^2 - x + c = 0) ↔
  (∀ c : ℝ, c > 0 → ∀ x : ℝ, x^2 - x + c ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existence_proposition_l2527_252706


namespace NUMINAMATH_CALUDE_kendras_change_l2527_252714

/-- Calculates the change received after a purchase -/
def calculate_change (toy_price hat_price : ℕ) (num_toys num_hats : ℕ) (total_money : ℕ) : ℕ :=
  total_money - (toy_price * num_toys + hat_price * num_hats)

/-- Proves that Kendra's change is $30 -/
theorem kendras_change :
  let toy_price : ℕ := 20
  let hat_price : ℕ := 10
  let num_toys : ℕ := 2
  let num_hats : ℕ := 3
  let total_money : ℕ := 100
  calculate_change toy_price hat_price num_toys num_hats total_money = 30 := by
  sorry

#eval calculate_change 20 10 2 3 100

end NUMINAMATH_CALUDE_kendras_change_l2527_252714


namespace NUMINAMATH_CALUDE_last_digit_2_pow_2004_l2527_252769

/-- The last digit of 2^n -/
def lastDigitPow2 (n : ℕ) : ℕ :=
  (2^n) % 10

/-- The sequence of last digits for powers of 2 from 2^1 to 2^8 -/
def lastDigitSequence : List ℕ := [2, 4, 8, 6, 2, 4, 8, 6]

theorem last_digit_2_pow_2004 :
  lastDigitPow2 2004 = 6 :=
sorry

end NUMINAMATH_CALUDE_last_digit_2_pow_2004_l2527_252769


namespace NUMINAMATH_CALUDE_complex_pure_imaginary_l2527_252794

theorem complex_pure_imaginary (a : ℝ) : 
  (Complex.I * (a - 2 * Complex.I) + (2 : ℂ) * (a - 2 * Complex.I)).re = 0 → a = -1 :=
by sorry

end NUMINAMATH_CALUDE_complex_pure_imaginary_l2527_252794


namespace NUMINAMATH_CALUDE_gcd_of_4557_1953_5115_l2527_252751

theorem gcd_of_4557_1953_5115 : Nat.gcd 4557 (Nat.gcd 1953 5115) = 93 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_4557_1953_5115_l2527_252751


namespace NUMINAMATH_CALUDE_domain_of_f_l2527_252741

theorem domain_of_f (m : ℝ) : 
  (∀ x : ℝ, (m^2 - 3*m + 2)*x^2 + (m - 1)*x + 1 > 0) ↔ (m > 7/3 ∨ m ≤ 1) :=
by sorry

end NUMINAMATH_CALUDE_domain_of_f_l2527_252741


namespace NUMINAMATH_CALUDE_sarah_trucks_l2527_252783

-- Define the initial number of trucks Sarah had
def initial_trucks : ℕ := 51

-- Define the number of trucks Sarah gave away
def trucks_given_away : ℕ := 13

-- Define the number of trucks Sarah has left
def trucks_left : ℕ := 38

-- Theorem statement
theorem sarah_trucks : 
  initial_trucks = trucks_given_away + trucks_left :=
by sorry

end NUMINAMATH_CALUDE_sarah_trucks_l2527_252783


namespace NUMINAMATH_CALUDE_arrangement_count_is_factorial_squared_l2527_252738

/-- The number of ways to arrange 5 different objects in a 5x5 grid,
    such that each row and each column contains exactly one object. -/
def arrangement_count : ℕ := (5 : ℕ).factorial ^ 2

/-- Theorem stating that the number of arrangements is equal to (5!)^2 -/
theorem arrangement_count_is_factorial_squared :
  arrangement_count = 14400 := by sorry

end NUMINAMATH_CALUDE_arrangement_count_is_factorial_squared_l2527_252738


namespace NUMINAMATH_CALUDE_max_value_theorem_l2527_252758

theorem max_value_theorem (a b : ℝ) (h1 : a ≥ 0) (h2 : b ≥ 0) (h3 : a^2 + b^2/2 = 1) :
  ∃ (M : ℝ), M = (3 * Real.sqrt 2) / 4 ∧ a * Real.sqrt (1 + b^2) ≤ M ∧
  ∃ (a₀ b₀ : ℝ), a₀ * Real.sqrt (1 + b₀^2) = M :=
sorry

end NUMINAMATH_CALUDE_max_value_theorem_l2527_252758


namespace NUMINAMATH_CALUDE_sum_of_digits_of_1996_digit_multiple_of_9_l2527_252752

/-- A function that returns the sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- A function that checks if a number is a 1996-digit integer -/
def is1996Digit (n : ℕ) : Prop := sorry

theorem sum_of_digits_of_1996_digit_multiple_of_9 (n : ℕ) 
  (h1 : is1996Digit n) 
  (h2 : n % 9 = 0) : 
  let p := sumOfDigits n
  let q := sumOfDigits p
  let r := sumOfDigits q
  r = 9 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_of_1996_digit_multiple_of_9_l2527_252752


namespace NUMINAMATH_CALUDE_arithmetic_sequence_properties_l2527_252792

/-- An arithmetic sequence with specific terms -/
structure ArithmeticSequence where
  a : ℕ → ℤ
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  a7_eq : a 7 = -2
  a20_eq : a 20 = -28

/-- The general term of the arithmetic sequence -/
def generalTerm (seq : ArithmeticSequence) (n : ℕ) : ℤ :=
  14 - 2 * n

/-- The sum of the first n terms of the arithmetic sequence -/
def sumFirstN (seq : ArithmeticSequence) (n : ℕ) : ℤ :=
  n * (seq.a 1 + seq.a n) / 2

theorem arithmetic_sequence_properties (seq : ArithmeticSequence) :
  (∀ n, seq.a n = generalTerm seq n) ∧
  (∃ n, sumFirstN seq n = 42 ∧ ∀ m, sumFirstN seq m ≤ 42) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_properties_l2527_252792


namespace NUMINAMATH_CALUDE_bird_feeder_problem_l2527_252782

theorem bird_feeder_problem (feeder_capacity : ℝ) (birds_per_cup : ℝ) (stolen_amount : ℝ) :
  feeder_capacity = 2 ∧ birds_per_cup = 14 ∧ stolen_amount = 0.5 →
  (feeder_capacity - stolen_amount) * birds_per_cup = 21 := by
  sorry

end NUMINAMATH_CALUDE_bird_feeder_problem_l2527_252782


namespace NUMINAMATH_CALUDE_two_fixed_points_l2527_252762

/-- A function satisfying the given property -/
def satisfies_property (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x + y) = f x + f y + x * y + 1

/-- The main theorem -/
theorem two_fixed_points
  (f : ℝ → ℝ)
  (h1 : satisfies_property f)
  (h2 : f (-2) = -2) :
  ∃! (s : Finset ℤ), s.card = 2 ∧ ∀ a : ℤ, a ∈ s ↔ f a = a :=
sorry

end NUMINAMATH_CALUDE_two_fixed_points_l2527_252762


namespace NUMINAMATH_CALUDE_ratio_equality_solution_l2527_252729

theorem ratio_equality_solution : 
  ∃! x : ℝ, (3 * x + 1) / (5 * x + 2) = (6 * x + 4) / (10 * x + 7) :=
by
  sorry

end NUMINAMATH_CALUDE_ratio_equality_solution_l2527_252729


namespace NUMINAMATH_CALUDE_specific_trapezoid_area_l2527_252704

/-- Represents a trapezoid with an inscribed circle -/
structure InscribedCircleTrapezoid where
  -- Length of the top base
  AB : ℝ
  -- Distance from point D to point N on the bottom base
  DN : ℝ
  -- Radius of the inscribed circle
  r : ℝ

/-- The area of a trapezoid with an inscribed circle -/
def trapezoidArea (t : InscribedCircleTrapezoid) : ℝ :=
  sorry

/-- Theorem stating that the area of the specific trapezoid is 27 -/
theorem specific_trapezoid_area :
  ∀ (t : InscribedCircleTrapezoid),
    t.AB = 12 ∧ t.DN = 1 ∧ t.r = 2 →
    trapezoidArea t = 27 :=
  sorry

end NUMINAMATH_CALUDE_specific_trapezoid_area_l2527_252704


namespace NUMINAMATH_CALUDE_smallest_covering_triangular_number_l2527_252771

def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

def all_remainders_covered (m : ℕ) : Prop :=
  ∀ r : Fin 7, ∃ k : ℕ, k ≤ m ∧ triangular_number k % 7 = r.val

theorem smallest_covering_triangular_number :
  (all_remainders_covered 10) ∧
  (∀ n < 10, ¬ all_remainders_covered n) :=
sorry

end NUMINAMATH_CALUDE_smallest_covering_triangular_number_l2527_252771


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l2527_252791

theorem arithmetic_calculation : 4 * (8 - 3) - 7 = 13 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l2527_252791


namespace NUMINAMATH_CALUDE_exam_boys_count_total_boys_is_120_l2527_252705

/-- The number of boys who passed the examination -/
def passed_boys : ℕ := 100

/-- The average marks of all boys -/
def total_average : ℚ := 35

/-- The average marks of passed boys -/
def passed_average : ℚ := 39

/-- The average marks of failed boys -/
def failed_average : ℚ := 15

/-- The total number of boys who took the examination -/
def total_boys : ℕ := sorry

theorem exam_boys_count :
  total_boys = passed_boys +
    (total_boys * total_average - passed_boys * passed_average) / (failed_average - total_average) :=
by sorry

theorem total_boys_is_120 : total_boys = 120 :=
by sorry

end NUMINAMATH_CALUDE_exam_boys_count_total_boys_is_120_l2527_252705


namespace NUMINAMATH_CALUDE_f_zero_at_three_l2527_252724

/-- The function f(x) = 3x^3 + 2x^2 - 5x + s -/
def f (s : ℝ) (x : ℝ) : ℝ := 3 * x^3 + 2 * x^2 - 5 * x + s

/-- Theorem: f(3) = 0 if and only if s = -84 -/
theorem f_zero_at_three (s : ℝ) : f s 3 = 0 ↔ s = -84 := by
  sorry

end NUMINAMATH_CALUDE_f_zero_at_three_l2527_252724


namespace NUMINAMATH_CALUDE_prime_square_plus_two_prime_l2527_252774

theorem prime_square_plus_two_prime (P : ℕ) (h1 : Nat.Prime P) (h2 : Nat.Prime (P^2 + 2)) :
  P^4 + 1921 = 2002 := by
sorry

end NUMINAMATH_CALUDE_prime_square_plus_two_prime_l2527_252774


namespace NUMINAMATH_CALUDE_sum_of_absolute_coefficients_l2527_252711

theorem sum_of_absolute_coefficients (a a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ : ℝ) :
  (∀ x : ℝ, (1 - 3*x)^9 = a + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6 + a₇*x^7 + a₈*x^8 + a₉*x^9) →
  |a| + |a₁| + |a₂| + |a₃| + |a₄| + |a₅| + |a₆| + |a₇| + |a₈| + |a₉| = 4^9 :=
by
  sorry

end NUMINAMATH_CALUDE_sum_of_absolute_coefficients_l2527_252711


namespace NUMINAMATH_CALUDE_inequality_transformation_l2527_252710

theorem inequality_transformation (a b : ℝ) (h : a > b) : 2 * a + 1 > 2 * b + 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_transformation_l2527_252710


namespace NUMINAMATH_CALUDE_parakeet_cost_graph_is_finite_distinct_points_l2527_252742

def parakeet_cost (n : ℕ) : ℚ :=
  if n ≤ 10 then 20 * n
  else if n ≤ 20 then 18 * n
  else if n ≤ 25 then 18 * n
  else 0

def cost_graph : Set (ℕ × ℚ) :=
  {p | ∃ n : ℕ, 1 ≤ n ∧ n ≤ 25 ∧ p = (n, parakeet_cost n)}

theorem parakeet_cost_graph_is_finite_distinct_points :
  Finite cost_graph ∧ ∀ p q : ℕ × ℚ, p ∈ cost_graph → q ∈ cost_graph → p ≠ q → p.1 ≠ q.1 :=
sorry

end NUMINAMATH_CALUDE_parakeet_cost_graph_is_finite_distinct_points_l2527_252742


namespace NUMINAMATH_CALUDE_jordan_no_quiz_probability_l2527_252736

theorem jordan_no_quiz_probability (p_quiz : ℚ) (h : p_quiz = 5/9) :
  1 - p_quiz = 4/9 := by
sorry

end NUMINAMATH_CALUDE_jordan_no_quiz_probability_l2527_252736


namespace NUMINAMATH_CALUDE_min_value_product_l2527_252716

theorem min_value_product (a b c x y z : ℝ) 
  (non_neg : a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0)
  (sum_abc : a + b + c = 1)
  (sum_xyz : x + y + z = 1) :
  (a - x^2) * (b - y^2) * (c - z^2) ≥ -1/4 :=
sorry

end NUMINAMATH_CALUDE_min_value_product_l2527_252716


namespace NUMINAMATH_CALUDE_range_of_a_given_quadratic_inequality_l2527_252739

theorem range_of_a_given_quadratic_inequality (a : ℝ) : 
  (∀ x : ℝ, 4 * x^2 + (a - 2) * x + (1/4 : ℝ) > 0) → 
  0 < a ∧ a < 4 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_given_quadratic_inequality_l2527_252739


namespace NUMINAMATH_CALUDE_square_fraction_above_line_l2527_252735

-- Define the square
def square_vertices : List (ℝ × ℝ) := [(4, 1), (7, 1), (7, 4), (4, 4)]

-- Define the line passing through two points
def line_points : List (ℝ × ℝ) := [(4, 3), (7, 1)]

-- Function to calculate the fraction of square area above the line
def fraction_above_line (square : List (ℝ × ℝ)) (line : List (ℝ × ℝ)) : ℚ :=
  sorry

-- Theorem statement
theorem square_fraction_above_line :
  fraction_above_line square_vertices line_points = 1/2 :=
sorry

end NUMINAMATH_CALUDE_square_fraction_above_line_l2527_252735


namespace NUMINAMATH_CALUDE_product_of_numbers_with_given_sum_and_difference_l2527_252745

theorem product_of_numbers_with_given_sum_and_difference :
  ∀ x y : ℝ, x + y = 25 ∧ x - y = 3 → x * y = 154 := by
  sorry

end NUMINAMATH_CALUDE_product_of_numbers_with_given_sum_and_difference_l2527_252745


namespace NUMINAMATH_CALUDE_range_of_a_l2527_252760

def f (a : ℝ) (x : ℝ) : ℝ := (x - 2)^2 * |x - a|

theorem range_of_a (a : ℝ) :
  (∀ x ∈ Set.Icc 2 4, x * (deriv (f a) x) ≥ 0) ↔ a ∈ Set.Iic 2 ∪ Set.Ici 5 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l2527_252760


namespace NUMINAMATH_CALUDE_x_power_four_minus_reciprocal_l2527_252709

theorem x_power_four_minus_reciprocal (x : ℝ) (h : x - 1/x = 5) : x^4 - 1/x^4 = 727 := by
  sorry

end NUMINAMATH_CALUDE_x_power_four_minus_reciprocal_l2527_252709


namespace NUMINAMATH_CALUDE_equation_satisfied_l2527_252781

theorem equation_satisfied (a b c : ℤ) (h1 : a = c - 1) (h2 : b = a - 1) :
  a * (a - b) + b * (b - c) + c * (c - a) = 1 := by
  sorry

end NUMINAMATH_CALUDE_equation_satisfied_l2527_252781


namespace NUMINAMATH_CALUDE_triangle_special_condition_right_angle_l2527_252795

/-- Given a triangle ABC, if b cos C + c cos B = a sin A, then angle A is 90° -/
theorem triangle_special_condition_right_angle 
  (A B C : ℝ) (a b c : ℝ) : 
  0 < A ∧ A < π ∧ 
  0 < B ∧ B < π ∧ 
  0 < C ∧ C < π ∧ 
  A + B + C = π ∧ 
  a > 0 ∧ b > 0 ∧ c > 0 ∧
  b * Real.cos C + c * Real.cos B = a * Real.sin A → 
  A = π / 2 := by
sorry

end NUMINAMATH_CALUDE_triangle_special_condition_right_angle_l2527_252795


namespace NUMINAMATH_CALUDE_polynomial_degree_three_l2527_252737

def f (x : ℝ) : ℝ := 2 - 15*x + 4*x^2 - 5*x^3 + 7*x^4
def g (x : ℝ) : ℝ := 4 - 3*x - 8*x^3 + 12*x^4

def c : ℚ := -7/12

theorem polynomial_degree_three :
  ∃ (a b d : ℝ), ∀ (x : ℝ),
    f x + c * g x = a*x^3 + b*x^2 + d*x + (2 + 4*c) ∧ a ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_degree_three_l2527_252737


namespace NUMINAMATH_CALUDE_triangle_altitude_length_l2527_252765

/-- Given a rectangle with sides a and b, and a triangle with its base as the diagonal of the rectangle
    and area twice that of the rectangle, the length of the altitude of the triangle to its base
    (the diagonal) is (4ab)/√(a² + b²). -/
theorem triangle_altitude_length (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let rectangle_area := a * b
  let diagonal := Real.sqrt (a^2 + b^2)
  let triangle_area := 2 * rectangle_area
  let altitude := (2 * triangle_area) / diagonal
  altitude = (4 * a * b) / Real.sqrt (a^2 + b^2) := by
  sorry

end NUMINAMATH_CALUDE_triangle_altitude_length_l2527_252765


namespace NUMINAMATH_CALUDE_john_driving_distance_john_driving_distance_proof_l2527_252719

theorem john_driving_distance : ℝ → Prop :=
  fun total_distance =>
    let speed1 : ℝ := 45
    let time1 : ℝ := 2
    let speed2 : ℝ := 50
    let time2 : ℝ := 3
    let distance1 := speed1 * time1
    let distance2 := speed2 * time2
    total_distance = distance1 + distance2 ∧ total_distance = 240

-- Proof
theorem john_driving_distance_proof : ∃ d : ℝ, john_driving_distance d := by
  sorry

end NUMINAMATH_CALUDE_john_driving_distance_john_driving_distance_proof_l2527_252719


namespace NUMINAMATH_CALUDE_sum_of_first_20_odd_integers_greater_than_10_l2527_252777

/-- The sum of the first n terms of an arithmetic sequence -/
def arithmetic_sum (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a₁ + (n - 1) * d) / 2

/-- The 20th term of the arithmetic sequence starting at 11 with common difference 2 -/
def a₂₀ : ℕ := 11 + 19 * 2

theorem sum_of_first_20_odd_integers_greater_than_10 :
  arithmetic_sum 11 2 20 = 600 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_first_20_odd_integers_greater_than_10_l2527_252777


namespace NUMINAMATH_CALUDE_farm_pets_after_changes_l2527_252707

/-- Calculates the total number of pets after changes to a farm's pet population -/
theorem farm_pets_after_changes 
  (initial_dogs : ℕ) 
  (initial_fish : ℕ) 
  (initial_cats : ℕ) 
  (dogs_left : ℕ) 
  (rabbits_added : ℕ) 
  (h_initial_dogs : initial_dogs = 43)
  (h_initial_fish : initial_fish = 72)
  (h_initial_cats : initial_cats = 34)
  (h_dogs_left : dogs_left = 5)
  (h_rabbits_added : rabbits_added = 10) :
  initial_dogs - dogs_left + 2 * initial_fish + initial_cats + rabbits_added = 226 := by
  sorry

end NUMINAMATH_CALUDE_farm_pets_after_changes_l2527_252707


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2527_252757

theorem complex_equation_solution (z : ℂ) : (1 - Complex.I) * z = 2 → z = 1 + Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2527_252757


namespace NUMINAMATH_CALUDE_product_comparison_l2527_252748

theorem product_comparison (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (1.1 * a) * (1.13 * b) * (0.8 * c) < a * b * c := by
  sorry

end NUMINAMATH_CALUDE_product_comparison_l2527_252748


namespace NUMINAMATH_CALUDE_rita_dress_count_l2527_252798

def initial_amount : ℕ := 400
def final_amount : ℕ := 139
def pants_count : ℕ := 3
def jackets_count : ℕ := 4
def dress_price : ℕ := 20
def pants_price : ℕ := 12
def jacket_price : ℕ := 30
def transportation_cost : ℕ := 5

theorem rita_dress_count :
  let total_spent := initial_amount - final_amount
  let pants_jackets_cost := pants_count * pants_price + jackets_count * jacket_price
  let dress_total_cost := total_spent - pants_jackets_cost - transportation_cost
  dress_total_cost / dress_price = 5 := by sorry

end NUMINAMATH_CALUDE_rita_dress_count_l2527_252798


namespace NUMINAMATH_CALUDE_book_pages_count_book_pages_count_proof_l2527_252713

theorem book_pages_count : ℕ → ℕ → ℕ → ℕ → Prop :=
  fun (days : ℕ) (avg_first_three : ℕ) (avg_next_three : ℕ) (last_day : ℕ) =>
    days = 7 →
    avg_first_three = 42 →
    avg_next_three = 39 →
    last_day = 28 →
    3 * avg_first_three + 3 * avg_next_three + last_day = 271

-- The proof is omitted
theorem book_pages_count_proof : book_pages_count 7 42 39 28 := by sorry

end NUMINAMATH_CALUDE_book_pages_count_book_pages_count_proof_l2527_252713


namespace NUMINAMATH_CALUDE_arithmetic_sequence_properties_l2527_252780

/-- An arithmetic sequence and its partial sum sequence -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The arithmetic sequence
  d : ℝ      -- Common difference
  S : ℕ → ℝ  -- Partial sum sequence
  h1 : d ≠ 0
  h2 : ∀ n, a (n + 1) = a n + d
  h3 : ∀ n, S n = (n : ℝ) * (2 * a 1 + (n - 1) * d) / 2

theorem arithmetic_sequence_properties (seq : ArithmeticSequence) :
  (seq.d < 0 → ∃ M, ∀ n, seq.S n ≤ M) ∧
  ((∃ M, ∀ n, seq.S n ≤ M) → seq.d < 0) ∧
  (∃ seq : ArithmeticSequence, (∀ n, seq.S (n + 1) > seq.S n) ∧ ∃ k, seq.S k ≤ 0) ∧
  ((∀ n, seq.S n > 0) → ∀ n, seq.S (n + 1) > seq.S n) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_properties_l2527_252780


namespace NUMINAMATH_CALUDE_remainder_n_cubed_plus_three_l2527_252770

theorem remainder_n_cubed_plus_three (n : ℕ) (h : n > 2) : 
  (n^3 + 3) % (n + 1) = 2 := by
sorry

end NUMINAMATH_CALUDE_remainder_n_cubed_plus_three_l2527_252770


namespace NUMINAMATH_CALUDE_inequality_holds_iff_p_in_interval_l2527_252797

theorem inequality_holds_iff_p_in_interval (p q : ℝ) :
  q > 0 →
  2*p + q ≠ 0 →
  (4*(2*p*q^2 + p^2*q + 4*q^2 + 4*p*q) / (2*p + q) > 3*p^2*q) ↔
  (0 ≤ p ∧ p < 4) :=
by sorry

end NUMINAMATH_CALUDE_inequality_holds_iff_p_in_interval_l2527_252797


namespace NUMINAMATH_CALUDE_isosceles_right_triangle_angle_l2527_252744

/-- An isosceles right triangle has two equal angles and one right angle (90°) -/
structure IsoscelesRightTriangle where
  angle1 : ℝ
  angle2 : ℝ
  angle3 : ℝ
  is_isosceles : angle1 = angle2
  is_right : angle3 = 90
  sum_of_angles : angle1 + angle2 + angle3 = 180

/-- In an isosceles right triangle, if one of the angles is x°, then x = 45° -/
theorem isosceles_right_triangle_angle (t : IsoscelesRightTriangle) (x : ℝ) 
  (h : x = t.angle1 ∨ x = t.angle2 ∨ x = t.angle3) : x = 45 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_right_triangle_angle_l2527_252744


namespace NUMINAMATH_CALUDE_honey_tax_calculation_l2527_252788

/-- Represents the tax per pound of honey -/
def tax_per_pound : ℝ := 1

theorem honey_tax_calculation 
  (bulk_price : ℝ) 
  (minimum_spend : ℝ) 
  (total_paid : ℝ) 
  (excess_pounds : ℝ) 
  (h1 : bulk_price = 5)
  (h2 : minimum_spend = 40)
  (h3 : total_paid = 240)
  (h4 : excess_pounds = 32)
  : tax_per_pound = 1 := by
  sorry

#check honey_tax_calculation

end NUMINAMATH_CALUDE_honey_tax_calculation_l2527_252788


namespace NUMINAMATH_CALUDE_round_trip_average_speed_l2527_252755

/-- Calculates the average speed for a round trip given uphill and downhill times and distances -/
theorem round_trip_average_speed
  (uphill_distance : ℝ)
  (uphill_time : ℝ)
  (downhill_distance : ℝ)
  (downhill_time : ℝ)
  (h1 : uphill_distance = 2)
  (h2 : uphill_time = 45 / 60)
  (h3 : downhill_distance = 2)
  (h4 : downhill_time = 15 / 60)
  : (uphill_distance + downhill_distance) / (uphill_time + downhill_time) = 4 := by
  sorry

end NUMINAMATH_CALUDE_round_trip_average_speed_l2527_252755


namespace NUMINAMATH_CALUDE_two_by_six_grid_triangles_l2527_252726

/-- Represents a rectangular grid with diagonal lines --/
structure DiagonalGrid :=
  (rows : ℕ)
  (cols : ℕ)
  (has_center_diagonals : Bool)

/-- Counts the number of triangles in a diagonal grid --/
def count_triangles (grid : DiagonalGrid) : ℕ :=
  sorry

/-- Theorem stating that a 2x6 grid with center diagonals has at least 88 triangles --/
theorem two_by_six_grid_triangles :
  ∀ (grid : DiagonalGrid),
    grid.rows = 2 ∧ 
    grid.cols = 6 ∧ 
    grid.has_center_diagonals = true →
    count_triangles grid ≥ 88 :=
by sorry

end NUMINAMATH_CALUDE_two_by_six_grid_triangles_l2527_252726


namespace NUMINAMATH_CALUDE_root_equation_result_l2527_252768

theorem root_equation_result (α β : ℝ) : 
  α^2 - 4*α - 5 = 0 → β^2 - 4*β - 5 = 0 → 3*α^4 + 10*β^3 = 2593 := by
  sorry

end NUMINAMATH_CALUDE_root_equation_result_l2527_252768


namespace NUMINAMATH_CALUDE_f_of_2_eq_0_l2527_252750

def f (x : ℝ) : ℝ := (x - 1)^2 - (x - 1)

theorem f_of_2_eq_0 : f 2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_f_of_2_eq_0_l2527_252750


namespace NUMINAMATH_CALUDE_trigonometric_identity_l2527_252793

theorem trigonometric_identity (x y : ℝ) : 
  Real.sin (x + y) * Real.sin x + Real.cos (x + y) * Real.cos x = Real.cos y := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l2527_252793


namespace NUMINAMATH_CALUDE_jelly_bean_probability_l2527_252749

theorem jelly_bean_probability (p_red p_orange p_green : ℝ) 
  (h_red : p_red = 0.1)
  (h_orange : p_orange = 0.4)
  (h_green : p_green = 0.2)
  (h_sum : p_red + p_orange + p_green + p_yellow = 1)
  (h_nonneg : p_yellow ≥ 0) :
  p_yellow = 0.3 := by
sorry

end NUMINAMATH_CALUDE_jelly_bean_probability_l2527_252749


namespace NUMINAMATH_CALUDE_largest_difference_l2527_252753

def A : ℕ := 3 * 2003^2002
def B : ℕ := 2003^2002
def C : ℕ := 2002 * 2003^2001
def D : ℕ := 3 * 2003^2001
def E : ℕ := 2003^2001
def F : ℕ := 2003^2000

theorem largest_difference (A B C D E F : ℕ) 
  (hA : A = 3 * 2003^2002)
  (hB : B = 2003^2002)
  (hC : C = 2002 * 2003^2001)
  (hD : D = 3 * 2003^2001)
  (hE : E = 2003^2001)
  (hF : F = 2003^2000) :
  (A - B > B - C) ∧ 
  (A - B > C - D) ∧ 
  (A - B > D - E) ∧ 
  (A - B > E - F) :=
by sorry

end NUMINAMATH_CALUDE_largest_difference_l2527_252753


namespace NUMINAMATH_CALUDE_sum_base3_equals_100212_l2527_252743

/-- Converts a base-3 number represented as a list of digits to its decimal equivalent -/
def base3ToDecimal (digits : List Nat) : Nat :=
  digits.foldl (fun acc d => 3 * acc + d) 0

/-- Converts a decimal number to its base-3 representation as a list of digits -/
def decimalToBase3 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec aux (m : Nat) (acc : List Nat) : List Nat :=
      if m = 0 then acc else aux (m / 3) ((m % 3) :: acc)
    aux n []

theorem sum_base3_equals_100212 :
  let a := base3ToDecimal [1]
  let b := base3ToDecimal [1, 0, 2]
  let c := base3ToDecimal [2, 0, 2, 1]
  let d := base3ToDecimal [1, 1, 0, 1, 2]
  let e := base3ToDecimal [2, 2, 1, 1, 1]
  decimalToBase3 (a + b + c + d + e) = [1, 0, 0, 2, 1, 2] := by
  sorry

end NUMINAMATH_CALUDE_sum_base3_equals_100212_l2527_252743


namespace NUMINAMATH_CALUDE_collinear_points_imply_a_eq_two_l2527_252799

/-- Three points are collinear if the slope between any two pairs of points is the same. -/
def collinear (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) : Prop :=
  (y₂ - y₁) * (x₃ - x₁) = (y₃ - y₁) * (x₂ - x₁)

/-- Given three points A(a,2), B(5,1), and C(-4,2a) are collinear, prove that a = 2. -/
theorem collinear_points_imply_a_eq_two (a : ℝ) :
  collinear a 2 5 1 (-4) (2*a) → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_collinear_points_imply_a_eq_two_l2527_252799


namespace NUMINAMATH_CALUDE_birds_joining_fence_l2527_252786

theorem birds_joining_fence (initial_birds : ℕ) (total_birds : ℕ) (joined_birds : ℕ) : 
  initial_birds = 1 → total_birds = 5 → joined_birds = total_birds - initial_birds → joined_birds = 4 := by
  sorry

end NUMINAMATH_CALUDE_birds_joining_fence_l2527_252786


namespace NUMINAMATH_CALUDE_power_sum_equality_l2527_252761

theorem power_sum_equality : (-1)^53 + 3^(2^3 + 5^2 - 7^2) = -43046720 / 43046721 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_equality_l2527_252761


namespace NUMINAMATH_CALUDE_arithmetic_sum_2_to_20_l2527_252715

def arithmetic_sum (a₁ : ℕ) (aₙ : ℕ) (d : ℕ) : ℕ :=
  let n := (aₙ - a₁) / d + 1
  (a₁ + aₙ) * n / 2

theorem arithmetic_sum_2_to_20 :
  arithmetic_sum 2 20 2 = 110 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sum_2_to_20_l2527_252715


namespace NUMINAMATH_CALUDE_envelope_area_l2527_252727

/-- The area of a rectangular envelope with width and height both 6 inches is 36 square inches. -/
theorem envelope_area (width height : ℝ) (h1 : width = 6) (h2 : height = 6) :
  width * height = 36 := by
  sorry

end NUMINAMATH_CALUDE_envelope_area_l2527_252727


namespace NUMINAMATH_CALUDE_sequence_inequality_l2527_252779

theorem sequence_inequality (k : ℝ) : 
  (∀ n : ℕ+, n^2 - k*n ≥ 3^2 - k*3) → 
  5 ≤ k ∧ k ≤ 7 := by
  sorry

end NUMINAMATH_CALUDE_sequence_inequality_l2527_252779


namespace NUMINAMATH_CALUDE_fractional_equation_root_l2527_252721

theorem fractional_equation_root (n : ℤ) : 
  (∃ x : ℝ, x > 0 ∧ (x - 2) / (x - 3) = (n + 1) / (3 - x)) → n = -2 := by
  sorry

end NUMINAMATH_CALUDE_fractional_equation_root_l2527_252721


namespace NUMINAMATH_CALUDE_appears_in_31st_equation_l2527_252773

/-- The first term of the nth equation in the sequence -/
def first_term (n : ℕ) : ℕ := 2 * n^2

/-- The proposition that 2016 appears in the 31st equation -/
theorem appears_in_31st_equation : ∃ k : ℕ, k ≥ first_term 31 ∧ k ≤ first_term 32 ∧ k = 2016 :=
sorry

end NUMINAMATH_CALUDE_appears_in_31st_equation_l2527_252773


namespace NUMINAMATH_CALUDE_max_min_difference_l2527_252766

theorem max_min_difference (a b c d : ℕ+) 
  (h1 : a + b = 20)
  (h2 : a + c = 24)
  (h3 : a + d = 22) : 
  (Nat.max (a + b + c + d) (a + b + c + d) : ℤ) - 
  (Nat.min (a + b + c + d) (a + b + c + d) : ℤ) = 36 :=
by sorry

end NUMINAMATH_CALUDE_max_min_difference_l2527_252766


namespace NUMINAMATH_CALUDE_ceiling_neg_sqrt_fraction_l2527_252708

theorem ceiling_neg_sqrt_fraction : ⌈-Real.sqrt (36 / 9)⌉ = -2 := by sorry

end NUMINAMATH_CALUDE_ceiling_neg_sqrt_fraction_l2527_252708


namespace NUMINAMATH_CALUDE_precision_improves_with_sample_size_l2527_252784

/-- A structure representing a statistical sample -/
structure Sample (α : Type*) where
  data : List α
  size : Nat

/-- A measure of precision for an estimate -/
def precision (α : Type*) : Sample α → ℝ := sorry

/-- Theorem: As sample size increases, precision improves -/
theorem precision_improves_with_sample_size (α : Type*) :
  ∀ (s1 s2 : Sample α), s1.size < s2.size → precision α s1 < precision α s2 :=
sorry

end NUMINAMATH_CALUDE_precision_improves_with_sample_size_l2527_252784


namespace NUMINAMATH_CALUDE_exponent_division_l2527_252700

theorem exponent_division (a : ℝ) (h : a ≠ 0) : a^3 / a = a^2 :=
sorry

end NUMINAMATH_CALUDE_exponent_division_l2527_252700


namespace NUMINAMATH_CALUDE_parallel_line_equation_perpendicular_lines_equation_l2527_252763

-- Define the slope of line l₁
def slope_l1 : ℚ := -3 / 4

-- Define a point that l₂ passes through
def point_l2 : ℚ × ℚ := (-1, 3)

-- Define the area of the triangle formed by l₂ and the coordinate axes
def triangle_area : ℚ := 4

-- Theorem for the parallel line
theorem parallel_line_equation :
  ∃ (c : ℚ), 3 * point_l2.1 + 4 * point_l2.2 + c = 0 ∧
  ∀ (x y : ℚ), 3 * x + 4 * y + c = 0 ↔ 3 * x + 4 * y - 9 = 0 :=
sorry

-- Theorem for the perpendicular lines
theorem perpendicular_lines_equation :
  ∃ (n : ℚ), (n^2 = 96) ∧
  (∀ (x y : ℚ), 4 * x - 3 * y + n = 0 ↔ 4 * x - 3 * y + 4 * Real.sqrt 6 = 0 ∨
                                        4 * x - 3 * y - 4 * Real.sqrt 6 = 0) ∧
  (1/2 * |n/4| * |n/3| = triangle_area) :=
sorry

end NUMINAMATH_CALUDE_parallel_line_equation_perpendicular_lines_equation_l2527_252763


namespace NUMINAMATH_CALUDE_bus_journey_distance_l2527_252790

/-- Given a bus journey with the following parameters:
  * total_distance: The total distance covered by the bus
  * speed1: The first speed at which the bus travels for part of the journey
  * speed2: The second speed at which the bus travels for the remaining part of the journey
  * total_time: The total time taken for the entire journey

  This theorem proves that the distance covered at the first speed (speed1) is equal to
  the calculated value when the given conditions are met.
-/
theorem bus_journey_distance (total_distance : ℝ) (speed1 speed2 : ℝ) (total_time : ℝ)
  (h1 : total_distance = 250)
  (h2 : speed1 = 40)
  (h3 : speed2 = 60)
  (h4 : total_time = 5)
  (h5 : speed1 > 0)
  (h6 : speed2 > 0) :
  ∃ (distance1 : ℝ),
    distance1 / speed1 + (total_distance - distance1) / speed2 = total_time ∧
    distance1 = 100 := by
  sorry


end NUMINAMATH_CALUDE_bus_journey_distance_l2527_252790


namespace NUMINAMATH_CALUDE_sum_of_exponents_outside_radical_l2527_252722

-- Define the original expression
def original_expression (a b c : ℝ) : ℝ := (48 * a^5 * b^8 * c^14)^(1/4)

-- Define the simplified expression
def simplified_expression (a b c : ℝ) : ℝ := 2 * a * b^2 * c^3 * (3 * a * c^2)^(1/4)

-- Theorem statement
theorem sum_of_exponents_outside_radical (a b c : ℝ) : 
  original_expression a b c = simplified_expression a b c → 
  (1 : ℕ) + 2 + 3 = 6 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_exponents_outside_radical_l2527_252722


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l2527_252717

/-- Sum of a geometric sequence -/
def geometric_sum (a₀ : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a₀ * (1 - r^n) / (1 - r)

/-- The problem statement -/
theorem geometric_sequence_sum :
  let a₀ : ℚ := 1/3
  let r : ℚ := 1/3
  let n : ℕ := 8
  geometric_sum a₀ r n = 3280/6561 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l2527_252717


namespace NUMINAMATH_CALUDE_extremum_and_tangent_l2527_252720

/-- The function f(x) with parameter a -/
def f (a : ℝ) (x : ℝ) : ℝ := 2 * x^3 - 3 * (a + 1) * x^2 + 6 * a * x + 8

/-- The derivative of f(x) with respect to x -/
def f' (a : ℝ) (x : ℝ) : ℝ := 6 * x^2 - 6 * (a + 1) * x + 6 * a

theorem extremum_and_tangent (a : ℝ) :
  (∃ (c : ℝ), f' a 3 = c ∧ c = 0) →
  a = 3 ∧ f' 3 1 = 0 := by sorry

end NUMINAMATH_CALUDE_extremum_and_tangent_l2527_252720


namespace NUMINAMATH_CALUDE_negative_two_times_negative_three_l2527_252754

theorem negative_two_times_negative_three : (-2) * (-3) = 6 := by
  sorry

end NUMINAMATH_CALUDE_negative_two_times_negative_three_l2527_252754


namespace NUMINAMATH_CALUDE_brenda_final_lead_l2527_252730

theorem brenda_final_lead (initial_lead : ℕ) (brenda_play : ℕ) (david_play : ℕ) : 
  initial_lead = 22 → brenda_play = 15 → david_play = 32 → 
  initial_lead + brenda_play - david_play = 5 := by
  sorry

end NUMINAMATH_CALUDE_brenda_final_lead_l2527_252730


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l2527_252728

theorem sum_of_coefficients (a₀ a₁ a₂ a₃ a₄ : ℝ) :
  (∀ x, (1 - 2*x)^4 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4) →
  a₀ + a₁ + a₂ + a₃ = -15 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l2527_252728


namespace NUMINAMATH_CALUDE_expression_evaluation_l2527_252775

theorem expression_evaluation (x y z : ℝ) 
  (hz : z = y - 11)
  (hy : y = x + 3)
  (hx : x = 5) :
  ((x + 3) / (x + 2)) * ((y - 2) / (y - 3)) * ((z + 9) / (z + 7)) = 72 / 35 ∧
  x + 2 ≠ 0 ∧ y - 3 ≠ 0 ∧ z + 7 ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2527_252775


namespace NUMINAMATH_CALUDE_age_difference_proof_l2527_252756

theorem age_difference_proof (younger_age elder_age : ℕ) 
  (h1 : younger_age = 30)
  (h2 : elder_age = 50)
  (h3 : elder_age - 5 = 5 * (younger_age - 5)) :
  elder_age - younger_age = 20 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_proof_l2527_252756


namespace NUMINAMATH_CALUDE_specific_triangle_BD_length_l2527_252740

/-- A right triangle with a perpendicular from the right angle to the hypotenuse -/
structure RightTriangleWithAltitude where
  -- The lengths of the sides
  AB : ℝ
  AC : ℝ
  BC : ℝ
  -- The length of the altitude
  AD : ℝ
  -- The length of the segment from B to D
  BD : ℝ
  -- Conditions
  right_angle : AB^2 + AC^2 = BC^2
  altitude_perpendicular : AD * BC = AB * AC
  pythagoras_BD : BD^2 + AD^2 = BC^2

/-- The main theorem about the specific triangle in the problem -/
theorem specific_triangle_BD_length 
  (triangle : RightTriangleWithAltitude)
  (h_AB : triangle.AB = 45)
  (h_AC : triangle.AC = 60) :
  triangle.BD = 63 := by
  sorry

#check specific_triangle_BD_length

end NUMINAMATH_CALUDE_specific_triangle_BD_length_l2527_252740


namespace NUMINAMATH_CALUDE_a_percentage_less_than_b_l2527_252789

def full_marks : ℕ := 500
def d_marks : ℕ := (80 * full_marks) / 100
def c_marks : ℕ := (80 * d_marks) / 100
def b_marks : ℕ := (125 * c_marks) / 100
def a_marks : ℕ := 360

theorem a_percentage_less_than_b :
  (b_marks - a_marks) * 100 / b_marks = 10 := by sorry

end NUMINAMATH_CALUDE_a_percentage_less_than_b_l2527_252789


namespace NUMINAMATH_CALUDE_double_reflection_result_l2527_252732

/-- Reflects a point about the line y = x -/
def reflect_y_eq_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.2, p.1)

/-- Reflects a point about the line y = -x -/
def reflect_y_eq_neg_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.2, -p.1)

/-- The initial point -/
def initial_point : ℝ × ℝ := (3, -8)

theorem double_reflection_result :
  (reflect_y_eq_neg_x ∘ reflect_y_eq_x) initial_point = (-3, 8) := by
sorry

end NUMINAMATH_CALUDE_double_reflection_result_l2527_252732


namespace NUMINAMATH_CALUDE_hemisphere_surface_area_l2527_252734

theorem hemisphere_surface_area (r : ℝ) (h : π * r^2 = 144 * π) :
  2 * π * r^2 + π * r^2 = 432 * π := by
  sorry

end NUMINAMATH_CALUDE_hemisphere_surface_area_l2527_252734


namespace NUMINAMATH_CALUDE_simplify_expression_l2527_252759

theorem simplify_expression (a b : ℝ) : (1 : ℝ) * (2 * a) * (3 * a^2 * b) * (4 * a^3 * b^2) * (5 * a^4 * b^3) = 120 * a^10 * b^6 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2527_252759


namespace NUMINAMATH_CALUDE_all_cloaks_still_too_short_l2527_252712

/-- Represents a knight with a height and a cloak length -/
structure Knight where
  height : ℝ
  cloakLength : ℝ

/-- Predicate to check if a cloak is too short for a knight -/
def isCloakTooShort (k : Knight) : Prop := k.cloakLength < k.height

/-- Function to redistribute cloaks -/
def redistributeCloaks (knights : List Knight) : List Knight :=
  sorry

theorem all_cloaks_still_too_short (knights : List Knight) 
  (h1 : knights.length = 20)
  (h2 : ∀ k ∈ knights, isCloakTooShort k)
  (h3 : List.Pairwise (λ k1 k2 => k1.height ≤ k2.height) knights)
  : ∀ k ∈ redistributeCloaks knights, isCloakTooShort k :=
by sorry

end NUMINAMATH_CALUDE_all_cloaks_still_too_short_l2527_252712


namespace NUMINAMATH_CALUDE_greatest_x_value_l2527_252772

theorem greatest_x_value (x : ℝ) : 
  x ≠ 2 → 
  (x^2 - 5*x - 14) / (x - 2) = 4 / (x + 4) → 
  x ≤ -2 :=
by sorry

end NUMINAMATH_CALUDE_greatest_x_value_l2527_252772


namespace NUMINAMATH_CALUDE_quadrupled_base_exponent_l2527_252785

theorem quadrupled_base_exponent (a b x : ℝ) (ha : a > 0) (hb : b > 0) (hx : x > 0) :
  (4 * a)^(4 * b) = a^b * x^2 → x = 16^b * a^(3/2 * b) := by
  sorry

end NUMINAMATH_CALUDE_quadrupled_base_exponent_l2527_252785


namespace NUMINAMATH_CALUDE_fraction_ratio_l2527_252746

theorem fraction_ratio (N : ℝ) (h1 : (1/3) * (2/5) * N = 14) (h2 : 0.4 * N = 168) :
  14 / ((1/3) * (2/5) * N) = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_ratio_l2527_252746


namespace NUMINAMATH_CALUDE_complex_product_theorem_l2527_252731

theorem complex_product_theorem (Q E D : ℂ) : 
  Q = 3 + 4*I ∧ E = 2*I ∧ D = 3 - 4*I → 2 * Q * E * D = 100 * I :=
by sorry

end NUMINAMATH_CALUDE_complex_product_theorem_l2527_252731


namespace NUMINAMATH_CALUDE_triangle_angle_range_l2527_252718

theorem triangle_angle_range (a b : ℝ) (h_a : a = 2) (h_b : b = 2 * Real.sqrt 2) :
  ∃ (A : ℝ), 0 < A ∧ A ≤ π / 4 ∧
  ∀ (c : ℝ), c > 0 → a ^ 2 = b ^ 2 + c ^ 2 - 2 * b * c * Real.cos A :=
by sorry

end NUMINAMATH_CALUDE_triangle_angle_range_l2527_252718


namespace NUMINAMATH_CALUDE_units_digit_27_times_64_l2527_252725

/-- The units digit of a natural number -/
def units_digit (n : ℕ) : ℕ := n % 10

/-- The property that the units digit of a product depends only on the units digits of its factors -/
axiom units_digit_product (a b : ℕ) : 
  units_digit (a * b) = units_digit (units_digit a * units_digit b)

/-- The theorem stating that the units digit of 27 · 64 is 8 -/
theorem units_digit_27_times_64 : units_digit (27 * 64) = 8 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_27_times_64_l2527_252725


namespace NUMINAMATH_CALUDE_factorization_equality_l2527_252702

theorem factorization_equality (x : ℝ) : 90 * x^2 + 60 * x + 30 = 30 * (3 * x^2 + 2 * x + 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l2527_252702


namespace NUMINAMATH_CALUDE_ceiling_negative_example_l2527_252747

theorem ceiling_negative_example : ⌈(-3.7 : ℝ)⌉ = -3 := by sorry

end NUMINAMATH_CALUDE_ceiling_negative_example_l2527_252747


namespace NUMINAMATH_CALUDE_nonempty_set_implies_nonnegative_a_l2527_252796

theorem nonempty_set_implies_nonnegative_a (a : ℝ) :
  (∅ : Set ℝ) ⊂ {x : ℝ | x^2 ≤ a} → a ∈ Set.Ici (0 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_nonempty_set_implies_nonnegative_a_l2527_252796
