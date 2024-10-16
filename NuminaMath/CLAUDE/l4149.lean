import Mathlib

namespace NUMINAMATH_CALUDE_age_sum_problem_l4149_414964

theorem age_sum_problem (a b c : ℕ+) : 
  a = b ∧ a > c ∧ a * b * c = 162 → a + b + c = 20 := by
  sorry

end NUMINAMATH_CALUDE_age_sum_problem_l4149_414964


namespace NUMINAMATH_CALUDE_sum_of_consecutive_iff_not_power_of_two_l4149_414917

/-- A function that checks if a number is a sum of consecutive integers -/
def is_sum_of_consecutive (n : ℕ) : Prop :=
  ∃ (start k : ℕ), k ≥ 2 ∧ n = (k * (2 * start + k + 1)) / 2

/-- A function that checks if a number is a power of 2 -/
def is_power_of_two (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 2^k

/-- Theorem stating that a positive integer is a sum of two or more consecutive
    positive integers if and only if it is not a power of 2 -/
theorem sum_of_consecutive_iff_not_power_of_two (n : ℕ) (h : n > 0) :
  is_sum_of_consecutive n ↔ ¬ is_power_of_two n :=
sorry

end NUMINAMATH_CALUDE_sum_of_consecutive_iff_not_power_of_two_l4149_414917


namespace NUMINAMATH_CALUDE_large_circle_diameter_is_32_l4149_414909

/-- Represents the arrangement of circles as described in the problem -/
structure CircleArrangement where
  small_circle_radius : ℝ
  num_small_circles : ℕ
  num_layers : ℕ

/-- The specific arrangement described in the problem -/
def problem_arrangement : CircleArrangement :=
  { small_circle_radius := 4
  , num_small_circles := 8
  , num_layers := 2 }

/-- The diameter of the large circle in the arrangement -/
def large_circle_diameter (ca : CircleArrangement) : ℝ := 32

/-- Theorem stating that the diameter of the large circle in the problem arrangement is 32 units -/
theorem large_circle_diameter_is_32 :
  large_circle_diameter problem_arrangement = 32 := by
  sorry

end NUMINAMATH_CALUDE_large_circle_diameter_is_32_l4149_414909


namespace NUMINAMATH_CALUDE_difference_of_squares_l4149_414992

theorem difference_of_squares (m n : ℝ) : (3*m + n) * (3*m - n) = (3*m)^2 - n^2 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l4149_414992


namespace NUMINAMATH_CALUDE_smallest_number_with_conditions_l4149_414905

theorem smallest_number_with_conditions : ∃ n : ℕ, 
  (∀ k : ℕ, k < n → ¬(11 ∣ k ∧ (∀ i : ℕ, 3 ≤ i ∧ i ≤ 7 → k % i = 2))) ∧ 
  11 ∣ n ∧ 
  (∀ i : ℕ, 3 ≤ i ∧ i ≤ 7 → n % i = 2) ∧ 
  n = 2102 :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_with_conditions_l4149_414905


namespace NUMINAMATH_CALUDE_unique_linear_equation_solution_l4149_414906

theorem unique_linear_equation_solution (m n : ℕ+) :
  ∃ (a b c : ℤ), ∀ (x y : ℕ+),
    (a * x.val + b * y.val = c) ↔ (x = m ∧ y = n) :=
sorry

end NUMINAMATH_CALUDE_unique_linear_equation_solution_l4149_414906


namespace NUMINAMATH_CALUDE_largest_possible_s_value_l4149_414948

theorem largest_possible_s_value (r s : ℕ) (hr : r ≥ s) (hs : s ≥ 3) : 
  (((r - 2) * 180 : ℚ) / r) / (((s - 2) * 180 : ℚ) / s) = 101 / 97 → s ≤ 100 := by
  sorry

end NUMINAMATH_CALUDE_largest_possible_s_value_l4149_414948


namespace NUMINAMATH_CALUDE_value_of_y_l4149_414968

theorem value_of_y : ∃ y : ℝ, (3 * y - 9) / 3 = 18 ∧ y = 21 := by
  sorry

end NUMINAMATH_CALUDE_value_of_y_l4149_414968


namespace NUMINAMATH_CALUDE_opposite_angles_equal_l4149_414946

/-- Two angles are opposite if they are formed by two intersecting lines and are not adjacent. -/
def are_opposite_angles (α β : Real) : Prop := sorry

/-- The measure of an angle in radians. -/
def angle_measure (α : Real) : ℝ := sorry

theorem opposite_angles_equal (α β : Real) :
  are_opposite_angles α β → angle_measure α = angle_measure β := by sorry

end NUMINAMATH_CALUDE_opposite_angles_equal_l4149_414946


namespace NUMINAMATH_CALUDE_at_least_two_primes_in_sequence_l4149_414934

theorem at_least_two_primes_in_sequence : ∃ (m n : ℕ), 
  2 ≤ m ∧ 2 ≤ n ∧ m ≠ n ∧ 
  Nat.Prime (m^3 + m + 1) ∧ 
  Nat.Prime (n^3 + n + 1) :=
sorry

end NUMINAMATH_CALUDE_at_least_two_primes_in_sequence_l4149_414934


namespace NUMINAMATH_CALUDE_find_a_l4149_414950

-- Define the inequality
def inequality (a : ℝ) (x : ℝ) : Prop := (a * x) / (x - 1) < 1

-- Define the solution set
def solution_set (a : ℝ) : Set ℝ := {x | x < 1 ∨ x > 2}

-- Theorem statement
theorem find_a : ∃ a : ℝ, (∀ x : ℝ, inequality a x ↔ x ∈ solution_set a) ∧ a = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_find_a_l4149_414950


namespace NUMINAMATH_CALUDE_equation_one_integral_root_l4149_414907

theorem equation_one_integral_root :
  ∃! (x : ℤ), x - 9 / (x - 5 : ℚ) = 4 - 9 / (x - 5 : ℚ) :=
by sorry

end NUMINAMATH_CALUDE_equation_one_integral_root_l4149_414907


namespace NUMINAMATH_CALUDE_least_subtraction_for_divisibility_l4149_414921

theorem least_subtraction_for_divisibility : ∃ (n : ℕ), 
  (n = 33) ∧ 
  (∀ m : ℕ, m < n → ¬(79 ∣ (123457 - m))) ∧ 
  (79 ∣ (123457 - n)) := by
sorry

end NUMINAMATH_CALUDE_least_subtraction_for_divisibility_l4149_414921


namespace NUMINAMATH_CALUDE_quadratic_root_zero_l4149_414977

theorem quadratic_root_zero (m : ℝ) : 
  (∃ x, (m - 1) * x^2 + 2 * x + m^2 - 1 = 0) ∧ 
  ((m - 1) * 0^2 + 2 * 0 + m^2 - 1 = 0) → 
  m = -1 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_zero_l4149_414977


namespace NUMINAMATH_CALUDE_height_relation_l4149_414935

/-- Two right circular cylinders with equal volumes and related radii -/
structure TwoCylinders where
  r₁ : ℝ  -- radius of the first cylinder
  h₁ : ℝ  -- height of the first cylinder
  r₂ : ℝ  -- radius of the second cylinder
  h₂ : ℝ  -- height of the second cylinder
  r₁_pos : 0 < r₁
  h₁_pos : 0 < h₁
  r₂_pos : 0 < r₂
  h₂_pos : 0 < h₂
  equal_volume : r₁^2 * h₁ = r₂^2 * h₂
  radius_relation : r₂ = 1.2 * r₁

theorem height_relation (c : TwoCylinders) : c.h₁ = 1.44 * c.h₂ := by
  sorry

end NUMINAMATH_CALUDE_height_relation_l4149_414935


namespace NUMINAMATH_CALUDE_decimal_to_fraction_times_three_l4149_414990

theorem decimal_to_fraction_times_three :
  (2.36 : ℚ) * 3 = 177 / 25 := by
sorry

end NUMINAMATH_CALUDE_decimal_to_fraction_times_three_l4149_414990


namespace NUMINAMATH_CALUDE_range_of_a_l4149_414988

-- Define the propositions p and q
def p (x : ℝ) : Prop := abs (x + 1) > 2
def q (x a : ℝ) : Prop := abs x > a

-- Define the necessary but not sufficient condition
def necessary_not_sufficient (a : ℝ) : Prop :=
  (∀ x, ¬(q x a) → ¬(p x)) ∧ 
  (∃ x, ¬(p x) ∧ (q x a))

-- State the theorem
theorem range_of_a : 
  ∀ a : ℝ, (∀ x, p x → q x a) ∧ necessary_not_sufficient a ↔ a ≤ 1 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l4149_414988


namespace NUMINAMATH_CALUDE_square_octagon_exterior_angle_l4149_414936

/-- The measure of an interior angle of a regular polygon with n sides -/
def interior_angle (n : ℕ) : ℚ := 180 * (n - 2) / n

/-- The configuration of a square and regular octagon sharing a side -/
structure SquareOctagonConfig where
  square_angle : ℚ  -- Interior angle of the square
  octagon_angle : ℚ -- Interior angle of the octagon
  common_side : ℚ   -- Length of the common side (not used in this problem, but included for completeness)

/-- The exterior angle formed by the non-shared sides of the square and octagon -/
def exterior_angle (config : SquareOctagonConfig) : ℚ :=
  360 - config.square_angle - config.octagon_angle

/-- Theorem: The exterior angle in the square-octagon configuration is 135° -/
theorem square_octagon_exterior_angle :
  ∀ (config : SquareOctagonConfig),
    config.square_angle = 90 ∧
    config.octagon_angle = interior_angle 8 →
    exterior_angle config = 135 := by
  sorry


end NUMINAMATH_CALUDE_square_octagon_exterior_angle_l4149_414936


namespace NUMINAMATH_CALUDE_order_of_expressions_l4149_414955

theorem order_of_expressions :
  let a : ℝ := (1/2)^(1/3)
  let b : ℝ := (1/3)^(1/2)
  let c : ℝ := Real.log (3/Real.pi)
  c < b ∧ b < a := by sorry

end NUMINAMATH_CALUDE_order_of_expressions_l4149_414955


namespace NUMINAMATH_CALUDE_num_routes_eq_binomial_num_routes_is_six_l4149_414925

/-- The number of different routes from the bottom-left corner to the top-right corner of a 2x2 grid,
    moving only upwards or to the right one square at a time. -/
def num_routes : ℕ := 6

/-- The size of the grid (2x2 in this case) -/
def grid_size : ℕ := 2

/-- The total number of moves required to reach the top-right corner from the bottom-left corner -/
def total_moves : ℕ := grid_size * 2

/-- Theorem stating that the number of routes is equal to the binomial coefficient (total_moves choose grid_size) -/
theorem num_routes_eq_binomial :
  num_routes = Nat.choose total_moves grid_size :=
by sorry

/-- Theorem proving that the number of routes is 6 -/
theorem num_routes_is_six :
  num_routes = 6 :=
by sorry

end NUMINAMATH_CALUDE_num_routes_eq_binomial_num_routes_is_six_l4149_414925


namespace NUMINAMATH_CALUDE_line_equation_l4149_414965

/-- A line in the 2D plane represented by its equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if a point (x, y) lies on a given line -/
def Line.contains (l : Line) (x y : ℝ) : Prop :=
  l.a * x + l.b * y + l.c = 0

/-- Check if a line has equal intercepts on x and y axes -/
def Line.has_equal_intercepts (l : Line) : Prop :=
  l.a ≠ 0 ∧ l.b ≠ 0 ∧ (-l.c / l.a = -l.c / l.b)

theorem line_equation (l : Line) :
  l.has_equal_intercepts ∧ l.contains 1 2 →
  (l.a = 2 ∧ l.b = -1 ∧ l.c = 0) ∨ (l.a = 1 ∧ l.b = 1 ∧ l.c = -3) :=
sorry

end NUMINAMATH_CALUDE_line_equation_l4149_414965


namespace NUMINAMATH_CALUDE_dentist_age_l4149_414947

/-- The dentist's current age satisfies the given condition and is equal to 32. -/
theorem dentist_age : ∃ (x : ℕ), (x - 8) / 6 = (x + 8) / 10 ∧ x = 32 := by
  sorry

end NUMINAMATH_CALUDE_dentist_age_l4149_414947


namespace NUMINAMATH_CALUDE_zeros_of_f_l4149_414943

def f (x : ℝ) : ℝ := (x - 1) * (x^2 - 2*x - 3)

theorem zeros_of_f :
  {x : ℝ | f x = 0} = {1, -1, 3} := by sorry

end NUMINAMATH_CALUDE_zeros_of_f_l4149_414943


namespace NUMINAMATH_CALUDE_div_power_eq_reciprocal_pow_l4149_414918

/-- Definition of division power for rational numbers -/
def div_power (a : ℚ) (n : ℕ) : ℚ :=
  if n = 0 then 1
  else if n = 1 then a
  else a / (div_power a (n - 1))

/-- Theorem: Division power is equivalent to reciprocal exponentiation -/
theorem div_power_eq_reciprocal_pow (a : ℚ) (n : ℕ) (h : a ≠ 0) :
  div_power a n = (1 / a) ^ (n - 2) :=
sorry

end NUMINAMATH_CALUDE_div_power_eq_reciprocal_pow_l4149_414918


namespace NUMINAMATH_CALUDE_subsidy_calculation_l4149_414930

/-- Represents the "Home Appliances to the Countryside" initiative subsidy calculation -/
theorem subsidy_calculation (x : ℝ) : 
  (20 * x * 0.13 = 2340) ↔ 
  (∃ (subsidy_rate : ℝ) (num_phones : ℕ) (total_subsidy : ℝ),
    subsidy_rate = 0.13 ∧ 
    num_phones = 20 ∧ 
    total_subsidy = 2340 ∧
    num_phones * (x * subsidy_rate) = total_subsidy) :=
by sorry

end NUMINAMATH_CALUDE_subsidy_calculation_l4149_414930


namespace NUMINAMATH_CALUDE_joes_money_from_mother_l4149_414927

def notebook_cost : ℕ := 4
def book_cost : ℕ := 7
def notebooks_bought : ℕ := 7
def books_bought : ℕ := 2
def money_left : ℕ := 14

theorem joes_money_from_mother : 
  notebook_cost * notebooks_bought + book_cost * books_bought + money_left = 56 := by
  sorry

end NUMINAMATH_CALUDE_joes_money_from_mother_l4149_414927


namespace NUMINAMATH_CALUDE_math_problem_l4149_414978

theorem math_problem (m n : ℕ) (hm : m > 0) (hn : n > 0) (h_sum : 3 * m + 2 * n = 225) :
  (gcd m n = 15 → m + n = 105) ∧ (lcm m n = 45 → m + n = 90) := by
  sorry

end NUMINAMATH_CALUDE_math_problem_l4149_414978


namespace NUMINAMATH_CALUDE_no_integer_root_2016_l4149_414951

theorem no_integer_root_2016 (a b c d : ℤ) (p : ℤ → ℤ) :
  (∀ x : ℤ, p x = a * x^3 + b * x^2 + c * x + d) →
  p 1 = 2015 →
  p 2 = 2017 →
  ∀ x : ℤ, p x ≠ 2016 := by
sorry

end NUMINAMATH_CALUDE_no_integer_root_2016_l4149_414951


namespace NUMINAMATH_CALUDE_probability_in_specific_rectangle_l4149_414940

/-- A rectangle in 2D space --/
structure Rectangle where
  x1 : ℝ
  y1 : ℝ
  x2 : ℝ
  y2 : ℝ

/-- The probability that a randomly selected point in the rectangle is closer to one point than another --/
def probability_closer_to_point (r : Rectangle) (p1 : ℝ × ℝ) (p2 : ℝ × ℝ) : ℝ :=
  sorry

/-- The theorem statement --/
theorem probability_in_specific_rectangle : 
  let r : Rectangle := { x1 := 0, y1 := 0, x2 := 3, y2 := 2 }
  probability_closer_to_point r (0, 0) (4, 2) = 5/6 := by
  sorry

end NUMINAMATH_CALUDE_probability_in_specific_rectangle_l4149_414940


namespace NUMINAMATH_CALUDE_leapYearsIn123Years_l4149_414975

/-- In a calendrical system where leap years occur every three years, 
    this function calculates the number of leap years in a given period. -/
def leapYearsCount (periodLength : ℕ) : ℕ :=
  periodLength / 3

/-- Theorem stating that in a 123-year period, the number of leap years is 41. -/
theorem leapYearsIn123Years : leapYearsCount 123 = 41 := by
  sorry

end NUMINAMATH_CALUDE_leapYearsIn123Years_l4149_414975


namespace NUMINAMATH_CALUDE_moving_circle_center_trajectory_l4149_414973

/-- A moving circle that passes through (1, 0) and is tangent to x = -1 -/
structure MovingCircle where
  center : ℝ × ℝ
  passes_through_one_zero : (center.1 - 1)^2 + center.2^2 = (center.1 + 1)^2
  tangent_to_neg_one : True  -- This condition is implied by the equation above

/-- The trajectory of the center of the moving circle is y² = 4x -/
theorem moving_circle_center_trajectory (M : MovingCircle) : 
  M.center.2^2 = 4 * M.center.1 := by
  sorry

end NUMINAMATH_CALUDE_moving_circle_center_trajectory_l4149_414973


namespace NUMINAMATH_CALUDE_pipe_length_l4149_414911

theorem pipe_length : ∀ (shorter_piece longer_piece total_length : ℕ),
  shorter_piece = 28 →
  longer_piece = shorter_piece + 12 →
  total_length = shorter_piece + longer_piece →
  total_length = 68 :=
by
  sorry

end NUMINAMATH_CALUDE_pipe_length_l4149_414911


namespace NUMINAMATH_CALUDE_star_distance_l4149_414938

/-- The distance between a star and Earth given the speed of light and time taken for light to reach Earth -/
theorem star_distance (c : ℝ) (t : ℝ) (y : ℝ) (h1 : c = 3 * 10^5) (h2 : t = 10) (h3 : y = 3.1 * 10^7) :
  c * (t * y) = 9.3 * 10^13 := by
  sorry

end NUMINAMATH_CALUDE_star_distance_l4149_414938


namespace NUMINAMATH_CALUDE_midpoint_area_in_square_l4149_414945

/-- The area enclosed by midpoints of line segments in a square --/
theorem midpoint_area_in_square (s : ℝ) (h : s = 3) : 
  let midpoint_area := s^2 - (s^2 * Real.pi) / 4
  midpoint_area = 9 - (9 * Real.pi) / 4 := by
  sorry

end NUMINAMATH_CALUDE_midpoint_area_in_square_l4149_414945


namespace NUMINAMATH_CALUDE_probability_sum_six_l4149_414976

/-- The number of faces on each die -/
def numFaces : ℕ := 6

/-- The target sum we're looking for -/
def targetSum : ℕ := 6

/-- The set of possible outcomes when rolling two dice -/
def outcomes : Finset (ℕ × ℕ) :=
  Finset.product (Finset.range numFaces) (Finset.range numFaces)

/-- The set of favorable outcomes (sum equals targetSum) -/
def favorableOutcomes : Finset (ℕ × ℕ) :=
  outcomes.filter (fun p => p.1 + p.2 + 2 = targetSum)

/-- The probability of rolling a sum of 6 with two fair six-sided dice -/
theorem probability_sum_six :
  Nat.card favorableOutcomes / Nat.card outcomes = 5 / 36 := by
  sorry


end NUMINAMATH_CALUDE_probability_sum_six_l4149_414976


namespace NUMINAMATH_CALUDE_mork_tax_rate_calculation_l4149_414982

-- Define the variables
def mork_income : ℝ := sorry
def mork_tax_rate : ℝ := sorry
def mindy_tax_rate : ℝ := 0.25
def combined_tax_rate : ℝ := 0.28

-- Define the theorem
theorem mork_tax_rate_calculation :
  mork_tax_rate = 0.4 :=
by
  -- Assume the conditions
  have h1 : mindy_tax_rate = 0.25 := rfl
  have h2 : combined_tax_rate = 0.28 := rfl
  have h3 : mork_income > 0 := sorry
  have h4 : mork_tax_rate * mork_income + mindy_tax_rate * (4 * mork_income) = combined_tax_rate * (5 * mork_income) := sorry

  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_mork_tax_rate_calculation_l4149_414982


namespace NUMINAMATH_CALUDE_car_speed_problem_l4149_414931

-- Define the parameters of the problem
def initial_distance : ℝ := 10
def final_distance : ℝ := 8
def time : ℝ := 2.25
def speed_A : ℝ := 58

-- Define the speed of Car B as a variable
def speed_B : ℝ := 50

-- Theorem statement
theorem car_speed_problem :
  initial_distance + 
  speed_A * time = 
  speed_B * time + 
  initial_distance + 
  final_distance := by sorry

end NUMINAMATH_CALUDE_car_speed_problem_l4149_414931


namespace NUMINAMATH_CALUDE_least_three_digit_multiple_of_11_l4149_414986

theorem least_three_digit_multiple_of_11 : ∃ n : ℕ, 
  (n ≥ 100 ∧ n < 1000) ∧ 
  (∃ k : ℕ, n = 11 * k) ∧
  (∀ m : ℕ, m ≥ 100 ∧ m < 1000 ∧ (∃ j : ℕ, m = 11 * j) → n ≤ m) ∧
  n = 110 := by
  sorry

end NUMINAMATH_CALUDE_least_three_digit_multiple_of_11_l4149_414986


namespace NUMINAMATH_CALUDE_F_is_odd_l4149_414985

-- Define a real-valued function f
variable (f : ℝ → ℝ)

-- Define F in terms of f
def F (f : ℝ → ℝ) : ℝ → ℝ := λ x ↦ f x - f (-x)

-- Theorem: F is an odd function
theorem F_is_odd (f : ℝ → ℝ) : ∀ x : ℝ, F f (-x) = -(F f x) :=
by
  sorry

end NUMINAMATH_CALUDE_F_is_odd_l4149_414985


namespace NUMINAMATH_CALUDE_committee_formation_ways_l4149_414995

theorem committee_formation_ways (n m : ℕ) (hn : n = 10) (hm : m = 4) : 
  Nat.choose n m = 210 := by
  sorry

end NUMINAMATH_CALUDE_committee_formation_ways_l4149_414995


namespace NUMINAMATH_CALUDE_total_eggs_l4149_414929

theorem total_eggs (num_students : ℕ) (eggs_per_student : ℕ) (h1 : num_students = 7) (h2 : eggs_per_student = 8) :
  num_students * eggs_per_student = 56 := by
  sorry

end NUMINAMATH_CALUDE_total_eggs_l4149_414929


namespace NUMINAMATH_CALUDE_f_divisible_by_two_l4149_414922

/-- A polynomial of the form x^2 + px + q -/
def f (p q : ℤ) (x : ℤ) : ℤ := x^2 + p*x + q

/-- The polynomial f is divisible by 2 for all integer x if and only if p is odd and q is even -/
theorem f_divisible_by_two (p q : ℤ) :
  (∀ x : ℤ, 2 ∣ f p q x) ↔ (Odd p ∧ Even q) :=
sorry

end NUMINAMATH_CALUDE_f_divisible_by_two_l4149_414922


namespace NUMINAMATH_CALUDE_safe_flight_probability_l4149_414956

/-- Represents a rectangular prism with given dimensions -/
structure RectangularPrism where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of a rectangular prism -/
def volume (prism : RectangularPrism) : ℝ :=
  prism.length * prism.width * prism.height

/-- Represents the problem setup -/
def problem_setup : Prop :=
  let outer_prism : RectangularPrism := { length := 5, width := 4, height := 3 }
  let inner_prism : RectangularPrism := { length := 3, width := 2, height := 1 }
  let outer_volume := volume outer_prism
  let inner_volume := volume inner_prism
  (inner_volume / outer_volume) = (1 : ℝ) / 10

/-- The main theorem to prove -/
theorem safe_flight_probability : problem_setup := by
  sorry

end NUMINAMATH_CALUDE_safe_flight_probability_l4149_414956


namespace NUMINAMATH_CALUDE_solve_y_equation_l4149_414939

theorem solve_y_equation : ∃ y : ℚ, (3 * y) / 7 = 21 ∧ y = 49 := by
  sorry

end NUMINAMATH_CALUDE_solve_y_equation_l4149_414939


namespace NUMINAMATH_CALUDE_quadratic_roots_properties_l4149_414952

theorem quadratic_roots_properties (m : ℝ) (x₁ x₂ : ℝ) :
  (∀ x, x^2 - 2*(m+1)*x + m^2 + 3 = 0 ↔ x = x₁ ∨ x = x₂) →
  (m ≥ 1 ∧ ∃ m', m' ≥ 1 ∧ (x₁ - 1)*(x₂ - 1) = m' + 6 ∧ m' = 4) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_properties_l4149_414952


namespace NUMINAMATH_CALUDE_constant_value_proof_l4149_414997

theorem constant_value_proof (x y : ℝ) (a : ℝ) 
  (h1 : (a * x + 8 * y) / (x - 2 * y) = 29)
  (h2 : x / (2 * y) = 3 / 2) : 
  a = 7 := by sorry

end NUMINAMATH_CALUDE_constant_value_proof_l4149_414997


namespace NUMINAMATH_CALUDE_min_lines_for_31_segments_l4149_414969

/-- A broken line represented by its number of segments -/
structure BrokenLine where
  segments : ℕ
  no_self_intersections : Bool
  distinct_endpoints : Bool

/-- The minimum number of straight lines formed by extending all segments of a broken line -/
def min_straight_lines (bl : BrokenLine) : ℕ :=
  (bl.segments + 1) / 2

/-- Theorem stating the minimum number of straight lines for a specific broken line -/
theorem min_lines_for_31_segments :
  ∀ (bl : BrokenLine),
    bl.segments = 31 →
    bl.no_self_intersections = true →
    bl.distinct_endpoints = true →
    min_straight_lines bl = 16 := by
  sorry

#eval min_straight_lines { segments := 31, no_self_intersections := true, distinct_endpoints := true }

end NUMINAMATH_CALUDE_min_lines_for_31_segments_l4149_414969


namespace NUMINAMATH_CALUDE_stating_calculate_total_applicants_l4149_414933

/-- Represents the proportion of students who applied to first-tier colleges in a sample -/
def sample_proportion (sample_size : ℕ) (applicants_in_sample : ℕ) : ℚ :=
  applicants_in_sample / sample_size

/-- Represents the proportion of students who applied to first-tier colleges in the population -/
def population_proportion (population_size : ℕ) (total_applicants : ℕ) : ℚ :=
  total_applicants / population_size

/-- 
Theorem stating that if the sample proportion equals the population proportion,
then the total number of applicants in the population can be calculated.
-/
theorem calculate_total_applicants 
  (population_size : ℕ) 
  (sample_size : ℕ) 
  (applicants_in_sample : ℕ) 
  (h1 : population_size = 1000)
  (h2 : sample_size = 150)
  (h3 : applicants_in_sample = 60) :
  ∃ (total_applicants : ℕ),
    sample_proportion sample_size applicants_in_sample = 
    population_proportion population_size total_applicants ∧ 
    total_applicants = 400 := by
  sorry

end NUMINAMATH_CALUDE_stating_calculate_total_applicants_l4149_414933


namespace NUMINAMATH_CALUDE_maria_stamp_collection_l4149_414937

/-- Given that Maria has 40 stamps and wants to increase her collection by 20%,
    prove that she will have a total of 48 stamps. -/
theorem maria_stamp_collection (initial_stamps : ℕ) (increase_percentage : ℚ) : 
  initial_stamps = 40 → 
  increase_percentage = 20 / 100 → 
  initial_stamps + (initial_stamps * increase_percentage).floor = 48 := by
  sorry

end NUMINAMATH_CALUDE_maria_stamp_collection_l4149_414937


namespace NUMINAMATH_CALUDE_some_number_value_l4149_414904

theorem some_number_value : ∃ (some_number : ℝ), 
  (0.0077 * 3.6) / (0.04 * some_number * 0.007) = 990.0000000000001 ∧ some_number = 10 := by
  sorry

end NUMINAMATH_CALUDE_some_number_value_l4149_414904


namespace NUMINAMATH_CALUDE_second_group_size_l4149_414989

theorem second_group_size :
  ∀ (n : ℕ), 
  -- First group has 20 students with average height 20 cm
  (20 : ℝ) * 20 = 400 ∧
  -- Second group has n students with average height 20 cm
  (n : ℝ) * 20 = 20 * n ∧
  -- Combined group has 31 students with average height 20 cm
  (31 : ℝ) * 20 = 620 ∧
  -- Total height of combined groups equals sum of individual group heights
  400 + 20 * n = 620
  →
  n = 11 := by
sorry

end NUMINAMATH_CALUDE_second_group_size_l4149_414989


namespace NUMINAMATH_CALUDE_p_sufficient_not_necessary_for_q_l4149_414972

-- Define the propositions p and q
def p (x : ℝ) : Prop := x = 2
def q (x : ℝ) : Prop := 0 < x ∧ x < 3

-- Theorem stating that p is a sufficient but not necessary condition for q
theorem p_sufficient_not_necessary_for_q :
  (∀ x, p x → q x) ∧ ¬(∀ x, q x → p x) := by
  sorry

end NUMINAMATH_CALUDE_p_sufficient_not_necessary_for_q_l4149_414972


namespace NUMINAMATH_CALUDE_lauren_reaches_andrea_l4149_414987

/-- The initial distance between Andrea and Lauren in kilometers -/
def initial_distance : ℝ := 30

/-- The rate at which the distance between Andrea and Lauren decreases in km/min -/
def distance_decrease_rate : ℝ := 2

/-- The duration of initial biking in minutes -/
def initial_biking_time : ℝ := 10

/-- The duration of the stop in minutes -/
def stop_time : ℝ := 5

/-- Andrea's speed in km/h -/
def andrea_speed : ℝ := 40

/-- Lauren's speed in km/h -/
def lauren_speed : ℝ := 80

/-- The total time it takes for Lauren to reach Andrea -/
def total_time : ℝ := 22.5

theorem lauren_reaches_andrea :
  let distance_covered := distance_decrease_rate * initial_biking_time
  let remaining_distance := initial_distance - distance_covered
  let lauren_final_time := remaining_distance / (lauren_speed / 60)
  total_time = initial_biking_time + stop_time + lauren_final_time :=
by
  sorry

#check lauren_reaches_andrea

end NUMINAMATH_CALUDE_lauren_reaches_andrea_l4149_414987


namespace NUMINAMATH_CALUDE_quadratic_polynomial_value_l4149_414981

/-- A quadratic polynomial with integer coefficients -/
def QuadraticPoly (p : ℤ → ℤ) : Prop :=
  ∃ a b c : ℤ, ∀ x, p x = a * x^2 + b * x + c

theorem quadratic_polynomial_value (p : ℤ → ℤ) :
  QuadraticPoly p →
  p 41 = 42 →
  (∃ a b : ℤ, a > 41 ∧ b > 41 ∧ p a = 13 ∧ p b = 73) →
  p 1 = 2842 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_polynomial_value_l4149_414981


namespace NUMINAMATH_CALUDE_craig_appliance_sales_l4149_414984

/-- The number of appliances sold by Craig in a week -/
def num_appliances : ℕ := 6

/-- The total selling price of appliances in dollars -/
def total_selling_price : ℚ := 3620

/-- The total commission Craig earned in dollars -/
def total_commission : ℚ := 662

/-- The fixed commission per appliance in dollars -/
def fixed_commission : ℚ := 50

/-- The percentage of selling price Craig receives as commission -/
def commission_rate : ℚ := 1/10

theorem craig_appliance_sales :
  num_appliances = 6 ∧
  (num_appliances : ℚ) * fixed_commission + commission_rate * total_selling_price = total_commission :=
sorry

end NUMINAMATH_CALUDE_craig_appliance_sales_l4149_414984


namespace NUMINAMATH_CALUDE_junk_mail_distribution_l4149_414993

theorem junk_mail_distribution (total_mail : ℕ) (total_houses : ℕ) (white_houses : ℕ) (red_houses : ℕ)
  (h1 : total_mail = 48)
  (h2 : total_houses = 8)
  (h3 : white_houses = 2)
  (h4 : red_houses = 3)
  (h5 : total_houses > 0) :
  let colored_houses := white_houses + red_houses
  let mail_per_house := total_mail / total_houses
  mail_per_house = 6 ∧ colored_houses * mail_per_house = colored_houses * 6 :=
by sorry

end NUMINAMATH_CALUDE_junk_mail_distribution_l4149_414993


namespace NUMINAMATH_CALUDE_normal_to_curve_l4149_414971

-- Define the curve
def curve (x y a : ℝ) : Prop := x^(2/3) + y^(2/3) = a^(2/3)

-- Define the normal equation
def normal_equation (x y a θ : ℝ) : Prop := y * Real.cos θ - x * Real.sin θ = a * Real.cos (2 * θ)

-- Theorem statement
theorem normal_to_curve (x y a θ : ℝ) :
  curve x y a →
  (∃ (p q : ℝ), curve p q a ∧ 
    -- The point (p, q) is on the curve and the normal at this point makes an angle θ with the X-axis
    (y - q) * Real.cos θ = (x - p) * Real.sin θ) →
  normal_equation x y a θ :=
by sorry

end NUMINAMATH_CALUDE_normal_to_curve_l4149_414971


namespace NUMINAMATH_CALUDE_art_dealer_loss_l4149_414966

theorem art_dealer_loss (selling_price : ℝ) (selling_price_positive : selling_price > 0) :
  let profit_percentage : ℝ := 0.1
  let loss_percentage : ℝ := 0.1
  let cost_price_1 : ℝ := selling_price / (1 + profit_percentage)
  let cost_price_2 : ℝ := selling_price / (1 - loss_percentage)
  let profit : ℝ := selling_price - cost_price_1
  let loss : ℝ := cost_price_2 - selling_price
  let net_loss : ℝ := loss - profit
  net_loss = 0.02 * selling_price :=
by sorry

end NUMINAMATH_CALUDE_art_dealer_loss_l4149_414966


namespace NUMINAMATH_CALUDE_probability_three_yellow_one_white_l4149_414920

/-- The probability of drawing 3 yellow balls followed by 1 white ball from a box
    containing 5 yellow balls and 4 white balls, where yellow balls are returned
    after being drawn. -/
theorem probability_three_yellow_one_white (yellow_balls : ℕ) (white_balls : ℕ)
    (h_yellow : yellow_balls = 5) (h_white : white_balls = 4) :
    (yellow_balls / (yellow_balls + white_balls : ℚ))^3 *
    (white_balls / (yellow_balls + white_balls : ℚ)) =
    (5/9 : ℚ)^3 * (4/9 : ℚ) :=
by sorry

end NUMINAMATH_CALUDE_probability_three_yellow_one_white_l4149_414920


namespace NUMINAMATH_CALUDE_mary_always_wins_l4149_414957

/-- Represents a player in the game -/
inductive Player : Type
| john : Player
| mary : Player

/-- Represents a move in the game -/
inductive Move : Type
| plus : Move
| minus : Move

/-- Represents the state of the game -/
structure GameState :=
(moves : List Move)

/-- The list of numbers in the game -/
def numbers : List Int := [-1, -2, -3, -4, -5, -6, -7, -8]

/-- Calculate the final sum based on the moves and numbers -/
def finalSum (state : GameState) : Int :=
  sorry

/-- Check if Mary wins given the final sum -/
def maryWins (sum : Int) : Prop :=
  sum = -4 ∨ sum = -2 ∨ sum = 0 ∨ sum = 2 ∨ sum = 4

/-- Mary's strategy function -/
def maryStrategy (state : GameState) : Move :=
  sorry

/-- Theorem stating that Mary always wins -/
theorem mary_always_wins :
  ∀ (game : List Move),
    game.length ≤ 8 →
    maryWins (finalSum { moves := game ++ [maryStrategy { moves := game }] }) :=
sorry

end NUMINAMATH_CALUDE_mary_always_wins_l4149_414957


namespace NUMINAMATH_CALUDE_cone_base_circumference_l4149_414932

/-- The circumference of the base of a right circular cone formed from a sector of a circle --/
theorem cone_base_circumference (r : ℝ) (θ : ℝ) (h_r : r = 6) (h_θ : θ = 240) :
  let original_circumference := 2 * π * r
  let sector_proportion := θ / 360
  let base_circumference := sector_proportion * original_circumference
  base_circumference = 8 * π := by
  sorry

end NUMINAMATH_CALUDE_cone_base_circumference_l4149_414932


namespace NUMINAMATH_CALUDE_hexagon_area_equal_perimeter_l4149_414942

theorem hexagon_area_equal_perimeter (s t : ℝ) : 
  s > 0 → 
  t > 0 → 
  3 * s = 6 * t → -- Equal perimeters condition
  s^2 * Real.sqrt 3 / 4 = 16 → -- Triangle area condition
  6 * (t^2 * Real.sqrt 3 / 4) = 24 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_area_equal_perimeter_l4149_414942


namespace NUMINAMATH_CALUDE_sunglasses_wearers_l4149_414967

theorem sunglasses_wearers (total_adults : ℕ) (women_percentage : ℚ) (men_percentage : ℚ) : 
  total_adults = 1800 → 
  women_percentage = 25 / 100 →
  men_percentage = 10 / 100 →
  (total_adults / 2 * women_percentage + total_adults / 2 * men_percentage : ℚ) = 315 := by
  sorry

end NUMINAMATH_CALUDE_sunglasses_wearers_l4149_414967


namespace NUMINAMATH_CALUDE_number_problem_l4149_414998

theorem number_problem : ∃ x : ℝ, 4 * x + 7 * x = 55 ∧ x = 5 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l4149_414998


namespace NUMINAMATH_CALUDE_least_with_twelve_factors_l4149_414908

/-- The number of positive factors of a positive integer -/
def num_factors (n : ℕ+) : ℕ := sorry

/-- n is the least positive integer with exactly k positive factors -/
def is_least_with_factors (n : ℕ+) (k : ℕ) : Prop :=
  num_factors n = k ∧ ∀ m : ℕ+, m < n → num_factors m ≠ k

theorem least_with_twelve_factors :
  is_least_with_factors 96 12 := by sorry

end NUMINAMATH_CALUDE_least_with_twelve_factors_l4149_414908


namespace NUMINAMATH_CALUDE_min_sum_squares_l4149_414902

def S : Finset Int := {-8, -6, -4, -1, 3, 5, 7, 10}

theorem min_sum_squares (p q r s t u v w : Int) 
  (h_distinct : p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ p ≠ t ∧ p ≠ u ∧ p ≠ v ∧ p ≠ w ∧
                q ≠ r ∧ q ≠ s ∧ q ≠ t ∧ q ≠ u ∧ q ≠ v ∧ q ≠ w ∧
                r ≠ s ∧ r ≠ t ∧ r ≠ u ∧ r ≠ v ∧ r ≠ w ∧
                s ≠ t ∧ s ≠ u ∧ s ≠ v ∧ s ≠ w ∧
                t ≠ u ∧ t ≠ v ∧ t ≠ w ∧
                u ≠ v ∧ u ≠ w ∧
                v ≠ w)
  (h_in_S : p ∈ S ∧ q ∈ S ∧ r ∈ S ∧ s ∈ S ∧ t ∈ S ∧ u ∈ S ∧ v ∈ S ∧ w ∈ S) :
  (p + q + r + s)^2 + (t + u + v + w)^2 ≥ 18 :=
by sorry

end NUMINAMATH_CALUDE_min_sum_squares_l4149_414902


namespace NUMINAMATH_CALUDE_minimum_value_and_tangent_line_l4149_414926

noncomputable def f (a b x : ℝ) : ℝ := a * Real.exp x + 1 / (a * Real.exp x) + b

theorem minimum_value_and_tangent_line (a b : ℝ) (ha : a > 0) :
  (∀ x ≥ 0, f a b x ≥ (if a ≥ 1 then a + 1/a + b else b + 2)) ∧
  (∃ x ≥ 0, f a b x = (if a ≥ 1 then a + 1/a + b else b + 2)) ∧
  ((f a b 2 = 3 ∧ (deriv (f a b)) 2 = 3/2) → a = 2 / Real.exp 2 ∧ b = 1/2) :=
sorry

end NUMINAMATH_CALUDE_minimum_value_and_tangent_line_l4149_414926


namespace NUMINAMATH_CALUDE_cost_price_calculation_l4149_414910

/-- Given an article sold at a 30% profit with a selling price of 364,
    prove that the cost price of the article is 280. -/
theorem cost_price_calculation (selling_price : ℝ) (profit_percentage : ℝ) 
    (h1 : selling_price = 364)
    (h2 : profit_percentage = 0.30) : 
  ∃ (cost_price : ℝ), cost_price = 280 ∧ 
    selling_price = cost_price * (1 + profit_percentage) := by
  sorry

end NUMINAMATH_CALUDE_cost_price_calculation_l4149_414910


namespace NUMINAMATH_CALUDE_max_min_sum_implies_a_value_l4149_414983

def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*x + a

theorem max_min_sum_implies_a_value (a : ℝ) :
  (∃ (max min : ℝ), 
    (∀ x ∈ Set.Icc 2 3, f a x ≤ max) ∧
    (∃ y ∈ Set.Icc 2 3, f a y = max) ∧
    (∀ x ∈ Set.Icc 2 3, min ≤ f a x) ∧
    (∃ y ∈ Set.Icc 2 3, f a y = min) ∧
    max + min = 5) →
  a = 1 := by
sorry

end NUMINAMATH_CALUDE_max_min_sum_implies_a_value_l4149_414983


namespace NUMINAMATH_CALUDE_figure_area_proof_l4149_414900

theorem figure_area_proof (r1_height r1_width r2_height r2_width r3_height r3_width r4_height r4_width : ℕ) 
  (h1 : r1_height = 6 ∧ r1_width = 5)
  (h2 : r2_height = 3 ∧ r2_width = 5)
  (h3 : r3_height = 3 ∧ r3_width = 10)
  (h4 : r4_height = 8 ∧ r4_width = 2) :
  r1_height * r1_width + r2_height * r2_width + r3_height * r3_width + r4_height * r4_width = 91 := by
  sorry

#check figure_area_proof

end NUMINAMATH_CALUDE_figure_area_proof_l4149_414900


namespace NUMINAMATH_CALUDE_equal_intercept_line_theorem_tangent_circle_theorem_l4149_414970

-- Define the line l
def line_l (x y : ℝ) : Prop := 3 * x + 4 * y - 12 = 0

-- Define point P
def point_P : ℝ × ℝ := (1, 2)

-- Define the line with equal intercepts passing through P
def equal_intercept_line (x y : ℝ) : Prop := x + y = 3

-- Define the circle
def tangent_circle (x y : ℝ) : Prop := (x - 1)^2 + (y - 1)^2 = 1

-- Theorem for the line with equal intercepts
theorem equal_intercept_line_theorem :
  ∃ (a : ℝ), a ≠ 0 ∧ 
  (∀ x y : ℝ, equal_intercept_line x y ↔ x / a + y / a = 1) ∧
  equal_intercept_line point_P.1 point_P.2 :=
sorry

-- Theorem for the tangent circle
theorem tangent_circle_theorem :
  ∃ A B : ℝ × ℝ,
  (line_l A.1 A.2 ∧ A.2 = 0) ∧
  (line_l B.1 B.2 ∧ B.1 = 0) ∧
  (∀ x y : ℝ, tangent_circle x y →
    (x = 0 ∨ y = 0 ∨ line_l x y)) :=
sorry

end NUMINAMATH_CALUDE_equal_intercept_line_theorem_tangent_circle_theorem_l4149_414970


namespace NUMINAMATH_CALUDE_triangle_triple_sine_sum_l4149_414924

theorem triangle_triple_sine_sum (A B C : ℝ) : 
  A + B + C = π ∧ (A = π/3 ∨ B = π/3 ∨ C = π/3) → 
  Real.sin (3*A) + Real.sin (3*B) + Real.sin (3*C) = 0 := by
  sorry

end NUMINAMATH_CALUDE_triangle_triple_sine_sum_l4149_414924


namespace NUMINAMATH_CALUDE_shopkeeper_gain_percentage_l4149_414958

/-- Calculates the gain percentage given the cost price and selling price -/
def gain_percentage (cost_price selling_price : ℚ) : ℚ :=
  (selling_price - cost_price) / cost_price * 100

/-- Theorem stating that for an article with a cost price of 220 and selling price of 264,
    the gain percentage is 20% -/
theorem shopkeeper_gain_percentage :
  let cost_price : ℚ := 220
  let selling_price : ℚ := 264
  gain_percentage cost_price selling_price = 20 := by sorry

end NUMINAMATH_CALUDE_shopkeeper_gain_percentage_l4149_414958


namespace NUMINAMATH_CALUDE_ellipse_sum_l4149_414912

/-- Represents an ellipse with center (h, k) and semi-axes a and b -/
structure Ellipse where
  h : ℝ
  k : ℝ
  a : ℝ
  b : ℝ

/-- The equation of the ellipse -/
def ellipse_equation (e : Ellipse) (x y : ℝ) : Prop :=
  (x - e.h)^2 / e.a^2 + (y - e.k)^2 / e.b^2 = 1

theorem ellipse_sum (e : Ellipse) :
  e.h = 5 ∧ e.k = -3 ∧ e.a = 7 ∧ e.b = 4 →
  e.h + e.k + e.a + e.b = 13 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_sum_l4149_414912


namespace NUMINAMATH_CALUDE_units_digit_of_special_number_l4149_414901

def is_product_of_one_digit_numbers (n : ℕ) : Prop :=
  ∃ (factors : List ℕ), (factors.all (λ x => x > 0 ∧ x < 10)) ∧ 
    (factors.prod = n)

def digit_product (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) * digit_product (n / 10)

theorem units_digit_of_special_number (n : ℕ) :
  n > 10 ∧ 
  is_product_of_one_digit_numbers n ∧ 
  Odd (digit_product n) →
  n % 10 = 5 := by
sorry

end NUMINAMATH_CALUDE_units_digit_of_special_number_l4149_414901


namespace NUMINAMATH_CALUDE_parabola_with_focus_on_line_l4149_414923

-- Define the line on which the focus lies
def focus_line (x y : ℝ) : Prop := 3 * x - 4 * y - 12 = 0

-- Define the standard equations of the parabolas
def parabola_eq1 (x y : ℝ) : Prop := y^2 = 16 * x
def parabola_eq2 (x y : ℝ) : Prop := x^2 = -12 * y

-- Theorem statement
theorem parabola_with_focus_on_line :
  ∀ (x y : ℝ), focus_line x y → (parabola_eq1 x y ∨ parabola_eq2 x y) :=
sorry

end NUMINAMATH_CALUDE_parabola_with_focus_on_line_l4149_414923


namespace NUMINAMATH_CALUDE_geometric_series_first_term_l4149_414915

theorem geometric_series_first_term 
  (r : ℝ) 
  (S : ℝ) 
  (h1 : r = 1/5)
  (h2 : S = 100)
  (h3 : S = a / (1 - r)) :
  a = 80 := by
sorry

end NUMINAMATH_CALUDE_geometric_series_first_term_l4149_414915


namespace NUMINAMATH_CALUDE_line_one_point_not_always_tangent_l4149_414913

-- Define a curve as a set of points in 2D space
def Curve := Set (ℝ × ℝ)

-- Define a line as a set of points in 2D space
def Line := Set (ℝ × ℝ)

-- Define what it means for a line to be tangent to a curve
def IsTangent (l : Line) (c : Curve) : Prop := sorry

-- Define what it means for a line to have only one common point with a curve
def HasOneCommonPoint (l : Line) (c : Curve) : Prop := sorry

-- Theorem statement
theorem line_one_point_not_always_tangent :
  ∃ (l : Line) (c : Curve), HasOneCommonPoint l c ∧ ¬IsTangent l c := by sorry

end NUMINAMATH_CALUDE_line_one_point_not_always_tangent_l4149_414913


namespace NUMINAMATH_CALUDE_cyclist_speed_solution_l4149_414961

/-- Represents the speeds and distance of two cyclists traveling in opposite directions. -/
structure CyclistProblem where
  slower_speed : ℝ
  time : ℝ
  distance_apart : ℝ
  speed_difference : ℝ

/-- Calculates the total distance traveled by both cyclists. -/
def total_distance (p : CyclistProblem) : ℝ :=
  p.time * (2 * p.slower_speed + p.speed_difference)

/-- Theorem stating the conditions and solution for the cyclist problem. -/
theorem cyclist_speed_solution (p : CyclistProblem) 
  (h1 : p.time = 6)
  (h2 : p.distance_apart = 246)
  (h3 : p.speed_difference = 5) :
  p.slower_speed = 18 ∧ p.slower_speed + p.speed_difference = 23 :=
by
  sorry

#check cyclist_speed_solution

end NUMINAMATH_CALUDE_cyclist_speed_solution_l4149_414961


namespace NUMINAMATH_CALUDE_sqrt_sum_equation_solutions_l4149_414941

theorem sqrt_sum_equation_solutions :
  ∀ x : ℝ, (Real.sqrt ((3 + Real.sqrt 5) ^ x) + Real.sqrt ((3 - Real.sqrt 5) ^ x) = 6) ↔ (x = 2 ∨ x = -2) :=
by sorry

end NUMINAMATH_CALUDE_sqrt_sum_equation_solutions_l4149_414941


namespace NUMINAMATH_CALUDE_track_length_l4149_414962

theorem track_length : ∀ (x : ℝ), 
  (x > 0) →  -- track length is positive
  (120 / (x/2 - 120) = (x/2 + 50) / (3*x/2 - 170)) →  -- ratio of distances is constant
  x = 418 := by
sorry

end NUMINAMATH_CALUDE_track_length_l4149_414962


namespace NUMINAMATH_CALUDE_paiges_files_l4149_414954

theorem paiges_files (deleted_files : ℕ) (files_per_folder : ℕ) (num_folders : ℕ) :
  deleted_files = 9 →
  files_per_folder = 6 →
  num_folders = 3 →
  deleted_files + (files_per_folder * num_folders) = 27 :=
by
  sorry

end NUMINAMATH_CALUDE_paiges_files_l4149_414954


namespace NUMINAMATH_CALUDE_shaded_area_of_intersecting_diameters_l4149_414960

theorem shaded_area_of_intersecting_diameters (r : ℝ) (θ : ℝ) : 
  r = 6 → θ = π / 3 → 2 * (θ / (2 * π)) * (π * r^2) = 12 * π := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_of_intersecting_diameters_l4149_414960


namespace NUMINAMATH_CALUDE_invoice_error_correction_l4149_414974

/-- Two-digit number -/
def TwoDigitNumber (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

/-- The proposition to be proved -/
theorem invoice_error_correction (x y : ℕ) 
  (hx : TwoDigitNumber x) (hy : TwoDigitNumber y)
  (h_diff : 100 * x + y - (100 * y + x) = 3654) :
  x = 63 ∧ y = 26 := by
  sorry

end NUMINAMATH_CALUDE_invoice_error_correction_l4149_414974


namespace NUMINAMATH_CALUDE_find_s_value_l4149_414999

/-- Given a relationship between R, S, and T, prove the value of S for specific R and T -/
theorem find_s_value (c : ℝ) (R S T : ℝ → ℝ) :
  (∀ x, R x = c * (S x / T x)) →  -- Relationship between R, S, and T
  R 1 = 2 →                       -- Initial condition for R
  S 1 = 1/2 →                     -- Initial condition for S
  T 1 = 4/3 →                     -- Initial condition for T
  R 2 = Real.sqrt 75 →            -- New condition for R
  T 2 = Real.sqrt 32 →            -- New condition for T
  S 2 = 45/4 :=                   -- Conclusion: value of S
by sorry

end NUMINAMATH_CALUDE_find_s_value_l4149_414999


namespace NUMINAMATH_CALUDE_problem_solution_l4149_414916

theorem problem_solution (w x y : ℝ) 
  (h1 : 7 / w + 7 / x = 7 / y) 
  (h2 : w * x = y) 
  (h3 : (w + x) / 2 = 0.5) : 
  y = 0.25 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l4149_414916


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_difference_l4149_414979

theorem quadratic_equation_solution_difference : ∃ (x₁ x₂ : ℝ),
  (2 * x₁^2 - 6 * x₁ + 18 = 2 * x₁ + 82) ∧
  (2 * x₂^2 - 6 * x₂ + 18 = 2 * x₂ + 82) ∧
  x₁ ≠ x₂ ∧
  |x₁ - x₂| = 12 :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_difference_l4149_414979


namespace NUMINAMATH_CALUDE_discount_calculation_l4149_414903

/-- Calculates the final amount paid after applying a discount -/
def finalAmount (initialAmount : ℕ) (discountPer100 : ℕ) : ℕ :=
  let fullDiscountUnits := initialAmount / 100
  let totalDiscount := fullDiscountUnits * discountPer100
  initialAmount - totalDiscount

/-- Theorem stating that for a $250 purchase with a $10 discount per $100 spent, the final amount is $230 -/
theorem discount_calculation :
  finalAmount 250 10 = 230 := by
  sorry

end NUMINAMATH_CALUDE_discount_calculation_l4149_414903


namespace NUMINAMATH_CALUDE_dave_trips_l4149_414914

/-- The number of trips Dave needs to make to carry all trays -/
def number_of_trips (trays_per_trip : ℕ) (trays_table1 : ℕ) (trays_table2 : ℕ) : ℕ :=
  (trays_table1 + trays_table2 + trays_per_trip - 1) / trays_per_trip

theorem dave_trips :
  number_of_trips 9 17 55 = 8 :=
by sorry

end NUMINAMATH_CALUDE_dave_trips_l4149_414914


namespace NUMINAMATH_CALUDE_binomial_10_9_l4149_414928

theorem binomial_10_9 : (10 : ℕ).choose 9 = 10 := by sorry

end NUMINAMATH_CALUDE_binomial_10_9_l4149_414928


namespace NUMINAMATH_CALUDE_expected_value_of_three_marbles_l4149_414953

def marbles : Finset ℕ := {1, 2, 3, 4, 5, 6}

def sumOfThree (s : Finset ℕ) : ℕ := s.sum id

def allCombinations : Finset (Finset ℕ) :=
  marbles.powerset.filter (λ s => s.card = 3)

def expectedValue : ℚ :=
  (allCombinations.sum sumOfThree) / allCombinations.card

theorem expected_value_of_three_marbles :
  expectedValue = 21/2 := by sorry

end NUMINAMATH_CALUDE_expected_value_of_three_marbles_l4149_414953


namespace NUMINAMATH_CALUDE_tickets_at_door_correct_l4149_414959

/-- Represents the number of tickets sold at the door -/
def tickets_at_door : ℕ := 672

/-- Represents the number of advanced tickets sold -/
def advanced_tickets : ℕ := 800 - tickets_at_door

/-- The total number of tickets sold -/
def total_tickets : ℕ := 800

/-- The price of an advanced ticket in cents -/
def advanced_price : ℕ := 1450

/-- The price of a ticket at the door in cents -/
def door_price : ℕ := 2200

/-- The total amount of money taken in cents -/
def total_revenue : ℕ := 1664000

theorem tickets_at_door_correct :
  (advanced_tickets * advanced_price + tickets_at_door * door_price = total_revenue) ∧
  (advanced_tickets + tickets_at_door = total_tickets) := by
  sorry

end NUMINAMATH_CALUDE_tickets_at_door_correct_l4149_414959


namespace NUMINAMATH_CALUDE_smallest_transformed_sum_l4149_414919

/-- The number of faces on a standard die -/
def facesOnDie : ℕ := 6

/-- The target sum we want to achieve -/
def targetSum : ℕ := 1994

/-- The function to calculate the transformed sum given the number of dice -/
def transformedSum (n : ℕ) : ℕ := 7 * n - targetSum

/-- The theorem stating the smallest possible value of the transformed sum -/
theorem smallest_transformed_sum :
  ∃ (n : ℕ), 
    (n * facesOnDie ≥ targetSum) ∧ 
    (∀ m : ℕ, m * facesOnDie ≥ targetSum → n ≤ m) ∧
    (transformedSum n = 337) := by
  sorry

end NUMINAMATH_CALUDE_smallest_transformed_sum_l4149_414919


namespace NUMINAMATH_CALUDE_range_of_a_l4149_414949

/-- A function f(x) with parameter a -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*a*x + 15 - 2*a

/-- Predicate to check if there are exactly two positive integers in an open interval -/
def exactly_two_positive_integers (lower upper : ℝ) : Prop :=
  ∃ (n m : ℕ), n < m ∧ 
    (∀ (k : ℕ), lower < k ∧ k < upper ↔ k = n ∨ k = m)

/-- Theorem stating the range of a given the conditions -/
theorem range_of_a (a : ℝ) :
  (∃ (x₁ x₂ : ℝ), x₁ < x₂ ∧ 
    f a x₁ = 0 ∧ f a x₂ = 0 ∧ 
    exactly_two_positive_integers x₁ x₂) →
  (31/10 < a ∧ a ≤ 19/6) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l4149_414949


namespace NUMINAMATH_CALUDE_no_prime_root_solution_l4149_414980

/-- A quadratic equation x^2 - 67x + k = 0 with prime roots -/
def has_prime_roots (k : ℤ) : Prop :=
  ∃ p q : ℕ, Nat.Prime p ∧ Nat.Prime q ∧ 
  (p : ℤ) + q = 67 ∧ (p : ℤ) * q = k

/-- There are no integer values of k for which the equation x^2 - 67x + k = 0 has two prime roots -/
theorem no_prime_root_solution : ¬∃ k : ℤ, has_prime_roots k := by
  sorry

end NUMINAMATH_CALUDE_no_prime_root_solution_l4149_414980


namespace NUMINAMATH_CALUDE_division_fraction_problem_l4149_414944

theorem division_fraction_problem : (1 / 60) / ((2 / 3) - (1 / 5) - (2 / 5)) = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_division_fraction_problem_l4149_414944


namespace NUMINAMATH_CALUDE_power_function_domain_and_oddness_l4149_414994

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def has_real_domain (f : ℝ → ℝ) : Prop :=
  ∀ x, ∃ y, f x = y

theorem power_function_domain_and_oddness (a : ℝ) :
  a ∈ ({-1, 0, 1/2, 1, 2, 3} : Set ℝ) →
  (has_real_domain (fun x ↦ x^a) ∧ is_odd_function (fun x ↦ x^a)) ↔ (a = 1 ∨ a = 3) :=
sorry

end NUMINAMATH_CALUDE_power_function_domain_and_oddness_l4149_414994


namespace NUMINAMATH_CALUDE_right_triangle_and_multiplicative_inverse_l4149_414991

theorem right_triangle_and_multiplicative_inverse :
  (30^2 + 272^2 = 278^2) ∧
  ((550 * 6) % 4079 = 1) ∧
  (0 ≤ 6 ∧ 6 < 4079) := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_and_multiplicative_inverse_l4149_414991


namespace NUMINAMATH_CALUDE_herd_division_l4149_414996

theorem herd_division (herd : ℚ) : 
  (1/3 : ℚ) + (1/6 : ℚ) + (1/9 : ℚ) + (8 : ℚ) / herd = 1 → 
  herd = 144/7 := by
  sorry

end NUMINAMATH_CALUDE_herd_division_l4149_414996


namespace NUMINAMATH_CALUDE_carols_peanuts_l4149_414963

/-- Represents the number of peanuts Carol's father gave her -/
def peanuts_from_father (initial : ℕ) (final : ℕ) : ℕ := final - initial

theorem carols_peanuts : peanuts_from_father 2 7 = 5 := by
  sorry

end NUMINAMATH_CALUDE_carols_peanuts_l4149_414963
