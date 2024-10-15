import Mathlib

namespace NUMINAMATH_CALUDE_lefty_points_lefty_scored_20_points_l2586_258687

theorem lefty_points : ℝ → Prop :=
  fun L : ℝ =>
    let righty : ℝ := L / 2
    let third_teammate : ℝ := 3 * L
    let total_points : ℝ := L + righty + third_teammate
    let average_points : ℝ := total_points / 3
    average_points = 30 → L = 20

-- Proof
theorem lefty_scored_20_points : ∃ L : ℝ, lefty_points L :=
  sorry

end NUMINAMATH_CALUDE_lefty_points_lefty_scored_20_points_l2586_258687


namespace NUMINAMATH_CALUDE_line_parameterization_l2586_258632

/-- Given a line y = 2x - 10 parameterized by (x,y) = (g(t), 10t - 4), prove that g(t) = 5t + 3 -/
theorem line_parameterization (g : ℝ → ℝ) : 
  (∀ t, 2 * g t - 10 = 10 * t - 4) → 
  (∀ t, g t = 5 * t + 3) := by
sorry

end NUMINAMATH_CALUDE_line_parameterization_l2586_258632


namespace NUMINAMATH_CALUDE_g_of_8_eq_neg_46_l2586_258656

/-- A function g : ℝ → ℝ satisfying the given functional equation for all real x and y -/
def g_equation (g : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, g x + g (3*x + y) + 7*x*y = g (4*x - 2*y) + 3*x^2 + 2

/-- Theorem stating that if g satisfies the functional equation, then g(8) = -46 -/
theorem g_of_8_eq_neg_46 (g : ℝ → ℝ) (h : g_equation g) : g 8 = -46 := by
  sorry

end NUMINAMATH_CALUDE_g_of_8_eq_neg_46_l2586_258656


namespace NUMINAMATH_CALUDE_least_positive_integer_for_multiple_of_five_l2586_258613

theorem least_positive_integer_for_multiple_of_five : 
  ∃ (n : ℕ), n > 0 ∧ (525 + n) % 5 = 0 ∧ ∀ (m : ℕ), m > 0 ∧ (525 + m) % 5 = 0 → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_least_positive_integer_for_multiple_of_five_l2586_258613


namespace NUMINAMATH_CALUDE_rectangles_on_4x4_grid_l2586_258669

/-- The number of lines in a 4x4 grid (both horizontal and vertical) -/
def gridLines : ℕ := 5

/-- The number of lines needed to form a rectangle (both horizontal and vertical) -/
def linesNeeded : ℕ := 2

/-- The number of ways to choose horizontal lines for a rectangle -/
def horizontalChoices : ℕ := Nat.choose gridLines linesNeeded

/-- The number of ways to choose vertical lines for a rectangle -/
def verticalChoices : ℕ := Nat.choose gridLines linesNeeded

/-- Theorem: The number of rectangles on a 4x4 grid is 100 -/
theorem rectangles_on_4x4_grid : horizontalChoices * verticalChoices = 100 := by
  sorry


end NUMINAMATH_CALUDE_rectangles_on_4x4_grid_l2586_258669


namespace NUMINAMATH_CALUDE_cube_difference_l2586_258699

theorem cube_difference (x y : ℝ) (h1 : x + y = 8) (h2 : 3 * x + y = 14) :
  x^3 - y^3 = -98 := by
  sorry

end NUMINAMATH_CALUDE_cube_difference_l2586_258699


namespace NUMINAMATH_CALUDE_simplify_algebraic_expression_l2586_258670

theorem simplify_algebraic_expression (a b : ℝ) : 5*a*b - 7*a*b + 3*a*b = a*b := by
  sorry

end NUMINAMATH_CALUDE_simplify_algebraic_expression_l2586_258670


namespace NUMINAMATH_CALUDE_circle_diameter_from_area_l2586_258682

theorem circle_diameter_from_area (A : ℝ) (d : ℝ) :
  A = 225 * Real.pi → d = 30 → A = Real.pi * (d / 2)^2 :=
by sorry

end NUMINAMATH_CALUDE_circle_diameter_from_area_l2586_258682


namespace NUMINAMATH_CALUDE_train_speed_l2586_258660

/-- The speed of a train given its length and time to cross a pole -/
theorem train_speed (length : ℝ) (time : ℝ) (h1 : length = 800) (h2 : time = 10) :
  length / time = 80 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_l2586_258660


namespace NUMINAMATH_CALUDE_octal_127_equals_87_l2586_258695

-- Define the octal number as a list of digits
def octal_127 : List Nat := [1, 2, 7]

-- Function to convert octal to decimal
def octal_to_decimal (digits : List Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * (8 ^ i)) 0

-- Theorem statement
theorem octal_127_equals_87 :
  octal_to_decimal octal_127 = 87 := by
  sorry

end NUMINAMATH_CALUDE_octal_127_equals_87_l2586_258695


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l2586_258601

theorem sufficient_not_necessary (x y : ℝ) :
  (∀ x y : ℝ, x > 1 ∧ y > 1 → x + y > 2) ∧
  (∃ x y : ℝ, x + y > 2 ∧ ¬(x > 1 ∧ y > 1)) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l2586_258601


namespace NUMINAMATH_CALUDE_shaded_area_of_square_grid_l2586_258637

/-- The area of a square composed of 25 congruent smaller squares, 
    where the diagonal of the larger square is 10 cm, is 50 square cm. -/
theorem shaded_area_of_square_grid (d : ℝ) (n : ℕ) : 
  d = 10 → n = 25 → (d^2 / 2) * (n / n^(1/2) : ℝ)^2 = 50 := by sorry

end NUMINAMATH_CALUDE_shaded_area_of_square_grid_l2586_258637


namespace NUMINAMATH_CALUDE_equation_real_roots_range_l2586_258606

theorem equation_real_roots_range (a : ℝ) : 
  (∀ x : ℝ, (2 + 3*a) / (5 - a) > 0) ↔ a ∈ Set.Ioo (-2/3 : ℝ) 5 := by sorry

end NUMINAMATH_CALUDE_equation_real_roots_range_l2586_258606


namespace NUMINAMATH_CALUDE_sixth_power_to_third_power_l2586_258647

theorem sixth_power_to_third_power (x : ℝ) (h : 728 = x^6 + 1/x^6) : 
  x^3 + 1/x^3 = Real.sqrt 730 := by
  sorry

end NUMINAMATH_CALUDE_sixth_power_to_third_power_l2586_258647


namespace NUMINAMATH_CALUDE_bobby_candy_l2586_258676

def candy_problem (initial : ℕ) (final : ℕ) (second_round : ℕ) : Prop :=
  ∃ (first_round : ℕ), 
    initial - (first_round + second_round) = final ∧
    first_round + second_round < initial

theorem bobby_candy : candy_problem 21 7 9 → ∃ (x : ℕ), x = 5 := by
  sorry

end NUMINAMATH_CALUDE_bobby_candy_l2586_258676


namespace NUMINAMATH_CALUDE_minimum_point_of_translated_graph_l2586_258623

def f (x : ℝ) : ℝ := |x + 4| - 9

theorem minimum_point_of_translated_graph :
  ∃! (x y : ℝ), f x = y ∧ ∀ z : ℝ, f z ≥ y ∧ (x, y) = (-4, -9) := by sorry

end NUMINAMATH_CALUDE_minimum_point_of_translated_graph_l2586_258623


namespace NUMINAMATH_CALUDE_quadratic_root_zero_l2586_258684

theorem quadratic_root_zero (m : ℝ) : 
  (∃ x, (m - 1) * x^2 + 2 * x + m^2 - 1 = 0) ∧ 
  ((m - 1) * 0^2 + 2 * 0 + m^2 - 1 = 0) → 
  m = -1 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_zero_l2586_258684


namespace NUMINAMATH_CALUDE_quadrilateral_existence_l2586_258653

structure Plane where
  dummy : Unit

structure Line where
  dummy : Unit

structure Point where
  dummy : Unit

def lies_in (p : Point) (plane : Plane) : Prop := sorry

def not_in (p : Point) (plane : Plane) : Prop := sorry

def on_line (p : Point) (l : Line) : Prop := sorry

def intersect (p1 p2 : Plane) (l : Line) : Prop := sorry

def parallel (l1 l2 : Line) : Prop := sorry

def perpendicular (l1 l2 : Line) : Prop := sorry

def length_eq (s1 s2 : Point × Point) : Prop := sorry

def has_inscribed_circle (q : Point × Point × Point × Point) : Prop := sorry

theorem quadrilateral_existence 
  (P Q : Plane) (p : Line) (A C : Point) :
  intersect P Q p →
  lies_in A P →
  not_in A Q →
  lies_in C Q →
  not_in C P →
  ¬on_line A p →
  ¬on_line C p →
  ∃ (B D E : Point) (AB CD CE : Line),
    lies_in B P ∧
    lies_in D Q ∧
    parallel AB CD ∧
    parallel AB p ∧
    parallel CD p ∧
    perpendicular CE AB ∧
    length_eq (A, D) (B, C) ∧
    has_inscribed_circle (A, B, C, D) ∧
    (∃ (AE CE : Point × Point),
      (length_eq AE CE → ∃! (ABCD : Point × Point × Point × Point), ABCD = (A, B, C, D)) ∧
      (∀ x y, length_eq x y → x = AE → y = CE → 
        (∃ (ABCD1 ABCD2 : Point × Point × Point × Point), ABCD1 ≠ ABCD2 ∧ 
          (ABCD1 = (A, B, C, D) ∨ ABCD2 = (A, B, C, D))))) :=
by sorry

end NUMINAMATH_CALUDE_quadrilateral_existence_l2586_258653


namespace NUMINAMATH_CALUDE_min_value_sum_reciprocals_l2586_258614

theorem min_value_sum_reciprocals (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : a^2 + 4*b^2 + 9*c^2 = 4*b + 12*c - 2) :
  (1/a + 2/b + 3/c) ≥ 6 :=
sorry

end NUMINAMATH_CALUDE_min_value_sum_reciprocals_l2586_258614


namespace NUMINAMATH_CALUDE_correct_regression_equation_l2586_258605

/-- Represents the selling price of a product in yuan per piece -/
def SellingPrice : Type := ℝ

/-- Represents the sales volume of a product in pieces -/
def SalesVolume : Type := ℝ

/-- Represents a regression equation for sales volume based on selling price -/
structure RegressionEquation where
  slope : ℝ
  intercept : ℝ

/-- Indicates that two variables are negatively correlated -/
def NegativelyCorrelated (x : Type) (y : Type) : Prop := sorry

/-- Checks if a regression equation is valid for negatively correlated variables -/
def IsValidRegression (eq : RegressionEquation) (x : Type) (y : Type) : Prop := 
  NegativelyCorrelated x y → eq.slope < 0

/-- The correct regression equation for the given problem -/
def CorrectEquation : RegressionEquation := { slope := -2, intercept := 100 }

/-- Theorem stating that the CorrectEquation is valid for the given problem -/
theorem correct_regression_equation : 
  IsValidRegression CorrectEquation SellingPrice SalesVolume := sorry

end NUMINAMATH_CALUDE_correct_regression_equation_l2586_258605


namespace NUMINAMATH_CALUDE_bus_time_calculation_l2586_258615

def wake_up_time : ℕ := 6 * 60 + 45
def bus_departure_time : ℕ := 7 * 60 + 15
def class_duration : ℕ := 45
def num_classes : ℕ := 7
def lunch_duration : ℕ := 20
def science_lab_duration : ℕ := 60
def additional_time : ℕ := 90
def arrival_time : ℕ := 15 * 60 + 50

def total_school_time : ℕ := 
  num_classes * class_duration + lunch_duration + science_lab_duration + additional_time

def total_away_time : ℕ := arrival_time - bus_departure_time

theorem bus_time_calculation : 
  total_away_time - total_school_time = 30 := by sorry

end NUMINAMATH_CALUDE_bus_time_calculation_l2586_258615


namespace NUMINAMATH_CALUDE_special_circle_equation_l2586_258602

/-- A circle passing through two points with a specific sum of intercepts -/
structure SpecialCircle where
  -- The circle passes through (4,2) and (-2,-6)
  passes_through_1 : x^2 + y^2 + D*x + E*y + F = 0 → 4^2 + 2^2 + 4*D + 2*E + F = 0
  passes_through_2 : x^2 + y^2 + D*x + E*y + F = 0 → (-2)^2 + (-6)^2 + (-2)*D + (-6)*E + F = 0
  -- Sum of intercepts is -2
  sum_of_intercepts : D + E = 2

/-- The standard equation of the special circle -/
def standard_equation (c : SpecialCircle) : Prop :=
  ∃ (x y : ℝ), (x - 1)^2 + (y + 2)^2 = 25

/-- Theorem stating that the given circle has the specified standard equation -/
theorem special_circle_equation (c : SpecialCircle) : standard_equation c :=
  sorry

end NUMINAMATH_CALUDE_special_circle_equation_l2586_258602


namespace NUMINAMATH_CALUDE_simplify_expression_l2586_258677

theorem simplify_expression (x y z : ℝ) (hx : x ≠ 0) (hz : z ≠ 0) (hxy : x^2 ≠ y^2) :
  (x^2 - y^2)⁻¹ * (x⁻¹ - z⁻¹) = (z - x) * x⁻¹ * z⁻¹ * (x^2 - y^2)⁻¹ :=
by sorry

end NUMINAMATH_CALUDE_simplify_expression_l2586_258677


namespace NUMINAMATH_CALUDE_greatest_common_factor_40_120_100_l2586_258648

theorem greatest_common_factor_40_120_100 : Nat.gcd 40 (Nat.gcd 120 100) = 20 := by
  sorry

end NUMINAMATH_CALUDE_greatest_common_factor_40_120_100_l2586_258648


namespace NUMINAMATH_CALUDE_probability_sum_six_l2586_258683

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


end NUMINAMATH_CALUDE_probability_sum_six_l2586_258683


namespace NUMINAMATH_CALUDE_sara_is_45_inches_tall_l2586_258672

-- Define the heights as natural numbers
def roy_height : ℕ := 36
def joe_height : ℕ := roy_height + 3
def sara_height : ℕ := joe_height + 6

-- Theorem statement
theorem sara_is_45_inches_tall : sara_height = 45 := by
  sorry

end NUMINAMATH_CALUDE_sara_is_45_inches_tall_l2586_258672


namespace NUMINAMATH_CALUDE_steve_bench_wood_length_l2586_258641

/-- Calculates the total length of wood needed for Steve's bench. -/
theorem steve_bench_wood_length : 
  let long_pieces : ℕ := 6
  let long_length : ℕ := 4
  let short_pieces : ℕ := 2
  let short_length : ℕ := 2
  long_pieces * long_length + short_pieces * short_length = 28 :=
by sorry

end NUMINAMATH_CALUDE_steve_bench_wood_length_l2586_258641


namespace NUMINAMATH_CALUDE_no_natural_solution_l2586_258603

theorem no_natural_solution : ¬∃ (x y : ℕ), 2 * x + 3 * y = 6 := by sorry

end NUMINAMATH_CALUDE_no_natural_solution_l2586_258603


namespace NUMINAMATH_CALUDE_tan_theta_value_l2586_258626

/-- If the terminal side of angle θ passes through the point (-√3/2, 1/2), then tan θ = -√3/3 -/
theorem tan_theta_value (θ : Real) (h : ∃ (t : Real), t > 0 ∧ t * (-Real.sqrt 3 / 2) = Real.cos θ ∧ t * (1 / 2) = Real.sin θ) : 
  Real.tan θ = -Real.sqrt 3 / 3 := by
sorry

end NUMINAMATH_CALUDE_tan_theta_value_l2586_258626


namespace NUMINAMATH_CALUDE_solution_set_characterization_l2586_258652

-- Define the properties of function f
def IsOddFunction (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def IsDecreasingFunction (f : ℝ → ℝ) : Prop := ∀ x y, x < y → f x > f y

-- Define the solution set
def SolutionSet (f : ℝ → ℝ) : Set ℝ := {a | f (a^2) + f (2*a) > 0}

-- State the theorem
theorem solution_set_characterization 
  (f : ℝ → ℝ) 
  (h_odd : IsOddFunction f) 
  (h_decreasing : IsDecreasingFunction f) : 
  SolutionSet f = Set.Ioo (-2) 0 := by sorry

end NUMINAMATH_CALUDE_solution_set_characterization_l2586_258652


namespace NUMINAMATH_CALUDE_average_problem_l2586_258664

theorem average_problem (t b c : ℝ) (h : (t + b + c + 29) / 4 = 15) :
  (t + b + c + 14 + 15) / 5 = 12 := by
  sorry

end NUMINAMATH_CALUDE_average_problem_l2586_258664


namespace NUMINAMATH_CALUDE_inequality_implication_l2586_258658

theorem inequality_implication (p q r : ℝ) 
  (hr : r > 0) (hpq : p * q ≠ 0) (hineq : p * r < q * r) : 
  1 < q / p :=
sorry

end NUMINAMATH_CALUDE_inequality_implication_l2586_258658


namespace NUMINAMATH_CALUDE_number_equation_solution_l2586_258673

theorem number_equation_solution : 
  ∃ x : ℝ, (3 * x = 2 * x - 7) ∧ (x = -7) := by sorry

end NUMINAMATH_CALUDE_number_equation_solution_l2586_258673


namespace NUMINAMATH_CALUDE_tax_rate_calculation_l2586_258685

/-- Proves that the tax rate is 30% given the specified conditions --/
theorem tax_rate_calculation (total_cost tax_free_cost : ℝ) 
  (h1 : total_cost = 20)
  (h2 : tax_free_cost = 14.7)
  (h3 : (total_cost - tax_free_cost) * 0.3 = (total_cost - tax_free_cost) * (30 / 100)) : 
  (((total_cost - tax_free_cost) * 0.3) / (total_cost - tax_free_cost)) * 100 = 30 := by
  sorry

#check tax_rate_calculation

end NUMINAMATH_CALUDE_tax_rate_calculation_l2586_258685


namespace NUMINAMATH_CALUDE_square_side_length_l2586_258674

/-- The area enclosed between the circumferences of four circles described about the corners of a square -/
def enclosed_area : ℝ := 42.06195997410015

/-- Theorem: Given four equal circles described about the four corners of a square, 
    each touching two others, with the area enclosed between the circumferences 
    of the circles being 42.06195997410015 cm², the length of a side of the square is 14 cm. -/
theorem square_side_length (r : ℝ) (h1 : r > 0) 
  (h2 : 4 * r^2 - Real.pi * r^2 = enclosed_area) : 
  2 * r = 14 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_l2586_258674


namespace NUMINAMATH_CALUDE_exists_composite_in_sequence_l2586_258689

-- Define the sequence type
def RecurrenceSequence := ℕ → ℕ

-- Define the recurrence relation
def SatisfiesRecurrence (a : RecurrenceSequence) : Prop :=
  ∀ n : ℕ, (a (n + 1) = 2 * a n + 1) ∨ (a (n + 1) = 2 * a n - 1)

-- Define a non-constant sequence
def NonConstant (a : RecurrenceSequence) : Prop :=
  ∃ m n : ℕ, a m ≠ a n

-- Define a positive sequence
def Positive (a : RecurrenceSequence) : Prop :=
  ∀ n : ℕ, a n > 0

-- Define a composite number
def Composite (n : ℕ) : Prop :=
  n > 1 ∧ ∃ k : ℕ, 1 < k ∧ k < n ∧ n % k = 0

-- The main theorem
theorem exists_composite_in_sequence (a : RecurrenceSequence)
  (h1 : SatisfiesRecurrence a)
  (h2 : NonConstant a)
  (h3 : Positive a) :
  ∃ n : ℕ, Composite (a n) :=
  sorry

end NUMINAMATH_CALUDE_exists_composite_in_sequence_l2586_258689


namespace NUMINAMATH_CALUDE_marys_number_l2586_258643

/-- Represents the scenario described in the problem -/
structure Scenario where
  j : Nat  -- John's number
  m : Nat  -- Mary's number
  sum : Nat := j + m
  product : Nat := j * m

/-- Predicate to check if a number has multiple factorizations -/
def hasMultipleFactorizations (n : Nat) : Prop :=
  ∃ a b c d : Nat, a * b = n ∧ c * d = n ∧ a ≠ c ∧ a ≠ d ∧ a > 1 ∧ b > 1 ∧ c > 1 ∧ d > 1

/-- The main theorem representing the problem -/
theorem marys_number (s : Scenario) : 
  s.product = 2002 ∧ 
  hasMultipleFactorizations 2002 ∧
  (∀ x : Nat, x * s.m = 2002 → hasMultipleFactorizations x) →
  s.m = 1001 := by
  sorry

#eval 1001 * 2  -- Should output 2002

end NUMINAMATH_CALUDE_marys_number_l2586_258643


namespace NUMINAMATH_CALUDE_sum_of_three_greater_than_five_l2586_258633

theorem sum_of_three_greater_than_five (a b c : ℕ) :
  a ∈ Finset.range 10 →
  b ∈ Finset.range 10 →
  c ∈ Finset.range 10 →
  a ≠ b →
  a ≠ c →
  b ≠ c →
  a + b + c > 5 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_three_greater_than_five_l2586_258633


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_slopes_l2586_258636

/-- The slopes of the asymptotes for the hyperbola described by the equation x²/144 - y²/81 = 1 are ±3/4 -/
theorem hyperbola_asymptote_slopes (x y : ℝ) :
  x^2 / 144 - y^2 / 81 = 1 →
  ∃ (m : ℝ), m = 3/4 ∧ (∀ (x' y' : ℝ), x'^2 / 144 - y'^2 / 81 = 0 → y' = m * x' ∨ y' = -m * x') :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_slopes_l2586_258636


namespace NUMINAMATH_CALUDE_division_problem_l2586_258675

theorem division_problem (dividend : ℕ) (divisor : ℕ) (remainder : ℕ) (quotient : ℕ) :
  dividend = 122 →
  divisor = 20 →
  remainder = 2 →
  dividend = divisor * quotient + remainder →
  quotient = 6 := by
sorry

end NUMINAMATH_CALUDE_division_problem_l2586_258675


namespace NUMINAMATH_CALUDE_unique_prime_square_l2586_258666

theorem unique_prime_square (p : ℕ) : 
  Prime p ∧ ∃ k : ℕ, 2 * p^4 - p^2 + 16 = k^2 ↔ p = 3 := by
sorry

end NUMINAMATH_CALUDE_unique_prime_square_l2586_258666


namespace NUMINAMATH_CALUDE_glass_bowls_problem_l2586_258667

/-- The number of glass bowls initially bought -/
def initial_bowls : ℕ := 2393

/-- The buying price per bowl in rupees -/
def buying_price : ℚ := 18

/-- The selling price per bowl in rupees -/
def selling_price : ℚ := 20

/-- The number of bowls sold -/
def bowls_sold : ℕ := 104

/-- The percentage gain -/
def percentage_gain : ℚ := 0.4830917874396135

theorem glass_bowls_problem :
  let total_cost : ℚ := initial_bowls * buying_price
  let revenue : ℚ := bowls_sold * selling_price
  let gain : ℚ := revenue - (bowls_sold * buying_price)
  percentage_gain = (gain / total_cost) * 100 :=
by sorry

end NUMINAMATH_CALUDE_glass_bowls_problem_l2586_258667


namespace NUMINAMATH_CALUDE_pat_stickers_l2586_258611

def stickers_problem (initial_stickers end_stickers : ℝ) : Prop :=
  initial_stickers - end_stickers = 22

theorem pat_stickers : stickers_problem 39 17 := by
  sorry

end NUMINAMATH_CALUDE_pat_stickers_l2586_258611


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_difference_l2586_258600

theorem quadratic_equation_solution_difference : ∃ (x₁ x₂ : ℝ),
  (2 * x₁^2 - 6 * x₁ + 18 = 2 * x₁ + 82) ∧
  (2 * x₂^2 - 6 * x₂ + 18 = 2 * x₂ + 82) ∧
  x₁ ≠ x₂ ∧
  |x₁ - x₂| = 12 :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_difference_l2586_258600


namespace NUMINAMATH_CALUDE_perfect_square_trinomial_condition_l2586_258624

/-- A trinomial ax^2 + bx + c is a perfect square if there exist real numbers p and q
    such that ax^2 + bx + c = (px + q)^2 for all x. -/
def is_perfect_square_trinomial (a b c : ℝ) : Prop :=
  ∃ p q : ℝ, ∀ x, a * x^2 + b * x + c = (p * x + q)^2

/-- If 4x^2 + 2kx + 25 is a perfect square trinomial, then k = ±10. -/
theorem perfect_square_trinomial_condition (k : ℝ) :
  is_perfect_square_trinomial 4 (2*k) 25 → k = 10 ∨ k = -10 := by
  sorry


end NUMINAMATH_CALUDE_perfect_square_trinomial_condition_l2586_258624


namespace NUMINAMATH_CALUDE_rationalize_denominator_l2586_258686

theorem rationalize_denominator :
  (1 : ℝ) / (Real.rpow 3 (1/3) + Real.rpow 27 (1/3)) = Real.rpow 9 (1/3) / 12 := by
  sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l2586_258686


namespace NUMINAMATH_CALUDE_gcd_420_882_l2586_258608

theorem gcd_420_882 : Nat.gcd 420 882 = 42 := by
  sorry

end NUMINAMATH_CALUDE_gcd_420_882_l2586_258608


namespace NUMINAMATH_CALUDE_unique_positive_solution_l2586_258663

theorem unique_positive_solution : ∃! (x : ℝ), x > 0 ∧ x^2 + x = 210 := by
  sorry

end NUMINAMATH_CALUDE_unique_positive_solution_l2586_258663


namespace NUMINAMATH_CALUDE_donuts_distribution_l2586_258627

/-- Calculates the number of donuts each student who likes donuts receives -/
def donuts_per_student (total_donuts : ℕ) (total_students : ℕ) (donut_liking_ratio : ℚ) : ℚ :=
  total_donuts / (total_students * donut_liking_ratio)

/-- Proves that given 4 dozen donuts distributed among 80% of 30 students, 
    each student who likes donuts receives 2 donuts -/
theorem donuts_distribution : 
  donuts_per_student (4 * 12) 30 (4/5) = 2 := by
  sorry

end NUMINAMATH_CALUDE_donuts_distribution_l2586_258627


namespace NUMINAMATH_CALUDE_isosceles_triangle_relationship_l2586_258609

/-- An isosceles triangle with perimeter 10 cm -/
structure IsoscelesTriangle where
  /-- Length of each equal side in cm -/
  x : ℝ
  /-- Length of the base in cm -/
  y : ℝ
  /-- The triangle is isosceles -/
  isIsosceles : true
  /-- The perimeter is 10 cm -/
  perimeterIs10 : x + x + y = 10

/-- The relationship between y and x, and the range of x for the isosceles triangle -/
theorem isosceles_triangle_relationship (t : IsoscelesTriangle) :
  t.y = 10 - 2 * t.x ∧ 5/2 < t.x ∧ t.x < 5 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_relationship_l2586_258609


namespace NUMINAMATH_CALUDE_max_value_at_point_one_two_l2586_258681

/-- The feasible region defined by the given constraints -/
def FeasibleRegion (x y : ℝ) : Prop :=
  x + 2*y ≤ 5 ∧ 2*x + y ≤ 4 ∧ x ≥ 0 ∧ y ≥ 0

/-- The objective function to be maximized -/
def ObjectiveFunction (x y : ℝ) : ℝ := 3*x + 4*y

/-- Theorem stating that the maximum value of the objective function
    in the feasible region is 11, achieved at (1, 2) -/
theorem max_value_at_point_one_two :
  ∃ (max : ℝ), max = 11 ∧
  ∃ (x₀ y₀ : ℝ), x₀ = 1 ∧ y₀ = 2 ∧
  FeasibleRegion x₀ y₀ ∧
  ObjectiveFunction x₀ y₀ = max ∧
  ∀ (x y : ℝ), FeasibleRegion x y → ObjectiveFunction x y ≤ max :=
by
  sorry

end NUMINAMATH_CALUDE_max_value_at_point_one_two_l2586_258681


namespace NUMINAMATH_CALUDE_cylinder_radius_approximation_l2586_258659

noncomputable def cylinder_radius (square_area : ℝ) (rectangle_breadth : ℝ) : ℝ :=
  let square_side := Real.sqrt square_area
  let circle_radius := square_side
  let rectangle_length := (2 / 5) * circle_radius
  let rectangle_area := rectangle_length * rectangle_breadth
  let cylinder_volume := 2 * rectangle_area
  let cylinder_height := square_side
  Real.sqrt (cylinder_volume / (Real.pi * cylinder_height))

theorem cylinder_radius_approximation :
  ∀ (ε : ℝ), ε > 0 →
  abs (cylinder_radius 2500 10 - 1.59514) < ε :=
sorry

end NUMINAMATH_CALUDE_cylinder_radius_approximation_l2586_258659


namespace NUMINAMATH_CALUDE_orange_profit_problem_l2586_258644

/-- The number of oranges needed to make a profit of 200 cents -/
def oranges_needed (buy_price : ℚ) (sell_price : ℚ) (profit_goal : ℚ) : ℕ :=
  (profit_goal / (sell_price - buy_price)).ceil.toNat

/-- The problem statement -/
theorem orange_profit_problem :
  let buy_price : ℚ := 14 / 4
  let sell_price : ℚ := 25 / 6
  let profit_goal : ℚ := 200
  oranges_needed buy_price sell_price profit_goal = 300 := by
sorry

end NUMINAMATH_CALUDE_orange_profit_problem_l2586_258644


namespace NUMINAMATH_CALUDE_smallest_yellow_marbles_l2586_258639

theorem smallest_yellow_marbles (n : ℕ) (h1 : n % 3 = 0) 
  (h2 : n ≥ 30) : ∃ (blue red green yellow : ℕ),
  blue = n / 3 ∧ 
  red = n / 3 ∧ 
  green = 10 ∧ 
  yellow = n - (blue + red + green) ∧ 
  yellow ≥ 0 ∧ 
  ∀ m : ℕ, m < n → ¬(∃ (b r g y : ℕ), 
    b = m / 3 ∧ 
    r = m / 3 ∧ 
    g = 10 ∧ 
    y = m - (b + r + g) ∧ 
    y ≥ 0 ∧ 
    m % 3 = 0) :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_yellow_marbles_l2586_258639


namespace NUMINAMATH_CALUDE_equation_solutions_l2586_258631

theorem equation_solutions :
  (∀ X : ℝ, X - 12 = 81 → X = 93) ∧
  (∀ X : ℝ, 5.1 + X = 10.5 → X = 5.4) ∧
  (∀ X : ℝ, 6 * X = 4.2 → X = 0.7) ∧
  (∀ X : ℝ, X / 0.4 = 12.5 → X = 5) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l2586_258631


namespace NUMINAMATH_CALUDE_solution_set_f_gt_2x_plus_1_range_of_t_for_f_geq_g_l2586_258654

-- Define the functions f and g
def f (x : ℝ) := |x - 1|
def g (t x : ℝ) := t * |x| - 2

-- Statement 1: Solution set of f(x) > 2x+1
theorem solution_set_f_gt_2x_plus_1 :
  {x : ℝ | f x > 2 * x + 1} = {x : ℝ | x < 0} := by sorry

-- Statement 2: Range of t for which f(x) ≥ g(x) holds for all x ∈ ℝ
theorem range_of_t_for_f_geq_g :
  ∀ t : ℝ, (∀ x : ℝ, f x ≥ g t x) ↔ t ≤ 1 := by sorry

end NUMINAMATH_CALUDE_solution_set_f_gt_2x_plus_1_range_of_t_for_f_geq_g_l2586_258654


namespace NUMINAMATH_CALUDE_intersection_point_d_l2586_258694

def g (c : ℤ) (x : ℝ) : ℝ := 5 * x + c

theorem intersection_point_d (c : ℤ) (d : ℤ) :
  (g c (-5) = d) ∧ (g c d = -5) → d = -5 := by sorry

end NUMINAMATH_CALUDE_intersection_point_d_l2586_258694


namespace NUMINAMATH_CALUDE_center_value_of_arithmetic_array_l2586_258655

/-- Represents a 3x3 array with arithmetic sequences in rows and columns -/
def ArithmeticArray := Matrix (Fin 3) (Fin 3) ℝ

/-- Checks if a sequence of three real numbers is arithmetic -/
def is_arithmetic_sequence (a b c : ℝ) : Prop :=
  b - a = c - b

/-- Properties of the arithmetic array -/
def arithmetic_array_properties (A : ArithmeticArray) : Prop :=
  ∀ i : Fin 3,
    (is_arithmetic_sequence (A i 0) (A i 1) (A i 2)) ∧
    (is_arithmetic_sequence (A 0 i) (A 1 i) (A 2 i))

theorem center_value_of_arithmetic_array (A : ArithmeticArray) 
  (h_props : arithmetic_array_properties A)
  (h_first_row : A 0 0 = 3 ∧ A 0 2 = 15)
  (h_last_row : A 2 0 = 9 ∧ A 2 2 = 33) :
  A 1 1 = 15 := by
  sorry

end NUMINAMATH_CALUDE_center_value_of_arithmetic_array_l2586_258655


namespace NUMINAMATH_CALUDE_line_slope_l2586_258661

theorem line_slope (x y : ℝ) : 
  (3 * x - Real.sqrt 3 * y + 1 = 0) → 
  (∃ m : ℝ, y = m * x + (-1 / Real.sqrt 3) ∧ m = Real.sqrt 3) :=
sorry

end NUMINAMATH_CALUDE_line_slope_l2586_258661


namespace NUMINAMATH_CALUDE_cos_48_degrees_l2586_258692

theorem cos_48_degrees : Real.cos (48 * π / 180) = Real.cos (48 * π / 180) := by
  sorry

end NUMINAMATH_CALUDE_cos_48_degrees_l2586_258692


namespace NUMINAMATH_CALUDE_corrected_mean_calculation_l2586_258688

def original_mean : ℝ := 36
def num_observations : ℕ := 50
def error_1 : (ℝ × ℝ) := (46, 23)
def error_2 : (ℝ × ℝ) := (55, 40)
def error_3 : (ℝ × ℝ) := (28, 15)

theorem corrected_mean_calculation :
  let total_sum := original_mean * num_observations
  let error_sum := error_1.1 + error_2.1 + error_3.1 - (error_1.2 + error_2.2 + error_3.2)
  let corrected_sum := total_sum + error_sum
  corrected_sum / num_observations = 37.02 := by sorry

end NUMINAMATH_CALUDE_corrected_mean_calculation_l2586_258688


namespace NUMINAMATH_CALUDE_parabola_property_l2586_258607

/-- A parabola with equation y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The y-coordinate of a point on the parabola given its x-coordinate -/
def Parabola.y_at (p : Parabola) (x : ℝ) : ℝ :=
  p.a * x^2 + p.b * x + p.c

theorem parabola_property (p : Parabola) :
  p.y_at (-3) = 4 →  -- vertex at (-3, 4)
  p.y_at (-2) = 7 →  -- passes through (-2, 7)
  3 * p.a + 2 * p.b + p.c = 76 := by
  sorry

end NUMINAMATH_CALUDE_parabola_property_l2586_258607


namespace NUMINAMATH_CALUDE_shirt_cost_l2586_258646

theorem shirt_cost (total_cost shirt_cost coat_cost : ℝ) : 
  total_cost = 600 →
  shirt_cost = (1/3) * coat_cost →
  shirt_cost + coat_cost = total_cost →
  shirt_cost = 150 := by
sorry

end NUMINAMATH_CALUDE_shirt_cost_l2586_258646


namespace NUMINAMATH_CALUDE_inequality_proof_l2586_258638

theorem inequality_proof (a b c d : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0) 
  (h_prod : a * b * c * d = 1) 
  (h_ineq : a + b + c + d > a/b + b/c + c/d + d/a) : 
  a + b + c + d < b/a + c/b + d/c + a/d := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l2586_258638


namespace NUMINAMATH_CALUDE_product_of_integers_with_given_lcm_and_gcd_l2586_258630

theorem product_of_integers_with_given_lcm_and_gcd :
  ∀ a b : ℕ+, 
  (Nat.lcm a b = 60) → 
  (Nat.gcd a b = 12) → 
  (a * b = 720) :=
by
  sorry

end NUMINAMATH_CALUDE_product_of_integers_with_given_lcm_and_gcd_l2586_258630


namespace NUMINAMATH_CALUDE_tan_equation_solution_set_l2586_258679

theorem tan_equation_solution_set :
  {x : Real | Real.tan x = 2} = {x : Real | ∃ k : ℤ, x = k * Real.pi + Real.arctan 2} := by sorry

end NUMINAMATH_CALUDE_tan_equation_solution_set_l2586_258679


namespace NUMINAMATH_CALUDE_divisible_by_77_l2586_258622

theorem divisible_by_77 (n : ℕ) (h : ∀ k : ℕ, 2 ≤ k → k ≤ 76 → k ∣ n) : 77 ∣ n := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_77_l2586_258622


namespace NUMINAMATH_CALUDE_largest_divisor_of_consecutive_even_numbers_l2586_258628

theorem largest_divisor_of_consecutive_even_numbers : ∃ (m : ℕ), 
  (∀ (n : ℕ), (2*n) * (2*n + 2) * (2*n + 4) % m = 0) ∧ 
  (∀ (k : ℕ), k > m → ∃ (n : ℕ), (2*n) * (2*n + 2) * (2*n + 4) % k ≠ 0) :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_largest_divisor_of_consecutive_even_numbers_l2586_258628


namespace NUMINAMATH_CALUDE_gdp_scientific_notation_l2586_258618

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  valid : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a real number to scientific notation with a specified number of significant figures -/
def toScientificNotation (x : ℝ) (sigFigs : ℕ) : ScientificNotation :=
  sorry

/-- The original GDP value in trillions of dollars -/
def originalGDP : ℝ := 1.337

/-- The number of significant figures to use -/
def sigFigs : ℕ := 3

theorem gdp_scientific_notation :
  toScientificNotation (originalGDP * 1000000000000) sigFigs =
    ScientificNotation.mk 1.34 12 (by norm_num) :=
  sorry

end NUMINAMATH_CALUDE_gdp_scientific_notation_l2586_258618


namespace NUMINAMATH_CALUDE_arithmetic_arrangement_l2586_258649

theorem arithmetic_arrangement :
  (1 / 8 * 1 / 9 * 1 / 28 = 1 / 2016) ∧
  ((1 / 8 - 1 / 9) * 1 / 28 = 1 / 2016) := by sorry

end NUMINAMATH_CALUDE_arithmetic_arrangement_l2586_258649


namespace NUMINAMATH_CALUDE_edward_lives_problem_l2586_258690

theorem edward_lives_problem (lives_lost lives_remaining : ℕ) 
  (h1 : lives_lost = 8)
  (h2 : lives_remaining = 7) :
  lives_lost + lives_remaining = 15 :=
by sorry

end NUMINAMATH_CALUDE_edward_lives_problem_l2586_258690


namespace NUMINAMATH_CALUDE_parabola_translation_l2586_258665

/-- Represents a parabola of the form y = ax² + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Translates a parabola horizontally and vertically -/
def translate (p : Parabola) (h : ℝ) (v : ℝ) : Parabola :=
  { a := p.a
    b := -2 * p.a * h + p.b
    c := p.a * h^2 - p.b * h + p.c + v }

theorem parabola_translation (x : ℝ) :
  let original := Parabola.mk 5 0 0
  let translated := translate original 2 3
  (5 * x^2) + 3 = translated.a * (x - 2)^2 + translated.b * (x - 2) + translated.c := by
  sorry

end NUMINAMATH_CALUDE_parabola_translation_l2586_258665


namespace NUMINAMATH_CALUDE_price_reduction_theorem_l2586_258604

def original_price_A : ℝ := 500
def original_price_B : ℝ := 600
def original_price_C : ℝ := 700

def first_discount_rate : ℝ := 0.15
def second_discount_rate : ℝ := 0.10
def sales_tax_rate : ℝ := 0.07
def flat_discount_B : ℝ := 200

def total_original_price : ℝ := original_price_A + original_price_B + original_price_C

noncomputable def final_price_A : ℝ := 
  (original_price_A * (1 - first_discount_rate) * (1 - second_discount_rate)) * (1 + sales_tax_rate)

noncomputable def final_price_B : ℝ := 
  (original_price_B * (1 - first_discount_rate) * (1 - second_discount_rate)) - flat_discount_B

noncomputable def final_price_C : ℝ := 
  (original_price_C * (1 - second_discount_rate)) * (1 + sales_tax_rate)

noncomputable def total_final_price : ℝ := final_price_A + final_price_B + final_price_C

noncomputable def percentage_reduction : ℝ := 
  (total_original_price - total_final_price) / total_original_price * 100

theorem price_reduction_theorem : 
  25.42 ≤ percentage_reduction ∧ percentage_reduction < 25.43 :=
sorry

end NUMINAMATH_CALUDE_price_reduction_theorem_l2586_258604


namespace NUMINAMATH_CALUDE_prob_three_dice_l2586_258642

/-- The number of faces on a die -/
def num_faces : ℕ := 6

/-- The number of favorable outcomes on a single die (numbers greater than 2) -/
def favorable_outcomes : ℕ := 4

/-- The number of dice thrown simultaneously -/
def num_dice : ℕ := 3

/-- The probability of getting a number greater than 2 on a single die -/
def prob_single_die : ℚ := favorable_outcomes / num_faces

/-- The probability of getting a number greater than 2 on each of three dice -/
theorem prob_three_dice : (prob_single_die ^ num_dice : ℚ) = 8 / 27 := by
  sorry

end NUMINAMATH_CALUDE_prob_three_dice_l2586_258642


namespace NUMINAMATH_CALUDE_perpendicular_bisector_value_l2586_258619

/-- If the line x + y = b is the perpendicular bisector of the line segment from (2,4) to (6,10), then b = 11 -/
theorem perpendicular_bisector_value (b : ℝ) : 
  (∀ (x y : ℝ), x + y = b ↔ 
    ((x - 4)^2 + (y - 7)^2 = (2 - 4)^2 + (4 - 7)^2 ∧ 
     (x - 4)^2 + (y - 7)^2 = (6 - 4)^2 + (10 - 7)^2)) → 
  b = 11 :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_bisector_value_l2586_258619


namespace NUMINAMATH_CALUDE_complex_fraction_evaluation_l2586_258680

theorem complex_fraction_evaluation :
  let i : ℂ := Complex.I
  (3 + i) / (1 + i) = 2 - i :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_evaluation_l2586_258680


namespace NUMINAMATH_CALUDE_subset_P_l2586_258635

def P : Set ℝ := {x | x > -1}

theorem subset_P : {0} ⊆ P := by sorry

end NUMINAMATH_CALUDE_subset_P_l2586_258635


namespace NUMINAMATH_CALUDE_banana_boxes_l2586_258693

def total_bananas : ℕ := 40
def bananas_per_box : ℕ := 4

theorem banana_boxes : total_bananas / bananas_per_box = 10 := by
  sorry

end NUMINAMATH_CALUDE_banana_boxes_l2586_258693


namespace NUMINAMATH_CALUDE_problem_statement_l2586_258617

theorem problem_statement :
  (∀ x : ℝ, x < 0 → (2 : ℝ)^x > (3 : ℝ)^x) ∧
  (∃ x : ℝ, x > 0 ∧ Real.sqrt x > x^3) :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l2586_258617


namespace NUMINAMATH_CALUDE_derivative_of_one_minus_cosine_l2586_258698

theorem derivative_of_one_minus_cosine (α : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ 1 - Real.cos x
  (deriv f) α = Real.sin α := by
sorry

end NUMINAMATH_CALUDE_derivative_of_one_minus_cosine_l2586_258698


namespace NUMINAMATH_CALUDE_raisin_cost_fraction_l2586_258640

/-- Given a mixture of raisins and nuts, where the cost of nuts is twice that of raisins,
    prove that the cost of raisins is 3/11 of the total cost of the mixture. -/
theorem raisin_cost_fraction (raisin_cost : ℚ) : 
  let raisin_pounds : ℚ := 3
  let nut_pounds : ℚ := 4
  let nut_cost : ℚ := 2 * raisin_cost
  let total_raisin_cost : ℚ := raisin_pounds * raisin_cost
  let total_nut_cost : ℚ := nut_pounds * nut_cost
  let total_cost : ℚ := total_raisin_cost + total_nut_cost
  total_raisin_cost / total_cost = 3 / 11 := by
sorry

end NUMINAMATH_CALUDE_raisin_cost_fraction_l2586_258640


namespace NUMINAMATH_CALUDE_intersection_equality_implies_range_l2586_258691

-- Define the sets A and C
def A : Set ℝ := {x | 1 ≤ x ∧ x ≤ 5}
def C (a : ℝ) : Set ℝ := {x | -a ≤ x ∧ x ≤ a + 3}

-- State the theorem
theorem intersection_equality_implies_range (a : ℝ) :
  C a ∩ A = C a → -3/2 ≤ a ∧ a ≤ -1 :=
by sorry

end NUMINAMATH_CALUDE_intersection_equality_implies_range_l2586_258691


namespace NUMINAMATH_CALUDE_specific_pentagon_area_l2586_258697

/-- Pentagon with specified side lengths and right angles -/
structure Pentagon where
  PQ : ℝ
  QR : ℝ
  RS : ℝ
  ST : ℝ
  TP : ℝ
  angle_TPQ : ℝ
  angle_PQR : ℝ

/-- The area of a pentagon with the given properties -/
def pentagon_area (p : Pentagon) : ℝ := sorry

/-- Theorem stating the area of the specific pentagon is 100 -/
theorem specific_pentagon_area :
  let p : Pentagon := {
    PQ := 8,
    QR := 2,
    RS := 13,
    ST := 13,
    TP := 8,
    angle_TPQ := 90,
    angle_PQR := 90
  }
  pentagon_area p = 100 := by sorry

end NUMINAMATH_CALUDE_specific_pentagon_area_l2586_258697


namespace NUMINAMATH_CALUDE_amc10_paths_l2586_258657

/-- Represents the number of possible moves from each position -/
def num_moves : ℕ := 8

/-- Represents the length of the string "AMC10" -/
def word_length : ℕ := 5

/-- Calculates the number of paths to spell "AMC10" -/
def num_paths : ℕ := num_moves ^ (word_length - 1)

/-- Proves that the number of paths to spell "AMC10" is 4096 -/
theorem amc10_paths : num_paths = 4096 := by
  sorry

end NUMINAMATH_CALUDE_amc10_paths_l2586_258657


namespace NUMINAMATH_CALUDE_binomial_identity_solutions_l2586_258668

theorem binomial_identity_solutions (n : ℕ) :
  ∀ x y : ℝ, (x + y)^n = x^n + y^n ↔
    (n = 1 ∧ True) ∨
    (∃ k : ℕ, n = 2 * k ∧ (x = 0 ∨ y = 0)) ∨
    (∃ k : ℕ, n = 2 * k + 1 ∧ (x = 0 ∨ y = 0 ∨ x = -y)) :=
by sorry

end NUMINAMATH_CALUDE_binomial_identity_solutions_l2586_258668


namespace NUMINAMATH_CALUDE_cone_surface_area_minimization_l2586_258634

/-- For a cone with fixed volume, when the total surface area is minimized,
    there exists a relationship between the height and radius of the cone. -/
theorem cone_surface_area_minimization (V : ℝ) (V_pos : V > 0) :
  ∃ (R H : ℝ) (R_pos : R > 0) (H_pos : H > 0),
    (1/3 : ℝ) * Real.pi * R^2 * H = V ∧
    (∀ (r h : ℝ) (r_pos : r > 0) (h_pos : h > 0),
      (1/3 : ℝ) * Real.pi * r^2 * h = V →
      Real.pi * R^2 + Real.pi * R * Real.sqrt (R^2 + H^2) ≤
      Real.pi * r^2 + Real.pi * r * Real.sqrt (r^2 + h^2)) →
    ∃ (k : ℝ), H = k * R := by
  sorry

end NUMINAMATH_CALUDE_cone_surface_area_minimization_l2586_258634


namespace NUMINAMATH_CALUDE_math_problem_solution_l2586_258645

theorem math_problem_solution :
  ∀ (S₁ S₂ S₃ S₁₂ S₁₃ S₂₃ S₁₂₃ : ℕ),
  S₁ + S₂ + S₃ + S₁₂ + S₁₃ + S₂₃ + S₁₂₃ = 100 →
  S₁ + S₁₂ + S₁₃ + S₁₂₃ = 60 →
  S₂ + S₁₂ + S₂₃ + S₁₂₃ = 60 →
  S₃ + S₁₃ + S₂₃ + S₁₂₃ = 60 →
  (S₁ + S₂ + S₃) - S₁₂₃ = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_math_problem_solution_l2586_258645


namespace NUMINAMATH_CALUDE_tamara_height_l2586_258678

/-- Given that Tamara's height is 3 times Kim's height minus 4 inches,
    and their combined height is 92 inches, prove that Tamara is 68 inches tall. -/
theorem tamara_height (kim : ℝ) (tamara : ℝ) : 
  tamara = 3 * kim - 4 → 
  tamara + kim = 92 → 
  tamara = 68 := by
sorry

end NUMINAMATH_CALUDE_tamara_height_l2586_258678


namespace NUMINAMATH_CALUDE_vertex_x_coordinate_is_one_l2586_258662

-- Define the quadratic function
def quadratic (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- Define the theorem
theorem vertex_x_coordinate_is_one 
  (a b c : ℝ) 
  (h1 : quadratic a b c 0 = 3)
  (h2 : quadratic a b c 2 = 3)
  (h3 : quadratic a b c 4 = 11) :
  ∃ k : ℝ, quadratic a b c x = a * (x - 1)^2 + k := by
sorry


end NUMINAMATH_CALUDE_vertex_x_coordinate_is_one_l2586_258662


namespace NUMINAMATH_CALUDE_diophantine_equations_solutions_l2586_258625

-- Define the set of solutions for the first equation
def S₁ : Set (ℤ × ℤ) := {(x, y) | ∃ k : ℤ, x = 3 * k + 1 ∧ y = -2 * k + 1}

-- Define the set of solutions for the second equation
def S₂ : Set (ℤ × ℤ) := {(x, y) | ∃ k : ℤ, x = 5 * k ∧ y = 2 - 2 * k}

theorem diophantine_equations_solutions :
  (∀ (x y : ℤ), (2 * x + 3 * y = 5) ↔ (x, y) ∈ S₁) ∧
  (∀ (x y : ℤ), (2 * x + 5 * y = 10) ↔ (x, y) ∈ S₂) ∧
  (¬ ∃ (x y : ℤ), 3 * x + 9 * y = 2018) := by
  sorry

#check diophantine_equations_solutions

end NUMINAMATH_CALUDE_diophantine_equations_solutions_l2586_258625


namespace NUMINAMATH_CALUDE_find_m_l2586_258616

/-- The value of log base 10 of 2, approximated to 4 decimal places -/
def log10_2 : ℝ := 0.3010

/-- Theorem stating that the positive integer m satisfying the given inequality is 155 -/
theorem find_m (m : ℕ) (hm_pos : m > 0) 
  (h_ineq : (10 : ℝ)^(m-1) < (2 : ℝ)^512 ∧ (2 : ℝ)^512 < (10 : ℝ)^m) : 
  m = 155 := by
  sorry

#check find_m

end NUMINAMATH_CALUDE_find_m_l2586_258616


namespace NUMINAMATH_CALUDE_strikers_count_l2586_258629

/-- A soccer team composition -/
structure SoccerTeam where
  goalies : Nat
  defenders : Nat
  midfielders : Nat
  strikers : Nat

/-- The total number of players in a soccer team -/
def total_players (team : SoccerTeam) : Nat :=
  team.goalies + team.defenders + team.midfielders + team.strikers

/-- Theorem: Given the conditions, the number of strikers is 7 -/
theorem strikers_count (team : SoccerTeam)
  (h1 : team.goalies = 3)
  (h2 : team.defenders = 10)
  (h3 : team.midfielders = 2 * team.defenders)
  (h4 : total_players team = 40) :
  team.strikers = 7 := by
  sorry

end NUMINAMATH_CALUDE_strikers_count_l2586_258629


namespace NUMINAMATH_CALUDE_parabola_greatest_a_l2586_258696

/-- The greatest possible value of a for a parabola with given conditions -/
theorem parabola_greatest_a (a b c : ℝ) : 
  (∃ (x y : ℝ), y = a * x^2 + b * x + c ∧ x = 3/5 ∧ y = -1/5) → -- vertex condition
  a < 0 → -- a is negative
  (∃ (k : ℤ), b + 2*c = k) → -- b + 2c is an integer
  (∀ (a' : ℝ), (∃ (b' c' : ℝ), 
    (∃ (x y : ℝ), y = a' * x^2 + b' * x + c' ∧ x = 3/5 ∧ y = -1/5) ∧
    a' < 0 ∧
    (∃ (k : ℤ), b' + 2*c' = k)) →
    a' ≤ a) →
  a = -5/6 := by
sorry

end NUMINAMATH_CALUDE_parabola_greatest_a_l2586_258696


namespace NUMINAMATH_CALUDE_tournament_teams_count_l2586_258671

/-- Calculates the number of matches in a round-robin tournament for n teams -/
def matchesInGroup (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Represents a valid configuration of team groups -/
structure GroupConfig where
  g1 : ℕ
  g2 : ℕ
  g3 : ℕ
  g4 : ℕ
  h1 : g1 ≥ 2
  h2 : g2 ≥ 2
  h3 : g3 ≥ 2
  h4 : g4 ≥ 2
  h5 : matchesInGroup g1 + matchesInGroup g2 + matchesInGroup g3 + matchesInGroup g4 = 66

/-- The set of all possible total number of teams -/
def possibleTotalTeams : Set ℕ := {21, 22, 23, 24, 25}

theorem tournament_teams_count :
  ∀ (config : GroupConfig), (config.g1 + config.g2 + config.g3 + config.g4) ∈ possibleTotalTeams :=
by sorry

end NUMINAMATH_CALUDE_tournament_teams_count_l2586_258671


namespace NUMINAMATH_CALUDE_range_of_m_l2586_258650

def elliptical_region (x y : ℝ) : Prop := x^2 / 4 + y^2 ≤ 1

def dividing_lines (x y m : ℝ) : Prop :=
  (y = Real.sqrt 2 * x) ∨ (y = -Real.sqrt 2 * x) ∨ (x = m)

def valid_coloring (n : ℕ) : Prop :=
  n = 720 ∧ ∃ (colors : Fin 6 → Type) (parts : Type) (coloring : parts → Fin 6),
    ∀ (p1 p2 : parts), p1 ≠ p2 → coloring p1 ≠ coloring p2

theorem range_of_m :
  ∀ m : ℝ,
    (∀ x y : ℝ, elliptical_region x y → dividing_lines x y m → valid_coloring 720) ↔
    ((-2 < m ∧ m ≤ -2/3) ∨ m = 0 ∨ (2/3 ≤ m ∧ m < 2)) :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l2586_258650


namespace NUMINAMATH_CALUDE_average_sale_proof_l2586_258610

def sales_first_five : List Int := [2500, 6500, 9855, 7230, 7000]
def sales_sixth : Int := 11915
def num_months : Int := 6

theorem average_sale_proof :
  (sales_first_five.sum + sales_sixth) / num_months = 7500 := by
  sorry

end NUMINAMATH_CALUDE_average_sale_proof_l2586_258610


namespace NUMINAMATH_CALUDE_rope_ratio_proof_l2586_258612

theorem rope_ratio_proof (total_length shorter_length : ℕ) 
  (h1 : total_length = 40)
  (h2 : shorter_length = 16)
  (h3 : shorter_length < total_length) :
  (shorter_length : ℚ) / (total_length - shorter_length : ℚ) = 2 / 3 := by
sorry

end NUMINAMATH_CALUDE_rope_ratio_proof_l2586_258612


namespace NUMINAMATH_CALUDE_weight_of_four_moles_of_compound_l2586_258651

/-- The weight of a given number of moles of a compound -/
def weight_of_moles (molecular_weight : ℝ) (num_moles : ℝ) : ℝ :=
  molecular_weight * num_moles

/-- Theorem: The weight of 4 moles of a compound with molecular weight 312 g/mol is 1248 grams -/
theorem weight_of_four_moles_of_compound (molecular_weight : ℝ) 
  (h : molecular_weight = 312) : weight_of_moles molecular_weight 4 = 1248 := by
  sorry

end NUMINAMATH_CALUDE_weight_of_four_moles_of_compound_l2586_258651


namespace NUMINAMATH_CALUDE_triangle_area_from_perimeter_and_inradius_l2586_258620

/-- Given a triangle with perimeter 48 and inradius 2.5, prove its area is 60 -/
theorem triangle_area_from_perimeter_and_inradius :
  ∀ (T : Set ℝ) (perimeter inradius area : ℝ),
  (perimeter = 48) →
  (inradius = 2.5) →
  (area = inradius * (perimeter / 2)) →
  area = 60 := by
sorry

end NUMINAMATH_CALUDE_triangle_area_from_perimeter_and_inradius_l2586_258620


namespace NUMINAMATH_CALUDE_safe_count_theorem_l2586_258621

def is_p_safe (n p : ℕ) : Prop :=
  n % p > 2 ∧ n % p < p - 2

def count_safe (max : ℕ) : ℕ :=
  (max / (5 * 7 * 17)) * 48

theorem safe_count_theorem :
  count_safe 20000 = 1584 ∧
  ∀ n : ℕ, n ≤ 20000 →
    (is_p_safe n 5 ∧ is_p_safe n 7 ∧ is_p_safe n 17) ↔
    ∃ k : ℕ, k < 48 ∧ n ≡ k [MOD 595] :=
by sorry

end NUMINAMATH_CALUDE_safe_count_theorem_l2586_258621
