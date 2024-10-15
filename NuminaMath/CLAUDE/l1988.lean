import Mathlib

namespace NUMINAMATH_CALUDE_opposite_of_negative_2023_l1988_198895

def opposite (x : ℤ) : ℤ := -x

theorem opposite_of_negative_2023 : opposite (-2023) = 2023 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_negative_2023_l1988_198895


namespace NUMINAMATH_CALUDE_simplify_expression_l1988_198855

theorem simplify_expression (x : ℝ) (h : x ≠ -2) :
  4 / (x + 2) + x - 2 = x^2 / (x + 2) := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1988_198855


namespace NUMINAMATH_CALUDE_largest_prime_divisor_of_sum_of_squares_l1988_198894

theorem largest_prime_divisor_of_sum_of_squares :
  ∃ (p : ℕ), Nat.Prime p ∧ p ∣ (35^2 + 84^2) ∧ 
  ∀ (q : ℕ), Nat.Prime q → q ∣ (35^2 + 84^2) → q ≤ p :=
by sorry

end NUMINAMATH_CALUDE_largest_prime_divisor_of_sum_of_squares_l1988_198894


namespace NUMINAMATH_CALUDE_distinct_remainders_of_sums_l1988_198808

theorem distinct_remainders_of_sums (n : ℕ) (h : n > 1) :
  let S := Finset.range n
  ∀ (i j k l : ℕ) (hi : i ∈ S) (hj : j ∈ S) (hk : k ∈ S) (hl : l ∈ S)
    (hij : i ≤ j) (hkl : k ≤ l),
  (i + j) % (n * (n + 1) / 2) = (k + l) % (n * (n + 1) / 2) →
  i = k ∧ j = l :=
by sorry

end NUMINAMATH_CALUDE_distinct_remainders_of_sums_l1988_198808


namespace NUMINAMATH_CALUDE_log_27_3_l1988_198802

theorem log_27_3 : Real.log 3 / Real.log 27 = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_log_27_3_l1988_198802


namespace NUMINAMATH_CALUDE_not_divides_power_diff_l1988_198827

theorem not_divides_power_diff (n : ℕ+) : ¬ ∃ k : ℤ, (2^(n : ℕ) + 65) * k = 5^(n : ℕ) - 3^(n : ℕ) := by
  sorry

end NUMINAMATH_CALUDE_not_divides_power_diff_l1988_198827


namespace NUMINAMATH_CALUDE_tangent_circle_equation_l1988_198872

/-- A circle C tangent to the line x-2=0 at point (2,1) with radius 3 -/
structure TangentCircle where
  /-- The center of the circle -/
  center : ℝ × ℝ
  /-- The radius of the circle -/
  radius : ℝ
  /-- The circle is tangent to the line x-2=0 at point (2,1) -/
  tangent_point : center.1 - 2 = radius ∨ center.1 - 2 = -radius
  /-- The point (2,1) lies on the circle -/
  on_circle : (2 - center.1)^2 + (1 - center.2)^2 = radius^2
  /-- The radius is 3 -/
  radius_is_three : radius = 3

/-- The equation of the circle is either (x+1)^2+(y-1)^2=9 or (x-5)^2+(y-1)^2=9 -/
theorem tangent_circle_equation (c : TangentCircle) :
  (∀ x y, (x - c.center.1)^2 + (y - c.center.2)^2 = c.radius^2) →
  ((∀ x y, (x + 1)^2 + (y - 1)^2 = 9) ∨ (∀ x y, (x - 5)^2 + (y - 1)^2 = 9)) :=
by sorry

end NUMINAMATH_CALUDE_tangent_circle_equation_l1988_198872


namespace NUMINAMATH_CALUDE_allowance_equation_l1988_198865

/-- The student's monthly allowance in USD -/
def monthly_allowance : ℝ := 29.65

/-- Proposition: Given the spending pattern, the monthly allowance satisfies the equation -/
theorem allowance_equation : 
  (5 / 42 : ℝ) * monthly_allowance = 3 / 0.85 := by
  sorry

end NUMINAMATH_CALUDE_allowance_equation_l1988_198865


namespace NUMINAMATH_CALUDE_accounting_majors_l1988_198834

theorem accounting_majors (p q r s : ℕ+) 
  (h1 : p * q * r * s = 1365)
  (h2 : 1 < p)
  (h3 : p < q)
  (h4 : q < r)
  (h5 : r < s) :
  p = 3 := by sorry

end NUMINAMATH_CALUDE_accounting_majors_l1988_198834


namespace NUMINAMATH_CALUDE_not_sufficient_not_necessary_l1988_198876

open Set

/-- The set of real numbers x where x³ - 2x > 0 -/
def S : Set ℝ := {x | x^3 - 2*x > 0}

/-- The set of real numbers x where |x + 1| > 3 -/
def T : Set ℝ := {x | |x + 1| > 3}

theorem not_sufficient_not_necessary : ¬(S ⊆ T) ∧ ¬(T ⊆ S) := by sorry

end NUMINAMATH_CALUDE_not_sufficient_not_necessary_l1988_198876


namespace NUMINAMATH_CALUDE_square_perimeter_l1988_198880

/-- Given a square cut into four equal rectangles that form a shape with perimeter 56,
    prove that the original square's perimeter is 32. -/
theorem square_perimeter (width : ℝ) (length : ℝ) : 
  width > 0 →
  length = 4 * width →
  2 * (14 * width) = 56 →
  4 * (4 * width) = 32 :=
by
  sorry

#check square_perimeter

end NUMINAMATH_CALUDE_square_perimeter_l1988_198880


namespace NUMINAMATH_CALUDE_largest_decimal_l1988_198833

theorem largest_decimal : 
  let a := 0.9877
  let b := 0.9789
  let c := 0.9700
  let d := 0.9790
  let e := 0.9709
  (a > b) ∧ (a > c) ∧ (a > d) ∧ (a > e) :=
by sorry

end NUMINAMATH_CALUDE_largest_decimal_l1988_198833


namespace NUMINAMATH_CALUDE_largest_prime_divisor_factorial_sum_l1988_198884

theorem largest_prime_divisor_factorial_sum : 
  ∃ p : Nat, 
    Nat.Prime p ∧ 
    p ∣ (Nat.factorial 13 + Nat.factorial 14) ∧
    ∀ q : Nat, Nat.Prime q → q ∣ (Nat.factorial 13 + Nat.factorial 14) → q ≤ p :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_largest_prime_divisor_factorial_sum_l1988_198884


namespace NUMINAMATH_CALUDE_angle_in_second_quadrant_l1988_198801

theorem angle_in_second_quadrant (θ : Real) (h : θ = 27 * Real.pi / 4) :
  0 < θ % (2 * Real.pi) ∧ θ % (2 * Real.pi) < Real.pi :=
by sorry

end NUMINAMATH_CALUDE_angle_in_second_quadrant_l1988_198801


namespace NUMINAMATH_CALUDE_cone_lateral_area_l1988_198882

/-- The lateral area of a cone with base radius 3 cm and height 4 cm is 15π cm². -/
theorem cone_lateral_area :
  let r : ℝ := 3  -- radius in cm
  let h : ℝ := 4  -- height in cm
  let l : ℝ := Real.sqrt (r^2 + h^2)  -- slant height
  let lateral_area : ℝ := π * r * l  -- lateral area formula
  lateral_area = 15 * π := by sorry

end NUMINAMATH_CALUDE_cone_lateral_area_l1988_198882


namespace NUMINAMATH_CALUDE_terrence_earnings_l1988_198862

def total_earnings : ℕ := 90
def emilee_earnings : ℕ := 25

theorem terrence_earnings (jermaine_earnings terrence_earnings : ℕ) 
  (h1 : jermaine_earnings = terrence_earnings + 5)
  (h2 : jermaine_earnings + terrence_earnings + emilee_earnings = total_earnings) :
  terrence_earnings = 30 := by
  sorry

end NUMINAMATH_CALUDE_terrence_earnings_l1988_198862


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1988_198842

theorem inequality_solution_set : 
  {x : ℝ | (x - 2) * (2 * x + 1) > 0} = 
  {x : ℝ | x < -1/2 ∨ x > 2} := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1988_198842


namespace NUMINAMATH_CALUDE_factorization_3mx_6my_factorization_1_25x_squared_l1988_198871

-- For the first expression
theorem factorization_3mx_6my (m x y : ℝ) : 
  3 * m * x - 6 * m * y = 3 * m * (x - 2 * y) := by sorry

-- For the second expression
theorem factorization_1_25x_squared (x : ℝ) :
  1 - 25 * x^2 = (1 + 5 * x) * (1 - 5 * x) := by sorry

end NUMINAMATH_CALUDE_factorization_3mx_6my_factorization_1_25x_squared_l1988_198871


namespace NUMINAMATH_CALUDE_min_n_for_S_greater_than_1020_l1988_198821

def S (n : ℕ) : ℕ := 2 * (2^n - 1) - n

theorem min_n_for_S_greater_than_1020 :
  ∀ k : ℕ, k < 10 → S k ≤ 1020 ∧ S 10 > 1020 := by sorry

end NUMINAMATH_CALUDE_min_n_for_S_greater_than_1020_l1988_198821


namespace NUMINAMATH_CALUDE_recurring_decimal_calculation_l1988_198819

theorem recurring_decimal_calculation : ∀ (x y : ℚ),
  x = 1/3 → y = 1 → (8 * x) / y = 8/3 := by sorry

end NUMINAMATH_CALUDE_recurring_decimal_calculation_l1988_198819


namespace NUMINAMATH_CALUDE_P_sufficient_not_necessary_for_Q_l1988_198825

def P : Set ℝ := {1, 2, 3, 4}
def Q : Set ℝ := {x : ℝ | 0 < x ∧ x < 5}

theorem P_sufficient_not_necessary_for_Q :
  (∀ x, x ∈ P → x ∈ Q) ∧ (∃ x, x ∈ Q ∧ x ∉ P) := by
  sorry

end NUMINAMATH_CALUDE_P_sufficient_not_necessary_for_Q_l1988_198825


namespace NUMINAMATH_CALUDE_proposition_logic_l1988_198896

theorem proposition_logic (p q : Prop) : 
  (((p ∧ q) → (p ∨ q)) ∧ ¬((p ∨ q) → (p ∧ q))) ∧
  ((¬(p ∧ q) → p) ∧ ¬(p → ¬(p ∧ q))) := by
  sorry

end NUMINAMATH_CALUDE_proposition_logic_l1988_198896


namespace NUMINAMATH_CALUDE_infinitely_many_non_representable_primes_l1988_198898

/-- A polynomial with integer coefficients -/
def IntPolynomial := ℕ → ℤ

/-- The degree of a polynomial -/
def degree (p : IntPolynomial) : ℕ := sorry

/-- Evaluate a polynomial at a given point -/
def eval (p : IntPolynomial) (x : ℕ) : ℤ := sorry

/-- The set of values that can be represented by a list of polynomials -/
def representableSet (polys : List IntPolynomial) : Set ℕ :=
  {n : ℕ | ∃ (p : IntPolynomial) (a : ℕ), p ∈ polys ∧ eval p a = n}

/-- The main theorem -/
theorem infinitely_many_non_representable_primes
  (n : ℕ)
  (polys : List IntPolynomial)
  (h_degree : ∀ p ∈ polys, degree p ≥ 2)
  : Set.Infinite {p : ℕ | Nat.Prime p ∧ p ∉ representableSet polys} := by
  sorry

end NUMINAMATH_CALUDE_infinitely_many_non_representable_primes_l1988_198898


namespace NUMINAMATH_CALUDE_digitSquareSequenceReaches1Or4_l1988_198829

/-- Sum of squares of digits of a natural number -/
def sumOfSquaresOfDigits (n : ℕ) : ℕ := sorry

/-- The sequence of sum of squares of digits -/
def digitSquareSequence (start : ℕ) : ℕ → ℕ
  | 0 => start
  | n + 1 => sumOfSquaresOfDigits (digitSquareSequence start n)

/-- Predicate to check if a number is three digits -/
def isThreeDigit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

theorem digitSquareSequenceReaches1Or4 (start : ℕ) (h : isThreeDigit start) :
  ∃ (k : ℕ), digitSquareSequence start k = 1 ∨ digitSquareSequence start k = 4 := by sorry

end NUMINAMATH_CALUDE_digitSquareSequenceReaches1Or4_l1988_198829


namespace NUMINAMATH_CALUDE_draw_four_from_fifteen_l1988_198804

/-- The number of balls in the bin -/
def n : ℕ := 15

/-- The number of balls to be drawn -/
def k : ℕ := 4

/-- The number of ways to draw k balls from n balls in order, without replacement -/
def drawWithoutReplacement (n k : ℕ) : ℕ :=
  (n - k + 1).factorial / (n - k).factorial

theorem draw_four_from_fifteen :
  drawWithoutReplacement n k = 32760 := by
  sorry

end NUMINAMATH_CALUDE_draw_four_from_fifteen_l1988_198804


namespace NUMINAMATH_CALUDE_complex_power_sum_l1988_198878

theorem complex_power_sum (z : ℂ) (h : z + 1/z = 2 * Real.cos (5 * π / 180)) :
  z^3000 + 1/z^3000 = 1 := by sorry

end NUMINAMATH_CALUDE_complex_power_sum_l1988_198878


namespace NUMINAMATH_CALUDE_three_statements_imply_negation_l1988_198803

theorem three_statements_imply_negation (p q : Prop) :
  let statement1 := p ∨ q
  let statement2 := p ∨ ¬q
  let statement3 := p ∧ ¬q
  let statement4 := ¬p ∨ ¬q
  let negation_of_both_false := ¬(¬p ∧ ¬q)
  (statement1 → negation_of_both_false) ∧
  (statement2 → negation_of_both_false) ∧
  (statement3 → negation_of_both_false) ∧
  ¬(statement4 → negation_of_both_false) := by
  sorry

end NUMINAMATH_CALUDE_three_statements_imply_negation_l1988_198803


namespace NUMINAMATH_CALUDE_school_commute_properties_l1988_198879

/-- Represents the distribution of students' commute times -/
structure CommuteDistribution where
  less_than_20 : Nat
  between_20_and_40 : Nat
  between_41_and_60 : Nat
  more_than_60 : Nat

/-- The given distribution of students' commute times -/
def school_distribution : CommuteDistribution :=
  { less_than_20 := 90
  , between_20_and_40 := 60
  , between_41_and_60 := 10
  , more_than_60 := 20 }

/-- Theorem stating the properties of the school's commute distribution -/
theorem school_commute_properties (d : CommuteDistribution) 
  (h : d = school_distribution) : 
  (d.less_than_20 = 90) ∧ 
  (d.less_than_20 + d.between_20_and_40 + d.between_41_and_60 + d.more_than_60 = 180) ∧ 
  (d.between_41_and_60 + d.more_than_60 = 30) ∧
  ¬(d.between_20_and_40 + d.between_41_and_60 + d.more_than_60 > d.less_than_20) := by
  sorry

#check school_commute_properties

end NUMINAMATH_CALUDE_school_commute_properties_l1988_198879


namespace NUMINAMATH_CALUDE_quadratic_root_implies_k_l1988_198893

theorem quadratic_root_implies_k (k : ℚ) : 
  (4 * ((-15 - Real.sqrt 165) / 8)^2 + 15 * ((-15 - Real.sqrt 165) / 8) + k = 0) → 
  k = 15/4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_implies_k_l1988_198893


namespace NUMINAMATH_CALUDE_quadratic_equations_solutions_l1988_198832

theorem quadratic_equations_solutions :
  (∀ x : ℝ, 2 * (x - 1)^2 = 18 ↔ x = 4 ∨ x = -2) ∧
  (∀ x : ℝ, x^2 - 4*x - 3 = 0 ↔ x = 2 + Real.sqrt 7 ∨ x = 2 - Real.sqrt 7) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equations_solutions_l1988_198832


namespace NUMINAMATH_CALUDE_fibonacci_6_l1988_198874

def fibonacci : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fibonacci (n + 1) + fibonacci n

theorem fibonacci_6 : fibonacci 5 = 8 := by
  sorry

end NUMINAMATH_CALUDE_fibonacci_6_l1988_198874


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l1988_198889

theorem simplify_and_evaluate (a : ℝ) (h : a = 2) : 
  a / (a^2 - 1) - 1 / (a^2 - 1) = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l1988_198889


namespace NUMINAMATH_CALUDE_tom_reading_pages_l1988_198881

def pages_read (initial_speed : ℕ) (time : ℕ) (speed_factor : ℕ) : ℕ :=
  initial_speed * speed_factor * time

theorem tom_reading_pages : pages_read 12 2 3 = 72 := by
  sorry

end NUMINAMATH_CALUDE_tom_reading_pages_l1988_198881


namespace NUMINAMATH_CALUDE_surface_area_upper_bound_l1988_198852

/-- A convex broken line in 3D space -/
structure ConvexBrokenLine where
  points : List (Real × Real × Real)
  is_convex : Bool
  length : Real

/-- The surface area generated by rotating a convex broken line around an axis -/
def surface_area_of_rotation (line : ConvexBrokenLine) (axis : Real × Real × Real) : Real :=
  sorry

/-- The theorem stating the upper bound of the surface area of rotation -/
theorem surface_area_upper_bound (line : ConvexBrokenLine) (axis : Real × Real × Real) :
  surface_area_of_rotation line axis ≤ Real.pi * line.length ^ 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_surface_area_upper_bound_l1988_198852


namespace NUMINAMATH_CALUDE_quadratic_equation_proof_l1988_198870

/-- Given a quadratic equation x^2 - 3x + k = 0 with one root being 4,
    prove that the other root is -1 and k = -4 -/
theorem quadratic_equation_proof (k : ℝ) : 
  (∃ x : ℝ, x^2 - 3*x + k = 0 ∧ x = 4) → 
  (∃ y : ℝ, y^2 - 3*y + k = 0 ∧ y = -1) ∧ k = -4 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_proof_l1988_198870


namespace NUMINAMATH_CALUDE_reasoning_classification_l1988_198820

-- Define the types of reasoning
inductive ReasoningType
  | Inductive
  | Analogical
  | Deductive

-- Define the reasoning methods
def method1 : String := "Inferring the properties of a ball from the properties of a circle"
def method2 : String := "Inducing that the sum of the internal angles of all triangles is 180° from the sum of the internal angles of right triangles, isosceles triangles, and equilateral triangles"
def method3 : String := "Deducing that f(x) = sinx is an odd function from f(-x) = -f(x), x ∈ R"
def method4 : String := "Inducing that the sum of the internal angles of a convex polygon is (n-2)•180° from the sum of the internal angles of a triangle, quadrilateral, and pentagon"

-- Define a function to classify reasoning methods
def classifyReasoning (method : String) : ReasoningType := sorry

-- Theorem to prove
theorem reasoning_classification :
  (classifyReasoning method1 = ReasoningType.Analogical) ∧
  (classifyReasoning method2 = ReasoningType.Inductive) ∧
  (classifyReasoning method3 = ReasoningType.Deductive) ∧
  (classifyReasoning method4 = ReasoningType.Inductive) := by
  sorry

end NUMINAMATH_CALUDE_reasoning_classification_l1988_198820


namespace NUMINAMATH_CALUDE_hcf_lcm_sum_reciprocal_l1988_198839

theorem hcf_lcm_sum_reciprocal (m n : ℕ+) : 
  Nat.gcd m.val n.val = 6 → 
  Nat.lcm m.val n.val = 210 → 
  m.val + n.val = 60 → 
  (1 : ℚ) / m.val + (1 : ℚ) / n.val = 1 / 21 := by
sorry

end NUMINAMATH_CALUDE_hcf_lcm_sum_reciprocal_l1988_198839


namespace NUMINAMATH_CALUDE_charity_sale_result_l1988_198888

/-- Represents the number and prices of shirts in a charity sale --/
structure ShirtSale where
  total_shirts : ℕ
  total_cost : ℕ
  black_wholesale : ℕ
  black_retail : ℕ
  white_wholesale : ℕ
  white_retail : ℕ

/-- Calculates the number of black and white shirts and the total profit --/
def calculate_shirts_and_profit (sale : ShirtSale) : 
  (ℕ × ℕ × ℕ) := sorry

/-- Theorem stating the correct results for the given shirt sale --/
theorem charity_sale_result (sale : ShirtSale) 
  (h1 : sale.total_shirts = 200)
  (h2 : sale.total_cost = 3500)
  (h3 : sale.black_wholesale = 25)
  (h4 : sale.black_retail = 50)
  (h5 : sale.white_wholesale = 15)
  (h6 : sale.white_retail = 35) :
  calculate_shirts_and_profit sale = (50, 150, 4250) := by sorry

end NUMINAMATH_CALUDE_charity_sale_result_l1988_198888


namespace NUMINAMATH_CALUDE_male_students_bound_l1988_198830

/-- Represents the arrangement of students in a grid -/
structure StudentArrangement where
  rows : ℕ
  columns : ℕ
  total_students : ℕ
  same_gender_pairs_bound : ℕ

/-- Counts the number of male students in a given arrangement -/
def count_male_students (arrangement : StudentArrangement) : ℕ := sorry

/-- The main theorem to be proved -/
theorem male_students_bound (arrangement : StudentArrangement) 
  (h1 : arrangement.rows = 22)
  (h2 : arrangement.columns = 75)
  (h3 : arrangement.total_students = 1650)
  (h4 : arrangement.same_gender_pairs_bound = 11) :
  count_male_students arrangement ≤ 928 := by sorry

end NUMINAMATH_CALUDE_male_students_bound_l1988_198830


namespace NUMINAMATH_CALUDE_characterization_of_n_l1988_198877

-- Define the type of positive integers
def PositiveInt := { n : ℕ | n > 0 }

-- Define a function to get all positive divisors of a number
def positiveDivisors (n : PositiveInt) : List PositiveInt := sorry

-- Define a function to check if a list forms a geometric sequence
def isGeometricSequence (l : List ℝ) : Prop := sorry

-- Define the conditions for n
def satisfiesConditions (n : PositiveInt) : Prop :=
  let divisors := positiveDivisors n
  (divisors.length ≥ 4) ∧
  (isGeometricSequence (List.zipWith (λ a b => b - a) divisors (List.tail divisors)))

-- Define the form pᵃ where p is prime and a ≥ 3
def isPrimePower (n : PositiveInt) : Prop :=
  ∃ (p : ℕ) (a : ℕ), Prime p ∧ a ≥ 3 ∧ n = p^a

-- The main theorem
theorem characterization_of_n (n : PositiveInt) :
  satisfiesConditions n ↔ isPrimePower n := by sorry

end NUMINAMATH_CALUDE_characterization_of_n_l1988_198877


namespace NUMINAMATH_CALUDE_budget_increase_is_twenty_percent_l1988_198800

/-- The percentage increase in the gym budget -/
def budget_increase_percentage (original_dodgeball_count : ℕ) (dodgeball_price : ℚ)
  (new_softball_count : ℕ) (softball_price : ℚ) : ℚ :=
  let original_budget := original_dodgeball_count * dodgeball_price
  let new_budget := new_softball_count * softball_price
  ((new_budget - original_budget) / original_budget) * 100

/-- Theorem stating that the budget increase percentage is 20% -/
theorem budget_increase_is_twenty_percent :
  budget_increase_percentage 15 5 10 9 = 20 := by
  sorry

end NUMINAMATH_CALUDE_budget_increase_is_twenty_percent_l1988_198800


namespace NUMINAMATH_CALUDE_four_blocks_in_six_by_six_grid_l1988_198848

theorem four_blocks_in_six_by_six_grid : 
  let n : ℕ := 6
  let k : ℕ := 4
  let grid_size := n * n
  let combinations := (n.choose k) * (n.choose k) * (k.factorial)
  combinations = 5400 := by
  sorry

end NUMINAMATH_CALUDE_four_blocks_in_six_by_six_grid_l1988_198848


namespace NUMINAMATH_CALUDE_arithmetic_proof_l1988_198867

theorem arithmetic_proof : (56^2 + 56^2) / 28^2 = 8 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_proof_l1988_198867


namespace NUMINAMATH_CALUDE_min_value_implies_a_l1988_198869

/-- Given a function f(x) = 4x + a/x where x > 0 and a > 0,
    if the function takes its minimum value at x = 2,
    then a = 16 -/
theorem min_value_implies_a (a : ℝ) :
  (a > 0) →
  (∀ x : ℝ, x > 0 → 4*x + a/x ≥ 4*2 + a/2) →
  (∀ x : ℝ, x > 0 → x ≠ 2 → 4*x + a/x > 4*2 + a/2) →
  a = 16 := by
  sorry

end NUMINAMATH_CALUDE_min_value_implies_a_l1988_198869


namespace NUMINAMATH_CALUDE_light_bulb_probability_l1988_198818

theorem light_bulb_probability (pass_rate : ℝ) (h1 : 0 ≤ pass_rate ∧ pass_rate ≤ 1) :
  pass_rate = 0.99 → 
  ∃ (P : Set ℝ → ℝ), 
    (∀ A, 0 ≤ P A ∧ P A ≤ 1) ∧ 
    (P ∅ = 0) ∧ 
    (P univ = 1) ∧
    P {x | x ≤ pass_rate} = 0.99 :=
by sorry

end NUMINAMATH_CALUDE_light_bulb_probability_l1988_198818


namespace NUMINAMATH_CALUDE_cost_function_correct_l1988_198866

/-- The cost function for shipping a parcel -/
def cost (P : ℕ) : ℕ :=
  if P ≤ 5 then
    12 + 4 * (P - 1)
  else
    27 + 4 * P - 21

/-- Theorem stating the correctness of the cost function -/
theorem cost_function_correct (P : ℕ) :
  (P ≤ 5 → cost P = 12 + 4 * (P - 1)) ∧
  (P > 5 → cost P = 27 + 4 * P - 21) := by
  sorry

end NUMINAMATH_CALUDE_cost_function_correct_l1988_198866


namespace NUMINAMATH_CALUDE_pages_read_tomorrow_l1988_198873

/-- The number of pages Melody needs to read for her English class -/
def english_pages : ℕ := 50

/-- The number of pages Melody needs to read for her Math class -/
def math_pages : ℕ := 30

/-- The number of pages Melody needs to read for her History class -/
def history_pages : ℕ := 20

/-- The number of pages Melody needs to read for her Chinese class -/
def chinese_pages : ℕ := 40

/-- The fraction of English pages Melody will read tomorrow -/
def english_fraction : ℚ := 1 / 5

/-- The percentage of Math pages Melody will read tomorrow -/
def math_percentage : ℚ := 30 / 100

/-- The fraction of History pages Melody will read tomorrow -/
def history_fraction : ℚ := 1 / 4

/-- The percentage of Chinese pages Melody will read tomorrow -/
def chinese_percentage : ℚ := 125 / 1000

/-- Theorem stating the total number of pages Melody will read tomorrow -/
theorem pages_read_tomorrow :
  (english_fraction * english_pages).floor +
  (math_percentage * math_pages).floor +
  (history_fraction * history_pages).floor +
  (chinese_percentage * chinese_pages).floor = 29 := by
  sorry

end NUMINAMATH_CALUDE_pages_read_tomorrow_l1988_198873


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l1988_198849

theorem quadratic_inequality_range (a : ℝ) : 
  (∀ x : ℝ, a * x^2 + a * x + 1 > 0) ↔ a ∈ Set.Ici 0 ∩ Set.Iio 4 :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l1988_198849


namespace NUMINAMATH_CALUDE_shells_not_red_or_green_l1988_198831

theorem shells_not_red_or_green (total : ℕ) (red : ℕ) (green : ℕ) 
  (h1 : total = 291) (h2 : red = 76) (h3 : green = 49) :
  total - (red + green) = 166 := by
  sorry

end NUMINAMATH_CALUDE_shells_not_red_or_green_l1988_198831


namespace NUMINAMATH_CALUDE_largest_positive_solution_l1988_198897

theorem largest_positive_solution : 
  ∃ (x : ℝ), x > 0 ∧ 
    (2 * x^3 - x^2 - x + 1)^(1 + 1/(2*x + 1)) = 1 ∧ 
    ∀ (y : ℝ), y > 0 → 
      (2 * y^3 - y^2 - y + 1)^(1 + 1/(2*y + 1)) = 1 → 
      y ≤ x :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_largest_positive_solution_l1988_198897


namespace NUMINAMATH_CALUDE_wallace_jerky_production_l1988_198841

/-- Represents the jerky production scenario -/
structure JerkyProduction where
  total_order : ℕ
  already_made : ℕ
  days_to_fulfill : ℕ
  batches_per_day : ℕ

/-- Calculates the number of bags one batch can make -/
def bags_per_batch (jp : JerkyProduction) : ℕ :=
  ((jp.total_order - jp.already_made) / jp.days_to_fulfill) / jp.batches_per_day

/-- Theorem stating that under the given conditions, one batch makes 10 bags -/
theorem wallace_jerky_production :
  ∀ (jp : JerkyProduction),
    jp.total_order = 60 →
    jp.already_made = 20 →
    jp.days_to_fulfill = 4 →
    jp.batches_per_day = 1 →
    bags_per_batch jp = 10 := by
  sorry

end NUMINAMATH_CALUDE_wallace_jerky_production_l1988_198841


namespace NUMINAMATH_CALUDE_horizontal_asymptote_rational_function_l1988_198838

/-- The function f(x) = (7x^2 - 4) / (4x^2 + 8x - 3) has a horizontal asymptote at y = 7/4 -/
theorem horizontal_asymptote_rational_function :
  let f : ℝ → ℝ := λ x => (7 * x^2 - 4) / (4 * x^2 + 8 * x - 3)
  ∀ ε > 0, ∃ N : ℝ, ∀ x : ℝ, x > N → |f x - 7/4| < ε :=
by sorry

end NUMINAMATH_CALUDE_horizontal_asymptote_rational_function_l1988_198838


namespace NUMINAMATH_CALUDE_smallest_cube_ending_144_l1988_198883

theorem smallest_cube_ending_144 : ∃ (n : ℕ), n > 0 ∧ n^3 % 1000 = 144 ∧ ∀ (m : ℕ), m > 0 → m^3 % 1000 = 144 → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_smallest_cube_ending_144_l1988_198883


namespace NUMINAMATH_CALUDE_shopping_problem_l1988_198843

theorem shopping_problem (total : ℕ) (stores : ℕ) (initial_amount : ℕ) :
  total = stores ∧ 
  initial_amount = 100 ∧ 
  stores = 6 → 
  ∃ (spent_per_store : ℕ), 
    spent_per_store * stores ≤ initial_amount ∧ 
    spent_per_store > 0 ∧
    initial_amount - spent_per_store * stores ≤ 28 :=
by sorry

#check shopping_problem

end NUMINAMATH_CALUDE_shopping_problem_l1988_198843


namespace NUMINAMATH_CALUDE_parallelogram_base_length_l1988_198824

/-- Given a parallelogram with area 108 cm² and height 9 cm, its base length is 12 cm. -/
theorem parallelogram_base_length (area : ℝ) (height : ℝ) (base : ℝ) :
  area = 108 ∧ height = 9 ∧ area = base * height → base = 12 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_base_length_l1988_198824


namespace NUMINAMATH_CALUDE_systematic_sampling_interval_l1988_198805

/-- Calculates the sampling interval for systematic sampling -/
def samplingInterval (populationSize sampleSize : ℕ) : ℕ :=
  populationSize / sampleSize

/-- Theorem: The sampling interval for a population of 1200 and sample size of 40 is 30 -/
theorem systematic_sampling_interval :
  samplingInterval 1200 40 = 30 := by
  sorry

end NUMINAMATH_CALUDE_systematic_sampling_interval_l1988_198805


namespace NUMINAMATH_CALUDE_max_value_quadratic_l1988_198850

theorem max_value_quadratic (x : ℝ) :
  let y : ℝ → ℝ := λ x => -3 * x^2 + 4 * x + 6
  ∃ (max_y : ℝ), ∀ (x : ℝ), y x ≤ max_y ∧ max_y = 22/3 :=
by
  sorry

end NUMINAMATH_CALUDE_max_value_quadratic_l1988_198850


namespace NUMINAMATH_CALUDE_rectangle_max_area_l1988_198891

/-- Given a rectangle with perimeter 40 units and length twice its width, 
    the maximum area of the rectangle is 800/9 square units. -/
theorem rectangle_max_area : 
  ∀ w l : ℝ, 
  w > 0 → 
  l > 0 → 
  2 * (w + l) = 40 → 
  l = 2 * w → 
  ∀ a : ℝ, a = w * l → a ≤ 800 / 9 :=
by
  sorry

end NUMINAMATH_CALUDE_rectangle_max_area_l1988_198891


namespace NUMINAMATH_CALUDE_probability_five_heads_seven_flips_l1988_198851

theorem probability_five_heads_seven_flips :
  let n : ℕ := 7  -- total number of flips
  let k : ℕ := 5  -- number of heads we want
  let p : ℚ := 1/2  -- probability of heads on a single flip (fair coin)
  Nat.choose n k * p^k * (1 - p)^(n - k) = 21/128 :=
by sorry

end NUMINAMATH_CALUDE_probability_five_heads_seven_flips_l1988_198851


namespace NUMINAMATH_CALUDE_p_necessary_not_sufficient_for_q_l1988_198886

theorem p_necessary_not_sufficient_for_q :
  (∃ a : ℝ, a > 4 ∧ ¬(5 < a ∧ a < 6)) ∧
  (∀ a : ℝ, 5 < a ∧ a < 6 → a > 4) :=
by sorry

end NUMINAMATH_CALUDE_p_necessary_not_sufficient_for_q_l1988_198886


namespace NUMINAMATH_CALUDE_trapezium_other_side_length_l1988_198826

/-- Theorem: In a trapezium with one parallel side of 18 cm, a distance between parallel sides of 10 cm,
    and an area of 190 square centimeters, the length of the other parallel side is 20 cm. -/
theorem trapezium_other_side_length (a b h : ℝ) (h1 : a = 18) (h2 : h = 10) (h3 : (a + b) * h / 2 = 190) :
  b = 20 := by
  sorry

end NUMINAMATH_CALUDE_trapezium_other_side_length_l1988_198826


namespace NUMINAMATH_CALUDE_smallest_positive_integer_with_remainders_l1988_198806

theorem smallest_positive_integer_with_remainders : ∃! n : ℕ, 
  n > 0 ∧ 
  n % 2 = 1 ∧
  n % 3 = 2 ∧
  n % 4 = 3 ∧
  n % 5 = 4 ∧
  n % 6 = 5 ∧
  n % 7 = 6 ∧
  n % 8 = 7 ∧
  n % 9 = 8 ∧
  n % 10 = 9 ∧
  ∀ m : ℕ, m > 0 ∧ 
    m % 2 = 1 ∧
    m % 3 = 2 ∧
    m % 4 = 3 ∧
    m % 5 = 4 ∧
    m % 6 = 5 ∧
    m % 7 = 6 ∧
    m % 8 = 7 ∧
    m % 9 = 8 ∧
    m % 10 = 9 → m ≥ n :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_positive_integer_with_remainders_l1988_198806


namespace NUMINAMATH_CALUDE_sqrt_x_plus_one_over_sqrt_x_l1988_198814

theorem sqrt_x_plus_one_over_sqrt_x (x : ℝ) (h1 : x > 0) (h2 : x + 1/x = 50) : 
  Real.sqrt x + 1 / Real.sqrt x = Real.sqrt 52 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_x_plus_one_over_sqrt_x_l1988_198814


namespace NUMINAMATH_CALUDE_number_of_boys_number_of_boys_is_17_l1988_198875

theorem number_of_boys (total_children : ℕ) (happy_children : ℕ) (sad_children : ℕ) 
  (neither_children : ℕ) (girls : ℕ) (happy_boys : ℕ) (sad_girls : ℕ) 
  (neither_boys : ℕ) : ℕ :=
  by
  have h1 : total_children = 60 := by sorry
  have h2 : happy_children = 30 := by sorry
  have h3 : sad_children = 10 := by sorry
  have h4 : neither_children = 20 := by sorry
  have h5 : girls = 43 := by sorry
  have h6 : happy_boys = 6 := by sorry
  have h7 : sad_girls = 4 := by sorry
  have h8 : neither_boys = 5 := by sorry
  
  exact total_children - girls

theorem number_of_boys_is_17 : number_of_boys 60 30 10 20 43 6 4 5 = 17 := by sorry

end NUMINAMATH_CALUDE_number_of_boys_number_of_boys_is_17_l1988_198875


namespace NUMINAMATH_CALUDE_negation_of_proposition_l1988_198823

theorem negation_of_proposition (p : Prop) :
  (¬ (∃ m : ℝ, (m^2 + m - 6)⁻¹ > 0)) ↔ 
  (∀ m : ℝ, (m^2 + m - 6)⁻¹ < 0 ∨ m^2 + m - 6 = 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l1988_198823


namespace NUMINAMATH_CALUDE_find_number_l1988_198859

theorem find_number : ∃ x : ℚ, (x + 32/113) * 113 = 9637 ∧ x = 85 := by sorry

end NUMINAMATH_CALUDE_find_number_l1988_198859


namespace NUMINAMATH_CALUDE_inequality_proof_l1988_198837

theorem inequality_proof (x y a : ℝ) 
  (h1 : x + a < y + a) 
  (h2 : a * x > a * y) : 
  x < y ∧ a < 0 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l1988_198837


namespace NUMINAMATH_CALUDE_focus_to_asymptote_distance_l1988_198810

-- Define the hyperbola C
def hyperbola (x y : ℝ) : Prop := x^2 / 3 - y^2 / 3 = 1

-- Define a focus of the hyperbola
def focus (F : ℝ × ℝ) : Prop := 
  ∃ (x y : ℝ), hyperbola x y ∧ F = (Real.sqrt 6, 0)

-- Define an asymptote of the hyperbola
def asymptote (x y : ℝ) : Prop := y = x

-- Theorem statement
theorem focus_to_asymptote_distance (F : ℝ × ℝ) :
  focus F → (∃ (x y : ℝ), asymptote x y ∧ 
    Real.sqrt ((F.1 - x)^2 + (F.2 - y)^2) / Real.sqrt (1 + 1^2) = Real.sqrt 3) :=
sorry

end NUMINAMATH_CALUDE_focus_to_asymptote_distance_l1988_198810


namespace NUMINAMATH_CALUDE_pages_read_initially_l1988_198892

def book_chapters : ℕ := 8
def book_pages : ℕ := 95
def pages_read_later : ℕ := 25
def total_pages_read : ℕ := 62

theorem pages_read_initially : 
  total_pages_read - pages_read_later = 37 := by
  sorry

end NUMINAMATH_CALUDE_pages_read_initially_l1988_198892


namespace NUMINAMATH_CALUDE_game_c_higher_prob_l1988_198861

/-- A biased coin with probability of heads 3/5 and tails 2/5 -/
structure BiasedCoin where
  p_heads : ℚ
  p_tails : ℚ
  head_prob : p_heads = 3/5
  tail_prob : p_tails = 2/5
  total_prob : p_heads + p_tails = 1

/-- Game C: Win if all three outcomes are the same -/
def prob_win_game_c (coin : BiasedCoin) : ℚ :=
  coin.p_heads^3 + coin.p_tails^3

/-- Game D: Win if first two outcomes are the same and third is different -/
def prob_win_game_d (coin : BiasedCoin) : ℚ :=
  2 * (coin.p_heads^2 * coin.p_tails + coin.p_tails^2 * coin.p_heads)

/-- The main theorem stating that Game C has a 1/25 higher probability of winning -/
theorem game_c_higher_prob (coin : BiasedCoin) :
  prob_win_game_c coin - prob_win_game_d coin = 1/25 := by
  sorry

end NUMINAMATH_CALUDE_game_c_higher_prob_l1988_198861


namespace NUMINAMATH_CALUDE_fourth_pile_magazines_l1988_198853

def magazine_sequence (n : ℕ) : ℕ :=
  if n = 1 then 3
  else if n = 2 then 4
  else if n = 3 then 6
  else if n = 5 then 13
  else 0  -- For other values, we don't have information

def difference_sequence (n : ℕ) : ℕ :=
  magazine_sequence (n + 1) - magazine_sequence n

theorem fourth_pile_magazines :
  (∀ n : ℕ, n ≥ 1 ∧ n ≤ 3 → difference_sequence (n + 1) = difference_sequence n + 1) →
  magazine_sequence 4 = 9 := by
  sorry

end NUMINAMATH_CALUDE_fourth_pile_magazines_l1988_198853


namespace NUMINAMATH_CALUDE_joe_height_difference_l1988_198890

/-- Proves that Joe is 6 inches taller than double Sara's height -/
theorem joe_height_difference (sara : ℝ) (joe : ℝ) : 
  sara + joe = 120 →
  joe = 82 →
  joe - 2 * sara = 6 := by
sorry

end NUMINAMATH_CALUDE_joe_height_difference_l1988_198890


namespace NUMINAMATH_CALUDE_prime_divisibility_l1988_198822

theorem prime_divisibility (p q r : ℕ) : 
  Prime p → Prime q → Prime r → Odd p → (p ∣ q^r + 1) → 
  (2*r ∣ p - 1) ∨ (p ∣ q^2 - 1) := by
  sorry

end NUMINAMATH_CALUDE_prime_divisibility_l1988_198822


namespace NUMINAMATH_CALUDE_equation_root_approximation_l1988_198807

/-- The equation whose root we need to find -/
def equation (x : ℝ) : Prop :=
  (Real.sqrt 5 - Real.sqrt 2) * (1 + x) = (Real.sqrt 6 - Real.sqrt 3) * (1 - x)

/-- The approximate root of the equation -/
def approximate_root : ℝ := -0.068

/-- Theorem stating that the approximate root satisfies the equation within a small error -/
theorem equation_root_approximation :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.001 ∧ 
  |((Real.sqrt 5 - Real.sqrt 2) * (1 + approximate_root) - 
    (Real.sqrt 6 - Real.sqrt 3) * (1 - approximate_root))| < ε :=
sorry

end NUMINAMATH_CALUDE_equation_root_approximation_l1988_198807


namespace NUMINAMATH_CALUDE_quadratic_always_negative_l1988_198857

theorem quadratic_always_negative (m : ℝ) :
  (∀ x : ℝ, -x^2 + (2*m + 6)*x - m - 3 < 0) ↔ -3 < m ∧ m < -2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_always_negative_l1988_198857


namespace NUMINAMATH_CALUDE_parallelepiped_volume_and_lateral_area_l1988_198863

/-- 
Given a right parallelepiped with a rhombus base of area Q and diagonal section areas S₁ and S₂,
this theorem proves the formulas for its volume and lateral surface area.
-/
theorem parallelepiped_volume_and_lateral_area (Q S₁ S₂ : ℝ) 
  (hQ : Q > 0) (hS₁ : S₁ > 0) (hS₂ : S₂ > 0) :
  ∃ (V LSA : ℝ),
    V = Real.sqrt ((S₁ * S₂ * Q) / 2) ∧ 
    LSA = 2 * Real.sqrt (S₁^2 + S₂^2) := by
  sorry

end NUMINAMATH_CALUDE_parallelepiped_volume_and_lateral_area_l1988_198863


namespace NUMINAMATH_CALUDE_trig_fraction_equality_l1988_198858

theorem trig_fraction_equality (x : ℝ) (h : (1 - Real.sin x) / Real.cos x = 3/5) :
  Real.cos x / (1 + Real.sin x) = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_trig_fraction_equality_l1988_198858


namespace NUMINAMATH_CALUDE_irrational_minus_rational_is_irrational_pi_minus_3_14_irrational_l1988_198864

/-- π is irrational -/
axiom pi_irrational : Irrational Real.pi

/-- 3.14 is rational -/
axiom rational_3_14 : ∃ (q : ℚ), (q : ℝ) = 3.14

/-- The difference of an irrational number and a rational number is irrational -/
theorem irrational_minus_rational_is_irrational (x y : ℝ) (hx : Irrational x) (hy : ∃ (q : ℚ), (q : ℝ) = y) :
  Irrational (x - y) :=
sorry

/-- π - 3.14 is irrational -/
theorem pi_minus_3_14_irrational : Irrational (Real.pi - 3.14) :=
  irrational_minus_rational_is_irrational Real.pi 3.14 pi_irrational rational_3_14

end NUMINAMATH_CALUDE_irrational_minus_rational_is_irrational_pi_minus_3_14_irrational_l1988_198864


namespace NUMINAMATH_CALUDE_factors_of_N_l1988_198812

/-- The number of natural-number factors of N, where N = 2^4 * 3^3 * 5^2 * 7^2 -/
def num_factors (N : Nat) : Nat :=
  if N = 2^4 * 3^3 * 5^2 * 7^2 then 180 else 0

/-- Theorem stating that the number of natural-number factors of N is 180 -/
theorem factors_of_N :
  ∃ N : Nat, N = 2^4 * 3^3 * 5^2 * 7^2 ∧ num_factors N = 180 :=
by
  sorry

#check factors_of_N

end NUMINAMATH_CALUDE_factors_of_N_l1988_198812


namespace NUMINAMATH_CALUDE_problem_statement_l1988_198828

theorem problem_statement (x y : ℝ) (h : x^2 * y^2 - x * y - x / y - y / x = 4) :
  (x - 2) * (y - 2) = 3 - 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1988_198828


namespace NUMINAMATH_CALUDE_vintik_votes_l1988_198844

theorem vintik_votes (total_percentage : ℝ) (shpuntik_votes : ℕ) 
  (h1 : total_percentage = 146)
  (h2 : shpuntik_votes > 1000) :
  ∃ (vintik_votes : ℕ), vintik_votes > 850 := by
  sorry

end NUMINAMATH_CALUDE_vintik_votes_l1988_198844


namespace NUMINAMATH_CALUDE_expression_equals_negative_one_l1988_198817

theorem expression_equals_negative_one (x y : ℝ) 
  (hx : x ≠ 0) (hxy : x ≠ 2*y ∧ x ≠ -2*y) : 
  (x / (x + 2*y) + 2*y / (x - 2*y)) / (2*y / (x + 2*y) - x / (x - 2*y)) = -1 :=
by sorry

end NUMINAMATH_CALUDE_expression_equals_negative_one_l1988_198817


namespace NUMINAMATH_CALUDE_sum_and_round_to_nearest_ten_l1988_198809

-- Define a function to round to the nearest ten
def roundToNearestTen (n : ℤ) : ℤ :=
  10 * ((n + 5) / 10)

-- Theorem statement
theorem sum_and_round_to_nearest_ten :
  roundToNearestTen (54 + 29) = 80 := by
  sorry

end NUMINAMATH_CALUDE_sum_and_round_to_nearest_ten_l1988_198809


namespace NUMINAMATH_CALUDE_platform_length_l1988_198836

/-- The length of a platform given train specifications -/
theorem platform_length (train_length : ℝ) (cross_platform_time : ℝ) (cross_pole_time : ℝ)
  (h1 : train_length = 300)
  (h2 : cross_platform_time = 39)
  (h3 : cross_pole_time = 8) :
  let train_speed := train_length / cross_pole_time
  let platform_length := train_speed * cross_platform_time - train_length
  platform_length = 1162.5 := by sorry

end NUMINAMATH_CALUDE_platform_length_l1988_198836


namespace NUMINAMATH_CALUDE_equation_roots_range_l1988_198815

theorem equation_roots_range (n : ℕ) (k : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
   2*n - 1 < x₁ ∧ x₁ ≤ 2*n + 1 ∧
   2*n - 1 < x₂ ∧ x₂ ≤ 2*n + 1 ∧
   |x₁ - 2*n| = k * Real.sqrt x₁ ∧
   |x₂ - 2*n| = k * Real.sqrt x₂) →
  (0 < k ∧ k ≤ 1 / Real.sqrt (2*n + 1)) :=
by sorry

end NUMINAMATH_CALUDE_equation_roots_range_l1988_198815


namespace NUMINAMATH_CALUDE_office_network_connections_l1988_198840

/-- The number of connections in a network of switches where each switch is connected to a fixed number of other switches. -/
def network_connections (n : ℕ) (k : ℕ) : ℕ :=
  (n * k) / 2

/-- Theorem: In a network of 30 switches, where each switch is directly connected to exactly 4 other switches, the total number of connections is 60. -/
theorem office_network_connections :
  network_connections 30 4 = 60 := by
  sorry

end NUMINAMATH_CALUDE_office_network_connections_l1988_198840


namespace NUMINAMATH_CALUDE_projectile_meeting_time_l1988_198899

theorem projectile_meeting_time (distance : ℝ) (speed1 : ℝ) (speed2 : ℝ) : 
  distance = 1455 →
  speed1 = 470 →
  speed2 = 500 →
  (distance / (speed1 + speed2)) * 60 = 90 := by
sorry

end NUMINAMATH_CALUDE_projectile_meeting_time_l1988_198899


namespace NUMINAMATH_CALUDE_unique_nonnegative_solution_l1988_198885

theorem unique_nonnegative_solution (x y z : ℝ) :
  x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0 →
  x + y + z = 3 * x * y →
  x^2 + y^2 + z^2 = 3 * x * z →
  x^3 + y^3 + z^3 = 3 * y * z →
  x = 1 ∧ y = 1 ∧ z = 1 := by
sorry

end NUMINAMATH_CALUDE_unique_nonnegative_solution_l1988_198885


namespace NUMINAMATH_CALUDE_count_divisible_by_11_equals_v_l1988_198811

/-- Concatenates the squares of integers from 1 to n -/
def b (n : ℕ) : ℕ := sorry

/-- Counts how many numbers b_k are divisible by 11 for 1 ≤ k ≤ 50 -/
def count_divisible_by_11 : ℕ := sorry

/-- The correct count of numbers b_k divisible by 11 for 1 ≤ k ≤ 50 -/
def v : ℕ := sorry

theorem count_divisible_by_11_equals_v : count_divisible_by_11 = v := by sorry

end NUMINAMATH_CALUDE_count_divisible_by_11_equals_v_l1988_198811


namespace NUMINAMATH_CALUDE_martha_apples_l1988_198847

theorem martha_apples (tim harry martha : ℕ) 
  (h1 : martha = tim + 30)
  (h2 : harry = tim / 2)
  (h3 : harry = 19) : 
  martha = 68 := by sorry

end NUMINAMATH_CALUDE_martha_apples_l1988_198847


namespace NUMINAMATH_CALUDE_right_triangle_circumradius_l1988_198846

theorem right_triangle_circumradius (a b c : ℝ) (h1 : a = 9) (h2 : b = 12) (h3 : c = 15) :
  a^2 + b^2 = c^2 → (c / 2 : ℝ) = 7.5 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_circumradius_l1988_198846


namespace NUMINAMATH_CALUDE_greatest_bound_of_r2_l1988_198845

/-- The function f(x) = x^2 - r_2x + r_3 -/
def f (r_2 r_3 : ℝ) (x : ℝ) : ℝ := x^2 - r_2*x + r_3

/-- The sequence g_n defined recursively -/
def g (r_2 r_3 : ℝ) : ℕ → ℝ
  | 0 => 0
  | n + 1 => f r_2 r_3 (g r_2 r_3 n)

/-- The property that g_{2i} < g_{2i+1} and g_{2i+1} > g_{2i+2} for 0 ≤ i ≤ 2011 -/
def alternating_property (r_2 r_3 : ℝ) : Prop :=
  ∀ i : ℕ, i ≤ 2011 → g r_2 r_3 (2*i) < g r_2 r_3 (2*i + 1) ∧ g r_2 r_3 (2*i + 1) > g r_2 r_3 (2*i + 2)

/-- The property that there exists j such that g_{i+1} > g_i for all i > j -/
def eventually_increasing (r_2 r_3 : ℝ) : Prop :=
  ∃ j : ℕ, ∀ i : ℕ, i > j → g r_2 r_3 (i + 1) > g r_2 r_3 i

/-- The property that the sequence g_n is unbounded -/
def unbounded_sequence (r_2 r_3 : ℝ) : Prop :=
  ∀ M : ℝ, ∃ N : ℕ, g r_2 r_3 N > M

/-- The main theorem -/
theorem greatest_bound_of_r2 :
  (∃ A : ℝ, ∀ r_2 r_3 : ℝ, 
    alternating_property r_2 r_3 → 
    eventually_increasing r_2 r_3 → 
    unbounded_sequence r_2 r_3 → 
    A ≤ |r_2| ∧ 
    (∀ B : ℝ, (∀ r_2' r_3' : ℝ, 
      alternating_property r_2' r_3' → 
      eventually_increasing r_2' r_3' → 
      unbounded_sequence r_2' r_3' → 
      B ≤ |r_2'|) → B ≤ A)) ∧
  (∀ A : ℝ, (∀ r_2 r_3 : ℝ, 
    alternating_property r_2 r_3 → 
    eventually_increasing r_2 r_3 → 
    unbounded_sequence r_2 r_3 → 
    A ≤ |r_2| ∧ 
    (∀ B : ℝ, (∀ r_2' r_3' : ℝ, 
      alternating_property r_2' r_3' → 
      eventually_increasing r_2' r_3' → 
      unbounded_sequence r_2' r_3' → 
      B ≤ |r_2'|) → B ≤ A)) → A = 2) := by
  sorry

end NUMINAMATH_CALUDE_greatest_bound_of_r2_l1988_198845


namespace NUMINAMATH_CALUDE_sinusoidal_function_parameters_l1988_198856

open Real

theorem sinusoidal_function_parameters 
  (f : ℝ → ℝ)
  (ω φ : ℝ)
  (h1 : ∀ x, f x = 2 * sin (ω * x + φ))
  (h2 : ω > 0)
  (h3 : abs φ < π)
  (h4 : f (5 * π / 8) = 2)
  (h5 : f (11 * π / 8) = 0)
  (h6 : ∀ T > 0, (∀ x, f (x + T) = f x) → T ≥ 3 * π) :
  ω = 2 / 3 ∧ φ = π / 12 := by
sorry

end NUMINAMATH_CALUDE_sinusoidal_function_parameters_l1988_198856


namespace NUMINAMATH_CALUDE_equation_solution_l1988_198887

theorem equation_solution : ∃! x : ℝ, (3 / (x - 3) = 4 / (x - 4)) ∧ x ≠ 3 ∧ x ≠ 4 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1988_198887


namespace NUMINAMATH_CALUDE_last_home_game_score_l1988_198813

theorem last_home_game_score (H : ℕ) : 
  (H = 2 * (H / 2)) →  -- Last home game score is twice the first away game
  (∃ second_away : ℕ, second_away = H / 2 + 18) →  -- Second away game score
  (∃ third_away : ℕ, third_away = (H / 2 + 18) + 2) →  -- Third away game score
  ((5 * H) / 2 + 38 + 55 = 4 * H) →  -- Cumulative points condition
  H = 62 := by
sorry

end NUMINAMATH_CALUDE_last_home_game_score_l1988_198813


namespace NUMINAMATH_CALUDE_certain_number_problem_l1988_198854

theorem certain_number_problem (x y : ℝ) (h1 : x = 180) 
  (h2 : 0.25 * x = 0.10 * y - 5) : y = 500 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_problem_l1988_198854


namespace NUMINAMATH_CALUDE_mechanic_bill_calculation_l1988_198816

/-- Given a mechanic's hourly rate, parts cost, and hours worked, calculate the total bill -/
def total_bill (hourly_rate : ℕ) (parts_cost : ℕ) (hours_worked : ℕ) : ℕ :=
  hourly_rate * hours_worked + parts_cost

/-- Theorem: The total bill for a 5-hour job with $45/hour rate and $225 parts cost is $450 -/
theorem mechanic_bill_calculation :
  total_bill 45 225 5 = 450 := by
  sorry

end NUMINAMATH_CALUDE_mechanic_bill_calculation_l1988_198816


namespace NUMINAMATH_CALUDE_triangle_perimeter_is_minimal_l1988_198868

def triangle_side_1 : ℕ := 45
def triangle_side_2 : ℕ := 55
def triangle_side_3 : ℕ := 2 * triangle_side_1

def triangle_perimeter : ℕ := triangle_side_1 + triangle_side_2 + triangle_side_3

theorem triangle_perimeter_is_minimal : 
  triangle_perimeter = 190 ∧ 
  (∀ a b c : ℕ, a = triangle_side_1 → b = triangle_side_2 → c ≥ 2 * triangle_side_1 → 
   a + b > c ∧ a + c > b ∧ b + c > a → a + b + c ≥ triangle_perimeter) := by
  sorry

end NUMINAMATH_CALUDE_triangle_perimeter_is_minimal_l1988_198868


namespace NUMINAMATH_CALUDE_complement_of_A_l1988_198835

def U : Set ℝ := Set.univ

def A : Set ℝ := {x : ℝ | x ≥ 1} ∪ {x : ℝ | x ≤ 0}

theorem complement_of_A : Set.compl A = Set.Ioo 0 1 := by sorry

end NUMINAMATH_CALUDE_complement_of_A_l1988_198835


namespace NUMINAMATH_CALUDE_quadrilateral_area_inequality_l1988_198860

/-- A quadrilateral with sides a, b, c, d and area S -/
structure Quadrilateral where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  S : ℝ

/-- Predicate for a cyclic quadrilateral with perpendicular diagonals -/
def is_cyclic_perpendicular_diagonals (q : Quadrilateral) : Prop := sorry

theorem quadrilateral_area_inequality (q : Quadrilateral) :
  q.S ≤ (q.a * q.c + q.b * q.d) / 2 ∧
  (q.S = (q.a * q.c + q.b * q.d) / 2 ↔ is_cyclic_perpendicular_diagonals q) := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_area_inequality_l1988_198860
